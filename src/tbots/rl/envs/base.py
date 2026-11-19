"""Our Gymnasium environment, built on the SimBackend interface.

A `SimBackend`, not a `Backend`: training needs to place the ball, command
opponents and inject referee state, and none of those exist on real hardware.
Taking the narrower type here is what makes "you cannot train against the
robots" a type error instead of a crash forty minutes into a run.

THE ENVIRONMENT HOLDS TWO WORLDS (ARCHITECTURE section 7):

    truth       what rSim says happened. Drives reward, termination, and
                (later) the synthetic referee.
    perceived   what `Perception` says a camera would have shown. Drives the
                observation, and nothing else.

Rewarding on the perceived world would reward noise; observing the true one
would train a policy on information it never has at a match.

One `step()` may cover several physics ticks (`action_repeat`): `_decode`
returns one command list per tick, perception sees every tick, reward is
summed over them, and the first `Done` reason ends the step early.

We do NOT subclass rSoccer's SSLBaseEnv. That class hardcodes coordinate
conventions, referee handling, and opponent behaviour that we need to
control ourselves. We use rSoccer for its renderer and as a reference.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np

from tbots.backends.base import Scenario, SimBackend
from tbots.core.command import RobotCommand
from tbots.core.geometry import BALL_RADIUS
from tbots.core.state import WorldState
from tbots.core.transition import Done, Transition
from tbots.perception.sim import Perception, PerceptionConfig
from tbots.rl.rewards.registry import CompositeReward

ScenarioFn = Callable[[np.random.Generator], Scenario]
OUT_OF_BOUNDS_POLICIES = ("terminate", "restart")


class SSLEnv(gym.Env):
    """Base class. Subclasses define spaces, `_observe` and `_decode`."""

    metadata = {"render_modes": ["human", "vision"], "render_fps": 60}

    def __init__(
        self,
        backend: SimBackend,
        reward: CompositeReward,
        scenario_fn: ScenarioFn,
        *,
        perception: Perception | None = None,
        max_episode_ticks: int = 7200,
        ego_id: int | None = None,
        out_of_bounds: str = "terminate",
        render_mode: str | None = None,
    ) -> None:
        if out_of_bounds not in OUT_OF_BOUNDS_POLICIES:
            raise ValueError(f"out_of_bounds must be one of {OUT_OF_BOUNDS_POLICIES}")
        if out_of_bounds == "restart":
            # A synthetic free kick needs the SyntheticReferee (TASK-052) and
            # is the tactics env's default (TASK-051). Not silently ignored.
            raise NotImplementedError("out_of_bounds='restart' is TASK-051/TASK-052")
        if max_episode_ticks <= 0:
            raise ValueError("max_episode_ticks must be positive")
        self.backend = backend
        self.reward = reward
        self.scenario_fn = scenario_fn
        self.perception = perception or Perception(PerceptionConfig(mode="truth"),
                                                   dt=backend.dt,
                                                   geometry=backend.geometry)
        self.max_episode_ticks = max_episode_ticks
        self.ego_id = ego_id
        self.out_of_bounds = out_of_bounds
        self.render_mode = render_mode
        self._ticks = 0
        self._truth: WorldState | None = None
        self._perceived: WorldState | None = None
        self._publisher: Any = None  # set up lazily for render_mode == "vision"

    # -- subclass hooks -----------------------------------------------------

    def _observe(self, world: WorldState) -> Any:
        """Observation from the PERCEIVED world."""
        raise NotImplementedError

    def _decode(self, action: Any, world: WorldState) -> Sequence[Sequence[RobotCommand]]:
        """One list of our commands per physics tick this step covers.
        `world` is the perceived world the action was chosen on."""
        raise NotImplementedError

    def _opponent_commands(self, truth: WorldState) -> Sequence[RobotCommand]:
        """Their commands for the next tick. Default: they stand still."""
        return ()

    def _task_done(self, truth: WorldState, prev: WorldState) -> Done | None:
        """A subclass's own end condition (skill success/failure)."""
        return None

    def _on_reset(self, truth: WorldState) -> None:
        """Clear per-episode subclass state (codecs, skill instances)."""

    # -- the two worlds -----------------------------------------------------

    @property
    def truth(self) -> WorldState:
        assert self._truth is not None, "call reset() first"
        return self._truth

    @property
    def perceived(self) -> WorldState:
        assert self._perceived is not None, "call reset() first"
        return self._perceived

    # -- gym API ------------------------------------------------------------

    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        scenario = self.scenario_fn(self.np_random)
        self._truth = self.backend.reset(scenario)
        self._ticks = 0
        self.reward.reset()
        self._on_reset(self._truth)
        # Perception gets its own stream, drawn from the env's: one seed
        # determines both, and changing the perception mode does not change
        # which scenarios an episode sees (TASK-080).
        prng = np.random.default_rng(int(self.np_random.integers(2**63)))
        self._perceived = self.perception.reset(self._truth, prng)
        if self.render_mode is not None:
            self.render()
        return self._observe(self._perceived), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        assert self._truth is not None and self._perceived is not None, "call reset() first"
        per_tick = self._decode(action, self._perceived)
        if not per_tick:
            raise ValueError("_decode returned no ticks")

        total = 0.0
        terms: dict[str, float] = {}
        done: Done | None = None
        for commands in per_tick:
            prev = self._truth
            truth = self.backend.step(commands, self._opponent_commands(prev))
            self._truth = truth
            self._perceived = self.perception.observe(truth)
            self._ticks += 1
            done = self._done_reason(truth, prev)
            tr = Transition(world=truth, prev=prev, commands=tuple(commands),
                            ego_id=self.ego_id, geometry=self.backend.geometry,
                            dt=self.backend.dt, done=done)
            total += self.reward(tr)
            for name, v in self.reward.last.items():
                terms[name] = terms.get(name, 0.0) + v
            if self.render_mode is not None:
                self.render()
            if done is not None:
                break

        terminated = done is not None and done.terminal
        truncated = done is Done.TIMEOUT
        info: dict[str, Any] = {"reward_terms": terms,
                                "done": done.name if done is not None else None}
        return self._observe(self._perceived), total, terminated, truncated, info

    def _done_reason(self, truth: WorldState, prev: WorldState) -> Done | None:
        """Goals first, then out-of-bounds, then the subclass, then time."""
        g = self.backend.geometry
        b = truth.ball
        # The whole ball must cross the line (SSL rules), hence the radius.
        past_x = abs(b.x) > g.half_length + BALL_RADIUS
        if past_x and abs(b.y) < g.goal_width / 2.0:
            return Done.GOAL_US if b.x > 0 else Done.GOAL_THEM
        if past_x or abs(b.y) > g.half_width + BALL_RADIUS:
            return Done.OUT_OF_BOUNDS
        task = self._task_done(truth, prev)
        if task is not None:
            return task
        if self._ticks >= self.max_episode_ticks:
            return Done.TIMEOUT
        return None

    def render(self) -> None:
        if self.render_mode == "vision":
            if self._publisher is None:
                from tbots.net.vision_publisher import VisionPublisher
                self._publisher = VisionPublisher(geometry=self.backend.geometry)
                self._publisher.publish_geometry()
            self._publisher.publish(self._truth)

    def close(self) -> None:
        if self._publisher is not None:
            self._publisher.close()
            self._publisher = None
        self.backend.close()
