"""PerceptionSim: what a camera would have told us, made from rSim truth.

rSim hands the environment perfect, instantaneous state. A real vision
frame is 20-45 ms stale, a few millimetres off, and now and then missing an
object altogether. A policy trained on truth learns to lean on information
it can never have at a match. This module puts those three effects back,
on SEMANTIC state, below the environment (ARCHITECTURE section 7):

    rSim truth --> PerceptionSim --> WorldState            (`noisy`)
                                 \\-> DetectionFrame       (`tracked`)
                                       + RobotFeedback --> Tracker --> WorldState

`noisy` is truth delayed, jittered, and with dropped objects held at their
last perceived pose and marked `visible=False`. No tracker; velocities are
the delayed truth's. It is for day-one training.

`tracked` builds the same `DetectionFrame`s `net/vision.py` produces and runs
them through the same `Tracker` a match runs, so a policy trained on it has
seen the exact perception it will get. Anything exported must be trained in
this mode.

`truth` passes rSim's state through. Debugging only.

Every parameter is a range, sampled once per episode, because the point is a
policy that works across the plausible range, not one tuned to one lab.

Delay is quantised to whole physics ticks: rSim only has states at tick
boundaries, so a 30 ms latency at 60 Hz is 2 ticks (33 ms). That keeps the
delayed state a real rSim state rather than an interpolation nobody
simulated.

NOT MODELLED: command latency (a command reaching the robot late). It is
not perception; the superseded `rl/wrappers/domain_rand.py` had a queue for
it. `actuation_delay_s` here only tells the tracker how far to predict.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np

from tbots.core.geometry import DIV_B, FieldGeometry
from tbots.core.perspective import IDENTITY
from tbots.core.state import (
    DetectionBall,
    DetectionFrame,
    DetectionRobot,
    RobotFeedback,
    RobotState,
    WorldState,
)
from tbots.core.units import wrap_angle
from tbots.perception.tracker import Tracker

MODES = ("truth", "noisy", "tracked")

Range = tuple[float, float]


def _range(value: Any, name: str, hi: float | None = None) -> Range:
    """A scalar `v` means the fixed range (v, v)."""
    if isinstance(value, (int, float)):
        lo_hi = (float(value), float(value))
    else:
        seq = list(value)
        if len(seq) != 2:
            raise ValueError(f"{name} must be a number or [min, max], got {value!r}")
        lo_hi = (float(seq[0]), float(seq[1]))
    lo, up = lo_hi
    if lo < 0 or up < lo or (hi is not None and up > hi):
        bound = f" <= {hi}" if hi is not None else ""
        raise ValueError(f"{name} must satisfy 0 <= min <= max{bound}, got {value!r}")
    return lo_hi


@dataclass(frozen=True, slots=True)
class PerceptionConfig:
    """`perception:` in the env config. Distances in meters, angles in
    radians, times in seconds. Each range is sampled once per episode."""

    mode: str = "noisy"
    latency_s: Range = (0.020, 0.045)
    robot_pos_std_m: Range = (0.0, 0.005)
    robot_angle_std_rad: Range = (0.0, 0.02)
    ball_pos_std_m: Range = (0.0, 0.005)
    # Probability that one object is missing from one frame.
    dropout: Range = (0.0, 0.04)
    # How far ahead of now the tracker predicts (tracked mode only).
    actuation_delay_s: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"perception mode must be one of {MODES}, got {self.mode!r}")
        for f in ("latency_s", "robot_pos_std_m", "robot_angle_std_rad", "ball_pos_std_m"):
            object.__setattr__(self, f, _range(getattr(self, f), f))
        object.__setattr__(self, "dropout", _range(self.dropout, "dropout", hi=1.0))
        if self.actuation_delay_s < 0:
            raise ValueError("actuation_delay_s must be non-negative")

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> PerceptionConfig:
        """From a config block. An unknown key is an error, not ignored: the
        old `domain_randomization` keys would otherwise be silently dropped."""
        known = {f.name for f in fields(cls)}
        for k in cfg:
            if k not in known:
                raise KeyError(f"unknown perception key {k!r}; known: {sorted(known)}")
        return cls(**dict(cfg))


@dataclass(frozen=True, slots=True)
class EpisodeParams:
    """One episode's draw from a `PerceptionConfig`."""

    delay_steps: int
    robot_pos_std: float
    robot_angle_std: float
    ball_pos_std: float
    dropout: float


class PerceptionSim:
    """Truth in, delayed and corrupted perception out. One per environment.

    Call `reset()` with the episode's first truth state, then exactly one of
    `noisy()` or `tracked()` per physics tick with the newest truth.
    """

    def __init__(self, config: PerceptionConfig, dt: float) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.config = config
        self.dt = dt
        self._rng: np.random.Generator = np.random.default_rng(0)
        self.params = EpisodeParams(0, 0.0, 0.0, 0.0, 0.0)
        self._history: deque[WorldState] = deque(maxlen=1)
        self._prev: WorldState | None = None       # last noisy() output
        self._last_emitted: float | None = None    # last tracked() frame's t

    # -- episode ------------------------------------------------------------

    def reset(self, truth: WorldState, rng: np.random.Generator) -> EpisodeParams:
        self._rng = rng
        c = self.config
        u = rng.uniform
        latency = u(*c.latency_s)
        self.params = EpisodeParams(
            delay_steps=int(round(latency / self.dt)),
            robot_pos_std=float(u(*c.robot_pos_std_m)),
            robot_angle_std=float(u(*c.robot_angle_std_rad)),
            ball_pos_std=float(u(*c.ball_pos_std_m)),
            dropout=float(u(*c.dropout)),
        )
        # Until `delay_steps` ticks have passed, the newest thing a camera
        # could have seen is the episode's first state.
        self._history = deque([truth], maxlen=self.params.delay_steps + 1)
        self._prev = None
        self._last_emitted = None
        return self.params

    def _push(self, truth: WorldState) -> WorldState:
        self._history.append(truth)
        return self._history[0]

    # -- noisy ------------------------------------------------------------

    def initial_noisy(self) -> WorldState:
        """The perceived first state. Noisy, but nothing dropped: an object
        missing from the very first frame has no last pose to hold."""
        self._prev = self._corrupt(self._history[0], allow_dropout=False)
        return self._prev

    def noisy(self, truth: WorldState) -> WorldState:
        """`truth` from `delay_steps` ago, with noise and dropout."""
        self._prev = self._corrupt(self._push(truth), allow_dropout=True)
        return self._prev

    def _drop(self, allow: bool) -> bool:
        p = self.params.dropout
        # Draw even when not allowed, so the RNG stream does not depend on it.
        r = self._rng.random()
        return allow and r < p

    def _jitter(self, std: float) -> float:
        return float(self._rng.normal(0.0, std)) if std > 0 else 0.0

    def _corrupt_robot(self, r: RobotState, held: RobotState | None,
                       allow_dropout: bool) -> RobotState:
        if self._drop(allow_dropout and held is not None):
            assert held is not None
            return replace(held, visible=False)
        sp, sa = self.params.robot_pos_std, self.params.robot_angle_std
        if sp == 0 and sa == 0:
            return r
        return replace(r, x=r.x + self._jitter(sp), y=r.y + self._jitter(sp),
                       theta=wrap_angle(r.theta + self._jitter(sa)) if sa > 0 else r.theta)

    def _corrupt(self, w: WorldState, allow_dropout: bool) -> WorldState:
        prev = self._prev

        def team(robots: dict[int, RobotState], held: dict[int, RobotState]):
            return {i: self._corrupt_robot(r, held.get(i), allow_dropout)
                    for i, r in robots.items()}

        us = team(w.us, prev.us if prev else {})
        them = team(w.them, prev.them if prev else {})
        if self._drop(allow_dropout and prev is not None):
            assert prev is not None
            ball = replace(prev.ball, visible=False)
        elif self.params.ball_pos_std > 0:
            s = self.params.ball_pos_std
            ball = replace(w.ball, x=w.ball.x + self._jitter(s), y=w.ball.y + self._jitter(s))
        else:
            ball = w.ball
        return replace(w, us=us, them=them, ball=ball)

    # -- tracked ------------------------------------------------------------

    def initial_tracked(self, truth: WorldState) -> tuple[list[DetectionFrame],
                                                         dict[int, RobotFeedback]]:
        """The first frame, complete, plus feedback."""
        return self._emit(self._history[0], allow_dropout=False), self._feedback(truth)

    def tracked(self, truth: WorldState) -> tuple[list[DetectionFrame],
                                                 dict[int, RobotFeedback]]:
        """New camera frames (zero or one) and our robots' telemetry.

        Frames are what `net/vision.py` would hand the tracker: field frame,
        blue and yellow, positions only. rSim is `IDENTITY`, so we are blue.
        Feedback comes off the robots over the radio, not through the camera,
        so it is the CURRENT truth, not the delayed one.
        """
        return self._emit(self._push(truth), allow_dropout=True), self._feedback(truth)

    def _emit(self, w: WorldState, allow_dropout: bool) -> list[DetectionFrame]:
        # A delayed state already sent (the first few ticks of an episode,
        # before history has filled) is not a new camera frame.
        if self._last_emitted is not None and w.t <= self._last_emitted:
            return []
        self._last_emitted = w.t
        sp, sa = self.params.robot_pos_std, self.params.robot_angle_std

        def det(robots: dict[int, RobotState]) -> tuple[DetectionRobot, ...]:
            out = []
            for i, r in robots.items():
                if self._drop(allow_dropout):
                    continue
                out.append(DetectionRobot(robot_id=i, x=r.x + self._jitter(sp),
                                          y=r.y + self._jitter(sp),
                                          theta=wrap_angle(r.theta + self._jitter(sa))))
            return tuple(out)

        blue, yellow = det(w.us), det(w.them)
        balls: tuple[DetectionBall, ...] = ()
        if not self._drop(allow_dropout):
            s = self.params.ball_pos_std
            balls = (DetectionBall(x=w.ball.x + self._jitter(s),
                                   y=w.ball.y + self._jitter(s), z=w.ball.z),)
        return [DetectionFrame(t_capture=w.t, camera_id=0, balls=balls,
                               robots_blue=blue, robots_yellow=yellow)]

    @staticmethod
    def _feedback(truth: WorldState) -> dict[int, RobotFeedback]:
        return {i: RobotFeedback(robot_id=i, t=truth.t, has_ball=r.has_ball)
                for i, r in truth.us.items()}


class Perception:
    """One perception stage for an environment, in whichever mode.

    `reset(truth, rng)` at episode start and `observe(truth)` after every
    physics tick, each returning the perceived `WorldState` the observation
    builders see. Reward and termination never see it (ARCHITECTURE 7).
    """

    def __init__(self, config: PerceptionConfig, dt: float,
                 geometry: FieldGeometry = DIV_B) -> None:
        self.config = config
        self.mode = config.mode
        self._geom = geometry
        self.sim = PerceptionSim(config, dt)
        self._tracker: Tracker | None = None

    def reset(self, truth: WorldState, rng: np.random.Generator) -> WorldState:
        if self.mode == "truth":
            return truth
        self.sim.reset(truth, rng)
        if self.mode == "noisy":
            return self.sim.initial_noisy()
        # A fresh tracker per episode: a reset teleports everything, and a
        # filter that remembers the last episode's velocities would carry
        # them across the jump for a few frames.
        self._tracker = Tracker(self._geom, actuation_delay=self.config.actuation_delay_s)
        return self._fuse(truth, *self.sim.initial_tracked(truth))

    def observe(self, truth: WorldState) -> WorldState:
        if self.mode == "truth":
            return truth
        if self.mode == "noisy":
            return self.sim.noisy(truth)
        return self._fuse(truth, *self.sim.tracked(truth))

    def _fuse(self, truth: WorldState, frames: list[DetectionFrame],
              feedback: dict[int, RobotFeedback]) -> WorldState:
        assert self._tracker is not None
        world = self._tracker.update(frames, truth.t, IDENTITY)
        # Exactly what NetworkBackend.observe() does with real telemetry:
        # dribbler contact comes from the robot, not the camera.
        telemetry = {i: fb for i, fb in feedback.items() if i in world.us}
        us = {i: replace(r, has_ball=telemetry[i].has_ball) if i in telemetry else r
              for i, r in world.us.items()}
        return replace(world, us=us, game=truth.game, telemetry=telemetry)
