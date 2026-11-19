"""TASK-076: reward terms take a frozen Transition, and the env holds two
worlds -- truth for reward and termination, perception for observations."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from tbots.backends.base import Scenario
from tbots.backends.rsim import RSimBackend
from tbots.core.command import RobotCommand
from tbots.core.state import WorldState
from tbots.core.transition import Done, Transition
from tbots.perception.sim import Perception, PerceptionConfig
from tbots.rl.envs.base import SSLEnv
from tbots.rl.rewards.registry import CompositeReward, register_reward, reward_names

SEEN: list[Transition] = []


@register_reward("_test_reads_commands_and_done")
class ReadsCommandsAndDone:
    """Penalises commanded speed; pays out on a goal. Records what it saw."""

    def reset(self) -> None:
        pass

    def __call__(self, tr: Transition) -> float:
        SEEN.append(tr)
        energy = sum(c.vx ** 2 + c.vy ** 2 for c in tr.commands)
        return (10.0 if tr.done is Done.GOAL_US else 0.0) - 0.01 * energy


class _DriveEnv(SSLEnv):
    """The smallest concrete SSLEnv: robot 0 drives forward at action[0] m/s.
    Observation is the perceived ball x, so the test can tell the worlds apart."""

    action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
    observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float64)

    def _observe(self, world: WorldState) -> np.ndarray:
        return np.array([world.ball.x])

    def _decode(self, action, world: WorldState) -> list[list[RobotCommand]]:
        return [[RobotCommand(0, vx=float(action[0]))]]


def _env(perception: Perception | None = None, ball_x: float = 3.0,
         max_ticks: int = 600) -> _DriveEnv:
    backend = RSimBackend(n_us=1, n_them=0)

    def scenario(rng: np.random.Generator) -> Scenario:
        # The robot faces +x with the ball right in front, the goal 1.5 m on.
        return Scenario(ball=(ball_x, 0.0, 0.0, 0.0), us=((ball_x - 0.15, 0.0, 0.0),))

    return _DriveEnv(backend,
                     CompositeReward([{"name": "_test_reads_commands_and_done"}]),
                     scenario, perception=perception, max_episode_ticks=max_ticks,
                     ego_id=0)


def test_reward_spec_rejects_unknown_terms_and_registry_lists_them():
    assert "alive" in reward_names()
    with pytest.raises(KeyError, match="unknown reward term"):
        CompositeReward([{"name": "no_such_term"}])


def test_a_term_reading_commands_and_done_runs_for_one_episode():
    SEEN.clear()
    env = _env()
    env.reset(seed=0)
    total, info, terminated, truncated = 0.0, {}, False, False
    while not (terminated or truncated):
        _, r, terminated, truncated, info = env.step(np.array([1.0], np.float32))
        total += r
    env.close()

    # Pushing the ball at 1 m/s from 1.5 m out scores, well inside 10 s.
    assert terminated and not truncated
    assert info["done"] == "GOAL_US"
    assert SEEN[-1].done is Done.GOAL_US
    assert all(tr.done is None for tr in SEEN[:-1])
    # The term saw the commands we sent, on the ego robot, every step.
    assert all(tr.commands == (RobotCommand(0, vx=1.0),) for tr in SEEN)
    assert all(tr.ego_id == 0 and tr.dt == pytest.approx(1 / 60) for tr in SEEN)
    assert info["reward_terms"]["_test_reads_commands_and_done"] == pytest.approx(10.0 - 0.01)
    assert total == pytest.approx(10.0 - 0.01 * len(SEEN))


def test_timeout_is_a_truncation_and_the_term_sees_it():
    SEEN.clear()
    env = _env(max_ticks=5)
    env.reset(seed=0)
    for _ in range(5):
        _, _, terminated, truncated, info = env.step(np.array([0.0], np.float32))
    assert truncated and not terminated
    assert info["done"] == "TIMEOUT" and SEEN[-1].done is Done.TIMEOUT
    env.close()


def test_ball_leaving_the_field_terminates_a_skill_env():
    env = _env(ball_x=-3.0)

    def scenario(rng):  # ball rolling out over the touchline at y = +3
        return Scenario(ball=(0.0, 2.8, 0.0, 3.0), us=((-2.0, 0.0, 0.0),))

    env.scenario_fn = scenario
    env.reset(seed=0)
    for _ in range(120):
        _, _, terminated, _, info = env.step(np.array([0.0], np.float32))
        if terminated:
            break
    assert info["done"] == "OUT_OF_BOUNDS"
    env.close()


def test_observations_come_from_perception_and_reward_from_truth():
    SEEN.clear()
    perception = Perception(PerceptionConfig(
        mode="noisy", latency_s=0.05, robot_pos_std_m=0.0, robot_angle_std_rad=0.0,
        ball_pos_std_m=0.0, dropout=0.0), dt=1 / 60)
    env = _env(perception=perception, ball_x=0.0)
    obs0, _ = env.reset(seed=0)
    truth_ball_x = []
    observed = []
    for _ in range(30):
        obs, *_ = env.step(np.array([1.0], np.float32))
        observed.append(float(obs[0]))
        truth_ball_x.append(SEEN[-1].world.ball.x)
    d = perception.sim.params.delay_steps
    assert d == 3
    # Observation k is truth k - d; the reward term saw truth k.
    assert observed[d:] == pytest.approx(truth_ball_x[:-d])
    assert truth_ball_x[-1] > observed[-1] + 0.01
    env.close()
