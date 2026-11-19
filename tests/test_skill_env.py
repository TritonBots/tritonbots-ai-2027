"""TASK-050: SkillEnv over a SkillTask. Gate: one episode of go_to_ball runs
in truth, noisy and tracked modes."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from omegaconf import OmegaConf

from tbots.backends.rsim import RSimBackend
from tbots.core.transition import Done
from tbots.perception.sim import Perception, PerceptionConfig
from tbots.rl.envs.skill_env import SkillEnv
from tbots.rl.rewards.registry import CompositeReward
from tbots.rl.tasks import get_task, task_names

REWARD_YAML = Path(__file__).parents[1] / "configs" / "reward" / "go_to_ball.yaml"


def _env(mode: str, n_us: int = 1, n_them: int = 0) -> SkillEnv:
    backend = RSimBackend(n_us=n_us, n_them=n_them)
    spec = OmegaConf.to_container(OmegaConf.load(REWARD_YAML).terms)
    return SkillEnv(backend, get_task("go_to_ball"), CompositeReward(spec),  # type: ignore[arg-type]
                    n_us=n_us, n_them=n_them,
                    perception=Perception(PerceptionConfig(mode=mode), dt=backend.dt))


def _toward_ball(obs: np.ndarray) -> dict:
    """A hand-written policy on the observation: drive at the ball, ego frame."""
    bx, by = float(obs[3]), float(obs[4])
    n = math.hypot(bx, by) or 1.0
    return {"move": np.array([bx / n, by / n, 0.0], np.float32), "flags": np.zeros(2, np.int8)}


def test_go_to_ball_is_registered():
    assert "go_to_ball" in task_names()


@pytest.mark.parametrize("mode", ["truth", "noisy", "tracked"])
def test_one_episode_of_go_to_ball_runs(mode):
    env = _env(mode)
    obs, _ = env.reset(seed=4)
    assert env.observation_space.contains(obs)
    done, steps, ret, info = False, 0, 0.0, {}
    while not done:
        obs, r, term, trunc, info = env.step(_toward_ball(obs))
        ret += r
        steps += 1
        done = term or trunc
    env.close()
    # Driving at the ball reaches it well inside the 8 s limit in every mode.
    assert info["done"] == Done.SKILL_SUCCESS.name, (mode, info)
    assert steps < 8 * 60
    assert ret > 0.5


def test_random_actions_run_with_robots_on_both_sides_and_end_by_time():
    env = _env("tracked", n_us=3, n_them=3)
    env.reset(seed=0)
    env.action_space.seed(0)
    steps, done, info = 0, False, {}
    while not done:
        _, _, term, trunc, info = env.step(env.action_space.sample())
        steps += 1
        done = term or trunc
    env.close()
    assert info["done"] in {"TIMEOUT", "SKILL_SUCCESS", "OUT_OF_BOUNDS"}
    assert steps <= 8 * 60


def test_passes_the_gymnasium_env_checker():
    env = _env("noisy")
    check_env(env, skip_render_check=True)
    env.close()


def test_same_seed_same_episode_start():
    a, b = _env("noisy"), _env("noisy")
    oa, _ = a.reset(seed=123)
    ob, _ = b.reset(seed=123)
    np.testing.assert_array_equal(oa, ob)
    a.close()
    b.close()
