"""TASK-006: make_env is the only way an environment is built.

Gate: make_env builds every combination in configs/; rl/train.py,
apps/eval.py and rl/vec.py call nothing else.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import numpy as np
import pytest

from tbots.rl.envs.skill_env import SkillEnv
from tbots.rl.factory import CONFIG_DIR, compose_config, make_env, resolve_config
from tbots.rl.tasks import task_names

ENVS = sorted(p.stem for p in (CONFIG_DIR / "env").glob("*.yaml"))
REWARDS = [None, *sorted(p.stem for p in (CONFIG_DIR / "reward").glob("*.yaml"))]
MODES = ["truth", "noisy", "tracked"]


def test_the_config_groups_are_what_we_think():
    assert {"skill", "div_b_6v6"} <= set(ENVS)
    assert "go_to_ball" in task_names()


@pytest.mark.parametrize(("env", "task", "reward", "mode"),
                         list(itertools.product(ENVS, task_names(), REWARDS, MODES)))
def test_make_env_builds_every_combination(env, task, reward, mode):
    overrides = [f"env={env}", f"task={task}", f"env.perception.mode={mode}"]
    if reward is not None:
        overrides.append(f"reward={reward}")
    cfg = compose_config(overrides)
    e = make_env(cfg)
    try:
        assert isinstance(e.unwrapped, SkillEnv)
        assert e.unwrapped.perception.mode == mode
        assert e.unwrapped.backend.n_us == cfg.env.n_us
        obs, _ = e.reset(seed=0)
        assert obs.shape == e.observation_space.shape
        e.action_space.seed(0)
        obs, r, *_ = e.step(e.action_space.sample())
        assert np.isfinite(r)
    finally:
        e.close()


def test_the_task_default_reward_is_used_and_recorded_unless_overridden():
    cfg = resolve_config(compose_config(["task=go_to_ball"]))
    names = [t["name"] for t in cfg.reward.terms]
    assert names == ["ball_progress", "outcome", "time_cost"]
    cfg = resolve_config(compose_config(["reward=example"]))
    assert [t["name"] for t in cfg.reward.terms] == ["alive"]


def test_codec_overrides_from_the_env_config_reach_the_codec():
    cfg = compose_config(["+env.codec.action_repeat=3"])
    e = make_env(cfg)
    assert e.unwrapped.codec.action_repeat == 3
    e.close()


def test_unknown_env_kind_and_old_perception_keys_fail_loudly():
    cfg = compose_config([])
    cfg.env.kind = "tactics"
    with pytest.raises(NotImplementedError, match="TASK-051"):
        make_env(cfg)
    cfg = compose_config(["+env.perception.vision_latency_ms=[20,45]"])
    with pytest.raises(KeyError, match="unknown perception key"):
        make_env(cfg)


# -- "call nothing else" -----------------------------------------------------

SRC = Path(__file__).parents[1] / "src" / "tbots"
CONSUMERS = [SRC / "rl" / "train.py", SRC / "rl" / "vec.py", SRC / "apps" / "eval.py",
             SRC / "rl" / "evaluate.py"]
FORBIDDEN = {"RSimBackend", "SkillEnv", "SSLEnv", "TacticsEnv", "Perception",
             "PerceptionSim", "Tracker", "CompositeReward", "build_codec"}


def test_the_consumers_that_must_exist_do():
    for name in ("train.py", "vec.py"):
        assert (SRC / "rl" / name).is_file()


@pytest.mark.parametrize("path", [p for p in CONSUMERS if p.exists()], ids=lambda p: p.name)
def test_consumers_build_environments_only_through_make_env(path):
    tree = ast.parse(path.read_text())
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    imported = {a.asname or a.name for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not (used | imported) & FORBIDDEN, f"{path.name} builds an env by hand"
