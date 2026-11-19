"""make_env: the ONE way to build an environment.

    backend -> Perception (PerceptionSim, Tracker) -> env -> wrappers

Train, eval, the vector env, tests and the throughput benchmark all call
this, so there is no second hand-built stack to drift from the first --
the classic "works in training, not in eval" bug (ARCHITECTURE section 9).

`cfg` is the composed Hydra config (configs/config.yaml), or a run
directory's recorded config.yaml loaded back with OmegaConf.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from tbots.backends.rsim import RSimBackend
from tbots.core.geometry import DIV_B
from tbots.perception.sim import Perception, PerceptionConfig
from tbots.rl.artifacts import checkpoint_metadata
from tbots.rl.envs.skill_env import SkillEnv
from tbots.rl.obs.builders import obs_size
from tbots.rl.rewards.registry import CompositeReward
from tbots.rl.tasks import get_task

# src/tbots/rl/factory.py -> the repository root.
CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def compose_config(overrides: list[str] | None = None) -> DictConfig:
    """The same config `python -m tbots.rl.train <overrides>` would see."""
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose("config", overrides=list(overrides or []))


def load_reward_spec(name: str) -> list[dict[str, Any]]:
    path = CONFIG_DIR / "reward" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no reward config {path}")
    spec = OmegaConf.to_container(OmegaConf.load(path).terms, resolve=True)
    assert isinstance(spec, list)
    return spec  # type: ignore[return-value]


def resolve_config(cfg: DictConfig) -> DictConfig:
    """A copy of `cfg` with everything implicit made explicit: the task's
    default reward is filled in when `reward=` was not given. Write THIS to
    the run directory, so a run reloads the reward it trained on even if the
    task's YAML has changed since."""
    cfg = cfg.copy()
    if cfg.get("reward") is None:
        with open_dict(cfg):
            cfg.reward = {"terms": load_reward_spec(get_task(cfg.task).reward)}
    return cfg


def make_env(cfg: DictConfig, *, render_mode: str | None = None) -> gym.Env:
    cfg = resolve_config(cfg)
    e = cfg.env
    if e.kind != "skill":
        raise NotImplementedError(f"env kind {e.kind!r}: only 'skill' exists; the tactics "
                                  "env is TASK-051")
    if e.division != "B":
        raise ValueError("only Division B is supported (core.geometry.DIV_B)")
    if e.backend.kind != "rsim":
        raise ValueError(f"training needs a SimBackend; {e.backend.kind!r} is not one we build")

    dt = 1.0 / float(e.control_hz)
    backend = RSimBackend(n_us=int(e.n_us), n_them=int(e.n_them), dt=dt,
                          geometry=DIV_B, field_type=int(e.backend.field_type))
    perception_cfg = OmegaConf.to_container(e.perception, resolve=True)
    assert isinstance(perception_cfg, dict)
    perception = Perception(PerceptionConfig.from_mapping(perception_cfg),  # type: ignore[arg-type]
                            dt=dt, geometry=DIV_B)
    reward_spec = OmegaConf.to_container(cfg.reward.terms, resolve=True)
    codec = OmegaConf.to_container(e.codec, resolve=True) if e.get("codec") is not None else {}
    env: gym.Env = SkillEnv(
        backend, get_task(cfg.task), CompositeReward(reward_spec),  # type: ignore[arg-type]
        n_us=int(e.n_us), n_them=int(e.n_them), perception=perception,
        codec_overrides=codec,  # type: ignore[arg-type]
        out_of_bounds=str(e.out_of_bounds), render_mode=render_mode)
    # Wrappers go here, in this order, for every caller. There are none yet:
    # the trainer keeps its own episode statistics.
    return env


def env_metadata(env: gym.Env, cfg: DictConfig) -> dict[str, Any]:
    """The checkpoint metadata (TASK-059) describing `env` as `make_env`
    built it from `cfg`: what a policy for it observes and emits."""
    u = env.unwrapped
    assert isinstance(u, SkillEnv)
    return checkpoint_metadata(
        obs_builder=u.task.obs_builder, obs_size=obs_size(u.task.obs_builder),
        codec=u.codec.spec(), dt=u.backend.dt, perception_mode=u.perception.mode,
        stage=cfg.train.get("stage") if "train" in cfg else None, task=str(cfg.task))
