"""Evaluation: a deterministic policy on a fixed seed set, recorded.

Shared by the trainer's per-checkpoint eval (TASK-056) and `apps/eval.py`
(TASK-058). Training-rollout statistics are measured on a moving
distribution with exploration noise and are never used to compare runs;
these numbers are (ARCHITECTURE section 9).

Always `tracked` perception unless told otherwise: an evaluation should
see what a match sees.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from tbots.core.transition import Done, Transition
from tbots.rl.algo.ppo import Agent
from tbots.rl.artifacts import load_checkpoint
from tbots.rl.factory import env_metadata, make_env, resolve_config
from tbots.rl.rollout import RolloutRecorder, prune, save_rollout

# Evaluation seeds live far from training seeds (train.seed + worker index),
# so an eval episode is never one the policy trained on.
EVAL_SEED_BASE = 1_000_000


def eval_seeds(n: int) -> list[int]:
    return list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + n))


@dataclass
class EpisodeResult:
    seed: int
    ret: float
    ticks: int
    seconds: float
    done: str | None
    goals_us: int
    goals_them: int
    terms: dict[str, float] = field(default_factory=dict)
    rollout: str | None = None

    @property
    def goal_difference(self) -> int:
        return self.goals_us - self.goals_them

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "goal_difference": self.goal_difference}


def eval_config(cfg: DictConfig, perception_mode: str = "tracked") -> DictConfig:
    cfg = resolve_config(cfg)
    cfg.env.perception.mode = perception_mode
    return cfg


def build_agent(env: Any, hidden: int) -> Agent:
    return Agent(int(env.observation_space.shape[0]), env.action_space, hidden=hidden)


def load_policy(checkpoint: str | Path, env: Any, cfg: DictConfig) -> Agent:
    """The checkpoint's policy, refused if it cannot run on `env`."""
    state = load_checkpoint(checkpoint, expected_metadata=env_metadata(env, cfg))
    agent = build_agent(env, int(state["agent"]["hidden"]))
    agent.load_state_dict(state["model"])
    agent.eval()
    return agent


def evaluate(agent: Agent, cfg: DictConfig, seeds: Sequence[int], *,
             record_dir: str | Path | None = None, keep: int = 10,
             label: str = "eval", render_mode: str | None = None,
             realtime: bool = False) -> list[EpisodeResult]:
    """Run one deterministic episode per seed on `make_env(cfg)`.

    `cfg` is used as given -- pass it through `eval_config` for `tracked`.
    With `record_dir`, each episode is saved as `<label>_seed<seed>.npz`
    and only the newest `keep` rollouts in that directory survive.
    """
    env = make_env(cfg, render_mode=render_mode)
    u = env.unwrapped
    names = [n for n, _, _ in u.reward.terms]
    results: list[EpisodeResult] = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            rec = RolloutRecorder(names, meta={
                "task": str(cfg.task), "seed": int(seed), "dt": u.backend.dt,
                "perception_mode": u.perception.mode, "label": label})
            rec.start(u.truth)
            goals = {"us": 0, "them": 0}
            wall0 = time.monotonic()

            def hook(tr: Transition, terms: dict[str, float], rec=rec, goals=goals,
                     wall0=wall0) -> None:
                rec.on_tick(tr, terms)
                if tr.done is Done.GOAL_US:
                    goals["us"] += 1
                elif tr.done is Done.GOAL_THEM:
                    goals["them"] += 1
                if realtime:
                    lag = tr.world.t - (time.monotonic() - wall0)
                    if lag > 0:
                        time.sleep(lag)

            u.tick_hook = hook
            ret, done = 0.0, False
            info: dict[str, Any] = {}
            while not done:
                with torch.no_grad():
                    a, _, _ = agent.act(torch.as_tensor(obs, dtype=torch.float32)[None],
                                        deterministic=True)
                obs, r, term, trunc, info = env.step(_single(agent.head.to_env(a)))
                ret += float(r)
                done = term or trunc
            u.tick_hook = None
            rollout = rec.finish()
            path = None
            if record_dir is not None:
                path = str(save_rollout(Path(record_dir) / f"{label}_seed{seed}.npz", rollout))
            results.append(EpisodeResult(
                seed=int(seed), ret=ret, ticks=len(rollout) - 1,
                seconds=float(rollout.t[-1] - rollout.t[0]), done=info.get("done"),
                goals_us=goals["us"], goals_them=goals["them"],
                terms=rollout.term_totals(), rollout=path))
    finally:
        env.close()
    if record_dir is not None:
        prune(record_dir, keep)
    return results


def _single(batched: Any) -> Any:
    """Strip the batch dimension the policy works in."""
    if isinstance(batched, dict):
        return {k: v[0] for k, v in batched.items()}
    return batched[0]


def summarize(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    """Per-episode records plus the distributions a run is compared on."""
    rets = np.array([r.ret for r in results], dtype=np.float64)
    names = sorted({k for r in results for k in r.terms})
    terms = {n: [r.terms.get(n, 0.0) for r in results] for n in names}
    reasons: dict[str, int] = {}
    for r in results:
        reasons[str(r.done)] = reasons.get(str(r.done), 0) + 1
    return {
        "episodes": [r.to_dict() for r in results],
        "mean_return": float(rets.mean()) if len(rets) else 0.0,
        "std_return": float(rets.std()) if len(rets) else 0.0,
        "goal_difference": int(sum(r.goal_difference for r in results)),
        "done_reasons": reasons,
        "reward_terms": {n: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                             "min": float(np.min(v)), "max": float(np.max(v))}
                         for n, v in terms.items()},
    }


def config_from_run(run_dir: str | Path) -> DictConfig:
    """The resolved config a run recorded."""
    cfg = OmegaConf.load(Path(run_dir) / "config.yaml")
    assert isinstance(cfg, DictConfig)
    return cfg
