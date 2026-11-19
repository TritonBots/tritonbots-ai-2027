"""Hydra entry point: train a policy with PPO.

    python -m tbots.rl.train                      # go_to_ball, default config
    python -m tbots.rl.train task=NAME reward=NAME train.run_dir=runs/NAME

Composes configs/ (env, task, reward, train), builds environments ONLY
through `make_env` / `make_vec_env`, and writes a run directory:

    runs/NAME/config.yaml              the resolved config, reward included
    runs/NAME/latest.pt                checkpoint (atomic; TASK-059 metadata)
    runs/NAME/events.out.tfevents.*    TensorBoard: `tensorboard --logdir runs`
    runs/NAME/rollouts/*.npz           the eval at every checkpoint, recorded

Resuming: `train.resume_from=PATH` loads the weights and optimizer, refusing a
checkpoint whose observation builder, codec or dt differ. Resuming from this
run's own `latest.pt` (walltime) continues the step count; resuming from
another run's (the next curriculum stage) starts it at zero.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from tbots.rl.algo.ppo import Agent, PPOConfig, bootstrap_truncations, compute_gae, ppo_update
from tbots.rl.artifacts import load_checkpoint, save_checkpoint, write_resolved_config
from tbots.rl.evaluate import build_agent, eval_config, eval_seeds, evaluate, summarize
from tbots.rl.factory import env_metadata, make_env, resolve_config
from tbots.rl.vec import make_vec_env


def prepare_run(cfg: DictConfig) -> Path:
    """Validate and record a requested run before a trainer consumes it."""
    run_dir = Path(cfg.train.run_dir)
    if not str(run_dir):
        raise ValueError("train.run_dir must not be empty")
    write_resolved_config(run_dir, cfg)
    return run_dir


def _final_obs(info: dict[str, Any], i: int) -> np.ndarray:
    return np.asarray(info["final_obs"][i], dtype=np.float32)


def _final_done(info: dict[str, Any], i: int) -> str | None:
    fi = info.get("final_info")
    if fi is None or "done" not in fi:
        return None
    v = fi["done"][i]
    return None if v is None else str(v)


@dataclass
class StepRecord:
    """One vector-env step as the trainer stores it."""

    obs: torch.Tensor                   # what the action was chosen on
    action: dict[str, torch.Tensor]
    logprob: torch.Tensor
    value: torch.Tensor
    reward: np.ndarray                  # the env's reward, before bootstrapping
    terminated: np.ndarray
    truncated: np.ndarray
    final_value: np.ndarray             # V(final_obs) where truncated only, else 0
    info: dict[str, Any]


class Collector:
    """Steps the vector env with the policy. ONE SEED (TASK-080) decides
    everything it does: worker i's scenarios and perception noise come from
    `seed + i` (vec reset), and action sampling from a generator seeded with
    `seed`. Opponents stand still today; when they act (TASK-057) they must
    draw from the same seed."""

    def __init__(self, cfg: DictConfig, agent: Agent, seed: int,
                 device: torch.device | None = None) -> None:
        n = int(cfg.train.num_envs)
        self.agent = agent
        self.device = device or torch.device("cpu")
        self.n = n
        self.envs = make_vec_env(cfg, n, asynchronous=n > 1)
        self.obs_dim = int(self.envs.single_observation_space.shape[0])
        self.rng = torch.Generator().manual_seed(seed)
        self.obs, _ = self.envs.reset(seed=seed)

    def tensor(self, x: Any) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)

    def step(self) -> StepRecord:
        o = self.tensor(self.obs)
        a, lp, v = self.agent.act(o, generator=self.rng)
        self.obs, r, term, trunc, info = self.envs.step(self.agent.head.to_env(a))
        term, trunc = np.asarray(term, bool), np.asarray(trunc, bool)
        final_v = np.zeros(self.n, dtype=np.float32)
        tr_only = trunc & ~term
        if tr_only.any():
            idx = np.flatnonzero(tr_only)
            with torch.no_grad():
                fv = self.agent.value(self.tensor(np.stack([_final_obs(info, i) for i in idx])))
            final_v[idx] = fv.cpu().numpy()
        return StepRecord(o, a, lp, v, np.asarray(r, dtype=np.float32), term, trunc,
                          final_v, info)

    def next_value(self) -> torch.Tensor:
        with torch.no_grad():
            return self.agent.value(self.tensor(self.obs))

    def close(self) -> None:
        self.envs.close()


def train(cfg: DictConfig) -> dict[str, Any]:
    """Run PPO as `cfg` says. Returns a summary (also written as
    run_dir/train_summary.json)."""
    cfg = resolve_config(cfg)
    run_dir = prepare_run(cfg)
    t = cfg.train
    seed = int(t.seed)
    device = torch.device(str(t.device))
    torch.manual_seed(seed)                                   # network init
    shuffle_rng = torch.Generator().manual_seed(seed + 1)     # minibatch order

    probe = make_env(cfg)
    metadata = env_metadata(probe, cfg)
    agent = build_agent(probe, int(t.hidden)).to(device)
    probe.close()
    optimizer = torch.optim.Adam(agent.parameters(), lr=float(t.learning_rate), eps=1e-5)

    global_step = 0
    if t.resume_from:
        state = load_checkpoint(t.resume_from, expected_metadata=metadata)
        agent.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if Path(t.resume_from).resolve() == (run_dir / "latest.pt").resolve():
            global_step = int(state["global_step"])

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(run_dir))
    writer.add_text("config", "```\n" + OmegaConf.to_yaml(cfg) + "\n```")

    n, T = int(t.num_envs), int(t.num_steps)
    ppo = PPOConfig(gamma=float(t.gamma), gae_lambda=float(t.gae_lambda),
                    update_epochs=int(t.update_epochs), num_minibatches=int(t.num_minibatches),
                    clip_coef=float(t.clip_coef), ent_coef=float(t.ent_coef),
                    vf_coef=float(t.vf_coef), max_grad_norm=float(t.max_grad_norm),
                    target_kl=None if t.target_kl is None else float(t.target_kl))
    collector = Collector(cfg, agent, seed, device)
    obs_dim = collector.obs_dim
    tensor = collector.tensor

    b_obs = torch.zeros((T, n, obs_dim), device=device)
    b_lp = torch.zeros((T, n), device=device)
    b_rew = torch.zeros((T, n), device=device)
    b_done = torch.zeros((T, n), device=device)
    b_val = torch.zeros((T, n), device=device)
    b_act: dict[str, torch.Tensor] = {}

    ep_ret = np.zeros(n)
    ep_len = np.zeros(n, dtype=np.int64)
    recent: list[float] = []
    reasons: dict[str, int] = {}
    truncations = 0
    eval_history: list[dict[str, Any]] = []

    total = int(t.total_steps)
    interval = int(t.checkpoint_interval_steps)
    next_ckpt = global_step + interval
    per_update = n * T
    updates = max(1, math.ceil((total - global_step) / per_update))
    t0 = time.monotonic()
    start_step = global_step

    def checkpoint_and_eval() -> None:
        save_checkpoint(run_dir, {
            "metadata": metadata, "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(), "global_step": global_step,
            "agent": {"hidden": agent.hidden, "obs_size": agent.obs_size},
            "config": OmegaConf.to_container(cfg, resolve=True)})
        if int(t.eval_episodes) > 0:
            res = evaluate(agent, eval_config(cfg), eval_seeds(int(t.eval_episodes)),
                           record_dir=run_dir / "rollouts", keep=int(t.keep_rollouts),
                           label=f"step{global_step:09d}")
            s = summarize(res)
            writer.add_scalar("eval/mean_return", s["mean_return"], global_step)
            writer.add_scalar("eval/goal_difference", s["goal_difference"], global_step)
            for name, d in s["reward_terms"].items():
                writer.add_scalar(f"eval_terms/{name}", d["mean"], global_step)
            eval_history.append({"step": global_step, "mean_return": s["mean_return"],
                                 "done_reasons": s["done_reasons"]})

    for update in range(updates):
        if t.anneal_lr:
            frac = max(0.0, 1.0 - global_step / total)
            optimizer.param_groups[0]["lr"] = frac * float(t.learning_rate)
        for step in range(T):
            rec = collector.step()
            global_step += n
            ep_ret += rec.reward
            ep_len += 1
            ended = rec.terminated | rec.truncated
            # Truncation bootstraps from V(final_obs); termination does not.
            tr_only = rec.truncated & ~rec.terminated
            truncations += int(tr_only.sum())
            r = bootstrap_truncations(rec.reward, tr_only, rec.final_value, ppo.gamma)
            info = rec.info

            b_obs[step] = rec.obs
            b_lp[step] = rec.logprob
            b_val[step] = rec.value
            b_rew[step] = tensor(r)
            b_done[step] = tensor(ended)
            for k, x in rec.action.items():
                if k not in b_act:
                    b_act[k] = torch.zeros((T, n, *x.shape[1:]), device=device)
                b_act[k][step] = x

            for i in np.flatnonzero(ended):
                recent.append(float(ep_ret[i]))
                writer.add_scalar("charts/episodic_return", ep_ret[i], global_step)
                writer.add_scalar("charts/episodic_length", ep_len[i], global_step)
                why = _final_done(info, i) or "UNKNOWN"
                reasons[why] = reasons.get(why, 0) + 1
                ep_ret[i], ep_len[i] = 0.0, 0

        next_v = collector.next_value()
        adv, ret = compute_gae(b_rew, b_val, b_done, next_v, ppo.gamma, ppo.gae_lambda)
        stats = ppo_update(
            agent, optimizer, ppo, b_obs.reshape(-1, obs_dim),
            {k: x.reshape(T * n, *x.shape[2:]) for k, x in b_act.items()},
            b_lp.reshape(-1), adv.reshape(-1), ret.reshape(-1), generator=shuffle_rng)

        sps = (global_step - start_step) / max(time.monotonic() - t0, 1e-9)
        for k, val in stats.items():
            writer.add_scalar(f"losses/{k}", val, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/truncations", truncations, global_step)
        tail = recent[-100:]
        mean100 = float(np.mean(tail)) if tail else float("nan")
        print(f"update {update + 1}/{updates}  step {global_step}  "
              f"return(last {len(tail)}) {mean100:.3f}  sps {sps:.0f}", flush=True)

        last = update == updates - 1
        if global_step >= next_ckpt or last:
            checkpoint_and_eval()
            while next_ckpt <= global_step:
                next_ckpt += interval

    collector.close()
    writer.close()
    summary = {
        "run_dir": str(run_dir), "global_step": global_step,
        "episodes": len(recent), "mean_return_last_100": float(np.mean(recent[-100:]))
        if recent else None, "done_reasons": reasons, "truncations": truncations,
        "evals": eval_history, "seconds": time.monotonic() - t0,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
