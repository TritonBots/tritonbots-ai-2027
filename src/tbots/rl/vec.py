"""Vectorised environment construction and throughput benchmarking.

rSim is in-process and fast, so throughput is the whole point of training
on it. One rSim per subprocess, built by `make_env` (the only construction
path), `OMP_NUM_THREADS=1` in every worker, robot counts fixed for the life
of the vector env -- no in-run reconfigure (ARCHITECTURE section 9).

Autoreset is SAME_STEP: when an episode ends the worker resets in the same
call and the true last observation travels in `info["final_obs"]`. The
trainer needs that observation to bootstrap a truncated episode.

Benchmark:

    python -m tbots.rl.vec --workers 1 8 32 --seconds 20 env=div_b_6v6 \\
        env.perception.mode=tracked
"""

from __future__ import annotations

import argparse
import functools
import os
import time
from typing import Any

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv
from omegaconf import DictConfig, OmegaConf


def _worker_env(cfg: dict[str, Any]) -> gym.Env:
    # Before anything numeric is imported in this process: one thread per
    # worker. N workers each spawning a core's worth of BLAS threads is how
    # a 32-worker run ends up slower than an 8-worker one.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from tbots.rl.factory import make_env  # in the worker, after the env vars
    return make_env(OmegaConf.create(cfg))


def make_vec_env(cfg: DictConfig, n_envs: int, asynchronous: bool = True) -> gym.vector.VectorEnv:
    """`n_envs` copies of `make_env(cfg)`. Seed with `vec.reset(seed=s)`:
    worker i gets seed s + i."""
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    # A plain dict pickles into a spawned worker; a DictConfig with Hydra
    # interpolations is not guaranteed to.
    plain = OmegaConf.to_container(cfg, resolve=True)
    fns = [functools.partial(_worker_env, plain) for _ in range(n_envs)]
    if not asynchronous:
        return SyncVectorEnv(fns, autoreset_mode=AutoresetMode.SAME_STEP)
    # spawn, not fork: the parent may hold torch's thread pools, and forking
    # a process with live threads can deadlock the child.
    return AsyncVectorEnv(fns, context="spawn", autoreset_mode=AutoresetMode.SAME_STEP)


def benchmark(env: gym.vector.VectorEnv, n_steps: int = 10_000, seed: int = 0,
              warmup: int = 50) -> float:
    """Environment steps per second, summed over workers, under random
    actions. Write the number down; scaling decisions use it."""
    env.reset(seed=seed)
    env.action_space.seed(seed)
    for _ in range(warmup):
        env.step(env.action_space.sample())
    calls = max(1, n_steps // env.num_envs)
    t0 = time.perf_counter()
    for _ in range(calls):
        env.step(env.action_space.sample())
    return calls * env.num_envs / (time.perf_counter() - t0)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--seconds", type=float, default=20.0,
                   help="approximate wall time per measurement")
    p.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. env=div_b_6v6")
    args = p.parse_args(argv)

    from tbots.rl.factory import compose_config
    cfg = compose_config(args.overrides)
    print(f"env={cfg.env.n_us}v{cfg.env.n_them} perception={cfg.env.perception.mode} "
          f"task={cfg.task} cpus={os.cpu_count()}")
    for n in args.workers:
        env = make_vec_env(cfg, n)
        try:
            probe = benchmark(env, n_steps=20 * n)
            sps = benchmark(env, n_steps=max(n, int(probe * args.seconds)))
        finally:
            env.close()
        print(f"workers={n:3d}  {sps:9.0f} steps/s  ({sps / n:7.0f} per worker)")


if __name__ == "__main__":
    main()
