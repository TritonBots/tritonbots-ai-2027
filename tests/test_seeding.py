"""TASK-080: one seed through scenarios, perception noise and policy
sampling. Gate: two runs with the same seed produce identical first 100
transitions per worker -- through the trainer's own Collector."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tbots.rl.evaluate import build_agent
from tbots.rl.factory import compose_config, make_env
from tbots.rl.train import Collector

N = 100


def _first_transitions(seed: int, mode: str) -> list[dict[str, np.ndarray]]:
    cfg = compose_config(["train.num_envs=3", f"train.seed={seed}",
                          f"env.perception.mode={mode}"])
    torch.manual_seed(seed)
    probe = make_env(cfg)
    agent = build_agent(probe, 64)
    probe.close()
    c = Collector(cfg, agent, seed)
    out = []
    try:
        for _ in range(N):
            r = c.step()
            out.append({"obs": r.obs.numpy().copy(),
                        **{f"a_{k}": v.numpy().copy() for k, v in r.action.items()},
                        "reward": r.reward.copy(), "term": r.terminated.copy(),
                        "trunc": r.truncated.copy()})
    finally:
        c.close()
    return out


@pytest.mark.timeout(300)
@pytest.mark.parametrize("mode", ["noisy", "tracked"])
def test_same_seed_identical_first_100_transitions_per_worker(mode):
    a, b = _first_transitions(7, mode), _first_transitions(7, mode)
    for t, (x, y) in enumerate(zip(a, b, strict=True)):
        for k in x:
            np.testing.assert_array_equal(x[k], y[k], err_msg=f"step {t} {k}")
    # And the seed matters: another one differs, and workers differ from
    # each other within a run.
    c = _first_transitions(8, mode)
    assert not np.array_equal(a[0]["obs"], c[0]["obs"])
    assert not np.array_equal(a[0]["obs"][0], a[0]["obs"][1])
