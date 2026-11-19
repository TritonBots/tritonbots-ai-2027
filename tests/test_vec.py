"""TASK-055: the subprocess vector env."""

from __future__ import annotations

import numpy as np

from tbots.rl.factory import compose_config
from tbots.rl.vec import benchmark, make_vec_env


def test_async_vector_env_steps_and_hands_back_final_observations():
    cfg = compose_config(["env.perception.mode=tracked", "task=go_to_ball"])
    env = make_vec_env(cfg, 2)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (2, *env.single_observation_space.shape)
        env.action_space.seed(0)
        ended = 0
        # 8 s limit at 60 Hz: every worker must have ended an episode by 500.
        for _ in range(500):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert r.shape == (2,)
            for i in np.flatnonzero(term | trunc):
                ended += 1
                assert info["_final_obs"][i]
                assert info["final_obs"][i].shape == env.single_observation_space.shape
        assert ended >= 2
    finally:
        env.close()


def test_workers_are_seeded_apart_and_reproducibly():
    cfg = compose_config([])
    a, b = make_vec_env(cfg, 2), make_vec_env(cfg, 2)
    try:
        oa, _ = a.reset(seed=5)
        ob, _ = b.reset(seed=5)
        np.testing.assert_array_equal(oa, ob)
        assert not np.array_equal(oa[0], oa[1])
    finally:
        a.close()
        b.close()


def test_benchmark_returns_a_rate():
    env = make_vec_env(compose_config([]), 1, asynchronous=False)
    try:
        assert benchmark(env, n_steps=50, warmup=5) > 0
    finally:
        env.close()
