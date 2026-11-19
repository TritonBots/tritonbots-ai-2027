"""TASK-056: the PPO pieces. The truncation bootstrap is the one ARCHITECTURE
says must be tested."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from tbots.rl.algo.distributions import ActionHead
from tbots.rl.algo.ppo import Agent, PPOConfig, bootstrap_truncations, compute_gae, ppo_update

SPACE = gym.spaces.Dict({"move": gym.spaces.Box(-1, 1, (3,), np.float32),
                         "flags": gym.spaces.MultiBinary(2)})
G = 0.9


def test_truncation_bootstraps_and_termination_does_not():
    """Two one-step episodes ending at t=0, in two envs, identical but for
    HOW they ended. V(final_obs) = 10 for both."""
    rewards = np.array([[1.0, 1.0]], np.float32)
    truncated = np.array([[True, False]])
    terminated = np.array([[False, True]])
    final_v = np.array([[10.0, 10.0]], np.float32)
    r = bootstrap_truncations(rewards, truncated, final_v, G)
    assert r[0].tolist() == pytest.approx([1.0 + G * 10.0, 1.0])

    values = torch.tensor([[2.0, 2.0]])
    dones = torch.tensor((truncated | terminated).astype(np.float32))
    # The value after the last step is a NEW episode's start; it must not leak.
    next_value = torch.tensor([1000.0, 1000.0])
    adv, ret = compute_gae(torch.tensor(r), values, dones, next_value, G, 0.95)
    assert ret[0].tolist() == pytest.approx([1.0 + G * 10.0, 1.0])
    assert adv[0].tolist() == pytest.approx([1.0 + G * 10.0 - 2.0, 1.0 - 2.0])


def test_gae_matches_a_hand_computation_across_an_episode_boundary():
    rewards = torch.tensor([[1.0], [0.0], [2.0]])
    values = torch.tensor([[0.5], [0.4], [0.3]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    next_value = torch.tensor([0.7])
    lam = 0.8
    adv, _ = compute_gae(rewards, values, dones, next_value, G, lam)
    d2 = 2.0 + G * 0.7 - 0.3
    d1 = 0.0 - 0.4                       # episode ends at t=1: no bootstrap
    d0 = 1.0 + G * 0.4 - 0.5
    want = [d0 + G * lam * d1, d1, d2]
    assert adv.squeeze(-1).tolist() == pytest.approx(want)


def test_composite_log_prob_is_the_sum_of_its_parts():
    head = ActionHead(SPACE, 8)
    d = head(torch.randn(5, 8))
    a = d.sample()
    parts = d.parts["move"].log_prob(a["move"]) + d.parts["flags"].log_prob(a["flags"])
    torch.testing.assert_close(d.log_prob(a), parts)
    assert a["move"].shape == (5, 3) and a["flags"].shape == (5, 2)
    assert set(torch.unique(a["flags"]).tolist()) <= {0.0, 1.0}
    env_a = head.to_env(a)
    assert env_a["flags"].dtype == np.int8 and SPACE.contains(
        {"move": np.clip(env_a["move"][0], -1, 1).astype(np.float32),
         "flags": env_a["flags"][0]})


def test_seeded_sampling_is_reproducible():
    torch.manual_seed(0)
    agent = Agent(4, SPACE)
    obs = torch.randn(3, 4)
    a1, *_ = agent.act(obs, generator=torch.Generator().manual_seed(9))
    a2, *_ = agent.act(obs, generator=torch.Generator().manual_seed(9))
    for k in a1:
        torch.testing.assert_close(a1[k], a2[k])


def test_ppo_update_moves_the_policy_toward_advantaged_actions():
    torch.manual_seed(0)
    agent = Agent(2, SPACE)
    opt = torch.optim.Adam(agent.parameters(), lr=3e-3)
    obs = torch.zeros(256, 2)
    with torch.no_grad():
        a, lp, _ = agent.act(obs)
    # Reward forward motion: advantage = move[0].
    adv = a["move"][:, 0].clone()
    before = agent.dist(obs[:1]).parts["move"].mode()[0, 0].item()
    for _ in range(5):
        ppo_update(agent, opt, PPOConfig(update_epochs=4), obs, a, lp, adv, adv.clone())
    after = agent.dist(obs[:1]).parts["move"].mode()[0, 0].item()
    assert after > before + 0.05
