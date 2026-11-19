"""PPO, vendored and CleanRL-shaped: one file a recruit can read.

    Agent           actor-critic MLP; the actor ends in an ActionHead built
                    from the action space (algo/distributions.py)
    compute_gae     advantages and returns, with the truncation rule below
    ppo_update      the clipped-surrogate update over one rollout

THE TRUNCATION RULE (ARCHITECTURE section 9). An episode that ends because
time ran out did not end in the world: the robot would have kept collecting
reward. So a truncated step's reward gets `gamma * V(final_obs)` added --
the bootstrap -- and a terminated step's does not. Get this wrong and a
policy learns that the clock is a cliff. `tests/test_ppo.py` checks it.

The trainer loop that drives these lives in rl/train.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from tbots.rl.algo.distributions import Action, ActionHead, Composite, _layer


class Agent(nn.Module):
    def __init__(self, obs_size: int, action_space: gym.Space, hidden: int = 64) -> None:
        super().__init__()
        self.obs_size = obs_size
        self.hidden = hidden
        self.critic = nn.Sequential(
            _layer(obs_size, hidden, np.sqrt(2)), nn.Tanh(),
            _layer(hidden, hidden, np.sqrt(2)), nn.Tanh(),
            _layer(hidden, 1, 1.0))
        self.actor_body = nn.Sequential(
            _layer(obs_size, hidden, np.sqrt(2)), nn.Tanh(),
            _layer(hidden, hidden, np.sqrt(2)), nn.Tanh())
        self.head = ActionHead(action_space, hidden)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def dist(self, obs: torch.Tensor) -> Composite:
        return self.head(self.actor_body(obs))

    @torch.no_grad()
    def act(self, obs: torch.Tensor, *, deterministic: bool = False,
            generator: torch.Generator | None = None
            ) -> tuple[Action, torch.Tensor, torch.Tensor]:
        """(action, log-prob, value) for a batch of observations."""
        d = self.dist(obs)
        a = d.mode() if deterministic else d.sample(generator)
        return a, d.log_prob(a), self.value(obs)

    def evaluate(self, obs: torch.Tensor, a: Action
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(log-prob, entropy, value) of given actions."""
        d = self.dist(obs)
        return d.log_prob(a), d.entropy(), self.value(obs)


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                next_value: torch.Tensor, gamma: float, lam: float
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE over a [T, N] rollout.

    `dones[t]` is 1 where step t ended an episode, by termination OR by
    truncation: either way the next observation belongs to a new episode
    and nothing may flow back across it. Truncation bootstraps are already
    in `rewards[t]` (see `bootstrap_truncations`). `next_value` is V of the
    observation after the last step.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        nv = next_value if t == T - 1 else values[t + 1]
        live = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * live - values[t]
        last = delta + gamma * lam * live * last
        adv[t] = last
    return adv, adv + values


def bootstrap_truncations(rewards: np.ndarray, truncated: np.ndarray,
                          final_values: np.ndarray, gamma: float) -> np.ndarray:
    """rewards + gamma * V(final_obs) where truncated (and not terminated)."""
    return rewards + gamma * final_values * truncated.astype(rewards.dtype)


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 10
    num_minibatches: int = 8
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None


def ppo_update(agent: Agent, optimizer: torch.optim.Optimizer, cfg: PPOConfig,
               obs: torch.Tensor, actions: Action, logprobs: torch.Tensor,
               advantages: torch.Tensor, returns: torch.Tensor,
               generator: torch.Generator | None = None) -> dict[str, float]:
    """One PPO update over a flattened batch. Returns diagnostics."""
    batch = obs.shape[0]
    mb = max(1, batch // cfg.num_minibatches)
    stats: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": [],
                                     "approx_kl": [], "clipfrac": []}
    for _ in range(cfg.update_epochs):
        order = torch.randperm(batch, generator=generator)
        stop = False
        for start in range(0, batch, mb):
            idx = order[start:start + mb]
            new_lp, ent, v = agent.evaluate(obs[idx], {k: a[idx] for k, a in actions.items()})
            log_ratio = new_lp - logprobs[idx]
            ratio = log_ratio.exp()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()
            a = advantages[idx]
            a = (a - a.mean()) / (a.std() + 1e-8)
            pg = torch.max(-a * ratio,
                           -a * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
            vl = 0.5 * ((v - returns[idx]) ** 2).mean()
            loss = pg - cfg.ent_coef * ent.mean() + cfg.vf_coef * vl
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()
            for k, val in (("policy_loss", pg), ("value_loss", vl), ("entropy", ent.mean()),
                           ("approx_kl", approx_kl), ("clipfrac", clipfrac)):
                stats[k].append(float(val.detach()))
            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                stop = True
                break
        if stop:
            break
    return {k: float(np.mean(v)) for k, v in stats.items()}
