"""Composable action distributions.

Built once, around the action SPACE, so the trainer never knows what an
action looks like. A skill's action is a Dict of a Box (Gaussian) and a
MultiBinary (Bernoulli); a factorised tactics policy (MAPPO, TASK-084) will be
a Composite of one head per robot. Either way the joint log-probability is
the sum of the parts', which is what makes per-robot heads tractable where a
joint softmax over six robots' choices is not (ARCHITECTURE section 9).

Actions travel through the trainer as `dict[str, Tensor]` (a single-space
policy uses the key "action").
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

Action = dict[str, torch.Tensor]


class DiagGaussian:
    """Independent normals; log-prob and entropy summed over the last dim.
    Unsquashed, as in CleanRL: the codec clips to the space's bounds."""

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor) -> None:
        self.dist = torch.distributions.Normal(mean, log_std.exp().expand_as(mean))

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(self.dist.mean.shape, generator=generator,
                            dtype=self.dist.mean.dtype, device=self.dist.mean.device)
        return self.dist.mean + self.dist.stddev * noise

    def mode(self) -> torch.Tensor:
        return self.dist.mean

    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(a).sum(-1)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy().sum(-1)


class Bernoulli:
    """Independent coin flips, one per element; summed over the last dim."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.dist = torch.distributions.Bernoulli(logits=logits)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        u = torch.rand(self.dist.probs.shape, generator=generator,
                       dtype=self.dist.probs.dtype, device=self.dist.probs.device)
        return (u < self.dist.probs).to(self.dist.probs.dtype)

    def mode(self) -> torch.Tensor:
        return (self.dist.logits > 0).to(self.dist.logits.dtype)

    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(a).sum(-1)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy().sum(-1)


Part = DiagGaussian | Bernoulli


class Composite:
    """Named parts; the joint log-prob and entropy are the sums."""

    def __init__(self, parts: dict[str, Part]) -> None:
        self.parts = parts

    def sample(self, generator: torch.Generator | None = None) -> Action:
        return {k: d.sample(generator) for k, d in self.parts.items()}

    def mode(self) -> Action:
        return {k: d.mode() for k, d in self.parts.items()}

    def log_prob(self, a: Action) -> torch.Tensor:
        return torch.stack([d.log_prob(a[k]) for k, d in self.parts.items()]).sum(0)

    def entropy(self) -> torch.Tensor:
        return torch.stack([d.entropy() for d in self.parts.values()]).sum(0)


def _layer(n_in: int, n_out: int, std: float) -> nn.Linear:
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class _GaussianHead(nn.Module):
    def __init__(self, n_in: int, n: int) -> None:
        super().__init__()
        self.mean = _layer(n_in, n, 0.01)
        self.log_std = nn.Parameter(torch.zeros(n))

    def forward(self, h: torch.Tensor) -> DiagGaussian:
        return DiagGaussian(self.mean(h), self.log_std)


class _BernoulliHead(nn.Module):
    def __init__(self, n_in: int, n: int) -> None:
        super().__init__()
        self.logits = _layer(n_in, n, 0.01)

    def forward(self, h: torch.Tensor) -> Bernoulli:
        return Bernoulli(self.logits(h))


class ActionHead(nn.Module):
    """Features -> Composite distribution, laid out from a gymnasium space.

    Supported: Box (flat), MultiBinary (flat), and a Dict of those.
    """

    def __init__(self, space: gym.Space, n_in: int) -> None:
        super().__init__()
        subspaces = dict(space.spaces) if isinstance(space, gym.spaces.Dict) \
            else {"action": space}
        self.keys = list(subspaces)
        self.heads = nn.ModuleDict()
        for k, s in subspaces.items():
            if isinstance(s, gym.spaces.Box) and len(s.shape) == 1:
                self.heads[k] = _GaussianHead(n_in, s.shape[0])
            elif isinstance(s, gym.spaces.MultiBinary) and isinstance(s.n, int):
                self.heads[k] = _BernoulliHead(n_in, s.n)
            else:
                raise TypeError(f"no head for action subspace {k!r}: {s}")
        self._single = not isinstance(space, gym.spaces.Dict)

    def forward(self, h: torch.Tensor) -> Composite:
        return Composite({k: self.heads[k](h) for k in self.keys})

    def to_env(self, a: Action) -> Any:
        """A batch of actions as the (vector) env wants them."""
        out = {k: v.detach().cpu().numpy() for k, v in a.items()}
        for k, head in self.heads.items():
            if isinstance(head, _BernoulliHead):
                out[k] = out[k].astype(np.int8)
        return out["action"] if self._single else out
