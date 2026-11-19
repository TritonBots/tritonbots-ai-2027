"""Composable reward terms. This is the file recruits touch first.

A reward function is a weighted sum of registered terms, configured in
YAML. Nobody edits an environment to change a reward.

A term is a class with `reset()` and `__call__(tr: Transition) -> float`.
`Transition` (core/transition.py) is ground truth only: `tr.world` and
`tr.prev` around the step, `tr.commands` we sent, `tr.ego_id`,
`tr.geometry`, `tr.dt`, and `tr.done` -- `None` while the episode runs,
else a `Done` reason. The signature is frozen; it will not change under you.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

from tbots.core.transition import Transition


@runtime_checkable
class RewardTerm(Protocol):
    def __call__(self, tr: Transition) -> float: ...

    def reset(self) -> None: ...


_REGISTRY: dict[str, Callable[..., RewardTerm]] = {}


def register_reward(name: str):
    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"reward term '{name}' already registered")
        _REGISTRY[name] = cls
        return cls
    return deco


def reward_names() -> list[str]:
    return sorted(_REGISTRY)


class CompositeReward:
    """Weighted sum of named terms. Also records per-term contributions so
    you can see WHICH term is driving a policy, which is the difference
    between debugging a reward in an hour and debugging it in a week.

    `spec` is the `terms:` list of a reward YAML: each item has a `name`, an
    optional `weight` (default 1.0), and any other keys are passed to the
    term's constructor.
    """

    def __init__(self, spec: Iterable[Mapping[str, Any]]) -> None:
        self.terms: list[tuple[str, float, RewardTerm]] = []
        for item in spec:
            name = item["name"]
            weight = float(item.get("weight", 1.0))
            kwargs = {k: v for k, v in item.items() if k not in ("name", "weight")}
            if name not in _REGISTRY:
                raise KeyError(f"unknown reward term '{name}'. "
                               f"known: {reward_names()}")
            self.terms.append((name, weight, _REGISTRY[name](**kwargs)))
        self.last: dict[str, float] = {}

    def reset(self) -> None:
        self.last = {}
        for _, _, term in self.terms:
            term.reset()

    def __call__(self, tr: Transition) -> float:
        total = 0.0
        for name, weight, term in self.terms:
            v = weight * float(term(tr))
            self.last[name] = v
            total += v
        return total
