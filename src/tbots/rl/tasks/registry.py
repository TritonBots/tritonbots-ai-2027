"""SkillTask: everything that defines one learned skill, in one object.

Training a skill used to mean touching six files. Now a task is ONE file in
`src/tbots/rl/tasks/` that builds a `SkillTask` and calls `register_task`.
Every module in this package is imported when the package is, so a new
file is found without editing anything else. Start one with:

    python -m tbots.rl.new_task <name>

then train it with:

    python -m tbots.rl.train env=skill task=<name>
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tbots.backends.base import Scenario
from tbots.core.state import WorldState

# scenario(rng, n_us, n_them) -> Scenario. Draw every random number from
# `rng`: it is seeded, and that is what makes runs reproducible.
ScenarioSampler = Callable[[np.random.Generator, int, int], Scenario]
# predicate(truth, ego_id) -> bool. Ground truth, never the perceived world.
Predicate = Callable[[WorldState, int], bool]


def never(world: WorldState, ego_id: int) -> bool:
    """A predicate that is never true. The default failure condition."""
    return False


@dataclass(frozen=True)
class SkillTask:
    name: str
    scenario: ScenarioSampler
    success: Predicate
    failure: Predicate = never
    obs_builder: str = "ego_v1"
    codec: str = "skill_v1"
    codec_kwargs: Mapping[str, Any] = field(default_factory=dict)
    time_limit_s: float = 10.0
    # A file in configs/reward/, without the .yaml. Used unless the command
    # line picks another with reward=NAME.
    reward: str = ""
    # The robot being trained. Always 0: builders never see the ID, so a
    # policy trained as robot 0 plays as any robot.
    ego_id: int = 0

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"task name must be a Python identifier, got {self.name!r}")
        if self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive")
        if not self.reward:
            object.__setattr__(self, "reward", self.name)


_REGISTRY: dict[str, SkillTask] = {}


def register_task(task: SkillTask) -> SkillTask:
    if task.name in _REGISTRY:
        raise ValueError(f"skill task '{task.name}' already registered")
    _REGISTRY[task.name] = task
    return task


def get_task(name: str) -> SkillTask:
    if name not in _REGISTRY:
        raise KeyError(f"unknown skill task '{name}'. known: {task_names()}")
    return _REGISTRY[name]


def task_names() -> list[str]:
    return sorted(_REGISTRY)
