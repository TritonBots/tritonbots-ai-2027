"""Skill tasks. Importing this package registers every task in it.

Each module here (other than the registry and helpers) is one task, found
automatically, so `python -m tbots.rl.new_task NAME` never has to edit this
file.
"""

from __future__ import annotations

import importlib
import pkgutil

from tbots.rl.tasks.registry import (
    SkillTask,
    get_task,
    never,
    register_task,
    task_names,
)

_NOT_TASKS = {"registry", "sampling"}

for _m in pkgutil.iter_modules(__path__):
    if _m.name not in _NOT_TASKS and not _m.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_m.name}")

__all__ = ["SkillTask", "get_task", "never", "register_task", "task_names"]
