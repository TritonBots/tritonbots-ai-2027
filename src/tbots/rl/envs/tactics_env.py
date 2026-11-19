"""Options wrapper: one step() runs many backend ticks.

An action here is a set of skill assignments, not a velocity. The env then
runs the underlying skills at 60 Hz until one terminates (or a timeout),
which is what collapses a ~7,200-tick episode into ~240 decisions. See
SETUP.md 11.4 for why that decomposition is the whole bet.

The observation MUST stay a fixed size across every curriculum stage --
see SETUP.md 9.2a -- or nothing transfers between stages.
"""


from __future__ import annotations

import numpy as np

from tbots.core.command import RobotCommand
from tbots.core.state import WorldState
from tbots.rl.envs.base import SSLEnv


class TacticsEnv(SSLEnv):
    def _observe(self, world: WorldState) -> np.ndarray:
        raise NotImplementedError("TASK-051")

    def _decode(self, action, world: WorldState) -> list[list[RobotCommand]]:
        raise NotImplementedError("TASK-051")
