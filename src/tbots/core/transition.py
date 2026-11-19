"""What a reward term receives: one step of ground truth, and why it ended.

Decided once, in the architecture review, so the signature never changes
under recruits (ARCHITECTURE section 9, "Reward terms"):

    world, prev   ground truth after and before the step -- NEVER perceived
    commands      what we sent, for kick-spam and energy terms
    ego_id        the trained robot in a skill env, None in tactics
    geometry      so terms never import field constants
    dt            seconds of simulation covered, so shaping is rate-independent
    done          None while running, else the Done reason

Deliberately absent: the perceived world and the observation vector, so a
term cannot reward what the policy could not know; and any RNG, so terms are
deterministic.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from tbots.core.command import RobotCommand
from tbots.core.geometry import FieldGeometry
from tbots.core.state import WorldState


class Done(enum.Enum):
    """Why an episode ended."""

    GOAL_US = "goal_us"                 # we scored
    GOAL_THEM = "goal_them"             # we conceded
    OUT_OF_BOUNDS = "out_of_bounds"     # the ball left the field
    TIMEOUT = "timeout"                 # the time limit; a truncation
    SKILL_SUCCESS = "skill_success"     # the skill task's success predicate
    SKILL_FAILURE = "skill_failure"     # the skill task's failure predicate
    FOUL_DRIBBLING = "foul_dribbling"   # excessive dribbling (synthetic referee)
    FOUL_DOUBLE_TOUCH = "foul_double_touch"
    FOUL_NO_PROGRESS = "foul_no_progress"

    @property
    def terminal(self) -> bool:
        """True for a real end of the episode (Gymnasium `terminated`).

        TIMEOUT is the one truncation: the world did not end, we stopped
        watching. The trainer bootstraps across it and must not across the
        others, which is why the distinction lives on the enum.
        """
        return self is not Done.TIMEOUT


@dataclass(frozen=True, slots=True)
class Transition:
    world: WorldState
    prev: WorldState
    commands: tuple[RobotCommand, ...]
    ego_id: int | None
    geometry: FieldGeometry
    dt: float
    done: Done | None = None

    def __post_init__(self) -> None:
        # A list inside a frozen dataclass is still mutable, and a term that
        # appended to it would change what the next term sees.
        if not isinstance(self.commands, tuple):
            object.__setattr__(self, "commands", tuple(self.commands))
