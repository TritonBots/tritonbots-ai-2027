"""Built-in skills available through :func:`build_skill`.

Keep this list deliberate.  Importing the package is the registration point,
but it must not pull in optional ML dependencies or unfinished skills.
"""

from tbots.skills.attack import Attack
from tbots.skills.base import Skill, SkillStatus, build_skill, register_skill, skill_names
from tbots.skills.go_to_point import GoToPoint
from tbots.skills.goalkeep import Goalkeep

__all__ = [
    "Attack",
    "GoToPoint",
    "Goalkeep",
    "Skill",
    "SkillStatus",
    "build_skill",
    "register_skill",
    "skill_names",
]
