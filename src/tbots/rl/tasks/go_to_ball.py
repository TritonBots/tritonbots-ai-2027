"""go_to_ball: drive to a stationary ball from anywhere on the field.

The reference task: the one `train env=skill` runs by default, and the one
the trainer's step budget is documented against (TASK-056). Deliberately
easy -- it exists to prove the pipeline learns, not to be a good skill.
"""

from __future__ import annotations

import math

import numpy as np

from tbots.backends.base import Scenario
from tbots.core.state import WorldState
from tbots.rl.tasks.registry import SkillTask, register_task
from tbots.rl.tasks.sampling import random_point, random_pose, scatter

# Centre to centre. Touching is 0.09 + 0.0215 = 0.11 m; 0.18 m is "at the
# ball" without requiring contact, which would push it away.
REACHED = 0.18


def scenario(rng: np.random.Generator, n_us: int, n_them: int) -> Scenario:
    ego = random_pose(rng, margin=0.5)
    while True:
        ball = random_point(rng, margin=0.5)
        if math.hypot(ball[0] - ego[0], ball[1] - ego[1]) >= 0.6:
            break
    others = scatter(rng, n_us - 1 + n_them, avoid=[ball, (ego[0], ego[1])])
    return Scenario(ball=(ball[0], ball[1], 0.0, 0.0),
                    us=(ego, *others[:n_us - 1]), them=tuple(others[n_us - 1:]))


def success(world: WorldState, ego_id: int) -> bool:
    me = world.us[ego_id]
    return math.hypot(me.x - world.ball.x, me.y - world.ball.y) < REACHED


register_task(SkillTask(
    name="go_to_ball",
    scenario=scenario,
    success=success,
    time_limit_s=8.0,
))
