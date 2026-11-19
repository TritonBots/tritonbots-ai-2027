"""Physical limits of our robots, as the ai-team stack used them on hardware.

Source: the ``ai-hardcoded`` branch of https://github.com/TritonBots/ai-team,
the branch the July 2026 tournament ran from. Each value says where it came
from and whether anyone measured it, because they are not equally solid.

The ai-team stack ran on rcssserver at 1 unit = 0.1 m and 1 cycle = 0.1 s,
so a speed in units/cycle is the same number in m/s.
"""

from __future__ import annotations

import math

# MEASURED. ``networking/data_utils.py``: MAX_ANGULAR_VELOCITY =
# radians(20.0), "physical robot limit", clamped onto every turn command sent
# to the robots and the simulator (docs/CHANGES.md, "Enforce the physical
# robot's 20 deg/s angular-velocity limit").
MAX_ANGULAR_SPEED: float = math.radians(20.0)       # rad/s

# NOT MEASURED. ``configs/server.conf`` player_speed_max = 1.05 units/cycle.
# That is rcssserver's stock default, never recalibrated. It is the top speed
# every ai-team behaviour was tuned against, which is why it is used here.
# The robot firmware's mapping of `dash` power to wheel speed is not in the
# ai-team repo.
MAX_LINEAR_SPEED: float = 1.05                      # m/s

# MEASURED. ``configs/server.conf`` kick_power_rate = 0.0575, set in commit
# d20356a "Set sim kick exit speed to ~5.75 m/s to match real robots" from
# mechanical-team video analysis. A kick at power P leaves at 0.0575 * P m/s.
KICK_SPEED_PER_POWER: float = 0.0575                # (m/s) per unit of power

# MEASURED (CAD/mechanical). ``player_constants.py``
# FRONT_RECEPTION_CENTER_ANGLE_DEG = 100: the ball centre must be within this
# total angle of the robot's heading to be in the dribbler mouth.
RECEPTION_HALF_ANGLE: float = math.radians(100.0) / 2.0

# DESIGN MARGIN. ``docs/DRIBBLE_TO_MECHANICAL_REPORT.md``: carry at most
# 0.85 m per dribble segment, under the 1 m excessive-dribbling rule.
DRIBBLE_SEGMENT_LIMIT: float = 0.85                 # m


def kick_speed(power: float) -> float:
    """Ball exit speed for an rcss-style kick power in [0, 100]."""
    return KICK_SPEED_PER_POWER * max(0.0, min(100.0, power))
