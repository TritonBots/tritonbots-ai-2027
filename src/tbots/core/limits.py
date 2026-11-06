"""What a robot can physically follow. Pure data, canonical units.

The motion planner (``tbots.motion``) plans inside these limits, and the
action codec (TASK-078) is specified to rate-limit learned actions with the
same object, so a skill and a policy can never disagree about what a motor
can do.

The defaults are MEASURED, not chosen: see docs/RSIM_FACTS.md, "Robot
dynamics". rSim's wheels are torque-limited and the limit depends on the
direction of travel. A command ramped at <= 1.9 m/s^2 is followed in every
direction; faster than 2.0 m/s^2 a diagonal saturates a wheel and the robot
falls behind and yaws. ``max_a = 1.8`` keeps a margin under that cliff.
Real robots will want their own measured values; pass them explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MotionLimits:
    """Isotropic limits on one robot's commanded motion.

    max_v  linear speed, m/s, as a NORM (not per axis)
    max_a  linear acceleration of the command, m/s^2, as a norm
    max_w  angular speed, rad/s
    """

    max_v: float = 2.5
    max_a: float = 1.8
    max_w: float = 8.0

    def __post_init__(self) -> None:
        if self.max_v <= 0.0 or self.max_a <= 0.0 or self.max_w <= 0.0:
            raise ValueError("motion limits must be positive")
