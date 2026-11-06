"""The MOTION layer: classical, always (docs/ARCHITECTURE.md section 8).

``MPCPlanner`` drives one robot to a target around MOVING obstacles, within
``core.limits.MotionLimits``. It imports ``core`` and nothing else, so a
skill, a future state-machine tactic, or an RL action codec can all drive
through the same planner.
"""

from tbots.motion.mpc import MPCConfig, MPCPlanner, swept_clearance
from tbots.motion.obstacles import (
    Circle,
    KeepOuts,
    Obstacle,
    ObstacleKind,
    Rect,
    keepouts_from_world,
    obstacles_from_world,
)
from tbots.motion.prediction import (
    ConstantAccelerationPredictor,
    ConstantVelocityPredictor,
    ObstaclePredictor,
    build_predictor,
)

__all__ = [
    "Circle",
    "ConstantAccelerationPredictor",
    "ConstantVelocityPredictor",
    "KeepOuts",
    "MPCConfig",
    "MPCPlanner",
    "Obstacle",
    "ObstacleKind",
    "ObstaclePredictor",
    "Rect",
    "build_predictor",
    "keepouts_from_world",
    "obstacles_from_world",
    "swept_clearance",
]
