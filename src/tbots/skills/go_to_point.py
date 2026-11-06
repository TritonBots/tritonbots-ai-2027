"""Drive to a pose. Classical, not learned -- and that is on purpose.

A well-tuned velocity profile beats six months of PPO at this task,
transfers to real hardware for free, and can be debugged with a print
statement. RL earns its keep where physics is hard to model (dribbler
contact, interception under uncertainty), not on rigid-body motion across a
flat floor.

With ``avoid=True`` (the default) translation goes through the dynamic MPC
planner in ``tbots.motion``: the robot drives around moving opponents,
teammates and the rule keep-out zones, within ``core.limits.MotionLimits``.
In free space the planner reduces to a time-optimal straight-line profile.

``hold=True`` keeps station after arriving instead of reporting success:
the skill stays ``running`` (``arrived`` says it got there), so the Coach
keeps stepping it and the robot still dodges anything that comes at it. Use
it for positions a robot should keep, not for legs of a journey.
``avoid=False`` is the original proportional controller, kept for tests and
for anything that must drive straight through.

Heading is always the proportional controller below; an omnidirectional
robot can turn and translate independently.
"""

from __future__ import annotations

import math

from tbots.core.command import RobotCommand
from tbots.core.geometry import DIV_B, FieldGeometry
from tbots.core.limits import MotionLimits
from tbots.core.state import WorldState
from tbots.core.units import angle_diff
from tbots.motion import MPCPlanner, build_predictor, obstacles_from_world
from tbots.skills.base import Skill, SkillStatus, register_skill

# With avoidance, "arrived" also means "slow enough to stop in about a tick",
# so the stop the Coach sends next is not a velocity cliff.
SETTLE_SPEED: float = 0.1


@register_skill("go_to_point")
class GoToPoint(Skill):
    def __init__(
        self,
        target: tuple[float, float],
        face: float | None = None,
        pos_tol: float = 0.04,
        ang_tol: float = 0.08,
        max_v: float = 2.5,
        max_w: float = 8.0,
        kp_pos: float = 3.0,
        kp_ang: float = 4.0,
        avoid: bool = True,
        avoid_ball: bool = False,
        predictor: str = "constant_velocity",
        max_a: float = MotionLimits().max_a,
        geometry: FieldGeometry = DIV_B,
        hold: bool = False,
    ) -> None:
        self.target = (float(target[0]), float(target[1]))
        self.face = face
        self.pos_tol = pos_tol
        self.ang_tol = ang_tol
        self.max_v = max_v
        self.max_w = max_w
        self.kp_pos = kp_pos
        self.kp_ang = kp_ang
        self.avoid = avoid
        self.avoid_ball = avoid_ball
        self.geometry = geometry
        self.hold = hold
        self._arrived = False
        # Built even when avoid=False so a bad predictor name fails at
        # construction, not forty minutes in.
        self._planner = MPCPlanner(
            limits=MotionLimits(max_v=max_v, max_a=max_a, max_w=max_w),
            predictor=build_predictor(predictor),
        )
        self._status: SkillStatus = "running"

    def reset(self, world: WorldState, robot_id: int) -> None:
        self._status = "running"
        self._arrived = False
        self._planner.reset()

    @property
    def arrived(self) -> bool:
        """At the pose as of the last step. With ``hold=True`` this, not
        ``status()``, is the signal that the robot got there."""
        return self._arrived

    def step(self, world: WorldState, robot_id: int) -> RobotCommand:
        me = world.us.get(robot_id)
        if me is None:
            self._status = "failure"
            return RobotCommand(robot_id=robot_id)

        ex = self.target[0] - me.x
        ey = self.target[1] - me.y
        err = math.hypot(ex, ey)

        desired_theta = self.face if self.face is not None else me.theta
        eang = angle_diff(desired_theta, me.theta)

        self._arrived = err < self.pos_tol and abs(eang) < self.ang_tol and self._settled()
        if self._arrived and not self.hold:
            self._status = "success"
            return RobotCommand(robot_id=robot_id)
        # hold=True: keep station. The skill stays "running", so the Coach
        # keeps stepping it, and the planner keeps dodging -- a robot whose
        # skill reported success is parked by the Coach and cannot move out
        # of anyone's way.

        if self.avoid:
            obstacles, keepouts = obstacles_from_world(world, robot_id, self.geometry,
                                                       avoid_ball=self.avoid_ball)
            gx, gy = self._planner.plan(me.pos, (me.vx, me.vy), self.target,
                                        obstacles, keepouts, now=world.t)
        else:
            # Global-frame P control.
            speed = min(self.kp_pos * err, self.max_v)
            gx = speed * ex / max(err, 1e-6)
            gy = speed * ey / max(err, 1e-6)

        # Rotate into the robot's local frame.
        c, s = math.cos(-me.theta), math.sin(-me.theta)
        vx = c * gx - s * gy
        vy = s * gx + c * gy

        command = RobotCommand(
            robot_id=robot_id, vx=vx, vy=vy,
            vtheta=max(-self.max_w, min(self.max_w, self.kp_ang * eang)),
        )
        # The planner already bounds the speed as a norm; the per-axis clamp
        # is the original controller's and is a no-op on planned commands.
        return command if self.avoid else command.clamped(max_v=self.max_v, max_w=self.max_w)

    def _settled(self) -> bool:
        if not self.avoid:
            return True
        last = self._planner.last_command
        return last is None or math.hypot(*last) <= SETTLE_SPEED

    def status(self) -> SkillStatus:
        return self._status
