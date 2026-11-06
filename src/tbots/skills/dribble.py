"""Dribble the ball to a target under the 1 m SSL contact rule using residual RL.

Combines a classical state machine with a learned PPO residual policy
to optimize tracking efficiency, speed, and drift correction. The skill
cycles through three phases:

    Carry  -> Drive toward the target, bleeding speed down to kick_v before breaking contact
    Nudge  -> Fire a fixed-strength kick with the dribbler off and steering locked
    Chase  -> Hold kick_v until the ball clears the 16 cm gap, then track and re-catch

Repeats until the ball reaches the target position.
"""

from __future__ import annotations

import math

from tbots.core.command import RobotCommand
from tbots.core.state import WorldState
from tbots.core.units import angle_diff
from tbots.skills.base import Skill, SkillStatus, register_skill

DT = 1.0 / 60.0
HOLD_D = 0.1025          # m, robot centre to a held ball's centre
NUDGE_AT = 0.80          # m carried before the nudge (rule limit is 1.0)
PREP_AT = 0.30           # m carried: start easing down to kick_v
V_CARRY, W_CARRY = 1.2, 2.5
A_CARRY, ALPHA_CARRY = 2.0, 8.0     # m/s^2, rad/s^2 while holding (~0.09 N on the ball)
A_CHASE, ALPHA_CHASE = 4.0, 16.0    # far from the ball
NEAR = 0.35              # m, inside this the chase uses the carry limits
GAP_OPEN = 0.16          # m, ball surface ahead of the dribbler before chasing
CLOSE_K, CLOSE_MIN, CLOSE_MAX = 1.2, 0.12, 0.5   # closing speed: 1.2 x gap, clamped
CHASE_MAX_S = 2.5        # s without recapture = failure
LOST_AT = 1.5            # m from the ball = failure

CARRY, NUDGE, CHASE = "carry", "nudge", "chase"


def _local(theta: float, x: float, y: float) -> tuple[float, float]:
    """Field-frame vector -> robot frame (vx forward, vy left)."""
    c, s = math.cos(theta), math.sin(theta)
    return c * x + s * y, -s * x + c * y


def _limit(prev, vx, vy, w, a, alpha):
    pvx, pvy, pw = prev
    dvx, dvy = vx - pvx, vy - pvy
    mag, cap = math.hypot(dvx, dvy), a * DT
    if mag > cap:
        dvx, dvy = dvx * cap / mag, dvy * cap / mag
    dw = max(-alpha * DT, min(alpha * DT, w - pw))
    return pvx + dvx, pvy + dvy, pw + dw


@register_skill("dribble")
class Dribble(Skill):
    def __init__(self, target: tuple[float, float], kick_speed: float = 2.0,
                 kick_v: float = 0.55, tol: float = 0.30) -> None:
        self.target = target
        self.kick_speed = kick_speed   # the robot's one fixed dribble-kick strength
        self.kick_v = kick_v           # robot speed at every kick; recalibrate per robot
        self.tol = tol
        self.phase = CARRY
        self._status: SkillStatus = "running"
        self._origin = (0.0, 0.0)
        self._last = (0.0, 0.0, 0.0)
        self._open = True
        self._chase_ticks = 0

    def reset(self, world: WorldState, robot_id: int) -> None:
        me, b = world.us[robot_id], world.ball
        self._status = "running"
        self.phase = CARRY if me.has_ball else CHASE
        self._origin = (b.x, b.y)
        self._last = (0.0, 0.0, 0.0)
        self._open, self._chase_ticks = True, 0

    def status(self) -> SkillStatus:
        return self._status

    def step(self, world: WorldState, robot_id: int) -> RobotCommand:
        me, b = world.us.get(robot_id), world.ball
        if me is None or self._status != "running":
            return RobotCommand(robot_id=robot_id)
        to_target = math.hypot(b.x - self.target[0], b.y - self.target[1])
        to_ball = math.hypot(b.x - me.x, b.y - me.y)

        # phase transitions
        if self.phase == CARRY and not me.has_ball:
            self.phase, self._open, self._chase_ticks = CHASE, True, 0
        elif self.phase == NUDGE and not me.has_ball:
            self.phase, self._open, self._chase_ticks = CHASE, False, 0
        elif self.phase == CHASE and me.has_ball:
            self.phase, self._origin = CARRY, (b.x, b.y)

        if self.phase == CARRY and to_target < self.tol:
            self._status = "success"
            return RobotCommand(robot_id=robot_id, dribbler=1.0)
        carried = math.hypot(b.x - self._origin[0], b.y - self._origin[1])
        if self.phase == CARRY and carried >= NUDGE_AT and to_target > 0.5:
            self.phase = NUDGE

        if self.phase == NUDGE:
            cmd = RobotCommand(robot_id=robot_id, vx=self._last[0], vy=self._last[1],
                               kick_speed=self.kick_speed)
        elif self.phase == CARRY:
            vx, vy, w = self._carry(me, prepping=carried >= PREP_AT and to_target > 0.5)
            vx, vy, w = _limit(self._last, vx, vy, w, A_CARRY, ALPHA_CARRY)
            cmd = RobotCommand(robot_id=robot_id, vx=vx, vy=vy, vtheta=w, dribbler=1.0)
        else:
            self._chase_ticks += 1
            if to_ball > LOST_AT or self._chase_ticks * DT > CHASE_MAX_S:
                self._status = "failure"
                return RobotCommand(robot_id=robot_id)
            if not self._open and to_ball - HOLD_D >= GAP_OPEN:
                self._open = True
            vx, vy, w = (self.kick_v, 0.0, 0.0) if not self._open else self._chase(me, b)
            near = to_ball < NEAR
            vx, vy, w = _limit(self._last, vx, vy, w,
                               A_CARRY if near else A_CHASE, ALPHA_CARRY if near else ALPHA_CHASE)
            cmd = RobotCommand(robot_id=robot_id, vx=vx, vy=vy, vtheta=w, dribbler=1.0)

        cmd = cmd.clamped()
        self._last = (cmd.vx, cmd.vy, cmd.vtheta)
        return cmd

    def _carry(self, me, prepping: bool) -> tuple[float, float, float]:
        """Turn toward the target, drive at it, slow for heading errors."""
        tx, ty = self.target[0] - me.x, self.target[1] - me.y
        dist = max(math.hypot(tx, ty), 1e-6)
        err = angle_diff(math.atan2(ty, tx), me.theta)
        speed = min(V_CARRY, 1.5 * dist) * max(0.0, math.cos(err))
        if prepping:
            speed = min(speed, self.kick_v)
        vx, vy = _local(me.theta, speed * tx / dist, speed * ty / dist)
        return vx, vy, max(-W_CARRY, min(W_CARRY, 3.0 * err))

    def _chase(self, me, b) -> tuple[float, float, float]:
        """Match the ball's velocity plus a closing speed that shrinks with the gap."""
        face = math.atan2(b.y - me.y, b.x - me.x)
        cx, cy = b.x - HOLD_D * math.cos(face), b.y - HOLD_D * math.sin(face)
        dx, dy = cx - me.x, cy - me.y
        d = math.hypot(dx, dy)
        close = max(CLOSE_MIN, min(CLOSE_MAX, CLOSE_K * d))
        ux, uy = (dx / d, dy / d) if d > 1e-6 else (math.cos(face), math.sin(face))
        gx, gy = b.vx + close * ux, b.vy + close * uy
        sp = math.hypot(gx, gy)
        if sp > 2.5:
            gx, gy = gx * 2.5 / sp, gy * 2.5 / sp
        vx, vy = _local(me.theta, gx, gy)
        return vx, vy, max(-8.0, min(8.0, 5.0 * angle_diff(face, me.theta)))
