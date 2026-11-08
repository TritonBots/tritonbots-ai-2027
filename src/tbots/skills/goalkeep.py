"""Defend our goal. Hand-written baseline, ported from the ai-team rcssserver keeper.

Source: ``ai_interface/goalie.py`` on the ``support`` branch of
https://github.com/TritonBots/ai-team (2026-06-17). That code drove a 10x
scaled Division B pitch in rcssserver (90 x 60 units, 0.1 s cycles), so every
length below is the original constant times 0.1 m and every speed in
units/cycle is the same number in m/s. Constants that did not survive the
move to rSim say so where they are defined.

Decision making is a small state machine evaluated every tick:

- POSITION: stand on the bisector of the ball-to-post angle, stepped out a
  bounded distance from the goal line to cut the shooting angle.
- BLOCK: the ball's velocity ray crosses the goal mouth with enough speed to
  get there; move at full speed to the earliest reachable point on its path.
- CLEAR: the ball is held, or close and slow; dribble it, turn to a safe
  upfield direction, and kick it clear.
- CHARGE: a loose, slow ball in our defense area with no opponent contesting
  it; advance to claim it.

What changed in the port: we always defend -x (Rule 3), so side inference is
gone; ball velocity comes from the tracker instead of differencing positions;
the ball model is rSim's rolling deceleration instead of rcss's per-cycle
decay; and every target is clamped inside our defense area, which the SSL
rules give the keeper and nobody else.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

from tbots.core.command import RobotCommand
from tbots.core.geometry import BALL_RADIUS, DIV_B, ROBOT_RADIUS, FieldGeometry
from tbots.core.state import RobotState, WorldState
from tbots.core.units import angle_diff, wrap_angle
from tbots.skills.base import Skill, SkillStatus, register_skill
from tbots.skills.go_to_point import GoToPoint

KeeperMode = Literal["POSITION", "BLOCK", "CLEAR", "CHARGE"]

# Rolling deceleration of the ball in rSim, m/s^2. Measured 2026-11-07 (see
# docs/RSIM_FACTS.md): after the initial slide a ball loses ~0.7 m/s per
# second. Using the rolling figure for the whole trajectory overestimates how
# far a shot travels, so a keeper built on it errs toward blocking.
BALL_ROLLING_DECEL = 0.7


def ball_travel_time(dist: float, v0: float, decel: float = BALL_ROLLING_DECEL) -> float:
    """Seconds for a ball rolling at ``v0`` to cover ``dist``; inf if it stops first."""
    if dist <= 0.0:
        return 0.0
    if v0 <= 1e-9:
        return math.inf
    disc = v0 * v0 - 2.0 * decel * dist
    if disc < 0.0:
        return math.inf
    return (v0 - math.sqrt(disc)) / decel


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def bisector_target(
    ball: tuple[float, float],
    geometry: FieldGeometry = DIV_B,
    max_step_out: float = 0.4,
    close_zone: float = 0.8,
    far_zone: float = 2.5,
    y_margin: float = 0.2,
) -> tuple[float, float]:
    """Default keeper position for a ball at ``ball``.

    Stand on the bisector of the angle the two posts subtend at the ball,
    stepped out from the goal line toward the ball. Step-out is zero while the
    ball is within ``close_zone`` of the goal centre and grows linearly to
    ``max_step_out`` once it is beyond ``far_zone``.
    """
    goal_x, goal_y = geometry.our_goal
    half = geometry.goal_width / 2.0
    bx, by = ball

    ulx, uly = goal_x - bx, half - by
    urx, ury = goal_x - bx, -half - by
    nl, nr = math.hypot(ulx, uly), math.hypot(urx, ury)
    if nl < 1e-6 or nr < 1e-6:
        dx, dy = goal_x - bx, goal_y - by
    else:
        dx, dy = ulx / nl + urx / nr, uly / nl + ury / nr
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return (goal_x + ROBOT_RADIUS, goal_y)
    dx, dy = dx / n, dy / n

    # Where the bisector meets the goal line.
    if abs(dx) < 1e-6:
        iy = goal_y
    else:
        iy = by + (goal_x - bx) / dx * dy

    ball_goal = math.hypot(goal_x - bx, goal_y - by)
    factor = _clamp((ball_goal - close_zone) / (far_zone - close_zone), 0.0, 1.0)
    step = max_step_out * factor

    # (dx, dy) points from the ball at the goal, so step out along -(dx, dy).
    kx = _clamp(goal_x - step * dx, goal_x + ROBOT_RADIUS, goal_x + max_step_out)
    ky = _clamp(iy - step * dy, -half - y_margin, half + y_margin)
    return (kx, ky)


@register_skill("goalkeep")
class Goalkeep(Skill):
    """Keep goal. Never finishes on its own; the Coach reassigns it.

    Holds per-tick state (current mode, block hysteresis), so one instance
    must be kept for the whole time a robot is the keeper.
    """

    def __init__(
        self,
        geometry: FieldGeometry = DIV_B,
        max_step_out: float = 0.4,
        close_zone: float = 0.8,
        far_zone: float = 2.5,
        shot_speed: float = 0.3,
        goal_margin: float = 0.15,
        # rcss used 0.35 units/cycle (0.35 m/s), the speed of an rcss player.
        # rSim robots reach 3 m/s within a second; 1.5 m/s is a conservative
        # average over a short dash from rest.
        keeper_speed: float = 1.5,
        catch_dist: float = 0.12,
        charge_ball_speed: float = 0.5,
        contest_radius: float = 0.3,
        clear_kick_speed: float = 5.0,
        clear_align_tol: float = math.radians(6.0),
        block_release_ticks: int = 12,
        max_v: float = 2.5,
        block_max_v: float = 3.0,
    ) -> None:
        self.geometry = geometry
        self.max_step_out = min(max_step_out, geometry.penalty_depth - ROBOT_RADIUS)
        self.close_zone = close_zone
        self.far_zone = far_zone
        self.shot_speed = shot_speed
        self.goal_margin = goal_margin
        self.keeper_speed = keeper_speed
        self.catch_dist = catch_dist
        self.charge_ball_speed = charge_ball_speed
        self.contest_radius = contest_radius
        self.clear_kick_speed = clear_kick_speed
        self.clear_align_tol = clear_align_tol
        # rcss released a block after 2 quiet cycles (0.2 s); 12 ticks at 60 Hz.
        self.block_release_ticks = block_release_ticks
        self.max_v = max_v
        self.block_max_v = block_max_v

        self.mode: KeeperMode = "POSITION"
        self._release = 0
        self._status: SkillStatus = "running"

    # -- Skill protocol -------------------------------------------------------

    def reset(self, world: WorldState, robot_id: int) -> None:
        self.mode = "POSITION"
        self._release = 0
        self._status = "running"

    def status(self) -> SkillStatus:
        return self._status

    def step(self, world: WorldState, robot_id: int) -> RobotCommand:
        me = world.us.get(robot_id)
        if me is None:
            self._status = "failure"
            return RobotCommand(robot_id=robot_id)
        self._status = "running"

        held = self._holding(world, me)
        ball_dist = math.hypot(world.ball.x - me.x, world.ball.y - me.y)
        desired = self._select_mode(world, me, held, ball_dist)

        # Hysteresis: one noisy velocity frame must not drop a block.
        if self.mode == "BLOCK" and desired == "POSITION":
            self._release += 1
            if self._release < self.block_release_ticks:
                desired = "BLOCK"
            else:
                self._release = 0
        else:
            self._release = 0
        self.mode = desired

        if self.mode == "CLEAR":
            return self._clear(world, me, held)
        if self.mode == "BLOCK":
            return self._block(world, me)
        if self.mode == "CHARGE":
            return self._charge(world, me)
        return self._position(world, me)

    # -- perception helpers ---------------------------------------------------

    @staticmethod
    def _holding(world: WorldState, me: RobotState) -> bool:
        if me.has_ball:
            return True
        dx, dy = world.ball.x - me.x, world.ball.y - me.y
        d = math.hypot(dx, dy)
        facing = abs(angle_diff(math.atan2(dy, dx), me.theta)) < math.radians(25.0)
        return facing and d < ROBOT_RADIUS + BALL_RADIUS + 0.015

    def _ball_speed(self, world: WorldState) -> float:
        return math.hypot(world.ball.vx, world.ball.vy)

    def shot_crossing(self, world: WorldState) -> float | None:
        """The y at which the ball will cross our goal line, if it is a shot.

        A shot moves faster than ``shot_speed``, is headed at our goal line,
        crosses it within the goal mouth plus ``goal_margin``, and has the
        speed to reach the line before friction stops it.
        """
        speed = self._ball_speed(world)
        if speed < self.shot_speed:
            return None
        goal_x = self.geometry.our_goal[0]
        bx, by, vx, vy = world.ball.x, world.ball.y, world.ball.vx, world.ball.vy
        if vx >= -1e-9:
            return None
        t = (goal_x - bx) / vx
        y_cross = by + t * vy
        half = self.geometry.goal_width / 2.0 + self.goal_margin
        if abs(y_cross) > half:
            return None
        if ball_travel_time(math.hypot(goal_x - bx, y_cross - by), speed) == math.inf:
            return None
        return y_cross

    def _opponent_near_ball(self, world: WorldState) -> bool:
        bx, by = world.ball.pos
        return any(math.hypot(r.x - bx, r.y - by) < self.contest_radius
                   for r in world.them.values())

    def _select_mode(self, world: WorldState, me: RobotState, held: bool,
                     ball_dist: float) -> KeeperMode:
        speed = self._ball_speed(world)
        if held or (ball_dist < self.catch_dist and speed < self.shot_speed):
            return "CLEAR"
        if self.shot_crossing(world) is not None:
            return "BLOCK"
        # rcss charged inside a 6-unit (0.6 m) radius of the goal. In SSL the
        # whole defense area is the keeper's alone -- a ball left stopped in
        # it is one nobody else may fetch -- so the area is the trigger.
        if (speed < self.charge_ball_speed
                and self.geometry.inside_our_defense_area(world.ball.x, world.ball.y)
                and not self._opponent_near_ball(world)):
            return "CHARGE"
        return "POSITION"

    # -- targets --------------------------------------------------------------

    def _clamp_to_area(self, x: float, y: float) -> tuple[float, float]:
        """Keep the keeper in our defense area, never behind the goal line."""
        g = self.geometry
        goal_x = g.our_goal[0]
        x = _clamp(x, goal_x + ROBOT_RADIUS, goal_x + g.penalty_depth - ROBOT_RADIUS)
        lim = g.penalty_width / 2.0 - ROBOT_RADIUS
        return (x, _clamp(y, -lim, lim))

    def block_point(self, world: WorldState, me: RobotState) -> tuple[float, float]:
        """Earliest point on the ball's path the keeper reaches no later than the ball.

        Points before the keeper's perpendicular foot on the ray are both
        farther from the keeper and reached sooner by the ball, so only the
        stretch from the foot to the goal line is searched. If nothing on it is
        reachable in time, guard the predicted crossing.
        """
        goal_x = self.geometry.our_goal[0]
        bx, by = world.ball.pos
        speed = self._ball_speed(world)
        half = self.geometry.goal_width / 2.0
        if speed < 1e-6:
            return (goal_x + ROBOT_RADIUS, _clamp(by, -half, half))
        dx, dy = world.ball.vx / speed, world.ball.vy / speed
        if abs(dx) < 1e-9:
            return (me.x, _clamp(by, -half, half))

        t_goal = max((goal_x + ROBOT_RADIUS - bx) / dx, 0.0)
        t_foot = _clamp((me.x - bx) * dx + (me.y - by) * dy, 0.0, t_goal)
        n = 24
        for i in range(n + 1):
            s = t_foot + (t_goal - t_foot) * i / n
            px, py = bx + s * dx, by + s * dy
            keeper_t = math.hypot(px - me.x, py - me.y) / self.keeper_speed
            if keeper_t <= ball_travel_time(s, speed):
                return (px, py)
        return (goal_x + ROBOT_RADIUS, by + t_goal * dy)

    def clearance_angle(self, world: WorldState, me: RobotState) -> float:
        """A safe global direction to clear toward.

        Candidates fan out around straight upfield (+x, never at our own
        goal), scored by how upfield they are, a slight bias toward the nearer
        touchline, and a bonus when a teammate sits in the candidate's cone.
        """
        mates = [r for rid, r in world.us.items() if rid != me.robot_id]
        best, best_score = 0.0, -math.inf
        for offset in (-0.9, -0.5, -0.25, 0.0, 0.25, 0.5, 0.9):
            cand = wrap_angle(offset)
            score = math.cos(offset)
            if math.sin(cand) * me.y > 0:
                score += 0.15
            for r in mates:
                if abs(angle_diff(math.atan2(r.y - me.y, r.x - me.x), cand)) < 0.3:
                    score += 1.0
                    break
            if score > best_score:
                best, best_score = cand, score
        return best

    # -- per-mode commands ----------------------------------------------------

    def _drive(self, world: WorldState, me: RobotState, target: tuple[float, float],
               face: float, max_v: float, kp: float = 3.0) -> RobotCommand:
        return GoToPoint(target=target, face=face, max_v=max_v,
                         kp_pos=kp, pos_tol=0.02).step(world, me.robot_id)

    def _face_ball(self, world: WorldState, me: RobotState) -> float:
        return math.atan2(world.ball.y - me.y, world.ball.x - me.x)

    def _position(self, world: WorldState, me: RobotState) -> RobotCommand:
        goal_x = self.geometry.our_goal[0]
        half = self.geometry.goal_width / 2.0
        bx, by = world.ball.pos
        if bx - goal_x < 0.05:
            # Ball at or behind our goal line: do not chase it into the net;
            # hold the near-post area just in front of the line.
            target = (goal_x + 0.1, _clamp(by, -half + 0.1, half - 0.1))
        else:
            target = bisector_target(world.ball.pos, self.geometry, self.max_step_out,
                                     self.close_zone, self.far_zone)
        target = self._clamp_to_area(*target)
        return self._drive(world, me, target, self._face_ball(world, me), self.max_v)

    def _block(self, world: WorldState, me: RobotState) -> RobotCommand:
        half = self.geometry.goal_width / 2.0
        tx, ty = self.block_point(world, me)
        goal_x = self.geometry.our_goal[0]
        tx = _clamp(tx, goal_x + ROBOT_RADIUS, goal_x + self.max_step_out)
        ty = _clamp(ty, -half - 0.05, half + 0.05)
        target = self._clamp_to_area(tx, ty)
        # Full speed: reaching the line beats a smooth arrival.
        return self._drive(world, me, target, self._face_ball(world, me),
                           self.block_max_v, kp=6.0)

    def _clear(self, world: WorldState, me: RobotState, held: bool) -> RobotCommand:
        if held:
            # Holding the ball: rotate to a safe direction with the dribbler
            # on, and only kick once aligned, so we never clear blindly.
            want = self.clearance_angle(world, me)
            err = angle_diff(want, me.theta)
            kick = self.clear_kick_speed if abs(err) < self.clear_align_tol else 0.0
            return RobotCommand(robot_id=me.robot_id, vx=0.15,
                                vtheta=_clamp(4.0 * err, -3.0, 3.0),
                                dribbler=1.0, kick_speed=kick).clamped()
        # Close but not held: ease onto the ball with the dribbler on.
        cmd = self._drive(world, me, world.ball.pos, self._face_ball(world, me), 0.8)
        return replace(cmd, dribbler=1.0)

    def _charge(self, world: WorldState, me: RobotState) -> RobotCommand:
        target = self._clamp_to_area(*world.ball.pos)
        cmd = self._drive(world, me, target, self._face_ball(world, me), 1.5, kp=4.0)
        return replace(cmd, dribbler=1.0)
