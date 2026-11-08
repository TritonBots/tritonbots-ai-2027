"""Win the ball and score. Hand-written baseline, ported from the ai-team attacker.

Source: ``ai_interface/attacker.py`` (``SmartAttacker``) on the
``minghan-foundation`` branch of https://github.com/TritonBots/ai-team
(2026-05-01). That code drove a 10x scaled Division B pitch in rcssserver
(90 x 60 units, 0.1 s cycles), so lengths below are the original constants
times 0.1 m, speeds in units/cycle are the same number in m/s, and the shot
score weights are per meter instead of per unit (x10) so the scoring is
unchanged. Constants that did not survive the move to rSim say so.

Without the ball the attacker chases where the ball will be, then stages
behind it on the line to its target and closes in with the dribbler on. With
the ball it scores both goal corners against the keeper and every opponent in
the lane, shoots if in range and the lane is good enough, and otherwise
dribbles by short pushes toward goal, away from the nearest opponent, pulled
toward the middle near a touchline.

What changed in the port: rcss's turn/catch/kick string commands are
velocity commands with the dribbler and kicker; the "lane blocked" check runs
against every opponent instead of a single named defender; we never enter
their defense area (an SSL foul) -- this replaces the rcss "back off when the
keeper crowds the ball" step, which is kept for a ball just outside it. Two
rcss behaviours are dropped: the kickoff special case (restarts are the
Coach's job, TASK-043) and the "defender near ball, run to (10, 0)"
workaround.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from tbots.core.command import RobotCommand
from tbots.core.geometry import BALL_RADIUS, DIV_B, ROBOT_RADIUS, FieldGeometry
from tbots.core.state import RobotState, WorldState
from tbots.core.units import angle_diff
from tbots.skills.base import Skill, SkillStatus, register_skill
from tbots.skills.go_to_point import GoToPoint


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def dist_point_to_segment(p: tuple[float, float], a: tuple[float, float],
                          b: tuple[float, float]) -> float:
    abx, aby = b[0] - a[0], b[1] - a[1]
    n2 = abx * abx + aby * aby
    if n2 < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = _clamp(((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / n2, 0.0, 1.0)
    return math.hypot(p[0] - (a[0] + t * abx), p[1] - (a[1] + t * aby))


@dataclass(frozen=True)
class ShotChoice:
    target: tuple[float, float]
    blocked: bool
    score: float


def choose_shot_target(
    ball: tuple[float, float],
    world: WorldState,
    geometry: FieldGeometry = DIV_B,
    corner_inset: float = 0.35,
    lane_block_dist: float = 0.26,
    w_keeper_gap: float = 30.0,
    w_blocked: float = 2.2,
    w_distance: float = 0.3,
    w_keeper_advanced: float = 0.5,
    keeper_advanced_radius: float = 1.8,
) -> ShotChoice:
    """Pick the better of the two goal corners.

    Score = w_keeper_gap * |corner_y - keeper_y| - w_blocked * blocked
            - w_distance * shot_length + w_keeper_advanced * closeness,
    where ``blocked`` is any opponent within ``lane_block_dist`` of the
    ball-to-corner segment. The keeper is their robot nearest their goal.
    """
    gx = geometry.their_goal[0]
    inset = min(corner_inset, geometry.goal_width / 2.0)
    keeper = their_keeper(world, geometry)
    ky = keeper.y if keeper is not None else 0.0
    keeper_ball = (math.hypot(keeper.x - ball[0], keeper.y - ball[1])
                   if keeper is not None else math.inf)

    best: ShotChoice | None = None
    for t in ((gx, inset), (gx, -inset)):
        blocked = any(dist_point_to_segment(r.pos, ball, t) < lane_block_dist
                      for r in world.them.values())
        score = (w_keeper_gap * abs(t[1] - ky)
                 - w_blocked * (1.0 if blocked else 0.0)
                 - w_distance * math.hypot(t[0] - ball[0], t[1] - ball[1]))
        if keeper is not None:
            score += w_keeper_advanced * max(0.0, keeper_advanced_radius - keeper_ball)
        if best is None or score > best.score:
            best = ShotChoice(t, blocked, score)
    assert best is not None
    return best


def their_keeper(world: WorldState, geometry: FieldGeometry = DIV_B) -> RobotState | None:
    """Their robot nearest their goal centre.

    ``GameState`` does not carry the opponent's goalkeeper id (only ours), so
    proximity stands in for it.
    """
    if not world.them:
        return None
    gx, gy = geometry.their_goal
    return min(world.them.values(), key=lambda r: math.hypot(r.x - gx, r.y - gy))


@register_skill("attack")
class Attack(Skill):
    """Chase, carry and shoot. Never finishes on its own; the Coach reassigns it.

    Holds per-tick state (possession hysteresis, approach staging), so one
    instance must be kept for as long as a robot is the attacker.
    """

    def __init__(
        self,
        geometry: FieldGeometry = DIV_B,
        # Shooting
        shoot_range: float = 3.8,
        very_close_frac: float = 0.55,
        shot_score_threshold: float = 5.5,
        # rcss "kick 50..60" is 1.35-1.6 m/s, a pass in SSL; a shot is harder.
        shot_speed: float = 5.0,
        kick_align_tol: float = math.radians(2.0),
        # Dribbling: short pushes, which also keeps us inside the 1 m
        # dribbling limit. rcss "skick 22" (~0.6 m/s) dies within a few cm
        # on rSim's carpet.
        dribble_touch_speed: float = 1.0,
        dribble_forward: float = 0.42,
        dribble_side_base: float = 0.22,
        center_pull_strength: float = 0.55,
        sideline_risk_band: float = 0.6,
        boundary_buffer: float = 0.25,
        # Approach
        approach_back_extra: float = 0.12,
        stage_tol: float = 0.05,
        contact_speed: float = 0.8,
        chase_far_dist: float = 0.32,
        predict_horizon: float = 0.4,
        keeper_backoff_dist: float = 0.3,
        keeper_backoff_step: float = 1.0,
        # rcss kept "with ball" for 10 cycles (1 s) after losing contact; at
        # 60 Hz a quarter second is enough to bridge dribbler bounces.
        possession_ticks: int = 15,
        max_v: float = 2.5,
    ) -> None:
        self.geometry = geometry
        self.shoot_range = shoot_range
        self.very_close_frac = very_close_frac
        self.shot_score_threshold = shot_score_threshold
        self.shot_speed = shot_speed
        self.kick_align_tol = kick_align_tol
        self.dribble_touch_speed = dribble_touch_speed
        self.dribble_forward = dribble_forward
        self.dribble_side_base = dribble_side_base
        self.center_pull_strength = center_pull_strength
        self.sideline_risk_band = sideline_risk_band
        self.boundary_buffer = boundary_buffer
        self.approach_back_extra = approach_back_extra
        self.stage_tol = stage_tol
        self.contact_speed = contact_speed
        self.chase_far_dist = chase_far_dist
        self.predict_horizon = predict_horizon
        self.keeper_backoff_dist = keeper_backoff_dist
        self.keeper_backoff_step = keeper_backoff_step
        self.possession_ticks = possession_ticks
        self.max_v = max_v

        self._with_ball = 0
        self.last_action = "chase"
        self._status: SkillStatus = "running"

    # -- Skill protocol -------------------------------------------------------

    def reset(self, world: WorldState, robot_id: int) -> None:
        self._with_ball = 0
        self.last_action = "chase"
        self._status = "running"

    def status(self) -> SkillStatus:
        return self._status

    def step(self, world: WorldState, robot_id: int) -> RobotCommand:
        me = world.us.get(robot_id)
        if me is None:
            self._status = "failure"
            return RobotCommand(robot_id=robot_id)
        self._status = "running"

        g = self.geometry
        if g.inside_their_defense_area(world.ball.x, world.ball.y):
            self.last_action = "hold"
            return self._hold_outside_area(world, me)

        if self._has_ball(world, me):
            self._with_ball = self.possession_ticks + 1
        if self._with_ball > 0:
            self._with_ball -= 1
            return self._with_ball_cmd(world, me)
        return self._without_ball_cmd(world, me)

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _has_ball(world: WorldState, me: RobotState) -> bool:
        """Ball on the kicker: the dribbler's sensor, or touching and in front."""
        if me.has_ball:
            return True
        dx, dy = world.ball.x - me.x, world.ball.y - me.y
        in_front = abs(angle_diff(math.atan2(dy, dx), me.theta)) < math.radians(25.0)
        return in_front and math.hypot(dx, dy) < ROBOT_RADIUS + BALL_RADIUS + 0.02

    def _goto(self, world: WorldState, me: RobotState, target: tuple[float, float],
              face: float, max_v: float | None = None, kp: float = 3.0) -> RobotCommand:
        return GoToPoint(target=self._legal(target), face=face,
                         max_v=self.max_v if max_v is None else max_v,
                         kp_pos=kp, pos_tol=0.01).step(world, me.robot_id)

    def _legal(self, p: tuple[float, float]) -> tuple[float, float]:
        """Clamp a target into the field and out of both defense areas."""
        g = self.geometry
        buf = ROBOT_RADIUS
        x = _clamp(p[0], -g.half_length + buf, g.half_length - buf)
        y = _clamp(p[1], -g.half_width + buf, g.half_width - buf)
        edge = g.half_length - g.penalty_depth - ROBOT_RADIUS
        if abs(y) <= g.penalty_width / 2.0 + ROBOT_RADIUS:
            x = _clamp(x, -edge, edge)
        return (x, y)

    def sideline_risk(self, y: float) -> float:
        """0 in the middle of the field, 1 on a touchline."""
        gap = self.geometry.half_width - abs(y)
        return _clamp(1.0 - gap / self.sideline_risk_band, 0.0, 1.0)

    def _hold_outside_area(self, world: WorldState, me: RobotState) -> RobotCommand:
        bx, by = world.ball.pos
        target = (bx - 0.5, by)
        return self._goto(world, me, target, math.atan2(by - me.y, bx - me.x))

    def _approach(self, world: WorldState, me: RobotState,
                  target: tuple[float, float]) -> RobotCommand | None:
        """Stage behind the ball on the line to ``target``, then close in.

        Returns None once the robot has the ball on its kicker; the caller
        decides what to do with it.

        rcss kept a "deep approach" flag for this; here it is a geometric test
        each tick instead -- already behind the ball, on the line, and facing
        along it -- so it cannot latch in the wrong state.
        """
        if self._has_ball(world, me):
            return None
        bx, by = world.ball.pos
        dx, dy = target[0] - bx, target[1] - by
        n = math.hypot(dx, dy)
        ux, uy = (dx / n, dy / n) if n > 1e-6 else (1.0, 0.0)
        face = math.atan2(uy, ux)
        reach = ROBOT_RADIUS + BALL_RADIUS
        back = reach + self.approach_back_extra

        rel_x, rel_y = me.x - bx, me.y - by
        along = rel_x * ux + rel_y * uy          # negative = behind the ball
        lateral = abs(-rel_x * uy + rel_y * ux)
        on_line = (lateral < self.stage_tol
                   and -(back + self.stage_tol) < along < -reach + 0.03
                   and abs(angle_diff(face, me.theta)) < math.radians(10.0))
        if not on_line:
            stage = (bx - ux * back, by - uy * back)
            return self._goto(world, me, self._around_ball(world, me, stage), face, kp=4.0)

        # Drive through the ball centre with the dribbler on to take it.
        cmd = self._goto(world, me, (bx - ux * (reach - 0.03), by - uy * (reach - 0.03)),
                         face, max_v=self.contact_speed, kp=4.0)
        return replace(cmd, dribbler=1.0)

    @staticmethod
    def _around_ball(world: WorldState, me: RobotState,
                     stage: tuple[float, float]) -> tuple[float, float]:
        """Detour waypoint if the straight path to ``stage`` hits the ball."""
        clear = ROBOT_RADIUS + BALL_RADIUS + 0.06
        if dist_point_to_segment(world.ball.pos, me.pos, stage) > clear:
            return stage
        bx, by = world.ball.pos
        sx, sy = stage[0] - bx, stage[1] - by
        n = math.hypot(sx, sy) or 1.0
        px, py = -sy / n, sx / n
        side = 1.0 if (me.x - bx) * px + (me.y - by) * py >= 0 else -1.0
        return (bx + side * px * (clear + 0.05) + sx / n * 0.05,
                by + side * py * (clear + 0.05) + sy / n * 0.05)

    def _aim_and_kick(self, world: WorldState, me: RobotState,
                      target: tuple[float, float], speed: float) -> RobotCommand:
        """Rotate with the ball on the dribbler; kick only once aligned."""
        want = math.atan2(target[1] - me.y, target[0] - me.x)
        err = angle_diff(want, me.theta)
        kick = speed if abs(err) < self.kick_align_tol else 0.0
        return RobotCommand(robot_id=me.robot_id, vx=0.2,
                            vtheta=_clamp(4.0 * err, -3.0, 3.0),
                            dribbler=1.0, kick_speed=kick).clamped()

    # -- with the ball --------------------------------------------------------

    def _with_ball_cmd(self, world: WorldState, me: RobotState) -> RobotCommand:
        g = self.geometry
        bx, by = world.ball.pos
        gx, gy = g.their_goal

        keeper = their_keeper(world, g)
        if keeper is not None and keeper.x > 0:
            kb = math.hypot(keeper.x - bx, keeper.y - by)
            if kb < self.keeper_backoff_dist:
                # The keeper is on the ball: do not wrestle for it.
                ax, ay = bx - keeper.x, by - keeper.y
                n = math.hypot(ax, ay) or 1.0
                self.last_action = "backoff"
                return self._goto(world, me, (bx + ax / n * self.keeper_backoff_step,
                                              by + ay / n * self.keeper_backoff_step),
                                  math.atan2(by - me.y, bx - me.x))

        shot = choose_shot_target((bx, by), world, g)
        range_x = abs(me.x - gx)
        in_range = range_x < self.shoot_range
        very_close = range_x < self.shoot_range * self.very_close_frac
        if (in_range and (not shot.blocked or shot.score > self.shot_score_threshold)) \
                or very_close:
            self.last_action = "shoot"
            cmd = self._approach(world, me, shot.target)
            if cmd is not None:
                return cmd
            return self._aim_and_kick(world, me, shot.target, self.shot_speed)

        # Dribble by short pushes.
        risk = self.sideline_risk(by)
        sign = 1.0
        near = min(world.them.values(), default=None,
                   key=lambda r: math.hypot(r.x - bx, r.y - by))
        if near is not None:
            sign = -1.0 if near.y > by else 1.0
        if keeper is not None and abs(keeper.y) > 0.1:
            sign = -1.0 if keeper.y > 0 else 1.0
        if risk > 0.35:
            sign = -1.0 if by > 0 else 1.0

        dx, dy = gx - bx, gy - by
        n = math.hypot(dx, dy)
        ux, uy = (dx / n, dy / n) if n > 1e-6 else (1.0, 0.0)
        sx, sy = -uy, ux
        side = (1.0 - risk) * self.dribble_side_base * sign
        tx = bx + ux * self.dribble_forward + sx * side
        ty = (by + uy * self.dribble_forward + sy * side) \
            * (1.0 - self.center_pull_strength * risk)
        buf = self.boundary_buffer
        tx = _clamp(tx, -g.half_length + buf, g.half_length - buf)
        ty = _clamp(ty, -g.half_width + buf, g.half_width - buf)

        self.last_action = "dribble"
        cmd = self._approach(world, me, (tx, ty))
        if cmd is not None:
            return cmd
        return self._aim_and_kick(world, me, (tx, ty), self.dribble_touch_speed)

    # -- without the ball -----------------------------------------------------

    def _without_ball_cmd(self, world: WorldState, me: RobotState) -> RobotCommand:
        bx, by = world.ball.pos
        shot = choose_shot_target((bx, by), world, self.geometry)
        speed = math.hypot(world.ball.vx, world.ball.vy)
        if math.hypot(bx - me.x, by - me.y) > self.chase_far_dist and speed > 0.2:
            # A moving ball: run to where it will be, not where it is.
            self.last_action = "chase"
            px = bx + world.ball.vx * self.predict_horizon
            py = by + world.ball.vy * self.predict_horizon
            return self._goto(world, me, (px, py), math.atan2(py - me.y, px - me.x), kp=4.0)
        # Slow or close: line up behind the ball toward the better corner.
        self.last_action = "approach"
        cmd = self._approach(world, me, shot.target)
        if cmd is not None:
            return cmd
        # Ball is on the kicker: possession logic takes over next tick.
        return RobotCommand(robot_id=me.robot_id, dribbler=1.0)
