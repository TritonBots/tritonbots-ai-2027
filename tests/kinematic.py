"""A tiny kinematic world for closed-loop motion tests.

Not a stand-in for rSim: it is the ground truth the planner is tested
against *before* physics enters, so a failure here is the planner's and not
the simulator's. Our robots move exactly as commanded (the command's
robot-local velocity rotated into the global frame); opponents follow
scripted velocity functions of time. The real-physics version of the same
scenarios lives in tests/test_motion_rsim.py.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from tbots.core.command import RobotCommand
from tbots.core.gamestate import GameState, Play
from tbots.core.state import BallState, RobotState, WorldState

DT = 1.0 / 60.0
RUNNING = GameState(play=Play.RUN, can_move=True, min_ball_distance=0.0, our_goalkeeper=0)

# t -> (vx, vy), global frame
VelocityFn = Callable[[float], tuple[float, float]]


def constant(vx: float, vy: float) -> VelocityFn:
    return lambda t: (vx, vy)


def switch_at(t_switch: float, before: tuple[float, float],
              after: tuple[float, float]) -> VelocityFn:
    return lambda t: before if t < t_switch else after


@dataclass
class Opponent:
    x: float
    y: float
    velocity: VelocityFn = field(default_factory=lambda: constant(0.0, 0.0))


def to_global(command: RobotCommand, theta: float) -> tuple[float, float]:
    c, s = math.cos(theta), math.sin(theta)
    return (c * command.vx - s * command.vy, s * command.vx + c * command.vy)


def segment_clearance(p0, p1, o0, o1) -> float:
    """Exact minimum distance between two points moving linearly over a tick."""
    dx0, dy0 = p0[0] - o0[0], p0[1] - o0[1]
    ddx = (p1[0] - o1[0]) - dx0
    ddy = (p1[1] - o1[1]) - dy0
    den = ddx * ddx + ddy * ddy
    s = 0.0 if den < 1e-15 else max(0.0, min(1.0, -(dx0 * ddx + dy0 * ddy) / den))
    return math.hypot(dx0 + s * ddx, dy0 + s * ddy)


class KinematicWorld:
    """Our robots integrate their commands; opponents follow their scripts."""

    def __init__(self, us: dict[int, tuple[float, float, float]],
                 them: dict[int, Opponent] | None = None,
                 ball: tuple[float, float] = (4.0, 2.8),
                 game: GameState = RUNNING) -> None:
        self.t = 0.0
        self.us = {i: RobotState(i, x, y, th) for i, (x, y, th) in us.items()}
        self.them_scripts = dict(them or {})
        self.ball = BallState(*ball)
        self.game = game
        self.min_clearance = math.inf          # us vs everyone, continuous time
        self._them = self._them_states(0.0)

    def _them_states(self, t: float) -> dict[int, RobotState]:
        out = {}
        for i, opp in self.them_scripts.items():
            vx, vy = opp.velocity(t)
            out[i] = RobotState(i, opp.x, opp.y, 0.0, vx=vx, vy=vy)
        return out

    def world(self) -> WorldState:
        return WorldState(t=self.t, t_capture=self.t, ball=self.ball, us=dict(self.us),
                          them=dict(self._them), game=self.game)

    def step(self, commands: list[RobotCommand]) -> WorldState:
        by_id = {c.robot_id: c for c in commands}
        new_us = {}
        for i, r in self.us.items():
            c = by_id.get(i, RobotCommand(i))
            gvx, gvy = to_global(c, r.theta)
            new_us[i] = RobotState(i, r.x + gvx * DT, r.y + gvy * DT,
                                   r.theta + c.vtheta * DT, vx=gvx, vy=gvy, vtheta=c.vtheta)
        for opp in self.them_scripts.values():
            vx, vy = opp.velocity(self.t)
            opp.x += vx * DT
            opp.y += vy * DT
        self.t += DT
        new_them = self._them_states(self.t)

        everyone_old = list(self.us.items()) + [(-1 - j, r) for j, r in self._them.items()]
        everyone_new = dict(new_us)
        everyone_new.update({-1 - j: r for j, r in new_them.items()})
        for i in self.us:
            for j, old in everyone_old:
                if j == i or (j >= 0 and j < i):
                    continue
                d = segment_clearance(self.us[i].pos, new_us[i].pos,
                                      old.pos, everyone_new[j].pos)
                self.min_clearance = min(self.min_clearance, d)
        self.us = new_us
        self._them = new_them
        return self.world()
