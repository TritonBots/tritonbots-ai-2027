"""What the planner must not hit, built from one ``WorldState``.

Two kinds of thing:

* ``Obstacle`` -- something that MOVES: an opponent, a teammate, the ball.
  It carries its current velocity, and an ``ObstaclePredictor`` turns that
  into a trajectory over the planning horizon. A standing robot is just an
  obstacle whose velocity is zero.
* ``KeepOuts`` -- regions the rules forbid, mirroring the Coach's rule filter
  (tactics/coach.py): our defense area for every non-keeper, their defense
  area inflated by 0.2 m outside RUN, and the ``min_ball_distance`` circle.
  Planning around them is what keeps the filter from having to freeze a
  robot; the filter stays as the final guarantee.

Global frame, canonical units, we attack +x (Rules 3 and 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from tbots.core.gamestate import Play
from tbots.core.geometry import BALL_RADIUS, ROBOT_RADIUS, FieldGeometry
from tbots.core.state import WorldState

# Clearance kept on top of physical contact. Contact between two robots is
# a centre distance of 2 * ROBOT_RADIUS.
ROBOT_MARGIN: float = 0.03
# An extrapolated (not currently seen) robot is where we THINK it is.
HIDDEN_MARGIN: float = 0.05
# Kept inside rule boundaries, so a plan that grazes one never trips the
# Coach's filter, which tests the robot centre against the exact boundary.
RULE_MARGIN: float = 0.05
# Their defense area is inflated by this much outside RUN; same value the
# Coach filter uses.
THEIR_AREA_STOPPAGE_CLEARANCE: float = 0.20


class ObstacleKind(Enum):
    OPPONENT = auto()
    TEAMMATE = auto()
    BALL = auto()


@dataclass(frozen=True, slots=True)
class Obstacle:
    """A moving circle. ``radius`` is the COMBINED clearance: our centre must
    stay at least this far from its centre."""

    id: int
    kind: ObstacleKind
    x: float
    y: float
    vx: float
    vy: float
    radius: float


@dataclass(frozen=True, slots=True)
class Rect:
    """Axis-aligned keep-out: our centre must stay outside."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True, slots=True)
class Circle:
    """Round keep-out: our centre must stay outside."""

    x: float
    y: float
    radius: float


@dataclass(frozen=True, slots=True)
class KeepOuts:
    rects: tuple[Rect, ...] = ()
    circles: tuple[Circle, ...] = ()
    # Our centre must stay INSIDE this; None means unbounded.
    bounds: Rect | None = None


def obstacles_from_world(
    world: WorldState,
    robot_id: int,
    geometry: FieldGeometry,
    *,
    avoid_ball: bool = False,
    robot_margin: float = ROBOT_MARGIN,
) -> tuple[list[Obstacle], KeepOuts]:
    """Everything robot ``robot_id`` must avoid this tick."""
    contact = 2.0 * ROBOT_RADIUS + robot_margin
    obstacles: list[Obstacle] = []
    for kind, team in ((ObstacleKind.TEAMMATE, world.us), (ObstacleKind.OPPONENT, world.them)):
        for other in team.values():
            if kind is ObstacleKind.TEAMMATE and other.robot_id == robot_id:
                continue
            radius = contact + (0.0 if other.visible else HIDDEN_MARGIN)
            obstacles.append(Obstacle(other.robot_id, kind, other.x, other.y,
                                      other.vx, other.vy, radius))
    if avoid_ball:
        ball = world.ball
        obstacles.append(Obstacle(0, ObstacleKind.BALL, ball.x, ball.y, ball.vx, ball.vy,
                                  ROBOT_RADIUS + BALL_RADIUS + robot_margin))
    return obstacles, keepouts_from_world(world, robot_id, geometry)


def keepouts_from_world(world: WorldState, robot_id: int,
                        geometry: FieldGeometry) -> KeepOuts:
    game = world.game
    hl, hw = geometry.half_length, geometry.half_width
    far = hl + geometry.boundary_width + 1.0       # past the wall: "to infinity"
    half_area = geometry.penalty_width / 2.0

    rects: list[Rect] = []
    if robot_id != game.our_goalkeeper:
        m = RULE_MARGIN
        rects.append(Rect(-far, -hl + geometry.penalty_depth + m, -half_area - m, half_area + m))
    if game.play is not Play.RUN:
        m = THEIR_AREA_STOPPAGE_CLEARANCE + RULE_MARGIN
        rects.append(Rect(hl - geometry.penalty_depth - m, far, -half_area - m, half_area + m))

    circles: list[Circle] = []
    if game.min_ball_distance > 0.0:
        circles.append(Circle(world.ball.x, world.ball.y, game.min_ball_distance + RULE_MARGIN))

    # The physical wall is boundary_width outside the lines.
    reach_x = hl + geometry.boundary_width - ROBOT_RADIUS
    reach_y = hw + geometry.boundary_width - ROBOT_RADIUS
    return KeepOuts(tuple(rects), tuple(circles), Rect(-reach_x, reach_x, -reach_y, reach_y))

