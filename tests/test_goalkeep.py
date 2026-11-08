"""Goalkeep, the hand-written keeper ported from ai-team. Real rSim throughout.

The shot test carries its own control: every shot is first played with the
keeper parked out of the way, and must score. A keeper "saving" shots that
would have missed anyway proves nothing.
"""

from __future__ import annotations

import math
import random

import pytest

from tbots.backends.base import Scenario
from tbots.backends.rsim import RSimBackend
from tbots.core.geometry import DIV_B
from tbots.core.state import BallState, RobotState, WorldState
from tbots.skills import Goalkeep, build_skill
from tbots.skills.goalkeep import ball_travel_time, bisector_target

GOAL_X = DIV_B.our_goal[0]
HALF_GOAL = DIV_B.goal_width / 2.0
KEEPER_HOME = (GOAL_X + 0.2, 0.0, 0.0)
PARKED = (GOAL_X + 0.2, DIV_B.half_width - 0.2, 0.0)


@pytest.fixture(scope="module")
def backend():
    b = RSimBackend(n_us=1, n_them=0)
    yield b
    b.close()


def _world(ball: tuple[float, float], vel: tuple[float, float] = (0.0, 0.0),
           keeper: tuple[float, float] = (GOAL_X + 0.2, 0.0)) -> WorldState:
    return WorldState(t=0.0, ball=BallState(*ball, vx=vel[0], vy=vel[1]),
                      us={0: RobotState(0, *keeper, 0.0)})


def _shots(n: int, seed: int) -> list[tuple[float, float, float, float]]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        sx, sy = rng.uniform(-2.5, 0.5), rng.uniform(-2.5, 2.5)
        ty, speed = rng.uniform(-0.45, 0.45), rng.uniform(3.0, 6.0)
        dx, dy = GOAL_X - sx, ty - sy
        d = math.hypot(dx, dy)
        out.append((sx, sy, speed * dx / d, speed * dy / d))
    return out


def _play(backend, ball, keeper_pose, keeper_on: bool, ticks: int = 240):
    world = backend.reset(Scenario(ball=ball, us=(keeper_pose,)))
    keeper = Goalkeep()
    keeper.reset(world, 0)
    left_area = False
    for _ in range(ticks):
        world = backend.step([keeper.step(world, 0)] if keeper_on else [])
        me = world.us[0]
        if keeper_on and not DIV_B.inside_our_defense_area(me.x, me.y, margin=0.01):
            left_area = True
        if world.ball.x < GOAL_X:
            return ("goal" if abs(world.ball.y) < HALF_GOAL else "wide"), left_area, world
    return "no_goal", left_area, world


def test_registered():
    assert isinstance(build_skill("goalkeep"), Goalkeep)


def test_ball_travel_time():
    assert ball_travel_time(0.0, 1.0) == 0.0
    assert ball_travel_time(1.0, 0.0) == math.inf
    # 1 m/s stops after 1 / (2 * 0.7) = 0.714 m.
    assert ball_travel_time(0.8, 1.0) == math.inf
    assert ball_travel_time(1.0, 4.0) == pytest.approx((4 - math.sqrt(16 - 1.4)) / 0.7)


def test_bisector_target_is_on_the_line_for_a_central_ball():
    x, y = bisector_target((0.0, 0.0))
    assert y == pytest.approx(0.0, abs=1e-9)
    assert GOAL_X < x <= GOAL_X + 0.4 + 1e-9


def test_bisector_target_shifts_toward_the_ball_and_stays_near_the_mouth():
    for by in (-2.5, -1.0, 1.0, 2.5):
        x, y = bisector_target((-2.0, by))
        assert math.copysign(1.0, y) == math.copysign(1.0, by)
        assert abs(y) <= HALF_GOAL + 0.2
        assert DIV_B.inside_our_defense_area(x, y)


def test_bisector_hugs_the_line_when_the_ball_is_close():
    x, _ = bisector_target((GOAL_X + 0.5, 0.0))
    far_x, _ = bisector_target((0.0, 0.0))
    assert x < far_x


def test_shot_detection_needs_pace_and_direction():
    keeper = Goalkeep()
    assert keeper.shot_crossing(_world((-2.0, 0.0), (-4.0, 0.0))) == pytest.approx(0.0)
    assert keeper.shot_crossing(_world((-2.0, 0.0), (4.0, 0.0))) is None       # away
    assert keeper.shot_crossing(_world((-2.0, 0.0), (-4.0, 4.0))) is None      # wide
    assert keeper.shot_crossing(_world((-2.0, 0.0), (-1.0, 0.0))) is None      # dies short


def test_clearance_points_upfield():
    keeper = Goalkeep()
    world = _world((GOAL_X + 0.3, 0.5), keeper=(GOAL_X + 0.2, 0.5))
    angle = keeper.clearance_angle(world, world.us[0])
    assert math.cos(angle) > 0.5


def test_keeper_saves_shots_that_would_score(backend):
    """Ten seeded on-target shots at 3-6 m/s: all score unguarded, none past the keeper."""
    for shot in _shots(10, seed=0):
        unguarded, _, _ = _play(backend, shot, PARKED, keeper_on=False)
        if unguarded != "goal":
            continue  # the control says this shot would not have scored
        outcome, left_area, _ = _play(backend, shot, KEEPER_HOME, keeper_on=True)
        assert outcome != "goal", f"conceded {shot}"
        assert not left_area, f"keeper left the defense area on {shot}"


def test_seeded_shots_mostly_score_unguarded(backend):
    """Guard on the test above: the control must actually be scoring."""
    goals = sum(_play(backend, s, PARKED, keeper_on=False)[0] == "goal"
                for s in _shots(10, seed=0))
    assert goals >= 8


def test_keeper_claims_and_clears_a_loose_ball_near_goal(backend):
    """A slow ball inside the 0.6 m charge radius (the tournament default) is
    claimed and cleared upfield."""
    outcome, left_area, world = _play(
        backend, (GOAL_X + 0.6, 0.25, -0.4, 0.0), KEEPER_HOME, keeper_on=True, ticks=600)
    assert outcome != "goal"
    assert not left_area
    assert not DIV_B.inside_our_defense_area(world.ball.x, world.ball.y)
    assert world.ball.x > -2.0
