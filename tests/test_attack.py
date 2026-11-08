"""Attack, the hand-written attacker ported from ai-team. Real rSim throughout."""

from __future__ import annotations

import math
import random

import pytest

from tbots.backends.base import Scenario
from tbots.backends.rsim import RSimBackend
from tbots.core.geometry import DIV_B
from tbots.core.state import BallState, RobotState, WorldState
from tbots.skills import Attack, build_skill
from tbots.skills.attack import choose_shot_target, dist_point_to_segment

GOAL_X = DIV_B.their_goal[0]
HALF_GOAL = DIV_B.goal_width / 2.0


def _world(ball, them=()) -> WorldState:
    return WorldState(t=0.0, ball=BallState(*ball),
                      us={0: RobotState(0, ball[0] - 0.5, ball[1], 0.0)},
                      them={i: RobotState(i, *p) for i, p in enumerate(them)})


def _play(backend, scenario: Scenario, ticks: int):
    world = backend.reset(scenario)
    skill = Attack()
    skill.reset(world, 0)
    entered = False
    for _ in range(ticks):
        world = backend.step([skill.step(world, 0)])
        me = world.us[0]
        if DIV_B.inside_their_defense_area(me.x, me.y):
            entered = True
        if world.ball.x > GOAL_X:
            return ("goal" if abs(world.ball.y) < HALF_GOAL else "wide"), entered
    return "no_goal", entered


def test_registered():
    assert isinstance(build_skill("attack"), Attack)


def test_dist_point_to_segment():
    assert dist_point_to_segment((0.0, 1.0), (-1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert dist_point_to_segment((3.0, 0.0), (-1.0, 0.0), (1.0, 0.0)) == pytest.approx(2.0)


def test_shot_goes_away_from_the_keeper():
    high = choose_shot_target((3.0, 0.0), _world((3.0, 0.0), them=[(4.4, 0.2, math.pi)]))
    low = choose_shot_target((3.0, 0.0), _world((3.0, 0.0), them=[(4.4, -0.2, math.pi)]))
    assert high.target[1] < 0 < low.target[1]


def test_a_robot_in_the_lane_marks_it_blocked():
    # Keeper central, a defender sitting on the line to the -y corner.
    world = _world((3.0, 0.0), them=[(4.4, 0.0, math.pi), (3.7, -0.17, math.pi)])
    shot = choose_shot_target((3.0, 0.0), world)
    assert shot.target[1] > 0
    assert not shot.blocked


def test_scores_on_an_empty_goal_from_anywhere(backend_1v0):
    """Twelve seeded starts across the field, each within 15 s, never in their area."""
    rng = random.Random(0)
    for _ in range(12):
        ball = (rng.uniform(-1.5, 3.0), rng.uniform(-2.5, 2.5))
        me = (rng.uniform(-3.0, 1.0), rng.uniform(-2.5, 2.5), rng.uniform(-3.0, 3.0))
        scenario = Scenario(ball=(*ball, 0.0, 0.0), us=(me,))
        outcome, entered = _play(backend_1v0, scenario, ticks=900)
        assert outcome == "goal", f"{outcome} from ball={ball} me={me}"
        assert not entered, f"entered their defense area from ball={ball} me={me}"


def test_scores_past_a_keeper_standing_off_centre():
    b = RSimBackend(n_us=1, n_them=1)
    try:
        for ky in (0.2, -0.2):
            scenario = Scenario(ball=(2.5, 0.3, 0.0, 0.0), us=((1.0, 0.0, 0.0),),
                                them=((GOAL_X - 0.1, ky, math.pi),))
            outcome, _ = _play(b, scenario, ticks=600)
            assert outcome == "goal", f"keeper at y={ky}"
    finally:
        b.close()


@pytest.fixture(scope="module")
def backend_1v0():
    b = RSimBackend(n_us=1, n_them=0)
    yield b
    b.close()
