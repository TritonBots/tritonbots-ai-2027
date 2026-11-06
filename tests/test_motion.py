"""TASK-086: dynamic MPC obstacle avoidance, closed loop, no physics.

Every scenario that matters has MOVING opponents. The world is rebuilt every
tick from their true positions and velocities, and clearance is checked in
continuous time between ticks, so a robot that passes through an opponent
between two samples is caught. tests/test_motion_rsim.py repeats the
important scenarios through real rSim physics.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import replace

import numpy as np
import pytest

from tbots.core.command import RobotCommand
from tbots.core.gamestate import GameState, Play
from tbots.core.geometry import DIV_B, ROBOT_RADIUS
from tbots.core.limits import MotionLimits
from tbots.motion import (
    ConstantAccelerationPredictor,
    ConstantVelocityPredictor,
    MPCPlanner,
    Obstacle,
    ObstacleKind,
    obstacles_from_world,
    swept_clearance,
)
from tbots.skills.go_to_point import GoToPoint
from tests.kinematic import DT, KinematicWorld, Opponent, constant, switch_at, to_global

CONTACT = 2.0 * ROBOT_RADIUS
START = (-2.0, 0.0, 0.0)
TARGET = (2.0, 0.0)
LIMITS = MotionLimits()


def drive(sim: KinematicWorld, skill: GoToPoint, seconds: float = 5.0,
          robot_id: int = 0) -> tuple[float | None, list[RobotCommand]]:
    """Run until success. Returns (arrival time or None, running commands)."""
    world = sim.world()
    skill.reset(world, robot_id)
    commands: list[RobotCommand] = []
    for _ in range(int(round(seconds / DT))):
        command = skill.step(world, robot_id)
        if skill.status() == "success":
            return sim.t, commands
        commands.append(command)
        world = sim.step([command])
    return None, commands


# -- dynamic scenarios -------------------------------------------------------


# name -> (opponents, predictor). Every scenario is a real conflict: with the
# opponents hidden from the planner, our robot touches one
# (test_scenarios_are_real_conflicts).
SCENARIOS = {
    # Free-space arrival at x = 0 is ~1.5 s; this opponent reaches y = 0
    # there at 1.5 s.
    "crossing": (lambda: {0: Opponent(0.0, -2.25, constant(0.0, 1.5))},
                 "constant_velocity"),
    "head_on": (lambda: {0: Opponent(2.5, 0.0, constant(-2.0, 0.0))},
                "constant_velocity"),
    # Harmless until t = 0.6 s, then turns onto our path.
    "manoeuvring": (lambda: {0: Opponent(1.5, -1.0, switch_at(0.6, (0.5, 0.0), (-0.85, 0.85)))},
                    "constant_acceleration"),
    "three_sweeping": (lambda: {0: Opponent(-0.8, -1.5, constant(0.0, 1.0)),
                                1: Opponent(0.5, 2.0, constant(0.0, -1.2)),
                                2: Opponent(1.4, -2.4, constant(0.0, 1.5))},
                       "constant_velocity"),
    "static": (lambda: {0: Opponent(0.0, 0.0)}, "constant_velocity"),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_moving_opponents_are_avoided(name):
    opponents, predictor = SCENARIOS[name]
    sim = KinematicWorld({0: START}, opponents())
    arrived, _ = drive(sim, GoToPoint(TARGET, predictor=predictor))
    assert sim.min_clearance >= CONTACT
    assert arrived is not None and arrived < 5.0


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenarios_are_real_conflicts(name):
    """Same planner, same limits, opponents hidden from it: contact.

    Without this, a scenario could pass because nobody was ever in the way.
    (The old ``avoid=False`` controller has no acceleration limit, so in this
    kinematic world it arrives early and is no control at all.)
    """
    opponents, predictor = SCENARIOS[name]
    sim = KinematicWorld({0: START}, opponents())
    skill = GoToPoint(TARGET, predictor=predictor)
    world = sim.world()
    skill.reset(world, 0)
    for _ in range(300):
        command = skill.step(replace(world, them={}), 0)
        if skill.status() == "success":
            break
        world = sim.step([command])
    assert sim.min_clearance < CONTACT


# -- limits and free space ----------------------------------------------------


def test_every_command_respects_speed_and_acceleration_limits():
    sim = KinematicWorld({0: START}, {
        0: Opponent(0.0, -2.25, constant(0.0, 1.5)),
        1: Opponent(2.5, 0.4, constant(-2.0, 0.0)),
    })
    _, commands = drive(sim, GoToPoint(TARGET))
    previous = (0.0, 0.0)
    for command in commands:
        v = to_global(command, 0.0)   # heading stays 0 with face=None from 0
        assert math.hypot(*v) <= LIMITS.max_v + 1e-9
        dv = math.hypot(v[0] - previous[0], v[1] - previous[1])
        assert dv <= LIMITS.max_a * DT + 1e-9
        previous = v


def test_free_space_arrival_is_near_time_optimal():
    # Rest-to-rest over 4 m under max_v / max_a: accelerate, cruise, brake.
    d, v, a = 4.0, LIMITS.max_v, LIMITS.max_a
    bound = 2.0 * v / a + (d - v * v / a) / v
    arrived, _ = drive(KinematicWorld({0: START}), GoToPoint(TARGET))
    assert arrived is not None
    assert arrived <= 1.15 * bound


# -- rules ---------------------------------------------------------------------


def test_stop_keeps_half_a_metre_from_the_ball():
    stop = GameState(play=Play.STOP, can_move=True, min_ball_distance=0.5)
    sim = KinematicWorld({0: START}, ball=(0.0, 0.0), game=stop)
    world = sim.world()
    skill = GoToPoint(TARGET)
    skill.reset(world, 0)
    closest = math.inf
    for _ in range(300):
        command = skill.step(world, 0)
        if skill.status() == "success":
            break
        world = sim.step([command])
        closest = min(closest, math.hypot(*world.us[0].pos))
    assert skill.status() == "success"
    assert closest >= 0.5


@pytest.mark.parametrize("keeper", [True, False])
def test_only_the_keeper_crosses_our_defense_area(keeper: bool):
    game = GameState(play=Play.RUN, can_move=True, min_ball_distance=0.0,
                     our_goalkeeper=0 if keeper else 5)
    sim = KinematicWorld({0: (-4.0, -1.5, 0.0)}, game=game)
    world = sim.world()
    skill = GoToPoint((-4.0, 1.5))
    skill.reset(world, 0)
    entered = False
    for _ in range(360):
        command = skill.step(world, 0)
        if skill.status() == "success":
            break
        world = sim.step([command])
        entered |= DIV_B.inside_our_defense_area(*world.us[0].pos)
    assert skill.status() == "success"
    assert entered is keeper


# -- the pieces ----------------------------------------------------------------


def test_identical_seeds_give_identical_commands():
    def run():
        sim = KinematicWorld({0: START}, {0: Opponent(0.0, -2.25, constant(0.0, 1.5))})
        return drive(sim, GoToPoint(TARGET))[1]
    assert run() == run()


def test_constant_velocity_prediction_is_exact_and_radius_grows_capped():
    predictor = ConstantVelocityPredictor()
    times = np.linspace(0.0, 1.0, 21)
    robot = Obstacle(id=3, kind=ObstacleKind.OPPONENT, x=1.0, y=2.0, vx=0.5, vy=-1.0, radius=0.2)
    ball = Obstacle(id=0, kind=ObstacleKind.BALL, x=0.0, y=0.0, vx=2.0, vy=0.0, radius=0.12)
    centres, radii = predictor.predict([robot, ball], times, now=0.0)
    assert centres.shape == (2, 21, 2) and radii.shape == (2, 21)
    np.testing.assert_allclose(centres[0, :, 0], 1.0 + 0.5 * times)
    np.testing.assert_allclose(centres[0, :, 1], 2.0 - 1.0 * times)
    assert radii[0, 0] == pytest.approx(0.2)
    assert np.all(np.diff(radii[0]) >= 0.0) and radii[0, -1] > radii[0, 0]
    assert radii[0, -1] <= 0.2 + predictor.growth_cap + 1e-12
    np.testing.assert_allclose(radii[1], 0.12)


def test_constant_acceleration_prediction_follows_a_turn():
    predictor = ConstantAccelerationPredictor()
    times = np.linspace(0.0, 0.5, 11)
    vx = 0.0
    for k in range(10):               # an opponent accelerating at 2 m/s^2 in +x
        vx = 2.0 * k * DT
        robot = Obstacle(id=1, kind=ObstacleKind.OPPONENT, x=0.0, y=0.0, vx=vx, vy=0.0,
                         radius=0.2)
        centres, _ = predictor.predict([robot], times, now=k * DT)
    constant_v = vx * times[-1]
    assert centres[0, -1, 0] > constant_v + 0.1


def test_swept_clearance_catches_a_pass_between_samples():
    # Head-on swap: 2 m apart at both ends of the step, coincident halfway.
    d = swept_clearance(np.array([[-1.0, 0.0]]), np.array([[1.0, 0.0]]),
                        np.array([[1.0, 0.0]]), np.array([[-1.0, 0.0]]))
    assert d[0] == pytest.approx(0.0, abs=1e-12)
    d = swept_clearance(np.array([[-1.0, 0.0]]), np.array([[1.0, 0.0]]),
                        np.array([[0.0, -1.0]]), np.array([[0.0, 1.0]]))
    assert d[0] == pytest.approx(0.0, abs=1e-12)


def test_obstacles_from_world_excludes_self_and_reads_the_rules():
    sim = KinematicWorld({0: START, 1: (0.0, 1.0, 0.0)},
                         {0: Opponent(1.0, 0.0, constant(-1.0, 0.0))},
                         ball=(0.5, 0.5),
                         game=GameState(play=Play.STOP, can_move=True, min_ball_distance=0.5))
    obstacles, keepouts = obstacles_from_world(sim.world(), 0, DIV_B)
    kinds = sorted((o.kind.name, o.id) for o in obstacles)
    assert kinds == [("OPPONENT", 0), ("TEAMMATE", 1)]
    opponent = next(o for o in obstacles if o.kind is ObstacleKind.OPPONENT)
    assert (opponent.vx, opponent.vy) == (-1.0, 0.0)
    assert any(c.radius >= 0.5 for c in keepouts.circles)
    assert keepouts.rects        # non-keeper: our defense area, and theirs on STOP


def test_plan_is_fast_enough_for_six_robots_at_60_hz():
    planner = MPCPlanner(limits=LIMITS)
    rng = np.random.default_rng(0)
    obstacles = [Obstacle(id=i, kind=ObstacleKind.OPPONENT, x=float(x), y=float(y),
                          vx=float(vx), vy=float(vy), radius=0.21)
                 for i, (x, y, vx, vy) in enumerate(rng.uniform(-2, 2, size=(11, 4)))]
    _, keepouts = obstacles_from_world(KinematicWorld({0: START}).world(), 0, DIV_B)
    samples = []
    for k in range(120):
        t0 = time.perf_counter()
        planner.plan((-2.0, 0.0), (0.5, 0.0), TARGET, obstacles, keepouts, now=k * DT)
        samples.append(time.perf_counter() - t0)
    # Loose for CI; the measured figure is recorded in docs/SETUP_LOG.md.
    assert statistics.median(samples[10:]) < 2e-3


def test_route_goes_round_the_defense_area_and_is_direct_when_clear():
    from tbots.motion.route import route

    game = GameState(play=Play.RUN, can_move=True, min_ball_distance=0.0, our_goalkeeper=5)
    _, keepouts = obstacles_from_world(KinematicWorld({0: START}, game=game).world(), 0, DIV_B)
    area = keepouts.rects[0]
    path = route((-4.0, -1.5), (-4.0, 1.5), keepouts)
    assert path[-1] == (-4.0, 1.5)
    assert len(path) == 3
    assert all(x > area.x_max for x, _ in path[:-1])     # round the front, not the wall
    assert route((-2.0, 0.0), (2.0, 0.0), keepouts) == [(2.0, 0.0)]


def test_hold_keeps_station_and_dodges_after_arriving():
    """A finished skill is parked by the Coach and cannot dodge. hold=True
    keeps the skill running at its target, so it still gets out of the way."""
    # Arrive first, then an opponent drives straight through the target.
    sim = KinematicWorld({0: (1.9, 0.0, 0.0)}, {0: Opponent(4.0, 0.0, switch_at(
        1.0, (0.0, 0.0), (-1.5, 0.0)))})
    skill = GoToPoint(TARGET, hold=True)
    world = sim.world()
    skill.reset(world, 0)
    arrived_at = None
    for _ in range(300):
        command = skill.step(world, 0)
        assert skill.status() == "running"
        if skill.arrived and arrived_at is None:
            arrived_at = sim.t
        world = sim.step([command])
    assert arrived_at is not None and arrived_at < 1.0
    assert sim.min_clearance >= CONTACT
    assert math.dist(world.us[0].pos, TARGET) < 0.1      # and came back
