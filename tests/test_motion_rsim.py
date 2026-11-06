"""TASK-086 through real physics: moving opponents, driven by rSim.

The opponents are real rSim robots, commanded every tick through
``SimBackend.step(commands, opponent_commands)``. Their velocities reach the
planner the same way they will in training: differenced by rSim, read once
per step, in the ``WorldState`` the backend returns.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from tbots.backends.base import Scenario
from tbots.backends.rsim import RSimBackend
from tbots.core.command import RobotCommand
from tbots.core.geometry import ROBOT_RADIUS, dist
from tbots.skills.go_to_point import GoToPoint

CONTACT = 2.0 * ROBOT_RADIUS
TARGET = (2.0, 0.0)

# name -> (opponent pose, opponent robot-local forward speed). Each one is a
# real conflict: with the opponent hidden from the planner, our robot touches
# it (test_scenarios_are_real_conflicts).
SCENARIOS = {
    "crossing": ((0.0, -2.0, math.pi / 2), 1.5),
    "head_on": ((2.5, 0.0, math.pi), 1.5),
    "static": ((0.0, 0.0, 0.0), 0.0),
}


@pytest.fixture
def backend():
    b = RSimBackend(n_us=1, n_them=1)
    yield b
    b.close()


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_moving_opponent_is_avoided_in_rsim(backend, name):
    pose, speed = SCENARIOS[name]
    world = backend.reset(Scenario(ball=(4.0, 2.8, 0.0, 0.0),
                                   us=((-2.0, 0.0, 0.0),), them=(pose,)))
    skill = GoToPoint(TARGET)
    skill.reset(world, 0)

    closest = math.inf
    worst_tracking = 0.0
    previous_global = None
    opponent_moved = 0.0
    start_them = world.them[0].pos
    for tick in range(300):
        command = skill.step(world, 0)
        if skill.status() == "success":
            break
        world = backend.step([command], [RobotCommand(0, vx=speed)])
        me = world.us[0]
        closest = min(closest, dist(me.pos, world.them[0].pos))
        opponent_moved = max(opponent_moved, dist(start_them, world.them[0].pos))
        # The planner only asks for what rSim follows (docs/RSIM_FACTS.md).
        if previous_global is not None and tick > 3:
            worst_tracking = max(worst_tracking, math.hypot(me.vx - previous_global[0],
                                                            me.vy - previous_global[1]))
        c, s = math.cos(me.theta), math.sin(me.theta)
        previous_global = (c * command.vx - s * command.vy, s * command.vx + c * command.vy)

    assert skill.status() == "success", f"did not arrive; at {world.us[0].pos}"
    assert closest > CONTACT
    if speed > 0.0:
        assert opponent_moved > 1.0, "the opponent must really have moved"
    assert worst_tracking < 0.15


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenarios_are_real_conflicts(backend, name):
    """Same planner, opponent hidden from it: contact. Otherwise the test
    above could pass because nobody was ever in the way."""
    pose, speed = SCENARIOS[name]
    world = backend.reset(Scenario(ball=(4.0, 2.8, 0.0, 0.0),
                                   us=((-2.0, 0.0, 0.0),), them=(pose,)))
    skill = GoToPoint(TARGET)
    skill.reset(world, 0)
    closest = math.inf
    for _ in range(300):
        command = skill.step(replace(world, them={}), 0)
        if skill.status() == "success":
            break
        world = backend.step([command], [RobotCommand(0, vx=speed)])
        closest = min(closest, dist(world.us[0].pos, world.them[0].pos))
    assert closest < CONTACT + 0.005
