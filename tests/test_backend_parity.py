"""Same skill, same result, both backends.

If GoToPoint converges in rSim but not against ER-Force, we have a
sim-to-sim gap -- and we would much rather find that in September than in April.
"""

import os

import pytest

from tbots.backends.base import Scenario
from tbots.backends.rsim import RSimBackend
from tbots.core.command import stop
from tbots.core.geometry import dist
from tbots.skills.go_to_point import GoToPoint

TARGET = (2.0, 1.0)


START = (-1.0, -1.0)


def _run(backend):
    return _drive(backend, backend.reset(Scenario.single_robot_at(*START)))


def _drive(backend, world, target=TARGET):
    skill = GoToPoint(target=target)
    skill.reset(world, 0)
    for _ in range(600):
        world = backend.step([skill.step(world, 0)])
        if skill.status() == "success":
            break
    return dist(world.us[0].pos, target)


def test_rsim_converges():
    b = RSimBackend(n_us=1, n_them=0)
    try:
        assert _run(b) < 0.06
    finally:
        b.close()


@pytest.mark.skipif(
    os.environ.get("TBOTS_NETWORK_TESTS") != "1",
    reason="requires a running simulator; set TBOTS_NETWORK_TESTS=1",
)
def test_network_converges():
    """GoToPoint against ER-Force, through vision, the tracker and the wire.

    No reset(): teleporting needs SimulatorCommand (TASK-015, blocked on the
    proto collision in SETUP_LOG Step 7). Instead robot 0 DRIVES to rSim's
    start pose first, so the measured trip is the same trip rSim makes and a
    second run is not trivially already there.
    """
    from tests.test_network import make_backend

    b = make_backend()
    try:
        world = b.observe()
        assert 0 in world.us, "robot 0 of our colour is not on the field"
        assert _drive(b, world, START) < 0.06
        assert _drive(b, b.observe()) < 0.06
    finally:
        b.step([stop(0)])
        b.close()
