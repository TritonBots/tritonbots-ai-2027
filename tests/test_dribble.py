"""Dribble skill: outputs robot-frame velocities, executes a nudge before hitting the 1 m rule limit, and counts as a success once the ball reaches the target."""

import math

from tbots.core.state import BallState, RobotState, WorldState
from tbots.skills.dribble import NUDGE_AT, Dribble


def world(rx, ry, theta, bx, by, has_ball=True):
    return WorldState(t=0.0, ball=BallState(x=bx, y=by),
                      us={0: RobotState(robot_id=0, x=rx, y=ry, theta=theta, has_ball=has_ball)})


def test_drives_forward_in_robot_frame():
    # Robot faces +y with the target straight ahead: the command must be "forward"
    # (vx > 0), not a field-frame +y velocity handed over as vy.
    w = world(0.0, 0.0, math.pi / 2, 0.0, 0.1)
    s = Dribble(target=(0.0, 3.0))
    s.reset(w, 0)
    cmd = s.step(w, 0)
    assert cmd.vx > 0
    assert abs(cmd.vy) < 1e-6
    assert cmd.dribbler == 1.0


def test_nudges_before_one_metre():
    s = Dribble(target=(4.0, 0.0))
    s.reset(world(0.0, 0.0, 0.0, 0.1, 0.0), 0)
    cmd = s.step(world(NUDGE_AT, 0.0, 0.0, NUDGE_AT + 0.1, 0.0), 0)
    assert cmd.kick_speed > 0
    assert cmd.dribbler == 0.0


def test_success_at_target():
    s = Dribble(target=(1.0, 0.0))
    s.reset(world(0.0, 0.0, 0.0, 0.1, 0.0), 0)
    s.step(world(0.8, 0.0, 0.0, 0.9, 0.0), 0)
    assert s.status() == "success"
