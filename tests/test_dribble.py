"""Dribble skill: robot-frame output, nudge before the 1 m rule, success at target,
observation layout, and that the residual is bounded by the rate limit."""

import math

from tbots.core.state import BallState, RobotState, WorldState
from tbots.skills.dribble import A_CARRY, DT, NUDGE_AT, OBS_SIZE, Dribble


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


def test_residual_sees_fixed_layout_and_is_rate_limited():
    seen = []

    def full_throttle(obs):
        seen.append(obs)
        return [1.0, 1.0, 1.0]          # the largest correction a policy can ask for

    w = world(0.0, 0.0, 0.0, 0.1, 0.0)
    s = Dribble(target=(3.0, 0.0), residual=full_throttle)
    s.reset(w, 0)
    cmd = s.step(w, 0)
    assert len(seen[0]) == OBS_SIZE
    # even a maximal residual cannot change velocity faster than the carry limit
    assert math.hypot(cmd.vx, cmd.vy) <= A_CARRY * DT + 1e-9
    assert s.kick_v == 0.40              # the speed the residual was trained with
