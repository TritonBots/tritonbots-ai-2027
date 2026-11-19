"""TASK-078: the action codec.

Gate: a fixed action sequence decodes to a fixed RobotCommand sequence
(golden); a step change in commanded velocity is rate-limited.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tbots.core.command import RobotCommand
from tbots.core.geometry import MAX_KICK_SPEED
from tbots.core.limits import MAX_ANGULAR_ACCEL, MAX_LINEAR_ACCEL
from tbots.rl.codec import build_codec, codec_names

DT = 1.0 / 60.0


def act(vx: float, vy: float, w: float, kick: int = 0, dribble: int = 0) -> dict:
    return {"move": np.array([vx, vy, w], np.float32),
            "flags": np.array([kick, dribble], np.int8)}


def test_registry_lists_and_builds_the_skill_codec():
    assert "skill_v1" in codec_names()
    with pytest.raises(KeyError, match="unknown codec"):
        build_codec("nope")


def test_golden_sequence():
    codec = build_codec("skill_v1", max_speed=2.0, max_angular_speed=6.0,
                        kick_speed=4.0, dribbler_speed=0.5, dt=DT)
    codec.reset()
    seq = [act(1, 0, 0), act(1, 0, 0), act(0, 1, -1, kick=1),
           act(0, 0, 0, dribble=1), act(-1, -1, 1, 1, 1)]
    out = [c for a in seq for c in codec.decode(a, robot_id=0)]
    dv = MAX_LINEAR_ACCEL * DT     # 0.05 m/s per tick
    dw = MAX_ANGULAR_ACCEL * DT    # 0.333.. rad/s per tick
    s = math.sqrt(0.5)             # a unit step at 45 degrees, split over axes
    golden = [
        RobotCommand(0, vx=dv, vy=0.0, vtheta=0.0),
        RobotCommand(0, vx=2 * dv, vy=0.0, vtheta=0.0),
        # (0.1, 0) -> (0, 2): the 0.05 step points along (-0.1, 2), normalised.
        RobotCommand(0, vx=0.1 - dv * 0.1 / math.hypot(0.1, 2.0),
                     vy=dv * 2.0 / math.hypot(0.1, 2.0), vtheta=-dw, kick_speed=4.0),
    ]
    for got, want in zip(out[:3], golden, strict=True):
        assert got.robot_id == want.robot_id
        assert got.vx == pytest.approx(want.vx, abs=1e-12)
        assert got.vy == pytest.approx(want.vy, abs=1e-12)
        assert got.vtheta == pytest.approx(want.vtheta, abs=1e-12)
        assert got.kick_speed == want.kick_speed and got.dribbler == want.dribbler
    # Decelerating toward zero, dribbler on, no kick.
    assert out[3].dribbler == 0.5 and out[3].kick_speed == 0.0
    assert math.hypot(out[3].vx - out[2].vx, out[3].vy - out[2].vy) == pytest.approx(dv)
    assert out[3].vtheta == pytest.approx(-dw + dw)
    # The pinned table. skill_v1 is immutable once a checkpoint exists: if
    # this fails, you changed what an old policy's outputs mean. Register a
    # new codec name instead.
    pinned = [
        (0.05, 0.0, 0.0, 0.0, 0.0),
        (0.1, 0.0, 0.0, 0.0, 0.0),
        (0.09750311915280539, 0.04993761694389223, -0.3333333333333333, 4.0, 0.0),
        (0.05300039440517364, 0.027144909995472876, 0.0, 0.0, 0.5),
        (0.01733217659494487, -0.007894756644523566, 0.3333333333333333, 4.0, 0.5),
    ]
    assert [(c.vx, c.vy, c.vtheta, c.kick_speed, c.dribbler) for c in out] == [
        pytest.approx(row, abs=1e-12) for row in pinned]
    assert all(type(v) is float for c in out for v in (c.vx, c.vy, c.vtheta))
    # Diagonal full-stick is the speed limit along the diagonal, not sqrt(2) x it.
    codec.reset()
    for _ in range(200):
        (c,) = codec.decode(act(-1, -1, 1, 1, 1), robot_id=3)
    assert (c.vx, c.vy) == pytest.approx((-2.0 * s, -2.0 * s))
    assert c.vtheta == pytest.approx(6.0)
    assert (c.robot_id, c.kick_speed, c.dribbler) == (3, 4.0, 0.5)


def test_a_step_change_is_rate_limited_and_then_reached():
    codec = build_codec("skill_v1", max_speed=2.0, max_angular_speed=6.0, dt=DT)
    codec.reset()
    vxs, ws = [], []
    for _ in range(60):
        (c,) = codec.decode(act(1, 0, 1), robot_id=0)
        vxs.append(c.vx)
        ws.append(c.vtheta)
    steps = np.diff([0.0, *vxs])
    assert max(steps) == pytest.approx(MAX_LINEAR_ACCEL * DT)
    assert max(np.diff([0.0, *ws])) == pytest.approx(MAX_ANGULAR_ACCEL * DT)
    # 2 m/s at 3 m/s^2 is 40 ticks; 6 rad/s at 20 rad/s^2 is 18.
    assert vxs[39] == pytest.approx(2.0) and vxs[38] < 2.0
    assert ws[17] == pytest.approx(6.0) and ws[16] < 6.0


def test_kick_speed_is_clamped_to_the_rule():
    codec = build_codec("skill_v1", kick_speed=9.0, dt=DT)
    assert codec.kick_speed == MAX_KICK_SPEED
    (c,) = codec.decode(act(0, 0, 0, kick=1), robot_id=0)
    assert c.kick_speed == MAX_KICK_SPEED


def test_action_repeat_emits_one_rate_limited_command_per_tick():
    codec = build_codec("skill_v1", action_repeat=4, dt=DT)
    codec.reset()
    out = codec.decode(act(1, 0, 0), robot_id=0)
    assert len(out) == 4
    assert [c.vx for c in out] == pytest.approx([MAX_LINEAR_ACCEL * DT * k for k in (1, 2, 3, 4)])


def test_out_of_range_actions_are_clipped_and_reset_forgets_velocity():
    codec = build_codec("skill_v1", max_speed=2.0, dt=DT)
    codec.reset()
    for _ in range(100):
        (c,) = codec.decode(act(5, 0, 0), robot_id=0)
    assert c.vx == pytest.approx(2.0)
    codec.reset()
    (c,) = codec.decode(act(0, 0, 0), robot_id=0)
    assert c.vx == 0.0


def test_spec_records_every_parameter_and_the_space_matches():
    codec = build_codec("skill_v1", max_speed=1.5, action_repeat=2, dt=DT)
    spec = codec.spec()
    assert spec["name"] == "skill_v1"
    for k in ("max_speed", "max_angular_speed", "kick_speed", "chip", "dribbler_speed",
              "max_linear_accel", "max_angular_accel", "action_repeat", "dt"):
        assert k in spec
    assert spec["max_speed"] == 1.5 and spec["action_repeat"] == 2
    sample = codec.action_space.sample()
    assert len(codec.decode(sample, robot_id=0)) == 2
