"""TASK-011: RobotControlSender. Encoding, port choice, and the wire.

The packet is captured on a real UDP socket on this host. The live check
against the ER-Force simulator is in test_network.py.
"""

from __future__ import annotations

import socket

import pytest

from tbots._pb.ssl_simulation_robot_control_pb2 import RobotControl
from tbots.core.command import RobotCommand
from tbots.core.perspective import UNRESOLVED, Perspective
from tbots.net.robot_control import (
    BLUE_PORT,
    DEFAULT_CHIP_ANGLE_DEG,
    YELLOW_PORT,
    RobotControlSender,
    encode,
)


def test_encode_is_local_frame_si_and_clamped():
    rc = encode([RobotCommand(4, vx=1.0, vy=-0.5, vtheta=2.0, kick_speed=9.0,
                              chip=True, dribbler=0.5)], dribbler_max_rpm=1000.0)
    (c,) = rc.robot_commands
    lv = c.move_command.local_velocity
    assert (c.id, lv.forward, lv.left, lv.angular) == (4, 1.0, -0.5, 2.0)
    assert c.kick_speed == pytest.approx(6.5)           # MAX_KICK_SPEED clamp
    assert c.kick_angle == DEFAULT_CHIP_ANGLE_DEG
    assert c.dribbler_speed == pytest.approx(500.0)


def test_flat_kick_and_no_kick_have_zero_angle():
    rc = encode([RobotCommand(0, kick_speed=3.0), RobotCommand(1, chip=True)])
    assert [c.kick_angle for c in rc.robot_commands] == [0.0, 0.0]


def test_zero_command_still_serialises():
    """proto2 `required` fields: a stopped robot must still be encodable."""
    data = encode([RobotCommand(0)]).SerializeToString()
    assert RobotControl.FromString(data).robot_commands[0].move_command.HasField(
        "local_velocity")


def test_port_follows_colour_and_refuses_to_guess():
    s = RobotControlSender()
    try:
        assert s.port_for(Perspective(we_are_yellow=False)) == BLUE_PORT
        assert s.port_for(Perspective(we_are_yellow=True, flip=True)) == YELLOW_PORT
        with pytest.raises(RuntimeError):
            s.port_for(UNRESOLVED)
    finally:
        s.close()


def test_one_packet_per_tick_with_every_robot():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(1.0)
    s = RobotControlSender(port=rx.getsockname()[1])
    try:
        s.send([RobotCommand(i, vx=0.1 * i) for i in range(6)])
        rc = RobotControl.FromString(rx.recv(65535))
        assert [c.id for c in rc.robot_commands] == list(range(6))
        rx.settimeout(0.05)
        with pytest.raises(TimeoutError):
            rx.recv(65535)
    finally:
        s.close()
        rx.close()


def test_feedback_is_empty_when_nothing_answers():
    s = RobotControlSender(port=9)           # discard; nothing replies
    try:
        s.send([RobotCommand(0)])
        assert s.feedback() == {}
        assert s.errors == []
    finally:
        s.close()
