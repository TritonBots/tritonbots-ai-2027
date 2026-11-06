"""Send RobotControl to the simulator (or the radio) over UDP.

Builds ONE RobotControl containing all six RobotCommands and sends it to
UDP 10301 (blue) or 10302 (yellow). One packet per tick, containing every
robot — never one packet per robot.

Also reads RobotControlResponse for dribbler-contact feedback and
simulator errors.

THE RADIO IS NOT HERE YET. This speaks the SSL simulation protocol, which is
what the ER-Force simulator accepts. Real robots need our own radio frame and
firmware fields, which docs/TASKS.md §6 lists as an open problem; when they
exist they belong in a sibling sender with this class's three methods, so
`NetworkBackend` does not change.

WIRE FACTS (verified against the ER-Force simulator, SETUP_LOG TASK-011):
  - The protocol is proto2 and `MoveLocalVelocity`'s three fields are
    `required`. A packet missing one does not serialise.
  - The simulator answers on the socket that sent the command, one
    RobotControlResponse per packet, with `feedback` for every commanded
    robot that exists.
  - A command for a robot id that is not on the field is dropped silently:
    no error, no feedback.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Sequence

from tbots._pb.ssl_simulation_robot_control_pb2 import RobotControl
from tbots._pb.ssl_simulation_robot_feedback_pb2 import RobotControlResponse
from tbots.core.command import RobotCommand
from tbots.core.perspective import IDENTITY, Perspective
from tbots.core.state import RobotFeedback

BLUE_PORT = 10301
YELLOW_PORT = 10302

# Interim, not measured. The simulation protocol takes kick angle in degrees
# and dribbler speed in rpm; our command carries a chip flag and a 0..1
# dribbler. Neither the protocol nor the ER-Force spec protos state the
# robot's chip angle or dribbler maximum. Both are constructor arguments so
# that a measurement replaces them without touching this file.
DEFAULT_CHIP_ANGLE_DEG = 45.0
DEFAULT_DRIBBLER_MAX_RPM = 1000.0


def encode(commands: Sequence[RobotCommand], chip_angle_deg: float = DEFAULT_CHIP_ANGLE_DEG,
           dribbler_max_rpm: float = DEFAULT_DRIBBLER_MAX_RPM) -> RobotControl:
    """Our commands -> one RobotControl. No frame transform: see __init__."""
    rc = RobotControl()
    for cmd in commands:
        c = cmd.clamped()
        out = rc.robot_commands.add()
        out.id = c.robot_id
        lv = out.move_command.local_velocity
        lv.forward = c.vx          # m/s and rad/s on both sides of the wire:
        lv.left = c.vy             # the simulation protocol is SI already.
        lv.angular = c.vtheta
        out.kick_speed = c.kick_speed
        out.kick_angle = chip_angle_deg if (c.chip and c.kick_speed > 0.0) else 0.0
        out.dribbler_speed = c.dribbler * dribbler_max_rpm
    return rc


class RobotControlSender:
    def __init__(self, host: str = "127.0.0.1", port: int | None = None,
                 perspective: Perspective = IDENTITY, *,
                 chip_angle_deg: float = DEFAULT_CHIP_ANGLE_DEG,
                 dribbler_max_rpm: float = DEFAULT_DRIBBLER_MAX_RPM) -> None:
        """`port` defaults to BLUE_PORT / YELLOW_PORT per the perspective.

        The perspective decides which port we talk on -- and nothing else.
        RobotCommand velocities are in the ROBOT'S LOCAL frame and must NOT
        be flipped: the robot's heading was already normalised on the way in,
        so flipping here would apply the rotation twice and drive every robot
        backwards for one half of every match.

        Colours can change at half time (SETUP_LOG TASK-013), so `send()`
        also accepts the current perspective and re-picks the port on every
        packet. An explicit `port` pins it and wins over both.
        """
        self._host = host
        self._port = port
        self._perspective = perspective
        self._chip_angle_deg = chip_angle_deg
        self._dribbler_max_rpm = dribbler_max_rpm
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._sock.setblocking(False)
        self._feedback: dict[int, RobotFeedback] = {}
        self._errors: list[tuple[str, str]] = []
        self.sent = 0

    def port_for(self, perspective: Perspective | None = None) -> int:
        if self._port is not None:
            return self._port
        p = perspective if perspective is not None else self._perspective
        if not p.resolved:
            # Guessing blue would drive the OTHER team's robots in a
            # simulator, and nothing at all on a real field.
            raise RuntimeError("our colour is unresolved; refusing to pick a "
                               "command port. Has the referee named us?")
        return YELLOW_PORT if p.we_are_yellow else BLUE_PORT

    def send(self, commands: Sequence[RobotCommand],
             perspective: Perspective | None = None) -> None:
        """One packet, every robot. `perspective` only picks the port."""
        pkt = encode(commands, self._chip_angle_deg, self._dribbler_max_rpm)
        self._sock.sendto(pkt.SerializeToString(), (self._host, self.port_for(perspective)))
        self.sent += 1

    def _pump(self) -> None:
        while True:
            try:
                data = self._sock.recv(65535)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                # ICMP port-unreachable surfaces here on Linux when nothing
                # listens. That is "no simulator", not a reason to crash.
                return
            t = time.time()
            resp = RobotControlResponse()
            resp.ParseFromString(data)
            for fb in resp.feedback:
                self._feedback[fb.id] = RobotFeedback(
                    robot_id=fb.id, t=t, has_ball=fb.dribbler_ball_contact)
            if resp.errors:
                self._errors = [(e.code, e.message) for e in resp.errors]

    def feedback(self) -> dict[int, RobotFeedback]:
        """Latest RobotControlResponse, decoded. Dribbler contact and errors.

        Latched per robot: an entry stays until that robot reports again, so
        check `t` to know how old it is. Errors are on `errors`.
        """
        self._pump()
        return dict(self._feedback)

    @property
    def errors(self) -> list[tuple[str, str]]:
        """(code, message) from the newest response that carried any."""
        self._pump()
        return list(self._errors)

    def close(self) -> None:
        self._sock.close()
