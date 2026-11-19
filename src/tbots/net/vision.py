"""Receive SSL-Vision detection frames and merge them into world frames.

Joins 224.5.23.2:10006, parses SSL_WrapperPacket, buffers detection frames
per camera_id, and merges them into one world frame by t_capture.

Division B uses four cameras at roughly 60 Hz each, so you will receive
~240 packets/second, staggered. DO NOT run the control loop once per
packet — merge, then tick once.

WHAT "MERGE" MEANS HERE. The receiver does not average robots across
cameras; that is fusion, and fusion is the tracker's job (it owns the
filters that know how much to trust each measurement). The receiver's job is
the bookkeeping that makes fusion possible:

  - decode, convert mm to m (Rule 4: this module is the wire boundary),
  - drop duplicate packets (same camera, same frame_number),
  - put every frame on OUR clock (see below),
  - hand over all frames since the last tick in t_capture order, and
  - decide when a tick's worth has arrived: `wait_for_next_frame` returns
    once every camera heard recently has delivered a new frame, so four
    staggered 60 Hz cameras produce one 60 Hz tick, not four.

CLOCKS. `t_capture` is stamped by the vision computer's clock, which at a
real event is not synchronised with ours. Every frame is re-stamped onto our
`time.time()` clock: the offset between the two is estimated as the minimum,
over recent packets, of `t_received - t_sent`. A minimum, because network
delay is never negative and its smallest value is the closest we get to
zero; the residual error is the LAN's fastest one-way delay, well under a
millisecond on a wired field network. Against the simulator on this machine
the two clocks are the same clock and the offset converges to ~0.

The receive time is read when the packet is drained, so the estimate needs
SOME packets drained the moment they land. The match loop does that:
`wait_for_next_frame` wakes on each arrival. Packets that queued while the
loop was busy read late, over-estimate the offset, and the minimum ignores
them. (Kernel receive timestamps, SO_TIMESTAMP, would remove the
requirement, but on this WSL2 host loopback multicast packets carry no
arrival stamp and the kernel stamps them at read time -- no better.
SETUP_LOG TASK-010.)
"""

from __future__ import annotations

import collections
import select
import time
from dataclasses import dataclass

from tbots._pb.messages_robocup_ssl_detection_pb2 import SSL_DetectionFrame
from tbots._pb.messages_robocup_ssl_wrapper_pb2 import SSL_WrapperPacket
from tbots.core.state import DetectionBall, DetectionFrame, DetectionRobot
from tbots.core.units import mm_to_m
from tbots.net.multicast import drain, rx_socket

# A camera that has sent nothing for this long no longer gates the tick.
# Generous against 60 Hz (a 30-frame gap), short enough that one camera
# dying mid-match costs one late tick, not a frozen control loop.
_CAMERA_STALE_S = 0.5

# Clock-offset window. Long enough to contain an uncongested packet, short
# enough to follow a drifting vision-PC clock.
_OFFSET_WINDOW = 600


@dataclass(frozen=True, slots=True)
class WireGeometry:
    """The field dimensions SSL-Vision announced, in meters.

    Deliberately only the scalars every source populates. The ER-Force
    simulator leaves `penalty_area_depth`, `penalty_area_width` and
    `center_circle_radius` at 0 and describes them only as field lines, and
    proto3 cannot tell 0 from absent (SETUP_LOG TASK-071). Checking a
    Division on those would fail every simulated match.
    """

    length: float
    width: float
    goal_width: float
    goal_depth: float
    boundary_width: float


def decode_detection(d: SSL_DetectionFrame, offset: float = 0.0) -> DetectionFrame:
    """One SSL_DetectionFrame -> one DetectionFrame, meters, field frame.

    Colours stay wire colours and the frame stays the field's frame. Rule 3
    is applied once, by the tracker, after fusion.
    """
    return DetectionFrame(
        t_capture=d.t_capture + offset,
        camera_id=d.camera_id,
        balls=tuple(
            DetectionBall(x=mm_to_m(b.x), y=mm_to_m(b.y),
                          z=mm_to_m(b.z) if b.HasField("z") else 0.0,
                          confidence=b.confidence)
            for b in d.balls),
        robots_blue=_robots(d.robots_blue),
        robots_yellow=_robots(d.robots_yellow),
    )


def _robots(rs) -> tuple[DetectionRobot, ...]:
    # Orientation is optional on the wire. A robot reported without one would
    # feed heading 0 into its filter, so it is dropped rather than guessed.
    return tuple(DetectionRobot(robot_id=r.robot_id, x=mm_to_m(r.x), y=mm_to_m(r.y),
                                theta=r.orientation, confidence=r.confidence)
                 for r in rs if r.HasField("orientation"))


class VisionReceiver:
    def __init__(self, group: str = "224.5.23.2", port: int = 10006,
                 iface: str = "0.0.0.0") -> None:
        self._sock = rx_socket(group, port, iface)
        self._pending: list[DetectionFrame] = []
        self._last_frame_number: dict[int, int] = {}
        self._last_heard: dict[int, float] = {}       # camera -> local time
        self._fresh: set[int] = set()                 # cameras since last tick
        self._offsets: collections.deque[float] = collections.deque(maxlen=_OFFSET_WINDOW)
        self._geometry: WireGeometry | None = None
        self.packets = 0

    # -- clock --------------------------------------------------------------

    @property
    def clock_offset(self) -> float:
        """Add to a vision-PC timestamp to get our `time.time()`."""
        return min(self._offsets) if self._offsets else 0.0

    # -- geometry -----------------------------------------------------------

    @property
    def geometry(self) -> WireGeometry | None:
        """The newest geometry packet, or None until one has arrived."""
        return self._geometry

    # -- frames -------------------------------------------------------------

    def _pump(self) -> None:
        for data in drain(self._sock):
            t_rx = time.time()
            pkt = SSL_WrapperPacket()
            pkt.ParseFromString(data)          # bare protobuf, no framing
            self.packets += 1
            if pkt.HasField("geometry"):
                f = pkt.geometry.field
                self._geometry = WireGeometry(
                    length=mm_to_m(f.field_length), width=mm_to_m(f.field_width),
                    goal_width=mm_to_m(f.goal_width), goal_depth=mm_to_m(f.goal_depth),
                    boundary_width=mm_to_m(f.boundary_width))
            if not pkt.HasField("detection"):
                continue
            d = pkt.detection
            cam = d.camera_id
            # The same frame can arrive twice (a duplicated datagram, or two
            # vision processes for one camera). Never feed a filter twice.
            if self._last_frame_number.get(cam) == d.frame_number:
                continue
            self._last_frame_number[cam] = d.frame_number
            sent = d.t_sent if d.t_sent > 0.0 else d.t_capture
            self._offsets.append(t_rx - sent)
            self._last_heard[cam] = t_rx
            self._fresh.add(cam)
            # Stamped with the offset as it stands now; it only ever moves
            # by the LAN's jitter, so re-stamping old frames gains nothing.
            self._pending.append(decode_detection(d, self.clock_offset))

    def _tick_ready(self, now: float) -> bool:
        live = {c for c, t in self._last_heard.items() if now - t < _CAMERA_STALE_S}
        return bool(live) and live <= self._fresh

    def wait_for_next_frame(self, timeout: float = 0.05) -> bool:
        """Block until every live camera has sent a new frame, or `timeout`.

        True if a full tick's worth arrived. False on timeout -- which is not
        an error: the caller ticks anyway on whatever partial frames came in,
        because a robot steered by a 50 ms old world is better than a robot
        that stopped being steered.
        """
        deadline = time.monotonic() + timeout
        while True:
            self._pump()
            if self._tick_ready(time.time()):
                return True
            left = deadline - time.monotonic()
            if left <= 0.0:
                return False
            select.select([self._sock], [], [], left)

    def frames(self) -> list[DetectionFrame]:
        """The merged detection frames since the last call.

        Every camera's frames, in t_capture order, on our clock. Empty if
        nothing has arrived -- the tracker extrapolates through that.
        """
        self._pump()
        out = sorted(self._pending, key=lambda f: f.t_capture)
        self._pending = []
        self._fresh = set()
        return out

    def close(self) -> None:
        self._sock.close()

