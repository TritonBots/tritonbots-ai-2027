"""TASK-010: VisionReceiver against real multicast sockets.

TTL 0 and a port nothing in the league uses, so nothing leaves this host.
The packets are real SSL_WrapperPackets, serialised and sent over UDP; the
receiver under test is the one the match path uses. The live check against
the ER-Force simulator is in test_network.py.
"""

from __future__ import annotations

import time

import pytest

from tbots._pb.messages_robocup_ssl_wrapper_pb2 import SSL_WrapperPacket
from tbots.net.multicast import tx_socket
from tbots.net.vision import VisionReceiver

GROUP = "224.5.23.2"
PORT = 10597  # nothing in the stack or the league uses this


def _detection(cam: int, frame: int, t_capture: float, t_sent: float,
               blue: tuple = (), yellow: tuple = (), ball: tuple | None = None) -> bytes:
    pkt = SSL_WrapperPacket()
    d = pkt.detection
    d.frame_number = frame
    d.t_capture = t_capture
    d.t_sent = t_sent
    d.camera_id = cam
    for group, robots in ((d.robots_blue, blue), (d.robots_yellow, yellow)):
        for rid, x, y, th in robots:
            r = group.add()
            r.confidence, r.robot_id, r.x, r.y, r.orientation = 0.9, rid, x, y, th
            r.pixel_x = r.pixel_y = 0.0
    if ball is not None:
        b = d.balls.add()
        b.confidence, b.x, b.y, b.z = 0.8, *ball
        b.pixel_x = b.pixel_y = 0.0
    return pkt.SerializeToString()


@pytest.fixture
def link():
    rx = VisionReceiver(GROUP, PORT)
    tx = tx_socket(ttl=0)
    yield rx, (lambda data: tx.sendto(data, (GROUP, PORT)))
    rx.close()
    tx.close()


def test_decodes_to_meters_and_keeps_wire_colours(link):
    rx, send = link
    now = time.time()
    send(_detection(0, 1, now - 0.02, now - 0.01,
                    blue=((3, 1500.0, -250.0, 0.5),), yellow=((7, -4000.0, 2000.0, -3.0),),
                    ball=(100.0, 200.0, 50.0)))
    assert rx.wait_for_next_frame(timeout=1.0)
    (f,) = rx.frames()
    assert f.camera_id == 0
    (b,) = f.robots_blue
    assert (b.robot_id, b.x, b.y, b.theta) == (3, 1.5, -0.25, 0.5)
    assert b.confidence == pytest.approx(0.9)
    (y,) = f.robots_yellow
    assert (y.robot_id, y.x, y.y) == (7, -4.0, 2.0)
    assert f.balls[0].x == pytest.approx(0.1) and f.balls[0].z == pytest.approx(0.05)


def test_one_tick_per_camera_cycle_not_per_packet(link):
    """Two staggered cameras: the tick waits for both, then returns both."""
    rx, send = link
    now = time.time()
    send(_detection(0, 10, now, now))
    send(_detection(1, 20, now + 0.008, now + 0.008))
    assert rx.wait_for_next_frame(timeout=1.0)
    assert [f.camera_id for f in rx.frames()] == [0, 1]

    # Only camera 0 reports: not a full tick, so wait times out...
    send(_detection(0, 11, now + 0.016, now + 0.016))
    assert not rx.wait_for_next_frame(timeout=0.1)
    # ...until camera 1 catches up. Frames come back in capture order even
    # though camera 1's older frame arrives last.
    send(_detection(1, 21, now + 0.012, now + 0.024))
    assert rx.wait_for_next_frame(timeout=1.0)
    fs = rx.frames()
    assert [f.camera_id for f in fs] == [1, 0]
    assert fs[0].t_capture < fs[1].t_capture


def test_duplicate_frames_are_dropped(link):
    rx, send = link
    now = time.time()
    pkt = _detection(0, 5, now, now, blue=((0, 0.0, 0.0, 0.0),))
    send(pkt)
    send(pkt)
    time.sleep(0.05)
    assert len(rx.frames()) == 1


def test_robot_without_orientation_is_dropped(link):
    rx, send = link
    pkt = SSL_WrapperPacket()
    d = pkt.detection
    d.frame_number, d.t_capture, d.t_sent, d.camera_id = 1, time.time(), time.time(), 0
    r = d.robots_blue.add()
    r.confidence, r.robot_id, r.x, r.y, r.pixel_x, r.pixel_y = 1.0, 0, 0.0, 0.0, 0.0, 0.0
    send(pkt.SerializeToString())
    assert rx.wait_for_next_frame(timeout=1.0)
    assert rx.frames()[0].robots_blue == ()


def test_unsynchronised_vision_clock_is_mapped_onto_ours(link):
    """Vision PC clock 1000 s behind ours: capture times still land on ours."""
    rx, send = link
    skew = -1000.0
    for i in range(5):
        now = time.time()
        send(_detection(0, 100 + i, now + skew - 0.010, now + skew))
        # The match loop reads each packet as it lands (wait_for_next_frame
        # wakes on arrival); that is what the minimum-delay estimate needs.
        assert rx.wait_for_next_frame(timeout=1.0)
        f = rx.frames()[-1]
        assert f.t_capture == pytest.approx(now - 0.010, abs=0.005)
    assert rx.clock_offset == pytest.approx(-skew, abs=0.005)


def test_packets_read_late_do_not_skew_the_clock(link):
    """A tick's worth queued while we computed: the minimum ignores them."""
    rx, send = link
    now = time.time()
    send(_detection(0, 1, now - 0.010, now))
    assert rx.wait_for_next_frame(timeout=1.0)
    rx.frames()
    good = rx.clock_offset
    for i in range(3):
        t = time.time()
        send(_detection(0, 2 + i, t - 0.010, t))
    time.sleep(0.05)                     # read 50 ms late
    rx.frames()
    assert rx.clock_offset == pytest.approx(good, abs=0.002)


def test_geometry_is_decoded_to_meters(link):
    rx, send = link
    assert rx.geometry is None
    pkt = SSL_WrapperPacket()
    f = pkt.geometry.field
    f.field_length, f.field_width, f.goal_width, f.goal_depth, f.boundary_width = (
        9000, 6000, 1000, 180, 300)
    send(pkt.SerializeToString())
    time.sleep(0.05)
    rx.frames()
    g = rx.geometry
    assert g is not None
    assert (g.length, g.width, g.goal_width, g.goal_depth, g.boundary_width) == (
        9.0, 6.0, 1.0, 0.18, 0.3)
