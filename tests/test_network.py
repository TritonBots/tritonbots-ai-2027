"""The match path against the real stack: ER-Force simulator + game controller.

    docker compose up -d simulator game-controller
    # in the GC UI (:8081), name one team exactly "TritonBots"
    TBOTS_NETWORK_TESTS=1 pytest tests/test_network.py tests/test_backend_parity.py

Skipped without TBOTS_NETWORK_TESTS=1, like the parity test. When it is set,
a missing stack FAILS rather than skips: green must mean the wire works.
Addresses are configs/net/dev.yaml's.
"""

from __future__ import annotations

import os
import time

import pytest

from tbots.apps import wiggle
from tbots.backends.network import NetworkBackend
from tbots.core.command import RobotCommand, stop
from tbots.core.geometry import DIV_A, DIV_B, FieldGeometry
from tbots.net.referee import RefereeReceiver
from tbots.net.robot_control import RobotControlSender
from tbots.net.vision import VisionReceiver

pytestmark = pytest.mark.skipif(
    os.environ.get("TBOTS_NETWORK_TESTS") != "1",
    reason="requires a running simulator; set TBOTS_NETWORK_TESTS=1",
)

VISION = ("224.5.23.2", 10020)
REFEREE = ("224.5.23.1", 10003)


def make_backend(geometry: FieldGeometry = DIV_B) -> NetworkBackend:
    """A NetworkBackend on the dev stack, with our colour already resolved."""
    referee = RefereeReceiver(*REFEREE, team_name="TritonBots")
    deadline = time.monotonic() + 3.0
    while not referee.perspective.resolved and time.monotonic() < deadline:
        referee.poll()
        time.sleep(0.05)
    if not referee.perspective.resolved:
        referee.close()
        pytest.fail("the game controller has not named 'TritonBots'; set a team "
                    "name in its UI (:8081) and re-run")
    vision = VisionReceiver(*VISION)
    b = NetworkBackend(team_name="TritonBots", vision=vision, referee=referee,
                       control=RobotControlSender(), geometry=geometry)
    # Fill the tracker before anyone reads it.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and vision.geometry is None:
        vision.wait_for_next_frame(timeout=0.05)
    return b


def test_vision_receives_the_simulator():
    rx = VisionReceiver(*VISION)
    try:
        ticks = 0
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            ticks += rx.wait_for_next_frame(timeout=0.1)
            fs = rx.frames()
        assert 50 <= ticks <= 80, ticks          # ~66 Hz, one camera
        assert fs and fs[-1].robots_blue and fs[-1].balls
        assert abs(rx.clock_offset) < 0.05       # same host, same clock
        g = rx.geometry
        assert g is not None and (g.length, g.width) == (9.0, 6.0)
    finally:
        rx.close()


def test_wiggle_drives_a_circle():
    assert wiggle.main(["--color", "blue", "--robot", "4", "--seconds", "4"]) == 0


def test_control_feedback_from_the_simulator():
    s = RobotControlSender(port=10301)
    try:
        for _ in range(10):
            s.send([RobotCommand(5), RobotCommand(15)])   # 15 is not on the field
            time.sleep(0.02)
        fb = s.feedback()
        assert 5 in fb and 15 not in fb
        assert fb[5].has_ball is False
    finally:
        s.close()


def test_observe_is_our_frame_with_the_game_attached():
    b = make_backend()
    try:
        # Telemetry is the simulator's reply to a command, so it trails the
        # first command by a tick; a few steps settle it.
        for _ in range(5):
            w = b.step([stop(0)])
        p = b.perspective
        assert p.resolved
        assert len(w.us) >= 6 and len(w.them) >= 6
        assert all(r.visible for r in w.us.values())
        assert w.ball.visible
        assert w.t_capture is not None and 0.0 < w.t - w.t_capture < 0.1
        assert 0 in w.telemetry
        # Rule 3: the colour the referee gave us is the colour on `us`.
        rx = VisionReceiver(*VISION)
        try:
            rx.wait_for_next_frame(timeout=0.5)
            f = rx.frames()[-1]
            raw = f.robots_yellow if p.we_are_yellow else f.robots_blue
            r0 = next(r for r in raw if r.robot_id == 0)
            x, y = p.point(r0.x, r0.y)
            assert (w.us[0].x, w.us[0].y) == pytest.approx((x, y), abs=0.05)
        finally:
            rx.close()
    finally:
        b.close()


def test_division_mismatch_fails_loudly():
    b = make_backend(geometry=DIV_A)
    try:
        with pytest.raises(RuntimeError, match="wrong Division"):
            b.observe()
    finally:
        b.close()
