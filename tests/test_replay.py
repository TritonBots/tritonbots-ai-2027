"""TASK-082: apps/replay.py plays a rollout back, pausable and seekable,
through the real VisionPublisher."""

from __future__ import annotations

import time

import numpy as np
import pytest

from tbots.apps.replay import Player, handle_key, play
from tbots.backends.base import Scenario
from tbots.backends.rsim import RSimBackend
from tbots.core.command import RobotCommand
from tbots.core.transition import Transition
from tbots.net.multicast import rx_socket
from tbots.rl.rollout import RolloutRecorder, load_rollout, save_rollout

DT = 1 / 60


def _rollout(ticks: int = 120):
    b = RSimBackend(n_us=1, n_them=1)
    w = b.reset(Scenario(ball=(0.5, 0.0, 0.0, 0.0), us=((-1.0, 0.0, 0.0),),
                         them=((1.0, 0.0, 3.14),)))
    rec = RolloutRecorder(["x"], meta={"dt": DT})
    rec.start(w)
    for _ in range(ticks):
        prev, w = w, b.step([RobotCommand(0, vx=1.0)])
        rec.on_tick(Transition(w, prev, (RobotCommand(0, vx=1.0),), 0, b.geometry, DT),
                    {"x": 1.0})
    b.close()
    return rec.finish()


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_frames_follow_wall_clock_at_realtime_and_at_double_speed():
    r = _rollout()
    clock, shown = FakeClock(), []
    p = Player(r, lambda w: shown.append(w.t), clock=clock)
    p.update()
    assert shown == [0.0]
    clock.now += 0.5
    p.update()
    assert p.index == 30 and shown[-1] == pytest.approx(r.t[30])
    p.set_speed(2.0)
    clock.now += 0.25
    p.update()
    assert p.index == 60


def test_pause_holds_seek_jumps_and_frame_step_pauses():
    r = _rollout()
    clock = FakeClock()
    p = Player(r, lambda w: None, clock=clock)
    clock.now += 1.0
    p.update()
    assert p.index == 60
    handle_key(p, " ")
    clock.now += 5.0
    p.update()
    assert p.index == 60 and p.paused
    handle_key(p, "left")
    assert p.index == 0
    handle_key(p, "right")
    handle_key(p, "right")
    assert p.index == 120                       # clamped to the last frame
    handle_key(p, ",")
    assert p.index == 119 and p.paused
    handle_key(p, " ")                          # resume from 119
    clock.now += 1 / 60 + 1e-6
    p.update()
    assert p.index == 120 and p.at_end
    assert handle_key(p, "q") is False


def test_update_reports_time_until_the_next_frame():
    p = Player(_rollout(), lambda w: None, clock=FakeClock())
    assert p.update() == pytest.approx(DT)


def test_rollout_round_trips_through_a_file(tmp_path):
    r = _rollout(30)
    back = load_rollout(save_rollout(tmp_path / "a.npz", r))
    assert len(back) == 31 and back.done == "" and back.meta["dt"] == DT
    w0, w1 = r.world(17), back.world(17)
    assert w0.us[0].x == pytest.approx(w1.us[0].x) and w0.ball.y == pytest.approx(w1.ball.y)
    assert back.commands[0, 0, 0] == -1 and back.commands[5, 0, 1] == pytest.approx(1.0)


def test_play_streams_every_frame_to_the_wire_in_realtime():
    """End to end: the real publisher, a real multicast socket, wall time."""
    port = 10096
    rx = rx_socket("224.5.23.2", port)
    rx.setblocking(False)
    r = _rollout(60)              # 1 s of play
    t0 = time.monotonic()
    play(r, port=port, interactive=False)
    elapsed = time.monotonic() - t0
    frames = 0
    from tbots._pb.messages_robocup_ssl_wrapper_pb2 import SSL_WrapperPacket
    xs = []
    while True:
        try:
            data = rx.recv(65535)
        except (TimeoutError, BlockingIOError):
            break
        pkt = SSL_WrapperPacket()
        pkt.ParseFromString(data)
        if pkt.HasField("detection"):
            frames += 1
            xs.append(pkt.detection.robots_blue[0].x)
    rx.close()
    assert frames == len(r)
    assert 0.95 <= elapsed < 1.6
    # The robot drove +x for a second: the wire shows it, in millimetres.
    assert np.all(np.diff(xs) >= -1.0) and xs[-1] - xs[0] > 300
