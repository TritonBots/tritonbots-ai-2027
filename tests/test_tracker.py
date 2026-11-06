"""TASK-020: the tracker against a real rSim trajectory.

The gate: frames synthesised from a known rSim trajectory, delivered with
30 ms latency; the tracked positions at `t` are within 2 cm of the truth at
`t`. Nothing here is stubbed -- the trajectory is rSim's, the noise is
Gaussian at SSL-Vision's scale, and the frames are split across two
overlapping cameras so that merging is exercised.

The trajectory is chosen to be CONTACT-FREE, and the test asserts that it
is. A robot that hits a wall at 1.8 m/s stops in one frame (~100 m/s^2);
no motion model predicts that 30 ms ahead, so a contact would measure the
wall, not the tracker. Contacts are covered by the manoeuvre reset, not by
this gate.
"""

from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pytest

from tbots.backends.base import Scenario
from tbots.backends.rsim import RSimBackend
from tbots.core.command import RobotCommand
from tbots.core.geometry import DIV_B
from tbots.core.perspective import IDENTITY, UNRESOLVED, Perspective
from tbots.core.state import (
    DetectionBall,
    DetectionFrame,
    DetectionRobot,
    WorldState,
)
from tbots.perception.tracker import Tracker

LATENCY = 0.030
NOISE_POS = 0.002
NOISE_THETA = 0.01

# Each robot follows its own Lissajous figure. Centres are far enough apart,
# and from the walls, that no two paths can touch.
CENTRES = {("us", 0): (-3.0, 1.4), ("us", 1): (-3.0, -1.3), ("us", 2): (-1.1, 0.2),
           ("them", 0): (3.0, 1.4), ("them", 1): (3.0, -1.3), ("them", 2): (1.1, 1.0)}


def _ref(key: tuple[str, int], t: float) -> tuple[float, float, float, float]:
    cx, cy = CENTRES[key]
    ph = key[1] * 1.3 + (0.0 if key[0] == "us" else 0.7)
    a, b = 0.6, 0.4
    return (cx + a * math.sin(1.6 * t + ph), cy + b * math.sin(2.1 * t + 2 * ph),
            a * 1.6 * math.cos(1.6 * t + ph), b * 2.1 * math.cos(2.1 * t + 2 * ph))


def _follow(world: WorldState, side: str, t: float, dt: float) -> list[RobotCommand]:
    out = []
    for i, r in getattr(world, side).items():
        x, y, vx, vy = _ref((side, i), t + dt)
        gx, gy = vx + 4.0 * (x - r.x), vy + 4.0 * (y - r.y)
        c, s = math.cos(r.theta), math.sin(r.theta)
        out.append(RobotCommand(i, vx=c * gx + s * gy, vy=-s * gx + c * gy,
                                vtheta=2.0 * math.sin(1.2 * t + i)))
    return out


@pytest.fixture(scope="module")
def trajectory() -> list[WorldState]:
    b = RSimBackend(n_us=3, n_them=3)
    try:
        w = b.reset(Scenario(
            ball=(-2.0, -2.4, 3.0, 0.05),
            us=tuple((*_ref(("us", i), 0.0)[:2], 0.0) for i in range(3)),
            them=tuple((*_ref(("them", i), 0.0)[:2], 3.1) for i in range(3))))
        truth = [w]
        for k in range(480):
            t = k * b.dt
            w = b.step(_follow(w, "us", t, b.dt), _follow(w, "them", t, b.dt))
            truth.append(w)
        return truth
    finally:
        b.close()


def _frames(x: WorldState, rng: random.Random) -> list[DetectionFrame]:
    """Two cameras, overlapping by a metre around x = 0. Field frame, us blue."""
    def n() -> float:
        return rng.gauss(0.0, NOISE_POS)

    def seen(rs: dict, lo: float, hi: float) -> tuple[DetectionRobot, ...]:
        return tuple(DetectionRobot(r.robot_id, r.x + n(), r.y + n(),
                                    r.theta + rng.gauss(0.0, NOISE_THETA))
                     for r in rs.values() if lo <= r.x <= hi)

    out = []
    for cam, (lo, hi) in enumerate(((-9.0, 0.5), (-0.5, 9.0))):
        balls = ((DetectionBall(x.ball.x + n(), x.ball.y + n()),)
                 if lo <= x.ball.x <= hi else ())
        out.append(DetectionFrame(x.t, cam, balls, seen(x.us, lo, hi), seen(x.them, lo, hi)))
    return out


def _interp(truth: list[WorldState], ts: np.ndarray, side: str, rid: int,
            t: float) -> np.ndarray:
    i = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
    a = (t - ts[i - 1]) / (ts[i] - ts[i - 1])

    def pos(w: WorldState) -> np.ndarray:
        return np.array(w.ball.pos if side == "ball" else getattr(w, side)[rid].pos)

    return (1 - a) * pos(truth[i - 1]) + a * pos(truth[i])


def test_trajectory_is_contact_free(trajectory):
    for w in trajectory:
        rs = list(w.us.values()) + list(w.them.values())
        for p, q in itertools.combinations(rs, 2):
            assert math.dist(p.pos, q.pos) > 0.5
        for r in rs:
            assert math.dist(r.pos, w.ball.pos) > 0.5
            assert DIV_B.inside_field(r.x, r.y, margin=0.5)


def test_gate_tracked_within_2cm_of_truth_at_30ms_latency(trajectory):
    ts = np.array([w.t for w in trajectory])
    rng = random.Random(0)
    tracker = Tracker(DIV_B)
    worst = 0.0
    checked = 0
    for x in trajectory:
        t_now = x.t + LATENCY
        if t_now > ts[-1]:
            break
        est = tracker.update(_frames(x, rng), t_now, IDENTITY)
        assert est.t == pytest.approx(t_now)
        assert est.t_capture == pytest.approx(x.t)
        if x.t < 0.5:            # filters converging from a cold start
            continue
        assert set(est.us) == {0, 1, 2} and set(est.them) == {0, 1, 2}
        for side in ("us", "them"):
            for rid, r in getattr(est, side).items():
                truth = _interp(trajectory, ts, side, rid, est.t)
                err = float(np.hypot(*(np.array(r.pos) - truth)))
                worst = max(worst, err)
                checked += 1
        err = float(np.hypot(*(np.array(est.ball.pos) - _interp(trajectory, ts, "ball", 0, est.t))))
        worst = max(worst, err)
    assert checked > 2000
    assert worst < 0.02, f"worst tracked error {worst * 1000:.1f} mm"


def test_dropout_extrapolates_and_marks_invisible():
    tracker = Tracker(DIV_B)
    p = IDENTITY
    for k in range(60):                      # 1 m/s along +x for a second
        t = k / 60
        f = DetectionFrame(t, 0, (), (DetectionRobot(0, t, 0.0, 0.0),), ())
        w = tracker.update([f], t, p)
    assert w.us[0].visible
    assert w.us[0].vx == pytest.approx(1.0, abs=0.05)

    # The robot drops out; frames keep arriving without it.
    t_last = 59 / 60
    for k in range(60, 66):
        t = k / 60
        w = tracker.update([DetectionFrame(t, 0)], t, p)
    assert not w.us[0].visible
    assert w.us[0].x == pytest.approx(65 / 60, abs=0.01)

    # Extrapolation is capped, then the track is dropped.
    t = t_last + 0.5
    w = tracker.update([DetectionFrame(t, 0)], t, p)
    assert w.us[0].x == pytest.approx(t_last + 0.25, abs=0.02)
    t = t_last + 1.1
    w = tracker.update([DetectionFrame(t, 0)], t, p)
    assert 0 not in w.us


def test_vision_loss_is_not_visible():
    """No frames at all: the newest frame ages, so nothing looks fresh."""
    tracker = Tracker(DIV_B)
    f = DetectionFrame(0.0, 0, (DetectionBall(1.0, 1.0),), (DetectionRobot(0, 0.0, 0.0, 0.0),))
    w = tracker.update([f], 0.0, IDENTITY)
    assert w.us[0].visible and w.ball.visible
    w = tracker.update([], 0.3, IDENTITY)
    assert not w.us[0].visible and not w.ball.visible


def test_perspective_is_applied_once_at_the_end():
    """Yellow, defending +x: colours swap into us/them and the frame rotates."""
    tracker = Tracker(DIV_B)
    p = Perspective(we_are_yellow=True, flip=True)
    for k in range(30):
        t = k / 60
        f = DetectionFrame(t, 0, (DetectionBall(1.0, 0.5),),
                           robots_blue=(DetectionRobot(3, 2.0, 1.0, 0.0),),
                           robots_yellow=(DetectionRobot(1, -2.0 + t, 0.5, math.pi / 2),))
        w = tracker.update([f], t, p)
    assert set(w.us) == {1} and set(w.them) == {3}
    us = w.us[1]
    assert us.pos == pytest.approx((2.0 - 29 / 60, -0.5), abs=0.01)
    assert us.vx == pytest.approx(-1.0, abs=0.05)      # moving +x in the field
    assert us.theta == pytest.approx(-math.pi / 2, abs=0.02)
    assert w.them[3].pos == pytest.approx((-2.0, -1.0), abs=0.01)
    assert w.ball.pos == pytest.approx((-1.0, -0.5), abs=0.01)


def test_unresolved_perspective_raises():
    with pytest.raises(ValueError):
        Tracker(DIV_B).update([], 0.0, UNRESOLVED)


def test_late_camera_cannot_rewind_a_track():
    tracker = Tracker(DIV_B)
    for k in range(30):
        t = k / 60
        tracker.update([DetectionFrame(t, 0, (), (DetectionRobot(0, t, 0.0, 0.0),))], t, IDENTITY)
    # A second camera reports an older capture of the same robot, far behind.
    stale = DetectionFrame(0.1, 1, (), (DetectionRobot(0, 0.1, 0.0, 0.0),))
    w = tracker.update([stale], 29 / 60, IDENTITY)
    assert w.us[0].x == pytest.approx(29 / 60, abs=0.01)


def test_kick_is_reacquired_within_three_frames():
    """Ball at rest, then 5 m/s. The manoeuvre reset re-learns velocity."""
    rng = random.Random(1)
    tracker = Tracker(DIV_B)

    def bx(t: float) -> float:
        return 0.0 if t < 1.0 else 5.0 * (t - 1.0)

    for k in range(120):
        t = k / 60
        f = DetectionFrame(t, 0, (DetectionBall(bx(t) + rng.gauss(0, 0.002),
                                                rng.gauss(0, 0.002)),))
        w = tracker.update([f], t + LATENCY, IDENTITY)
        if t >= 1.0 + 3 / 60:
            assert abs(w.ball.x - bx(w.t)) < 0.02, (t, w.ball.x, bx(w.t))
