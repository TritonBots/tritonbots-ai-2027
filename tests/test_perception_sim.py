"""TASK-075: PerceptionSim on real rSim trajectories.

The gate: with zero noise the output equals the input shifted by the
sampled delay; with dropout, some robots are `visible=False`; seeded runs
are identical. Plus the `tracked` mode, which must go through the real
Tracker and come out in our frame.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tbots.backends.base import Scenario
from tbots.backends.rsim import RSimBackend
from tbots.core.command import RobotCommand
from tbots.core.state import DetectionFrame, WorldState
from tbots.perception.sim import Perception, PerceptionConfig, PerceptionSim

DT = 1.0 / 60.0


def _trajectory(n: int = 90) -> list[WorldState]:
    """A real rSim run: robot 0 drives a curve, the ball rolls, them stand."""
    backend = RSimBackend(n_us=2, n_them=2)
    try:
        world = backend.reset(Scenario(ball=(0.5, 0.2, 1.0, -0.5, ),
                                       us=((-1.0, 0.0, 0.0), (-2.0, 1.0, 0.0)),
                                       them=((1.0, 1.0, math.pi), (2.0, -1.0, math.pi))))
        out = [world]
        for _ in range(n):
            out.append(backend.step([RobotCommand(0, vx=1.0, vtheta=1.0)]))
        return out
    finally:
        backend.close()


TRAJ = _trajectory()


def _zero(**kw) -> PerceptionConfig:
    base: dict = {"latency_s": (0.020, 0.045), "robot_pos_std_m": 0.0,
                  "robot_angle_std_rad": 0.0, "ball_pos_std_m": 0.0, "dropout": 0.0}
    base.update(kw)
    return PerceptionConfig(**base)


def test_zero_noise_output_is_the_input_shifted_by_the_sampled_delay():
    sim = PerceptionSim(_zero(), dt=DT)
    sim.reset(TRAJ[0], np.random.default_rng(3))
    d = sim.params.delay_steps
    # 20..45 ms at 60 Hz is 1..3 ticks; a zero delay would make this vacuous.
    assert 1 <= d <= 3
    for k, truth in enumerate(TRAJ[1:], start=1):
        assert sim.noisy(truth) == TRAJ[max(0, k - d)]


def test_delay_is_sampled_per_episode_from_the_configured_range():
    sim = PerceptionSim(_zero(latency_s=(0.0, 0.1)), dt=DT)
    rng = np.random.default_rng(0)
    delays = set()
    for _ in range(40):
        sim.reset(TRAJ[0], rng)
        delays.add(sim.params.delay_steps)
    assert min(delays) >= 0 and max(delays) <= 6
    assert len(delays) > 3


def test_dropout_marks_robots_invisible_and_holds_their_last_perceived_state():
    sim = PerceptionSim(_zero(dropout=0.3), dt=DT)
    sim.reset(TRAJ[0], np.random.default_rng(1))
    dropped = 0
    prev = None
    for truth in TRAJ[1:]:
        w = sim.noisy(truth)
        for side in ("us", "them"):
            for rid, r in getattr(w, side).items():
                if not r.visible:
                    dropped += 1
                    # Held, not invented: same pose as the last frame we had.
                    assert prev is not None
                    p = getattr(prev, side)[rid]
                    assert (r.x, r.y, r.theta) == (p.x, p.y, p.theta)
        prev = w
    assert dropped > 0


def test_noise_moves_positions_by_roughly_the_sampled_std():
    cfg = _zero(latency_s=(0.0, 0.0), robot_pos_std_m=0.01)
    sim = PerceptionSim(cfg, dt=DT)
    sim.reset(TRAJ[0], np.random.default_rng(7))
    errs = []
    for truth in TRAJ[1:]:
        w = sim.noisy(truth)
        errs += [w.us[i].x - truth.us[i].x for i in truth.us]
    assert 0.005 < float(np.std(errs)) < 0.02


@pytest.mark.parametrize("mode", ["noisy", "tracked"])
def test_seeded_runs_are_identical(mode):
    cfg = PerceptionConfig(mode=mode, dropout=(0.0, 0.1))

    def run(seed: int) -> list[WorldState]:
        p = Perception(cfg, dt=DT)
        out = [p.reset(TRAJ[0], np.random.default_rng(seed))]
        out += [p.observe(t) for t in TRAJ[1:]]
        return out

    assert run(11) == run(11)
    assert run(11) != run(12)


def test_tracked_mode_emits_detection_frames_and_synthetic_feedback():
    sim = PerceptionSim(_zero(), dt=DT)
    sim.reset(TRAJ[0], np.random.default_rng(0))
    for truth in TRAJ[1:10]:
        sim.tracked(truth)
    frames, feedback = sim.tracked(TRAJ[10])
    assert all(isinstance(f, DetectionFrame) for f in frames)
    assert len(frames) == 1
    f = frames[0]
    # rSim is IDENTITY: we are blue. Frames carry colours, never us/them.
    assert {r.robot_id for r in f.robots_blue} == set(TRAJ[10].us)
    assert {r.robot_id for r in f.robots_yellow} == set(TRAJ[10].them)
    assert f.t_capture == pytest.approx(TRAJ[10 - sim.params.delay_steps].t)
    # Feedback is ours only, and comes off the robot, not off the camera.
    assert set(feedback) == set(TRAJ[10].us)
    assert all(fb.t == TRAJ[10].t for fb in feedback.values())


def test_a_frame_is_never_emitted_twice():
    sim = PerceptionSim(_zero(latency_s=(0.05, 0.05)), dt=DT)
    sim.reset(TRAJ[0], np.random.default_rng(0))
    seen = []
    for truth in TRAJ[1:10]:
        seen += [f.t_capture for f in sim.tracked(truth)[0]]
    assert len(seen) == len(set(seen))


def test_tracked_mode_runs_the_real_tracker_and_lands_near_truth():
    p = Perception(_zero(mode="tracked", latency_s=(0.03, 0.03),
                         robot_pos_std_m=0.003, ball_pos_std_m=0.003), dt=DT)
    p.reset(TRAJ[0], np.random.default_rng(0))
    errs = []
    for truth in TRAJ[1:]:
        w = p.observe(truth)
        assert w.t == pytest.approx(truth.t)
        assert w.t_capture is not None and w.t_capture < truth.t
        r, t = w.us[0], truth.us[0]
        errs.append(math.hypot(r.x - t.x, r.y - t.y))
    # The tracker predicts across the 30 ms delay; settled, it is centimetre-good.
    assert max(errs[30:]) < 0.05


def test_truth_mode_is_the_identity():
    p = Perception(PerceptionConfig(mode="truth"), dt=DT)
    assert p.reset(TRAJ[0], np.random.default_rng(0)) is TRAJ[0]
    assert all(p.observe(t) is t for t in TRAJ[1:])


def test_config_accepts_scalars_and_ranges_and_rejects_bad_modes():
    cfg = PerceptionConfig.from_mapping({"mode": "noisy", "latency_s": 0.03,
                                         "dropout": [0.0, 0.1]})
    assert cfg.latency_s == (0.03, 0.03)
    assert cfg.dropout == (0.0, 0.1)
    with pytest.raises(ValueError, match="mode"):
        PerceptionConfig(mode="perfect")
    with pytest.raises(ValueError, match="dropout"):
        PerceptionConfig(dropout=(0.0, 1.5))
    with pytest.raises(KeyError, match="unknown perception key"):
        PerceptionConfig.from_mapping({"vision_latency_ms": [20, 45]})
