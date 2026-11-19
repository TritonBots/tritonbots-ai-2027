"""TASK-058: apps/eval.py and rollout files.

Gate: `eval --checkpoint runs/x/latest.pt` writes runs/x/eval.json and
runs/x/rollouts/*.npz; a 6v6 two-minute rollout is under 5 MB.
"""

from __future__ import annotations

import json

import pytest

from tbots.apps.eval import main as eval_main
from tbots.backends.base import Scenario
from tbots.backends.rsim import RSimBackend
from tbots.core.command import RobotCommand
from tbots.core.transition import Transition
from tbots.rl.factory import compose_config
from tbots.rl.rollout import RolloutRecorder, load_rollout, prune, save_rollout
from tbots.rl.train import train


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    run = tmp_path_factory.mktemp("evalrun") / "x"
    train(compose_config(["train.total_steps=512", "train.num_envs=2", "train.num_steps=256",
                          "train.eval_episodes=0", "train.update_epochs=1",
                          f"train.run_dir={run}"]))
    return run


@pytest.mark.timeout(300)
def test_eval_writes_eval_json_and_rollouts(run_dir, capsys):
    eval_main(["--checkpoint", str(run_dir / "latest.pt"), "--episodes", "3"])
    report = json.loads((run_dir / "eval.json").read_text())
    assert report["perception_mode"] == "tracked"
    assert len(report["episodes"]) == 3
    assert report["seeds"] == [1_000_000, 1_000_001, 1_000_002]
    for e in report["episodes"]:
        assert {"ret", "done", "goal_difference", "terms", "rollout"} <= set(e)
        assert set(e["terms"]) == {"ball_progress", "outcome", "time_cost"}
        assert sum(e["terms"].values()) == pytest.approx(e["ret"], abs=1e-4)
    assert set(report["reward_terms"]["ball_progress"]) == {"mean", "std", "min", "max"}
    files = sorted((run_dir / "rollouts").glob("*.npz"))
    assert len(files) == 3
    r = load_rollout(files[0])
    assert r.done == report["episodes"][0]["done"]
    assert "mean return" in capsys.readouterr().out


@pytest.mark.timeout(300)
def test_eval_is_deterministic_on_its_fixed_seed_set(run_dir):
    eval_main(["--checkpoint", str(run_dir / "latest.pt"), "--episodes", "2"])
    a = json.loads((run_dir / "eval.json").read_text())
    eval_main(["--checkpoint", str(run_dir / "latest.pt"), "--episodes", "2"])
    b = json.loads((run_dir / "eval.json").read_text())
    assert [e["ret"] for e in a["episodes"]] == [e["ret"] for e in b["episodes"]]


def test_only_the_newest_rollouts_are_kept(tmp_path):
    import os
    for i in range(13):
        p = tmp_path / f"r{i:02d}.npz"
        p.write_bytes(b"x")
        os.utime(p, ns=(i * 10**9, i * 10**9))
    gone = prune(tmp_path, 10)
    assert sorted(p.name for p in gone) == ["r00.npz", "r01.npz", "r02.npz"]
    assert len(list(tmp_path.glob("*.npz"))) == 10


@pytest.mark.timeout(600)
def test_a_6v6_two_minute_rollout_is_under_5_mb(tmp_path):
    """7200 ticks of 6v6 rSim, every robot moving, recorded the way eval
    records (RolloutRecorder fed Transitions), saved, measured."""
    b = RSimBackend(n_us=6, n_them=6)
    w = b.reset(Scenario.kickoff())
    rec = RolloutRecorder(["a", "b", "c", "d"], meta={"dt": b.dt})
    rec.start(w)
    for k in range(7200):
        cmds = tuple(RobotCommand(i, vx=0.8 * ((k // 120 + i) % 2 * 2 - 1), vtheta=0.5)
                     for i in range(6))
        opp = [RobotCommand(i, vy=0.5 * ((k // 90 + i) % 2 * 2 - 1)) for i in range(6)]
        prev, w = w, b.step(cmds, opp)
        rec.on_tick(Transition(w, prev, cmds, None, b.geometry, b.dt),
                    {"a": 0.1, "b": -0.01 * k, "c": 0.0, "d": float(k % 7)})
    b.close()
    path = save_rollout(tmp_path / "match.npz", rec.finish())
    size = path.stat().st_size
    print(f"6v6 120 s rollout: {size / 1e6:.2f} MB")
    assert size < 5_000_000
    back = load_rollout(path)
    assert len(back) == 7201 and back.us.shape == (7201, 6, 8)
