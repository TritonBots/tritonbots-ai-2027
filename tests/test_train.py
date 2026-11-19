"""TASK-056: the trainer, end to end on real rSim.

The gate's learning half (go_to_ball reaches a positive mean return within
the documented budget) is a multi-minute run, recorded in SETUP_LOG; this
file checks the machinery on a short run: the run directory's contents, the
truncation bootstrap actually firing, eval rollouts, and resume.
"""

from __future__ import annotations

import pytest

from tbots.rl.artifacts import load_checkpoint
from tbots.rl.factory import compose_config
from tbots.rl.train import train

SHORT = ["train.total_steps=1024", "train.num_envs=2", "train.num_steps=256",
         "train.checkpoint_interval_steps=512", "train.eval_episodes=1",
         "train.update_epochs=2"]


@pytest.mark.timeout(300)
def test_a_short_run_writes_the_whole_run_directory(tmp_path):
    run = tmp_path / "run"
    s = train(compose_config([*SHORT, f"train.run_dir={run}"]))
    assert (run / "latest.pt").is_file()
    assert (run / "config.yaml").is_file()
    assert list(run.glob("events.out.tfevents.*"))
    assert len(list((run / "rollouts").glob("*.npz"))) == 2      # two checkpoints x 1
    assert s["global_step"] == 1024
    # go_to_ball's 8 s limit is 480 ticks; 512 ticks per worker under a
    # random policy truncates, and every truncation was bootstrapped.
    assert s["done_reasons"].get("TIMEOUT", 0) >= 1
    assert s["truncations"] == s["done_reasons"]["TIMEOUT"]
    state = load_checkpoint(run / "latest.pt")
    assert state["global_step"] == 1024
    assert state["metadata"]["obs_builder"] == "ego_v1"
    assert state["metadata"]["perception_mode"] == "noisy"


@pytest.mark.timeout(300)
def test_resume_continues_its_own_run_and_restarts_the_count_for_a_new_one(tmp_path):
    a = tmp_path / "a"
    train(compose_config([*SHORT, f"train.run_dir={a}", "train.eval_episodes=0"]))

    # Walltime resume: same run directory, the step count continues.
    s = train(compose_config([*SHORT, f"train.run_dir={a}", "train.eval_episodes=0",
                              "train.total_steps=1536", f"train.resume_from={a}/latest.pt"]))
    assert s["global_step"] == 1536

    # Next stage: a new run directory from a's checkpoint starts at zero
    # with a's weights, and a changed observation would be refused.
    b = tmp_path / "b"
    resumed = train(compose_config([*SHORT, f"train.run_dir={b}", "train.eval_episodes=0",
                                    "train.total_steps=512", "env.n_them=1",
                                    f"train.resume_from={a}/latest.pt"]))
    assert resumed["global_step"] == 512
    with pytest.raises(ValueError, match="dt"):
        train(compose_config([*SHORT, f"train.run_dir={tmp_path / 'c'}",
                              "env.control_hz=50", f"train.resume_from={a}/latest.pt"]))
