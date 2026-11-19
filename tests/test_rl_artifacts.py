from __future__ import annotations

import warnings

import pytest
from omegaconf import OmegaConf

from tbots.rl.artifacts import (
    check_compatible,
    checkpoint_metadata,
    load_checkpoint,
    save_checkpoint,
    write_resolved_config,
)


def test_resolved_config_records_overrides(tmp_path):
    config = OmegaConf.create({"train": {"run_dir": "runs/example", "num_envs": 8}})

    path = write_resolved_config(tmp_path, config)

    assert OmegaConf.load(path).train.num_envs == 8


# -- TASK-059 ----------------------------------------------------------------

CODEC = {"name": "skill_v1", "max_speed": 2.0, "max_angular_speed": 6.0, "kick_speed": 3.0,
         "chip": False, "dribbler_speed": 1.0, "max_linear_accel": 3.0,
         "max_angular_accel": 20.0, "action_repeat": 1, "dt": 1 / 60}


def meta(**over):
    base = {"obs_builder": "ego_v1", "obs_size": 33, "codec": dict(CODEC), "dt": 1 / 60,
            "perception_mode": "tracked", "stage": "solo", "task": "go_to_ball"}
    codec_over = over.pop("codec", {})
    base.update(over)
    m = checkpoint_metadata(**base)
    m["codec"].update(codec_over)
    return m


def test_metadata_records_everything_the_architecture_lists():
    m = meta()
    for key in ("obs_builder", "obs_size", "codec", "dt", "action_repeat", "accel_limits",
                "perception_mode", "stage", "git_sha", "task"):
        assert key in m, key
    assert m["action_repeat"] == 1
    assert m["accel_limits"] == {"linear": 3.0, "angular": 20.0}
    # This checkout is a git repository, so the SHA is a real one.
    assert isinstance(m["git_sha"], str) and len(m["git_sha"]) == 40


def test_checkpoint_round_trip(tmp_path):
    path = save_checkpoint(tmp_path, {"metadata": meta(), "step": 42})
    assert load_checkpoint(path, expected_metadata=meta())["step"] == 42


@pytest.mark.parametrize(("what", "other"), [
    ("obs_builder", meta(obs_builder="ego_v2")),
    ("obs_size", meta(obs_size=34)),
    ("codec", meta(codec={"name": "skill_v2"})),
    ("codec", meta(codec={"max_speed": 1.05})),
    ("dt", meta(dt=1 / 50)),
    ("action_repeat", meta(codec={"action_repeat": 2})),
    ("accel_limits", meta(codec={"max_linear_accel": 4.0})),
])
def test_each_incompatibility_refuses(tmp_path, what, other):
    path = save_checkpoint(tmp_path, {"metadata": meta()})
    with pytest.raises(ValueError, match=what):
        load_checkpoint(path, expected_metadata=other)


def test_a_perception_mode_mismatch_warns_and_loads(tmp_path):
    path = save_checkpoint(tmp_path, {"metadata": meta(), "step": 7})
    with pytest.warns(UserWarning, match="perception"):
        state = load_checkpoint(path, expected_metadata=meta(perception_mode="noisy"))
    assert state["step"] == 7


def test_the_sha_stage_task_and_kick_power_are_never_checked():
    saved, now = meta(), meta(stage="full", task="other", codec={"kick_speed": 5.0})
    now["git_sha"] = "0" * 40
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_compatible(saved, now)


def test_a_checkpoint_without_metadata_refuses_when_metadata_is_expected(tmp_path):
    path = save_checkpoint(tmp_path, {"step": 1})
    with pytest.raises(ValueError, match="no metadata"):
        load_checkpoint(path, expected_metadata=meta())
