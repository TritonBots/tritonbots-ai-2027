"""Durable run metadata and checkpoints for training jobs.

This module deliberately has no policy or optimizer dependency.  A future
trainer supplies a serialisable mapping; this module makes writing, loading,
and compatibility validation reliable.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

FORMAT_VERSION = 1


def write_resolved_config(run_dir: str | Path, config: DictConfig) -> Path:
    """Atomically persist the fully resolved configuration for a run."""
    destination = Path(run_dir) / "config.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(destination, OmegaConf.to_yaml(config, resolve=True))
    return destination


def save_checkpoint(run_dir: str | Path, state: dict[str, Any]) -> Path:
    """Atomically replace ``latest.pt`` with a versioned checkpoint."""
    destination = Path(run_dir) / "latest.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": FORMAT_VERSION, **state}
    with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        _torch().save(payload, temporary)
        with temporary.open("rb") as saved:
            os.fsync(saved.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_checkpoint(
    path: str | Path, *, expected_metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Load a checkpoint; if `expected_metadata` is given, refuse one that
    this configuration cannot use (see `check_compatible`)."""
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    state = _torch().load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format: {checkpoint}")
    if expected_metadata is not None:
        saved = state.get("metadata")
        if not isinstance(saved, Mapping):
            raise ValueError(f"checkpoint has no metadata to check: {checkpoint}")
        check_compatible(saved, expected_metadata)
    return state


# -- metadata (TASK-059) -------------------------------------------------------
#
# What a checkpoint records, and what loading it checks (ARCHITECTURE section
# 9, "Checkpoints"):
#
#   REFUSE  observation builder name or size, codec, dt. A policy fed a
#           different observation, or whose outputs mean different commands,
#           is not the policy that was trained. Within the codec that means
#           its name and every parameter that changes what an output means:
#           the speed scales, the acceleration limits, action_repeat.
#   WARN    perception mode. Legitimate (fine-tune a `noisy` policy in
#           `tracked`), but worth knowing; anything exported must have been
#           trained in `tracked`.
#   NEVER   the git SHA, the curriculum stage, the task name, and the codec's
#           kick and dribbler speeds -- those are a skill's kwargs, chosen per
#           use (a pass and a shot share a policy and kick at different
#           powers). Immutable builder and codec names are what let an
#           August checkpoint load in April, so the SHA is recorded only.

_CODEC_MEANING = ("name", "max_speed", "max_angular_speed")


def git_sha() -> str | None:
    """HEAD of the checkout this code runs from, or None outside git."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                             capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def checkpoint_metadata(*, obs_builder: str, obs_size: int, codec: Mapping[str, Any],
                        dt: float, perception_mode: str, stage: str | None = None,
                        task: str | None = None) -> dict[str, Any]:
    """Everything a checkpoint records about how its policy was trained."""
    return {
        "obs_builder": obs_builder,
        "obs_size": int(obs_size),
        "codec": dict(codec),
        "dt": float(dt),
        "action_repeat": int(codec["action_repeat"]),
        "accel_limits": {"linear": float(codec["max_linear_accel"]),
                         "angular": float(codec["max_angular_accel"])},
        "perception_mode": perception_mode,
        "stage": stage,
        "task": task,
        "git_sha": git_sha(),
    }


def _codec_view(m: Mapping[str, Any]) -> dict[str, Any]:
    c = m.get("codec") or {}
    return {k: c.get(k) for k in _CODEC_MEANING}


def _accel(m: Mapping[str, Any]) -> Any:
    c = m.get("codec") or {}
    return (c.get("max_linear_accel"), c.get("max_angular_accel"))


def _repeat(m: Mapping[str, Any]) -> Any:
    return (m.get("codec") or {}).get("action_repeat")


def check_compatible(saved: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Raise ValueError if a policy with `saved` metadata cannot run under
    `expected`; warn if it can but trained under different perception."""
    problems = []
    for key in ("obs_builder", "obs_size"):
        if saved.get(key) != expected.get(key):
            problems.append(f"{key}: checkpoint {saved.get(key)!r}, now {expected.get(key)!r}")
    if _codec_view(saved) != _codec_view(expected):
        problems.append(f"codec: checkpoint {_codec_view(saved)}, now {_codec_view(expected)}")
    if _repeat(saved) != _repeat(expected):
        problems.append(f"action_repeat: checkpoint {_repeat(saved)}, now {_repeat(expected)}")
    if _accel(saved) != _accel(expected):
        problems.append(f"accel_limits: checkpoint {_accel(saved)}, now {_accel(expected)}")
    s_dt, e_dt = saved.get("dt"), expected.get("dt")
    if s_dt is None or e_dt is None or abs(float(s_dt) - float(e_dt)) > 1e-12:
        problems.append(f"dt: checkpoint {s_dt!r}, now {e_dt!r}")
    if problems:
        raise ValueError("checkpoint is incompatible with this configuration:\n  "
                         + "\n  ".join(problems))
    if saved.get("perception_mode") != expected.get("perception_mode"):
        warnings.warn(f"checkpoint was trained with perception "
                      f"{saved.get('perception_mode')!r}, running with "
                      f"{expected.get('perception_mode')!r}", UserWarning, stacklevel=2)


def _atomic_text(destination: Path, contents: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as tmp:
        tmp.write(contents)
        tmp.flush()
        os.fsync(tmp.fileno())
        temporary = Path(tmp.name)
    os.replace(temporary, destination)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "checkpoint support needs the optional train dependency (torch)"
        ) from exc
    return torch
