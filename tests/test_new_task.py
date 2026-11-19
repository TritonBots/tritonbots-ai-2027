"""TASK-079: the new_task scaffold. The full gate (scaffold `demo` on a clean
checkout, run its test, train it) is recorded in SETUP_LOG; this checks the
generator without touching the real tree."""

from __future__ import annotations

import ast

import pytest

from tbots.rl.new_task import scaffold


def _tree(root):
    for d in ("src/tbots/rl/tasks", "configs/reward", "tests"):
        (root / d).mkdir(parents=True)
    return root


def test_scaffold_writes_three_valid_files(tmp_path):
    root = _tree(tmp_path)
    written = scaffold("my_skill", root=root)
    assert sorted(p.relative_to(root).as_posix() for p in written) == [
        "configs/reward/my_skill.yaml", "src/tbots/rl/tasks/my_skill.py",
        "tests/test_task_my_skill.py"]
    for p in written:
        if p.suffix == ".py":
            ast.parse(p.read_text())
    from omegaconf import OmegaConf
    terms = OmegaConf.load(root / "configs/reward/my_skill.yaml").terms
    from tbots.rl.rewards import reward_names
    assert all(t.name in reward_names() for t in terms)
    task_src = (root / "src/tbots/rl/tasks/my_skill.py").read_text()
    for hook in ("def scenario", "def success", "def failure", "obs_builder=", "codec=",
                 "time_limit_s=", 'reward="my_skill"'):
        assert hook in task_src


@pytest.mark.parametrize("bad", ["Demo", "my-skill", "class", "registry", "_x", "go_to_ball"])
def test_scaffold_refuses_bad_or_taken_names(tmp_path, bad):
    with pytest.raises(ValueError):
        scaffold(bad, root=_tree(tmp_path))


def test_scaffold_never_overwrites(tmp_path):
    root = _tree(tmp_path)
    (root / "configs/reward/again.yaml").write_text("terms: []\n")
    with pytest.raises(FileExistsError):
        scaffold("again", root=root)
    assert not (root / "src/tbots/rl/tasks/again.py").exists()
