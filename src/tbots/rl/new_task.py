"""Scaffold a new learned skill.

    python -m tbots.rl.new_task dribble

writes three files and changes nothing else:

    src/tbots/rl/tasks/dribble.py      the task: every hook stubbed and explained
    configs/reward/dribble.yaml        its default reward
    tests/test_task_dribble.py         it registers, and one episode runs

The task works as written (it is go_to_ball with a 0.3 m radius), so the
test passes before you change anything. Then make it yours, and train it:

    pytest tests/test_task_dribble.py
    python -m tbots.rl.train env=skill task=dribble train.run_dir=runs/dribble
"""

from __future__ import annotations

import argparse
import keyword
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

TASK = '''"""{name}: <one line saying what this skill does>.

Made by `python -m tbots.rl.new_task {name}`. It already works -- it is a
copy of go_to_ball with a looser radius -- so start by running its test,
then change one hook at a time and re-run it.

Train:  python -m tbots.rl.train env=skill task={name} train.run_dir=runs/{name}
Watch:  python -m tbots.apps.eval --checkpoint runs/{name}/latest.pt --render vision --realtime
"""

from __future__ import annotations

import math

import numpy as np

from tbots.backends.base import Scenario
from tbots.core.state import WorldState
from tbots.rl.tasks import SkillTask, register_task
from tbots.rl.tasks.sampling import random_point, random_pose, scatter


def scenario(rng: np.random.Generator, n_us: int, n_them: int) -> Scenario:
    """Where everything starts. Called at every episode reset.

    Return a Scenario with exactly `n_us` of our poses (the FIRST is the
    robot being trained) and `n_them` of theirs, each (x, y, theta) in
    meters and radians, plus the ball as (x, y, vx, vy). We attack +x.

    Draw every random number from `rng`. It is seeded, which is what makes
    a run reproducible; `random` or `np.random` directly would not be.
    """
    ego = random_pose(rng, margin=0.5)
    ball = random_point(rng, margin=0.5)
    others = scatter(rng, n_us - 1 + n_them, avoid=[ball, (ego[0], ego[1])])
    return Scenario(ball=(ball[0], ball[1], 0.0, 0.0),
                    us=(ego, *others[:n_us - 1]), them=tuple(others[n_us - 1:]))


def success(world: WorldState, ego_id: int) -> bool:
    """True ends the episode as SKILL_SUCCESS. `world` is ground truth.

    Keep it cheap and unambiguous: it runs every physics tick.
    """
    me = world.us[ego_id]
    return math.hypot(me.x - world.ball.x, me.y - world.ball.y) < 0.3


def failure(world: WorldState, ego_id: int) -> bool:
    """True ends the episode as SKILL_FAILURE. Return False if there is no
    failure short of running out of time (the ball leaving the field
    already ends the episode as OUT_OF_BOUNDS)."""
    return False


register_task(SkillTask(
    name="{name}",
    scenario=scenario,
    success=success,
    failure=failure,
    # What the policy sees. Egocentric, registered in rl/obs/builders.py.
    # Builders are immutable once a checkpoint exists: need a different
    # view? Register a new builder name, never edit an old one.
    obs_builder="ego_v1",
    # What the policy's outputs mean (rl/codec.py), and this skill's
    # settings for it: speeds in m/s and rad/s, kick speed in m/s (clamped
    # to 6.5), dribbler 0..1, action_repeat ticks per decision.
    codec="skill_v1",
    codec_kwargs={{"max_speed": 2.0, "max_angular_speed": 6.0, "kick_speed": 3.0}},
    # Episode length in simulated seconds; reaching it is a TIMEOUT.
    time_limit_s=8.0,
    # configs/reward/{name}.yaml. `reward=OTHER` on the command line overrides.
    reward="{name}",
))
'''

REWARD = '''# Default reward for the {name} task: a weighted sum of registered terms
# (src/tbots/rl/rewards/). Each item: name, weight, and any keyword
# arguments for the term. Write new terms in src/tbots/rl/rewards/.
terms:
  - {{name: ball_progress, weight: 1.0}}           # meters closed on the ball
  - {{name: outcome, weight: 1.0, success: 1.0}}   # paid once, when it ends
  - {{name: time_cost, weight: 0.1}}               # -1 per simulated second
'''

TEST = '''"""Made by `python -m tbots.rl.new_task {name}`. Keep it passing."""

from __future__ import annotations

import numpy as np

from tbots.rl.factory import compose_config, make_env
from tbots.rl.tasks import get_task, task_names


def test_{name}_registers():
    assert "{name}" in task_names()
    assert get_task("{name}").reward == "{name}"


def test_{name}_scenario_places_every_robot():
    for n_us, n_them in ((1, 0), (3, 2), (6, 6)):
        s = get_task("{name}").scenario(np.random.default_rng(0), n_us, n_them)
        assert len(s.us) == n_us and len(s.them) == n_them


def test_one_{name}_episode_runs():
    env = make_env(compose_config(["task={name}", "env.perception.mode=tracked"]))
    try:
        obs, _ = env.reset(seed=0)
        env.action_space.seed(0)
        done, info = False, {{}}
        while not done:
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert np.isfinite(r) and env.observation_space.contains(obs)
            done = term or trunc
        assert info["done"] in {{"SKILL_SUCCESS", "SKILL_FAILURE", "TIMEOUT", "OUT_OF_BOUNDS"}}
    finally:
        env.close()
'''


def scaffold(name: str, root: Path = ROOT) -> list[Path]:
    if not name.isidentifier() or keyword.iskeyword(name) or name != name.lower():
        raise ValueError(f"task name must be a lower-case Python identifier, got {name!r}")
    if name in {"registry", "sampling"} or name.startswith("_"):
        raise ValueError(f"{name!r} is reserved")
    from tbots.rl.tasks import task_names
    if name in task_names():
        raise ValueError(f"a task named {name!r} is already registered")
    files = {
        root / "src" / "tbots" / "rl" / "tasks" / f"{name}.py": TASK.format(name=name),
        root / "configs" / "reward" / f"{name}.yaml": REWARD.format(name=name),
        root / "tests" / f"test_task_{name}.py": TEST.format(name=name),
    }
    existing = [p for p in files if p.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {', '.join(map(str, existing))}")
    for path, text in files.items():
        path.write_text(text)
    return list(files)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="the task name, e.g. dribble")
    args = p.parse_args(argv)
    try:
        written = scaffold(args.name)
    except (ValueError, FileExistsError) as e:
        sys.exit(f"new_task: {e}")
    for path in written:
        print(f"wrote {path.relative_to(ROOT)}")
    print(f"\nnext:  pytest tests/test_task_{args.name}.py\n"
          f"       python -m tbots.rl.train env=skill task={args.name} "
          f"train.run_dir=runs/{args.name}")


if __name__ == "__main__":
    main()
