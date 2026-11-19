"""TASK-053 observation builders, and the builder half of TASK-081.

Gate: obs_size(name) matches build_obs output for every registered name;
golden vectors pass; relabelling robot IDs changes no builder's output.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tbots.core.state import BallState, RobotState, WorldState
from tbots.rl.obs.builders import build_obs, is_egocentric, obs_names, obs_size

# Ego robot 0 at the origin facing +y; three teammates, two opponents.
WORLD = WorldState(
    t=1.0, ball=BallState(1.0, 0.5, vx=-1.0, vy=0.5),
    us={0: RobotState(0, 0.0, 0.0, math.pi / 2, vx=0.3, vy=0.6, vtheta=1.0),
        1: RobotState(1, -1.0, 1.0, 0.0),
        2: RobotState(2, -2.0, -2.0, math.pi, vx=1.0),
        3: RobotState(3, 3.0, 0.0, 0.0)},
    them={0: RobotState(0, 0.5, 0.0, math.pi, vy=-1.0),
          4: RobotState(4, 2.0, 2.0, -math.pi / 2, has_ball=True)})

# PINNED. A builder is immutable once any checkpoint exists. If one of these
# fails you changed what an old policy sees: register a new name instead.
GOLDEN = {
    "ego_v1": [
        0.2, -0.1, 0.1,                     # ego velocity (fwd, left), spin
        1 / 9, -2 / 9, 1 / 6, 1 / 3,        # ball rel pos, ball vel
        1.0, 0.0,                           # ball visible, has ball
        0.0, -1.0, 0.0, 1.0,                # their goal, our goal
        1.0, 2 / 9, 2 / 9, -0.2, 0.1,       # nearest teammate (id 1)
        1.0, -4 / 9, 4 / 9, -0.2, -0.7 / 3,  # next teammate (id 2)
        1.0, 0.0, -1 / 9, -1.6 / 3, 0.1,    # nearest opponent (id 0)
        1.0, 4 / 9, -4 / 9, -0.2, 0.1,      # next opponent (id 4)
    ],
    "global_v1": [
        2 / 9, 1 / 9, -1 / 3, 1 / 6, 1.0,
        # us, ordered by x: ids 2, 1, 0, 3, then two empty slots
        1.0, -4 / 9, -4 / 9, -1.0, 0.0, 1 / 3, 0.0, 0.0, 0.0,
        1.0, -2 / 9, 2 / 9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.1, 0.0,
        1.0, 2 / 3, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        *[0.0] * 18,
        # them, ordered by x: ids 0, 4, then four empty slots
        1.0, 1 / 9, 0.0, -1.0, 0.0, 0.0, -1 / 3, 0.0, 0.0,
        1.0, 4 / 9, 4 / 9, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0,
        *[0.0] * 36,
    ],
}


def _ego(name: str) -> int | None:
    return 0 if is_egocentric(name) else None


def test_every_registered_builder_has_a_golden_vector():
    assert set(obs_names()) == set(GOLDEN), "a new builder needs a golden vector here"


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_size_matches_output_and_golden_vector(name):
    v = build_obs(name, WORLD, _ego(name))
    assert v.shape == (obs_size(name),) and v.dtype == np.float32
    np.testing.assert_allclose(v, np.asarray(GOLDEN[name], np.float32), atol=1e-6)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_size_is_fixed_from_one_robot_to_twelve(name):
    solo = WorldState(t=0.0, ball=BallState(0.0, 0.0), us={0: RobotState(0, -1.0, 0.0, 0.0)})
    assert build_obs(name, solo, _ego(name)).shape == (obs_size(name),)


def _relabel(world: WorldState, us_map: dict[int, int], them_map: dict[int, int]) -> WorldState:
    from dataclasses import replace
    return replace(world,
                   us={us_map[i]: replace(r, robot_id=us_map[i]) for i, r in world.us.items()},
                   them={them_map[i]: replace(r, robot_id=them_map[i])
                         for i, r in world.them.items()})


# The match-day labels from ARCHITECTURE section 9, and a shuffle.
RELABELS = [({0: 7, 1: 1, 2: 12, 3: 4}, {0: 9, 4: 3}),
            ({0: 3, 1: 2, 2: 1, 3: 0}, {0: 4, 4: 0})]


@pytest.mark.parametrize("name", sorted(GOLDEN))
@pytest.mark.parametrize(("us_map", "them_map"), RELABELS)
def test_relabelling_robot_ids_changes_no_builder_output(name, us_map, them_map):
    """TASK-081, builder half: no builder encodes an ID value."""
    ego = _ego(name)
    original = build_obs(name, WORLD, ego)
    relabelled = build_obs(name, _relabel(WORLD, us_map, them_map),
                           None if ego is None else us_map[ego])
    np.testing.assert_array_equal(original, relabelled)


def test_egocentric_is_invariant_to_rotating_the_whole_scene_about_the_ego():
    """What 'egocentric' buys: the ball and teammates seen from the ego look
    the same wherever on the field the ego is facing -- except the goals,
    which are fixed in the field and so legitimately move."""
    from dataclasses import replace
    a = 0.7

    def rot(x, y):
        return (x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a))

    def rr(r: RobotState) -> RobotState:
        (x, y), (vx, vy) = rot(r.x, r.y), rot(r.vx, r.vy)
        return replace(r, x=x, y=y, vx=vx, vy=vy, theta=r.theta + a)

    b = WORLD.ball
    (bx, by), (bvx, bvy) = rot(b.x, b.y), rot(b.vx, b.vy)
    turned = replace(WORLD, ball=replace(b, x=bx, y=by, vx=bvx, vy=bvy),
                     us={i: rr(r) for i, r in WORLD.us.items()},
                     them={i: rr(r) for i, r in WORLD.them.items()})
    v0, v1 = build_obs("ego_v1", WORLD, 0), build_obs("ego_v1", turned, 0)
    goals = slice(9, 13)
    keep = np.ones(v0.shape, bool)
    keep[goals] = False
    np.testing.assert_allclose(v0[keep], v1[keep], atol=1e-6)


def test_unknown_builder_and_missing_ego_are_actionable_errors():
    with pytest.raises(KeyError, match="unknown observation builder"):
        build_obs("nope", WORLD, 0)
    with pytest.raises(KeyError, match="ego robot 5"):
        build_obs("ego_v1", WORLD, 5)
    with pytest.raises(ValueError, match="needs a robot_id"):
        build_obs("ego_v1", WORLD, None)


@pytest.mark.parametrize("name", sorted(GOLDEN))
@pytest.mark.parametrize(("us_map", "them_map"), RELABELS)
def test_relabelling_robot_ids_changes_no_policy_output(name, us_map, them_map):
    """TASK-081, policy half: the PPO actor-critic sees the same thing, so it
    does the same thing -- mean action, flag logits, and value."""
    import torch

    from tbots.rl.algo.ppo import Agent
    from tbots.rl.codec import build_codec

    torch.manual_seed(0)
    agent = Agent(obs_size(name), build_codec("skill_v1").action_space)
    ego = _ego(name)
    a = torch.as_tensor(build_obs(name, WORLD, ego))[None]
    b = torch.as_tensor(build_obs(name, _relabel(WORLD, us_map, them_map),
                                  None if ego is None else us_map[ego]))[None]
    with torch.no_grad():
        da, db = agent.dist(a), agent.dist(b)
        for k in da.parts:
            torch.testing.assert_close(da.parts[k].mode(), db.parts[k].mode(), rtol=0, atol=0)
        torch.testing.assert_close(agent.value(a), agent.value(b), rtol=0, atol=0)
