"""Named observation builders.

A builder turns a WorldState into a fixed-size float32 vector. Named, so a
checkpoint can record which one it was trained with and skills/learned.py
can look it up by name -- an observation encoding silently drifting apart
from the policy that was trained on it is a very expensive bug.

RULES (ARCHITECTURE section 9, "Observation builders"):

- IMMUTABLE ONCE A CHECKPOINT EXISTS. `tests/test_obs.py` pins a golden
  vector per builder. If it fails, you changed what an old policy sees:
  register a new name (`ego_v2`) instead of editing.
- No builder encodes a robot ID value. Our robots at a match might be
  {1, 3, 4, 7, 9, 12}; every skill env trains robot 0. Other robots are
  ordered by distance (egocentric) or by position (global), never by ID,
  and `tests/test_obs.py` asserts relabelling changes nothing (TASK-081).
- No running normalisation. Values are divided by the fixed constants
  below, so there is no hidden statistic to export with a policy. The
  constants are an open problem (TASKS.md section 6) and are part of each
  builder's definition: changing them is a new builder.
- Division B only. The field never shrinks across a curriculum, so goal
  positions are fixed constants, not read from config.

Skill builders are EGOCENTRIC: everything relative to the ego robot's pose
and heading, so a dribble learned facing +x works facing anywhere. The
tactics builder here is GLOBAL and fixed-size; the permutation-invariant
set encoder that should eventually consume robot sets is TASK-083.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tbots.core.geometry import DIV_B
from tbots.core.state import RobotState, WorldState

# Fixed scaling constants, shared by the v1 builders.
POS_SCALE = DIV_B.half_length       # 4.5 m
VEL_SCALE = 3.0                     # m/s
SPIN_SCALE = 10.0                   # rad/s
THEIR_GOAL = DIV_B.their_goal
OUR_GOAL = DIV_B.our_goal

BuilderFn = Callable[[WorldState, "int | None"], np.ndarray]


@dataclass(frozen=True)
class _Builder:
    name: str
    size: int
    fn: BuilderFn
    egocentric: bool


_REGISTRY: dict[str, _Builder] = {}


def register_obs(name: str, size: int, *, egocentric: bool):
    """Register `fn(world, robot_id) -> vector` under `name` with a fixed size.
    `egocentric` builders need a `robot_id`; global ones take `None`."""
    def deco(fn: BuilderFn) -> BuilderFn:
        if name in _REGISTRY:
            raise ValueError(f"observation builder '{name}' already registered")
        _REGISTRY[name] = _Builder(name, size, fn, egocentric)
        return fn
    return deco


def obs_names() -> list[str]:
    return sorted(_REGISTRY)


def _get(name: str) -> _Builder:
    if name not in _REGISTRY:
        raise KeyError(f"unknown observation builder '{name}'. known: {obs_names()}")
    return _REGISTRY[name]


def obs_size(name: str) -> int:
    return _get(name).size


def is_egocentric(name: str) -> bool:
    return _get(name).egocentric


def build_obs(name: str, world: WorldState, robot_id: int | None) -> np.ndarray:
    b = _get(name)
    if b.egocentric and robot_id is None:
        raise ValueError(f"'{name}' is egocentric and needs a robot_id")
    out = b.fn(world, robot_id)
    if out.shape != (b.size,) or out.dtype != np.float32:
        raise RuntimeError(f"builder '{name}' returned {out.dtype}{out.shape}, "
                           f"registered as float32({b.size},)")
    return out


# -- ego_v1 ------------------------------------------------------------------

EGO_NEIGHBOURS = 2      # nearest teammates, and separately nearest opponents


@register_obs("ego_v1", size=13 + 2 * EGO_NEIGHBOURS * 5, egocentric=True)
def _ego_v1(world: WorldState, robot_id: int | None) -> np.ndarray:
    """Layout (33 floats), every vector in the ego robot's frame
    (x forward out of the kicker, y left):

        0-1   ego velocity / VEL_SCALE
        2     ego spin / SPIN_SCALE
        3-4   ball position relative to ego / POS_SCALE
        5-6   ball velocity / VEL_SCALE
        7     ball visible (1/0)
        8     ego has the ball (1/0)
        9-10  their goal centre relative to ego / POS_SCALE
        11-12 our goal centre relative to ego / POS_SCALE
        13-22 nearest 2 teammates, 23-32 nearest 2 opponents, nearest first,
              each [present, dx, dy, dvx, dvy] (positions / POS_SCALE,
              velocities / VEL_SCALE); absent slots are all zero.
    """
    if robot_id is None or robot_id not in world.us:
        raise KeyError(f"ego robot {robot_id} is not in world.us {sorted(world.us)}")
    me = world.us[robot_id]
    c, s = math.cos(me.theta), math.sin(me.theta)

    def rot(dx: float, dy: float) -> tuple[float, float]:
        return (c * dx + s * dy, -s * dx + c * dy)

    def rel(x: float, y: float) -> tuple[float, float]:
        return rot(x - me.x, y - me.y)

    v = rot(me.vx, me.vy)
    ball = rel(world.ball.x, world.ball.y)
    bv = rot(world.ball.vx, world.ball.vy)
    tg = rel(*THEIR_GOAL)
    og = rel(*OUR_GOAL)
    out = [v[0] / VEL_SCALE, v[1] / VEL_SCALE, me.vtheta / SPIN_SCALE,
           ball[0] / POS_SCALE, ball[1] / POS_SCALE, bv[0] / VEL_SCALE, bv[1] / VEL_SCALE,
           1.0 if world.ball.visible else 0.0, 1.0 if me.has_ball else 0.0,
           tg[0] / POS_SCALE, tg[1] / POS_SCALE, og[0] / POS_SCALE, og[1] / POS_SCALE]

    def nearest(robots: list[RobotState]) -> list[float]:
        rows = []
        for r in robots:
            d = rel(r.x, r.y)
            dv = rot(r.vx - me.vx, r.vy - me.vy)
            rows.append((math.hypot(*d), d, dv))
        # Distance, then position, as the tie-break: never the ID.
        rows.sort(key=lambda t: (t[0], t[1]))
        flat: list[float] = []
        for _, d, dv in rows[:EGO_NEIGHBOURS]:
            flat += [1.0, d[0] / POS_SCALE, d[1] / POS_SCALE, dv[0] / VEL_SCALE, dv[1] / VEL_SCALE]
        flat += [0.0] * (5 * EGO_NEIGHBOURS - len(flat))
        return flat

    out += nearest([r for i, r in world.us.items() if i != robot_id])
    out += nearest(list(world.them.values()))
    return np.asarray(out, dtype=np.float32)


# -- global_v1 ---------------------------------------------------------------

GLOBAL_SLOTS = DIV_B.max_robots     # 6 per side, always: the size never changes
_ROBOT_FEATURES = 9


@register_obs("global_v1", size=5 + 2 * GLOBAL_SLOTS * _ROBOT_FEATURES, egocentric=False)
def _global_v1(world: WorldState, robot_id: int | None = None) -> np.ndarray:
    """Layout (113 floats), field frame (we attack +x):

        0-4    ball x, y / POS_SCALE, vx, vy / VEL_SCALE, visible
        5-58   our 6 slots, 59-112 their 6 slots, each
               [present, x, y (/POS_SCALE), cos theta, sin theta,
                vx, vy (/VEL_SCALE), spin (/SPIN_SCALE), has_ball]

    Slots are filled in order of (x, y), not of robot ID, and padded with
    zeros, so the vector is the same size from 1v0 to 6v6 and a stage can
    resume from the one before it.
    """
    b = world.ball
    out = [b.x / POS_SCALE, b.y / POS_SCALE, b.vx / VEL_SCALE, b.vy / VEL_SCALE,
           1.0 if b.visible else 0.0]
    for robots in (world.us, world.them):
        if len(robots) > GLOBAL_SLOTS:
            raise ValueError(f"global_v1 holds {GLOBAL_SLOTS} robots a side, got {len(robots)}")
        for r in sorted(robots.values(), key=lambda r: (r.x, r.y)):
            out += [1.0, r.x / POS_SCALE, r.y / POS_SCALE, math.cos(r.theta), math.sin(r.theta),
                    r.vx / VEL_SCALE, r.vy / VEL_SCALE, r.vtheta / SPIN_SCALE,
                    1.0 if r.has_ball else 0.0]
        out += [0.0] * (_ROBOT_FEATURES * (GLOBAL_SLOTS - len(robots)))
    return np.asarray(out, dtype=np.float32)
