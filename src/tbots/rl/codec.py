"""Action codecs: what a learned skill's network output means as commands.

A codec is named, registered, and IMMUTABLE once a checkpoint exists: the
checkpoint records `codec.spec()`, and loading refuses a mismatch (TASK-059).
A different meaning is a new name (`skill_v2`), never an edit.

`skill_v1` (ARCHITECTURE section 9, "Action codec"):

    action = {"move":  Box(-1, 1, (3,))   forward, left, counter-clockwise
              "flags": MultiBinary(2)}    kick, dribble

- `move` is scaled by `max_speed` and `max_angular_speed`. The linear part
  is limited by its NORM, so full stick on a diagonal is `max_speed`, not
  1.4 times it.
- The booleans map to `kick_speed` and `dribbler_speed`, which come from
  the skill's kwargs: a pass and a shot kick at different powers while the
  network only learns WHEN. The kick is clamped to the 6.5 m/s rule.
- Every tick the commanded velocity moves toward the target by at most
  `max_linear_accel * dt` (as a vector) and `max_angular_accel * dt`, so the
  simulator only executes what a motor can follow. The limit is applied to
  the robot-frame command, which is what the motors see.
- `action_repeat` ticks per decision, each separately rate-limited.

A codec carries the previous command, so it is per-robot and per-episode:
call `reset()` at the start of every episode.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import gymnasium as gym
import numpy as np

from tbots.core.command import RobotCommand
from tbots.core.geometry import MAX_KICK_SPEED
from tbots.core.limits import MAX_ANGULAR_ACCEL, MAX_LINEAR_ACCEL


@runtime_checkable
class ActionCodec(Protocol):
    name: str
    action_space: gym.Space
    action_repeat: int

    def reset(self) -> None: ...

    def decode(self, action: Any, robot_id: int) -> list[RobotCommand]:
        """One command per physics tick this action covers."""
        ...

    def spec(self) -> dict[str, Any]:
        """Everything that determines `decode`, for the checkpoint."""
        ...


_REGISTRY: dict[str, Callable[..., ActionCodec]] = {}


def register_codec(name: str):
    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"codec '{name}' already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def build_codec(name: str, **kwargs: Any) -> ActionCodec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown codec '{name}'. known: {codec_names()}")
    return _REGISTRY[name](**kwargs)


def codec_names() -> list[str]:
    return sorted(_REGISTRY)


def _toward(cur: float, target: float, step: float) -> float:
    return target if abs(target - cur) <= step else cur + math.copysign(step, target - cur)


@register_codec("skill_v1")
class SkillCodecV1:
    name: str

    def __init__(self, *, max_speed: float = 2.0, max_angular_speed: float = 6.0,
                 kick_speed: float = 3.0, chip: bool = False, dribbler_speed: float = 1.0,
                 max_linear_accel: float = MAX_LINEAR_ACCEL,
                 max_angular_accel: float = MAX_ANGULAR_ACCEL,
                 action_repeat: int = 1, dt: float = 1.0 / 60.0) -> None:
        if action_repeat < 1:
            raise ValueError("action_repeat must be >= 1")
        if min(max_speed, max_angular_speed, max_linear_accel, max_angular_accel, dt) <= 0:
            raise ValueError("speeds, accelerations and dt must be positive")
        if not 0.0 <= dribbler_speed <= 1.0:
            raise ValueError("dribbler_speed is a fraction, 0..1")
        self.max_speed = float(max_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.kick_speed = min(max(float(kick_speed), 0.0), MAX_KICK_SPEED)
        self.chip = bool(chip)
        self.dribbler_speed = float(dribbler_speed)
        self.max_linear_accel = float(max_linear_accel)
        self.max_angular_accel = float(max_angular_accel)
        self.action_repeat = int(action_repeat)
        self.dt = float(dt)
        self.action_space = gym.spaces.Dict({
            "move": gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
            "flags": gym.spaces.MultiBinary(2),
        })
        self._v = (0.0, 0.0)
        self._w = 0.0

    def reset(self) -> None:
        self._v = (0.0, 0.0)
        self._w = 0.0

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "max_speed": self.max_speed,
                "max_angular_speed": self.max_angular_speed,
                "kick_speed": self.kick_speed, "chip": self.chip,
                "dribbler_speed": self.dribbler_speed,
                "max_linear_accel": self.max_linear_accel,
                "max_angular_accel": self.max_angular_accel,
                "action_repeat": self.action_repeat, "dt": self.dt}

    def decode(self, action: Any, robot_id: int) -> list[RobotCommand]:
        move = np.clip(np.asarray(action["move"], dtype=np.float64).reshape(3), -1.0, 1.0)
        flags = np.asarray(action["flags"]).reshape(2)
        tx, ty = float(move[0]) * self.max_speed, float(move[1]) * self.max_speed
        norm = math.hypot(tx, ty)
        if norm > self.max_speed:
            tx, ty = tx * self.max_speed / norm, ty * self.max_speed / norm
        tw = float(move[2]) * self.max_angular_speed
        kick = self.kick_speed if flags[0] > 0 else 0.0
        dribble = self.dribbler_speed if flags[1] > 0 else 0.0

        dv = self.max_linear_accel * self.dt
        dw = self.max_angular_accel * self.dt
        out = []
        for _ in range(self.action_repeat):
            vx, vy = self._v
            ex, ey = tx - vx, ty - vy
            err = math.hypot(ex, ey)
            if err > dv:
                vx, vy = vx + ex * dv / err, vy + ey * dv / err
            else:
                vx, vy = tx, ty
            self._v = (vx, vy)
            self._w = _toward(self._w, tw, dw)
            out.append(RobotCommand(robot_id=robot_id, vx=vx, vy=vy, vtheta=self._w,
                                    kick_speed=kick, chip=self.chip and kick > 0,
                                    dribbler=dribble))
        return out
