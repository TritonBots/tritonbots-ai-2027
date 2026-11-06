"""Where each moving obstacle will be over the planning horizon.

This is what makes the MPC DYNAMIC: the planner never checks a candidate
trajectory against where an opponent IS, only against where it is predicted
to be at the same instant. A predictor returns, for every obstacle and every
time in the horizon grid, a centre and a clearance radius.

The radius GROWS with lookahead. Opponents do not hold a constant velocity,
so a prediction one second out is a region, not a point. The growth is a
reachable-set bound -- how far a robot accelerating at ``accel_bound`` could
deviate from the prediction, ``1/2 a t^2`` -- capped so a long horizon does
not wall off the whole field. Near-term predictions stay tight, which is
where collisions are actually decided; receding-horizon replanning every
tick takes care of the rest.

The ball is exempt from growth: it does not decide to turn.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from tbots.motion.obstacles import Obstacle, ObstacleKind


@runtime_checkable
class ObstaclePredictor(Protocol):
    def predict(self, obstacles: Sequence[Obstacle], times: np.ndarray,
                now: float) -> tuple[np.ndarray, np.ndarray]:
        """``times`` is the horizon grid (N,), starting at 0, relative to ``now``.

        Returns centres (M, N, 2) and radii (M, N). ``now`` is the world time,
        for predictors that keep a history.
        """
        ...

    def reset(self) -> None: ...


def _clip_speed(v: np.ndarray, max_speed: float) -> np.ndarray:
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_speed / np.maximum(speed, 1e-12))
    return v * scale


class ConstantVelocityPredictor:
    """Straight lines at the reported velocity. The default."""

    def __init__(self, *, max_speed: float = 4.0, accel_bound: float = 0.5,
                 growth_cap: float = 0.10) -> None:
        self.max_speed = max_speed
        self.accel_bound = accel_bound
        self.growth_cap = growth_cap

    def reset(self) -> None:
        pass

    def predict(self, obstacles: Sequence[Obstacle], times: np.ndarray,
                now: float) -> tuple[np.ndarray, np.ndarray]:
        pos, vel, r0, grows = _unpack(obstacles)
        vel = _clip_speed(vel, self.max_speed)
        centres = pos[:, None, :] + vel[:, None, :] * times[None, :, None]
        return centres, self._radii(r0, grows, times)

    def _radii(self, r0: np.ndarray, grows: np.ndarray, times: np.ndarray) -> np.ndarray:
        growth = np.minimum(0.5 * self.accel_bound * times ** 2, self.growth_cap)
        return r0[:, None] + grows[:, None] * growth[None, :]


class ConstantAccelerationPredictor(ConstantVelocityPredictor):
    """Extrapolates each obstacle's estimated acceleration for a short while.

    Acceleration is estimated from the velocity history it is shown, one
    smoothed finite difference per obstacle per call, so it follows an
    opponent that is turning or braking. The acceleration is applied only for
    ``accel_horizon`` seconds and the speed is clipped: extrapolating a
    manoeuvre for a whole second predicts nonsense.

    Stateful: give each planner its own instance, and call ``reset()`` when
    the world jumps (a new episode, a ``place()``).
    """

    def __init__(self, *, max_speed: float = 4.0, accel_bound: float = 0.5,
                 growth_cap: float = 0.10, accel_horizon: float = 0.4,
                 max_accel: float = 6.0, smoothing: float = 0.5) -> None:
        super().__init__(max_speed=max_speed, accel_bound=accel_bound, growth_cap=growth_cap)
        self.accel_horizon = accel_horizon
        self.max_accel = max_accel
        self.smoothing = smoothing
        self._history: dict[tuple[ObstacleKind, int], tuple[float, np.ndarray, np.ndarray]] = {}

    def reset(self) -> None:
        self._history.clear()

    def predict(self, obstacles: Sequence[Obstacle], times: np.ndarray,
                now: float) -> tuple[np.ndarray, np.ndarray]:
        pos, vel, r0, grows = _unpack(obstacles)
        acc = np.zeros_like(vel)
        for i, o in enumerate(obstacles):
            key = (o.kind, o.id)
            previous = self._history.get(key)
            estimate = np.zeros(2)
            if previous is not None and now > previous[0]:
                raw = (vel[i] - previous[1]) / (now - previous[0])
                estimate = self.smoothing * raw + (1.0 - self.smoothing) * previous[2]
                estimate = _clip_speed(estimate, self.max_accel)
            self._history[key] = (now, vel[i].copy(), estimate)
            if o.kind is not ObstacleKind.BALL:
                acc[i] = estimate

        # Velocity at each grid time, then integrate (trapezoid) to positions.
        t_acc = np.minimum(times, self.accel_horizon)
        v_t = _clip_speed(vel[:, None, :] + acc[:, None, :] * t_acc[None, :, None],
                          self.max_speed)
        dt = np.diff(times)
        steps = 0.5 * (v_t[:, 1:, :] + v_t[:, :-1, :]) * dt[None, :, None]
        centres = np.concatenate([pos[:, None, :], pos[:, None, :] + np.cumsum(steps, axis=1)],
                                 axis=1)
        return centres, self._radii(r0, grows, times)


def _unpack(obstacles: Sequence[Obstacle]):
    m = len(obstacles)
    pos = np.empty((m, 2))
    vel = np.empty((m, 2))
    r0 = np.empty(m)
    grows = np.empty(m)
    for i, o in enumerate(obstacles):
        pos[i] = (o.x, o.y)
        vel[i] = (o.vx, o.vy)
        r0[i] = o.radius
        grows[i] = 0.0 if o.kind is ObstacleKind.BALL else 1.0
    return pos, vel, r0, grows


PREDICTORS = {
    "constant_velocity": ConstantVelocityPredictor,
    "constant_acceleration": ConstantAccelerationPredictor,
}


def build_predictor(name: str) -> ObstaclePredictor:
    if name not in PREDICTORS:
        raise ValueError(f"unknown predictor '{name}'. known: {sorted(PREDICTORS)}")
    return PREDICTORS[name]()
