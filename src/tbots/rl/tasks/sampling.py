"""Helpers for scenario samplers. Positions in meters, our frame (+x attacks)."""

from __future__ import annotations

import math

import numpy as np

from tbots.backends.base import Pose
from tbots.core.geometry import DIV_B, FieldGeometry

# Robots this close (centre to centre) would start overlapping.
MIN_SEPARATION = 0.3


def random_point(rng: np.random.Generator, margin: float = 0.3,
                 geometry: FieldGeometry = DIV_B) -> tuple[float, float]:
    """Uniform over the field, `margin` meters inside the lines."""
    return (float(rng.uniform(-geometry.half_length + margin, geometry.half_length - margin)),
            float(rng.uniform(-geometry.half_width + margin, geometry.half_width - margin)))


def random_pose(rng: np.random.Generator, margin: float = 0.3,
                geometry: FieldGeometry = DIV_B) -> Pose:
    x, y = random_point(rng, margin, geometry)
    return (x, y, float(rng.uniform(-math.pi, math.pi)))


def scatter(rng: np.random.Generator, n: int, avoid: list[tuple[float, float]],
            margin: float = 0.3, geometry: FieldGeometry = DIV_B) -> list[Pose]:
    """`n` random poses at least MIN_SEPARATION from each other and from
    every point in `avoid` (the ball, robots already placed)."""
    taken = list(avoid)
    out: list[Pose] = []
    for _ in range(n):
        for _attempt in range(1000):
            p = random_pose(rng, margin, geometry)
            if all(math.hypot(p[0] - x, p[1] - y) >= MIN_SEPARATION for x, y in taken):
                break
        else:
            raise RuntimeError("could not place a robot clear of the others")
        taken.append((p[0], p[1]))
        out.append(p)
    return out
