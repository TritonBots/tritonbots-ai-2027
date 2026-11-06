"""Dynamic, sampling-based model predictive control for one robot.

Every control tick, from where the robot is and how fast it is going:

0. route round the STATIC rule regions (``motion/route.py``) and aim at the
   first waypoint -- a defense area is a local minimum for a one-second
   horizon, and it does not move, so it is routed, not sampled;
1. predict every obstacle over the horizon (``ObstaclePredictor``) -- the
   obstacles MOVE, and each candidate is checked against where they will be
   at the same instant, never against where they are now;
2. build K candidate acceleration sequences: last tick's plan shifted on,
   a time-optimal straight-line "nominal", a full brake, four swerves, and
   smooth random perturbations of the warm start and the nominal;
3. project every candidate onto the robot's limits (``MotionLimits``) and
   roll it out as a double integrator;
4. score it: progress to the target, effort, jerk, swept space-time
   clearance to every predicted obstacle, closing speed (time to collision),
   and the rule keep-out regions;
5. take the MPPI softmax-weighted average, keep whichever of that and the
   best single candidate scores lower, and command its first step.

Two shortcuts keep the planner honest and cheap:

* if the nominal straight line is collision-free over the whole horizon, it
  is used as-is, so in free space this IS the classical time-optimal
  controller and nothing random is ever commanded;
* if no candidate avoids every predicted collision (an opponent is charging
  at us), the planner picks the one whose first collision is LATEST, not the
  brake: braking in front of a moving robot is not safe.

Pure numpy, no state other than the warm start, the last command and a
seeded generator, so identical inputs give identical commands. Imports
``core`` only (Rule 1); global frame, canonical units (Rules 3 and 4).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tbots.core.limits import MotionLimits
from tbots.motion.obstacles import KeepOuts, Obstacle
from tbots.motion.prediction import ConstantVelocityPredictor, ObstaclePredictor
from tbots.motion.route import route


@dataclass(frozen=True, slots=True)
class MPCConfig:
    horizon: int = 20                 # steps
    dt_plan: float = 0.05             # s per step; 20 x 0.05 = a one-second lookahead
    samples: int = 128                # candidates per solve, including the structured ones
    knots: int = 5                    # random perturbations are smooth: interpolated knots
    noise_scale: float = 0.8          # perturbation std, as a fraction of max_a
    temperature: float = 0.1          # MPPI temperature, relative to the cost spread
    brake_fraction: float = 0.8       # nominal decelerates at this fraction of max_a

    w_progress: float = 1.0           # per metre-second of distance to target
    w_terminal: float = 2.0           # per metre of distance at the horizon
    w_effort: float = 0.005           # per (m/s^2)^2 s
    w_jerk: float = 0.0005            # per (m/s^3)^2 s
    w_collision: float = 200.0        # per step spent inside a clearance radius
    w_penetration: float = 2000.0     # per metre of depth inside it
    w_proximity: float = 1.0          # soft buffer just outside it
    proximity: float = 0.15           # m
    w_ttc: float = 1.0                # closing fast on something
    ttc_horizon: float = 0.5          # s
    obstacle_decay: float = 0.98      # per step; uncertainty is mostly in the radius growth
    w_keepout: float = 200.0
    w_keepout_depth: float = 2000.0

    resync_speed: float = 0.5         # m/s; measured vs last command beyond this -> trust measured
    replan_every: int = 1             # ticks between solves; >1 trades fidelity for throughput
    seed: int = 0


def swept_clearance(p0: np.ndarray, p1: np.ndarray, o0: np.ndarray,
                    o1: np.ndarray) -> np.ndarray:
    """Minimum distance between two points each moving linearly over a step.

    Arrays broadcast; the last axis is (x, y). This is the check that does not
    tunnel: two robots closing head-on at 5 m/s cover 0.25 m in one 0.05 s
    plan step, more than a robot diameter, so comparing endpoints alone would
    miss the collision entirely.
    """
    d0 = p0 - o0
    dd = (p1 - o1) - d0
    den = np.einsum("...i,...i->...", dd, dd)
    s = np.clip(-np.einsum("...i,...i->...", d0, dd) / np.maximum(den, 1e-12), 0.0, 1.0)
    closest = d0 + s[..., None] * dd
    return np.sqrt(np.einsum("...i,...i->...", closest, closest))


def _clip_norm(v: np.ndarray, limit: float) -> np.ndarray:
    n = np.sqrt(np.einsum("...i,...i->...", v, v))[..., None]
    return v * np.minimum(1.0, limit / np.maximum(n, 1e-12))


class MPCPlanner:
    """One robot's planner. Keep one instance per robot per behaviour."""

    def __init__(self, config: MPCConfig | None = None, limits: MotionLimits | None = None,
                 predictor: ObstaclePredictor | None = None,
                 control_dt: float = 1.0 / 60.0) -> None:
        self.config = config or MPCConfig()
        self.limits = limits or MotionLimits()
        self.predictor: ObstaclePredictor = predictor or ConstantVelocityPredictor()
        self.control_dt = control_dt
        cfg = self.config
        if cfg.horizon < 2 or cfg.samples < 8 or cfg.knots < 2 or cfg.replan_every < 1:
            raise ValueError("MPCConfig: horizon >= 2, samples >= 8, knots >= 2, replan_every >= 1")
        n = cfg.horizon
        self._times = np.arange(n + 1) * cfg.dt_plan
        self._step_times = np.arange(n) * cfg.dt_plan
        self._decay = cfg.obstacle_decay ** np.arange(n)
        # Linear interpolation from knots to steps, so perturbations are smooth.
        knot_pos = np.linspace(0.0, n - 1, cfg.knots)
        eye = np.eye(cfg.knots)
        self._knot_matrix = np.stack([np.interp(np.arange(n), knot_pos, eye[j])
                                      for j in range(cfg.knots)], axis=1)   # (N, knots)
        self.reset()

    # -- public -------------------------------------------------------------

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.config.seed)
        self._warm: np.ndarray | None = None
        self._last_command: np.ndarray | None = None
        self._trajectory: np.ndarray | None = None
        self._waypoints: list[tuple[float, float]] = []
        self._tick = 0
        self.predictor.reset()

    @property
    def last_command(self) -> tuple[float, float] | None:
        if self._last_command is None:
            return None
        return (float(self._last_command[0]), float(self._last_command[1]))

    @property
    def last_trajectory(self) -> np.ndarray | None:
        """The chosen plan's positions (N+1, 2). For debugging, drawing, and
        one day for teammates to plan against."""
        return None if self._trajectory is None else self._trajectory.copy()

    @property
    def waypoints(self) -> list[tuple[float, float]]:
        """The current global route round the static keep-outs."""
        return list(self._waypoints)

    def plan(self, pos: tuple[float, float], vel: tuple[float, float],
             target: tuple[float, float], obstacles: Sequence[Obstacle],
             keepouts: KeepOuts | None = None, now: float = 0.0) -> tuple[float, float]:
        """Return the global-frame velocity to command this tick."""
        cfg = self.config
        n = cfg.horizon
        p0 = np.asarray(pos, dtype=np.float64)
        v0 = self._initial_velocity(np.asarray(vel, dtype=np.float64))
        keepouts = keepouts or KeepOuts()
        # Static rule regions are routed round globally; the MPC aims at the
        # first waypoint and brakes for the whole remaining route.
        waypoints = route((float(p0[0]), float(p0[1])),
                          (float(target[0]), float(target[1])), keepouts)
        goal = np.asarray(waypoints[0], dtype=np.float64)
        beyond = sum(math.dist(a, b) for a, b in zip(waypoints, waypoints[1:], strict=False))
        self._waypoints = waypoints

        if self._warm is not None and self._tick % cfg.replan_every != 0:
            self._tick += 1
            return self._emit(v0, self._warm[0], self._warm)
        self._tick += 1

        if obstacles:
            centres, radii = self.predictor.predict(obstacles, self._times, now)
            centres, radii = self._reachable(p0, v0, centres, radii)
        else:
            centres, radii = np.zeros((0, n + 1, 2)), np.zeros((0, n + 1))

        # The nominal straight line first, alone. If it is clear for the whole
        # horizon, command the classical controller at control rate and skip
        # sampling: free space costs one rollout, not K.
        nominal = self._track(p0, v0, goal, cfg.dt_plan, n, beyond)
        An, _, Pn = self._rollout(p0, v0, nominal[None])
        _, hit_n = self._cost(An, Pn, goal, centres, radii, keepouts)
        if hit_n[0] == n:
            a0 = self._track(p0, v0, goal, self.control_dt, 1, beyond)[0]
            self._trajectory = Pn[0]
            return self._emit(v0, a0, An[0])

        A, _, P = self._rollout(p0, v0, self._candidates(p0, v0, goal, nominal))
        cost, first_hit = self._cost(A, P, goal, centres, radii, keepouts)

        # MPPI average, then keep the better of it and the best sample.
        spread = float(np.median(cost) - cost.min()) + 1e-9
        w = np.exp(-(cost - cost.min()) / (cfg.temperature * spread))
        w /= w.sum()
        Am, _, Pm = self._rollout(p0, v0, np.einsum("k,kni->ni", w, A)[None])
        cm, hm = self._cost(Am, Pm, goal, centres, radii, keepouts)
        A = np.concatenate([A, Am])
        P = np.concatenate([P, Pm])
        cost = np.concatenate([cost, cm])
        first_hit = np.concatenate([first_hit, hm])

        safe = first_hit == n
        if safe.any():
            best = int(np.argmin(np.where(safe, cost, np.inf)))
        else:
            # Evasion: delay the first collision as long as possible.
            latest = first_hit.max()
            best = int(np.argmin(np.where(first_hit == latest, cost, np.inf)))
        self._trajectory = P[best]
        return self._emit(v0, A[best, 0], A[best])

    # -- internals ----------------------------------------------------------

    def _initial_velocity(self, measured: np.ndarray) -> np.ndarray:
        # Commands are absolute and the robot tracks them (docs/RSIM_FACTS.md,
        # "Robot dynamics"), so the last command is a cleaner estimate of the
        # velocity than a differenced measurement -- until something pushes
        # the robot, a rule filter scales the command, or a new instance
        # starts, in which case the measurement wins.
        measured = _clip_norm(measured, self.limits.max_v)
        last = self._last_command
        if last is None or float(np.hypot(*(measured - last))) > self.config.resync_speed:
            return measured
        return last

    def _emit(self, v0: np.ndarray, a0: np.ndarray, plan: np.ndarray) -> tuple[float, float]:
        a0 = _clip_norm(a0, self.limits.max_a)
        v_cmd = _clip_norm(v0 + a0 * self.control_dt, self.limits.max_v)
        self._last_command = v_cmd
        self._warm = self._shift(plan)
        return (float(v_cmd[0]), float(v_cmd[1]))

    def _shift(self, plan: np.ndarray) -> np.ndarray:
        # Plans are sampled every dt_plan, executed every control_dt: shift by
        # the fraction actually executed. np.interp holds the final value.
        t = self._step_times + self.control_dt
        return np.stack([np.interp(t, self._step_times, plan[:, i]) for i in range(2)], axis=1)

    def _reachable(self, p0: np.ndarray, v0: np.ndarray, centres: np.ndarray,
                   radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Drop obstacles we cannot reach within the horizon, whatever we do."""
        lim = self.limits
        t = self._times
        speed0 = float(np.hypot(*v0))
        reach = np.minimum(speed0 * t + 0.5 * lim.max_a * t * t, lim.max_v * t)
        gap = np.hypot(centres[..., 0] - p0[0], centres[..., 1] - p0[1]) - radii
        keep = (gap <= reach[None, :] + self.config.proximity).any(axis=1)
        return centres[keep], radii[keep]

    def _track(self, p0: np.ndarray, v0: np.ndarray, goal: np.ndarray, dt: float,
               steps: int, beyond: float = 0.0) -> np.ndarray:
        """Accelerations of the time-optimal straight-line controller.

        ``beyond`` is route length still to cover after ``goal``: it sets the
        braking distance, so a waypoint is passed at speed, not stopped at.

        Plain floats: this runs for every solve, and per-element numpy calls
        on two-vectors cost more than the arithmetic.
        """
        max_a, max_v = self.limits.max_a, self.limits.max_v
        a_brake = self.config.brake_fraction * max_a
        gx, gy = float(goal[0]), float(goal[1])
        px, py = float(p0[0]), float(p0[1])
        vx, vy = float(v0[0]), float(v0[1])
        out = np.empty((steps, 2))
        for k in range(steps):
            dx, dy = gx - px, gy - py
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                wx = wy = 0.0
            else:
                speed = min(max_v, math.sqrt(2.0 * a_brake * (dist + beyond)))
                wx, wy = dx * speed / dist, dy * speed / dist
            ax, ay = (wx - vx) / dt, (wy - vy) / dt
            a = math.hypot(ax, ay)
            if a > max_a:
                ax, ay = ax * max_a / a, ay * max_a / a
            nx, ny = vx + ax * dt, vy + ay * dt
            s = math.hypot(nx, ny)
            if s > max_v:
                nx, ny = nx * max_v / s, ny * max_v / s
            out[k] = ((nx - vx) / dt, (ny - vy) / dt)
            px += 0.5 * (vx + nx) * dt
            py += 0.5 * (vy + ny) * dt
            vx, vy = nx, ny
        return out

    def _candidates(self, p0: np.ndarray, v0: np.ndarray, goal: np.ndarray,
                    nominal: np.ndarray) -> np.ndarray:
        cfg, lim = self.config, self.limits
        n = cfg.horizon
        warm = self._warm if self._warm is not None else nominal

        # Full brake: decelerate at max_a along -v until stopped.
        speed0 = float(np.hypot(*v0))
        brake = np.zeros((n, 2))
        if speed0 > 1e-9:
            t_stop = speed0 / lim.max_a
            frac = np.clip((t_stop - self._step_times) / cfg.dt_plan, 0.0, 1.0)
            brake = -(v0 / speed0) * lim.max_a * frac[:, None]

        heading = goal - p0
        norm = float(np.hypot(*heading))
        if norm < 1e-9:
            heading = v0 if speed0 > 1e-9 else np.array([1.0, 0.0])
            norm = float(np.hypot(*heading))
        perp = np.array([-heading[1], heading[0]]) / norm
        first_half = (np.arange(n) < n // 2)[:, None]
        swerves = [nominal + side * strength * lim.max_a * perp * first_half
                   for side in (1.0, -1.0) for strength in (1.0, 0.5)]

        structured = np.stack([warm, nominal, brake, *swerves])
        n_random = cfg.samples - len(structured)
        knots = self._rng.normal(0.0, cfg.noise_scale * lim.max_a,
                                 size=(n_random, cfg.knots, 2))
        noise = np.einsum("nj,kji->kni", self._knot_matrix, knots)
        around_warm = n_random * 2 // 3
        base = np.concatenate([np.broadcast_to(warm, (around_warm, n, 2)),
                               np.broadcast_to(nominal, (n_random - around_warm, n, 2))])
        return np.concatenate([structured, base + noise])

    def _rollout(self, p0: np.ndarray, v0: np.ndarray, A: np.ndarray):
        """Project onto the limits and integrate. A (K, N, 2)."""
        cfg, lim = self.config, self.limits
        dt = cfg.dt_plan
        A = _clip_norm(A, lim.max_a)
        k_count = A.shape[0]
        V = np.empty((k_count, cfg.horizon + 1, 2))
        V[:, 0] = v0
        for k in range(cfg.horizon):
            # Projection onto the speed disc is non-expansive, so this never
            # makes the effective acceleration exceed max_a.
            V[:, k + 1] = _clip_norm(V[:, k] + A[:, k] * dt, lim.max_v)
        A_eff = np.diff(V, axis=1) / dt
        steps = 0.5 * (V[:, 1:] + V[:, :-1]) * dt
        P = np.concatenate([np.broadcast_to(p0, (k_count, 1, 2)),
                            p0 + np.cumsum(steps, axis=1)], axis=1)
        return A_eff, V, P

    def _cost(self, A: np.ndarray, P: np.ndarray, goal: np.ndarray, centres: np.ndarray,
              radii: np.ndarray, keepouts: KeepOuts) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        dt, n = cfg.dt_plan, cfg.horizon
        x, y = P[:, 1:, 0], P[:, 1:, 1]                                  # (K, N)
        dist = np.hypot(x - goal[0], y - goal[1])
        cost = cfg.w_progress * dist.sum(axis=1) * dt + cfg.w_terminal * dist[:, -1]
        cost += cfg.w_effort * (A * A).sum(axis=(1, 2)) * dt
        jerk = np.diff(A, axis=1)
        cost += cfg.w_jerk * (jerk * jerk).sum(axis=(1, 2)) / dt
        hit = np.zeros(x.shape, dtype=bool)

        if centres.shape[0]:
            # Swept space-time clearance, by components (see swept_clearance;
            # this is the same algebra, sharing the dot product with the
            # closing-speed term). Shapes: (K, M, N).
            px, py = P[:, None, :, 0], P[:, None, :, 1]
            ox, oy = centres[None, :, :, 0], centres[None, :, :, 1]
            rx, ry = px - ox, py - oy                                    # (K, M, N+1)
            dx0, dy0 = rx[..., :-1], ry[..., :-1]
            ddx, ddy = rx[..., 1:] - dx0, ry[..., 1:] - dy0
            dot = dx0 * ddx + dy0 * ddy
            den = np.maximum(ddx * ddx + ddy * ddy, 1e-12)
            s = np.clip(-dot / den, 0.0, 1.0)
            cx, cy = dx0 + s * ddx, dy0 + s * ddy
            gap = np.sqrt(cx * cx + cy * cy) - radii[None, :, 1:]
            inside = gap < 0.0
            hit |= inside.any(axis=1)
            per_step = np.where(
                inside,
                cfg.w_collision - cfg.w_penetration * gap,
                cfg.w_proximity * np.clip(1.0 - gap / cfg.proximity, 0.0, 1.0) ** 2,
            )
            # Time to collision from the closing speed at the start of each step.
            sep = np.sqrt(dx0 * dx0 + dy0 * dy0)
            closing = -dot / (np.maximum(sep, 1e-9) * dt)
            gap0 = np.maximum(sep - radii[None, :, :-1], 0.0)
            ttc = gap0 / np.maximum(closing, 1e-6)
            urgency = np.where(closing > 1e-6,
                               np.clip(1.0 - ttc / cfg.ttc_horizon, 0.0, 1.0) ** 2, 0.0)
            per_step += cfg.w_ttc * urgency
            cost += (per_step.sum(axis=1) * self._decay).sum(axis=1)

        depths = []
        for r in keepouts.rects:
            depths.append(np.minimum(np.minimum(x - r.x_min, r.x_max - x),
                                     np.minimum(y - r.y_min, r.y_max - y)))
        for c in keepouts.circles:
            depths.append(c.radius - np.hypot(x - c.x, y - c.y))
        if keepouts.bounds is not None:
            b = keepouts.bounds
            depths.append(np.maximum(np.maximum(b.x_min - x, x - b.x_max),
                                     np.maximum(b.y_min - y, y - b.y_max)))
        for depth in depths:
            bad = depth > 0.0
            hit |= bad
            cost += np.where(bad, cfg.w_keepout + cfg.w_keepout_depth * depth, 0.0).sum(axis=1)

        first_hit = np.where(hit.any(axis=1), hit.argmax(axis=1), n)
        return cost, first_hit
