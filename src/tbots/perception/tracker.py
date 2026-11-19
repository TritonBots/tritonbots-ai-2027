"""Fuse multi-camera detections into one clean world model.

Responsibilities:
  - merge detections of the same robot seen by overlapping cameras
  - associate detections across frames; keep IDs stable
  - estimate velocities (SSL-Vision reports position only)
  - extrapolate through dropouts and mark those states visible=False
  - forward-predict to compensate latency

LATENCY IS THE POINT. By the time a vision frame reaches us it is already
20-40 ms old, and our command takes another 10-20 ms to reach the robot.
At 3 m/s that is ~15 cm of error. rSim hands us perfect instantaneous
state, so a policy trained without simulated latency will depend on
information it can never have at a match.

HOW. Every object is a kinematic Kalman filter, run in the field frame on
capture timestamps: constant acceleration for robot position, constant
velocity for robot heading and for the ball.

  - Robots are associated by (colour, robot_id). SSL-Vision's pattern
    already identifies them; there is nothing to guess. Two cameras that see
    the same robot each update its filter, in capture order -- that is the
    merge, and it weights each sighting by the filter's own uncertainty.
  - The ball has no identity, so each frame's candidate nearest the
    prediction is taken, within a gate. With no track, the most confident
    candidate starts one.
  - Heading is filtered separately, with the innovation wrapped, so a robot
    turning through +-pi does not spin the filter the long way round.
  - A measurement older than the filter's own time is dropped: a late
    camera cannot rewind a track.
  - A measurement more than `manoeuvre` from the prediction (a kick, a
    collision) resets the velocity and acceleration uncertainty, so the
    filter re-learns them in a few frames instead of lagging for many.

The output is the filter state predicted forward to `t_now +
actuation_delay`, the moment a command computed from it takes effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tbots.core.geometry import FieldGeometry
from tbots.core.perspective import IDENTITY, Perspective
from tbots.core.state import (
    BallState,
    DetectionBall,
    DetectionFrame,
    DetectionRobot,
    RobotState,
    WorldState,
)
from tbots.core.units import wrap_angle


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """Tuning. Standard deviations; meters, radians, seconds."""

    # Robot position model: 2 constant velocity, 3 constant acceleration.
    # Constant acceleration wins on rSim trajectories: robots ramp, and a
    # 45 ms forward prediction under a 3-4 m/s^2 ramp is where a CV model
    # loses its centimetre (SETUP_LOG TASK-020 has the sweep).
    robot_order: int = 3
    # Process noise, as the square root of the spectral density of the white
    # noise driving each model (jerk for order 3, acceleration for order 2).
    # Low on purpose: with 2-3 mm vision noise, velocity noise times the
    # prediction horizon costs more than lag does. Sudden changes are
    # handled by the manoeuvre reset below, not by a nervous filter.
    robot_noise: float = 5.0
    robot_rot_noise: float = 3.0
    ball_noise: float = 1.0
    # Measurement noise. SSL-Vision is ~2-5 mm on position for robots.
    robot_pos_std: float = 0.003
    robot_theta_std: float = 0.02
    ball_pos_std: float = 0.005
    # A ball candidate further than this from the prediction is ignored
    # while the track is alive; a robot measurement this far away restarts
    # the robot's track (a teleport, or the robot was carried).
    ball_gate: float = 0.5
    robot_jump: float = 0.5
    min_confidence: float = 0.1
    # A measurement this far from the prediction means the model is wrong
    # right now -- a kick, a collision, a wall. The filter forgets its
    # derivatives and re-learns them from the next few frames.
    manoeuvre: float = 0.03
    # Visible means "in a camera frame no more than visible_timeout older
    # than the newest frame", measured on capture time so that ordinary
    # vision latency does not count as a dropout -- AND that newest frame is
    # itself no older than max_latency, so that losing vision altogether
    # does not leave the whole world looking fresh.
    visible_timeout: float = 0.05
    max_latency: float = 0.15
    # Older than this: dropped from the world entirely.
    drop_timeout: float = 1.0
    # Extrapolation stops here. Beyond it a constant-velocity guess is
    # worse than standing still.
    max_extrapolation: float = 0.25


class _KF:
    """Kinematic Kalman filter over n independent axes.

    `order` 2 is constant velocity, state (p, v), driven by white
    acceleration noise. `order` 3 is constant acceleration, state (p, v, a),
    driven by white jerk noise. `noise` is the square root of that white
    noise's spectral density. Axes share a time and differ only in data, so
    they are stored side by side.
    """

    def __init__(self, z: np.ndarray, t: float, order: int, noise: float,
                 meas_std: float, wrap: bool = False) -> None:
        n = len(z)
        self.k = order
        self.x = np.zeros((n, order))
        self.x[:, 0] = z
        # Position known to the measurement; the derivatives unknown.
        p0 = np.diag([meas_std ** 2, 4.0, 25.0][:order])
        self.P = np.tile(p0, (n, 1, 1))
        self.t = t
        self.q = noise ** 2
        self.r = meas_std ** 2
        self.wrap = wrap

    def _F(self, dt: float) -> np.ndarray:
        if self.k == 2:
            return np.array([[1.0, dt], [0.0, 1.0]])
        return np.array([[1.0, dt, dt * dt / 2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])

    def _Q(self, dt: float) -> np.ndarray:
        if self.k == 2:
            return self.q * np.array([[dt ** 3 / 3, dt ** 2 / 2],
                                      [dt ** 2 / 2, dt]])
        return self.q * np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                                  [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                                  [dt ** 3 / 6, dt ** 2 / 2, dt]])

    def forget_derivatives(self) -> None:
        """Keep position, reset velocity (and acceleration) uncertainty."""
        p0 = np.diag([0.0, 4.0, 25.0][:self.k])
        self.P[:, 1:, :] = 0.0
        self.P[:, :, 1:] = 0.0
        self.P[:, 1:, 1:] = p0[1:, 1:]

    def _wrap(self) -> None:
        if self.wrap:
            self.x[:, 0] = [wrap_angle(a) for a in self.x[:, 0]]

    def predict(self, t: float) -> None:
        dt = t - self.t
        if dt <= 0.0:
            return
        F = self._F(dt)
        self.x = self.x @ F.T
        self.P = F @ self.P @ F.T + self._Q(dt)
        self._wrap()
        self.t = t

    def update(self, z: np.ndarray) -> None:
        y = z - self.x[:, 0]
        if self.wrap:
            y = np.array([wrap_angle(a) for a in y])
        S = self.P[:, 0, 0] + self.r                    # (n,)
        K = self.P[:, :, 0] / S[:, None]                # (n, k)
        self.x = self.x + K * y[:, None]
        # H picks the first state, so P -= K H P is an outer product.
        self.P = self.P - K[:, :, None] * self.P[:, 0:1, :]
        self._wrap()

    def peek(self, t: float) -> np.ndarray:
        """(n, 2) position and velocity at `t`, filter unchanged."""
        dt = max(0.0, t - self.t)
        x = self.x @ self._F(dt).T
        return x[:, :2]

    def innovation(self, z: np.ndarray, t: float) -> float:
        d = z - self.peek(t)[:, 0]
        if self.wrap:
            d = np.array([wrap_angle(a) for a in d])
        return float(np.hypot(*d[:2])) if len(d) >= 2 else float(abs(d[0]))


class _RobotTrack:
    def __init__(self, m: DetectionRobot, t: float, cfg: TrackerConfig) -> None:
        self.manoeuvre = cfg.manoeuvre
        self.pos = _KF(np.array([m.x, m.y]), t, cfg.robot_order, cfg.robot_noise,
                       cfg.robot_pos_std)
        self.rot = _KF(np.array([m.theta]), t, 2, cfg.robot_rot_noise, cfg.robot_theta_std,
                       wrap=True)
        self.last_seen = t

    def update(self, m: DetectionRobot, t: float) -> None:
        z = np.array([m.x, m.y])
        if self.pos.innovation(z, t) > self.manoeuvre:
            self.pos.forget_derivatives()
        self.pos.predict(t)
        self.pos.update(z)
        self.rot.predict(t)
        self.rot.update(np.array([m.theta]))
        self.last_seen = t


class _BallTrack:
    def __init__(self, m: DetectionBall, t: float, cfg: TrackerConfig) -> None:
        self.manoeuvre = cfg.manoeuvre
        self.pos = _KF(np.array([m.x, m.y]), t, 2, cfg.ball_noise, cfg.ball_pos_std)
        self.z = m.z
        self.last_seen = t

    def update(self, m: DetectionBall, t: float) -> None:
        z = np.array([m.x, m.y])
        if self.pos.innovation(z, t) > self.manoeuvre:
            self.pos.forget_derivatives()
        self.pos.predict(t)
        self.pos.update(z)
        self.z = m.z
        self.last_seen = t


class Tracker:
    def __init__(self, geometry: FieldGeometry, actuation_delay: float = 0.0,
                 config: TrackerConfig | None = None) -> None:
        """`actuation_delay`: seconds from a command leaving us to the robot
        acting on it. Output states are predicted to `t_now` plus this."""
        self._geom = geometry
        self.actuation_delay = actuation_delay
        self.cfg = config or TrackerConfig()
        self._robots: dict[tuple[bool, int], _RobotTrack] = {}  # (yellow, id)
        self._ball: _BallTrack | None = None
        self._t_capture: float | None = None

    # -- ingest ---------------------------------------------------------------

    def _robot(self, yellow: bool, m: DetectionRobot, t: float) -> None:
        if m.confidence < self.cfg.min_confidence:
            return
        key = (yellow, m.robot_id)
        tr = self._robots.get(key)
        if tr is None:
            self._robots[key] = _RobotTrack(m, t, self.cfg)
            return
        if t < tr.pos.t:
            return          # a late camera cannot rewind the track
        if tr.pos.innovation(np.array([m.x, m.y]), t) > self.cfg.robot_jump:
            self._robots[key] = _RobotTrack(m, t, self.cfg)
            return
        tr.update(m, t)

    def _balls(self, ms: tuple[DetectionBall, ...], t: float) -> None:
        ms = tuple(m for m in ms if m.confidence >= self.cfg.min_confidence)
        if not ms:
            return
        tr = self._ball
        stale = tr is None or t - tr.last_seen > self.cfg.drop_timeout
        if stale:
            best = max(ms, key=lambda m: m.confidence)
            self._ball = _BallTrack(best, t, self.cfg)
            return
        assert tr is not None
        if t < tr.pos.t:
            return
        d, m = min(((tr.pos.innovation(np.array([m.x, m.y]), t), m) for m in ms),
                   key=lambda p: p[0])
        if d <= self.cfg.ball_gate:
            tr.update(m, t)
        elif t - tr.last_seen > self.cfg.visible_timeout:
            # The tracked ball has been gone a while and something is
            # somewhere else: the ball moved faster than we could follow
            # (a kick while out of view). Follow the new sighting.
            self._ball = _BallTrack(m, t, self.cfg)

    def ingest(self, frames: list[DetectionFrame]) -> None:
        for f in sorted(frames, key=lambda f: f.t_capture):
            t = f.t_capture
            for m in f.robots_blue:
                self._robot(False, m, t)
            for m in f.robots_yellow:
                self._robot(True, m, t)
            self._balls(f.balls, t)
            if self._t_capture is None or t > self._t_capture:
                self._t_capture = t

    # -- output ---------------------------------------------------------------

    def _visible(self, last_seen: float, t_now: float) -> bool:
        newest = self._t_capture if self._t_capture is not None else last_seen
        return (newest - last_seen <= self.cfg.visible_timeout
                and t_now - newest <= self.cfg.max_latency)

    def _robot_state(self, rid: int, tr: _RobotTrack, t: float, t_now: float) -> RobotState:
        horizon = min(t, tr.last_seen + self.cfg.max_extrapolation)
        p = tr.pos.peek(horizon)
        r = tr.rot.peek(horizon)
        return RobotState(robot_id=rid, x=float(p[0, 0]), y=float(p[1, 0]),
                          theta=wrap_angle(float(r[0, 0])),
                          vx=float(p[0, 1]), vy=float(p[1, 1]), vtheta=float(r[0, 1]),
                          visible=self._visible(tr.last_seen, t_now))

    def update(self, frames: list, t_now: float,
               perspective: Perspective = IDENTITY) -> WorldState:
        """Fuse `frames` (raw, field-frame, blue/yellow) into one WorldState.

        `perspective` is passed per call rather than held, because sides swap
        at half time and this object outlives that. Apply it EXACTLY ONCE, at
        the end: fuse and estimate velocities in the field frame the cameras
        report, then rotate the finished frame with
        `perspective.world_state(...)` and split blue/yellow into us/them by
        `perspective.we_are_yellow`. Rotating first and fusing second gives
        the same answer for positions and a subtly wrong one for anything
        that remembers a previous frame.
        """
        if not perspective.resolved:
            raise ValueError("Tracker.update needs a resolved Perspective: "
                             "splitting blue/yellow into us/them without knowing "
                             "our colour would silently swap the teams")
        self.ingest(frames)
        t = t_now + self.actuation_delay

        for key in [k for k, tr in self._robots.items()
                    if t_now - tr.last_seen > self.cfg.drop_timeout]:
            del self._robots[key]

        blue: dict[int, RobotState] = {}
        yellow: dict[int, RobotState] = {}
        for (is_yellow, rid), tr in self._robots.items():
            (yellow if is_yellow else blue)[rid] = self._robot_state(rid, tr, t, t_now)

        if self._ball is None or t_now - self._ball.last_seen > self.cfg.drop_timeout:
            ball = BallState(0.0, 0.0, visible=False)
        else:
            b = self._ball
            p = b.pos.peek(min(t, b.last_seen + self.cfg.max_extrapolation))
            ball = BallState(x=float(p[0, 0]), y=float(p[1, 0]), z=b.z,
                             vx=float(p[0, 1]), vy=float(p[1, 1]),
                             visible=self._visible(b.last_seen, t_now))

        us, them = (yellow, blue) if perspective.we_are_yellow else (blue, yellow)
        field = WorldState(t=t, ball=ball, us=us, them=them, t_capture=self._t_capture)
        return perspective.world_state(field)
