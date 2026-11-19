"""Rollout files: one recorded episode, ground truth, tick by tick.

What `apps/eval.py` writes and `apps/replay.py` streams to the browser. Also
a regression artifact: attach one to a pull request and a reviewer can
watch exactly what the policy did (ARCHITECTURE section 9).

One compressed `.npz` per episode:

    t           (T,)              seconds; index 0 is the state after reset
    ball        (T, 6)            x, y, z, vx, vy, vz
    us, them    (T, n, 8)         x, y, theta, vx, vy, vtheta, has_ball, visible
    us_ids, them_ids  (n,)        robot ids, in the column order above
    commands    (T, n_us, 7)      robot_id (-1: none), vx, vy, vtheta,
                                  kick_speed, chip, dribbler; row 0 is empty
    term_names  (K,)              reward term names
    terms       (T, K)            per-term reward each tick; row 0 is zero
    done        ()                Done name, or "" if the recording stopped early
    meta        ()                JSON: task, seed, dt, perception mode, ...

Ground truth only: a rollout shows what HAPPENED, not what the policy saw.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tbots.core.command import RobotCommand
from tbots.core.state import BallState, RobotState, WorldState
from tbots.core.transition import Transition

ROBOT_COLS = 8
CMD_COLS = 7


def _robot_row(r: RobotState) -> list[float]:
    return [r.x, r.y, r.theta, r.vx, r.vy, r.vtheta, float(r.has_ball), float(r.visible)]


def _robot(rid: int, row: np.ndarray) -> RobotState:
    return RobotState(robot_id=int(rid), x=float(row[0]), y=float(row[1]), theta=float(row[2]),
                      vx=float(row[3]), vy=float(row[4]), vtheta=float(row[5]),
                      has_ball=bool(row[6] > 0.5), visible=bool(row[7] > 0.5))


@dataclass
class Rollout:
    t: np.ndarray
    ball: np.ndarray
    us: np.ndarray
    them: np.ndarray
    us_ids: np.ndarray
    them_ids: np.ndarray
    commands: np.ndarray
    term_names: list[str]
    terms: np.ndarray
    done: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def dt(self) -> float:
        return float(self.meta.get("dt", self.t[1] - self.t[0] if len(self.t) > 1 else 1 / 60))

    def world(self, i: int) -> WorldState:
        b = self.ball[i]
        return WorldState(
            t=float(self.t[i]), t_capture=float(self.t[i]),
            ball=BallState(x=float(b[0]), y=float(b[1]), z=float(b[2]),
                           vx=float(b[3]), vy=float(b[4]), vz=float(b[5])),
            us={int(r): _robot(r, self.us[i, j]) for j, r in enumerate(self.us_ids)},
            them={int(r): _robot(r, self.them[i, j]) for j, r in enumerate(self.them_ids)})

    def worlds(self) -> Iterator[WorldState]:
        for i in range(len(self)):
            yield self.world(i)

    def term_totals(self) -> dict[str, float]:
        return {n: float(self.terms[:, k].sum()) for k, n in enumerate(self.term_names)}


class RolloutRecorder:
    """Collects one episode from an env's `tick_hook`.

        rec = RolloutRecorder(term_names, meta)
        rec.start(env.truth)           after reset
        env.unwrapped.tick_hook = rec.on_tick
        ... run the episode ...
        rollout = rec.finish()
    """

    def __init__(self, term_names: list[str], meta: dict[str, Any] | None = None) -> None:
        self.term_names = list(term_names)
        self.meta = dict(meta or {})
        self._t: list[float] = []
        self._ball: list[list[float]] = []
        self._us: list[list[list[float]]] = []
        self._them: list[list[list[float]]] = []
        self._cmd: list[list[list[float]]] = []
        self._terms: list[list[float]] = []
        self._us_ids: list[int] = []
        self._them_ids: list[int] = []
        self._done = ""

    def _frame(self, w: WorldState) -> None:
        b = w.ball
        self._t.append(w.t)
        self._ball.append([b.x, b.y, b.z, b.vx, b.vy, b.vz])
        self._us.append([_robot_row(w.us[i]) for i in self._us_ids])
        self._them.append([_robot_row(w.them[i]) for i in self._them_ids])

    def start(self, truth: WorldState) -> None:
        self._us_ids, self._them_ids = sorted(truth.us), sorted(truth.them)
        self._frame(truth)
        self._cmd.append([[-1.0] + [0.0] * (CMD_COLS - 1) for _ in self._us_ids])
        self._terms.append([0.0] * len(self.term_names))

    def on_tick(self, tr: Transition, terms: dict[str, float]) -> None:
        self._frame(tr.world)
        by_id: dict[int, RobotCommand] = {c.robot_id: c for c in tr.commands}
        rows = []
        for i in self._us_ids:
            c = by_id.get(i)
            rows.append([-1.0] + [0.0] * (CMD_COLS - 1) if c is None else
                        [float(c.robot_id), c.vx, c.vy, c.vtheta, c.kick_speed,
                         float(c.chip), c.dribbler])
        self._cmd.append(rows)
        self._terms.append([terms.get(n, 0.0) for n in self.term_names])
        if tr.done is not None:
            self._done = tr.done.name

    def finish(self) -> Rollout:
        n_us, n_them = len(self._us_ids), len(self._them_ids)
        T = len(self._t)
        return Rollout(
            t=np.asarray(self._t, np.float64),
            ball=np.asarray(self._ball, np.float32).reshape(T, 6),
            us=np.asarray(self._us, np.float32).reshape(T, n_us, ROBOT_COLS),
            them=np.asarray(self._them, np.float32).reshape(T, n_them, ROBOT_COLS),
            us_ids=np.asarray(self._us_ids, np.int16),
            them_ids=np.asarray(self._them_ids, np.int16),
            commands=np.asarray(self._cmd, np.float32).reshape(T, n_us, CMD_COLS),
            term_names=self.term_names,
            terms=np.asarray(self._terms, np.float32).reshape(T, len(self.term_names)),
            done=self._done, meta=self.meta)


def save_rollout(path: str | Path, r: Rollout) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, t=r.t, ball=r.ball, us=r.us, them=r.them, us_ids=r.us_ids,
                        them_ids=r.them_ids, commands=r.commands,
                        term_names=np.asarray(r.term_names, dtype=str), terms=r.terms,
                        done=np.asarray(r.done), meta=np.asarray(json.dumps(r.meta)))
    tmp.replace(path)
    return path


def load_rollout(path: str | Path) -> Rollout:
    with np.load(Path(path), allow_pickle=False) as z:
        return Rollout(t=z["t"], ball=z["ball"], us=z["us"], them=z["them"],
                       us_ids=z["us_ids"], them_ids=z["them_ids"], commands=z["commands"],
                       term_names=[str(n) for n in z["term_names"]], terms=z["terms"],
                       done=str(z["done"]), meta=json.loads(str(z["meta"])))


def prune(directory: str | Path, keep: int) -> list[Path]:
    """Delete all but the newest `keep` rollouts in `directory`. Returns the
    deleted paths."""
    files = sorted(Path(directory).glob("*.npz"), key=lambda p: (p.stat().st_mtime_ns, p.name))
    doomed = files[:-keep] if keep > 0 else files
    for p in doomed:
        p.unlink()
    return doomed
