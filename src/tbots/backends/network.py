"""Match backend: the ER-Force simulator, or real robots.

Commands go out over the SSL simulation protocol (or the radio).
Observations come in over SSL-Vision multicast.
Referee state comes in over the game-controller multicast.

This backend is realtime and lossy. It is for evaluation and match play,
never for training.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace

from tbots.backends.base import Backend, Scenario
from tbots.core.command import RobotCommand
from tbots.core.geometry import DIV_B, FieldGeometry
from tbots.core.perspective import Perspective
from tbots.core.state import WorldState
from tbots.net.referee import RefereeReceiver
from tbots.net.robot_control import RobotControlSender
from tbots.net.sim_control import SimControlSender
from tbots.net.vision import VisionReceiver
from tbots.perception.tracker import Tracker


class NetworkBackend(Backend):
    def __init__(
        self,
        *,
        team_name: str,
        vision: VisionReceiver,
        referee: RefereeReceiver,
        control: RobotControlSender,
        sim_control: SimControlSender | None = None,
        geometry: FieldGeometry = DIV_B,
        dt: float = 1.0 / 60.0,
        actuation_delay: float = 0.0,
    ) -> None:
        """`actuation_delay` is how far past "now" the tracker predicts: the
        time from a command leaving us to the robot acting on it. Unmeasured
        for real robots (docs/TASKS.md §6), so 0 until someone measures it."""
        self._team_name = team_name
        self._vision = vision
        self._referee = referee
        self._control = control
        self._sim_control = sim_control
        self._geom = geometry
        self._dt = dt
        self._tracker = Tracker(geometry, actuation_delay=actuation_delay)

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def geometry(self) -> FieldGeometry:
        return self._geom

    def reset(self, scenario: Scenario) -> WorldState:
        # TODO(setup): teleport ball and robots via SimulatorCommand.
        # Only possible against a simulator — raise if sim_control is None,
        # because at a real match there is nothing to teleport.
        if self._sim_control is None:
            raise RuntimeError("reset() requires simulation control; "
                               "not available against real robots")
        self._sim_control.place(scenario, self.perspective)
        return self.observe()

    def step(self, commands: Sequence[RobotCommand]) -> WorldState:
        # The port follows our colour, which the referee can change at half
        # time (SETUP_LOG TASK-013), so it is re-read on every packet.
        self._referee.poll()
        self._control.send(commands, self._require_perspective())
        self._vision.wait_for_next_frame(timeout=0.05)
        return self.observe()

    @property
    def perspective(self) -> Perspective:
        """Rule 3: which colour we are, and which way round the field is.

        Resolved by RefereeReceiver and simply read here. This used to be a
        second derivation of the same two facts from the same message
        (TASK-013); there is now exactly one, and half time needs no special
        handling because the referee tells us the moment it swaps.
        """
        return self._referee.perspective

    def _require_perspective(self) -> Perspective:
        p = self._referee.perspective
        if not p.resolved:
            raise RuntimeError(
                f"the referee has not named {self._team_name!r} as blue or "
                "yellow, so we do not know which robots are ours. Set the team "
                "name in the game controller (case-sensitive) and retry.")
        return p

    def _check_geometry(self) -> None:
        """Fail on a Division mismatch, keyed on scalars every source sends.

        Not the penalty-area scalars: the ER-Force simulator leaves them 0
        (SETUP_LOG TASK-071), and so would fail every simulated match.
        """
        wire = self._vision.geometry
        if wire is None:
            return
        want = self._geom
        for name, got, exp in (("length", wire.length, want.length),
                               ("width", wire.width, want.width),
                               ("goal_width", wire.goal_width, want.goal_width)):
            if abs(got - exp) > 0.01:
                raise RuntimeError(
                    f"vision reports field {name} {got:.3f} m, configured "
                    f"geometry says {exp:.3f} m: this is the wrong Division "
                    f"(or the wrong vision source)")

    def observe(self) -> WorldState:
        """Vision through the tracker, referee attached, our frame (Rule 3).

        The colour/side normalisation happens once, inside the tracker, on
        the finished frame.
        """
        game = self._referee.poll()
        perspective = self._require_perspective()
        self._check_geometry()
        world = self._tracker.update(self._vision.frames(), time.time(), perspective)

        # Dribbler contact is the only feedback the simulation protocol has.
        # Telemetry is ours only; a robot not on the field is not attached.
        telemetry = {i: fb for i, fb in self._control.feedback().items() if i in world.us}
        us = {i: replace(r, has_ball=telemetry[i].has_ball) if i in telemetry else r
              for i, r in world.us.items()}
        return replace(world, us=us, game=game, telemetry=telemetry)

    def close(self) -> None:
        self._vision.close()
        self._referee.close()
        self._control.close()
