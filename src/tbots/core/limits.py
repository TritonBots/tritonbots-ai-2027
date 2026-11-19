"""How fast our robots may change speed. Meters, radians, seconds.

The action codec (rl/codec.py) rate-limits every learned command to these,
so the simulator only ever executes what a motor can follow (ARCHITECTURE
section 9, "Action codec"). They are recorded in every checkpoint.

NEITHER VALUE IS MEASURED ON OUR HARDWARE. They are an open problem
(docs/TASKS.md section 6). What they are based on:

- Linear: rSim itself ramps a robot under a step velocity command at about
  3.5 m/s^2 (measured 2026-11-19: 0 -> 2 m/s in ~0.55 s). 3.0 sits just
  under that, so the codec, not rSim's internal limit, is the binding
  constraint, and the value is a plausible small-size-league figure.
- Angular: rSim reaches a commanded spin in about 50 ms (~100 rad/s^2),
  which no real motor does. 20 rad/s^2 reaches 6 rad/s in 0.3 s.

Change them only with a measurement, and expect a change to make old
checkpoints refuse to load (TASK-059 records them).
"""

from __future__ import annotations

MAX_LINEAR_ACCEL: float = 3.0       # m/s^2, NOT MEASURED on hardware
MAX_ANGULAR_ACCEL: float = 20.0     # rad/s^2, NOT MEASURED on hardware
