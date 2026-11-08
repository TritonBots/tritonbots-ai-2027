# rSim facts — VERIFIED, do not guess

Verified on: 2026-09-07 (first verified 2026-08-10)
rSim commit: `56d02532f406987a9b9fee59b12da9b33468af1e` (our fork, YashTandon05/rSim)
Python 3.11.15, ODE 0.16.2 double-precision from `/usr/local/lib/libode.so.8`

Regenerate with `python scripts/verify_rsim.py` after **any** rSim fork update.

---

## The four answers

- **Division B (9.0 x 6.0 m) is `field_type = 1`.**
  Not 0. `field_type=0` is Division A (12.0 x 9.0), `field_type=2` is a
  6.0 x 4.0 field. The rSim README is right and rSoccer's README is wrong.

- **`BALL_STRIDE = 5`, `ROBOT_STRIDE = 11`.**
  `len(get_state()) == 5 + 11 * (n_blue + n_yellow)` — 137 for 6v6.
  Ball slice: `[x, y, z, vx, vy]`.
  Robot slice: `[x, y, dir, vx, vy, vdir, is_touching_ball, w0, w1, w2, w3]`.

- **Action vector length = 8** (per robot; indices 0..7 are all read).
  `[0]` use-wheels flag — if `> 0`, `[1..4]` are the four wheel speeds;
  otherwise `[1],[2],[3]` are local `vx, vy, vangular`.
  `[5]` flat kick speed, `[6]` chip kick speed, `[7]` dribbler on/off.

- **Angles are in DEGREES on the way out, RADIANS on the way in.**
  Reported headings, reset/constructor poses, and `vdir` are all **degrees**
  (`vdir` in deg/s). But the **commanded** angular velocity in the action
  vector is **radians/second**. The units are asymmetric.

---

## Four traps that cost real time

**1. `field_type = 0` is Division A, not Division B.** `docs/SETUP.md`
Step 9.2 and Step 14.3's `configs/env/div_b_6v6.yaml` have been corrected to
`field_type = 1` following this finding — see `docs/SETUP_LOG.md` Step 6 for
the correction. Left as the upstream-README-implied `0`, we would have
silently trained Division B policies on a 12 x 9 m Division A pitch. Nothing
errors; the goals are just in the wrong place. SETUP.md's original probe also
hardcoded `field_type=0` in PARTS 2 and 3, so it was measuring Division A
too — Step 6's script has been rewritten to discover the value in PART 1 and
carry it forward instead of hardcoding it.

**2. The action vector is 8, and a wrong length does not raise.**
`SSLWorld::setActions()` reads `rbtAction[0..7]` with `std::vector::operator[]`,
which performs no bounds checking. Pass a 6-element action and C++ reads two
elements past the end — undefined behaviour, silently interpreted as kick and
dribbler commands. It does not throw. Any probe that concludes "length 6 was
accepted, so 6 is correct" is reading garbage memory. The length comes from
the source, not from the absence of an exception. `docs/SETUP.md` Step 9.2
has been corrected to `ACTION_LEN: int = 8` with offsets
`A_KICK_FLAT, A_KICK_CHIP, A_DRIBBLER = 5, 6, 7` (the original collapsed
`4, 5, 5` would have fired the kicker on random data).

**3. Angle units are asymmetric.** Read degrees, write radians. Concretely,
in `backends/rsim.py`: convert `deg -> rad` on `dir` and `vdir` coming out of
`get_state()`, convert `rad -> deg` on the pose angles going into
`reset()`/the constructor — but pass `RobotCommand.vtheta` straight through
to action slot `[3]` with **no conversion**, because that slot is already
rad/s. Verified: commanding `vangular = 3.0` yields 162.85 deg/s = 2.84 rad/s.
Had it been deg/s we would have measured ~3 deg/s.

**4. `get_state()` must be called exactly once per `step()`.**
Velocities are not read from ODE. `getState()` finite-differences the current
positions against those captured at the *previous `get_state()` call*, and
always divides by exactly one `timeStep` — never by the time actually
elapsed. So:
  - two `get_state()` calls with no `step()` between them → all velocities `0`
  - one `get_state()` after N steps → velocities N times too large

A robot genuinely moving at 1.0 m/s reports `vx = 9.9979` if you call
`get_state()` once every 10 steps. The first call after construction or
`reset()` always reports zero velocity, because there is no previous state to
difference against. `RSimBackend._observe()` satisfies this as long as it is
called exactly once per `step()` — do not add extra `get_state()` calls for
logging or rendering.

**5. The timestep is a float in SECONDS — but an int still means
milliseconds.** FIXED 2026-09-07 (TASK-070); read this before you write
`SSL(...)` by hand.

Upstream's binding was `SSL(..., timeStep_ms: int, ...)` and divided by 1000.0
before handing the value to `SSLWorld`, which has always taken seconds as a
double. A 60 Hz caller therefore had to pass `int(round(1/60 * 1000))` = 17,
and physics advanced 17 ms per step — 58.8 Hz — while `RSimBackend._t`
advanced 1/60 s. Nothing errored and nothing looked wrong: `getState()`
divides its finite-differenced velocities by that same 0.017, so they read
back correct. Only simulated time drifted, 2% fast, silently. Measured: a
robot holding a commanded 1.0 m/s covered **1.02000 m** per 60 ticks.

Our fork adds a `double`-seconds constructor. The overload is chosen by the
argument's **type**:

| You pass | Means | 60 Hz |
|---|---|---|
| `1.0 / 60.0` (float) | seconds, used exactly | ✅ |
| `17` (int) | milliseconds, as before | ❌ 58.8 Hz |
| `17.0` (float) | **17 seconds**, and it will be honoured | ❌ |

The same 60 ticks now cover **1.00000 m**. `RSimBackend` passes
`float(self._dt)`; the `float()` is load-bearing, not decoration. The int
overload is kept so rSoccer, which passes ints, is unaffected.

`SSL.time_step` reads the value back, always in seconds, whichever
constructor was used — assert against it rather than recomputing.

## Smaller quirks

- A heading of exactly 0 is reported as **360.0**, not 0.0. `getDir()` ends
  with `(y > 0) ? absAng : 360 - absAng`, so the range is `(0, 360]`.
  `wrap_angle(deg_to_rad(360.0))` gives 0.0, so the normal conversion path
  handles it — but raw comparisons against 0 will surprise you.
- `step()` internally substeps 5 times at `timeStep * 0.2`.
- `reset()` tears down and reconstructs the whole `SSLWorld`; it is not cheap,
  and it resets the velocity-differencing baseline.
- **`reset()` also costs a rolling ball a fixed ~0.09 m/s.** The `SSLWorld`
  constructor ends with `for (i = 0; i < 30) physics->step(timeStep * 0.1)` —
  a settling loop worth three whole timesteps — and the ball rolls against
  friction throughout it. Measured 2026-09-07: the loss is 0.091 m/s at 1, 2
  and 4 m/s alike, so it is an absolute penalty and not a percentage —
  24% of a 0.5 m/s roll, 3% of a 4 m/s one. Positions are unaffected to
  within a micrometre. This matters because `RSimBackend.place()` is built on
  `reset()`, so every teleport pays it; see `backends/rsim.py`.
- **`reset()` takes robot POSES only** — `[x, y, dir]`, no velocity — and
  `ballPos` is `[x, y, vx, vy]`, with no `z` or `vz`. So a reset stops every
  robot dead, and drops a chipped ball to the ground. Neither is expressible
  through this API.
- Field params available from `get_field_params()`: `length`, `width`,
  `penalty_length`, `penalty_width`, `goal_width`, `goal_depth`,
  `ball_radius`, `rbt_radius`, `rbt_wheel_radius`, `rbt_motor_max_rpm`,
  `rbt_distance_center_kicker`, `rbt_kicker_thickness`, `rbt_kicker_width`,
  `rbt_wheel0_angle` .. `rbt_wheel3_angle`.
- rSim's Division B geometry agrees with `core/geometry.py`'s `DIV_B`
  (goal_width 1.0, penalty 1.0 x 2.0) and its Division A with `DIV_A`
  (goal_width 1.8, penalty 1.8 x 3.6).
- **A rolling ball slows by about 0.7 m/s every second, after a short,
  much faster initial loss.** Measured 2026-11-07 through `RSimBackend` at
  60 Hz, ball placed with velocity on an empty Division B field:
  | Placed at | after 0.5 s | after 1 s | after 2 s |
  |---|---|---|---|
  | 1.0 m/s | 0.37 | 0.02 | 0 |
  | 2.0 m/s | 1.08 | 0.73 | 0.03 |
  | 4.0 m/s | 2.51 | 2.16 | 1.46 |
  Between 0.5 s and 2 s every row loses ~0.7 m/s per second, so the rolling
  phase is close to constant deceleration, not rcssserver's per-cycle
  multiplicative decay. The first fraction of a second loses far more, and
  so does a kick: a 4.0 m/s flat kick left the robot at 3.90 m/s and was at
  2.73 m/s ten ticks (1/6 s) later. `skills/goalkeep.py` models the whole
  trajectory at 0.7 m/s², which overestimates how far a shot travels -- the
  safe direction for a keeper deciding whether to block.
- **A flat kick fires on contact.** Holding `kick_speed > 0` while driving
  into the ball releases it the tick the ball touches the kicker, at about
  the commanded speed (4.0 commanded, 3.90 measured). The dribbler sets
  `is_touching_ball`, which `RSimBackend` reports as `has_ball`.

---

## Constructor signature (from the pybind11 binding, not assumed)

Two overloads, differing only in the timestep. Use the first.

```
SSL(fieldType: int, nRobotsBlue: int, nRobotsYellow: int,
    timeStep_s: float,               # SECONDS. 1.0/60.0 for 60 Hz.
    ballPos: list[float],            # [x, y, vx, vy]
    blueRobotsPos: list[list[float]],   # [[x, y, dir_degrees], ...]
    yellowRobotsPos: list[list[float]]) # [[x, y, dir_degrees], ...]

SSL(fieldType: int, nRobotsBlue: int, nRobotsYellow: int,
    timeStep_ms: int,                # MILLISECONDS. Legacy; for rSoccer.
    ballPos, blueRobotsPos, yellowRobotsPos)
```

Arguments are **named** as of our fork (`fieldType`, `nRobotsBlue`,
`nRobotsYellow`, `timeStep_s` / `timeStep_ms`, `ballPos`, `blueRobotsPos`,
`yellowRobotsPos`); upstream declared none and was positional-only. Both
timestep arguments are `.noconvert()`, so the int/float distinction holds no
matter what the other six arguments look like.

Methods: `step(actions)`, `get_state()`, `reset(ballPos, bluePos, yellowPos)`,
`get_field_params()`. Property: `time_step` — simulated seconds per `step()`,
always in seconds.

`reset()` reuses the timestep the constructor was given, at full precision;
it does not requantise.

---

<raw script output follows>

```
======================================================================
PART 1 - field type mapping
======================================================================
Division B is 9.0 x 6.0 m. Division A is 12.0 x 9.0 m.

field_type=0  length=12.0  width=9.0  goal_width=1.8  penalty 1.8x3.6   <-- Division A
field_type=1  length=9.0  width=6.0  goal_width=1.0  penalty 1.0x2.0   <-- DIVISION B, this is OUR value
field_type=2  length=6.0  width=4.0  goal_width=0.7  penalty 0.8x2.0

ANSWER: Division B is field_type = 1
This MUST match FIELD_TYPE_DIV_B in backends/rsim.py (Step 9.2)
and field_type in configs/env/div_b_6v6.yaml (Step 14.3).

======================================================================
PART 2 - state array stride
======================================================================
Source: SSLWorld::getState() in src/robosim/sslworld.cpp indexes the
previous state as lastState[5 + (11 * i) + k] -- the strides are
written literally into the C++. Confirming that against the array:

len(get_state())  = 137
n_robots          = 12
  if BALL_STRIDE=5 -> ROBOT_STRIDE=11

Blue robots were placed at x = -0.5, -0.7, -0.9, ... y = 0, theta = 0.
Reading x back at BALL_STRIDE + ROBOT_STRIDE * i:
  robot 0: x=-0.5000  y=+0.0000  dir= 360.000   (expected x=-0.5, y=+0.0)
  robot 1: x=-0.7000  y=+0.0000  dir= 360.000   (expected x=-0.7, y=+0.0)
  robot 2: x=-0.9000  y=+0.0000  dir= 360.000   (expected x=-0.9, y=+0.0)
  robot 3: x=-1.1000  y=+0.0000  dir= 360.000   (expected x=-1.1, y=+0.0)
  robot 4: x=-1.3000  y=+0.0000  dir= 360.000   (expected x=-1.3, y=+0.0)
  robot 5: x=-1.5000  y=+0.0000  dir= 360.000   (expected x=-1.5, y=+0.0)

ANSWER: BALL_STRIDE = 5, ROBOT_STRIDE = 11
Ball slice  : [x, y, z, vx, vy]
Robot slice : [x, y, dir, vx, vy, vdir, is_touching_ball, w0, w1, w2, w3]

======================================================================
PART 3 - action vector length
======================================================================
Source: SSLWorld::setActions() in src/robosim/sslworld.cpp reads
rbtAction[0] through rbtAction[7] -- eight slots:
  [0]        use-wheels flag; >0 = treat [1..4] as wheel speeds,
             otherwise [1],[2],[3] are local vx, vy, vangular
  [1][2][3]  local vx, vy, vangular   (or wheels 0-2 if [0] > 0)
  [4]        wheel 3                  (only read when [0] > 0)
  [5]        kick speed, flat
  [6]        kick speed, chip
  [7]        dribbler on/off

WARNING: the naive 'does a short action vector raise?' probe is
meaningless here. std::vector::operator[] performs NO bounds check, so
a 6-element action silently reads garbage for [6] and [7] instead of
raising. Length is therefore established from the source, not caught
by an exception. Demonstrating that below:

  action length 6 -> no exception  <-- NOT actually safe: read out of bounds
  action length 7 -> no exception  <-- NOT actually safe: read out of bounds
  action length 8 -> no exception

ANSWER: action vector length = 8 (indices 0..7 are all read)

======================================================================
PART 4 - angle units
======================================================================
Source says degrees for reported heading:
  SSLRobot::getDir()  returns acos(...) * (180.0f / M_PI), mapped via
                      `(y > 0) ? absAng : 360 - absAng` -> range (0, 360]
  SSLRobot::setDir()  does `ang *= M_PI / 180.0f` -> reset poses are degrees
  smallestAngleDiff() compares in degrees -> state vdir is degrees/second
But the ACTION side is radians:
  setDesiredSpeedLocal(vx, vy, vw) computes (robotRadius * vw) and adds
  it to m/s terms, so vw must be rad/s for the units to balance.

4a. Place robots at known headings, read the reported heading back:
      placed    0.0 -> reported  360.000
      placed   45.0 -> reported   45.000
      placed   90.0 -> reported   90.000
      placed  135.0 -> reported  135.000
      placed  180.0 -> reported  180.000
      placed  270.0 -> reported  270.000
    1:1 in degrees. (Note 0.0 comes back as 360.0 -- getDir()'s
    `(y > 0) ? absAng : 360 - absAng` makes the range (0, 360], not
    [0, 360). A heading of exactly zero reports as 360.)

4b. Command vangular = 3.0 and measure the achieved rate:
      heading 323.172 -> 44.596 over 0.50s
      achieved = 162.85 deg/s = 2.842 rad/s
      commanded 3.0 -> got 2.842 rad/s, not 3 deg/s.

ANSWER: state angles and reset poses are DEGREES.
        Commanded vangular is RADIANS/second. This is asymmetric --
        it is the single easiest thing to get wrong in backends/rsim.py.

======================================================================
PART 5 - velocity fields are differenced per get_state() CALL
======================================================================
getState() derives vx/vy/vdir by differencing against the state captured
at the PREVIOUS getState() call, and always divides by exactly one
timeStep -- never by the time actually elapsed. Consequences:

  60 steps then the FIRST get_state()  -> vx =  0.0000   (no previous state to difference against)
  10 more steps, then get_state()      -> vx =  9.9979   (~10x too big: 10 steps of travel / 1 timestep)
  get_state() again, 0 steps between   -> vx =  0.0000   (nothing moved)

ANSWER: call get_state() EXACTLY ONCE PER step() or every velocity in
        the state array is wrong. Robot commanded at 1.0 m/s reads back
        as ~10 m/s above purely from calling get_state() too rarely.

======================================================================
PART 6 - the timestep is exact, and it is a float
======================================================================
Upstream's binding took an integer number of milliseconds. 1/60 s is
16.667 ms, so a 60 Hz caller passed 17 and got 58.8 Hz -- silently, because
getState() divides its velocities by the same wrong number and they
read back correct. Only simulated time drifts. Our fork adds a float-
seconds overload; this is what it does.

  SSL(..., 0.016666666666666666, ...)   -> time_step = 0.016666666666666666
  SSL(..., 17, ...)                        -> time_step = 0.017   (int = milliseconds)

Travel at a commanded 1.0 m/s, measured from steady state so the motor
ramp is not in the number:
  60 ticks of travel = 1.00000 m   (1.02000 before the fix)

ANSWER: pass dt as a float in SECONDS. int is still milliseconds.

======================================================================
SUMMARY
======================================================================
  Division B field_type   = 1
  BALL_STRIDE             = 5
  ROBOT_STRIDE            = 11
  action vector length    = 8
  state angles / poses    = degrees, heading in (0, 360], vdir in deg/s
  commanded vangular      = radians/second   <-- asymmetric, mind this
  get_state()             = must be called exactly once per step()
  timestep                = float seconds; 0.016666666666666666 s is exact 60 Hz
```
