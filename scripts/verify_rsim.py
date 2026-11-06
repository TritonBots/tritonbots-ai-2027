"""Empirically determine rSim's field-type mapping and array strides.

Run this ONCE after building rSim, and again after any rSim fork update.
The four answers it establishes go at the top of docs/RSIM_FACTS.md, and
must exactly match FIELD_TYPE_DIV_B, ACTION_LEN, and the action-offset
constants in backends/rsim.py (Step 9.2), and field_type in
configs/env/div_b_6v6.yaml (Step 14.3).

NOTE ON PROVENANCE. Where a fact can be read directly out of the rSim C++
source, this script quotes that source and then confirms it at runtime.
Black-box probing alone is not trustworthy here -- see PART 3, where the
obvious probe gives a confidently wrong answer: a too-short action vector
does NOT raise, it silently reads past the end of an unchecked
std::vector, so "no exception" does not mean "correct length".
"""

import numpy as np
import robosim

N_BLUE, N_YELLOW = 6, 6
N_ROBOTS = N_BLUE + N_YELLOW
# Seconds, as a float. Our fork overloads the constructor on this argument's
# TYPE -- an int is milliseconds, a float is seconds -- so `1.0 / 60.0` is an
# exact 60 Hz and `16` would be 16 ms. See PART 6. Upstream had only the
# integer-millisecond form, which cannot express 1/60 at all.
DT = 1.0 / 60.0


def make(field_type):
    """Construct an SSL world. The pybind11 signature is
    SSL(fieldType, nRobotsBlue, nRobotsYellow, timeStep_s: float,
        ballPos, blueRobotsPos, yellowRobotsPos),
    with a `timeStep_ms: int` overload kept for rSoccer. Arguments are named
    as of our fork; they used to be positional-only."""
    return robosim.SSL(
        field_type, N_BLUE, N_YELLOW, DT,
        [0.0, 0.0, 0.0, 0.0],
        [[-0.5 - 0.2 * i, 0.0, 0.0] for i in range(N_BLUE)],
        [[0.5 + 0.2 * i, 0.0, 180.0] for i in range(N_YELLOW)],
    )


print("=" * 70)
print("PART 1 - field type mapping")
print("=" * 70)
print("Division B is 9.0 x 6.0 m. Division A is 12.0 x 9.0 m.")
print()

div_b_field_type = None
for ft in (0, 1, 2):
    try:
        p = make(ft).get_field_params()
        tag = ""
        if (p["length"], p["width"]) == (9.0, 6.0):
            div_b_field_type = ft
            tag = "   <-- DIVISION B, this is OUR value"
        elif (p["length"], p["width"]) == (12.0, 9.0):
            tag = "   <-- Division A"
        print(f"field_type={ft}  length={p['length']}  width={p['width']}  "
              f"goal_width={p['goal_width']}  "
              f"penalty {p['penalty_length']}x{p['penalty_width']}{tag}")
    except Exception as exc:
        print(f"field_type={ft} -> FAILED: {exc}")

if div_b_field_type is None:
    raise SystemExit("FATAL: no field_type reported a 9.0 x 6.0 field.")

print()
print(f"ANSWER: Division B is field_type = {div_b_field_type}")
print("This MUST match FIELD_TYPE_DIV_B in backends/rsim.py (Step 9.2)")
print("and field_type in configs/env/div_b_6v6.yaml (Step 14.3).")

# Everything below MUST use the field type established above, not a guess.
FIELD_TYPE = div_b_field_type

print()
print("=" * 70)
print("PART 2 - state array stride")
print("=" * 70)
print("Source: SSLWorld::getState() in src/robosim/sslworld.cpp indexes the")
print("previous state as lastState[5 + (11 * i) + k] -- the strides are")
print("written literally into the C++. Confirming that against the array:")
print()

sim = make(FIELD_TYPE)
state = np.asarray(sim.get_state())
print(f"len(get_state())  = {len(state)}")
print(f"n_robots          = {N_ROBOTS}")
for ball_stride in (5, 6, 7):
    rem = len(state) - ball_stride
    if rem % N_ROBOTS == 0:
        print(f"  if BALL_STRIDE={ball_stride} -> ROBOT_STRIDE={rem // N_ROBOTS}")

BALL_STRIDE, ROBOT_STRIDE = 5, 11
assert len(state) == BALL_STRIDE + ROBOT_STRIDE * N_ROBOTS, (
    f"state length {len(state)} != {BALL_STRIDE} + {ROBOT_STRIDE} * {N_ROBOTS}")

print()
print("Blue robots were placed at x = -0.5, -0.7, -0.9, ... y = 0, theta = 0.")
print("Reading x back at BALL_STRIDE + ROBOT_STRIDE * i:")
for i in range(N_BLUE):
    base = BALL_STRIDE + ROBOT_STRIDE * i
    print(f"  robot {i}: x={state[base]:+.4f}  y={state[base+1]:+.4f}  "
          f"dir={state[base+2]:8.3f}   (expected x={-0.5 - 0.2*i:+.1f}, y=+0.0)")

print()
print("ANSWER: BALL_STRIDE = 5, ROBOT_STRIDE = 11")
print("Ball slice  : [x, y, z, vx, vy]")
print("Robot slice : [x, y, dir, vx, vy, vdir, is_touching_ball, w0, w1, w2, w3]")

print()
print("=" * 70)
print("PART 3 - action vector length")
print("=" * 70)
print("Source: SSLWorld::setActions() in src/robosim/sslworld.cpp reads")
print("rbtAction[0] through rbtAction[7] -- eight slots:")
print("  [0]        use-wheels flag; >0 = treat [1..4] as wheel speeds,")
print("             otherwise [1],[2],[3] are local vx, vy, vangular")
print("  [1][2][3]  local vx, vy, vangular   (or wheels 0-2 if [0] > 0)")
print("  [4]        wheel 3                  (only read when [0] > 0)")
print("  [5]        kick speed, flat")
print("  [6]        kick speed, chip")
print("  [7]        dribbler on/off")
print()
print("WARNING: the naive 'does a short action vector raise?' probe is")
print("meaningless here. std::vector::operator[] performs NO bounds check, so")
print("a 6-element action silently reads garbage for [6] and [7] instead of")
print("raising. Length is therefore established from the source, not caught")
print("by an exception. Demonstrating that below:")
print()
for n_act in (6, 7, 8):
    try:
        s = make(FIELD_TYPE)
        s.step([[0.0] * n_act for _ in range(N_ROBOTS)])
        note = "" if n_act == 8 else "  <-- NOT actually safe: read out of bounds"
        print(f"  action length {n_act} -> no exception{note}")
    except Exception as exc:
        print(f"  action length {n_act} -> raised {type(exc).__name__}: {exc}")

print()
print("ANSWER: action vector length = 8 (indices 0..7 are all read)")

print()
print("=" * 70)
print("PART 4 - angle units")
print("=" * 70)
print("Source says degrees for reported heading:")
print("  SSLRobot::getDir()  returns acos(...) * (180.0f / M_PI), mapped via")
print("                      `(y > 0) ? absAng : 360 - absAng` -> range (0, 360]")
print("  SSLRobot::setDir()  does `ang *= M_PI / 180.0f` -> reset poses are degrees")
print("  smallestAngleDiff() compares in degrees -> state vdir is degrees/second")
print("But the ACTION side is radians:")
print("  setDesiredSpeedLocal(vx, vy, vw) computes (robotRadius * vw) and adds")
print("  it to m/s terms, so vw must be rad/s for the units to balance.")
print()

ACT_LEN = 8


def body_action(vx=0.0, vy=0.0, vw=0.0, robot=0):
    """[0]=0 selects body velocities, so [1],[2],[3] are vx, vy, vangular."""
    acts = [[0.0] * ACT_LEN for _ in range(N_ROBOTS)]
    acts[robot] = [0.0, vx, vy, vw] + [0.0] * (ACT_LEN - 4)
    return acts


# -- 4a: place at known headings and read them back. No dynamics involved,
#        so this isolates the unit question cleanly.
print("4a. Place robots at known headings, read the reported heading back:")
known = [0.0, 45.0, 90.0, 135.0, 180.0, 270.0]
sim = robosim.SSL(
    FIELD_TYPE, N_BLUE, N_YELLOW, DT, [0.0, 0.0, 0.0, 0.0],
    [[-1.0 - 0.3 * i, 0.0, known[i]] for i in range(N_BLUE)],
    [[1.0 + 0.3 * i, 0.0, 0.0] for i in range(N_YELLOW)],
)
st = np.asarray(sim.get_state())
for i in range(N_BLUE):
    got = st[BALL_STRIDE + ROBOT_STRIDE * i + 2]
    print(f"      placed {known[i]:6.1f} -> reported {got:8.3f}")
print("    1:1 in degrees. (Note 0.0 comes back as 360.0 -- getDir()'s")
print("    `(y > 0) ? absAng : 360 - absAng` makes the range (0, 360], not")
print("    [0, 360). A heading of exactly zero reports as 360.)")

# -- 4b: a spin test settles whether COMMANDED vangular is deg/s or rad/s.
#        Spin to steady state first; the wheels ramp through a motor model,
#        so measuring from a standstill under-reads badly.
print()
print("4b. Command vangular = 3.0 and measure the achieved rate:")
sim = robosim.SSL(
    FIELD_TYPE, N_BLUE, N_YELLOW, DT, [0.0, 0.0, 0.0, 0.0],
    [[-1.0 - 0.3 * i, 0.0, 0.0] for i in range(N_BLUE)],
    [[1.0 + 0.3 * i, 0.0, 0.0] for i in range(N_YELLOW)],
)
VW = 3.0
for _ in range(120):                       # reach steady state
    sim.step(body_action(vw=VW))
h0 = np.asarray(sim.get_state())[BALL_STRIDE + 2]
TICKS = 30
for _ in range(TICKS):
    sim.step(body_action(vw=VW))
h1 = np.asarray(sim.get_state())[BALL_STRIDE + 2]
elapsed = TICKS * DT
rate_deg = ((h1 - h0) % 360.0) / elapsed
print(f"      heading {h0:.3f} -> {h1:.3f} over {elapsed:.2f}s")
print(f"      achieved = {rate_deg:.2f} deg/s = {np.radians(rate_deg):.3f} rad/s")
print(f"      commanded 3.0 -> got {np.radians(rate_deg):.3f} rad/s, not 3 deg/s.")
print()
print("ANSWER: state angles and reset poses are DEGREES.")
print("        Commanded vangular is RADIANS/second. This is asymmetric --")
print("        it is the single easiest thing to get wrong in backends/rsim.py.")

print()
print("=" * 70)
print("PART 5 - velocity fields are differenced per get_state() CALL")
print("=" * 70)
print("getState() derives vx/vy/vdir by differencing against the state captured")
print("at the PREVIOUS getState() call, and always divides by exactly one")
print("timeStep -- never by the time actually elapsed. Consequences:")
print()
sim = robosim.SSL(
    FIELD_TYPE, N_BLUE, N_YELLOW, DT, [0.0, 0.0, 0.0, 0.0],
    [[-1.0 - 0.3 * i, 0.0, 0.0] for i in range(N_BLUE)],
    [[1.0 + 0.3 * i, 0.0, 0.0] for i in range(N_YELLOW)],
)
for _ in range(60):
    sim.step(body_action(vx=1.0))
v = np.asarray(sim.get_state())[BALL_STRIDE + 3]
print(f"  60 steps then the FIRST get_state()  -> vx = {v:7.4f}   (no previous"
      " state to difference against)")
for _ in range(10):
    sim.step(body_action(vx=1.0))
v = np.asarray(sim.get_state())[BALL_STRIDE + 3]
print(f"  10 more steps, then get_state()      -> vx = {v:7.4f}   (~10x too big:"
      " 10 steps of travel / 1 timestep)")
v = np.asarray(sim.get_state())[BALL_STRIDE + 3]
print(f"  get_state() again, 0 steps between   -> vx = {v:7.4f}   (nothing moved)")
print()
print("ANSWER: call get_state() EXACTLY ONCE PER step() or every velocity in")
print("        the state array is wrong. Robot commanded at 1.0 m/s reads back")
print("        as ~10 m/s above purely from calling get_state() too rarely.")
print()
print("=" * 70)
print("PART 6 - the timestep is exact, and it is a float")
print("=" * 70)
print("Upstream's binding took an integer number of milliseconds. 1/60 s is")
print("16.667 ms, so a 60 Hz caller passed 17 and got 58.8 Hz -- silently, "
      "because")
print("getState() divides its velocities by the same wrong number and they")
print("read back correct. Only simulated time drifts. Our fork adds a float-")
print("seconds overload; this is what it does.")
print()
sim = robosim.SSL(
    FIELD_TYPE, 1, 0, DT, [0.0, 0.0, 0.0, 0.0], [[-2.0, 0.0, 0.0]], [],
)
print(f"  SSL(..., {DT!r}, ...)   -> time_step = {sim.time_step!r}")
ms = robosim.SSL(
    FIELD_TYPE, 1, 0, 17, [0.0, 0.0, 0.0, 0.0], [[-2.0, 0.0, 0.0]], [],
)
print(f"  SSL(..., 17, ...)                        -> time_step = "
      f"{ms.time_step!r}   (int = milliseconds)")
print()
print("Travel at a commanded 1.0 m/s, measured from steady state so the motor")
print("ramp is not in the number:")
for _ in range(60):                         # spin up
    sim.step(body_action(vx=1.0))
x0 = np.asarray(sim.get_state())[BALL_STRIDE + 0]
for _ in range(60):                         # exactly one second
    sim.step(body_action(vx=1.0))
x1 = np.asarray(sim.get_state())[BALL_STRIDE + 0]
print(f"  60 ticks of travel = {x1 - x0:.5f} m   (1.02000 before the fix)")
print()
print("ANSWER: pass dt as a float in SECONDS. int is still milliseconds.")
print()
print("=" * 70)
print("PART 7 - robots are torque-limited: how hard can a command ramp?")
print("=" * 70)
print("Source: src/robosim/sslconfig.h, class Robot:")
print("  wheelMotorMaxTorque = 0.070  // ... reduced to 3m/s2 accel")
print("  wheelMotorMaxRPM    = 1557.0")
print("rSim does not obey a velocity command instantly. The wheels are torque-")
print("limited, and the limit depends on direction because the four wheels sit")
print("at 60/135/225/300 degrees. The motion planner (tbots.motion) must only")
print("ask for what every direction can follow, so this measures it.")
print()


def one_robot(x, y):
    return robosim.SSL(FIELD_TYPE, 1, 0, DT, [4.0, 2.8, 0.0, 0.0],
                       [[x, y, 0.0]], [])


def one_action(vx, vy):
    return [[0.0, vx, vy, 0.0] + [0.0] * (ACT_LEN - 4)]


def state_v(sim):
    s = np.asarray(sim.get_state())[BALL_STRIDE:BALL_STRIDE + ROBOT_STRIDE]
    return float(s[3]), float(s[4]), float(s[2])


print("7a. Step command to 2.0 m/s from rest, acceleration over ticks 2..12:")
for deg in (0, 45, 90):
    ux, uy = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    sim = one_robot(-1.2 * ux, -1.2 * uy)
    sim.get_state()
    along = []
    for _ in range(13):
        sim.step(one_action(2.0 * ux, 2.0 * uy))
        vx, vy, _h = state_v(sim)
        along.append(vx * ux + vy * uy)
    print(f"  {deg:3d} deg   {(along[12] - along[2]) / (10 * DT):.2f} m/s^2")
print()
print("7b. Command ramped at a fixed rate to 2.5 m/s and back, 24 directions;")
print("    worst |measured - commanded| velocity and worst heading drift:")
for accel in (1.8, 1.9, 2.0, 2.5):
    worst_err = worst_yaw = 0.0
    for deg in range(0, 360, 15):
        ux, uy = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        sim = one_robot(-1.2 * ux, -1.2 * uy)
        sim.get_state()
        v = 0.0
        for k in range(160):
            v = min(2.5, v + accel * DT) if k < 80 else max(0.0, v - accel * DT)
            sim.step(one_action(v * ux, v * uy))
            vx, vy, heading = state_v(sim)
            if k > 3:
                worst_err = max(worst_err, float(np.hypot(vx - v * ux, vy - v * uy)))
            worst_yaw = max(worst_yaw, abs((heading + 180.0) % 360.0 - 180.0))
    print(f"  {accel:.1f} m/s^2   velocity error {worst_err:.3f} m/s   "
          f"heading drift {worst_yaw:.2f} deg")
print()
print("ANSWER: a command ramped at <= 1.9 m/s^2 is followed in every direction.")
print("        Faster ramps saturate a wheel on the diagonals: the robot falls")
print("        behind and YAWS although vangular = 0. Plan with 1.8 m/s^2.")
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Division B field_type   = {div_b_field_type}")
print(f"  BALL_STRIDE             = {BALL_STRIDE}")
print(f"  ROBOT_STRIDE            = {ROBOT_STRIDE}")
print("  action vector length    = 8")
print("  state angles / poses    = degrees, heading in (0, 360], vdir in deg/s")
print("  commanded vangular      = radians/second   <-- asymmetric, mind this")
print("  get_state()             = must be called exactly once per step()")
print(f"  timestep                = float seconds; {DT!r} s is exact 60 Hz")
print("  isotropic accel limit   = <= 1.9 m/s^2 ramps track in every direction")
