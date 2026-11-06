# TritonBots SSL Codebase — Architecture

> Why the codebase is shaped this way, and the decisions that shape it.
> Setup lives in `docs/SETUP.md`. The recruit walkthrough is `docs/ONBOARDING.md`.
> What is left to build is `docs/TASKS.md`.
>
> Rewritten 2026-09-04 after the architecture review. That review scoped
> itself to the RL training pipeline. Match-path decisions (radio, real
> robots) are out of scope and are marked as such. Section 13 lists the open
> problems.

---

## 1. Who and what

**Team:** TritonBots, RoboCup Small Size League, **Division B**: 6v6 robots on
a 9 × 6 m field. Everything in this codebase is Division B. Division A
geometry exists in `core/geometry.py` for completeness and is used nowhere.

**Python package:** `tbots`.

**Timeline:** recruits arrive mid-October 2026. On their first day they must
be able to write a reward function, train a policy against it, and watch it
play in a browser. Everything in this document exists to make that day work
and to make what they train valid on a real field later.

**Compute:** the Atlantis cluster (UCSD Supercomputing Club, SDSC-hosted). No
container runtime; training runs in a per-user venv against a shared
user-space ODE prefix. `docs/ONBOARDING.md` has the cluster workflow.

**Platforms:** WSL2 and macOS for development. Only the networked match
backend needs Linux.

---

## 2. Why a new codebase

The previous codebase used a simulator built for the Simulation League, not
SSL. Making it behave like SSL was a permanent source of bugs. Rather than
keep patching it, we started over with an SSL-native stack.

This year's strategic bet is reinforcement learning. That changes what "a
good simulator" means. We need one thing fast enough to generate millions of
training steps, and another thing protocol-accurate enough to prove the
result works at a competition. Those are different requirements, and
recognising that they are different is the central design idea here.

---

## 3. rSim and rSoccer

**rSim** is a C++/ODE physics core with Python bindings, derived from grSim
and FIRASim. You call `sim.step(actions)` in-process and get back a flat
state array. **rSoccer** is a thin Gymnasium layer over rSim plus a pygame
renderer.

**Good for:** fast, deterministic, in-process training. No sockets, no clock.

**Not good for:** anything protocol-accurate. rSoccer emits no SSL-Vision
packets, speaks no simulation protocol, and has no referee. It hands you
perfect ground truth.

**Things we verified empirically**, because the documentation contradicts
itself. Full detail in `docs/RSIM_FACTS.md`:

- Division B is `field_type=1`, not 0.
- The action vector is length 8. A shorter vector does not raise; it reads
  past the end of an unchecked `std::vector`.
- Angles are degrees coming out and radians per second going in.
- `get_state()` must be called exactly once per `step()` or velocities are
  wrong.
- **The timestep is a float in seconds — but an int still means
  milliseconds.** Upstream took only an integer number of milliseconds, so
  `1/60 s` became 17 ms and training ran at 58.8 Hz while believing it ran at
  60. Our fork added a `double`-seconds constructor overload (TASK-070, done
  2026-09-07); the int overload survives for rSoccer. Pass `float(dt)`.

Both packages are unmaintained since 2021, so we forked both. See §10 for
the fork policy.

---

## 4. Two simulators, one interface

A simulator fast enough for RL cannot be doing UDP round trips at 60 Hz. A
simulator that proves protocol compliance must be. So there are two:

```
                 Coach: tactics and skills   (backend-agnostic)
                              |
              WorldState  /  RobotCommand    <- the contract
                    /                  \
        rSim backend                 Network backend
   in-process, no sockets       SSL simulation protocol over UDP
   100x+ realtime               realtime
            |                            |
   RL training and eval        ER-Force simulator, real robots
```

Nothing above the backend layer knows which one is running.

### `Backend` and `SimBackend`

The two backends do not have identical capabilities, and pretending they do
was a source of leaks: `RSimBackend` had grown `reconfigure`,
`set_game_state` and other methods that no match backend could implement.
The protocol is now split in two:

```python
class Backend(Protocol):            # the match contract
    dt: float
    geometry: FieldGeometry
    def reset(self, scenario: Scenario) -> WorldState: ...
    def step(self, commands: Sequence[RobotCommand]) -> WorldState: ...
    def close(self) -> None: ...

class SimBackend(Backend, Protocol):  # simulator-only powers
    def step(self, commands, opponent_commands=()) -> WorldState: ...
    def place(self, ball=None, us=None, them=None) -> WorldState: ...
    def set_game_state(self, game: GameState) -> None: ...
```

(Sketches. `docs/SETUP.md` and the source are authoritative for code.)

- `opponent_commands` is always optional. Uncommanded opponents stand still.
  With `n_them=0` there is nothing to address.
- `place()` teleports whatever is given and leaves the rest. rSim has no
  partial teleport, so it implements `place()` as a full reset with the
  current poses filled in. ER-Force implements it with `SimulatorCommand`.
- A command whose `robot_id` is not one of ours is an **error**, not a
  silent drop.

rSim and ER-Force with simulation control are `SimBackend`s. Real robots are
only a `Backend`. A training environment requires a `SimBackend`, so the type
checker stops anyone pointing a training run at hardware.

### The acceptance test

The same `GoToPoint` skill must converge on both backends. Once the Coach
exists (§8) the parity test extends to the whole stack. If something works in
rSim and not against ER-Force, that is a sim-to-sim gap, and we want to find
it in September rather than April.

---

## 5. The architectural rules

Every file obeys these.

**Rule 1: `src/tbots/core/` imports nothing from the rest of the codebase.**
`core` defines data types. It never depends on a simulator, a socket, or a
neural network. Everything else imports `core`.

**Rule 2: Two backends, one match contract.** `Backend` is everything a match
needs: `dt`, `geometry`, `reset()`, `step(commands)`, `close()`. `RSimBackend`
and `NetworkBackend` both implement it, and no code on the match path names
either one. `SimBackend` extends it with the three powers only a simulator
has: `step(commands, opponent_commands)`, `place()`, `set_game_state()`. Code
that needs those takes a `SimBackend` in its type, which is why `SSLEnv` cannot
be pointed at hardware. `make lint` type-checks both. Detail in §4.

**Rule 3: We are always `us`, we always attack `+x`.** The world model has
`us` and `them`, never `blue` and `yellow`. `core/perspective.py` owns the
transform. It is a 180° rotation, not a mirror, and it is its own inverse.
`RobotCommand` velocities are in the robot's local frame and are never
transformed. An `as_opponent(world)` helper applies the same rotation and
swaps `us`/`them`, which is how a self-play opponent sees the field.

**Rule 4: Units convert exactly once, at the backend boundary.** Above it:
meters, radians, seconds. All conversion lives in `core/units.py` and the
backend adapters.

---

## 6. Core types

Immutable dataclasses. Everything else is a function of these. The snippets
below are abbreviated; do not transcribe them.

```python
WorldState:      t, t_capture, ball, us, them, game, telemetry
RobotState:      robot_id, x, y, theta, vx, vy, vtheta, has_ball, visible
RobotCommand:    robot_id, vx, vy, vtheta, kick_speed, chip, dribbler
GameState:       play, ours, can_move, can_touch_ball, min_ball_distance,
                 placement_target, counter, ...
```

Two choices worth preserving:

- **`RobotCommand` is velocity-level, not wheel-level.** Every backend and
  the real firmware accept body velocities. Wheel IK is a backend concern.
- **Commands are absolute, never deltas.** UDP drops packets. A lost delta
  corrupts state forever; a lost absolute command is a non-event.
- **`kick_speed` and `dribbler` stay floats.** The hardware supports variable
  kick strength and dribbler speed. Learned skills emit booleans (§9) and a
  codec maps them to speeds, but the command keeps the full range.

Types added by the review:

```python
DetectionFrame:  t_capture, camera_id, balls, robots_blue, robots_yellow
                 # what a camera saw. Field frame, colours, no velocities.
RobotFeedback:   robot_id, t, has_ball, battery, kick_charge, wheel_speeds
                 # telemetry from a robot, or from the simulator
Transition:      world, prev, commands, ego_id, geometry, dt, done
                 # what a reward term receives. See §9.
```

`DetectionFrame` and `RobotFeedback` live in `core` so the tracker never
touches a protobuf. `net/` converts wire packets into them; the training
pipeline synthesises them from rSim truth. That is what lets one perception
pipeline serve both backends.

### What `t` means

`WorldState.t` is the time the state is **valid for**, after the tracker has
forward-predicted to now plus the expected actuation delay. `t_capture` is
when the newest frame was captured. In rSim the two are equal. At a match the
difference is the latency, and keeping both is what makes it measurable.

---

## 7. Perception: one pipeline for both backends

rSim hands you perfect, instantaneous state. A real vision frame is 20 to
40 ms stale, noisy, and sometimes missing. A policy trained on perfect state
will not survive the real field.

The old design injected latency and noise into the encoded observation
vector with a Gym wrapper. That put the corruption at the wrong layer: the
wrapper could not know what the floats meant, and the match path would run a
tracker the policy had never seen. The review moved perception below the
environment.

```
                     rSim truth
                         |
                   PerceptionSim        latency, noise, dropouts,
                         |              synthetic telemetry
                   DetectionFrame       (same type net/vision.py produces)
                         |
                      Tracker           the SAME tracker a match uses
                         |
                  perceived WorldState  ->  observation builders
```

Three modes, selected by `perception.mode` in the env config:

| Mode | What the policy sees | Use |
|---|---|---|
| `truth` | rSim ground truth | Debugging only |
| `noisy` | Truth with delay, noise, dropouts (`visible=False`), no tracker | Day-one training before the tracker exists |
| `tracked` | Corrupted `DetectionFrame`s through the real `Tracker` | Anything that will be exported |

The tracker is pure math over 13 objects, tens of microseconds against
roughly 2 ms of ODE per step. It is not the packet-processing stack, which
stays in `net/` and never runs during training.

Single-camera frames only for now. We will have one camera for sim-to-real
testing before the competition, and single frames already exercise
association, velocity estimation and extrapolation, which is where the bugs
are.

**The environment holds two worlds.** Ground truth drives reward,
termination and the synthetic referee. The perceived world drives
observations and nothing else. Rewarding on the perceived world would reward
noise.

---

## 8. The behavioural layers and the Coach

```
  TACTICS   chooses WHICH skill each robot runs      4 Hz      <- the RL bet
     |
  SKILLS    closed-loop behaviour for ONE robot      60 Hz     <- some learned
     |
  MOTION    velocity control, acceleration limits    60 Hz     <- classical, always
```

**Motion stays classical.** A tuned velocity profile beats months of RL at
driving to a point and transfers to hardware for free. Learning go-to-point
is the most common way a student RL team wastes a season.

The motion layer is `src/tbots/motion/` (TASK-086). It imports `core` only, so
a skill, a state-machine tactic and a future RL "waypoint" codec can all
drive through it. `MPCPlanner` is a **dynamic** sampling MPC:

- every opponent and teammate is predicted forward over a one-second horizon
  by an `ObstaclePredictor` (constant velocity by default, or constant
  acceleration), with a clearance radius that grows with lookahead;
- candidates are checked against those predictions in space-time, with a
  swept check that cannot tunnel between samples;
- it plans inside `core/limits.py`'s `MotionLimits`, measured from rSim
  (`docs/RSIM_FACTS.md`, "Robot dynamics"), so the command sent is the motion
  produced.

The static rule regions (defense areas, the STOP ball circle) are routed
round globally by `motion/route.py`, because a two-metre defense area is a
local minimum for a one-second horizon. `GoToPoint(avoid=True)`, the
default, drives through the planner, so every role that moves avoids
without a tactic change. `hold=True` keeps station after arriving instead
of reporting success, because the Coach parks a finished skill and a parked
robot cannot dodge.

**Skills** share one interface whether they are forty lines of geometry or a
neural network: `reset`, `step`, `status`. `LearnedSkill` loads a checkpoint
and implements the same protocol, so a policy trained on Tuesday runs in the
match stack on Wednesday by editing one config line.

**Tactics** emit skill assignments, not velocities. The action space is
parameterised-discrete: `shoot`, `pass_to(teammate)`, `dribble_toward(region)`,
`mark(opponent)`, `reposition(zone)`. Targets are pointers over the robot
set, never slot indices.

### The Coach

One object plays in both places. `tactics/coach.py` owns the tactic, the
live skill instances, role assignment and restart routines, and exposes
`tick(world) -> list[RobotCommand]`. `TacticsEnv` drives it at simulation
speed; a match app drives it at realtime. There is no second play loop to
drift.

**Decisions happen on a fixed tick**, 4 Hz by default. Every 15 physics
ticks the tactic re-emits an assignment for every robot. A skill instance
persists across decisions while its `SkillSpec` is unchanged, so a learned
skill keeps its internal state; a changed assignment discards the old
instance. A skill that finishes early leaves its robot stopped until the
next decision, at most 250 ms away. The alternative, ending the decision
step whenever any one robot's skill finishes, makes step lengths erratic and
rebuilds every skill each time a teammate arrives somewhere.

**The Coach is also a rule filter.** Below the tactic, a deterministic layer
overrides commands whenever `GameState` forbids them: stop on HALT, stay
0.5 m from the ball on STOP and at opponents' free kicks, stay 0.2 m from the
opponent defense area during stoppages, keep non-keepers out of our defense
area, respect the stop-speed limit. No learned policy can commit a
positional foul, in training or at a match. The synthetic referee therefore
never has to punish them.

### Why the decomposition matters

At 60 Hz a two-minute episode is about 7,200 control ticks per robot. At the
tactics level it is about 480 decisions. Horizon is what destroys credit
assignment, and this is how we shorten it. It only works because the layers
below are solid.

---

## 9. Training

### One way to build an environment

`make_env(cfg) -> gym.Env` composes backend → `PerceptionSim` → tracker →
env → wrappers from the Hydra config. It is the only way train, eval, tests
and the throughput benchmark build an environment. Two hand-built stacks
that drift apart is the classic "works in training, not in eval" bug.

### What a recruit writes: a `SkillTask`

Training a skill used to mean touching six files. Now it is one registered
`SkillTask` that groups:

- a scenario sampler (initial positions, randomised),
- success and failure predicates,
- an observation builder name and an action codec name,
- a time limit,
- a default reward YAML.

`python -m tbots.rl.new_task dribble` scaffolds the file with every hook
stubbed and documented, plus a test that the task registers and one episode
runs. Then `python -m tbots.rl.train env=skill task=dribble` is the whole
command.

### Action codec

Learned skills act in `Box(vx, vy, vtheta)` plus two booleans, kick and
dribbler. The codec:

- maps the booleans to a kick speed and dribbler speed supplied by the
  skill's kwargs, so `PassTo` and `Shoot` kick at different powers while the
  network only learns *when*;
- applies per-tick acceleration and angular-acceleration limits from `core`,
  so the simulator only ever executes what a motor can follow;
- clamps kick speed to the 6.5 m/s rule;
- supports `action_repeat`, default 1.

Every codec parameter is recorded in the checkpoint.

### Observation builders

Named, registered, and **immutable once any checkpoint exists**. A layout
change is a new name (`egocentric_ball_v2`). A golden-vector test pins every
builder's size and output for a fixed `WorldState`.

- Skill builders are **egocentric**: everything relative to the ego robot's
  pose and heading, so a dribble learned facing `+x` works facing anywhere.
- Tactics builders are **global** and permutation-invariant over the robot
  set (a set encoder), with a fixed-size output regardless of robot count.
- No builder encodes the robot ID value. Our robots at a match might be
  `{1, 3, 4, 7, 9, 12}`; every skill env trains robot 0. A test asserts
  relabelling invariance.
- No running normalisation. Builders emit values pre-scaled by fixed
  constants. Running statistics are invisible checkpoint state that has to
  be exported and recomputed identically at a match.

### Reward terms

A term is a class with `reset()` and `__call__(tr: Transition) -> float`,
registered with `@register_reward("name")`, composed by weight in YAML.
`Transition` carries:

| Field | Why |
|---|---|
| `world`, `prev` | Ground truth after and before the step. Never the perceived world. |
| `commands` | What we sent, for kick-spam and energy terms. |
| `ego_id` | The trained robot in skill envs, `None` in tactics. |
| `geometry` | So terms never import field constants. |
| `dt` | So per-second shaping is rate-independent. |
| `done` | `None` while running, else `GOAL_US`, `GOAL_THEM`, `OUT_OF_BOUNDS`, `TIMEOUT`, `SKILL_SUCCESS`, `SKILL_FAILURE`, `FOUL_*`. |

Deliberately absent: the perceived state and the observation vector, so a
term cannot reward what the policy could not know; and any RNG, so terms are
deterministic. `info["reward_terms"]` reports per-term contributions every
step, which is the difference between debugging a reward in an hour and a
week.

In the tactics env, reward accumulates over the 15 inner physics ticks.
Sampling once per decision turns shaping terms into noise about where the
skill happened to end.

### Episode boundaries and the synthetic referee

rSim knows nothing about the referee. `SyntheticReferee` produces
`GameState` from ground truth and feeds `SimBackend.set_game_state()`.

Out-of-bounds is a per-env parameter. Skill envs default to **terminate**:
episodes are short and a dribble that left the pitch has failed. The tactics
env defaults to **restart**: a synthetic free kick via `place()`, and the
episode continues. Terminating a 6v6 episode on every out-of-bounds teaches
"score before the ball leaves" rather than possession, and the terminal
reward is a trap either way (a penalty breeds timidity, zero makes kicking
it out a free reset).

The referee models what the Coach's filter cannot prevent and rSim can
detect: excessive dribbling (1 m), double touch, no progress (10 s). Each is
a `done` reason with the same free-kick restart. Contact fouls (pushing,
crashing, ball holding) are skipped. This is an open problem (§13).

Episodes also start in restart states (kickoff, free kick, stop) with the
matching `GameState`, so policies see every `Play` value.

### The algorithm

A CleanRL-style PPO vendored into `rl/algo/ppo.py`. About three hundred
lines, no new dependency, readable by a recruit debugging a reward.
Stable-baselines3 fights you the moment you want a set encoder or a
multi-head action; two trainers means two checkpoint formats.

The PPO is built around a composable action distribution from day one,
because the tactics policy is **centralised and factorised**: a shared set
encoder, one independent action head per robot, joint log-probability equal
to the sum of per-robot log-probabilities, one centralised critic. That is
MAPPO. A naive joint softmax over six robots' choices is about 12⁶ actions
and cannot be output by any head. Skill envs are ordinary single-agent PPO
through the same code.

Truncation on the time limit bootstraps from the value function; termination
does not. The vendored PPO must get this right, and a test should check it.

### Checkpoints

`rl/artifacts.py` writes atomically with a `format_version`. Metadata
records: obs builder name and size, codec name, `dt`, `action_repeat`,
acceleration limits, perception mode, curriculum stage, git SHA.

Loading **refuses** on a builder, codec or `dt` mismatch, **warns** on a
perception-mode mismatch, and **never checks the SHA**. Immutable builder
names are what make an August checkpoint load in April.

Anything exported must have been trained in `tracked` mode.

### Evaluation, rollouts, and watching it play

`apps/eval.py` runs the deterministic policy on a **fixed seed set** of
scenarios, in `tracked` perception, against the configured opponent, and
reports goal difference and per-term reward distributions per episode.
Training-rollout statistics are measured on a moving distribution with
exploration noise and are never used to compare runs.

Each eval records its episodes as rollout files: ground-truth `WorldState`
sequence, commands, per-term rewards, `done` reason. The run directory keeps
the newest ten (configurable). A 6v6 two-minute episode is about 3 MB; a
ten-second skill episode is about 20 KB. `apps/replay.py` streams any
rollout to the browser through `VisionPublisher`, pausable and seekable. A
rollout file is also a regression artifact a recruit can attach to a pull
request.

The trainer runs a short eval at every checkpoint interval, so a run has a
visual history without recording training rollouts.

### Curriculum: a sequence of runs

Promotion between stages is a **human decision** after watching the policy
play. So a curriculum is not an in-run schedule. A stage is one training run
with a fixed robot count (`env.n_us`, `env.n_them`), resumed from a
checkpoint a person chose (`train.resume_from`). There is no automatic
promotion criterion and no `reconfigure()` inside a run.
`configs/train/curriculum_example.yaml` documents a recommended stage order.

Two rules still hold. The observation must be a fixed size across stages, or
stage 2 cannot resume from stage 1. And **the field never shrinks**: a 2v2
stage runs on the full Division B pitch. Constant geometry is what makes the
easy stage teach something true about the hard one.

### Opponents and self-play

The opponent runs its own Coach on `as_opponent(world)` and sees **ground
truth**. It is cheaper, and a better-informed opponent is a harder
curriculum. Opponents are sampled uniformly from a pool of frozen
checkpoints that always includes the scripted baseline; a config override
pins a specific checkpoint or the baseline for a run. Always-latest produces
cycling strategies; prioritised sampling is a later refinement.

### Seeding

One seed threads through scenario sampling, perception noise, opponent
sampling and policy sampling. A test asserts two runs with the same seed
produce identical first-N transitions per worker. rSim is single-threaded
ODE, so this holds.

### Parallelism and the cluster

Subprocess vector envs, one rSim per process, `OMP_NUM_THREADS=1`. A
GIL-release patch in our fork is a real speedup and waits for a measured
number that disappoints. Since robot counts never change inside a run, the
vector env is built once by `make_env` and torn down at the end.

Single-process throughput: 521 steps/s at 6v6, 60 Hz, on the WSL2 dev box;
about 250 on Atlantis. Atlantis compute nodes may have no outbound internet,
so W&B runs offline and syncs from the login node. Walltime limits make
checkpoint-and-resume non-optional.

---

## 10. Communication and external tools

**The game controller is not a simulator.** It is the referee: a state
machine with a web UI and no physics. It attaches to either backend, or
neither.

| Channel | Direction | Transport | Address / port |
|---|---|---|---|
| SSL-Vision detection + geometry | in | UDP multicast | `224.5.23.2:10006` (ER-Force uses **10020**) |
| Referee messages | in | UDP multicast | `224.5.23.1:10003` |
| Tracked vision | in | UDP multicast | `224.5.23.2:10010` |
| Simulator control | out | UDP | `10300` |
| Robot control, blue / yellow | out | UDP | `10301` / `10302` |
| GC team interface | both | TCP | `10008` (`10108` TLS) |
| GC CI mode | both | TCP | `10009` |
| GC web UI | | HTTP | `8081` |

Multicast packets are bare protobuf, one message per datagram. The TCP
interfaces are length-delimited streams. In development, multicast TTL is
always 0.

The control loop runs at 60 Hz, one `RobotControl` packet per tick with all
six robots. Vision is the real clock: tick on every merged frame, with a
watchdog that ticks on the tracker's extrapolation if no frame arrives
within 50 ms. The referee is a latched event stream; poll it, never block on
it. Trigger transitions on `GameState.counter`, never on `play` alone.

**ER-Force geometry.** The compose stack ran the match simulator with
`GEOMETRY=2020`, which is a 12.04 × 9.02 m Division A pitch. Nothing noticed
for the whole build. The Division B preset is `2020B` (TASK-071). The
network backend will also read geometry from the wire and fail loudly if it
disagrees with the configured division.

**There is no Python client library.** We write the socket code once in
`net/`.

### Fork policy

Fork rSim and rSoccer. Do not fork grSim, ER-Force, or the game controller.
Fork the thing you must modify and nobody validates against; run the thing
you must not modify because everyone validates against it.

### `VisionPublisher`

Converts any `WorldState` into SSL-Vision packets, so `ssl-vision-client`
renders both backends and every rollout in the same browser tab. It also
keeps our vision serialisation exercised constantly.

---

## 11. Decisions and rationale

| Decision | Rationale |
|---|---|
| Package `tbots` | `ssl` shadows the stdlib; `triton` collides with the compiler PyTorch installs. |
| Python 3.11 | 3.10 is EOL in October 2026, mid-season. |
| ER-Force `simulator-cli` as the only networked simulator | It is what the official virtual tournament runs, it is headless, and `ssl-vision-client` already covers visualization. |
| Two simulators | Speed and protocol accuracy are irreconcilable in one process. |
| `Backend` / `SimBackend` split | Names the real capability gap so training envs cannot be pointed at hardware and opponents have a path to the physics. |
| Perception below the environment, three modes | Noise belongs on semantic state, not on a float vector; `tracked` gives training and match one perception pipeline. |
| Reward on truth, observe on perception | Rewarding perceived state rewards noise. |
| One `Coach`, fixed 4 Hz tick, skill persistence | One play loop for both backends; predictable step lengths; learned skills keep their state. |
| Rule filter in the Coach | Positional fouls become impossible, so no policy has to learn them and no referee has to punish them. |
| Vendored PPO, factorised heads, centralised critic (MAPPO) | Readable by recruits; a joint softmax over six robots is infeasible; SB3 fights custom heads. |
| Booleans for kick and dribbler in the action space | On/off is what a skill decides; power comes from the skill's kwargs and the command keeps the float. |
| Acceleration limits in the codec | The simulator must only execute what a motor can follow. |
| Dynamic sampling MPC in `motion/`, inside `GoToPoint` | Opponents move, so obstacles are predicted trajectories, not circles. Sampling handles non-convex avoidance with numpy alone; free space falls back to the time-optimal profile; every role that moves gets it with no tactic change. |
| Immutable builders and codecs, new name on change | Checkpoints survive months of commits; the SHA is recorded, never checked. |
| Fixed scaling, no running normalisation | No invisible checkpoint state to export. |
| Out-of-bounds: terminate for skills, restart for tactics | Short skill episodes are fine; 6v6 needs possession across restarts. |
| Curriculum as a sequence of runs, human promotion | Matches how we validate; deletes fragile in-run reconfiguration. |
| Fixed eval seed set, recorded rollouts | Only comparable numbers; a human can watch what the numbers describe. |
| Opponent pool with the scripted baseline, ground-truth opponent | Avoids cycling; harder curriculum; cheaper. |
| Subprocess parallelism first | Boring and correct; measure before patching the fork. |
| Referee normalised into `GameState` | Nothing above `net/` imports a protobuf or reasons about colours. |
| macOS first-class for development | Team is split WSL2/macOS. Only the networked backend needs Linux. |

---

## 12. Repo shape

```
tritonbots/
├── protos/                 # pinned submodules: game controller, sim protocol, vision
├── third_party/
│   ├── rsim/               # OUR fork
│   └── rsoccer/            # OUR fork
├── src/tbots/
│   ├── _pb/                # generated protobufs (gitignored)
│   ├── core/               # the contract. depends on nothing.
│   │   └── perspective.py  # Rule 3, including as_opponent()
│   ├── backends/           # base.py (Backend, SimBackend), rsim.py, network.py
│   ├── net/                # sockets and protobuf glue
│   ├── perception/         # tracker.py, sim.py (PerceptionSim)
│   ├── motion/             # MPCPlanner, obstacle prediction, routing (imports core only)
│   ├── skills/             # single-robot behaviours, learned.py
│   ├── tactics/            # coach.py, base.py, scripted.py, learned.py, roles, restarts
│   ├── rl/
│   │   ├── algo/           # ppo.py, distributions
│   │   ├── envs/           # base.py, skill_env.py, tactics_env.py, synthetic_referee.py
│   │   ├── tasks/          # SkillTask registry, one file per task
│   │   ├── obs/            # named builders
│   │   ├── codec.py        # action codecs
│   │   ├── rewards/        # registry, terms
│   │   ├── rollout.py      # rollout files
│   │   ├── factory.py      # make_env
│   │   ├── new_task.py     # scaffold
│   │   ├── opponents.py, vec.py, artifacts.py, train.py
│   ├── viz/
│   └── apps/               # viz_rsim, ref_monitor, eval, replay, wiggle
├── configs/                # hydra: env, perception, task, reward, train, opponent, net
├── docs/
└── tests/
```

---

## 13. Open problems

Recorded so nobody mistakes an interim answer for a decision.

- **Contact fouls in training.** Pushing, crashing and ball holding are not
  modelled by the synthetic referee because rSim's contact behaviour is
  unverified. Revisit once ER-Force evaluation shows whether they occur.
- **Observation scaling constants.** Builders divide by fixed values
  (half-length, roughly 3 m/s). Real-world tests may move the perception
  noise parameters; they are unlikely to move these, but the question stays
  open until someone has run on a field.
- **Stop-speed threshold.** The rule filter needs the exact robot speed
  limit during STOP. Believed to be 1.5 m/s; verify against the current
  rulebook before it goes in code.
- **Match path.** Radio frame layout, the firmware's heading field, real
  robot telemetry. Out of scope for the training review; owned elsewhere.
- **Set encoder details.** DeepSets versus attention pooling, and the
  region/zone discretisation for `dribble_toward` and `reposition`.
- **GIL release in rSim.** Only if the measured multi-process throughput on
  Atlantis is not enough.

---

## 14. Status

The scaffolding described in `docs/SETUP.md` is built and verified; every
deviation is in `docs/SETUP_LOG.md`. What remains, in priority order with a
runnable gate per task, is `docs/TASKS.md`.
