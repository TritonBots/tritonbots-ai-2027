# TritonBots — RoboCup Small Size League, Division B

## Quick start

    git clone --recurse-submodules https://github.com/TritonBots/tritonbots-ai-2027.git tritonbots
    cd tritonbots
    # then follow docs/SETUP.md end to end
    # already set up? go straight to docs/ONBOARDING.md

## Once set up

    make proto              # regenerate protobufs
    make test                # run tests
    docker compose up -d    # referee + simulator + two visualizers
    python -m tbots.apps.viz_rsim --realtime   # watch rSim at localhost:8083

    # Two browser views. :8082 is the ER-Force match (vision on 10020).
    # :8083 is our own publishers -- rSim, eval, replay -- on 10006. Never
    # pass --port 10020: the simulator already publishes there, the client
    # flickers between both worlds, and viz_rsim now refuses to start.

## Measured throughput

`python -m tbots.rl.vec --workers 1 8 32 --seconds 15 env=div_b_6v6 env.perception.mode=MODE`
(TASK-055). Environment steps per second summed over workers, 6v6 Division B
at 60 Hz, random actions, one rSim per subprocess, `OMP_NUM_THREADS=1`.
Dev box: i7-1185G7, 8 logical CPUs, WSL2. 2026-11-19.

| Workers | `truth` | `tracked` |
|---:|---:|---:|
| 1  | 139 | 102 |
| 8  | 674 | 526 |
| 32 | 837 | 589 |

**Read these as ratios, not as this machine's ceiling.** The host was
throttled during the run: `python -m tbots.apps.viz_rsim --seconds 3`, which
measured 516-532 steps/s in September, gave 137 in the same session. Expect
roughly 3.8x the absolute numbers on an unthrottled run. What holds either way:

- 8 workers give about 5x one worker; 32 workers on 8 logical CPUs add only
  ~25% more. Use about one worker per logical CPU.
- `tracked` perception (the mode anything exported must train in) costs
  25-30% against `truth`.
- Raw rSim alone is ~266 steps/s at 6v6 in the same session; the env layers
  (`truth`) keep ~76% of that.

The earlier single-process figure, 521 steps/s, was `viz_rsim` measured
before TASK-070, and is superseded by the table above.

## Where things live

| I want to... | Go to |
|---|---|
| write a reward function | `src/tbots/rl/rewards/` |
| train a new learned skill | `python -m tbots.rl.new_task NAME` → `src/tbots/rl/tasks/` |
| train / evaluate / replay | `python -m tbots.rl.train`, `tbots.apps.eval`, `tbots.apps.replay` |
| write a skill | `src/tbots/skills/` |
| write a tactic | `src/tbots/tactics/` |
| change field dimensions | `src/tbots/core/geometry.py` |
| understand colour / side flipping | `src/tbots/core/perspective.py` |
| change ports | `configs/net/` |
| set up a new machine | `docs/SETUP.md` |
| get productive on day one | `docs/ONBOARDING.md` |
| understand the architecture | `docs/ARCHITECTURE.md` |
| pick up a task | `docs/TASKS.md` |

## The architectural rules

- **Rule 1.** `src/tbots/core/` imports nothing from the rest of the codebase.
- **Rule 2.** Two backends, one match contract. Match code sees only `Backend`; training takes `SimBackend`.
- **Rule 3.** We are always `us`, we always attack `+x`. The backend does the flipping.
- **Rule 4.** Units convert exactly once, at the backend boundary. Above it: meters, radians, seconds.
