# TritonBots — RoboCup Small Size League, Division B

## Quick start

    git clone --recurse-submodules https://github.com/YashTandon05/tritonbots.git
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

rSim, 6v6, 60 Hz, single process:  521 steps/s

## Where things live

| I want to... | Go to |
|---|---|
| write a reward function | `src/tbots/rl/rewards/` |
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
