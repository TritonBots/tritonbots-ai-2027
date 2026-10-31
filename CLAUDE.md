# TritonBots — agent working agreement

The staged build from `docs/SETUP.md` is finished. Work now is the tasks in
`docs/TASKS.md`, done one at a time, each closed by its own gate. Read this
file before your first action in any session.

---

## 1. Which document is authoritative for what

| Document | Authoritative for |
|---|---|
| **The source code** | What the system does. `src/`, `tests/`, `configs/`, `third_party/rsim` (our fork). |
| `docs/TASKS.md` | What to do, in what order, and the **Done when** gate that closes each task. |
| `docs/ARCHITECTURE.md` | Why it is shaped this way, and the decisions. Never a source of code. |
| `docs/RSIM_FACTS.md` | Empirically verified rSim behaviour. Do not reason about rSim; read this. |
| `docs/SETUP.md` | How the stack was built from nothing. A record, not a spec to build from again. |
| `docs/SETUP_LOG.md` | What actually happened, per step and per task. |
| `docs/ONBOARDING.md` | The recruit-facing walkthrough. Reference only. |
| `CLAUDE.md` (this file) | Process rules and environment facts. Wins on process. |

If two of these disagree on a fact, stop and ask. Do not pick one.

---

## 2. Environment facts

- WSL2 Ubuntu 24.04 on a Windows host. Everything runs inside WSL. The repo
  is `~/tritonbots`. Never work under `/mnt/c/...`.
- Python 3.11 from `uv`, in `.venv`. `source .venv/bin/activate` first.
  Noble's system Python is 3.12; never `apt install python3.11`.
- Docker is `docker.io` inside WSL, not Docker Desktop. `network_mode: host`
  is what carries multicast.
- ODE 0.16.2 is built from source into `/usr/local`. If rSim ever needs
  rebuilding, run all three checks first:
  ```
  pkg-config --modversion ode          # 0.16.2
  pkg-config --variable=libdir ode     # MUST be /usr/local/lib
  grep dDOUBLE /usr/local/include/ode/precision.h
  ```
  Noble's packaged `libode-dev` is also 0.16.2, so the version check alone
  passes against the wrong build, which compiles and gives wrong physics.
- **No sudo.** If a step needs it, stop and ask.
- This repo: https://github.com/TritonBots/tritonbots-ai-2027 (`origin`).
  It is a GitHub fork of the original https://github.com/YashTandon05/tritonbots
  (`personal`); changes reach `origin/main` by pull request.
- Forks: rSim https://github.com/YashTandon05/rSim, rSoccer
  https://github.com/YashTandon05/rSoccer.

---

## 3. Working protocol

**A task's Done-when gate is the unit of work.** Read the task in
`docs/TASKS.md` in full before acting. Write the failing test for the gate
first, make it pass, run `make lint` and `make test`, then commit.

One task per commit, prefixed `feat:`, `fix:`, `docs:`, `test:`, `build:`,
`chore:`. Each closed task gets a block in `docs/SETUP_LOG.md`:

```
## TASK-NNN — <title>            [PASS | BLOCKED]
Verification: <command run> -> <result>
Deviations:   <anything done differently from TASKS.md, and why>
Notes:        <anything the human should know>
```

Deviations is the field that matters. A gate that turned out to be
unsatisfiable as written, a signature that differed from the task text, a
finding outside scope: all go there. That log is how a multi-hour run is
reviewed by someone who did not watch it.

If a gate fails and you cannot see why, stop and report. Do not weaken the
gate. Do not start the next task.

**Scope.** Do the task you were given. A defect you notice in another task's
territory is a finding for the log and for `docs/TASKS.md`, not a licence to
fix it. The open problems in `docs/TASKS.md` §6 are decisions for a human;
implement the recorded assumption and say so.

**When you touch rSim.** After any change to `third_party/rsim`, rebuild,
re-run `scripts/verify_rsim.py`, and confirm every fact in
`docs/RSIM_FACTS.md` still reproduces. Record new facts there, in plain
English, before relying on them in code.

**When you touch the build.** If a change alters how the stack is built or
verified (a compose preset, a dependency, a tool version), update the matching
step in `docs/SETUP.md` so the record stays true.

---

## 4. Hard prohibitions

Each of these produces something that looks correct and is not.

- **Never `pip install rc-robosim`, `robosim`, or `rsoccer-gym` from PyPI.**
  They are stale 2021 wheels that overwrite the `robosim` we compile from
  source. Every physics result afterwards is wrong and every import check
  passes.
- **Never mock, stub, monkey-patch, or `pytest.skip` a component to make a
  gate pass.** Green must mean the real thing works.
- **Never invent an API signature.** If `robosim.SSL(...)`, a protobuf field,
  or a tool flag does not behave as documented, read the installed code and
  report the discrepancy.
- **Never guess a download URL.** Query the GitHub releases API for the real
  asset name.
- **Never fall back to Python 3.10 without asking.**
- **Never `git push --force`, rewrite history, or touch `main` on any
  upstream RoboCup-SSL repository.** Pushes to our own forks are fine.
- **Never use `pip --break-system-packages`.** `externally-managed-environment`
  means you are outside the venv.
- **Never weaken system security settings.** In particular, leave
  `kernel.apparmor_restrict_unprivileged_userns` alone.
- **Never add a dependency** without it being part of the task.

---

## 5. The architectural rules

Every file obeys these. The numbering is load-bearing: `Rule 3` and `Rule 4`
are referenced by name in `core/perspective.py`, `core/state.py`,
`net/referee.py` and elsewhere. Do not renumber.

**Rule 1 — `src/tbots/core/` imports nothing from the rest of the codebase.**
Everything else imports `core`. `core` defines the data types; it never
depends on a simulator, a socket, or a neural network. If you find yourself
adding `import robosim` or `import torch` to a file in `core/`, you have made
a mistake.

**Rule 2 — Two backends, one match contract.**
`Backend` in `backends/base.py` is everything a match needs: `dt`, `geometry`,
`reset()`, `step(commands)`, `close()`. `RSimBackend` and `NetworkBackend` both
implement it, and no code on the match path — skills, tactics, perception —
names either one. `SimBackend` extends `Backend` with the three powers only a
simulator has: `step(commands, opponent_commands)`, `place()`,
`set_game_state()`. Code that needs those takes a `SimBackend` in its type,
which is why `SSLEnv` cannot be pointed at hardware. `make lint` type-checks
both.

**Rule 3 — We are always `us`, we always attack `+x`.**
The world model has `us` and `them`, never `blue` and `yellow`. The backend
flips coordinates if we are yellow or defending the positive half.

**Rule 4 — Units convert exactly once, at the backend boundary.**
Above it: meters, radians, seconds. All conversion lives in `core/units.py`
and the backend adapters. Nowhere else.

Team name is exactly `TritonBots`, case-sensitive, no spaces. It
authenticates our game-controller connection; a typo fails later and
mysteriously.

---

## 6. Conventions

- Package `tbots`, importable as `from tbots.core.state import WorldState`.
- `uv` for all Python package operations, never bare `pip`.
- `make lint` runs ruff plus mypy over `core`, `backends` and the env base.
  `make test` runs the suite. Both must be clean before a commit.
- Commit messages say what changed and why, in the body, in prose. No
  attribution trailers.
