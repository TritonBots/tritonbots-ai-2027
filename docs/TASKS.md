# TritonBots — task board

**Status: 2026-09-04, after the architecture review.** Recruits arrive
**mid-October 2026** (about six weeks).

This is the single source of truth for what is left to build. The review
changed the shape of many tasks; §7 says what changed and why. IDs are kept
where a task survived and new IDs start at TASK-070.

| Document | Role |
|---|---|
| `docs/SETUP.md` | How to build the stack from nothing. Authoritative for the code it contains. |
| `docs/SETUP_LOG.md` | What actually happened during that build. |
| `docs/ARCHITECTURE.md` | Why the codebase is shaped this way, and the decisions. |
| `docs/ONBOARDING.md` | The recruit-facing walkthrough. |
| **`docs/TASKS.md`** (this file) | **What is left, in what order, and how we know it is done.** |

Every task has a **Done when** line. A task without a runnable gate is a task
nobody can honestly close. Sizes: **S** half a day or less, **M** one to
three days, **L** four days or more.

---

## 1. What "ready for recruitment" means

> A new member who has never seen this repo can, on a machine we did not
> prepare, follow `docs/ONBOARDING.md` from the top and by the end of the day
> have trained a policy against a reward function they wrote themselves and
> watched it play in their browser.

Tier 1 is everything between us and that sentence. The match path is not on
it: a recruit's first day is rSim → reward → train → watch. The cluster is on
it, because that is where real runs happen.

---

## 2. Done. Do not redo these.

| Task | Result | Evidence |
|---|---|---|
| TASK-001 | Division B is `field_type=1`; strides 5/11; action length 8; degrees out, rad/s in. | `docs/RSIM_FACTS.md` |
| TASK-002 | rSim builds and runs on Python 3.11 against our ODE; CI does it on every push. | SETUP_LOG Step 4 |
| TASK-003 | CI builds ODE against the system libccd like every dev box. | `.github/workflows/ci.yml` |
| TASK-013 | Rule 3 is a module: `core/perspective.py`, resolved once from the referee. | `tests/test_perspective.py` |
| Steps 1–16 | SETUP.md end to end, all seven acceptance checks. | SETUP_LOG Step 16 |
| Core contracts | `core/{units,geometry,gamestate,state,command,perspective}.py`. | `make lint` |
| rSim backend | `backends/rsim.py`, reshaped by TASK-072. | `pytest tests/test_rsim_backend.py` |
| Referee ingest | `net/referee.py` verified against the real game controller. | SETUP_LOG Step 16 |
| VisionPublisher | Any `WorldState` to SSL-Vision packets, with field lines. | SETUP_LOG Step 16 |
| GoToPoint | Reference skill, converges in rSim. | `tests/test_backend_parity.py` |
| Reward registry | `@register_reward`, `CompositeReward`, per-term contributions. Signature changes in TASK-076. | `rl/rewards/example.py` |
| Run artifacts | `rl/artifacts.py`: resolved config, atomic checkpoints, `format_version`. Metadata extends in TASK-059. | `tests/test_rl_artifacts.py` |
| Hydra composition | `configs/config.yaml` composes `env`/`reward`/`train`. | `python -m tbots.rl.train train.run_dir=/tmp/x` |
| External stack | `docker compose up -d`: game controller 3.21.0, ER-Force, vision-client. | SETUP_LOG Step 13 |
| Throughput | 521 steps/s, 6v6, single process on WSL2; about 250 on Atlantis. Measured before TASK-070, so it was 521 steps of 17 ms; re-measure with TASK-055. | README |
| TASK-070 | rSim steps at a true 1/60 s. Our fork's `SSL` takes a float-seconds timestep; an int is still milliseconds. | `tests/test_rsim_backend.py`, `docs/RSIM_FACTS.md` |
| TASK-071 | The compose stack runs Division B (`GEOMETRY: "2020B"`), verified off the wire and in `ssl-vision-client`. | SETUP_LOG TASK-071 |
| TASK-072 | `Backend` / `SimBackend` split. Opponents commandable, `place()`, unknown ids raise, `reconfigure()` gone. | `tests/test_rsim_backend.py` |
| TASK-004 | CI pins the verified rSim field type, layouts, action offsets, and 60 Hz timestep; it also runs the verification probe. | `tests/test_rsim_backend.py`, `.github/workflows/ci.yml` |
| TASK-073 | `as_opponent()` rotates a world by 180° and exchanges `us` and `them` for self-play. | `tests/test_perspective.py` |
| TASK-074 | Frozen perception dataclasses, capture time, and telemetry are part of the core world contract. | `tests/test_core.py`, `mypy src/tbots/core` |

**Superseded, not deleted:** TASK-054's `rl/wrappers/domain_rand.py` (Gym
wrapper on the observation vector). Replaced by `PerceptionSim` (TASK-075).
Its latency-queue code is reusable; its interface is not.

---

## 3. Tier 1: blocks recruitment

Ordered by the critical path. Later tasks depend on earlier ones unless
noted.

### 3.1 Fixes the review found

Three of the four are done — see §2. What is left:

| ID | Task | Size | Done when |
|---|---|---|---|

### 3.2 Backend and core types

| ID | Task | Size | Done when |
|---|---|---|---|

### 3.3 Perception in the training loop

| ID | Task | Size | Done when |
|---|---|---|---|
| **TASK-075** | **`PerceptionSim`** in `perception/sim.py`. Takes ground-truth `WorldState`, emits delayed, noisy, dropout-marked state. Two outputs: a `WorldState` (`noisy` mode) and `DetectionFrame`s plus synthetic `RobotFeedback` (`tracked` mode). Parameters randomised per episode from config. | M | Test: with zero noise the output equals the input shifted by the sampled delay; with dropout, some robots are `visible=False`; seeded runs are identical. |
| **TASK-020** | **`perception/tracker.py`.** Association, velocity estimation, extrapolation through dropouts, forward-prediction to `t = now + actuation delay`. Consumes `DetectionFrame`s, applies `Perspective` once at the end. Was Tier 2; `tracked` mode needs it, but `noisy` mode keeps day one unblocked if it slips. | L | Test: feed frames synthesised from a known rSim trajectory with 30 ms latency; tracked positions at `t` are within 2 cm of truth at `t`. |

### 3.4 Environment and the contracts recruits touch

| ID | Task | Size | Done when |
|---|---|---|---|
| **TASK-076** | **Two-world `SSLEnv` and `Transition`.** The env holds truth and perceived state; observations come from the perceived world, everything else from truth. Reward terms take a frozen `Transition(world, prev, commands, ego_id, geometry, dt, done)`. `done` is an enum. Update `registry.py` and `example.py`. | M | `pytest tests/test_core.py tests/test_rewards.py`; a term that reads `tr.commands` and `tr.done` runs in one episode. |
| **TASK-078** | **Action codec** in `rl/codec.py`. `Box(vx, vy, vtheta)` plus kick and dribbler booleans; booleans map to speeds from skill kwargs; acceleration limits from `core`; 6.5 m/s kick clamp; `action_repeat`. Named and registered. | M | Golden test: a fixed action sequence decodes to a fixed `RobotCommand` sequence; a step change in commanded velocity is rate-limited. |
| **TASK-053** | **Observation builders**, reshaped. Egocentric builder for skills, global fixed-size builder for tactics, fixed scaling constants, no robot-ID value anywhere. Golden-vector test per builder; builders are immutable once a checkpoint exists (a change is a new name). The set encoder proper is Tier 2. | M | `obs_size(name)` matches `build_obs` output for every registered name; golden vectors pass; TASK-081 passes. |
| **TASK-079** | **`SkillTask` registry and scaffold.** A registered dataclass grouping scenario sampler, success and failure predicates, obs builder, codec, time limit, default reward YAML. `python -m tbots.rl.new_task <name>` writes the file with every hook documented, a reward YAML, and a test. Must be simple enough for a recruit's first afternoon. | M | `new_task demo` followed by `pytest tests/test_task_demo.py` passes on a clean checkout; `train env=skill task=demo train.total_steps=1000` runs. |
| **TASK-050** | **`rl/envs/skill_env.py`**, now a thin env over a `SkillTask`: spaces from the codec and builder, `done` from the task predicates, out-of-bounds defaults to terminate. | S | One episode of `go_to_ball` runs in `truth`, `noisy` and `tracked` modes. |
| **TASK-006** | **`make_env(cfg)`** in `rl/factory.py`: backend → `PerceptionSim` → tracker → env → wrappers. The only construction path for train, eval, tests and benchmarks. The network-config half of this task stays Tier 2. | S | `make_env` builds every combination in `configs/`; `rl/train.py`, `apps/eval.py` and `rl/vec.py` call nothing else. |
| **TASK-007** | **Skill registry wiring.** `skills/__init__.py` imports nothing, so `skill_names()` is `[]` and `build_skill("go_to_point")` raises. | S | Test: registry non-empty and round-trips. |

### 3.5 Learning

| ID | Task | Size | Done when |
|---|---|---|---|
| **TASK-056** | **The trainer.** Vendored CleanRL-style PPO in `rl/algo/ppo.py` around a composable action distribution (Box, Bernoulli, and per-robot factorised heads for MAPPO later). Truncation bootstraps, termination does not. `rl/train.py` wires `make_env`, the step loop, TensorBoard, `artifacts.save_checkpoint`, and a short eval at each checkpoint interval. | L | `train env=skill task=go_to_ball` reaches a positive mean episode return within a documented step budget; `runs/NAME/{latest.pt, config.yaml, events.out.tfevents.*, rollouts/}` exist; a truncated-episode test checks the bootstrap. |
| **TASK-059** | **Checkpoint metadata**, extended: builder name and size, codec, `dt`, `action_repeat`, acceleration limits, perception mode, stage, git SHA. Refuse on builder/codec/`dt` mismatch, warn on perception mode, never check the SHA. | S | Tests for each refusal and the warning. |
| **TASK-080** | **Seeding and determinism.** One seed through scenarios, perception noise, opponents, policy sampling. | S | Two runs with the same seed produce identical first 100 transitions per worker. |
| **TASK-081** | **Robot-ID relabel invariance.** Relabel `us` and `them` IDs, assert every builder's output and a policy's output are unchanged. | S | The test exists and passes for every registered builder. |
| **TASK-055** | **`rl/vec.py`**: subprocess vector env via `make_env`, `OMP_NUM_THREADS=1`, and the throughput benchmark. No in-run reconfigure. | S | A measured steps/s number for 1, 8 and 32 workers is written into README. |

### 3.6 Watching it play

| ID | Task | Size | Done when |
|---|---|---|---|
| **TASK-058** | **`apps/eval.py` and rollouts.** Deterministic policy on a fixed seed set, `tracked` perception, configured opponent; reports goal difference and per-term reward distributions per episode; records episodes as rollout files (`rl/rollout.py`), keeping the newest ten. `--render vision --realtime` streams live. | M | `eval --checkpoint runs/x/latest.pt` writes `runs/x/eval.json` and `runs/x/rollouts/*.npz`; a 6v6 two-minute rollout is under 5 MB. |
| **TASK-082** | **`apps/replay.py`.** Streams a rollout file to `ssl-vision-client` through `VisionPublisher`, pausable and seekable. | S | A recorded skill episode plays back in the browser at realtime. |

### 3.7 Trust in the build

| ID | Task | Size | Done when |
|---|---|---|---|
| **TASK-005** | Verify the macOS path end to end. Step 1M has never been executed by anyone. | M | A Mac completes ONBOARDING §1.1 to §1.7 with notes. |
| **TASK-008** | Fresh-clone rehearsal by someone who did not build this. Every stumble is a doc bug. | M | A second person's notes are folded into ONBOARDING. |
| **TASK-009** | GitHub issues with owners for every Tier 1 task; a review rotation; branch protection on `main`. | S | Every row in this section has an issue link. |

### 3.8 The cluster

| ID | Task | Size | Done when |
|---|---|---|---|
| **TASK-060** | Commit `env-atlantis.sh` and the shared-prefix recipe. Today the script is gitignored and exists in one checkout. | S | A fresh clone on Atlantis sources it and `pytest -q` passes. |
| **TASK-061** | Commit `scripts/train_atlantis.slurm`. Delete SETUP.md Step 17's Apptainer recipe or label it "for a future cluster". | S | `sbatch` of the committed file runs. |
| **TASK-062** | Pin the torch/CUDA wheel and index for the GPU nodes; record which node types it supports. | S | `uv pip install -e ".[dev,train]"` on Atlantis reproduces the pinned wheel. |
| **TASK-063** | W&B entity and project; confirm whether compute nodes have outbound internet. | S | A job logs offline and syncs from the login node. |
| **TASK-064** | A cluster smoke job that runs `pytest -q` plus the benchmark, independent of the trainer. | S | The job writes to `logs/` and exits 0. |

---

## 4. Tier 2: first weeks of fall

| ID | Task | Notes |
|---|---|---|
| **TASK-077** | **The `Coach`** in `tactics/coach.py`: owns tactic, live skill instances, roles, restarts; `tick(world) -> list[RobotCommand]`; fixed 4 Hz decisions; skill persistence by `SkillSpec` equality; the rule filter (HALT, 0.5 m on STOP, 0.2 m from the opponent defense area, defense-area occupancy, stop speed). Done when: a test drives the Coach through every `Play` and no command violates the filtered rule. |
| **TASK-051** | **`rl/envs/tactics_env.py`**: drives the Coach, accumulates reward over the 15 inner ticks, out-of-bounds defaults to restart via `place()`. Needs TASK-077 and the set encoder. |
| **TASK-052** | **`SyntheticReferee`**: `GameState` from truth; excessive dribbling, double touch, no progress as `done` reasons with a free-kick restart; episodes may start in restart states. Contact fouls skipped (open problem). |
| **TASK-083** | **Set encoder** for tactics observations: DeepSets or attention pool, fixed-size output, pointer-style targets for `pass_to` and `mark`. Region and zone discretisation for `dribble_toward` and `reposition`. |
| **TASK-084** | **Factorised MAPPO heads**: per-robot independent heads on the shared encoder, joint log-prob as a sum, centralised critic, through the TASK-056 distribution interface without touching the trainer loop. |
| **TASK-057** | **Opponent pool**: uniform over frozen checkpoints plus the scripted baseline; config override pins one; opponent Coach runs on `as_opponent(world)` with ground truth. |
| **TASK-041** | **`tactics/scripted.py`**: the baseline opponent. Nothing is evaluable at 6v6 without it. Do not let it slip past week two. |
| TASK-030 to 037 | Skill implementations: `FacePoint`, `Shoot`, `PassTo`, `ReceivePass`, `Dribble`, `Intercept`, `Goalkeep`, `LearnedSkill`. Each learned one is a `SkillTask`. Classic recruit work. |
| TASK-040, 042, 043 | Roles (Hungarian), learned-tactic wrapper, restart routines. All live inside the Coach. |
| TASK-014 | `NetworkBackend.observe()`: merge frames, run the tracker, attach `GameState`, read geometry from the wire and fail on a Division mismatch. Unblocks the two-backend parity test. |
| TASK-010, 011, 012 | `net/vision.py`, `net/robot_control.py`, `apps/wiggle.py`. Moved out of Tier 1 on 2026-09-04: the review scoped itself to training, and these prove the match path. Still cheap and worth doing early. |
| TASK-015 | `net/sim_control.py`: ER-Force's `place()`. Blocked on the proto collision in SETUP_LOG Step 7. |
| TASK-018 | AutoReferee in the compose stack. |

---

## 5. Tier 3: after the season starts, or never

| ID | Task | Notes |
|---|---|---|
| TASK-016 | GC team client (TCP 10008, RSA signing). |
| TASK-017 | GC CI-mode client: drive the real referee from rSim. |
| TASK-085 | GIL release in rSim `step()`. Only if TASK-055's number disappoints. |
| — | Refresh the pins: GC submodule, docker image and `fetch_tools.sh` together. |
| — | `docs/PINNED_VERSIONS.txt` is stale; regenerate. |
| — | ER-Force geometry omits `line_thickness`, so `ssl-vision-client` 2.1.1 on :8082 draws every field line and both goals 0 mm wide during networked matches. The client ignores `field_lines` and redraws from the scalars. SETUP_LOG "viz_rsim flickers against the ER-Force simulator". |
| — | Doc drift: `CLAUDE.md` says `~/code/tritonbots`, repo is `~/tritonbots`; `docker-compose.yml` refers to `src/tbots/gen/`, path is `src/tbots/_pb/`. |

---

## 6. Open problems

Interim answers, not decisions. Detail in `docs/ARCHITECTURE.md` §13.

- Contact fouls in the synthetic referee.
- Observation scaling constants.
- The exact stop-speed threshold for the rule filter.
- Match path: radio frame, firmware heading field, real telemetry. Out of scope.
- Set encoder architecture and zone discretisation.

---

## 7. What changed on 2026-09-04, and why

The architecture review walked the whole design tree for the training
pipeline. Four findings were bugs, not opinions:

- rSim ran at 17 ms per step, not 1/60 s (TASK-070).
- The match simulator ran on a Division A field (TASK-071).
- `RSimBackend` silently dropped commands for unknown IDs (TASK-072).
- No path existed to command opponents, so self-play was impossible
  (TASK-072).

The decisions that reshaped tasks:

- **Perception moved below the environment.** The observation-vector wrapper
  (TASK-054) is superseded by `PerceptionSim` (TASK-075) with three modes,
  and the tracker (TASK-020) moved into Tier 1 because `tracked` mode runs
  it in training.
- **The `Backend` protocol split** into `Backend` and `SimBackend`
  (TASK-072). `reconfigure()` is gone.
- **The Coach** (TASK-077) is the one play loop; `TacticsEnv` drives it on
  a fixed 4 Hz tick.
- **Reward terms take a `Transition`** (TASK-076), decided now so the
  signature never changes under recruits.
- **`SkillTask` and a scaffold** (TASK-079) replace the six-file procedure
  for training a skill.
- **A curriculum is a sequence of runs** with human promotion. Automatic
  `promote_when` and in-run stage changes are deleted; `curriculum_example.yaml`
  becomes documentation.
- **Evaluation** (TASK-058) uses a fixed seed set and records rollouts;
  `apps/replay.py` (TASK-082) plays them back.
- **The trainer** (TASK-056) is a vendored PPO with a composable
  distribution, so MAPPO heads (TASK-084) drop in later.

Three config files still describe features that do not exist:
`domain_randomization` (now `perception`), `encoder: set_encoder`
(TASK-083), and `promote_when` (deleted). They are harmless until
`make_env` reads them. Land TASK-006 with the tasks whose keys it activates.

---

## 8. If you only get five things done

1. ~~**TASK-070, 071, 072.** The fixes.~~ **Done 2026-09-07.** TASK-004 is
   the remainder of §3.1: it is what stops any of the three regressing.
2. **TASK-076 + 078 + 079 + 050.** The contracts a recruit touches, frozen
   before anyone writes against them.
3. **TASK-056.** The trainer.
4. **TASK-058 + 082.** Without them a recruit trains a policy and never
   watches it play.
5. **TASK-060 + 061.** Atlantis reproducible from a clone.

TASK-005 (macOS) is the wildcard: zero cost if nobody brings a Mac, a
day-one blocker for every recruit who does.
