# Setup log

One block per step, per CLAUDE.md §3.

## Step 1 — System prerequisites (verification only)   [PASS]
Verification: `pkg-config --modversion ode` -> `0.16.2`
              `pkg-config --variable=libdir ode` -> `/usr/local/lib`
              `grep dDOUBLE /usr/local/include/ode/precision.h` -> `#define dDOUBLE` (x2)
Deviations:   None. Per CLAUDE.md §2, Step 1 was already performed by the human;
              only verification was run here, no installation and no sudo.
Notes:        libdir correctly resolves to `/usr/local/lib`, not
              `/usr/lib/x86_64-linux-gnu` — confirms the source-built ODE is
              the one being picked up, not Ubuntu 24.04's identically
              versioned packaged `libode-dev`.

## Step 2 — Create the repository                [PASS]
Verification: `find . -type d -not -path './.git*' | sort` -> matches the
              tree in SETUP.md Step 2.
Deviations:   The directory tree and `src/tbots/**/__init__.py` markers
              already existed from a prior commit (374f51f). Only the
              missing pieces were added here: `.gitignore` and
              `tools/bin/.gitkeep`.
Notes:        `.claude/` is a harness artifact, not part of SETUP.md's
              tree; harmless to have alongside it.

## Step 3 — Python environment                   [PASS]
Verification: `python -c "import tbots; print('ok')" 2>/dev/null || echo "expected..."` -> `ok`
              `python -c "import numpy, google.protobuf, gymnasium, torch; print('deps ok')"` -> `deps ok`
Deviations:   `import tbots` succeeded and printed `ok` instead of hitting
              the "expected" fallback in SETUP.md. SETUP.md's comment
              assumes the `tbots` package doesn't exist yet at this point;
              in our case Step 2's skeleton already created
              `src/tbots/__init__.py` (empty), so the editable install
              makes it trivially importable. Not a failure — the package
              still has no real content, it's just present rather than
              absent.
Notes:        `uv venv --python 3.11` downloaded a standalone CPython
              3.11.15 (uv-managed, not apt). `uv pip install -e ".[dev,train]"`
              resolved and installed 74 packages (torch 2.13.0, numpy 2.4.6,
              gymnasium 1.3.0, protobuf 5.29.6, etc.) cleanly, ~3.5 min due
              to CUDA wheel downloads for torch.

## Step 4 — Fork and build rSim                  [PASS]
Verification: `python -c "import robosim; print(robosim.__file__)"` ->
              `<venv>/lib/python3.11/site-packages/robosim/__init__.py`
              `ldd <_robosim .so> | grep ode` ->
              `libode.so.8 => /usr/local/lib/libode.so.8`  (OUR ODE, not the
              distro package — SETUP.md calls this the single most valuable
              check in the setup)
Deviations:   1. Submodule URLs. Used the CLAUDE.md §2 fork URLs
                 (github.com/YashTandon05/{rSim,rSoccer}); SETUP.md's
                 `github.com/tritonbots/*` URLs are placeholders and 404.
              2. pybind11 lives in CMakeLists.txt, not setup.py/pyproject.toml
                 as SETUP.md's failure table states. Bumped the FetchContent
                 pin 2.6.2 -> 2.13.6 there, and recomputed URL_HASH from the
                 actual tarball rather than guessing it.
              3. pybind11 2.6.2 failed for a reason SETUP.md does not list:
                 its own CMakeLists declares cmake_minimum_required < 3.5,
                 which CMake 4.x has REMOVED support for. It failed at
                 configure time, before the compiler ran. The 3.11 support
                 issue was real too, but this one hit first.
              4. `uv pip install -e .` (as SETUP.md specifies) SILENTLY
                 installs a broken package — see the finding below. Used a
                 non-editable `uv pip install .` instead.
              5. SETUP.md's ODE check `ldd "$(python -c 'import robosim;
                 print(robosim.__file__)')"` runs ldd against `__init__.py`,
                 a text file. Adapted it to target `robosim._robosim`'s
                 compiled .so, which is what actually links ODE.
              6. Repo lives at `/home/ytandon/tritonbots`, not
                 `~/code/tritonbots` as CLAUDE.md §2 states. Still inside the
                 WSL filesystem (not /mnt/c), so the 9p warning does not apply.
Notes:        C++ fixes needed for GCC 13, both committed and pushed to our
              fork (69f0d8e) and the submodule pointer bumped here:
                - `#include <cstdint>` added to src/robosim/sslworld.cpp and
                  src/robosim/vssworld.cpp (uint32_t/int32_t at sslworld
                  379/430/438 and vssworld 356/400/408). Exactly the
                  near-certain failure CLAUDE.md §6 predicted.
                - CMAKE_CXX_STANDARD 11 -> 17.
              Python 3.11.15 throughout. NO fallback to 3.10 was needed or made.

              *** FINDING (RESOLVED in a later session) — editable install
              was silently broken ***
              `uv pip install -e .` exited 0 and reported success, but never
              ran cmake: no _skbuild dir, no .so produced anywhere. It just
              wrote a .pth pointing at src/, so `import robosim` resolved
              but `from ._robosim import VSS, SSL` raised ModuleNotFoundError.
              Cause: the fork's pyproject.toml set
              `build-backend = "setuptools.build_meta"` while setup.py relied
              on `skbuild.setup`. Under PEP 660, setuptools handles the
              build_editable hook itself and scikit-build's cmake logic is
              never invoked. Classic scikit-build has no PEP 660 support.
              Impact: SETUP.md Step 4.4's own command, and Step 15.4's CI line
              `uv pip install -e third_party/rsim`, both produced a broken
              install. The Step 4.4 verification did catch it (import fails
              loudly), so it was not dangerous — just wrong as written.
              Workaround used at the time: non-editable `uv pip install .`.

              RESOLUTION: migrated the fork to `scikit-build-core`
              (commit `a2a7c70`, merged `39eacb1`, pushed), which is itself a
              PEP 517/660 backend and supports editable installs natively.
              `pyproject.toml` now declares
              `build-backend = "scikit_build_core.build"` with
              `wheel.packages = ["src/robosim"]` and
              `wheel.install-dir = "robosim"`; `setup.py` removed (nothing
              reads it under this backend). Verified in two disposable venvs,
              independent of this repo's working venv: editable install
              genuinely compiles the extension, links
              `/usr/local/lib/libode.so.8`, passes a full SSL()/step()/
              get_state() functional test, and a live edit to
              `src/robosim/__init__.py` (no reinstall) is picked up on the
              next import; non-editable install verified equivalently; the
              full rsoccer_gym pipeline re-verified against the new build.
              C++ changes still require a reinstall (deliberately did not set
              `editable.rebuild = true` — that would put a cmake/ninja/
              network round-trip on every `import robosim`, including inside
              training subprocess workers). Submodule pointer in this repo
              bumped to `39eacb1`; main venv's `robosim` reinstalled
              editable and re-verified against Steps 4/5/6's gates, all still
              PASS. SETUP.md Step 4.4 updated with a note on why `-e .` now
              works, so this pairing is not reintroduced by accident.

## Step 5 — Install rSoccer                      [PASS]
Verification: SETUP.md's script as written FAILS —
              `gym.make("SSLGoToBall-v0")` -> `NameNotFound: Environment
              `SSLGoToBall` doesn't exist.` That env ID does not exist in the
              pinned rSoccer revision. Ran the equivalent check against the
              env IDs the library actually registers; all 5 reset AND step:
                SSLStaticDefenders-v0      obs=(24,) act=(5,)  OK
                SSLDribbling-v0            obs=(21,) act=(4,)  OK
                SSLContestedPossession-v0  obs=(14,) act=(5,)  OK
                SSLPassEndurance-v0        obs=(16,) act=(3,)  OK
                VSS-v0                     obs=(40,) act=(2,)  OK
Deviations:   1. `SSLGoToBall-v0` is stale in SETUP.md. Substituted the real
                 registered IDs (listed above) rather than inventing a name.
                 The gate's intent — prove rSoccer and rSim work together — is
                 preserved and in fact strengthened: SETUP.md only calls
                 reset() on one env; this steps all five.
              2. The pin appears in BOTH files, not just setup.py as SETUP.md
                 implies: `pyproject.toml` `rc-robosim = "^1.2"` and
                 `setup.py` `"rc-robosim >= 1.2.0"`. Removed from both.
                 The effective build backend is `poetry.core.masonry.api`, so
                 pyproject.toml is the one that actually governs; setup.py is
                 vestigial but was carrying the same pin.
Notes:        Pin removed BEFORE installing, per CLAUDE.md §5. Confirmed no
              rc-robosim was fetched from PyPI, and that our source build is
              untouched afterwards: robosim still resolves to
              `<venv>/.../robosim/_robosim.cpython-311-x86_64-linux-gnu.so`,
              still links `/usr/local/lib/libode.so.8`, and its dist-info
              direct_url.json still reads
              `file:///home/ytandon/tritonbots/third_party/rsim`.
              This pin would definitely have fired rather than being
              harmlessly satisfied: our build reports version
              `0.1.dev134+g69f0d8e`, which does not satisfy `^1.2`.
              Unrelated latent issue, NOT hit: setup.py also pins
              `protobuf == 3.20.2`, which conflicts with our own
              `protobuf>=4.25` (5.29.6 installed). Harmless today because the
              poetry backend ignores setup.py entirely — but it would bite
              if anyone ever switches rSoccer's backend to setuptools.
              Committed to our fork (b9a0a63); submodule pointer bumped here.

## Step 6 — Verify rSim's undocumented behaviour   [PASS]
Verification: `python scripts/verify_rsim.py` -> exit 0, four facts
              established; `docs/RSIM_FACTS.md` written with the answers in
              plain English at the top and zero remaining placeholders.
              THE FOUR FACTS:
                Division B field_type = 1   (NOT 0)
                BALL_STRIDE = 5, ROBOT_STRIDE = 11  (len 137 for 6v6)
                action vector length = 8
                angles: DEGREES out (poses, heading, vdir),
                        RADIANS/sec in (commanded vangular)
Deviations:   Rewrote scripts/verify_rsim.py rather than transcribing it.
              CLAUDE.md §6 sanctions this ("a probe to iterate on"); as
              printed it could not produce correct answers:
              1. It hardcoded `field_type=0` in PARTS 2 and 3 after PART 1
                 was supposed to establish the value. field_type=0 is
                 Division A here, so those parts measured the wrong field.
                 The rewrite discovers the value and feeds it forward.
              2. It hardcoded `ACT_LEN = 6` with a comment to adjust by hand.
              3. Its action-length test assumed a too-short action vector
                 raises. It does not — see the finding below. As written the
                 probe reports "action length 6 -> ACCEPTED" and is wrong.
              4. Added PART 5 (velocity semantics) and a source cross-check
                 throughout: where a fact is written literally into the C++,
                 the script quotes it and then confirms it at runtime.
              Also probed each field_type in a separate subprocess first, so
              an invalid one could not take the whole run down with it.
Notes:        *** THREE FINDINGS THAT INVALIDATE VALUES ALREADY IN SETUP.md ***
              These are reported, not fixed — Steps 9 and 14 are out of scope
              for this session. They must be corrected when those steps land.

              (a) `field_type` must be 1, not 0. SETUP.md Step 9.2 ships
                  `FIELD_TYPE_DIV_B: int = 0` and Step 14.3's
                  configs/env/div_b_6v6.yaml ships `field_type: 0`. Both give
                  a 12x9 m Division A pitch. Nothing raises — we would just
                  silently train Division B policies on the wrong field.

              (b) `ACTION_LEN` must be 8, not 6, and SETUP.md's collapsed
                  offsets `A_KICK_FLAT, A_KICK_CHIP, A_DRIBBLER = 4, 5, 5`
                  are wrong. Real layout: [0] use-wheels flag, [1][2][3]
                  local vx/vy/vangular, [4] wheel3, [5] flat kick, [6] chip
                  kick, [7] dribbler. Critically, a wrong length does NOT
                  raise: setActions() uses std::vector::operator[], which is
                  unchecked, so a 6-element action reads two elements past
                  the end and feeds garbage to the kicker and dribbler.
                  SETUP.md's `if ACTION_LEN >= 8 ... else` branch takes the
                  wrong branch and fires the kicker on out-of-bounds memory.

              (c) Angle units are ASYMMETRIC, which SETUP.md's single
                  `ANGLES_IN_DEGREES` boolean cannot express. Out of the sim
                  (heading, vdir, and reset/ctor poses) is degrees; INTO the
                  sim, the commanded angular velocity in action slot [3] is
                  radians/second. SETUP.md's rsim.py happens to get this
                  right by accident (it converts state deg->rad and passes
                  c.vtheta through unconverted), but the single flag implies
                  both directions share a unit, and they do not.

              Two further behaviours worth knowing, documented in RSIM_FACTS:
              - get_state() must be called EXACTLY ONCE per step(). Velocities
                are finite-differenced against the previous get_state() call
                and always divided by one timeStep regardless of elapsed time.
                Call it once per 10 steps and a robot moving 1.0 m/s reports
                vx = 9.9979. Two calls back to back report 0.0.
              - A heading of exactly 0 is reported as 360.0; getDir()'s range
                is (0, 360], not [0, 360).
              Verified against rSim fork commit 69f0d8e on Python 3.11.15,
              linked to /usr/local/lib/libode.so.8.

## Step 7 — Protobuf submodules and code generation   [PASS]
Verification: SETUP.md's script as written FAILS at the first line —
              `from tbots._pb.ssl_gc_referee_message_pb2 import Referee`
              -> `ModuleNotFoundError`. The module is at
              `tbots._pb.state.ssl_gc_referee_message_pb2` (see Deviation 2).
              With the path corrected:
                Referee stages:   NORMAL_FIRST_HALF_PRE, NORMAL_FIRST_HALF,
                                  NORMAL_HALF_TIME, NORMAL_SECOND_HALF_PRE,
                                  NORMAL_SECOND_HALF ...
                Referee commands: all 18 present, exactly the list SETUP.md
                                  predicts (HALT ... BALL_PLACEMENT_BLUE)
                proto ok
              Also confirmed all 23 generated modules import together in one
              process (0 failures), and that `make proto` regenerates
              idempotently.
Deviations:   1. Proto paths all exist as SETUP.md expects. CLAUDE.md §6 warned
                 `protos/ssl-vision/src/shared/proto` might differ in the
                 pinned revision — it does not. Checked all three before
                 running: GC 20 protos, sim 11, ssl-vision 7 (38 total).
              2. GC modules are NESTED, not flat. The game controller's protos
                 import each other subdirectory-qualified
                 (`import "state/ssl_gc_common.proto";`), so the include root
                 must be `protos/ssl-game-controller/proto` and the generated
                 tree mirrors upstream: `_pb/state/`, `_pb/rcon/`, `_pb/geom/`,
                 `_pb/tracker/`, etc. Flattening would break those imports, so
                 this is not fixable — the import path is what changes.
                 Corrected in SETUP.md: Step 7's verification, Step 10.2's
                 `net/referee.py`, and Step 16's acceptance check 1 all used
                 the flat path and would all have failed.
                 The sim-protocol and ssl-vision protos import by bare
                 filename, so those land flat as SETUP.md assumed.
              3. gen_proto.sh substantially rewritten — it could not work as
                 printed. Two independent reasons, see Notes.
              4. GC's `v3.21.0` is a LIGHTWEIGHT tag, so `git submodule
                 status` / `git describe` report the misleading
                 `v2.7.1-774-ge51e1c7`. Confirmed HEAD e51e1c7 == v3.21.0^{commit}
                 exactly. Added a header to docs/PINNED_VERSIONS.txt so nobody
                 misreads the pin as v2.7.1.
              5. Tag availability differs from SETUP.md's framing ("Pin them to
                 a tag. Never track master"): ssl-simulation-protocol has NO
                 tags at all and ssl-vision's only tags are from 2014/2017.
                 Both are pinned by SHA off master, which is what the committed
                 gitlink records — SETUP.md's own inline comments already
                 anticipate this. Only the GC could be pinned to a real tag.
                 Kept v3.21.0 rather than the now-current v3.23.0, so the
                 protos match the GC binary/image version Step 13 and
                 docker-compose.yml pin. Bump all three together, deliberately.
Notes:        *** FINDING — not all league protos can be generated together ***
              None of these .proto files declare a `package`, so every message
              lands in the GLOBAL protobuf namespace, and all three repos
              vendor their own private copy of the SSL vision protos. Across
              the three submodules 26 top-level symbols are multiply defined:
              Team, Division, RobotId, SSL_DetectionFrame/Ball/Robot,
              SSL_GeometryData, SSL_FieldShapeType, SSL_WrapperPacket,
              TrackedFrame, TrackerWrapperPacket, Vector2, Vector3, ...
              protobuf's descriptor pool is process-global and keyed by symbol
              (impl here is `upb`), so the second definition raises at IMPORT
              time: `TypeError: Couldn't build proto file into descriptor
              pool: duplicate symbol 'Team'`.
              Measured: generating all 38 protos yields a package where 13 of
              38 modules cannot be imported, and which 13 depends on import
              ORDER. That is the failure SETUP.md's script would have shipped.
              Additionally, protoc cannot be given all three include roots at
              once (as SETUP.md does): the GC protos import deps
              subdirectory-qualified while the sim protos import the same
              names bare, so a shared -I list makes protoc resolve one
              physical file under two canonical names and collide with itself.
              That is the error the original script actually died on.

              RESOLUTION: each repo is compiled against only its own include
              root, and we generate a curated conflict-free subset — 23 of the
              38 protos — choosing the authoritative repo per SETUP.md 7.1's
              own ownership note and dropping the vendored duplicates. The
              selection was computed by taking the dependency closure of the
              protos we actually need, then greedily adding every other proto
              that introduces no symbol collision. Exclusions and rationale
              are in gen_proto.sh's header comment.
              Verified: all 23 import together, 0 failures.

              TWO CAPABILITIES THIS COSTS, both blocked rather than merely
              omitted — they cannot be recovered by regenerating:
              - `SimulatorCommand` is unavailable. sim-protocol's
                ssl_simulation_control.proto imports its VENDORED
                ssl_gc_common.proto, whose Team/Division/RobotId collide with
                the GC's own state/ssl_gc_common.proto that Referee needs.
                Referee and SimulatorCommand genuinely cannot coexist in one
                process as these repos are published. Blocks TASK-015
                (net/sim_control.py). Not on the match path — it is only used
                to teleport for episode resets against a simulator, and
                SETUP.md 14.2 already says that port is locked down at
                tournaments.
              - The GC `ci/` protos are unavailable (they pull the same
                vendored vision copies), blocking TASK-017's ci-mode client.
              Everything the match path needs IS present and coexists:
              Referee, RobotControl, RobotControlResponse, SSL_WrapperPacket,
              TrackerWrapperPacket, and the rcon team-client protos.

              Confirmed this is the modern GC proto, not archived refbox:
              TeamInfo has `goalkeeper` and no `goalie`. All fields Step 10.2
              depends on exist — command_counter, designated_position,
              blue_team_on_positive_half, current_action_time_remaining,
              stage_time_left, max_allowed_bots, yellow_cards, red_cards.

              Minor: protoletariat rewrites the generated `.py` imports
              correctly (verified: `from ..state import ...`, `from . import
              ...`, no un-rewritten top-level pb2 imports remain) but does NOT
              rewrite the `.pyi` type stubs, which keep flat imports. Harmless
              — stubs are never executed, `_pb` is gitignored, and the
              Makefile's lint target only runs mypy over `src/tbots/core`.
              Would matter only if a type-checker is ever pointed at `_pb`.

## Step 8 — The core contracts            [PASS]
Verification: the Step 8 python heredoc (imports WorldState/RobotCommand/DIV_B/
              wrap_angle) ->
                their goal: (4.5, 0.0)
                dist to ball: 1.0
                wrap(3pi): 3.141593
                clamped: RobotCommand(robot_id=0, vx=3.0, vy=0.0, vtheta=0.0,
                         kick_speed=0.0, chip=False, dribbler=0.0)
                core ok
Deviations:   None. Five files transcribed verbatim from SETUP.md 8.1-8.5:
              core/units.py, core/geometry.py, core/gamestate.py,
              core/state.py, core/command.py.
Notes:        Rule 1 holds — nothing in core/ imports outside core/. The only
              intra-package imports are state.py -> core.gamestate and
              command.py -> core.geometry.
              SETUP.md 8.2 carries a standing caveat worth repeating: the DIV_A
              / DIV_B constants are 2025-rulebook values and should be checked
              against the current rulebook before the first match; at runtime
              prefer the geometry from the SSL-Vision SSL_GeometryData packet
              and treat these as fallback.

## Step 9 — The backend layer            [PASS]
Verification: SETUP.md Step 9 specifies no verification command, only the
              commit. Smoke-checked instead:
                RSimBackend(6,6).reset(Scenario.single_robot_at(-1.0, 0.5))
                -> us0 (-1.0, 0.5), n_us 6, n_them 6  (state-length guard in
                reset() did not fire, so BALL_STRIDE/ROBOT_STRIDE agree with
                the installed rSim)
Deviations:   None. backends/base.py, backends/rsim.py and backends/network.py
              transcribed verbatim from SETUP.md 9.1, 9.2 and 9.3.
Notes:        Constants in backends/rsim.py were cross-checked against
              docs/RSIM_FACTS.md rather than taken on trust:
                FIELD_TYPE_DIV_B = 1, BALL_STRIDE = 5, ROBOT_STRIDE = 11,
                ACTION_LEN = 8, A_KICK_FLAT/A_KICK_CHIP/A_DRIBBLER = 5/6/7,
                ANGLES_IN_DEGREES = True, and vtheta passed to slot [3] with
                NO conversion (rSim's asymmetry: degrees out, radians in).
              SETUP.md Step 9.2 had already been corrected to these values in
              commit 1d2baa4, so transcription and RSIM_FACTS.md agree; no
              placeholder values were carried over.

              backends/network.py is not importable yet and will not be until
              TASK-010/011/015/020 land — it imports net.vision,
              net.robot_control, net.sim_control and perception.tracker at
              module level, and net/sim_control.py is itself blocked on the
              missing SimulatorCommand proto (see Step 7). This is expected:
              nothing imports it, and tests/test_backend_parity.py only
              imports it inside the test that is skipped without
              TBOTS_NETWORK_TESTS=1.

              `make lint` FAILS on this faithfully-transcribed code, and I did
              not "fix" it (CLAUDE.md §8: transcribe, do not reformat). ruff
              0.16.2's default rule set flags 8 issues in backends/ and 2 in
              core/ — all cosmetic: unused `field` import in base.py, unused
              `import math` in rsim.py, `typing.Sequence` vs
              `collections.abc.Sequence`, quoted self-referential annotations
              that `from __future__ import annotations` makes unnecessary, and
              `int(round(...))`. SETUP.md's pyproject pins no ruff `select`, so
              the effective rule set drifts with the ruff version. Worth a
              human decision: either pin `[tool.ruff.lint] select` in
              pyproject.toml, or apply `ruff format`/`--fix` once across the
              tree after transcription is complete.

## Step 10 — The network layer            [PASS]
Verification: SETUP.md Step 10 specifies no verification command, only the
              commit. Three checks run instead:
              1. All nine new/affected modules import cleanly, including
                 backends/network.py (the stubs raise on construction, not on
                 import).
              2. Live loopback round trip through the real multicast socket:
                 VisionPublisher -> 224.5.23.2:10006 -> rx_socket/drain ->
                 SSL_WrapperPacket.ParseFromString. 2 datagrams received;
                 geometry decoded as 9000 x 6000 mm; ball (1.5, -0.25) m read
                 back as 1500.0, -250.0 mm; us -> robots_blue, them ->
                 robots_yellow with ids preserved. Confirms the m->mm
                 conversion, the bare-protobuf framing, and TTL 0.
              3. The Step 10.2 field-name check SETUP.md explicitly asks for:
                 built a Referee message and ran it through to_gamestate().
                 max_allowed_bots, current_action_time_remaining,
                 stage_time_left, blue_team_on_positive_half, goalkeeper,
                 designated_position and command_counter all exist and decode.
                 DIRECT_FREE_BLUE as blue -> Play.FREE_KICK ours=True,
                 microseconds -> seconds (5.0 / 300.0), and flip_x negated the
                 placement target to (-1.0, 0.5).
Deviations:   net/multicast.py, net/referee.py and net/vision_publisher.py
              transcribed verbatim from SETUP.md 10.1-10.3.
              perception/tracker.py transcribed verbatim from 10.5.
              The four files in 10.4 (vision, robot_control, sim_control,
              team_client) are given in PROSE, not as code blocks — SETUP.md
              says only "create these four with the given signatures and
              NotImplementedError bodies". I therefore wrote the signatures
              rather than transcribed them, and derived each one from what
              backends/network.py actually calls, so the stubs and their only
              caller agree:
                VisionReceiver.wait_for_next_frame(timeout) / .close()
                RobotControlSender.send(commands) / .close()
                SimControlSender.place(scenario, flip_state) / .close()
              plus the extras the prose names (vision frames(), control
              feedback() for RobotControlResponse, and the team-client
              goalkeeper / substitution / advantage methods). Docstrings carry
              the prose contract verbatim, including "merge, then tick once"
              and "one packet per tick, containing every robot". Bodies are
              `raise NotImplementedError("TASK-0xx")` — nothing implemented.
Notes:        net/sim_control.py is stubbed AND blocked: SimulatorCommand is
              not generated at all (Step 7's proto collision), so TASK-015
              cannot be started until that is resolved. The docstring says so
              and points at the Step 7 log entry.
              net/team_client.py notes the framing difference that catches
              people out — TCP 10008 is a length-delimited stream, not the
              bare protobuf datagrams every other interface uses.

## Step 11 — Skills, tactics, and RL placeholders            [PASS]
Verification: SETUP.md Step 11 specifies no verification command, only the
              commit. Checked instead:
              - all 30 modules under skills/, tactics/ and rl/ import cleanly
              - registries populate on import: skill_names() -> ['go_to_point'],
                reward_names() -> ['alive'] (via the rl/rewards/__init__.py
                self-registration import SETUP.md 11.6 asks for)
              - CompositeReward([{'name':'alive','weight':2.0}]) returns 2.0
                and records {'alive': 2.0} in .last
              - build_skill('go_to_point', target=(1.0, 0.0)).step(...) on a
                robot at (-1, 0) returns vx=2.5 (max_v, correctly clamped),
                vy=0.0, status 'running'
Deviations:   Transcribed verbatim from code blocks: skills/base.py (11.1),
              skills/go_to_point.py (11.2), tactics/base.py (11.4),
              rl/rewards/registry.py + rl/rewards/example.py +
              rl/rewards/__init__.py (11.6), rl/envs/base.py (11.7).

              The 16 stub files in 11.3, 11.5 and 11.8 are specified by TABLE
              (file, class, task id, one-line purpose), not as code blocks, so
              as in Step 10.4 I wrote signatures rather than transcribing them:
                skills/  face_point, shoot, pass_to, receive_pass, dribble,
                         intercept, goalkeep  -> class with the Skill protocol
                         methods reset/step/status, each raising
                         NotImplementedError("TASK-03x")
                skills/learned.py -> LearnedSkill, docstring transcribed
                         VERBATIM from 11.3 (SETUP.md gives it in full)
                tactics/ roles (assign_roles function — the table describes a
                         matching routine, not a Tactic), scripted, learned,
                         restarts -> Tactic subclasses with decide()
                rl/      envs/skill_env, envs/tactics_env (SSLEnv subclasses
                         overriding the _observe/_decode hooks base.py already
                         declares), envs/synthetic_referee, obs/builders,
                         wrappers/domain_rand, vec, train, opponents
              Every body is `raise NotImplementedError("TASK-0xx")`. Nothing
              on the task board was implemented — CLAUDE.md §4.
Notes:        Each stub docstring carries the warning attached to it in
              SETUP.md rather than just the file's name, so a recruit opening
              the file gets the reasoning without re-reading the spec —
              domain_rand says sim-to-real fails without it, obs/builders says
              the encoding must be permutation-invariant and fixed-size across
              curriculum stages (9.2a), restarts says trigger on
              GameState.counter not on .play, opponents says build it now.

              rl/envs/base.py's SSLEnv._observe/_decode raise a bare
              NotImplementedError with no task id. That is correct and is not
              a stub — they are abstract hooks for subclasses, and SETUP.md
              writes them that way. The task ids live in the subclasses.

## Step 12 — Runnable apps            [PASS]
Verification: SETUP.md Step 12 specifies no verification command beyond
              "run viz_rsim once without --realtime and write the steps/s
              number down". Ran all three:
              - `python -m tbots.apps.viz_rsim`
                -> 3600 ticks in 7.59s = 474 steps/s (7.9x realtime)
                Full 6v6 rSim + GoToPoint x6 + a vision packet published every
                tick, so this is end-to-end, not a bare physics number.
              - `python -m tbots.apps.ref_monitor --help` -> parses; the live
                path needs a running game controller (Step 13), not yet built.
              - `python -m tbots.apps.wiggle` -> NotImplementedError: TASK-012,
                as specified.
Deviations:   All three files transcribed verbatim from SETUP.md 12.1-12.3,
              including wiggle.py's module-level `raise` (it is written that
              way in the spec — the file is a placeholder, not an importable
              module).
Notes:        THE STEPS/S NUMBER: 474 steps/s, 6v6, WSL2, with vision
              publishing on. SETUP.md asks for this to go in the README —
              there is no README.md in the repo yet; SETUP.md creates it in an
              appendix section after Step 17, which is outside the steps I was
              asked to build. Recording it here so it is not lost.

              Throughput is dominated by ODE contact handling, not by the
              publisher: a deliberately pathological run with all 12 robots
              commanded vx=1.0 into each other measured 97 steps/s 6v6, while
              1v1 measured 498 steps/s. Treat 474 as representative of normal
              play and expect it to sag in crowded scrimmages. Benchmark
              properly under TASK-055 before making a scaling decision on it.

## Step 15 — Tests and CI            [PASS]
Verification: `pytest -q` -> 10 passed, 1 skipped in 0.31s
              The one skip is test_backend_parity::test_network_converges
              ("requires a running simulator; set TBOTS_NETWORK_TESTS=1"),
              which is the correct state per SETUP.md 15.3.
              Breakdown: 5 core, 4 rSim backend, 1 rSim parity passing.
              test_rsim_converges reaches the target inside its 600-step
              budget in 0.04s, so GoToPoint genuinely converges in rSim — the
              acceptance test for the whole architecture, minus the network
              half that TASK-014 unblocks.
Deviations:   tests/test_core.py, tests/test_rsim_backend.py,
              tests/test_backend_parity.py and .github/workflows/ci.yml
              transcribed verbatim from SETUP.md 15.1-15.4.
Notes:        Two things about these tests a reader should not mistake for
              coverage, both faithful to the spec rather than my choices:
              - test_state_length_matches_constants asserts nothing. It passes
                by NOT raising: RSimBackend.reset() carries the state-length
                guard, so the test is really "the guard did not fire". Fine,
                but it will not catch a stride error that happens to preserve
                total length.
              - test_core.py imports `dist` and never uses it (ruff F401).
                Transcribed as written; not "fixed".

              CI CAVEAT — .github/workflows/ci.yml runs `make lint` before
              `make test`, and `make lint` currently FAILS (see the Step 9
              entry: ~10 cosmetic ruff findings across faithfully-transcribed
              core/ and backends/ code, plus the F401 above, under ruff
              0.16.2's default rule set). So the first push will show a red CI
              on the lint step even though every test passes. This needs a
              human decision before it is papered over: pin
              `[tool.ruff.lint] select` in pyproject.toml so the rule set stops
              drifting with the ruff version, or run `ruff check --fix` once
              across the tree now that transcription is done. I did neither —
              both are edits to SETUP.md-sourced code, which CLAUDE.md §8
              forbids me from making unilaterally.

              `make test` depends on `make proto`, so CI regenerates the
              protobufs before running; _pb is gitignored, as intended.

## Step 15 (follow-up) — pin ruff rule set, fix lint            [PASS]
Verification: `make lint` -> ruff "All checks passed!" + mypy "Success: no
              issues found in 6 source files"
              `pytest -q` -> 10 passed, 1 skipped (unchanged)
              Re-ran the app smoke checks afterwards, because the autofixes
              touched command.py's clamp helper and the referee/rsim modules:
              viz_rsim 440 steps/s; referee decode still FREE_KICK/ours/
              (-1.0, 0.5)/5.0s; clamped() still saturates to 3.0 / -3.0 / 6.5
              / 1.0.
              This clears the red-CI caveat recorded in the Step 15 entry
              above — ci.yml's `make lint` step now passes.
Deviations:   Reverses the earlier "transcribe, do not reformat" position on
              SETUP.md-sourced code, on explicit instruction from the human.
              Autofixes applied across src/, tests/ and scripts/: unused
              imports dropped, typing.Sequence -> collections.abc.Sequence,
              quoted self-referential annotations unquoted (safe under
              `from __future__ import annotations`), import order normalised,
              f-string `str(x)` -> `x!s`, redundant f-prefixes removed, and
              in core/command.py the clamp helper
              `lo if v < lo else hi if v > hi else v` -> `lo if v < lo else
              min(v, hi)`. All semantics-preserving, NaN behaviour included.
Notes:        THE RULE SET IS NOW PINNED in pyproject.toml under
              [tool.ruff.lint]: E, W, F, I, UP, B, C4. This is the actual fix
              — with no explicit `select`, ruff lints against whatever its
              current release defaults to, so a routine dependency refresh can
              turn CI red with no code change. That is exactly how the
              original failure appeared (SETUP.md's pyproject pinned only
              line-length; ruff 0.16.2's defaults had moved).

              RUF and FURB are deliberately NOT selected. Their only two
              findings here were noise and neither is worth editing code for:
                - RUF046 on `int(round(self._dt * 1000.0))` in backends/rsim.py
                  — redundant, but transcribed from SETUP.md and reads as
                  deliberate.
                - RUF012 on SSLEnv.metadata — that is the Gymnasium
                  convention, and gym.Env types it as a plain dict itself.
              Adding a rule family later is a decision to take on purpose.

              REVERTED: the fix pass had also reformatted our VENDORED FORKS —
              third_party/rsoccer (20 files) and third_party/rsim
              (src/robosim/__init__.py). Restyling upstream code we track as a
              fork buys nothing, is not covered by `make lint` (which lints
              only `src tests`), and would conflict on every upstream merge.
              Both submodule working trees were restored with `git checkout --
              .`; the committed Python 3.11 build fixes in our rSim fork are
              untouched. The discarded diffs are saved as
              rsim-ruff-churn.patch / rsoccer-ruff-churn.patch in this
              session's scratchpad if anyone wants them back.

## Step 13 — External tools            [PASS]
Verification: docker compose up -d; sleep 8; docker compose ps
              -> all three services Up (game-controller, simulator,
                 vision-client)
              curl -sf http://localhost:8081 > /dev/null  -> "GC UI ok"
              curl -sf http://localhost:8082 > /dev/null  -> "vision-client ok"
              Stack torn down again with `docker compose down` after the gate
              passed, so `restart: unless-stopped` does not resurrect three
              containers on every boot.
Deviations:   VERSIONS AND ASSET NAMES WERE QUERIED FROM THE GITHUB RELEASES
              API AND DOCKER HUB ON 2026-08-10, NOT COPIED FROM SETUP.md.
              What was actually used:

                ssl-game-controller  v3.21.0
                  asset: ssl-game-controller_v3.21.0_linux_amd64  (18264991 B)
                  url:   https://github.com/RoboCup-SSL/ssl-game-controller/
                         releases/download/v3.21.0/
                         ssl-game-controller_v3.21.0_linux_amd64
                  SETUP.md's URL pattern for this one is CORRECT and the
                  v3.21.0 tag still exists.
                  Latest available is v3.23.0 (2026-07-02). NOT taken --
                  protos/ssl-game-controller is pinned at tag v3.21.0 and our
                  referee protobufs were generated from that revision. Bumping
                  the binary alone would drift the wire format from the code.
                  Bump both together or neither.

                ssl-vision-client    v2.1.1  (latest; 2026-06-29)
                  asset: ssl-vision-client_v2.1.1_linux_amd64  (14762989 B)
                  SETUP.md Step 13.3 and its fetch_tools.sh both use
                  .../releases/latest/download/ssl-vision-client_linux_amd64.
                  THAT URL 404s -- verified, it redirects to
                  .../download/v2.1.1/ssl-vision-client_linux_amd64 which does
                  not exist. The real asset name carries the version, exactly
                  like the game controller's. scripts/fetch_tools.sh has been
                  corrected to a pinned VC_VERSION with the versioned asset
                  name; had it been transcribed as written it would have
                  failed for every recruit who ran it.

              docker-compose.yml image tags, all checked on hub.docker.com:
                robocupssl/ssl-game-controller:3.21.0    kept (exists; see above)
                robocupssl/ssl-vision-client:latest   -> :2.1.1
                  Pinned so the container and the tools/bin/ binary are the
                  same build.
                roboticserlangen/simulatorcli:latest  -> :commit-6a4e1c06533b
                  Same image, byte for byte: both tags resolve to digest
                  sha256:19d0df91697c82ebfd1f86eca5ccf6b8be2f0d64b22078725257c
                  3a5856b5ddc. Pinned so a future push to `latest` cannot
                  silently change our physics under a trained policy.

              VISION PORT 10020, not 10006, everywhere in the dev stack --
              on human instruction, and confirmed empirically (see Notes).
              Changed in the compose file's vision-client command (SETUP.md
              already had this right) and additionally in the commented-out
              simulation-controller block, whose -visionAddress was
              224.5.23.2:10006 in SETUP.md. That block's vision source is the
              ER-Force simulator two services above it, so 10006 was simply
              wrong and would have failed silently the day someone uncommented
              it.

              grSim NOT installed, per SETUP.md 13.2. No action needed --
              nothing in the build referenced it.

              Registered our team name in tools/gc-config/engine.yaml
              (SETUP.md 13.1(c)(a), the local-override half). Inserted
              "TritonBots" into the "teams" array, which the GC wrote there
              itself on first start.
Notes:        THE 10020 CLAIM IS NOW VERIFIED, NOT ASSUMED. With the stack up,
              a multicast listener saw 5/5 packets on 224.5.23.2:10020
              (1324-byte vision frames) and ZERO on 224.5.23.2:10006 in the
              same window. ER-Force publishes on 10020 only.

              OPEN GAP, needs a human decision, NOT fixed here.
              src/tbots/net/vision_publisher.py hardcodes a default of
              port=10006 in its constructor signature, and
              src/tbots/apps/viz_rsim.py constructs VisionPublisher without
              passing a port. Nothing in the tree reads configs/net/*.yaml at
              all yet -- the config loader is fall TASK work. Consequence:
              `docker compose up` + `python -m tbots.apps.viz_rsim` renders
              NOTHING, because the publisher sends on 10006 and the compose
              vision-client listens on 10020. I did not "fix" this by editing
              vision_publisher.py: that file is transcribed from SETUP.md
              Step 10.3 and changing its default would move the port
              constant back out of config and into code, which is the exact
              thing the ER-Force port note warns against. The workaround is
              documented in configs/net/dev.yaml -- run tools/bin/
              ssl-vision-client -visionAddress 224.5.23.2:10006 locally for
              rSim visualisation. The real fix is the config loader.

              ANOTHER STALE IMAGE, worth knowing before September.
              SETUP.md 13.2 drops grSim partly because its "published Docker
              image is 4+ years stale". roboticserlangen/simulatorcli's
              newest tag is from 2022-04-20 -- also 4+ years. The rest of the
              13.2 argument (headless, realism profiles, used by the official
              virtual tournament setup) is unaffected and the image runs fine
              here, so this is not a reason to revisit the decision. But the
              stated contrast is no longer accurate, and if we ever need a
              newer ER-Force build we will be compiling it from source.

              "Tritons RCSC" was already in the GC's default team list. That
              is a DIFFERENT team, not us. Ours is "TritonBots", one word, and
              it is now a separate 55th entry. Do not let anyone "deduplicate"
              these.

              tools/bin/ is gitignored, so the two binaries are NOT committed
              -- rerun scripts/fetch_tools.sh on a fresh clone. Added
              tools/gc-config/state-store.json.stream to .gitignore; the GC
              writes that runtime state into the mounted config directory,
              alongside the two config files we do want tracked.

              Upstream PR to add TritonBots to defaultTeams in the GC repo is
              deliberately NOT done -- SETUP.md 13.1 says do that in
              September, not April.

## Step 14 — Configuration            [PASS]
Verification: python -c "yaml.safe_load" over all five files in configs/
              -> all five parse; spot-checked values:
                 net/dev.yaml         vision {224.5.23.2, 10020}, ttl 0
                 net/competition.yaml vision {224.5.23.2, 10006}, ttl 1
                 env/div_b_6v6.yaml   field_type 1, control_hz 60
                 train/curriculum_example.yaml  max_steps parse as ints
                   (2_000_000 -> 2000000; PyYAML honours YAML 1.1 underscores,
                   so the underscored literals are not silently strings)
              team_name is exactly "TritonBots" in both net configs.
Deviations:   configs/net/dev.yaml has vision.port: 10020, where SETUP.md
              Step 14.1 writes 10006. On human instruction, and correct for
              this stack: the dev vision source IS the ER-Force simulator in
              docker-compose.yml. SETUP.md's own note in 13.2 says to set
              10020 when running against ER-Force -- 14.1's literal 10006 and
              13.2's instruction contradict each other, and 13.2 is the one
              that matches reality. The 10006 rationale is preserved in the
              file's comments rather than deleted, since our own
              VisionPublisher does use the classic port. See the OPEN GAP in
              the Step 13 entry above -- this config disagreeing with
              vision_publisher.py's hardcoded default is a real, currently
              unresolved inconsistency, not something these YAML edits fixed.

              configs/net/competition.yaml keeps 10006 unchanged. At a real
              event the source is real SSL-Vision on the classic port, and
              there is no ER-Force simulator in that path. Only dev.yaml moves.

              All five files otherwise transcribed verbatim from SETUP.md
              14.1-14.5. Added explanatory comments in the two net configs
              about which port applies where; no values changed beyond
              dev.yaml's vision.port.
Notes:        field_type: 1 in env/div_b_6v6.yaml is transcribed as SETUP.md
              14.3 writes it, and it agrees with the empirical finding in
              docs/RSIM_FACTS.md (Division B is field_type 1; 0 is Division
              A). Note that RSIM_FACTS.md line 129 warns that SETUP.md
              elsewhere still carries the wrong placeholder -- Step 9.2's
              FIELD_TYPE_DIV_B = 0. Step 14.3 itself is correct; the stale
              value is in Step 9.2. Already recorded under Step 6/9; repeated
              here because this is the file people will actually read.

              Nothing consumes configs/ yet. These files are declarative
              until the config loader lands, so a wrong value here fails
              nowhere today and everywhere later. They were checked by
              parsing and by eye, not by a running system.

## Step 16 — The acceptance checklist            [PASS]
Verification: All seven items in SETUP.md Step 16, run in order.
              1. `make proto` -> 23 modules generated from 23 protos;
                 `python -c "from tbots._pb.state.ssl_gc_referee_message_pb2
                 import Referee"` -> ok.
              2. `pytest -q tests/test_rsim_backend.py` -> 4 passed.
              3. `docker compose up -d game-controller`; :8081 team dropdown
                 shows `TritonBots`. Verified in an earlier session (human
                 confirmed, not re-clicked this pass).
              4. `python -m tbots.apps.ref_monitor --team "TritonBots"` with
                 GC running; Stop/Force Start/Halt in the GC UI each printed
                 a line within a second, counter incrementing. Verified in
                 an earlier session (human confirmed, not re-clicked this
                 pass).
              5. `docker compose up -d vision-client`; :8082 -> HTTP 200,
                 reachable and listening.
              6. `python -m tbots.apps.viz_rsim --realtime --seconds 60` ->
                 six robots orbiting the centre circle at :8082 (human
                 confirmed). `python -m tbots.apps.viz_rsim --seconds 60 |
                 tail -1` -> 521 steps/s, recorded in README.md.
              7. `docker compose down && docker compose up -d && sleep 5 &&
                 docker compose ps` -> all three services (game-controller,
                 simulator, vision-client) `Up`, no restart loops. Re-checked
                 8s later, still up, no churn.
Deviations:   src/tbots/net/vision_publisher.py's `publish_geometry()` now
              populates `field_lines`, `field_arcs`, `penalty_area_depth`,
              and `penalty_area_width`, none of which are in SETUP.md
              Step 10.3's transcribed code (that version sends only
              length/width/goal/boundary). Reason: ssl-vision-client draws
              nothing but a bare default centre circle without explicit
              line/arc geometry -- field_length/field_width alone size the
              canvas, they don't produce markings. Line/shape names and
              types are the league's (SSL_FieldShapeType), matched by the
              client, not invented here.

              Note for whoever reviews this: Step 13's log entry explicitly
              declined to edit this same file to fix the dev/competition
              port mismatch, on the grounds that it's transcribed verbatim
              from Step 10.3 and editing it would blur the config/code
              boundary. This change is a different kind of edit -- it adds
              protobuf field population (code), not a port default (which
              belongs in config) -- so it doesn't reopen that question, but
              flagging the tension since both touch the same "don't touch
              this file" instinct. Kept on human instruction (2026-08-10);
              not reverted to the literal spec.

              .gitignore: broadened the state-store ignore from the single
              literal `state-store.json.stream` (added in Step 13) to also
              match `*_state-store.json.stream`. The GC's "Reset Match"
              button archives the old store under a UTC-stamped filename
              before starting a new one; only the exact filename was
              ignored before, so archived copies were showing up as
              untracked.
Notes:        Step 16 is the last thing SETUP.md calls "simulator setup" --
              Step 17 (HPC image) is a separate concern (Apptainer packaging
              for the cluster), not part of this. tritonbots-simulator-1
              (ER-Force, the match backend) had been manually stopped
              (exit 143) before this pass; item 7's down/up cycle brought
              it back clean, so no action was needed beyond running the
              checklist as written.

## Post-Step-16 — resolved the vision-port OPEN GAP from Step 13/14   [PASS]
Verification: With the compose stack up, bound a raw multicast listener on
              224.5.23.2:10020 (what the compose vision-client actually
              listens on) and ran `python -m tbots.apps.viz_rsim --realtime
              --seconds 2 --port 10020`. Listener received 655 packets and
              decoded at least one detection frame with a 6-robot team --
              confirmed our own rSim traffic, not just the ER-Force
              container's idle broadcast, is reaching that port.
Deviations:   Root cause, empirically confirmed: `apps/viz_rsim.py`
              constructed `VisionPublisher(geometry=DIV_B)` with no port,
              so it always used VisionPublisher's SETUP.md-transcribed
              default of 10006. The docker-compose vision-client listens on
              10020 (it shares the ER-Force simulator container's port).
              Result: the README's own documented quick start --
              `docker compose up -d` then `python -m tbots.apps.viz_rsim
              --realtime` -- rendered nothing, silently, on this stack.
              This is the same gap Step 13/14 logged and explicitly left
              unfixed pending a human decision.

              Fix, on human instruction (2026-08-10): added a `--port`
              CLI argument to viz_rsim.py, default unchanged at 10006 (so
              VisionPublisher's own default and the macOS native-binary
              workflow in Step 16 -- native ssl-vision-client on 10006 --
              are both untouched). README's quick start now passes
              `--port 10020` explicitly, since `docker compose up -d` only
              ever appears in the Linux/WSL2 path (docker-compose.yml's own
              header says it does not work on macOS at all). Did not touch
              VisionPublisher's own default or its constructor signature --
              that stays exactly as SETUP.md Step 10.3 transcribes it; only
              the app-layer call site changed.
Notes:        The real fix is still the config loader reading
              configs/net/dev.yaml's `vision.port: 10020` -- that's fall
              TASK work, not touched here. This is a stopgap that makes the
              documented command line actually work today.

## Post-Step-16 — full readiness audit + ONBOARDING.md correction pass   [PASS]
Verification: `make lint` (ruff + mypy on core) and `make test` clean before
              and after. Every TASK-0xx stub file on the board (31 IDs)
              confirmed present with `NotImplementedError`/`pass` bodies,
              none accidentally implemented, none missing. `go_to_point.py`
              confirmed as the one deliberately-real skill (SETUP.md 11.2's
              reference implementation), not a gap. All five configs/*.yaml
              parse. `viz_rsim.py --seconds 3` and full `docker compose
              down && up -d && sleep 5 && ps` re-run clean, no restart
              loops.
Deviations:   docs/ONBOARDING.md had six confirmed-wrong or confirmed-broken
              claims, found by actually running what it says to run rather
              than reading it:
              1. Header said "Ubuntu 22.04 / WSL2" -- SETUP.md's own Step 1
                 was already corrected to 24.04; ONBOARDING never was.
              2. `docker-compose-plugin` -- not in Ubuntu's repos on any
                 release (SETUP.md Step 1.2 already says this); should be
                 `docker-compose-v2`. Fixed to match.
              3. The verify block was missing `pkg-config --variable=libdir
                 ode` entirely -- the check CLAUDE.md calls "not optional on
                 Ubuntu 24.04" and SETUP.md's own Step 1.4 calls "the single
                 most valuable check in the whole setup". Without it,
                 ONBOARDING's own two remaining checks (version string,
                 dDOUBLE grep) both pass against noble's packaged libode-dev
                 -- same version number as ours, wrong build -- so a recruit
                 following ONBOARDING literally could build rSim against the
                 wrong ODE and get wrong physics with a fully green verify
                 block. Added the check and rewrote the surrounding
                 paragraph, which had called this "the wrong version" -- it
                 isn't; the version string is identical, that's the whole
                 trap.
              4. Repo clone URL was `github.com/tritonbots/tritonbots`, the
                 placeholder CLAUDE.md section 2 explicitly warns 404s.
                 Fixed to `github.com/YashTandon05/tritonbots`.
              5. The protobuf import verify line was
                 `tbots._pb.ssl_gc_referee_message_pb2` -- confirmed by
                 running it, this raises `ModuleNotFoundError`. The real
                 path, `tbots._pb.state.ssl_gc_referee_message_pb2`, was
                 already fixed in SETUP.md itself by an earlier commit
                 (30d34a5) that never touched ONBOARDING's copy.
              6. 1.4's Linux/WSL2 `viz_rsim` command had no `--port`, so it
                 hits the same 10006-vs-10020 mismatch just fixed above.
                 Updated 1.4, the cheat sheet, and configs/net/dev.yaml's
                 comment to reference `--port 10020`.

              Beyond fixed bugs, two structural problems, addressed with a
              status note rather than code (implementing either is out of
              CLAUDE.md scope -- TASK-056 is explicitly recruit work, and
              apps/eval.py isn't even on the task board, so building it
              would be inventing scope, not doing recruit work early):
              - `rl/train.py` is TASK-056, a real `NotImplementedError`
                stub. ONBOARDING Part 1.6/1.7 presented it as something a
                new recruit runs successfully on day one.
              - `apps/eval.py`, referenced 4 times across 1.6/1.7/cheat
                sheet, does not exist anywhere in the tree AND is not on
                SETUP.md's task board under any ID -- it was never planned
                as a tracked deliverable, just assumed into existence by
                ONBOARDING's prose. Flagged prominently (status banner at
                the top, inline note at 1.6, cheat-sheet annotations)
                instead of silently fixed, since creating it is a scope
                decision for a human, not a doc-fidelity fix.

              Minor: `uv pip install -e third_party/rsoccer --no-deps` in
              1.2 never followed up with the second plain install SETUP.md
              Step 5 uses to pick up rsoccer's remaining deps (pygame etc).
              Since our fork already has the rc-robosim pin permanently
              removed (Step 5 log, committed b9a0a63), a plain
              `uv pip install -e third_party/rsoccer` with no `--no-deps`
              is now sufficient and simpler for a fresh clone -- confirmed
              pygame installs correctly in the current venv either way.
              Simplified to that single line.

              `ScriptedDefense()` in Part 3's self-play section renamed to
              `ScriptedTactic()` -- the real class name in
              tactics/scripted.py (TASK-041).
Notes:        Nothing above Step 16 (the actually-built simulator/backend/
              network layers) needed a code change in this pass -- only
              viz_rsim.py's call site, already covered above. Everything
              here is a documentation-accuracy pass on ONBOARDING.md, which
              CLAUDE.md designates reference-only / describes-the-end-state.
              The status banner keeps that framing intact rather than
              rewriting the doc to only describe today: 1.6 onward stays
              written for the finished system, now clearly labeled as such.

## Step 1 (addendum) — how ODE was actually installed, and doc corrections   [PASS]
Verification: All five documented checks re-run on this machine:
              `pkg-config --modversion ode`                -> 0.16.2
              `pkg-config --variable=libdir ode`           -> /usr/local/lib
              `ldd /usr/local/lib/libode.so.8 | grep libccd`
                                                           -> libccd.so.2 => /lib/x86_64-linux-gnu/libccd.so.2
              `dGetConfiguration()` via ctypes             -> "ODE ODE_EXT_no_debug
                 ODE_EXT_trimesh ODE_EXT_opcode ODE_OPC_new_collider
                 ODE_EXT_threading ODE_THR_builtin_impl ODE_double_precision"
              `ldd $(python -c 'import robosim._robosim as m; print(m.__file__)')`
                                                           -> libode.so.8 => /usr/local/lib/libode.so.8

              The installed ODE is correct. Double precision is confirmed from
              the compiled library itself, not just from precision.h. The
              packaged libode-dev/libode8t64 are absent (no /usr/include/ode,
              no /usr/lib/x86_64-linux-gnu/libode*), so nothing shadows it.

Deviations:   The human's actual install path differed from SETUP.md 1.4 and
              was recovered from fish history. The tarball step was attempted
              first and abandoned; the build that succeeded was:

                sudo apt-get remove libode-dev libode8t64
                sudo apt install -y build-essential autoconf automake libtool libccd-dev
                git clone https://bitbucket.org/odedevs/ode.git && cd ode
                git checkout 0.16.2        # NOT "ode-0.16.2" -- that tag does not exist
                autoreconf -fi             # git tree ships no ./configure
                ./configure --enable-double-precision --with-box-cylinder=libccd \
                            --enable-libccd --enable-shared --disable-demos --disable-asserts
                make -j"$(nproc)" && sudo make install && sudo ldconfig

              Three findings from reproducing this, each now written into
              docs/SETUP.md Step 1 and docs/ONBOARDING.md:

              1. `libccd-dev` was missing from the apt list in BOTH docs, and
                 it is load-bearing. ODE's `--with-libccd` defaults to
                 `system`, but when no system libccd is present configure
                 falls back to the bundled copy and STILL EXITS 0. Verified
                 by re-running configure with PKG_CONFIG_LIBDIR pointed at an
                 empty directory: "libccd source: internal", exit 0. Also
                 verified that passing `--with-libccd=system` explicitly does
                 NOT turn this into an error -- it falls back identically.
                 There is no flag that makes it fail, so the only defenses are
                 installing libccd-dev and checking afterwards. Our build is
                 system libccd (undefined ccd symbols in libode.so, runtime
                 link to Ubuntu libccd2 2.1, which is CCD_DOUBLE -- matches
                 ODE's double precision, no mismatch).

              2. The bitbucket tarball URL in both docs is NOT dead. Re-tested
                 today: HTTP 200, 2,627,992 bytes, and the exact ./configure
                 line from ONBOARDING.md runs clean (exit 0) against the
                 extracted tree with the same feature summary as the installed
                 library. Why the human's first attempt failed could not be
                 determined from history and is not recorded as a doc defect.
                 The git route is documented as a fallback only, with its two
                 real gotchas (tag is `0.16.2`; `autoreconf -fi` required).

              3. `ldd "$(python -c 'import robosim; print(robosim.__file__)')"`
                 in SETUP.md Step 1.4 and the troubleshooting appendix cannot
                 work -- `robosim.__file__` is the package `__init__.py` and
                 ldd on a .py file reports nothing useful. Corrected to
                 `robosim._robosim`, the compiled extension that actually
                 carries the ODE dependency.

              Also corrected, same root cause: SETUP.md Step 17.1's Apptainer
              recipe and Step 16's GitHub Actions job both built ODE without
              libccd-dev, so CI and the cluster container would have silently
              used internal libccd while developer machines used system --
              a physics divergence that no test would have caught. Added to
              both, and bumped the Actions ODE cache key to
              `ode-0.16.2-double-sysccd` so the stale cached library is not
              restored over the corrected build.

Notes:        No system state was changed by this pass -- verification and
              documentation only. The two throwaway configure runs used to
              prove the silent-fallback behaviour were done in the scratchpad
              against a freshly downloaded tarball, never installed.

## Post-Step-16 — task board rebuilt as docs/TASKS.md, CI ODE defect fixed   [PASS]
Verification: `make lint` -> ruff "All checks passed!" + mypy "Success: no
              issues found in 6 source files"; `pytest -q` -> 10 passed,
              1 skipped (both unchanged before and after).
              `.github/workflows/ci.yml` re-parsed as YAML (13 steps).
              `make proto` -> 23 modules from 23 protos.
              `python -m tbots.apps.viz_rsim --seconds 3` -> 516 steps/s.
              The new libccd assertion was checked against this machine's
              known-good ODE before being added to CI:
              `ldd /usr/local/lib/libode.so.8 | grep libccd` ->
              `libccd.so.2 => /lib/x86_64-linux-gnu/libccd.so.2`.
Deviations:   *** ONE REAL DEFECT FOUND AND FIXED ***
              `.github/workflows/ci.yml` did NOT install `libccd-dev`, and
              its ODE cache key was still `ode-0.16.2-double-${{ runner.os }}`.
              The "Step 1 (addendum)" entry above records this as corrected --
              but it was corrected only in SETUP.md's *printed copy* of the
              workflow (Step 15.4) and in Step 17.1's Apptainer recipe. The
              actual workflow file in `.github/` was never touched. So every
              green CI run since 2026-08-11 built ODE against ODE's BUNDLED
              libccd while every developer machine uses the system one --
              exactly the silent physics divergence that addendum was written
              to prevent, and one no test in the suite can detect.
              Fixed three ways: `libccd-dev` added to the apt list; cache key
              bumped to `ode-0.16.2-double-sysccd-${{ runner.os }}` so the
              stale artifacts cannot be restored over the corrected build; and
              an `ldd /usr/local/lib/libode.so.8 | grep -q libccd` gate added
              to the Install ODE step, since configure exits 0 either way and
              there is no flag that turns the fallback into an error.
              EXPECT THE NEXT CI RUN TO BE SLOW -- the key bump forces a full
              ODE rebuild from source. That is intended, once.
              Kept the repo's two existing improvements over SETUP.md's
              printed version: staging under `$HOME/ode-install` (actions/cache
              runs unprivileged and cannot write root-owned /usr/local), and
              the two-mirror download with a sha256 check.
              Known remaining divergence, deliberately not chased: the human's
              dev-box ODE was configured with `--disable-asserts`, CI's is not.
              Asserts abort on invalid state rather than changing physics, so
              this is a note for whoever debugs an ODE abort in CI, not a bug.

              DOC RESTRUCTURE (the actual request): SETUP.md's task board has
              been replaced by `docs/TASKS.md` and re-cut against what the
              build actually produced.
              - TASK-001 and TASK-002 closed with evidence. TASK-002 was
                labelled "the highest-risk item in the whole project" and
                landed without the Python 3.10 fallback ever being used.
              - TASK-013, TASK-014 and TASK-020 demoted from "before recruits
                arrive" to "first weeks of fall", on human decision. Rationale
                recorded in TASKS.md section 1: a recruit's day one is
                rSim -> reward -> train -> watch and never touches
                NetworkBackend, perception, or a socket. TASK-010/011/012 stay
                in Tier 1 -- small, self-contained, and they prove the
                outbound UDP direction end to end.
              - 12 tasks ADDED. Roughly half were already recorded as findings
                in this log with no owner attached; the rest came from auditing
                the tree against what the docs claim. Each was confirmed by
                running or grepping, not by reading:
                TASK-006 config loader -- `grep -r "yaml|hydra|OmegaConf"
                  src/tbots` returns NOTHING. `hydra-core` is a declared
                  dependency with zero consumers, every port is a constructor
                  default, and configs/ is not a Hydra tree (no root
                  config.yaml, no `defaults:` list anywhere) so
                  `env=div_b_6v6 reward=example` cannot compose. This is the
                  largest structural hole and the real fix for the
                  10006-vs-10020 workaround logged above.
                TASK-058 apps/eval.py -- referenced 4x in ONBOARDING, absent
                  from the tree, and absent from the old board. ONBOARDING's
                  own status banner flagged it as "a real gap"; it is now
                  tracked.
                TASK-059 checkpoint/resume -- implied by SETUP.md 17.2,
                  ONBOARDING's Atlantis job script, and ARCHITECTURE section 8
                  ("not optional"), owned by nobody. Atlantis walltime is 12h.
                TASK-007 skill registry -- `src/tbots/skills/__init__.py` is
                  empty, so nothing imports go_to_point and `skill_names()`
                  returns `[]` (verified by running it). ONBOARDING's cheat
                  sheet tells recruits to run exactly that, and
                  `build_skill("go_to_point")` raises KeyError, which breaks
                  the config-driven policy swap the Skill interface exists for.
                TASK-003/004/005/008 -- the build is verified only by the
                  machine and the process that built it. 004 notes that
                  `test_state_length_matches_constants` contains no assertion
                  at all, and that a wrong ACTION_LEN cannot raise because
                  setActions() indexes an unchecked std::vector. 005 notes
                  that no part of Step 1M (macOS) has ever been executed by
                  anyone or by CI, though ARCHITECTURE calls macOS
                  first-class.
                TASK-060..064 -- Atlantis. `env-atlantis.sh` is IN .gitignore
                  and exists only inside the shared checkout, while ONBOARDING
                  tells recruits it is "already in this checkout";
                  `scripts/train_atlantis.slurm` is `sbatch`-ed by ONBOARDING
                  but exists only as a fenced code block; `containers/` is
                  empty and SETUP.md Step 17 has no log entry, so the
                  Apptainer image was never built -- and Atlantis has no
                  container runtime, so that scope needs an explicit decision
                  rather than silent abandonment. Also: torch/CUDA wheel
                  unpinned against a `>=2.2` floor, W&B entity nonexistent,
                  and no SLURM smoke path that does not depend on TASK-056.
                TASK-018 autoref -- commented out in docker-compose.yml, and
                  without it the GC cannot enforce ball placement or robot
                  counts, so no realistic full-match evaluation is possible.
              - Named one recurring pattern: three config files already
                specify features that do not exist
                (`domain_randomization.enabled: true`, `encoder: set_encoder`,
                `backend.kind: rsim`). They are harmless only because nothing
                reads configs at all; TASK-006 makes them load-bearing on the
                day it lands. Schedule TASK-006 WITH the tasks whose keys it
                activates, not before them.
Notes:        Files touched: NEW `docs/TASKS.md`; `docs/SETUP.md` (board
              replaced by a pointer, and "You are done when" now references
              TASKS.md Tier 1); `docs/ARCHITECTURE.md` section 12 rewritten and
              a correction added noting that section 8's Apptainer-based HPC
              story does not match Atlantis, which has no container runtime;
              `docs/ONBOARDING.md` (status banner re-dated and pointed at
              TASKS.md, Part 2's "pick one from the board" likewise);
              `README.md` ("pick up a task" row); `.github/workflows/ci.yml`.
              Task IDs are unchanged throughout, so every
              `NotImplementedError("TASK-0xx")` marker in the source still
              resolves.
              No source file under `src/` was modified. Everything found in
              the audit became a task rather than a quiet fix, except the CI
              defect above, which the human explicitly asked to have fixed.
              Recruitment target used for the tiering: mid-October 2026, per
              the human, i.e. roughly 7 weeks from 2026-08-25.

## Post-Step-16 — Rule 3 extracted into core/perspective.py   [PASS]
Verification: `pytest -q` -> 29 passed, 1 skipped (was 10 passed, 1 skipped;
              +19 new tests, none of which need a socket or a running game
              controller).
              `make lint` -> ruff "All checks passed!" + mypy "Success: no
              issues found in 7 source files" (was 6 -- perspective.py is
              inside the mypy-checked core).
              NO-REGRESSION CHECK, reproducing the exact referee decode
              recorded in the Step 15 follow-up entry above:
                DIRECT_FREE_BLUE, blue="TritonBots", blue_team_on_positive_half,
                designated_position (1000, -500) mm, action time 5e6 us
                -> FREE_KICK | ours=True | target=(-1.0, 0.5) | t=5.0
              Byte-identical to the value logged before this change.
              `python -m tbots.apps.viz_rsim --seconds 3` -> 532 steps/s,
              in line with the 516-521 measured before (VisionPublisher's
              publish() gained a transform call, which is a no-op under the
              default IDENTITY perspective).
Deviations:   *** THIS IMPLEMENTS ARCHITECTURE CANDIDATE A AND CLOSES
              TASK-013, ON EXPLICIT HUMAN INSTRUCTION (2026-08-26). ***
              CLAUDE.md section 4 forbids implementing TASK items. The human
              asked for this one directly after an architecture review, so it
              is a sanctioned override, recorded here rather than assumed.
              Scope was held to the extraction: TASK-014 (observe()),
              TASK-010, TASK-011, TASK-015 and TASK-020 are all still
              NotImplementedError stubs. Only their signatures and docstrings
              changed, so the seam is in place for whoever writes them.

              THE PROBLEM. Rule 3 ("we are always `us`, we always attack +x")
              was knowledge, not a module. RefereeReceiver._resolve_sides()
              already derived (we_are_yellow, flip_x) from the referee
              message, and NetworkBackend._flip_state() -- TASK-013 -- was
              specified to derive the same two facts from the same message a
              second time. VisionReceiver, RobotControlSender, SimControlSender
              and Tracker each needed them too, and four of those five are
              still stubs, so the duplication had not been written yet. That
              is precisely why it was cheap to prevent now.

              WHAT WAS ADDED.
              NEW `src/tbots/core/perspective.py`. A frozen `Perspective`
              dataclass carrying the two answers (`we_are_yellow`, `flip`)
              plus the transform that follows: `point`, `velocity`, `angle`,
              `robot_state`, `ball_state`, `world_state`. Two module
              constants: `UNRESOLVED` (before the referee has named us) and
              `IDENTITY` (blue, attacking +x -- rSim's perspective, and the
              second adapter that makes this a real seam rather than a
              hypothetical one). It is in `core/` and imports only
              `core.state` and `core.units`, so Rule 1 holds.

              Three facts are now written down in one place instead of being
              folklore:
                - The transform is a 180-DEGREE ROTATION, NOT A MIRROR. It
                  negates x AND y and adds pi to headings. This matches what
                  the original to_gamestate() did to placement_target, so it
                  is a faithful extraction, not a behaviour change -- but
                  nothing said so anywhere. Negating x alone would flip
                  handedness and turn `vy = left` into `vy = right`.
                - It is ITS OWN INVERSE, so one set of methods serves both
                  directions: field->ours on the way in (vision, referee),
                  ours->field on the way out (vision publisher, teleport).
                - It must NEVER be applied to RobotCommand velocities. Those
                  are robot-LOCAL, and the heading they are relative to was
                  already normalised on the way in; flipping again drives
                  every robot backwards for one half of every match. Noted in
                  the module docstring and in robot_control.py's stub.

              WHAT WAS CHANGED.
              - `net/referee.py`: `_resolve_sides()` (a private method on a
                class that opens a multicast socket in its constructor)
                became `resolve_perspective(msg, team_name, current)`, a pure
                module-level function. Same logic, including the stickiness
                of both halves. `to_gamestate(msg, we_are_yellow, flip_x)` ->
                `to_gamestate(msg, perspective)`. RefereeReceiver now holds a
                Perspective and exposes it; its `we_are_yellow` and `flip_x`
                properties are kept as delegating properties so
                apps/ref_monitor.py and the troubleshooting sections of
                SETUP.md and ONBOARDING.md ("print we_are_yellow and flip_x
                from ref_monitor") still work unchanged.
              - `backends/network.py`: `_flip_state()` deleted; a
                `perspective` property reads it from the referee receiver.
                Half time needs no special handling anywhere -- the referee
                announces the swap and the next message updates it.
              - `net/vision_publisher.py`: constructor takes
                `perspective: Perspective = IDENTITY` in place of
                `we_are_yellow: bool = False`, and `publish()` applies
                `perspective.world_state(world)` to go our-frame ->
                field-frame. LATENT BUG FIXED: the publisher previously wrote
                our-frame coordinates straight onto the wire, so a match
                render would have been rotated 180 degrees for whichever half
                we defended +x. Nothing had noticed because the only caller
                today is rSim, whose perspective is IDENTITY, under which the
                new call is a no-op.
              - `perception/tracker.py`, `net/robot_control.py`,
                `net/sim_control.py`: still NotImplementedError, but their
                signatures now carry the Perspective and their docstrings say
                where to apply it (tracker: exactly once, at the END, after
                fusing in the camera's field frame -- rotating before fusing
                gives subtly wrong velocities).
              - `net/robot_control.py` also gained `BLUE_PORT`/`YELLOW_PORT`
                constants; the perspective chooses the port and nothing else.

              NEW `tests/test_perspective.py`, 19 tests, no mocks and no
              sockets. Builds real `Referee` protobuf messages in-process.
              Covers the transform (rotation not mirror, involution, vtheta
              and z untouched), all four colour x side combinations, case
              sensitivity of the team name, stickiness, the half-time swap,
              placement-target normalisation, `ours` following our colour
              rather than the command colour, and one end-to-end assertion
              that the goal we attack lands at x=+4.5 in all four cases.
              Before this, not one line of colour or side resolution was
              testable without joining a multicast group.

Notes:        Docs updated: `docs/ARCHITECTURE.md` section 5 (Rule 3 now names
              the module and records the rotation/involution/local-frame
              facts); `docs/TASKS.md` (TASK-013 marked DONE and absorbed;
              TASK-014 and TASK-020 notes narrowed accordingly; new row in
              section 2); `README.md` ("understand colour / side flipping").
              `docs/SETUP.md` Steps 9.3, 10.2 and 10.3 each got a SUPERSEDED
              note above the transcribed code block -- the blocks themselves
              were left alone, since SETUP.md is the record of the original
              build, but a rebuild that follows them literally would
              re-create the split this change removed.
              TASK-013 was in Tier 2 as of yesterday's re-tiering, so this is
              work pulled forward, not a Tier 1 item cleared.

## Post-Step-16 — Perspective verified against the real game controller   [PASS]
Verification: Three questions from the human, answered with evidence rather
              than inference. The stack (docker compose) was already up; it
              was left up, and the GC's side setting was restored to the value
              it held before this pass.

              (1) SEMANTICS, from the pinned proto rather than from our own
              comments. protos/ssl-game-controller/proto/state/
              ssl_gc_referee_message.proto:183-186 --
                "// True, if the blue team will have it's goal on the positive
                 x-axis of the ssl-vision coordinate system."
              So the flag is about which goal a team DEFENDS, which is what
              our `flip = (blue_team_on_positive_half != we_are_yellow)`
              assumes. Confirmed correct. Note it is `optional`.

              (2) THE FIELD IS ALWAYS THERE, on real packets. Sniffed the
              live GC (v3.21.0, the version we are pinned to) on
              224.5.23.1:10003: 404 packets, `blue_team_on_positive_half`
              present in 404 of 404 (100%). GC source shows why -- it is
              written unconditionally on every generated message
              (internal/app/publish/messagegenerator.go:114,157), not only on
              change. Real decode of a real packet:
                blue='Unknown' yellow='TritonBots' blue_on_positive=False
                -> we_are_yellow=True, flip=True.  Correct.

              (3) LIVE SIDE CHANGE TRACKED IN BOTH DIRECTIONS. Drove the GC's
              own websocket API (/api/control, an UpdateTeamState change with
              onPositiveHalf) with a hand-rolled RFC 6455 client, since no
              websocket library is in the venv:
                baseline              wire blue_on_positive=True  -> flip=False
                operator swaps ends   wire blue_on_positive=False -> flip=True
                swapped back          wire blue_on_positive=True  -> flip=False
              Colour unchanged throughout, as it must be.

              (4) TRAINING IS ONE-SIDED BY CONSTRUCTION. `backends/rsim.py`
              contains neither "perspective" nor "yellow" anywhere -- rSim has
              no colours and no half time, so its WorldState is already in our
              frame. Empirically: a robot commanded vx=+1.0 for one second
              moves x +0.00 -> +0.87, i.e. "forward" is always +x. And the
              same physical moment written in field coordinates for BOTH ends
              normalises to byte-identical positions, headings and velocities
              once the Perspective is applied.

Deviations:   None to the code. No source file changed in this pass.

Notes:        *** THE RULEBOOK DOES NOT SAY WHAT WE ASSUMED, AND IT DOES NOT
              MATTER -- BUT SOMETHING ELSE IN IT DOES. ***
              The published SSL rules (robocup-ssl.github.io/ssl-rules,
              chapters/gamestructure.adoc) state that "the winning team
              chooses the goal it will attack in the first half of the match"
              and list a 300-second half-time break, but contain NO explicit
              statement that teams change ends at half time. Our code comments
              asserted the swap as fact; that assertion was inherited from
              SETUP.md, not checked. It is now checked and it is not in the
              rules text.

              This turns out to be harmless, for a better reason than being
              right: the game controller does not swap sides automatically
              either. `proceedStage()` in internal/app/statemachine/
              change_stage.go does not touch OnPositiveHalf at any stage
              transition -- an operator sets it. So the side is whatever a
              human tells the GC it is, whenever they tell it, and the GC
              broadcasts the current truth in every packet. We are event-
              driven off that field rather than off a rule, so we are correct
              regardless of whether, or when, the swap happens.

              THE REAL RULEBOOK FINDING, which our code already handles but
              nobody had written down: COLOURS can also change mid-match.
              "if both teams prefer the same color, the referee assigns the
              colors by chance. In this case, the teams switch the colors
              after the first half of the match as well as after the first
              half of the overtime if applicable." We re-resolve colour from
              the team NAME on every packet, so this is handled -- but any
              future code that caches `we_are_yellow` once at startup would
              be wrong for half of such a match. Worth keeping in mind for
              TASK-011 (which picks port 10301 vs 10302 by colour) and
              TASK-016.

              ONE FAMILIARITY GAP THE FLIP DOES NOT CLOSE, for whoever writes
              the observation builder (TASK-050/053): rSim assigns our robots
              ids 0..n_us-1, always. At a real match our ids are whatever we
              put on the field, anywhere in 0..15, and they arrive that way in
              SSL-Vision's `robot_id`. An observation vector that indexes by
              robot id will therefore see a layout in a match that it never
              saw in training. The Perspective normalises the FRAME, not the
              INDEXING. Same class of gap as domain randomisation (TASK-054),
              which normalises neither: training state is perfect and
              instantaneous, match state is tracked, noisy and 20-40 ms stale.

## TASK-071 — Division B geometry in the compose stack   [PASS]
Verification: `docker compose up -d` with `GEOMETRY: "2020B"`, then two
              probes against the running stack.

              (1) ON THE WIRE. Joined 224.5.23.2:10020 and decoded the first
              `SSL_WrapperPacket` carrying `geometry`:

                field_length         = 9000 mm
                field_width          = 6000 mm
                goal_width           = 1000 mm
                goal_depth           =  180 mm
                boundary_width       =  300 mm
                n field_lines        = 12
                n field_arcs         = 1   (CenterCircle, r = 500 mm)
                n cameras (calib)    = 1
                TopTouchLine                : (-4500, 3000) -> (4500, 3000)
                LeftGoalLine                : (-4500,-3000) -> (-4500, 3000)
                LeftPenaltyStretch          : (-3500,-1000) -> (-3500, 1000)
                LeftFieldLeftPenaltyStretch : (-4500,-1000) -> (-3500,-1000)

              That is 9.0 x 6.0 m with 1.0 m goals and a 1.0 x 2.0 m defense
              area — `core/geometry.py`'s `DIV_B`, exactly, every field.
              Before the change the same probe would have read 12040 x 9020.

              (2) IN THE CLIENT. `ssl-vision-client` draws whatever geometry
              it last received, so it was read from the client itself over
              its `/api/vision/geometry` websocket:

                push 0:  44 B  field_length=12000 field_width=9000  lines=0
                push 1: 818 B  field_length=9000  field_width=6000  lines=12

              Push 0 is the client's BUILT-IN DEFAULT, sent on connect when it
              has no vision yet. Push 1 is the simulator's, and it is Division
              B. The client is drawing a 9 x 6 m field.

Deviations:   None. `2020B` is a stock preset that ships in the pinned image
              (`config/simulator/2020B.txt`, alongside 2014/2017/2018/2019/
              2020), so no config had to be mounted. Changed in
              `docker-compose.yml` and in all three places SETUP.md Step 13
              names a geometry (the `docker run` example, the `-g` option
              table, and the compose block), plus the from-source
              `simulator-cli -g` invocation.

Notes:        - The two presets differ in more than size: `2020` mounts TWO
                cameras (derived_camera_world_tx = -3010 and +3010), `2020B`
                mounts ONE at the centre. So Division B vision arrives as a
                single camera frame, not two half-field frames that need
                merging. Worth knowing for TASK-014's frame merge — on this
                stack there is nothing to merge, which means the merge path
                will NOT be exercised by local testing. At a real match with
                four cameras it will be.

              - THE SIMULATOR LEAVES THE SCALAR PENALTY-AREA FIELDS UNSET.
                In the packet above, `penalty_area_depth`, `penalty_area_width`
                and `center_circle_radius` all read 0; the defense area is
                described only by the `field_lines` entries
                (`LeftPenaltyStretch`, `LeftFieldLeftPenaltyStretch`, ...) and
                the circle only by `field_arcs`. proto3 cannot distinguish
                "absent" from 0 for those scalars. TASK-014 is specified to
                "read geometry from the wire and fail on a Division mismatch":
                it must key that check on `field_length`/`field_width`/
                `goal_width`, which ARE populated, and must not read
                `penalty_area_depth` — against this simulator that reads 0 and
                would fail every match. Found here, not by TASK-014, so it is
                written down before someone rediscovers it at a competition.

              - Geometry is republished continuously at ~66 Hz (800 packets in
                12 s), interleaved 1:1 with detection frames. The client's
                websocket, by contrast, pushes only on CHANGE — connecting to
                `/api/vision/geometry` against an already-running stack yields
                the default frame and then silence, which looks like a broken
                endpoint and is not. To observe a real push: stop the
                simulator, restart the client, connect, then start the
                simulator.

## TASK-070 — rSim runs at 60 Hz, not 58.8   [PASS]
Verification: `pytest tests/test_rsim_backend.py` (7 passed), of which the
              gate is `test_sixty_steps_is_one_second_of_travel`:

                before   1.0199999879350568 m  -> FAILS (1.0 +/- 0.01)
                after    1.00000            m  -> PASSES

              Measured from STEADY STATE, not from rest. A robot commanded
              1.0 m/s needs ~25 ticks to spin up through the motor model, and
              covers only 0.87 m over its first 60 ticks. That ramp is
              physics; timing it would have measured the wrong thing. Per-tick
              displacement at steady state is the clean signal:

                before   0.017000 m/tick   (60 ticks = 1.02000 m)
                after    0.016667 m/tick   (60 ticks = 1.00000 m)

              Exactly 2 cm of error per simulated second, which is the 17 ms
              vs 16.667 ms difference and nothing else.

              Two further tests pin the CAUSE rather than the symptom:
              `test_simulator_timestep_is_the_backend_dt` asserts
              `sim.time_step == backend.dt`, and
              `test_a_non_default_dt_reaches_the_simulator` runs at 1/120 s
              and re-checks after a `reset()`, since `reset()` rebuilds the
              world and used to requantise the step.

              Full suite 37 passed / 1 skipped; `make lint` clean.
              `scripts/verify_rsim.py` re-run end to end: all four TASK-001
              facts reproduce unchanged (notably PART 4's 162.85 deg/s =
              2.842 rad/s, bit-identical to the 2026-08-10 run), so the
              timestep change perturbed no other conclusion.
              `docs/RSIM_FACTS.md` regenerated from that run.

Deviations:   Four, all additive to what TASK-070 asked for.

              1. THE OVERLOAD IS RESOLVED BY TYPE, AND THAT NEEDED PROVING.
              The task says "add a double seconds overload, keep the int
              overload for rSoccer" -- which only works if pybind11 reliably
              routes an int to one and a float to the other. Tested all six
              call shapes (positional and keyword, float lists and int lists):
              all correct. The one that worried me was ints inside `ballPos`
              or a robot pose, because pybind11's overload resolution runs a
              strict pass over the WHOLE signature and then a converting pass:
              a stray int in a list could in principle fail the strict pass
              and let the converting pass hand a float timestep to the
              millisecond overload, where 1/60 truncates to 0 ms.
              MEASURED: it does not. pybind11 2.13 refuses to narrow a float
              into an int argument even when converting, so the millisecond
              overload cannot take a float at all. The hazard is real in
              shape and absent in this version.

              2. `.noconvert()` ON BOTH TIMESTEP ARGUMENTS ANYWAY. Given (1)
              it changes no behaviour we can observe today. It is there so the
              int/float split is a property of OUR binding rather than of
              pybind11's arithmetic-caster policy, which is a detail of a
              vendored dependency and invisible at the call site. I first
              wrote the opposite claim into the source comment -- that
              .noconvert() was load-bearing -- then tested it, found it false,
              and rewrote the comment to say what was actually measured.

              3. ARGUMENTS ARE NOW NAMED. `.noconvert()` requires a `py::arg`,
              and pybind11 requires all-or-none, so the binding went from
              positional-only to seven named arguments. This is a feature
              here: `timeStep_ms` vs `timeStep_s` now shows up in
              `help(robosim.SSL)`, and the whole bug was a units confusion
              that a reader could not see at the call site. `docs/RSIM_FACTS.md`
              said "positional only -- the binding declares no argument
              names"; that line is now wrong and has been corrected.

              4. `SSL.time_step` PROPERTY. The gate needs to assert what the
              simulator actually stepped at, and there was no way to ask. It
              always reads back in seconds, whichever constructor was used.

              Two things deliberately NOT done:
              - VSS keeps its millisecond-only constructor. We do not use VSS,
                so an overload there would be untested code shipped on
                symmetry alone.
              - `RSimBackend.dt` is unchanged and still defaults to 1/60.

Notes:        - `SSL(..., 17.0, ...)` IS SEVENTEEN SECONDS. The overload keys
                on type, not magnitude, and a float is always seconds. There
                is no plausible timestep where the two readings overlap, so
                this cannot be disambiguated by value and should not be: it is
                in the RSIM_FACTS table and in the binding's comment.

              - THE STORED TIMESTEP HAD TO MOVE TO SECONDS TOO, which the task
                did not mention. `SSL::reset()` tears down and rebuilds the
                whole `SSLWorld` (it does not rewind it), so it re-passes the
                stored timestep. Left as the `int m_timeStep_ms` member, the
                constructor would have honoured 1/60 and then the first
                `reset()` would have quantised it back to 17 ms -- meaning
                episode 1 at 60 Hz and every episode after it at 58.8, which
                is a far nastier bug than the one being fixed.
                `test_a_non_default_dt_reaches_the_simulator` re-checks after
                a reset for exactly this reason.

              - WHY THIS WAS INVISIBLE, worth internalising before trusting
                any other rSim number: the state array's velocities are
                finite-differenced and divided by the SAME wrong timestep, so
                they read back correct to full precision. A robot at 1 m/s
                reported `vx = 1.0` throughout. Nothing in the state array,
                and nothing in any velocity-based test, could have caught
                this. It was found by the architecture review reading the C++,
                and the only observable was position integrated over time.

              - CI needs no change: `.github/workflows/ci.yml` already
                checks out submodules recursively and runs
                `uv pip install -e third_party/rsim`, so the bumped pointer
                builds the new binding on every push. TASK-004 will pin the
                60 Hz timestep as a CI-guarded constant.

              - Fork: YashTandon05/rSim `56d0253`, pushed to `main`.
                Submodule pointer bumped 4e4619c -> 56d0253. C++ changes still
                require `uv pip install -e third_party/rsim` (the fork
                deliberately does not set `editable.rebuild`), so anyone
                pulling this must reinstall or they will keep the old .so and
                the old physics.

## TASK-072 — the `Backend` / `SimBackend` split   [PASS]
Verification: `pytest -q` -> 50 passed, 1 skipped (was 34 + 1 before this
              run of work). `make lint` clean.
              `MYPYPATH=src mypy --ignore-missing-imports src/tbots/backends
              src/tbots/core src/tbots/rl/envs/base.py` -> clean, 12 files.

              The three gates from the task board, each its own test in
              `tests/test_rsim_backend.py`:

                opponents move when commanded    test_opponents_move_when_commanded
                and stand still when not         test_opponents_stand_still_when_not_commanded
                place(ball=...) moves only ball  test_place_moves_only_the_ball
                unknown id raises ValueError     test_unknown_robot_id_raises

              Plus, because they are the ways this can silently go wrong:
              `test_our_id_is_not_silently_an_opponent_id` (n_us=1, n_them=1,
              so id 0 is valid on both sides and means two different robots),
              `test_place_moves_only_the_robots_named`,
              `test_place_with_nothing_named_changes_nothing`,
              `test_place_stops_the_robots`, `test_place_rejects_an_unknown_robot`,
              `test_place_before_reset_raises`, `test_reconfigure_is_gone`,
              and two protocol-conformance tests.

              THE SPLIT WAS VERIFIED TO ACTUALLY BITE, not just to compile.
              With `rl/envs/base.py` now taking a `SimBackend`, mypy on a
              probe that hands it a `NetworkBackend`:

                error: Argument 1 to "wants_sim" has incompatible type
                "NetworkBackend"; expected "SimBackend"
                note: "NetworkBackend" is missing following "SimBackend"
                      protocol members: place, set_game_state
                note: Following member(s) have conflicts:
                      Expected: def step(self, commands, opponent_commands=...)
                      Got:      def step(self, commands)

Deviations:   - `rl/envs/base.py` now annotates `backend: SimBackend` instead
                of `Backend`. Not listed in TASK-072, but it is the entire
                stated purpose of the split ("a training environment requires
                a SimBackend, so the type checker stops anyone pointing a
                training run at hardware", ARCHITECTURE §"Backend and
                SimBackend"). One word; the file is rewritten by TASK-076
                anyway.

              - `place()` also raises `ValueError` on an unknown id, and
                `RuntimeError` if called before `reset()`. The board specifies
                raising only for commands, but `place()` is new surface with
                no behaviour to preserve, and a silently-ignored placement is
                the same defect class the review found.

              - Kept `n_us` / `n_them` as properties. They are not in the
                protocol; robot counts being fixed for the run is now a stated
                property of the backend, so reading them is legitimate and
                `reconfigure()` was the only thing that had to go.

Notes:        - `place()` STOPS EVERY ROBOT ON THE FIELD, including ones
                nobody placed, and this cannot be fixed at our layer. rSim has
                no partial teleport, so `place()` is `reset()` with the current
                poses filled in — and rSim's reset takes robot poses only,
                `[x, y, dir]`, with no velocity field. The ball keeps its
                velocity, because `ballPos` carries `(vx, vy)`. Measured: a
                robot moving at 1 m/s is stationary after `place()` and does
                not coast. `test_place_stops_the_robots` pins it so it is a
                documented property rather than a surprise during TASK-052's
                restarts. If it ever needs to change, it needs a new entry
                point in the fork, not a change here.

              - AND THE VELOCITIES IT RETURNS ARE ALL ZERO, the ball's
                included. A reset clears the baseline `get_state()` differences
                against (RSIM_FACTS trap 4), so the frame `place()` returns
                reads zero for everything. The ball really is moving if you
                gave it a velocity; you cannot see it until the next `step()`.
                Anything computing a reward or a done-condition from the frame
                `place()` returns will read a stopped world.

              - `place()` does NOT rewind `self._t`. It is a mid-episode
                teleport; only `reset()` starts an episode.

              - THE BACKEND NOW CACHES ITS LAST `WorldState` (`self._last`),
                because `place()` needs the current poses and asking rSim for
                them would mean a second `get_state()` — which zeroes every
                velocity in the following frame (RSIM_FACTS trap 4). The
                once-per-step rule is still honoured exactly.

              - *** A FALSE GREEN WORTH KNOWING ABOUT. *** My first check of
                the split reported "Success: no issues found" and proved
                nothing: mypy could not resolve `tbots` from a file outside the
                repo (the editable install is a `.pth`, which mypy does not
                execute), so `--ignore-missing-imports` turned every tbots name
                into `Any` and silently checked nothing. `reveal_type` showed
                `Any` and caught it. `MYPYPATH=src` fixes it. This affects
                `make lint` too: `mypy src/tbots/core` works only because the
                path is passed directly, and there is no `[tool.mypy]` section
                in `pyproject.toml` to make it robust. Worth adding one.

              - FOUND, NOT FIXED: `RSimBackend._pad()` still TRUNCATES a
                scenario that has more robots than the backend was built for
                (`poses[:n]`), silently. That is the same shape of defect as
                the dropped commands this task fixed — `Scenario.kickoff()`
                against `RSimBackend(n_us=2)` quietly drops four robots and
                trains on a 2v2 that the caller believes is 6v6. Left alone
                deliberately: it is outside TASK-072's scope and changing it
                could break a workflow nobody asked me to change. Suggest
                folding it into TASK-004.

              - SETUP.md Step 9 has been brought back in line: 9.1 and 9.2 now
                carry the live `base.py` and `rsim.py`, and 9.2a (which taught
                curriculum-by-`reconfigure()`) and Step 14.5's `promote_when`
                paragraph carry "Superseded 2026-09-07" notes in the style of
                9.3. Its design points (a) and (b) — fixed observation size,
                constant field geometry — survive the change and are kept.

## TASK-072 (follow-up) — review findings fixed   [PASS]
Verification: `pytest -q` -> 52 passed, 1 skipped. `make lint` clean, and it
              now type-checks the backends as well as `core`.

              *** A REAL BUG, FOUND BY REVIEW AND NOT BY ME. ***
              TWO `place()` CALLS IN A ROW STOPPED THE BALL DEAD.
              `place()` sources anything the caller did not name from
              `self._last`, and then overwrote `self._last` with its own
              post-reset observation — in which every velocity reads zero,
              which is the trap its own docstring documented two lines above.
              The second call read that zero back out and placed a stationary
              ball. Nothing raised. Reproduced before the fix:

                rolling ball                    vx = 1.3165
                place, step                     vx = 1.2135
                place, place, step               vx = 0.0000   <- dead

              Exactly the defect class TASK-072 existed to fix, reintroduced
              one method over. Fixed by writing the placed velocity back over
              the post-reset read: the robots really are stopped, so zero is
              true for them, but for the ball it is a lie we can correct
              exactly, because we are the ones who just set it. After:

                place, place, place, step        vx = 1.2135   (was 0.0000)

              and two places now cost no more than one, because the second
              re-places the value recorded before the first settling loss.
              `test_place_keeps_the_ball_rolling` pins it.

              *** A SECOND MEASUREMENT THE REVIEW PROMPTED. ***
              Every `place()` costs a rolling ball a FIXED ~0.091 m/s, even
              when the ball is not the thing being placed:

                0.5 m/s -> lost 24.2%      2.0 m/s -> lost 7.0%
                1.0 m/s -> lost 15.5%      4.0 m/s -> lost 3.3%

              An absolute penalty, not a percentage — the same 0.091 m/s
              every time. Cause: `SSLWorld`'s constructor ends with
              `for (i = 0; i < 30) physics->step(timeStep * 0.1)`, a settling
              loop worth three whole timesteps, and the ball rolls against
              friction throughout it. Positions survive to within a
              micrometre. Recorded in `docs/RSIM_FACTS.md` because it is a
              property of rSim's `reset()`, not of our wrapper, and pinned by
              `test_place_costs_the_ball_a_little_speed` so that a future rSim
              change shows up here rather than in a reward curve. It is a real
              constraint on TASK-052's restarts: placing the ball for a free
              kick at walking pace loses a quarter of its speed.

              Ball `z`/`vz` are also dropped by any placement — `ballPos` is
              `[x, y, vx, vy]` and has nowhere to put them, so a chipped ball
              lands the moment you place it. Documented, not fixable here.

Deviations:   None beyond the review's own findings.

Notes:        Acted on, from the two review axes:

              - `mypy` NOW RUNS ON THE BACKENDS IN CI. The review's sharpest
                structural point: TASK-072's stated purpose is that "the type
                checker stops anyone pointing a training run at hardware", but
                `make lint` ran `mypy src/tbots/core` only, so CI never checked
                the file where that is enforced. The guarantee was a comment.
                `make lint` now covers `core`, `backends` and
                `rl/envs/base.py`, and `pyproject.toml` grew its first
                `[tool.mypy]` section: `mypy_path = "src"` plus a narrow
                `ignore_missing_imports` override for `robosim` alone (it is a
                compiled extension with no stubs; nothing else is silenced).
                VERIFIED BY BREAKING IT ON PURPOSE — assigning a
                `NetworkBackend` to a `SimBackend` makes `make lint` exit 1
                with "missing following SimBackend protocol members: place,
                set_game_state", and it goes green again on revert.

              - `SETUP.md` Step 0's **Rule 2** still read "Both implement the
                same `Backend` protocol", contradicting its own §9.1 and the
                shipped code. Amended in place with a dated note.
                *** CLAUDE.md §7 rule 2 QUOTES THAT SENTENCE VERBATIM AND IS
                NOW STALE. Left for a human: it is the working agreement that
                governs the build, and a build agent should not quietly edit
                the document that constrains it. ***

              - `_slot()` renamed `_checked_id()`. It validated an id and
                returned it unchanged while the caller did the slot
                arithmetic, so the name promised something it did not do.

              - `Scenario` now uses the `Pose` / `BallPlacement` aliases
                introduced five lines above it, instead of respelling the same
                tuple shapes by hand.

              - `test_reconfigure_is_gone` became
                `test_robot_counts_are_fixed_for_the_run`. Asserting an
                absence would have passed forever if the same power came back
                under a different name; it now asserts the invariant that was
                actually at stake — the counts are readable and cannot be set.

              - The stale ruff note in `pyproject.toml` citing
                `int(round(...))` in `backends/rsim.py` was corrected; TASK-070
                deleted that line.

              Considered and declined, with reasons:

              - A `Team` type to absorb the `(n, side)` pair travelling through
                `_checked_id` and `_poses`. Two private helpers in one file is
                thin justification for a new concept.

              - Making the tests stop reaching `backend._sim.time_step`. That
                gate is white-box on purpose: it asserts we passed `dt` through
                to the simulator, which has no public expression that would not
                be invented solely for the test.

              - "Speculative generality" on `backend: SimBackend`. It is a
                restriction, not a generality, and it is load-bearing today:
                `SSLEnv(NetworkBackend(...))` is a type error now, which is
                precisely what ARCHITECTURE.md asks the split to buy.

              Rejected on the facts:

              - The spec review called the SETUP.md §9.1 changes
                (`typing.Sequence` -> `collections.abc`, `-> "Scenario"` ->
                `-> Scenario`) unrequested restyling against CLAUDE.md §8.
                They are not. `git show 6d01e9a:src/tbots/backends/base.py`
                already imports from `collections.abc`: the LIVE CODE had
                diverged from SETUP.md's listing before this session, almost
                certainly in the Step 15 ruff `UP` pass. Replacing the listing
                wholesale brought the doc up to the code. No transcribed code
                was reformatted.

## Rule 2 removed from the architectural rules   [PASS]
Verification: `grep -rn "Rule 2|four rules|four architectural rules"` over all
              `*.md` and `*.py` -> no hits outside this log. `pytest -q` 52
              passed / 1 skipped, `make lint` clean.

Deviations:   Removed rather than rewritten, on the human's instruction
              (2026-09-07). The TASK-072 amendment note added earlier the same
              day is gone with it.

Notes:        - RULES 3 AND 4 KEEP THEIR NUMBERS. There is no Rule 2 now and
                the list reads 1, 3, 4. That is deliberate: "Rule 3" and
                "Rule 4" are referenced BY NAME in 26 places across
                `core/perspective.py`, `core/state.py`, `net/referee.py`,
                `backends/network.py`, `tests/test_perspective.py` and four
                docs. Renumbering would have silently invalidated every one of
                them, and would have falsified this log. A line in CLAUDE.md §7
                says so, so the next person does not "tidy" it.

              - The rules were stated in SIX documents, all now consistent:
                `CLAUDE.md` §7, `docs/SETUP.md` (Step 0 and the closing
                summary), `docs/ARCHITECTURE.md` §5, `docs/ONBOARDING.md`, and
                `README.md`. `docs/ONBOARDING.md` also had a prose callout
                ("This is Rule 2 working") next to the vision-client
                walkthrough; the sentence stands without the citation.

              - The ordered lists in `CLAUDE.md`, `README.md` and SETUP.md's
                summary became explicit `**Rule N**` labels. Markdown
                renumbers ordered lists on render, so a literal `1. / 3. / 4.`
                would have displayed as 1, 2, 3 and undone the point.

              - WHAT THE REMOVAL COSTS, for the record: the rule carried a
                second claim beyond the stale one, namely "nothing above the
                backend layer knows which one it is talking to". That is still
                true of `Backend` and is still the reason the same `GoToPoint`
                runs on rSim and ER-Force (`tests/test_backend_parity.py`). It
                now lives only in `docs/ARCHITECTURE.md`
                §"`Backend` and `SimBackend`", not in the rules list.

## TASK-070 (addendum) — rSoccer verified against the new binding   [PASS]
Verification: TASK-070 said "keep the int overload for rSoccer" and I had not
              actually run rSoccer against the rebuilt extension — only
              reasoned that it would work. Closed that gap:

                SSLStaticDefenders-v0      30 steps ok, obs (24,)
                SSLDribbling-v0            30 steps ok, obs (21,)
                SSLContestedPossession-v0  30 steps ok, obs (14,)
                SSLPassEndurance-v0        30 steps ok, obs (16,)
                VSS-v0                     30 steps ok, obs (40,)

              And it takes the path we intended, not by luck:
              `rsoccer_gym/Simulators/rsim.py:169` calls `robosim.SSL(...)`
              with `time_step_ms` positionally as an `int`, so it resolves to
              the millisecond overload and its physics is bit-for-bit what it
              was before. The four SSL envs are the ones that matter here;
              VSS goes through the untouched `VSS` binding.

Deviations:   None.

Notes:        The named arguments added to the SSL constructor are backward
                compatible for exactly this reason — every rSoccer call site is
                positional.

## Rule 2 reinstated; CLAUDE.md reframed for feature work   [PASS]
Verification: `grep -rn "Rule 2" README.md docs/*.md CLAUDE.md` -> the rule
              appears in all six places it was removed from (SETUP.md Step 0
              and its README listing, README.md, ONBOARDING.md, ARCHITECTURE.md
              §5, CLAUDE.md §5) with one wording. `make lint && make test`
              unchanged: 52 passed, 1 skipped.
Deviations:   Reverses the decision in "Rule 2 removed" above. The TASK-072
              split did not make the rule false: `Backend` is still the one
              interface for the match path, and `NetworkBackend` and
              `RSimBackend` both implement it. The rule is now stated in terms
              of the two protocols -- match code sees only `Backend`, training
              takes `SimBackend` -- so it names the enforcement (`make lint`
              type-checks both) rather than a slogan.
Notes:        CLAUDE.md no longer describes a one-time build from SETUP.md.
              It now says: source is authoritative, TASKS.md is what to do,
              the Done-when gate is the unit of work, one task per commit with
              a SETUP_LOG block, and the same hard prohibitions. Section 6
              ("unreliable parts of SETUP.md") is gone with the build. The
              repo path is corrected to ~/tritonbots.

## viz_rsim flickers against the ER-Force simulator   [PASS]
Verification: Reported: `python -m tbots.apps.viz_rsim --realtime --port 10020`
              with the compose stack up flickers at :8082 between the orbit
              and a pre-kickoff line-up, and the goals and field lines show
              for under a second at a time.

              On the wire, 224.5.23.2:10020 over 3 s: two senders, both
              `camera_id 0`. ER-Force (22 robots parked on the touchline,
              ~67 Hz, geometry in EVERY packet) and viz_rsim (12 robots,
              60 Hz, geometry every 300 ticks). Sampling the :8082 DOM every
              50 ms for 3 s with the pre-fix app running: 12 robots in 5
              samples, 22 in 55, and all 17 field/goal lines 0 wide in all
              60.

              After: `viz_rsim --port 10020` exits 1 naming ER-Force's source
              address; a second concurrent `viz_rsim` on 10006 exits 1 too.
              `viz_rsim --realtime` (default 10006) watched through
              ssl-vision-client v2.1.1 on :8083 <- 10006, the same version and
              flags as the new compose service: 60 samples, always 12 robots,
              17 lines, none 0 wide. `make lint` clean; `make test` 72 passed,
              1 skipped (the existing network-parity skip). New
              tests/test_multicast.py runs on real TTL-0 multicast, no mocks.
Deviations:   Not a TASKS.md task; a user-reported bug. Reverses the
              Post-Step-16 decision (2026-08-10) to watch rSim through the
              compose vision-client with `--port 10020`. That entry verified
              only that our packets REACHED 10020, never what the client drew
              with ER-Force also there. Two causes, both from sharing the port:

              1. Robots. ssl-vision-client keeps the newest frame per camera
                 id. Both sources are camera 0, so the browser shows whichever
                 arrived last.
              2. Markings. Client 2.1.1 ignores `field_lines`/`field_arcs`.
                 `SvgFieldLines` and `SvgGoal` rebuild every line and both
                 goals from the scalar fields, with stroke width
                 `lineThickness / 1000`. ER-Force's geometry omits
                 `line_thickness` (as well as the penalty and centre-circle
                 scalars TASK-071 found), so it draws them 0 wide. The client
                 pushes geometry to the browser only on change, so each of our
                 packets (every 5 s) flashed the lines in until ER-Force's
                 next packet ~15 ms later.

              Fix: rSim no longer shares a port with the simulator.
              - docker-compose.yml: new `vision-client-rsim` service, same
                pinned image, `:8083` <- `224.5.23.2:10006`. :8082 stays the
                match. Not exercised under `docker compose up` here (see
                Notes); the identical native binary was, and
                `docker compose config` parses.
              - net/multicast.py: `senders(group, port, seconds)` returns
                every source heard on a port. viz_rsim calls it before its
                first send and refuses to start if anyone is already
                publishing. TASK-058's `--render vision` and TASK-082's
                replay will hit the same trap and should do the same.
              - viz_rsim's docstring and `--port` help, README, SETUP.md 13.4
                (compose block and verify) and acceptance checks 5/6,
                ONBOARDING 1.4 and its cheat sheet, and configs/net/dev.yaml
                all now say :8083 / 10006. VisionPublisher is untouched.
Notes:        - FINDING, not fixed: because of cause 2, the MATCH view on
                :8082 draws no field lines and no goals even with nothing of
                ours running. TASK-071's "the client is drawing a 9 x 6 m
                field" is true of the canvas size only. Added to TASKS.md §5.
              - ENVIRONMENT: Docker Desktop's WSL integration proxy (started
                2026-10-31 13:03) has taken over /var/run/docker.sock. The
                stack runs on the native docker.io daemon (pid 466, socket-
                activated, started 2026-10-29), which the `docker` CLI can no
                longer reach: `docker ps` and `docker compose ps` come back
                empty while all three containers are up. CLAUDE.md says Docker
                is docker.io, not Desktop. Disabling Desktop's WSL integration
                for this distro and restarting docker.socket (sudo) puts it
                back. Until then `docker compose up -d vision-client-rsim`
                would start on the wrong engine.
