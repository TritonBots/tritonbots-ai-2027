# SSL Codebase — Setup Guide

**Audience:** anyone with a blank directory and a terminal. No prior SSL knowledge assumed.
**Outcome:** a working monorepo with a training simulator, a match simulator, a game controller, a browser visualizer, and every placeholder needed to start writing tactics and reward functions.
**Time:** 3–5 hours if nothing goes wrong. Budget a full day.

Follow the steps **in order**. Every step ends with a verification command. If the verification fails, stop and fix it — do not continue. Half of these steps depend on the previous one working.

---

## Table of contents

- [Step 0 — Assumptions and decisions already made](#step-0--assumptions-and-decisions-already-made)
- [Step 1 — System prerequisites (Ubuntu / WSL2)](#step-1--system-prerequisites)
- [Step 1M — System prerequisites (macOS)](#step-1m--system-prerequisites-macos)
- [Step 2 — Create the repository](#step-2--create-the-repository)
- [Step 3 — Python environment](#step-3--python-environment)
- [Step 4 — Fork and build rSim](#step-4--fork-and-build-rsim)
- [Step 5 — Install rSoccer](#step-5--install-rsoccer)
- [Step 6 — Verify rSim's undocumented behaviour](#step-6--verify-rsims-undocumented-behaviour)
- [Step 7 — Protobuf submodules and code generation](#step-7--protobuf-submodules-and-code-generation)
- [Step 8 — The core contracts](#step-8--the-core-contracts)
- [Step 9 — The backend layer](#step-9--the-backend-layer)
- [Step 10 — The network layer](#step-10--the-network-layer)
- [Step 11 — Skills, tactics, and RL placeholders](#step-11--skills-tactics-and-rl-placeholders)
- [Step 12 — Runnable apps](#step-12--runnable-apps)
- [Step 13 — External tools](#step-13--external-tools)
- [Step 14 — Configuration](#step-14--configuration)
- [Step 15 — Tests and CI](#step-15--tests-and-ci)
- [Step 16 — The acceptance checklist](#step-16--the-acceptance-checklist)
- [Step 17 — HPC image](#step-17--hpc-image)
- [Troubleshooting](#troubleshooting)
- [The task board](#the-task-board)

---

## Step 0 — Assumptions and decisions already made

Read this section. It explains *why* the rest of the guide looks the way it does.

### Assumptions

| Thing | Value |
|---|---|
| Team name | `TritonBots` (case-sensitive, must match the game controller exactly) |
| Python package | `tbots` |
| Repo | `github.com/tritonbots/tritonbots` |
| Division | Division B, 6v6, 9 m × 6 m field |
| Python | 3.11 |
| Supported platforms | Ubuntu 24.04 LTS, WSL2 (Ubuntu 24.04), macOS 13+ (Intel and Apple Silicon) |

### Why the package is called `tbots`

Not `ssl` — that shadows Python's standard-library TLS module, and the day a dependency imports it you get an hour of baffling errors.

Not `triton` — that collides with the **Triton compiler, which PyTorch installs as a dependency**. This one is a guaranteed, immediate breakage, not a hypothetical.

Not `tritonbots` — correct but verbose; you type it in every import.

`tbots` is short, unambiguous, and collides with nothing. `from tbots.core.state import WorldState`.

### Platform support matrix

macOS is a **first-class development and training platform**. The one thing it cannot easily do is run the networked match stack, because Docker Desktop on macOS does not support host networking and therefore cannot carry multicast to your host Python process.

| Capability | Ubuntu | WSL2 | macOS |
|---|---|---|---|
| Build rSim, run RL training | ✅ | ✅ | ✅ |
| rSoccer + pygame renderer | ✅ | ✅ | ✅ |
| Game controller (native binary) | ✅ | ✅ | ✅ |
| ssl-vision-client (native binary) | ✅ | ✅ | ✅ |
| `VisionPublisher` → browser view of rSim | ✅ | ✅ | ✅ |
| Referee multicast → our code | ✅ | ✅ | ✅ (needs a route, see Step 1M.5) |
| Networked match simulator via Docker | ✅ | ✅ | ❌ — no host networking |
| Networked match simulator built from source | ✅ | ✅ | ⚠️ possible, ~1 hour of Qt pain |
| Full `docker compose up` stack | ✅ | ✅ | ❌ |

**What this means in practice.** A recruit on a MacBook can do everything that matters day to day: train policies, write reward functions, watch rollouts in the browser, and test against the real referee. When they need the networked match backend, they use a lab Linux box, a teammate's WSL machine, or the HPC cluster. Nobody is blocked on hardware.

Anywhere you see `TritonBots`, that is the literal team name — do not change it.

### The architectural rules

These are not negotiable. They are the reason the codebase will still be maintainable in April.

**Rule 1 — `src/tbots/core/` imports nothing from the rest of the codebase.**
Everything else imports `core`. `core` defines the data types; it never depends on a simulator, a socket, or a neural network. If you find yourself adding `import robosim` or `import torch` to a file in `core/`, you have made a mistake.

**Rule 2 — Two backends, one match contract.**
`Backend` in `backends/base.py` is everything a match needs: `dt`, `geometry`, `reset()`, `step(commands)`, `close()`. `RSimBackend` and `NetworkBackend` both implement it, and no code on the match path — skills, tactics, perception — names either one. `SimBackend` extends `Backend` with the three powers only a simulator has: `step(commands, opponent_commands)`, `place()`, `set_game_state()`. Code that needs those takes a `SimBackend` in its type, which is why `SSLEnv` cannot be pointed at hardware. `make lint` type-checks both.

**Rule 3 — We are always `us`, we always attack `+x`.**
The world model has `us` and `them`, never `blue` and `yellow`. The backend flips coordinates if we are yellow or defending the positive half. Every skill, policy, and reward function is written as if we are blue attacking rightward. This eliminates an entire class of bug and halves what a policy has to learn.

**Rule 4 — Units convert exactly once, at the backend boundary.**
Above the boundary: **meters, radians, seconds**. Below it, the wire formats vary — SSL-Vision uses millimeters, rSim uses degrees. All conversion happens in `core/units.py` and the backend adapters. Nowhere else.

### What is what (the vocabulary)

New people confuse these constantly. Learn them now:

| Name | What it actually is | Physics? | Referee? |
|---|---|---|---|
| **rSim** | C++/ODE physics library with Python bindings. Runs *inside* our process. | Yes | No |
| **rSoccer** | Thin Gymnasium wrapper over rSim, plus a 2D pygame renderer. | No (delegates) | No |
| **ER-Force simulator-cli** | Standalone headless simulator. Separate process, talks UDP. This is our *match* simulator. | Yes | No |
| **grSim** | The other standalone simulator, with a 3D GUI. **We do not use it** — see Step 13.2. | Yes | No |
| **ssl-game-controller (GC)** | The **referee**. A state machine with a web UI. Has *no physics at all*. | **No** | Yes |
| **ssl-vision-client** | A browser page that draws a 2D field from vision packets. Read-only. | No | No |
| **autoRef** | Watches the game and proposes fouls to the GC. | No | Assists |

The single most common misunderstanding: **the game controller is not a simulator.** It never simulates anything. It watches, decides, and broadcasts. It can be attached to either backend, or to neither.

---

## Step 1 — System prerequisites

Everything in this step is installed system-wide, once per machine.

**On macOS, skip to [Step 1M](#step-1m--system-prerequisites-macos).** Everything after Step 1 is identical on all three platforms except where explicitly noted.

### 1.1 Base packages (Ubuntu 24.04 / WSL2)

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake pkg-config git curl wget unzip \
  autoconf automake libtool \
  libccd-dev \
  python3-dev python3-venv \
  libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev \
  libsdl2-dev \
  protobuf-compiler
```

> **`libccd-dev` was missing from the draft of this list and is load-bearing.**
> Step 1.4 builds ODE with `--enable-libccd`; without a system libccd present,
> `configure` silently substitutes ODE's bundled copy. See the note in 1.4.

**Verify:**

```bash
gcc --version && cmake --version && git --version && protoc --version
```

All four must print a version. `protoc` should be 3.x or newer.

### 1.2 Docker (for the game controller and match simulator)

```bash
sudo apt-get install -y docker.io docker-compose-v2
sudo usermod -aG docker "$USER"
newgrp docker   # or log out and back in
```

> **24.04 note — the compose package is named differently.**
> `docker-compose-plugin` is not in Ubuntu's repositories on any release; it
> comes from Docker's own apt repo. Ubuntu 24.04 ships `docker-compose-v2` in
> universe, which provides the same `docker compose` subcommand. Install that.
> Do not install the legacy hyphenated `docker-compose` — our
> `docker-compose.yml` is Compose v2 syntax.
>
> **WSL2 on 24.04:** if `docker` reports it cannot reach the daemon, start it
> with `sudo service docker start`, or enable systemd once by adding
> `[boot]` / `systemd=true` to `/etc/wsl.conf` and running `wsl --shutdown`
> from PowerShell.

**Verify:**

```bash
docker run --rm hello-world
```

Must print "Hello from Docker!". If it says "permission denied", you skipped the `usermod`/`newgrp`.

### 1.3 `uv` (Python package manager)

We use `uv` instead of pip/poetry because it is dramatically faster and it pins everything by default.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"   # or restart your shell
```

**Verify:**

```bash
uv --version
```

### 1.4 ODE 0.16.2 — the physics library rSim needs

**This is the step most likely to fail. Read it carefully.**

rSim requires ODE version **0.16.2**, built **with libccd collision support** and **double precision**. Ubuntu's `libode-dev` package is *not* built the way rSim expects. Build it from source.

> ### ⚠️ 24.04 WARNING — READ THIS BEFORE YOU RUN ANYTHING BELOW
>
> On Ubuntu 22.04 the packaged ODE was a visibly different version, so a
> stray `libode-dev` was easy to spot. **On Ubuntu 24.04, `libode-dev` is
> version 0.16.2 — the exact version string we want.**
>
> That means the verification command at the end of this step,
> `pkg-config --modversion ode`, **will print `0.16.2` and appear to pass
> even if you never built anything**, resolving instead to Debian's build,
> which is not configured with the flags rSim needs. rSim will then compile
> and import cleanly and produce wrong physics. This is the worst class of
> failure: silent, and downstream of everything.
>
> Remove the packaged ODE first, and check nothing else on the machine
> depends on it:
>
> ```bash
> dpkg -l | grep -i '^ii.*ode'
> apt-cache rdepends --installed libode-dev libode8t64 2>/dev/null
> sudo apt-get remove libode-dev libode8t64
> ```
>
> Read what `apt` proposes to remove alongside it before confirming. If a
> legacy simulator is the only reverse-dependency, that is fine. Then
> confirm the field is clear:
>
> ```bash
> pkg-config --modversion ode   # must now FAIL with "not found"
> ```
>
> If that still prints a version, something else installed an ODE. Find it
> with `pkg-config --variable=libdir ode` before continuing.

```bash
cd /tmp
wget https://bitbucket.org/odedevs/ode/downloads/ode-0.16.2.tar.gz
tar -xzf ode-0.16.2.tar.gz
cd ode-0.16.2

./configure \
  --enable-double-precision \
  --with-box-cylinder=libccd \
  --enable-libccd \
  --enable-shared \
  --disable-demos \
  --disable-asserts

make -j"$(nproc)"
sudo make install
sudo ldconfig
```

> ### ⚠️ Watch the `configure` summary for the libccd source
>
> `--enable-libccd` makes ODE *use* libccd colliders; `--with-libccd` chooses
> *which* libccd, and defaults to `system`. **If no system libccd is
> installed, `configure` falls back to ODE's bundled copy and still exits 0.**
> There is no flag that turns this into an error — passing
> `--with-libccd=system` explicitly does not help, it falls back just the same
> (verified). The only signal is two lines near the end of `configure`:
>
> ```
>   Use libccd:              yes
>   libccd source:           system      <-- must say system, not internal
> ```
>
> Our reference build is **system** libccd (Ubuntu `libccd2` 2.1, double
> precision). If yours says `internal`, `sudo apt-get install libccd-dev` and
> rebuild from a clean tree.

> ### If the bitbucket tarball is unavailable
>
> It is served through a signed S3 redirect and some networks block it. The
> git repository is equivalent, with two gotchas:
>
> ```bash
> cd /tmp
> git clone https://bitbucket.org/odedevs/ode.git && cd ode
> git checkout 0.16.2      # the tag is "0.16.2" — there is no "ode-0.16.2"
> autoreconf -fi           # the git tree ships no ./configure; generate it
> ```
>
> Then run the same `./configure … && make && sudo make install && sudo
> ldconfig`. This is what `autoconf automake libtool` are in the 1.1 list for.

**Verify:**

```bash
pkg-config --modversion ode
pkg-config --variable=libdir ode
ldd /usr/local/lib/libode.so.8 | grep libccd
python3 -c "import ctypes; l=ctypes.CDLL('/usr/local/lib/libode.so.8'); \
  l.dGetConfiguration.restype=ctypes.c_char_p; print(l.dGetConfiguration().decode())"
```

The third must print a `libccd.so.2` line — if it prints nothing, you got the
bundled libccd (see the note above). The fourth asks the compiled library what
it is, rather than trusting a header or a `.pc` file some other build could
have left behind, and must contain `ODE_double_precision`:

```
ODE ODE_EXT_no_debug ODE_EXT_trimesh ODE_EXT_opcode ODE_OPC_new_collider
ODE_EXT_threading ODE_THR_builtin_impl ODE_double_precision
```

The first must print `0.16.2`. **The second must print `/usr/local/lib`** —
if it prints `/usr/lib/x86_64-linux-gnu`, you are still resolving to the
packaged ODE and the build you just did is being ignored. Go back to the
24.04 warning above.

If `pkg-config` cannot find it at all, ODE installed to `/usr/local` but your `PKG_CONFIG_PATH` doesn't include it. Fix:

```bash
echo 'export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
pkg-config --modversion ode
```

Also verify double precision is on — this matters, because rSim will silently produce garbage physics if it's compiled against a single-precision ODE:

```bash
grep -r "dDOUBLE" /usr/local/include/ode/precision.h
```

Should show `#define dDOUBLE`.

Finally, confirm the linker will also prefer our build at runtime, not just
at compile time:

```bash
ldconfig -p | grep libode
```

`/usr/local/lib/libode.so` should appear, and no `/usr/lib/x86_64-linux-gnu`
entry should. After Step 4 builds rSim, re-check the binding itself:

```bash
ldd "$(python -c 'import robosim._robosim as m; print(m.__file__)')" | grep -i ode
```

Note `robosim._robosim`, not `robosim`. The latter is the package
`__init__.py`, and `ldd` on a `.py` file tells you nothing; the compiled
extension module is what actually carries the ODE dependency.

It must resolve to `/usr/local/lib/libode.so`. This is the single most
valuable check in the whole setup — it is what separates "rSim imports"
from "rSim is correct".

---

## Step 1M — System prerequisites (macOS)

Ubuntu/WSL users: skip this, you already did Step 1.

Works on macOS 13 (Ventura) and later, on both Intel and Apple Silicon. There are no prebuilt rSim wheels for macOS or ARM at all, so we build everything from source — which we were doing on Linux anyway.

### 1M.1 Xcode command line tools and Homebrew

```bash
xcode-select --install    # click through the dialog; skip if already installed

# Homebrew, if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

On Apple Silicon, Homebrew installs to `/opt/homebrew`. Make sure it is on your PATH:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
source ~/.zprofile
brew --prefix    # should print /opt/homebrew (ARM) or /usr/local (Intel)
```

### 1M.2 Base packages

```bash
brew install cmake pkg-config autoconf automake libtool git curl wget protobuf
```

**Verify:**

```bash
clang --version && cmake --version && protoc --version
```

> **Do not `brew install ode`.** Homebrew's ODE is a different version and is not built with the flags rSim needs. You will build it yourself in 1M.4.

### 1M.3 Docker Desktop (optional on macOS)

Install it if you like, but understand the limitation up front: **Docker Desktop on macOS does not support `network_mode: host`.** Containers live behind a VM's NAT, and multicast does not cross that boundary. This means the `docker-compose.yml` stack in Step 13 **will not work on macOS**.

Mac users run the game controller and ssl-vision-client as **native binaries** instead. Both are self-contained Go programs with official darwin builds. That covers everything except the networked match simulator.

### 1M.4 ODE 0.16.2 from source

Same version and flags as Linux. The only difference is telling `configure` where Homebrew lives.

```bash
export BREW_PREFIX="$(brew --prefix)"

cd /tmp
curl -LO https://bitbucket.org/odedevs/ode/downloads/ode-0.16.2.tar.gz
tar -xzf ode-0.16.2.tar.gz
cd ode-0.16.2

./configure \
  --prefix="$BREW_PREFIX" \
  --enable-double-precision \
  --with-box-cylinder=libccd \
  --enable-libccd \
  --enable-shared \
  --disable-demos \
  --disable-asserts

make -j"$(sysctl -n hw.ncpu)"
make install
```

We install into the Homebrew prefix rather than `/usr/local` so that (a) it does not need `sudo` on Apple Silicon and (b) `pkg-config` finds it without extra configuration.

**Verify:**

```bash
pkg-config --modversion ode          # must print 0.16.2
grep dDOUBLE "$(brew --prefix)/include/ode/precision.h"
```

If `pkg-config` cannot find it:

```bash
echo 'export PKG_CONFIG_PATH="$(brew --prefix)/lib/pkgconfig:$PKG_CONFIG_PATH"' >> ~/.zshrc
echo 'export DYLD_LIBRARY_PATH="$(brew --prefix)/lib:$DYLD_LIBRARY_PATH"' >> ~/.zshrc
source ~/.zshrc
```

> **Apple Silicon architecture trap.** If you installed Python or Homebrew under Rosetta at some point, you can end up with an x86_64 Python trying to load an arm64 ODE, and the error message will not tell you that. Check both:
> ```bash
> file "$(brew --prefix)/lib/libode.dylib"   # should say arm64 on M-series
> python3 -c "import platform; print(platform.machine())"   # should say arm64
> ```
> If they disagree, you have a mixed-architecture toolchain. Reinstall Homebrew natively and start this step over.

### 1M.5 Enable multicast on the loopback interface

**This is the macOS-specific step that nothing else warns you about.**

macOS does not route multicast to `lo0` by default. Without a route, our code will send referee and vision packets into the void and receive nothing, with no error message anywhere. Add the route:

```bash
sudo route -n add -net 224.0.0.0/4 -interface lo0
```

**Verify:**

```bash
netstat -rn | grep 224
```

This route does **not** survive a reboot. Either re-run it after each restart, or make it permanent with a LaunchDaemon. The simplest reliable option is to add it to the project's dev script — Step 12 apps will remind you.

There is a second macOS quirk: joining a multicast group on the wildcard interface `0.0.0.0` is unreliable. Our `net/multicast.py` takes an `iface` argument for exactly this reason. On macOS, set it to `127.0.0.1` in `configs/net/dev.yaml`:

```yaml
interface: "127.0.0.1"   # macOS: use loopback explicitly. Linux: "0.0.0.0" is fine.
```

### 1M.6 `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

### 1M.7 What macOS users skip

- **Step 13.4 (`docker-compose.yml`)** — will not work. Use the native binaries from Steps 13.1 and 13.3.
- **Acceptance check 7 (full compose stack)** — replaced by a macOS variant in Step 16.

Everything else in this guide applies unchanged.

---

## Step 2 — Create the repository

```bash
mkdir -p ~/code/tritonbots && cd ~/code/tritonbots
git init
git branch -M main
```

Now create the entire directory tree in one shot. Copy this whole block and paste it into your terminal:

```bash
mkdir -p \
  configs/{net,reward,env,train} \
  containers \
  docs \
  protos \
  scripts \
  src/tbots/{core,backends,net,perception,skills,tactics,viz,apps} \
  src/tbots/rl/{envs,rewards,obs,wrappers} \
  third_party \
  tools/bin \
  tests

# Python package markers
for d in \
  src/tbots \
  src/tbots/core src/tbots/backends src/tbots/net src/tbots/perception \
  src/tbots/skills src/tbots/tactics src/tbots/viz src/tbots/apps \
  src/tbots/rl src/tbots/rl/envs src/tbots/rl/rewards src/tbots/rl/obs src/tbots/rl/wrappers
do
  touch "$d/__init__.py"
done

touch tests/__init__.py
```

**Verify:**

```bash
find . -type d -not -path './.git*' | sort
```

You should see the full tree. The final layout, once you finish this guide, will be:

```
tritonbots/
├── .gitignore
├── .gitmodules
├── Makefile
├── README.md
├── docker-compose.yml
├── pyproject.toml
├── configs/
│   ├── net/{dev.yaml,competition.yaml}
│   ├── env/div_b_6v6.yaml
│   ├── reward/example.yaml
│   └── train/ppo_default.yaml
├── containers/{dev.Dockerfile,train.def}
├── docs/{ARCHITECTURE.md,ONBOARDING.md}
├── protos/                       # submodules — league .proto files
│   ├── ssl-game-controller/
│   ├── ssl-simulation-protocol/
│   └── ssl-vision/
├── scripts/{gen_proto.sh,verify_rsim.py,fetch_tools.sh}
├── src/tbots/
│   ├── _pb/                      # GENERATED. gitignored.
│   ├── core/                     # the contract. depends on nothing.
│   ├── backends/                 # rsim.py + network.py
│   ├── net/                      # sockets and protobuf glue
│   ├── perception/               # tracking and filtering
│   ├── skills/                   # single-robot behaviours
│   ├── tactics/                  # multi-robot decision making
│   ├── rl/                       # environments, rewards, training
│   ├── viz/                      # renderers
│   └── apps/                     # runnable entry points
├── third_party/
│   ├── rsim/                     # submodule of OUR fork
│   └── rsoccer/                  # submodule of OUR fork
├── tools/bin/                    # downloaded binaries. gitignored.
└── tests/
```

### 2.1 `.gitignore`

Create `.gitignore`:

```gitignore
# Generated protobuf code — never commit, always regenerate
src/tbots/_pb/

# Downloaded binaries
tools/bin/*
!tools/bin/.gitkeep

# Game controller runtime state
tools/gc-data/

# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.pytest_cache/
.ruff_cache/

# Training artifacts
runs/
checkpoints/
wandb/
*.pt
*.onnx

# Editors / OS
.vscode/
.idea/
.DS_Store
```

```bash
touch tools/bin/.gitkeep
git add .gitignore tools/bin/.gitkeep
git commit -m "chore: repository skeleton"
```

---

## Step 3 — Python environment

Create `pyproject.toml` in the repo root:

```toml
[project]
name = "tbots"
version = "0.1.0"
description = "RoboCup SSL Division B team software"
requires-python = ">=3.11,<3.12"

dependencies = [
  "numpy>=1.26",
  "protobuf>=4.25",
  "gymnasium>=0.29",
  "pyyaml>=6.0",
  "hydra-core>=1.3",
  "pygame>=2.5",
]

[project.optional-dependencies]
train = [
  "torch>=2.2",
  "wandb>=0.16",
  "tensorboard>=2.15",
]
dev = [
  "pytest>=8.0",
  "pytest-timeout>=2.2",
  "ruff>=0.4",
  "grpcio-tools>=1.60",
  "protoletariat>=3.2",
  "mypy>=1.9",
]

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
timeout = 120
```

Create the virtual environment and install:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,train]"
```

**Verify:**

```bash
python -c "import tbots; print('ok')" 2>/dev/null || echo "expected — tbots package is empty so far"
python -c "import numpy, google.protobuf, gymnasium, torch; print('deps ok')"
```

> **On macOS**, if `uv venv --python 3.11` cannot find an interpreter, install one: `brew install python@3.11`, then retry. Do not use the system Python at `/usr/bin/python3` — it is managed by Apple and you cannot install headers against it reliably.

---

## Step 4 — Fork and build rSim

### 4.1 Why we fork

rSim and rSoccer were last published to PyPI in **October 2021**, with binary wheels only for CPython 3.6–3.10 on x86-64 Linux. There are no wheels for Python 3.11+, none for macOS, none for ARM. Upstream is effectively dormant.

We are not "adding a dependency". We are **adopting an orphaned codebase**. That means we fork it, we build it from source, and when it breaks we fix it ourselves. Nobody upstream is coming to help.

By contrast we do **not** fork the simulator or the game controller. Those are the reference implementations that referees and opponents assume. The moment we fork one, "it works in our sim" stops being evidence that it works in theirs.

### 4.1a Why Python 3.11 and not 3.10

**Why 3.11 might fail.** rSim is a pybind11 extension, and its `setup.py` pins a 2021-era pybind11. **pybind11 gained Python 3.11 support in version 2.10.0**; anything older fails to compile against 3.11 headers, because CPython 3.11 reorganised the frame-object internals that pybind11 reaches into. You will see errors mentioning `PyFrameObject`, `f_code`, or `_PyEval_EvalFrameDefault`. A secondary risk is that 2021 C++ hits stricter modern compilers — GCC 12+ and recent clang require explicit `#include <cstdint>` where older versions leaked it in transitively. **On Ubuntu 24.04 this is not a risk, it is a near-certainty:** noble ships GCC 13 as the default compiler, where 22.04 shipped GCC 11. Expect `'uint32_t' was not declared in this scope` (or similar) on the first build, and expect to add the include. Budget for it rather than being surprised by it.

Both are small, known fixes: bump pybind11 to `>=2.11`, add the missing include, ensure `-std=c++17`. **That is exactly the situation we forked in order to be able to fix.**

**Can we guarantee 3.10 works?** Not guarantee — but it is much better evidenced. Upstream published cp310 wheels, which means they actually compiled and tested the source against 3.10. It is the newest version with proof of life.

**So why not just stay on 3.10?** Three reasons, and the first is decisive:

1. **Python 3.10 reaches end of life in October 2026** — during our season. Being on an EOL interpreter when a CVE lands in the dependency tree is a much worse problem than one afternoon of build debugging.
2. PyTorch and the wider scientific stack are steadily dropping 3.10. We do not want a dependency resolver fight in March.
3. Recent Ubuntu and most HPC modules default to 3.11 or 3.12. Matching the platform reduces friction on the cluster.

**The policy:** target 3.11. If someone cannot get it building within one working day, fall back to 3.10 to unblock the team — but open a ticket (TASK-002) and fix it properly before recruitment. Treat 3.10 as a temporary state, not a decision.

### 4.2 Fork on GitHub

In a browser:

1. Go to `https://github.com/robocin/rSim` → click **Fork** → select your org → create.
2. Go to `https://github.com/robocin/rSoccer` → click **Fork** → select your org → create.

### 4.3 Add as submodules

```bash
cd ~/code/tritonbots
git submodule add https://github.com/tritonbots/rSim.git third_party/rsim
git submodule add https://github.com/tritonbots/rSoccer.git third_party/rsoccer
git commit -m "chore: vendor rSim and rSoccer forks"
```

### 4.4 Build rSim

```bash
source .venv/bin/activate
cd third_party/rsim
uv pip install -e .
cd ../..
```

This compiles the C++ extension against the ODE you built in Step 1.4. It will take 1–3 minutes.

> **Why `-e .` works here.** Our fork's build backend is `scikit-build-core`
> (`pyproject.toml`), which supports editable installs natively: `.py` edits
> under `src/robosim/` take effect on the next import with no reinstall,
> while the compiled `_robosim` extension is built once at install time and
> needs a re-`install` after any C++ change. Upstream's original pairing of
> `setup.py` (`skbuild.setup(...)`) with a `setuptools.build_meta` backend
> in `pyproject.toml` cannot support this: under PEP 660 the setuptools
> backend handles `build_editable` itself and never invokes classic
> scikit-build's cmake step, so `-e .` exits 0 having silently built a
> package with no extension module at all: `import robosim` itself then
> fails with `ModuleNotFoundError: No module named 'robosim._robosim'`,
> because `robosim/__init__.py` does `from ._robosim import VSS, SSL` at
> import time. Do not reintroduce that pairing.

**Verify:**

```bash
python -c "import robosim; print(robosim.__file__)"
```

**If the build fails**, work through these in order:

| Error contains | Cause | Fix |
|---|---|---|
| `Could NOT find ODE` | `PKG_CONFIG_PATH` missing `/usr/local/lib/pkgconfig` | Redo Step 1.4's `PKG_CONFIG_PATH` export |
| `dSINGLE` / precision mismatch | ODE built single-precision | Rebuild ODE with `--enable-double-precision` |
| `PyFrameObject`, `f_code`, or pybind11 errors | Pinned pybind11 predates 3.11 support (needs >= 2.10.0) | Edit `third_party/rsim/setup.py` and/or `pyproject.toml`, bump pybind11 to `>=2.11`, commit to your fork |
| `error: 'X' was not declared` in C++ | Newer GCC is stricter than 2021's. On 24.04 you are on GCC 13. | Add the missing `#include <cstdint>` to the offending header, and ensure `-std=c++17` is in the compile flags in `setup.py` |
| ODE symbols resolve but physics is wrong / robots jitter | You linked against Ubuntu's packaged ODE, not our build — on 24.04 both report version 0.16.2 | Re-read the 24.04 warning in Step 1.4, remove `libode-dev`, rebuild ODE and rSim |
| `library not found` / `ld: symbol(s) not found` on macOS | ODE installed outside the linker path | `export LDFLAGS="-L$(brew --prefix)/lib"` and `export CPPFLAGS="-I$(brew --prefix)/include"`, then retry |
| Nothing works after one working day | — | Fall back to Python 3.10: `uv venv --python 3.10`, edit `requires-python` in `pyproject.toml`, redo Step 3, retry. **Open TASK-002 and fix 3.11 before recruitment.** |

Whatever you fix, **commit it to your fork** and push. That is the entire point of forking:

```bash
cd third_party/rsim
git add -A && git commit -m "fix: build on python 3.11" && git push
cd ../.. && git add third_party/rsim && git commit -m "chore: bump rsim"
```

---

## Step 5 — Install rSoccer

rSoccer's `setup.py` pins `rc-robosim==1.2.0` from PyPI. We must remove that pin, because we just built our own `robosim` from source and pip will otherwise happily overwrite it with the stale 2021 wheel.

```bash
cd third_party/rsoccer
grep -n "robosim" setup.py pyproject.toml 2>/dev/null
```

Edit whichever file lists it and **delete the `rc-robosim` line from the dependency list**. Then:

```bash
uv pip install -e . --no-deps
uv pip install -e .    # picks up the remaining deps
cd ../..
```

**Verify:**

```bash
python - << 'PYEOF'
import gymnasium as gym
import rsoccer_gym
env = gym.make("SSLGoToBall-v0")
obs, _ = env.reset(seed=0)
print("obs shape:", obs.shape)
print("action space:", env.action_space)
env.close()
print("rsoccer ok")
PYEOF
```

If this prints shapes and "rsoccer ok", both libraries are working.

> **What we use rSoccer for:** its 2D pygame renderer, its `SSLBaseEnv` as a reference implementation, and its benchmark task environments to sanity-check our own. We do **not** build our training environments on top of `SSLBaseEnv` — we write our own on top of our `Backend` interface, because rSoccer's base class hardcodes assumptions we need to control (coordinate conventions, referee handling, opponent policies).

---

## Step 6 — Verify rSim's undocumented behaviour

**Do not skip this step.** rSim's documentation contradicts itself in two places, and both contradictions will silently corrupt your training if you guess wrong. Empirical probing also surfaces two more undocumented behaviours that are just as dangerous and that no README mentions at all: a wrong action vector length does not raise an exception, and rSim's reported velocities are only correct if `get_state()` is called exactly once per `step()`.

The two documented conflicts:

1. **Field type.** The rSim README's SSL section says `0 = Division A, 1 = Division B, 2 = Hardware Challenges`. rSoccer's SSL README says `0 = the 6v6 competition field, 1 = 11v11, 2 = the 2021 hardware challenge field`. These are opposite claims about what `0` means.
2. **Action vector length.** The rSim README's comment describes 8 fields per robot, but the code example in the same README constructs 6.

We resolve all of this empirically, right now, and write the answers down.

Create `scripts/verify_rsim.py`:

```python
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
```

Run it and **save the output**:

```bash
mkdir -p docs
python scripts/verify_rsim.py | tee docs/RSIM_FACTS.md
```

Now open `docs/RSIM_FACTS.md`, read it, and at the top of the file write down the four answers in plain English. For reference, on our fork (rSim commit `69f0d8e`) these came back as:

```markdown
# rSim facts — VERIFIED, do not guess

Verified on: 2026-08-10, rSim commit <paste git sha>

- Division B (9.0 x 6.0 m) is `field_type = 1`     (0 is Division A, 2 is a 6x4 m field)
- BALL_STRIDE = 5   ROBOT_STRIDE = 11              (len(get_state()) == 5 + 11 * n_robots)
- Action vector length = 8                         (a short vector reads out of bounds -- it does NOT raise)
- Angles: state/reset poses are DEGREES, commanded angular velocity is RADIANS/second (asymmetric)

<raw script output follows>
```

If you are building against a *different* rSim fork commit, do not assume these values carry over — re-run the script and reconfirm. `field_type` in particular is not something the SSL rules fix; it is whatever this specific fork's C++ happens to map, and a different revision of the same fork could renumber it.

```bash
git add docs/RSIM_FACTS.md scripts/verify_rsim.py
git commit -m "docs: verified rSim conventions"
```

Everything in Step 9 depends on these four numbers being correct.

---

## Step 7 — Protobuf submodules and code generation

The league publishes its wire formats as `.proto` files. We never hand-write these, and we never copy-paste them — we pin the upstream repos and generate Python from them.

### 7.1 Add the submodules

```bash
cd ~/code/tritonbots

git submodule add https://github.com/RoboCup-SSL/ssl-game-controller.git      protos/ssl-game-controller
git submodule add https://github.com/RoboCup-SSL/ssl-simulation-protocol.git  protos/ssl-simulation-protocol
git submodule add https://github.com/RoboCup-SSL/ssl-vision.git               protos/ssl-vision
```

**Pin them to a tag.** Never track `master` — a silent upstream change during competition week is exactly the failure mode we are avoiding.

```bash
cd protos/ssl-game-controller && git checkout v3.21.0 && cd ../..
cd protos/ssl-simulation-protocol && git checkout master && cd ../..   # no tags; pin the SHA below
cd protos/ssl-vision && git checkout master && cd ../..

git add protos .gitmodules
git commit -m "chore: pin league protocol definitions"
```

Record the exact SHAs so a future you can reproduce this:

```bash
git submodule status > docs/PINNED_VERSIONS.txt
git add docs/PINNED_VERSIONS.txt && git commit -m "docs: record pinned protocol SHAs"
```

> **Which repo has what.** The **game controller** repo owns `ssl_gc_referee_message.proto` (the referee message teams receive), `ssl_gc_common.proto`, `ssl_gc_game_event.proto`, `ssl_gc_ci.proto`, `ssl_gc_rcon_team.proto`, and the API protos. The **simulation-protocol** repo owns `ssl_simulation_robot_control.proto`, `ssl_simulation_robot_feedback.proto`, `ssl_simulation_control.proto`, and `ssl_simulation_config.proto`. The **ssl-vision** repo owns the detection, geometry, wrapper, and tracked-wrapper protos.
>
> The old `ssl-refbox` repo is **archived and legacy**. Its message was called `SSL_Referee`; the current one is `Referee`, and `TeamInfo.goalie` is now `TeamInfo.goalkeeper`. The wire format is backwards-compatible, but generate from the game-controller repo — never from refbox.

### 7.2 The generation script

Create `scripts/gen_proto.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="src/tbots/_pb"

PROTO_DIRS=(
  "protos/ssl-game-controller/proto"
  "protos/ssl-simulation-protocol/proto"
  "protos/ssl-vision/src/shared/proto"
)

for d in "${PROTO_DIRS[@]}"; do
  [ -d "$d" ] || { echo "MISSING: $d — did you init submodules?"; exit 1; }
done

INCLUDES=()
FILES=()
for d in "${PROTO_DIRS[@]}"; do
  INCLUDES+=("-I$d")
  while IFS= read -r f; do FILES+=("$f"); done < <(find "$d" -name '*.proto')
done

echo "Generating ${#FILES[@]} proto files -> $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"

python -m grpc_tools.protoc "${INCLUDES[@]}" \
  --python_out="$OUT" --pyi_out="$OUT" "${FILES[@]}"

# Rewrite top-level imports into package-relative ones.
protol --create-package --in-place --python-out "$OUT" \
  protoc "${INCLUDES[@]/#-I/--proto-path=}" "${FILES[@]}"

touch "$OUT/__init__.py"
echo "done. $(find "$OUT" -name '*_pb2.py' | wc -l) modules generated."
```

```bash
chmod +x scripts/gen_proto.sh
./scripts/gen_proto.sh
```

> **The gotcha this script exists to solve.** These `.proto` files import each other by bare filename (`import "ssl_gc_common.proto";`). Python's `protoc` turns that into a *top-level* import: `import ssl_gc_common_pb2`. The moment the generated files live inside a package, that import fails with `ModuleNotFoundError: No module named 'ssl_gc_common_pb2'`. [`protoletariat`](https://pypi.org/project/protoletariat/) (the `protol` command) rewrites them into relative imports. Every team hits this on day one; now you won't.

> **The second gotcha: you cannot generate all of them.** None of the league's
> protos declare a `package`, so every message lands in the global protobuf
> namespace — and all three repos vendor their own private copy of the SSL
> vision protos. Across the three submodules, 26 top-level symbols are defined
> more than once (`Team`, `RobotId`, `SSL_DetectionFrame`, `SSL_GeometryData`,
> `SSL_WrapperPacket`, `TrackedFrame`, `Vector2`, …). protobuf's descriptor
> pool is process-global and keyed by symbol, so the second definition raises
> `TypeError: Couldn't build proto file into descriptor pool: duplicate symbol
> 'Team'` at *import* time. Generating all 38 protos yields a package in which
> 13 modules cannot be imported, and which 13 depends on import order.
>
> The script therefore generates a curated, conflict-free subset (23 protos),
> picking the authoritative repo for each file per the ownership note in 7.1
> and dropping the vendored duplicates. The exclusions are listed with their
> rationale in the script's header comment. Two consequences worth knowing
> before you reach Step 10:
>
> - **`SimulatorCommand` is not generated.** `ssl_simulation_control.proto`
>   imports simulation-protocol's *vendored* `ssl_gc_common.proto`, whose
>   `Team`/`Division`/`RobotId` collide with the game controller's own
>   `state/ssl_gc_common.proto` that `Referee` depends on. The referee and the
>   simulator-control protos genuinely cannot coexist in one process as these
>   repos are published. `net/sim_control.py` (TASK-015) is blocked on
>   resolving that — it is only needed for episode resets against a simulator,
>   never on the match path.
> - **The GC `ci/` protos are not generated** either (they pull in the same
>   vendored vision copies), so TASK-017's ci-mode client is likewise blocked.
>
> Everything the match path actually needs — `Referee`, `RobotControl`,
> `RobotControlResponse`, `SSL_WrapperPacket`, `TrackerWrapperPacket`, and the
> `rcon` team-client protos — is present and imports cleanly together.

**Verify:**

```bash
python - << 'PYEOF'
from tbots._pb.state.ssl_gc_referee_message_pb2 import Referee
from tbots._pb.ssl_simulation_robot_control_pb2 import RobotControl
from tbots._pb.messages_robocup_ssl_wrapper_pb2 import SSL_WrapperPacket

r = Referee()
print("Referee stages:", [f.name for f in Referee.Stage.DESCRIPTOR.values][:5], "...")
print("Referee commands:", [f.name for f in Referee.Command.DESCRIPTOR.values])
print("proto ok")
PYEOF
```

> **Note the `state.` in that first import.** The game controller's protos import
> each other subdirectory-qualified (`import "state/ssl_gc_common.proto";`), so
> the include root has to be `protos/ssl-game-controller/proto` and the
> generated package mirrors the upstream layout:
> `tbots._pb.state.ssl_gc_referee_message_pb2`, `tbots._pb.rcon.…`,
> `tbots._pb.tracker.…`. Flattening is not an option — it would break those
> imports. The simulation-protocol and ssl-vision protos import their deps by
> bare filename, so those two land flat, as above. Anywhere else in this guide
> that imports a `ssl_gc_*` module needs the same subdirectory prefix.

You should see the full `Command` enum: `HALT`, `STOP`, `NORMAL_START`, `FORCE_START`, `PREPARE_KICKOFF_YELLOW/BLUE`, `PREPARE_PENALTY_YELLOW/BLUE`, `DIRECT_FREE_YELLOW/BLUE`, `INDIRECT_FREE_YELLOW/BLUE`, `TIMEOUT_YELLOW/BLUE`, `GOAL_YELLOW/BLUE`, `BALL_PLACEMENT_YELLOW/BLUE`.

> The `INDIRECT_FREE_*` and `GOAL_*` values are **legacy**. Current SSL rules use only direct free kicks, and goals are signalled through game events plus a stop. They remain in the proto for compatibility. Do not build logic around them.

Add the generation step to a `Makefile` in the repo root:

```make
.PHONY: proto test lint fmt tools clean

proto:
	./scripts/gen_proto.sh

test: proto
	pytest -q

lint:
	ruff check src tests
	mypy src/tbots/core

fmt:
	ruff format src tests
	ruff check --fix src tests

tools:
	./scripts/fetch_tools.sh

clean:
	rm -rf src/tbots/_pb .pytest_cache .ruff_cache
```

```bash
git add scripts/gen_proto.sh Makefile && git commit -m "build: protobuf codegen"
```

---

## Step 8 — The core contracts

This is the heart of the codebase. Four small files that nothing else may contradict.

### 8.1 `src/tbots/core/units.py`

```python
"""Unit conventions. Read once; then never think about units again.

CANONICAL UNITS — used everywhere above the backend boundary:
    position           meters
    angle              radians, wrapped to (-pi, pi]
    linear velocity    m/s
    angular velocity   rad/s
    time               seconds (float)
    kick speed         m/s

WIRE UNITS — used only inside backends/ and net/:
    SSL-Vision         millimeters, radians
    rSim               meters, DEGREES  (verify against docs/RSIM_FACTS.md)
    sim protocol       meters, radians

Conversion happens exactly twice in this codebase: on the way in from a
backend, and on the way out to a backend. If you are converting units
anywhere else, you are creating a bug.
"""

import math

MM_PER_M: float = 1000.0


def mm_to_m(v: float) -> float:
    return v / MM_PER_M


def m_to_mm(v: float) -> float:
    return v * MM_PER_M


def deg_to_rad(v: float) -> float:
    return v * math.pi / 180.0


def rad_to_deg(v: float) -> float:
    return v * 180.0 / math.pi


def wrap_angle(a: float) -> float:
    """Wrap any angle into (-pi, pi]."""
    a = (a + math.pi) % (2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def angle_diff(target: float, current: float) -> float:
    """Shortest signed rotation from `current` to `target`, in (-pi, pi]."""
    return wrap_angle(target - current)
```

### 8.2 `src/tbots/core/geometry.py`

```python
"""Field dimensions and geometric helpers. All meters, all radians.

Coordinate system (after normalisation — see core/state.py):
    +x  points at THEIR goal. We always attack in the +x direction.
    +y  is 90 degrees counter-clockwise from +x.
    origin is the centre of the field.
    theta = 0 means the robot's kicker faces +x.

This is true regardless of which colour we are or which half we started on.
The backend does the flipping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FieldGeometry:
    length: float           # touchline to touchline, along x
    width: float            # goal line to goal line, along y
    goal_width: float
    goal_depth: float
    penalty_depth: float    # defense area extent along x
    penalty_width: float    # defense area extent along y
    boundary_width: float
    center_circle_radius: float
    max_robots: int

    @property
    def half_length(self) -> float:
        return self.length / 2.0

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    @property
    def their_goal(self) -> tuple[float, float]:
        return (self.half_length, 0.0)

    @property
    def our_goal(self) -> tuple[float, float]:
        return (-self.half_length, 0.0)

    def inside_field(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (abs(x) <= self.half_length - margin
                and abs(y) <= self.half_width - margin)

    def inside_our_defense_area(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (x <= -self.half_length + self.penalty_depth + margin
                and abs(y) <= self.penalty_width / 2.0 + margin)

    def inside_their_defense_area(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (x >= self.half_length - self.penalty_depth - margin
                and abs(y) <= self.penalty_width / 2.0 + margin)


# Official Division B geometry (9 x 6 m field, 6 robots per team).
DIV_B = FieldGeometry(
    length=9.0,
    width=6.0,
    goal_width=1.0,
    goal_depth=0.18,
    penalty_depth=1.0,
    penalty_width=2.0,
    boundary_width=0.3,
    center_circle_radius=0.5,
    max_robots=6,
)

# Official Division A geometry (12 x 9 m field, 11 robots per team).
DIV_A = FieldGeometry(
    length=12.0,
    width=9.0,
    goal_width=1.8,
    goal_depth=0.18,
    penalty_depth=1.8,
    penalty_width=3.6,
    boundary_width=0.3,
    center_circle_radius=0.5,
    max_robots=11,
)

# Robot physical constants (SSL rule limits).
ROBOT_RADIUS: float = 0.09          # 180 mm diameter limit
ROBOT_HEIGHT: float = 0.15
BALL_RADIUS: float = 0.0215         # golf ball
MAX_KICK_SPEED: float = 6.5         # m/s — rules cap kicks at 6.5 m/s


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_to(frm: tuple[float, float], to: tuple[float, float]) -> float:
    return math.atan2(to[1] - frm[1], to[0] - frm[0])
```

> **Verify these numbers against the current rulebook before your first match.** They are correct as of the 2025 rules, but the league adjusts field and robot dimensions between seasons. `https://robocup-ssl.github.io/ssl-rules/sslrules.html` is authoritative. Better still: at runtime, prefer the geometry that arrives in the SSL-Vision `SSL_GeometryData` packet over these constants, and use these only as a fallback.

### 8.3 `src/tbots/core/gamestate.py`

```python
"""Our normalised view of the referee. Nothing above net/ imports a protobuf."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class Play(Enum):
    """Coarse behavioural mode. Derived from the referee Command."""

    HALT = auto()              # all motion must stop immediately
    STOP = auto()              # move freely, stay 0.5 m from the ball
    PREPARE_KICKOFF = auto()   # take positions; ball is on the centre spot
    PREPARE_PENALTY = auto()   # take positions for a penalty
    RUN = auto()               # normal play — the ball is live
    FREE_KICK = auto()         # a free kick is being taken
    BALL_PLACEMENT = auto()    # like STOP, but someone must move the ball
    TIMEOUT = auto()


@dataclass(frozen=True, slots=True)
class GameState:
    """Everything our AI needs to know about the referee, colour-neutral.

    `ours` answers "is this OUR kickoff / free kick / placement / penalty?"
    It is meaningless for HALT, STOP, and RUN.
    """

    play: Play = Play.HALT
    ours: bool = False

    # Derived permissions — precomputed so no caller has to reason about rules.
    can_move: bool = False
    can_touch_ball: bool = False
    min_ball_distance: float = 0.5      # meters; 0.0 when we may approach

    # Ball placement target, in our normalised frame. None unless placing.
    placement_target: tuple[float, float] | None = None

    our_score: int = 0
    their_score: int = 0
    our_goalkeeper: int = 0
    our_max_robots: int = 6
    our_yellow_cards: int = 0
    our_red_cards: int = 0

    # Seconds remaining on the current action (free-kick shot clock). None if n/a.
    action_time_remaining: float | None = None
    # Seconds left in the current stage. None if n/a.
    stage_time_left: float | None = None

    # Monotonic counter from the referee. CHANGES mean "a new command was
    # issued". Trigger transitions on this, never on `play` alone, or your
    # kickoff routine will re-fire 60 times a second.
    counter: int = 0

    @property
    def is_stopped(self) -> bool:
        return self.play in (Play.HALT, Play.STOP, Play.TIMEOUT)


HALT = GameState()
```

### 8.4 `src/tbots/core/state.py`

```python
"""The world model. Immutable, colour-neutral, canonical units."""

from __future__ import annotations

from dataclasses import dataclass, field

from tbots.core.gamestate import HALT, GameState


@dataclass(frozen=True, slots=True)
class RobotState:
    robot_id: int
    x: float
    y: float
    theta: float            # radians, (-pi, pi]
    vx: float = 0.0         # m/s, GLOBAL frame (not robot-local)
    vy: float = 0.0
    vtheta: float = 0.0     # rad/s
    has_ball: bool = False  # dribbler infrared, or inferred
    visible: bool = True    # False -> this is an extrapolation, trust it less

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class BallState:
    x: float
    y: float
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    visible: bool = True

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class WorldState:
    """One frame of the world.

    `us` and `them` — NEVER `blue` and `yellow`. The backend resolves colour
    and field side, so every consumer can assume we attack +x. See Rule 3.
    """

    t: float                                        # seconds
    ball: BallState
    us: dict[int, RobotState] = field(default_factory=dict)
    them: dict[int, RobotState] = field(default_factory=dict)
    game: GameState = HALT

    def closest_to_ball(self, robots: dict[int, RobotState]) -> RobotState | None:
        if not robots:
            return None
        bx, by = self.ball.x, self.ball.y
        return min(robots.values(), key=lambda r: (r.x - bx) ** 2 + (r.y - by) ** 2)
```

### 8.5 `src/tbots/core/command.py`

```python
"""What we send to a robot. Deliberately velocity-level, not wheel-level."""

from __future__ import annotations

from dataclasses import dataclass, replace

from tbots.core.geometry import MAX_KICK_SPEED


@dataclass(frozen=True, slots=True)
class RobotCommand:
    """A complete, absolute command for one robot for one control tick.

    ABSOLUTE, never a delta. UDP drops packets; a lost delta corrupts state
    forever, a lost absolute command is a non-event.

    Velocities are in the ROBOT'S LOCAL FRAME:
        vx  forward (out of the kicker)
        vy  left
        vtheta  counter-clockwise
    """

    robot_id: int
    vx: float = 0.0
    vy: float = 0.0
    vtheta: float = 0.0
    kick_speed: float = 0.0     # m/s. 0.0 = do not kick.
    chip: bool = False          # True = chip kick, False = flat kick
    dribbler: float = 0.0       # 0.0 .. 1.0

    def clamped(self, max_v: float = 3.0, max_w: float = 12.0) -> "RobotCommand":
        def clamp(v: float, lo: float, hi: float) -> float:
            return lo if v < lo else hi if v > hi else v

        return replace(
            self,
            vx=clamp(self.vx, -max_v, max_v),
            vy=clamp(self.vy, -max_v, max_v),
            vtheta=clamp(self.vtheta, -max_w, max_w),
            kick_speed=clamp(self.kick_speed, 0.0, MAX_KICK_SPEED),
            dribbler=clamp(self.dribbler, 0.0, 1.0),
        )


def stop(robot_id: int) -> RobotCommand:
    """The command every robot gets on HALT."""
    return RobotCommand(robot_id=robot_id)
```

**Verify:**

```bash
python - << 'PYEOF'
from tbots.core.state import WorldState, BallState, RobotState
from tbots.core.command import RobotCommand
from tbots.core.geometry import DIV_B, dist
from tbots.core.units import wrap_angle
import math

w = WorldState(t=0.0, ball=BallState(x=0.0, y=0.0),
               us={0: RobotState(0, -1.0, 0.0, 0.0)})
print("their goal:", DIV_B.their_goal)
print("dist to ball:", dist(w.us[0].pos, w.ball.pos))
print("wrap(3pi):", round(wrap_angle(3 * math.pi), 6))
print("clamped:", RobotCommand(0, vx=99.0).clamped())
print("core ok")
PYEOF
```

```bash
git add src/tbots/core && git commit -m "feat: core contracts"
```

---

## Step 9 — The backend layer

### 9.1 `src/tbots/backends/base.py`

```python
"""The interfaces every backend implements. Two of them, not one.

`Backend` is the MATCH contract: everything a real robot on a real field can
do. `SimBackend` adds the powers only a simulator has -- commanding the
opposition, teleporting things, and being told what the referee said.

Keeping them apart is not tidiness. A training environment requires a
`SimBackend`, so the type checker refuses a training run pointed at hardware,
and nothing in the match path can quietly grow a dependency on a power that
disappears the moment we unplug the simulator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from tbots.core.command import RobotCommand
from tbots.core.gamestate import GameState
from tbots.core.geometry import DIV_B, FieldGeometry
from tbots.core.state import WorldState

# (x, y, theta) -- meters, meters, radians, in our normalised frame.
Pose = tuple[float, float, float]
# (x, y, vx, vy) -- the ball carries velocity; robots do not. See place().
BallPlacement = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Scenario:
    """A reproducible starting configuration.

    Positions are in our normalised frame: (x, y, theta), meters and radians.
    Ball is (x, y, vx, vy).
    """

    ball: BallPlacement = (0.0, 0.0, 0.0, 0.0)
    us: tuple[Pose, ...] = ()
    them: tuple[Pose, ...] = ()
    seed: int | None = None

    @staticmethod
    def single_robot_at(x: float, y: float, theta: float = 0.0) -> Scenario:
        return Scenario(ball=(2.0, 0.0, 0.0, 0.0), us=((x, y, theta),), them=())

    @staticmethod
    def kickoff(n_us: int = 6, n_them: int = 6,
                geom: FieldGeometry = DIV_B) -> Scenario:
        us = tuple((-0.5 - 0.6 * i, (-1) ** i * 0.7 * (i // 2), 0.0)
                   for i in range(n_us))
        them = tuple((0.5 + 0.6 * i, (-1) ** i * 0.7 * (i // 2), 3.14159)
                     for i in range(n_them))
        return Scenario(ball=(0.0, 0.0, 0.0, 0.0), us=us, them=them)


@runtime_checkable
class Backend(Protocol):
    """Anything that can be stepped and observed. The match contract.

    Implementations MUST return WorldState in canonical units (meters,
    radians) and in our normalised frame (we are `us`, we attack +x).

    Everything here is something real robots on a real field can do. If you
    are about to add a method that only a simulator could implement, it
    belongs on `SimBackend`.
    """

    @property
    def dt(self) -> float:
        """Seconds of simulated time per step()."""
        ...

    @property
    def geometry(self) -> FieldGeometry: ...

    def reset(self, scenario: Scenario) -> WorldState: ...

    def step(self, commands: Sequence[RobotCommand]) -> WorldState: ...

    def close(self) -> None: ...


@runtime_checkable
class SimBackend(Backend, Protocol):
    """A Backend that is a simulator, and can therefore do three more things.

    rSim is one. So is ER-Force once simulation control is wired up. Real
    robots are only ever a `Backend`.
    """

    def step(self, commands: Sequence[RobotCommand],
             opponent_commands: Sequence[RobotCommand] = ()) -> WorldState:
        """Advance one tick.

        `commands` address `us`, `opponent_commands` address `them`, and the
        same `robot_id` means a different robot in each list. Opponents are
        optional; uncommanded ones stand still, which is what makes a scripted
        or frozen-checkpoint opponent something you opt into rather than
        something you have to supply every tick.

        A `robot_id` that is not on the team being addressed raises
        `ValueError`. It used to be dropped in silence, which meant a typo in
        an observation builder trained a policy against a robot that never
        received anything.
        """
        ...

    def place(self, ball: BallPlacement | None = None,
              us: Mapping[int, Pose] | None = None,
              them: Mapping[int, Pose] | None = None) -> WorldState:
        """Teleport whatever is named and leave everything else where it is.

        `place(ball=...)` moves only the ball; `place(us={2: pose})` moves only
        our robot 2. An unknown `robot_id` raises `ValueError`.

        Robot placements carry no velocity, and that is not an oversight: rSim
        cannot express one, so every robot on the field stops. The ball keeps
        the velocity you give it, minus whatever the backend's teleport costs.
        Read `RSimBackend.place` before using this in a loop -- it is cheap in
        wall-clock terms and not free in physics.
        """
        ...

    def set_game_state(self, game: GameState) -> None:
        """Tell the simulator what the referee would have said.

        A simulator has no referee. In training this comes from a
        `SyntheticReferee`; a match backend reads the real thing off the wire
        and has nothing to inject.
        """
        ...
```

### 9.2 `src/tbots/backends/rsim.py`

> **Before you write this file, open `docs/RSIM_FACTS.md` from Step 6** and substitute the four verified numbers into the constants at the top. The constants below are already the values verified against our fork (rSim commit `56d0253`) — if you are building against a different fork commit, re-run `scripts/verify_rsim.py` and reconfirm before trusting them. Two of these are NOT what the upstream READMEs imply: `field_type` for Division B is `1`, not `0`, and the action vector is `8` elements, not `6` — see `docs/RSIM_FACTS.md` for why a wrong action length does not raise an exception, which is what makes it dangerous.

```python
"""Training backend: rSim (ODE) running in-process.

Fast, deterministic, no sockets, no clock. This is what training uses.
It is NOT protocol-accurate — it emits no vision packets and knows nothing
about the referee. That is the network backend's job.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import robosim

from tbots.backends.base import BallPlacement, Pose, Scenario, SimBackend
from tbots.core.command import RobotCommand
from tbots.core.gamestate import HALT, GameState
from tbots.core.geometry import DIV_B, FieldGeometry
from tbots.core.state import BallState, RobotState, WorldState
from tbots.core.units import deg_to_rad, rad_to_deg, wrap_angle

# ---------------------------------------------------------------------------
# VERIFIED CONSTANTS — see docs/RSIM_FACTS.md. Do not guess these.
# Re-run scripts/verify_rsim.py after every rSim fork update.
# ---------------------------------------------------------------------------
FIELD_TYPE_DIV_B: int = 1        # verified: 0 is Division A, 1 is Division B
BALL_STRIDE: int = 5             # ball_x, ball_y, ball_z, ball_vx, ball_vy
ROBOT_STRIDE: int = 11           # x, y, angle, vx, vy, vangle, ir, w0..w3
ACTION_LEN: int = 8              # verified: all 8 slots are read. A shorter
                                  # vector is NOT rejected — it silently reads
                                  # past the end of an unchecked std::vector
                                  # and feeds garbage to the kicker/dribbler.
ANGLES_IN_DEGREES: bool = True   # verified. Governs STATE decode (heading,
                                  # vdir) and POSE encode (reset/ctor) only.
                                  # Does NOT govern commanded angular velocity
                                  # — see the note on A_VTHETA in _encode().

# Offsets within one robot's slice
R_X, R_Y, R_THETA, R_VX, R_VY, R_VTHETA, R_IR = 0, 1, 2, 3, 4, 5, 6

# Offsets within one robot's action vector
A_USE_WHEELS, A_VX, A_VY, A_VTHETA = 0, 1, 2, 3
A_WHEEL3 = 4                     # only read when A_USE_WHEELS > 0; unused here
A_KICK_FLAT, A_KICK_CHIP, A_DRIBBLER = 5, 6, 7


def _ang_in(v: float) -> float:
    return wrap_angle(deg_to_rad(v) if ANGLES_IN_DEGREES else v)


def _ang_out(v: float) -> float:
    return rad_to_deg(v) if ANGLES_IN_DEGREES else v


class RSimBackend(SimBackend):
    def __init__(
        self,
        n_us: int = 6,
        n_them: int = 6,
        dt: float = 1.0 / 60.0,
        geometry: FieldGeometry = DIV_B,
        field_type: int = FIELD_TYPE_DIV_B,
    ) -> None:
        self._n_us = n_us
        self._n_them = n_them
        self._dt = dt
        self._geom = geometry
        self._field_type = field_type
        self._t = 0.0
        self._game: GameState = HALT
        self._sim: robosim.SSL | None = None
        # The last WorldState we produced. place() reads the current
        # poses from here rather than from the simulator, because a
        # second get_state() would wreck the velocity differencing.
        self._last: WorldState | None = None
        self._expected_state_len = BALL_STRIDE + ROBOT_STRIDE * (n_us + n_them)

    # -- Backend protocol ---------------------------------------------------
    # (step() carries SimBackend's widened signature; the extra argument
    #  defaults, so this is still the match contract.)

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def geometry(self) -> FieldGeometry:
        return self._geom

    def reset(self, scenario: Scenario) -> WorldState:
        ball = list(scenario.ball)
        us = self._pad(scenario.us, self._n_us, default_x=-1.0)
        them = self._pad(scenario.them, self._n_them, default_x=1.0)

        if self._sim is None:
            # float(self._dt), not int milliseconds. The binding overloads on
            # the timestep's TYPE: an int is milliseconds, a float is seconds
            # (our fork, see docs/RSIM_FACTS.md). The old
            # `int(round(dt * 1000))` gave 17 for 60 Hz, so physics advanced
            # 17 ms per step while `self._t` advanced 1/60 s -- 2% of drift a
            # second, silent, and invisible in the state array because
            # velocities are divided by that same 17 ms. The float() is not
            # decoration: pass an int dt and you are back on the old path.
            self._sim = robosim.SSL(
                self._field_type, self._n_us, self._n_them,
                float(self._dt), ball, us, them,
            )
            raw = np.asarray(self._sim.get_state(), dtype=np.float64)
            if len(raw) != self._expected_state_len:
                raise RuntimeError(
                    f"rSim state length {len(raw)} != expected "
                    f"{self._expected_state_len}. Your BALL_STRIDE / "
                    f"ROBOT_STRIDE constants are wrong. "
                    f"Re-run scripts/verify_rsim.py."
                )
        else:
            self._sim.reset(ball, us, them)

        self._t = 0.0
        return self._observe()

    def step(self, commands: Sequence[RobotCommand],
             opponent_commands: Sequence[RobotCommand] = ()) -> WorldState:
        """Advance one tick. See `SimBackend.step`.

        `commands` are ours, `opponent_commands` are theirs, and `robot_id` 0
        in each list is a different robot. Opponents left uncommanded stand
        still: their action slots stay zero.
        """
        if self._sim is None:
            raise RuntimeError("call reset() before step()")
        self._sim.step(self._encode(commands, opponent_commands))
        self._t += self._dt
        return self._observe()

    def close(self) -> None:
        self._sim = None
        self._last = None

    # -- simulator-only powers ---------------------------------------------

    def place(self, ball: BallPlacement | None = None,
              us: Mapping[int, Pose] | None = None,
              them: Mapping[int, Pose] | None = None) -> WorldState:
        """Teleport what is named, leave the rest. See `SimBackend.place`.

        rSim has no partial teleport, so this is a full `reset()` with the
        CURRENT poses filled in for everything the caller did not name. Two
        consequences follow from that, and both are visible to callers:

        1. ROBOTS STOP. rSim's reset takes robot poses only -- `[x, y, dir]`,
           no velocity -- so every robot on the field ends the call
           stationary, including ones nobody placed. The ball keeps its
           velocity, because `ballPos` carries `(vx, vy)`. Nothing can be done
           about this short of a new rSim entry point; it is why `place()` is
           for restarts and episode setup, not for nudging things mid-play.

        2. THE BALL LOSES A FIXED ~0.09 m/s EVERY TIME, even if you did not
           place it. rSim's reset runs a 30-substep settling loop before
           handing the world back, and the ball rolls against friction
           throughout it. Measured across speeds: the loss is 0.091 m/s
           whether the ball was doing 1 m/s or 4, i.e. crippling at walking
           pace and negligible at kick speed. Position is unaffected, to
           within a micrometre. Do not call `place()` on a loop.

        3. The reported velocities are honest, but they were not free. A reset
           clears the baseline `get_state()` differences against, so the raw
           read after it is zero for everything (docs/RSIM_FACTS.md, trap 4).
           For the robots that IS the truth. For the ball it is not, so the
           velocity we just placed is written back over it -- otherwise the
           next `place()` reads that zero out of the cache and places a dead
           ball. Two `place()` calls in a row did exactly that until
           2026-09-07.

        Ball `z` and `vz` are dropped: rSim's `ballPos` is `[x, y, vx, vy]` and
        has nowhere to put them. A chipped ball lands the moment you place it.

        Simulated time does NOT rewind. `place()` happens during an episode;
        only `reset()` starts a new one.
        """
        if self._sim is None or self._last is None:
            raise RuntimeError("call reset() before place()")

        cur = self._last
        ball_pos = (list(ball) if ball is not None
                    else [cur.ball.x, cur.ball.y, cur.ball.vx, cur.ball.vy])
        self._sim.reset(
            ball_pos,
            self._poses(cur.us, self._n_us, us, "ours"),
            self._poses(cur.them, self._n_them, them, "theirs"),
        )
        world = self._observe()
        # Put the ball's velocity back. See consequence 3 above: the raw read
        # after a reset is zero for everything, which is true of the robots and
        # false of the ball, and leaving it false is what let a second place()
        # stop the ball dead.
        self._last = replace(world,
                             ball=replace(world.ball,
                                          vx=ball_pos[2], vy=ball_pos[3]))
        return self._last

    def set_game_state(self, game: GameState) -> None:
        """Injected by the environment or a SyntheticReferee. rSim itself
        has no concept of a referee."""
        self._game = game

    # -- fixed for the whole run ---------------------------------------------
    # rSim fixes the robot count at construction, and so do we: `reconfigure()`
    # is gone (TASK-072). A curriculum is a sequence of RUNS, not a simulator
    # that changes shape underneath a running policy, and TASK-041/TASK-055
    # both assume the count cannot move.

    @property
    def n_us(self) -> int:
        return self._n_us

    @property
    def n_them(self) -> int:
        return self._n_them

    # -- internals ----------------------------------------------------------

    def _pad(self, poses, n: int, default_x: float):
        out = [[p[0], p[1], _ang_out(p[2])] for p in poses[:n]]
        while len(out) < n:
            i = len(out)
            out.append([default_x * (1.0 + 0.3 * i), -2.5, 0.0])
        return out

    def _checked_id(self, robot_id: int, n: int, side: str) -> int:
        """`robot_id` back, or ValueError if that robot is not on `side`.

        Not a silent skip. An out-of-range id used to be dropped without a
        word, so a policy could spend a whole run commanding a robot that did
        not exist and look merely bad at football.
        """
        if not (0 <= robot_id < n):
            have = f"ids are 0..{n - 1}" if n else "there are none on that side"
            raise ValueError(f"robot {robot_id} is not one of {side}: {have}")
        return robot_id

    def _poses(self, current: dict[int, RobotState], n: int,
               overrides: Mapping[int, Pose] | None,
               side: str) -> list[list[float]]:
        """One team's poses for a reset: `overrides` where given, else current."""
        for robot_id in overrides or ():
            self._checked_id(robot_id, n, side)
        out = []
        for i in range(n):
            if overrides is not None and i in overrides:
                x, y, theta = overrides[i]
            else:
                r = current[i]
                x, y, theta = r.x, r.y, r.theta
            out.append([x, y, _ang_out(theta)])
        return out

    def _encode(self, commands: Sequence[RobotCommand],
                opponent_commands: Sequence[RobotCommand]) -> list[list[float]]:
        # rSim's action list is blue-then-yellow, so our robot i is slot i and
        # their robot i is slot n_us + i. Everything starts at zero, which is
        # how an uncommanded robot stands still.
        n = self._n_us + self._n_them
        acts = [[0.0] * ACTION_LEN for _ in range(n)]
        pairs = [(c, self._checked_id(c.robot_id, self._n_us, "ours"))
                 for c in commands]
        pairs += [(c, self._n_us + self._checked_id(c.robot_id, self._n_them, "theirs"))
                  for c in opponent_commands]
        for c, slot in pairs:
            a = acts[slot]
            a[A_USE_WHEELS] = 0.0          # 0 = interpret as body velocities
            a[A_VX] = c.vx
            a[A_VY] = c.vy
            # c.vtheta is already radians/s (core units), and the action slot
            # wants radians/s too — pass through with NO conversion. This is
            # the one asymmetric spot in this file: everything coming OUT of
            # rSim's state is degrees and goes through _ang_in(); this value
            # going IN does not go through _ang_out(). Running it through
            # _ang_out() here is the single easiest mistake to make.
            a[A_VTHETA] = c.vtheta
            a[A_KICK_FLAT] = 0.0 if c.chip else c.kick_speed
            a[A_KICK_CHIP] = c.kick_speed if c.chip else 0.0
            a[A_DRIBBLER] = c.dribbler
        return acts

    def _observe(self) -> WorldState:
        # Call get_state() EXACTLY ONCE per reset()/step() and nowhere else.
        # rSim finite-differences velocities against whatever state it
        # captured at the PREVIOUS get_state() call, divided by a fixed
        # timeStep — never by time actually elapsed. Two calls with no
        # step() between them read back zero velocity; skipping a call
        # makes the next one read back a multiple too large. Do not add an
        # extra get_state() for logging or rendering.
        assert self._sim is not None
        raw = np.asarray(self._sim.get_state(), dtype=np.float64)

        ball = BallState(
            x=float(raw[0]), y=float(raw[1]), z=float(raw[2]),
            vx=float(raw[3]), vy=float(raw[4]), vz=0.0, visible=True,
        )

        us: dict[int, RobotState] = {}
        them: dict[int, RobotState] = {}
        for i in range(self._n_us + self._n_them):
            base = BALL_STRIDE + i * ROBOT_STRIDE
            s = raw[base:base + ROBOT_STRIDE]
            r = RobotState(
                robot_id=i if i < self._n_us else i - self._n_us,
                x=float(s[R_X]), y=float(s[R_Y]),
                theta=_ang_in(float(s[R_THETA])),
                vx=float(s[R_VX]), vy=float(s[R_VY]),
                vtheta=(deg_to_rad(float(s[R_VTHETA]))
                        if ANGLES_IN_DEGREES else float(s[R_VTHETA])),
                has_ball=bool(s[R_IR] > 0.5),
                visible=True,
            )
            (us if i < self._n_us else them)[r.robot_id] = r

        self._last = WorldState(t=self._t, ball=ball, us=us, them=them,
                                game=self._game)
        return self._last
```

> **Note on the `us`/`them` flip:** rSim always calls the first `n_us` robots "blue". Because we *choose* to be blue in training, no flip is needed here. The flip lives in the **network** backend, where the game controller tells us our real colour and field side. Keep it that way — training should never care about colour.

### 9.2a Variable robot counts and curriculum learning

> **Superseded 2026-09-07 (TASK-072) — `reconfigure()` no longer exists.**
> This section originally had the training loop call
> `backend.reconfigure(n_us, n_them)` to change stages mid-run. The
> architecture review deleted that: **a curriculum is a sequence of RUNS**,
> each with a fixed robot count, promoted by a human. A simulator that changes
> shape underneath a running policy invalidates the rollout buffer, the
> observation size and the checkpoint metadata all at once, and
> `rl/vec.py` (TASK-055) cannot do it at all across subprocess workers.
> Robot counts are now fixed for the life of a backend. The two design
> consequences below survive the change unaltered, and are the reason the
> section is kept.

**Arbitrary counts still work — at construction.** rSim takes
`n_robots_blue` and `n_robots_yellow` as constructor arguments, so 1v0, 1v1,
2v2, 3v3, and 6v6 are all valid: `RSimBackend(n_us=2, n_them=2)`. What changed
is only that you pick the count once, per run.

Two design consequences you must handle up front, because retrofitting them is
painful:

**(a) The observation vector must be a fixed size across the whole curriculum.** If your 2v2 observation is 18 floats and your 6v6 observation is 54, the policy cannot transfer between stages and the curriculum is pointless. Two ways out:

| Approach | How | Use for |
|---|---|---|
| **Pad and mask** | Always emit slots for `max_robots`; fill absent robots with zeros and append a validity bit per slot | Skill envs — simple, one robot matters anyway |
| **Set encoder** | Permutation-invariant DeepSets or attention pool over a variable-length robot set | Tactics — handles any N natively, and is the right architecture regardless |

The tactics layer wants the set encoder anyway (a flat MLP over concatenated positions has to relearn "opponent near ball is dangerous" separately for every slot index), so building it means the curriculum comes free.

**(b) The field size stays constant.** Do *not* shrink the pitch for 2v2. Keeping Division B geometry across every stage is what makes distances, angles, and goal positions mean the same thing at every difficulty — which is the entire mechanism by which the earlier stage teaches something useful about the later one.

A curriculum is then a schedule of runs, each launched with its own
`n_us`/`n_them` and starting from the previous stage's checkpoint.

### 9.3 `src/tbots/backends/network.py` (skeleton)

> **Superseded 2026-08-26 — Rule 3 now has a module.** The code block below is
> the original transcription and is kept for the record. The live files differ:
> colour and field-side resolution moved into `src/tbots/core/perspective.py`
> and `net.referee.resolve_perspective()`, so it is derived once instead of
> in each module that touches the wire. If you are rebuilding from this guide,
> take `core/perspective.py` from the repo rather than re-creating the split
> shown here. See `docs/ARCHITECTURE.md` §5 (Rule 3) and
> `tests/test_perspective.py`.

```python
"""Match backend: the ER-Force simulator, or real robots.

Commands go out over the SSL simulation protocol (or the radio).
Observations come in over SSL-Vision multicast.
Referee state comes in over the game-controller multicast.

This backend is realtime and lossy. It is for evaluation and match play,
never for training.
"""

from __future__ import annotations

from typing import Sequence

from tbots.backends.base import Backend, Scenario
from tbots.core.command import RobotCommand
from tbots.core.geometry import DIV_B, FieldGeometry
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
    ) -> None:
        self._team_name = team_name
        self._vision = vision
        self._referee = referee
        self._control = control
        self._sim_control = sim_control
        self._geom = geometry
        self._dt = dt
        self._tracker = Tracker(geometry)

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
        self._sim_control.place(scenario, self._flip_state())
        return self.observe()

    def step(self, commands: Sequence[RobotCommand]) -> WorldState:
        self._control.send(commands)
        self._vision.wait_for_next_frame(timeout=0.05)
        return self.observe()

    def observe(self) -> WorldState:
        # TODO(setup): merge detection frames, run the tracker, attach
        # GameState, apply the colour/side flip.
        raise NotImplementedError("TASK-014")

    def _flip_state(self):
        # TODO(setup): derive (we_are_yellow, we_defend_positive_x) from the
        # referee message and cache it. Recompute at every stage change,
        # because sides swap at half time.
        raise NotImplementedError("TASK-013")

    def close(self) -> None:
        self._vision.close()
        self._referee.close()
        self._control.close()
```

```bash
git add src/tbots/backends && git commit -m "feat: backend interface and rSim backend"
```

---

## Step 10 — The network layer

Nothing here is provided by a library. The league ships `.proto` files and Go reference clients; **there is no Python client library**. We write roughly 500 lines of socket code, once, and then never think about it again.

### 10.1 `src/tbots/net/multicast.py`

```python
"""UDP multicast helpers.

Two things every new person needs to know:

1. League multicast packets are BARE PROTOBUF. One message per datagram,
   no length prefix. Just ParseFromString(data).
   The TCP interfaces (game-controller team client on 10008, CI on 10009)
   are length-delimited streams instead — different framing entirely.

2. In development, ALWAYS set multicast TTL to 0. TTL 0 means the packet
   never leaves this host. Without it, two people on the same lab wifi will
   silently referee each other's matches, and you will lose an afternoon.
"""

from __future__ import annotations

import socket
import struct


def rx_socket(group: str, port: int, iface: str = "0.0.0.0",
              blocking: bool = False) -> socket.socket:
    """Join a multicast group and return a socket ready to recvfrom()."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except OSError:
            pass
    s.bind(("", port))
    mreq = struct.pack("4s4s", socket.inet_aton(group), socket.inet_aton(iface))
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    s.setblocking(blocking)
    return s


def tx_socket(ttl: int = 0, iface: str = "0.0.0.0") -> socket.socket:
    """Socket for sending multicast. ttl=0 keeps packets on this host."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                 socket.inet_aton(iface))
    return s


def drain(sock: socket.socket, bufsize: int = 65535) -> list[bytes]:
    """Read every pending datagram without blocking. Newest is last.

    Call this once per control tick. Never block the control loop on a
    socket: one dropped packet would stall all six robots.
    """
    out: list[bytes] = []
    while True:
        try:
            out.append(sock.recv(bufsize))
        except (BlockingIOError, InterruptedError):
            return out
        except OSError:
            return out
```

### 10.2 `src/tbots/net/referee.py`

> **Superseded 2026-08-26 — Rule 3 now has a module.** The code block below is
> the original transcription and is kept for the record. The live files differ:
> colour and field-side resolution moved into `src/tbots/core/perspective.py`
> and `net.referee.resolve_perspective()`, so it is derived once instead of
> in each module that touches the wire. If you are rebuilding from this guide,
> take `core/perspective.py` from the repo rather than re-creating the split
> shown here. See `docs/ARCHITECTURE.md` §5 (Rule 3) and
> `tests/test_perspective.py`.

```python
"""Receive Referee messages from the game controller and normalise them."""

from __future__ import annotations

import socket

from tbots._pb.state.ssl_gc_referee_message_pb2 import Referee
from tbots.core.gamestate import HALT, GameState, Play
from tbots.core.units import mm_to_m
from tbots.net.multicast import drain, rx_socket

_C = Referee.Command

# Which commands belong to which team, and which Play they map to.
_YELLOW_CMDS = {
    _C.PREPARE_KICKOFF_YELLOW, _C.PREPARE_PENALTY_YELLOW,
    _C.DIRECT_FREE_YELLOW, _C.INDIRECT_FREE_YELLOW,
    _C.TIMEOUT_YELLOW, _C.BALL_PLACEMENT_YELLOW,
}
_PLAY_MAP = {
    _C.HALT: Play.HALT,
    _C.STOP: Play.STOP,
    _C.NORMAL_START: Play.RUN,
    _C.FORCE_START: Play.RUN,
    _C.PREPARE_KICKOFF_YELLOW: Play.PREPARE_KICKOFF,
    _C.PREPARE_KICKOFF_BLUE: Play.PREPARE_KICKOFF,
    _C.PREPARE_PENALTY_YELLOW: Play.PREPARE_PENALTY,
    _C.PREPARE_PENALTY_BLUE: Play.PREPARE_PENALTY,
    _C.DIRECT_FREE_YELLOW: Play.FREE_KICK,
    _C.DIRECT_FREE_BLUE: Play.FREE_KICK,
    _C.INDIRECT_FREE_YELLOW: Play.FREE_KICK,
    _C.INDIRECT_FREE_BLUE: Play.FREE_KICK,
    _C.TIMEOUT_YELLOW: Play.TIMEOUT,
    _C.TIMEOUT_BLUE: Play.TIMEOUT,
    _C.BALL_PLACEMENT_YELLOW: Play.BALL_PLACEMENT,
    _C.BALL_PLACEMENT_BLUE: Play.BALL_PLACEMENT,
}


def to_gamestate(msg: Referee, we_are_yellow: bool,
                 flip_x: bool = False) -> GameState:
    play = _PLAY_MAP.get(msg.command, Play.HALT)
    cmd_is_yellow = msg.command in _YELLOW_CMDS
    ours = (cmd_is_yellow == we_are_yellow)

    mine = msg.yellow if we_are_yellow else msg.blue
    theirs = msg.blue if we_are_yellow else msg.yellow

    target = None
    if msg.HasField("designated_position"):
        tx = mm_to_m(msg.designated_position.x)
        ty = mm_to_m(msg.designated_position.y)
        target = (-tx if flip_x else tx, -ty if flip_x else ty)

    can_move = play is not Play.HALT
    can_touch = play is Play.RUN or (play is Play.BALL_PLACEMENT and ours)
    min_dist = 0.0 if can_touch else 0.5

    return GameState(
        play=play,
        ours=ours,
        can_move=can_move,
        can_touch_ball=can_touch,
        min_ball_distance=min_dist,
        placement_target=target,
        our_score=mine.score,
        their_score=theirs.score,
        our_goalkeeper=mine.goalkeeper,
        our_max_robots=(mine.max_allowed_bots
                        if mine.HasField("max_allowed_bots") else 6),
        our_yellow_cards=mine.yellow_cards,
        our_red_cards=mine.red_cards,
        action_time_remaining=(msg.current_action_time_remaining / 1e6
                               if msg.HasField("current_action_time_remaining")
                               else None),
        stage_time_left=(msg.stage_time_left / 1e6
                         if msg.HasField("stage_time_left") else None),
        counter=msg.command_counter,
    )


class RefereeReceiver:
    """Non-blocking latched receiver.

    The referee is an EVENT STREAM, not a clock. It only changes when an
    operator or an autoRef acts. Poll it every tick and use the latched
    value; never wait for a packet.
    """

    def __init__(self, group: str = "224.5.23.1", port: int = 10003,
                 team_name: str = "TritonBots") -> None:
        self._sock = rx_socket(group, port)
        self._team_name = team_name
        self._latest: GameState = HALT
        self._raw: Referee | None = None
        self._we_are_yellow: bool | None = None
        self._flip_x: bool = False

    @property
    def we_are_yellow(self) -> bool | None:
        return self._we_are_yellow

    @property
    def flip_x(self) -> bool:
        """True if we must negate x to keep attacking +x."""
        return self._flip_x

    def poll(self) -> GameState:
        for data in drain(self._sock):
            msg = Referee()
            msg.ParseFromString(data)          # bare protobuf, no framing
            self._raw = msg
            self._resolve_sides(msg)
            if self._we_are_yellow is not None:
                self._latest = to_gamestate(msg, self._we_are_yellow,
                                            self._flip_x)
        return self._latest

    def _resolve_sides(self, msg: Referee) -> None:
        # Colour: match our configured name against the referee's team names.
        # The name is CASE-SENSITIVE and must match exactly, spaces included.
        if msg.yellow.name == self._team_name:
            self._we_are_yellow = True
        elif msg.blue.name == self._team_name:
            self._we_are_yellow = False

        if self._we_are_yellow is None:
            return

        # Side: blue_team_on_positive_half tells us who defends +x.
        # We must ATTACK +x, so we flip when we DEFEND +x.
        if msg.HasField("blue_team_on_positive_half"):
            blue_pos = msg.blue_team_on_positive_half
            we_defend_positive = (blue_pos != self._we_are_yellow)
            self._flip_x = we_defend_positive

    def close(self) -> None:
        self._sock.close()
```

> **Field-name check.** `max_allowed_bots`, `current_action_time_remaining`, and `blue_team_on_positive_half` exist in the current game-controller proto but not in the archived refbox one. If any of these raise `ValueError: Unknown field`, open `protos/ssl-game-controller/proto/ssl_gc_referee_message.proto` and read the actual field names — that file is always the truth.

### 10.3 `src/tbots/net/vision_publisher.py` — build this early, it pays for itself

> **Superseded 2026-08-26 — Rule 3 now has a module.** The code block below is
> the original transcription and is kept for the record. The live files differ:
> colour and field-side resolution moved into `src/tbots/core/perspective.py`
> and `net.referee.resolve_perspective()`, so it is derived once instead of
> in each module that touches the wire. If you are rebuilding from this guide,
> take `core/perspective.py` from the repo rather than re-creating the split
> shown here. See `docs/ARCHITECTURE.md` §5 (Rule 3) and
> `tests/test_perspective.py`.

```python
"""Turn ANY WorldState into SSL-Vision packets.

This is the single highest-value 80 lines in the repo. Once it exists,
ssl-vision-client renders BOTH backends — you watch an rSim training
rollout and a live networked match in the same browser tab, same tool.

It also means our vision serialisation is exercised constantly, so it won't
be broken the first time we actually need it.
"""

from __future__ import annotations

import time

from tbots._pb.messages_robocup_ssl_wrapper_pb2 import SSL_WrapperPacket
from tbots.core.geometry import DIV_B, FieldGeometry
from tbots.core.state import WorldState
from tbots.core.units import m_to_mm
from tbots.net.multicast import tx_socket


class VisionPublisher:
    def __init__(self, group: str = "224.5.23.2", port: int = 10006,
                 ttl: int = 0, geometry: FieldGeometry = DIV_B,
                 we_are_yellow: bool = False) -> None:
        self._sock = tx_socket(ttl=ttl)
        self._addr = (group, port)
        self._geom = geometry
        self._yellow = we_are_yellow
        self._frame = 0

    def publish(self, world: WorldState, t_capture: float | None = None) -> None:
        pkt = SSL_WrapperPacket()
        d = pkt.detection
        d.frame_number = self._frame
        self._frame += 1
        d.t_capture = t_capture if t_capture is not None else time.time()
        d.t_sent = time.time()
        d.camera_id = 0

        b = d.balls.add()
        b.confidence = 1.0
        b.x = m_to_mm(world.ball.x)
        b.y = m_to_mm(world.ball.y)
        b.z = m_to_mm(world.ball.z)
        b.pixel_x = 0.0
        b.pixel_y = 0.0

        ours = d.robots_yellow if self._yellow else d.robots_blue
        theirs = d.robots_blue if self._yellow else d.robots_yellow

        for group, robots in ((ours, world.us), (theirs, world.them)):
            for r in robots.values():
                m = group.add()
                m.confidence = 1.0
                m.robot_id = r.robot_id
                m.x = m_to_mm(r.x)          # SSL-Vision is MILLIMETERS
                m.y = m_to_mm(r.y)
                m.orientation = r.theta      # ...but RADIANS
                m.pixel_x = 0.0
                m.pixel_y = 0.0

        self._sock.sendto(pkt.SerializeToString(), self._addr)

    def publish_geometry(self) -> None:
        """Send field dimensions so the client draws the right pitch.
        Call once at startup and every few seconds thereafter."""
        pkt = SSL_WrapperPacket()
        f = pkt.geometry.field
        f.field_length = int(m_to_mm(self._geom.length))
        f.field_width = int(m_to_mm(self._geom.width))
        f.goal_width = int(m_to_mm(self._geom.goal_width))
        f.goal_depth = int(m_to_mm(self._geom.goal_depth))
        f.boundary_width = int(m_to_mm(self._geom.boundary_width))
        self._sock.sendto(pkt.SerializeToString(), self._addr)

    def close(self) -> None:
        self._sock.close()
```

### 10.4 The remaining network files (skeletons)

Create these four with the given signatures and `NotImplementedError` bodies. They are tracked on [the task board](#the-task-board).

**`src/tbots/net/vision.py`** — `VisionReceiver`: joins `224.5.23.2:10006`, parses `SSL_WrapperPacket`, buffers detection frames per `camera_id`, and merges them into one world frame by `t_capture`. Division B uses four cameras at roughly 60 Hz each, so you will receive ~240 packets/second, staggered. **Do not run the control loop once per packet** — merge, then tick once.

**`src/tbots/net/robot_control.py`** — `RobotControlSender`: builds one `RobotControl` containing all six `RobotCommand`s and sends it to UDP 10301 (blue) or 10302 (yellow). One packet per tick, containing every robot — never one packet per robot. Also reads `RobotControlResponse` for dribbler-contact feedback and simulator errors.

**`src/tbots/net/sim_control.py`** — `SimControlSender`: sends `SimulatorCommand` to UDP 10300 to teleport the ball and robots. Used only for episode resets against a simulator. Note that at a tournament this port is normally locked down, so nothing in the match path may depend on it.

**`src/tbots/net/team_client.py`** — `TeamClient`: TCP 10008, length-delimited framing, optional RSA request signing. Used for goalkeeper changes, substitution intent, and the advantage choice. Build this **last**; we can play a full match without it. When you do build it: keys live in the GC's `config/trusted_keys/team/<teamName>.pub.pem`, the GC's `genKey.sh` generates the pair, the controller returns a token that must be echoed in the next request to prevent replays, and each team may connect only once.

```bash
git add src/tbots/net && git commit -m "feat: network layer"
```

### 10.5 `src/tbots/perception/tracker.py` (skeleton)

```python
"""Fuse multi-camera detections into one clean world model.

Responsibilities:
  - merge detections of the same robot seen by overlapping cameras
  - associate detections across frames; keep IDs stable
  - estimate velocities (SSL-Vision reports position only)
  - extrapolate through dropouts and mark those states visible=False
  - forward-predict to compensate latency

LATENCY IS THE POINT. By the time a vision frame reaches us it is already
20-40 ms old, and our command takes another 10-20 ms to reach the robot.
At 3 m/s that is ~15 cm of error. rSim hands us perfect instantaneous
state, so a policy trained without simulated latency will depend on
information it can never have at a match.
"""

from __future__ import annotations

from tbots.core.geometry import FieldGeometry
from tbots.core.state import WorldState


class Tracker:
    def __init__(self, geometry: FieldGeometry) -> None:
        self._geom = geometry

    def update(self, frames: list, t_now: float) -> WorldState:
        raise NotImplementedError("TASK-020")
```

---

## Step 11 — Skills, tactics, and RL placeholders

This is the scaffolding recruits will fill in. Every file here is either a working reference implementation or a clearly-marked stub. **Nothing here contains a reward function or a tactic** — those are the work, not the setup.

### 11.1 `src/tbots/skills/base.py`

```python
"""A Skill is a closed-loop behaviour for ONE robot over MANY control ticks.

The interface is identical whether the implementation is forty lines of
geometry or a neural network. That is what lets someone train a policy on
Tuesday and drop it into the match stack on Wednesday by editing a config.
"""

from __future__ import annotations

from typing import Callable, Literal, Protocol, runtime_checkable

from tbots.core.command import RobotCommand
from tbots.core.state import WorldState

SkillStatus = Literal["running", "success", "failure"]


@runtime_checkable
class Skill(Protocol):
    def reset(self, world: WorldState, robot_id: int) -> None: ...

    def step(self, world: WorldState, robot_id: int) -> RobotCommand: ...

    def status(self) -> SkillStatus: ...


_REGISTRY: dict[str, Callable[..., Skill]] = {}


def register_skill(name: str):
    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"skill '{name}' already registered")
        _REGISTRY[name] = cls
        return cls
    return deco


def build_skill(name: str, **kwargs) -> Skill:
    if name not in _REGISTRY:
        raise KeyError(f"unknown skill '{name}'. known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def skill_names() -> list[str]:
    return sorted(_REGISTRY)
```

### 11.2 `src/tbots/skills/go_to_point.py` — the reference implementation

Read this before writing any other skill. It is deliberately simple, deliberately classical, and deliberately **not** learned.

```python
"""Drive to a pose. Classical, not learned — and that is on purpose.

A well-tuned trapezoidal velocity profile beats six months of PPO at this
task, transfers to real hardware for free, and can be debugged with a
print statement. RL earns its keep where physics is hard to model
(dribbler contact, interception under uncertainty), not on rigid-body
motion across a flat floor.
"""

from __future__ import annotations

import math

from tbots.core.command import RobotCommand
from tbots.core.state import WorldState
from tbots.core.units import angle_diff
from tbots.skills.base import Skill, SkillStatus, register_skill


@register_skill("go_to_point")
class GoToPoint(Skill):
    def __init__(
        self,
        target: tuple[float, float],
        face: float | None = None,
        pos_tol: float = 0.04,
        ang_tol: float = 0.08,
        max_v: float = 2.5,
        max_w: float = 8.0,
        kp_pos: float = 3.0,
        kp_ang: float = 4.0,
    ) -> None:
        self.target = target
        self.face = face
        self.pos_tol = pos_tol
        self.ang_tol = ang_tol
        self.max_v = max_v
        self.max_w = max_w
        self.kp_pos = kp_pos
        self.kp_ang = kp_ang
        self._status: SkillStatus = "running"

    def reset(self, world: WorldState, robot_id: int) -> None:
        self._status = "running"

    def step(self, world: WorldState, robot_id: int) -> RobotCommand:
        me = world.us.get(robot_id)
        if me is None:
            self._status = "failure"
            return RobotCommand(robot_id=robot_id)

        ex = self.target[0] - me.x
        ey = self.target[1] - me.y
        err = math.hypot(ex, ey)

        desired_theta = self.face if self.face is not None else me.theta
        eang = angle_diff(desired_theta, me.theta)

        if err < self.pos_tol and abs(eang) < self.ang_tol:
            self._status = "success"
            return RobotCommand(robot_id=robot_id)

        # Global-frame P control, then rotate into the robot's local frame.
        speed = min(self.kp_pos * err, self.max_v)
        gx = speed * ex / max(err, 1e-6)
        gy = speed * ey / max(err, 1e-6)

        c, s = math.cos(-me.theta), math.sin(-me.theta)
        vx = c * gx - s * gy
        vy = s * gx + c * gy

        return RobotCommand(
            robot_id=robot_id, vx=vx, vy=vy,
            vtheta=max(-self.max_w, min(self.max_w, self.kp_ang * eang)),
        ).clamped(max_v=self.max_v, max_w=self.max_w)

    def status(self) -> SkillStatus:
        return self._status
```

> **Upgrade path, not a rewrite.** This P controller is good enough to build everything else on top of. When someone replaces it with a proper time-optimal trapezoidal profile plus acceleration limits, they change *this file only* — every caller and every trained policy is unaffected. That is the interface doing its job.

### 11.3 Skill stubs

Create these files. Each contains a docstring describing the contract and `raise NotImplementedError("TASK-0xx")`.

| File | Class | Learned? | Task |
|---|---|---|---|
| `skills/face_point.py` | `FacePoint` | no | TASK-030 |
| `skills/shoot.py` | `Shoot` | partly | TASK-031 |
| `skills/pass_to.py` | `PassTo` | partly | TASK-032 |
| `skills/receive_pass.py` | `ReceivePass` | **yes** | TASK-033 |
| `skills/dribble.py` | `Dribble` | **yes** | TASK-034 |
| `skills/intercept.py` | `Intercept` | **yes** | TASK-035 |
| `skills/goalkeep.py` | `Goalkeep` | **yes** | TASK-036 |
| `skills/learned.py` | `LearnedSkill` | wrapper | TASK-037 |

`LearnedSkill` is the load-bearing one. Its job:

```python
"""Wrap any trained checkpoint as a Skill.

    skill = LearnedSkill(checkpoint="checkpoints/dribble_v7.pt",
                         obs_builder="egocentric_ball")

Loads a TorchScript module, builds an observation from WorldState via a
named observation builder, runs a forward pass, decodes the output into a
RobotCommand. Must run on CPU in under ~1 ms.

Keep policies small — a few hundred thousand parameters. The virtual
tournament runs team software in a container without root, and a GPU has
to be requested from the technical committee in advance. Assume CPU.
"""
```

### 11.4 `src/tbots/tactics/base.py`

```python
"""Tactics: assign skills to robots. This is where our RL bet lives.

The tactics policy does NOT emit velocities. It emits skill assignments,
once every ~200-500 ms. That decomposition is what makes the learning
problem tractable:

    at 60 Hz, a two-minute episode is ~7,200 control ticks per robot
    at the tactics level, it is ~240 decisions

A ~30x shorter horizon, and horizon is what kills credit assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from tbots.core.state import WorldState


@dataclass(frozen=True, slots=True)
class SkillSpec:
    """A skill name plus its constructor kwargs. Serialisable on purpose."""

    name: str
    kwargs: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Assignment:
    robot_id: int
    skill: SkillSpec


@runtime_checkable
class Tactic(Protocol):
    def decide(self, world: WorldState) -> list[Assignment]: ...
```

### 11.5 Tactics stubs

| File | Purpose | Task |
|---|---|---|
| `tactics/roles.py` | Role assignment (keeper / defender / attacker), Hungarian matching | TASK-040 |
| `tactics/scripted.py` | A hand-written baseline. **Build this first** — you cannot evaluate a learned tactic without an opponent. | TASK-041 |
| `tactics/learned.py` | Wraps a trained tactics policy as a `Tactic` | TASK-042 |
| `tactics/restarts.py` | Kickoff, free kick, penalty, ball placement routines | TASK-043 |

> `tactics/restarts.py` is unglamorous and mandatory. Roughly half of match time is spent in stoppages. A team that handles restarts correctly and plays mediocre open-field soccer beats a team that does the reverse.

### 11.6 `src/tbots/rl/rewards/registry.py`

```python
"""Composable reward terms. This is the file recruits touch first.

A reward function is a weighted sum of registered terms, configured in
YAML. Nobody edits an environment to change a reward.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from tbots.core.state import WorldState


@runtime_checkable
class RewardTerm(Protocol):
    def __call__(self, world: WorldState, prev: WorldState) -> float: ...

    def reset(self) -> None: ...


_REGISTRY: dict[str, Callable[..., RewardTerm]] = {}


def register_reward(name: str):
    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"reward term '{name}' already registered")
        _REGISTRY[name] = cls
        return cls
    return deco


def reward_names() -> list[str]:
    return sorted(_REGISTRY)


class CompositeReward:
    """Weighted sum of named terms. Also records per-term contributions so
    you can see WHICH term is driving a policy, which is the difference
    between debugging a reward in an hour and debugging it in a week."""

    def __init__(self, spec: list[dict]) -> None:
        self.terms: list[tuple[str, float, RewardTerm]] = []
        for item in spec:
            name = item["name"]
            weight = float(item.get("weight", 1.0))
            kwargs = {k: v for k, v in item.items() if k not in ("name", "weight")}
            if name not in _REGISTRY:
                raise KeyError(f"unknown reward term '{name}'. "
                               f"known: {reward_names()}")
            self.terms.append((name, weight, _REGISTRY[name](**kwargs)))
        self.last: dict[str, float] = {}

    def reset(self) -> None:
        self.last = {}
        for _, _, term in self.terms:
            term.reset()

    def __call__(self, world: WorldState, prev: WorldState) -> float:
        total = 0.0
        for name, weight, term in self.terms:
            v = weight * term(world, prev)
            self.last[name] = v
            total += v
        return total
```

Also create `src/tbots/rl/rewards/example.py` with **one** trivial term, purely so the registry has something to load and the smoke test passes:

```python
"""One trivial term so the machinery has something to load.

This is NOT a template for good reward design — it is a syntax example.
"""

from tbots.core.state import WorldState
from tbots.rl.rewards.registry import register_reward


@register_reward("alive")
class Alive:
    """Returns 1.0 every step. Useless for learning; useful for testing."""

    def reset(self) -> None:
        pass

    def __call__(self, world: WorldState, prev: WorldState) -> float:
        return 1.0
```

Make sure terms get imported so they self-register — put this in `src/tbots/rl/rewards/__init__.py`:

```python
from tbots.rl.rewards.registry import (  # noqa: F401
    CompositeReward, RewardTerm, register_reward, reward_names,
)
from tbots.rl.rewards import example  # noqa: F401  (self-registers "alive")
```

### 11.7 `src/tbots/rl/envs/base.py`

```python
"""Our Gymnasium environment, built on the Backend interface.

We do NOT subclass rSoccer's SSLBaseEnv. That class hardcodes coordinate
conventions, referee handling, and opponent behaviour that we need to
control ourselves. We use rSoccer for its renderer and as a reference.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from tbots.backends.base import Backend, Scenario
from tbots.core.command import RobotCommand
from tbots.core.state import WorldState
from tbots.rl.rewards.registry import CompositeReward


class SSLEnv(gym.Env):
    """Base class. Subclasses define obs/action spaces and decode actions."""

    metadata = {"render_modes": ["human", "vision"], "render_fps": 60}

    def __init__(
        self,
        backend: Backend,
        reward: CompositeReward,
        scenario_fn,
        max_episode_steps: int = 3600,
        render_mode: str | None = None,
    ) -> None:
        self.backend = backend
        self.reward = reward
        self.scenario_fn = scenario_fn
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._steps = 0
        self._world: WorldState | None = None
        self._prev: WorldState | None = None
        self._publisher = None  # set up lazily for render_mode == "vision"

    # -- subclass hooks -----------------------------------------------------

    def _observe(self, world: WorldState) -> np.ndarray:
        raise NotImplementedError

    def _decode(self, action) -> list[RobotCommand]:
        raise NotImplementedError

    def _terminated(self, world: WorldState) -> bool:
        return False

    # -- gym API ------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        scenario: Scenario = self.scenario_fn(self.np_random)
        self._world = self.backend.reset(scenario)
        self._prev = self._world
        self._steps = 0
        self.reward.reset()
        return self._observe(self._world), {}

    def step(self, action):
        assert self._world is not None, "call reset() first"
        self._prev = self._world
        self._world = self.backend.step(self._decode(action))
        self._steps += 1

        r = self.reward(self._world, self._prev)
        terminated = self._terminated(self._world)
        truncated = self._steps >= self.max_episode_steps

        if self.render_mode is not None:
            self.render()

        info: dict[str, Any] = {"reward_terms": dict(self.reward.last)}
        return self._observe(self._world), r, terminated, truncated, info

    def render(self):
        if self.render_mode == "vision":
            if self._publisher is None:
                from tbots.net.vision_publisher import VisionPublisher
                self._publisher = VisionPublisher(geometry=self.backend.geometry)
                self._publisher.publish_geometry()
            self._publisher.publish(self._world)

    def close(self):
        self.backend.close()
```

### 11.8 Remaining RL stubs

| File | Purpose | Task |
|---|---|---|
| `rl/envs/skill_env.py` | Single-robot skill training env | TASK-050 |
| `rl/envs/tactics_env.py` | Options wrapper — one `step()` runs many backend ticks until a skill terminates | TASK-051 |
| `rl/envs/synthetic_referee.py` | Produces `GameState` from rSim ground truth (out of bounds, goals, fouls) | TASK-052 |
| `rl/obs/builders.py` | Named observation builders; permutation-invariant relational encoding for tactics | TASK-053 |
| `rl/wrappers/domain_rand.py` | Inject latency, detection noise, dropouts — **without this, sim-to-real will fail** | TASK-054 |
| `rl/vec.py` | `AsyncVectorEnv` construction and throughput benchmarking | TASK-055 |
| `rl/train.py` | Hydra entry point: `python -m tbots.rl.train reward=... env=...` | TASK-056 |
| `rl/opponents.py` | Opponent pool + frozen-checkpoint sampler for self-play | TASK-057 |

> **Build TASK-057 now, even though we start against a scripted opponent.** `env.set_opponent(policy)` defaulting to `ScriptedDefense()` costs an hour today. Retrofitting an opponent pool into an environment that assumed a static adversary is a miserable multi-day refactor.

```bash
git add src/tbots/skills src/tbots/tactics src/tbots/rl
git commit -m "feat: skills, tactics, and RL scaffolding"
```

---

## Step 12 — Runnable apps

Three small programs. Each is a checkpoint: if it runs, a whole layer works.

### 12.1 `src/tbots/apps/ref_monitor.py`

```python
"""Print the game state whenever it changes. Proves the referee link works.

    python -m tbots.apps.ref_monitor --team "TritonBots"

Then click buttons in the game controller UI and watch lines appear.
"""

from __future__ import annotations

import argparse
import time

from tbots.net.referee import RefereeReceiver


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--team", default="TritonBots")
    p.add_argument("--group", default="224.5.23.1")
    p.add_argument("--port", type=int, default=10003)
    args = p.parse_args()

    rx = RefereeReceiver(group=args.group, port=args.port, team_name=args.team)
    print(f"listening on {args.group}:{args.port} as '{args.team}' ...")
    print("(if nothing appears, the GC is not running or is on other ports)")

    last_counter = -1
    try:
        while True:
            gs = rx.poll()
            if gs.counter != last_counter:
                last_counter = gs.counter
                print(
                    f"[{gs.counter:5d}] {gs.play.name:<16} "
                    f"ours={str(gs.ours):<5} "
                    f"move={str(gs.can_move):<5} touch={str(gs.can_touch_ball):<5} "
                    f"score={gs.our_score}-{gs.their_score} "
                    f"gk={gs.our_goalkeeper} max_bots={gs.our_max_robots} "
                    f"yellow_team={rx.we_are_yellow} flip_x={rx.flip_x}"
                )
            time.sleep(0.02)
    except KeyboardInterrupt:
        rx.close()


if __name__ == "__main__":
    main()
```

### 12.2 `src/tbots/apps/viz_rsim.py` — the payoff app

```python
"""Run rSim and stream it to ssl-vision-client. The architecture proof.

    python -m tbots.apps.viz_rsim

Open http://localhost:8082 and you are watching an in-process training
simulator through the same browser tool you use for live matches.
"""

from __future__ import annotations

import argparse
import math
import time

from tbots.backends.base import Scenario
from tbots.backends.rsim import RSimBackend
from tbots.core.geometry import DIV_B
from tbots.net.vision_publisher import VisionPublisher
from tbots.skills.go_to_point import GoToPoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--realtime", action="store_true",
                   help="sleep to match wall clock (needed to actually watch)")
    p.add_argument("--seconds", type=float, default=60.0)
    args = p.parse_args()

    backend = RSimBackend(n_us=6, n_them=6, dt=1.0 / 60.0, geometry=DIV_B)
    pub = VisionPublisher(geometry=DIV_B)
    pub.publish_geometry()

    world = backend.reset(Scenario.kickoff())
    skills = {i: GoToPoint(target=(0.0, 0.0)) for i in range(6)}
    for i, s in skills.items():
        s.reset(world, i)

    n_ticks = int(args.seconds / backend.dt)
    t0 = time.time()
    for tick in range(n_ticks):
        # Make the robots orbit, so there is obviously something happening.
        phase = tick * backend.dt * 0.5
        for i, s in skills.items():
            a = phase + i * (2 * math.pi / 6)
            s.target = (2.0 * math.cos(a), 1.5 * math.sin(a))

        cmds = [skills[i].step(world, i) for i in range(6)]
        world = backend.step(cmds)
        pub.publish(world)

        if tick % 300 == 0:
            pub.publish_geometry()
        if args.realtime:
            target = t0 + (tick + 1) * backend.dt
            time.sleep(max(0.0, target - time.time()))

    elapsed = time.time() - t0
    print(f"{n_ticks} ticks in {elapsed:.2f}s "
          f"= {n_ticks / elapsed:,.0f} steps/s "
          f"({n_ticks * backend.dt / elapsed:.1f}x realtime)")
    pub.close()
    backend.close()


if __name__ == "__main__":
    main()
```

Run it **without** `--realtime` once and write the steps/s number in the README. Every scaling decision you make later depends on that number.

### 12.3 `src/tbots/apps/wiggle.py` (stub)

```python
"""Drive robot 0 in a circle in the MATCH backend. Proves the UDP path.

    python -m tbots.apps.wiggle --color blue --sim-host 127.0.0.1

Depends on TASK-011 (robot_control) and TASK-010 (vision).
"""
raise NotImplementedError("TASK-012")
```

```bash
git add src/tbots/apps && git commit -m "feat: runnable apps"
```

---

## Step 13 — External tools

Three separate programs run alongside our Python: the **referee**, a **match simulator**, and a **visualizer**. None of them is a Python library; all of them are standalone binaries or containers.

### 13.1 The game controller

<u>Remember what it is:</u> a referee. A state machine with a web UI. **It has no physics.** It watches the world (via tracker packets) and broadcasts `Referee` messages. It can be attached to either backend, or to neither.

#### Option A — release binary (use this for local development)

The binary is self-contained; no dependencies at all. Pre-built for 64-bit Linux, Windows, and macOS.

```bash
cd ~/code/tritonbots/tools/bin

# Check https://github.com/RoboCup-SSL/ssl-game-controller/releases/latest
# for the current version and the exact asset filename for your platform.
VERSION=v3.21.0
curl -L -o ssl-game-controller \
  "https://github.com/RoboCup-SSL/ssl-game-controller/releases/download/${VERSION}/ssl-game-controller_${VERSION}_linux_amd64"
chmod +x ssl-game-controller
cd ../..
```

Run it:

```bash
mkdir -p tools/gc-config tools/gc-data
cd tools/gc-config
../bin/ssl-game-controller -address :8081
```

On first start it writes a default config into the current directory's `config/` folder. Open `http://localhost:8081` — you should see the referee interface.

#### Option B — Docker (use this for the full stack)

```bash
docker pull robocupssl/ssl-game-controller:3.21.0
docker run --rm -p 8081:8081 \
  -v "$(pwd)/tools/gc-config:/config" \
  -v "$(pwd)/tools/gc-data:/data" \
  robocupssl/ssl-game-controller:3.21.0 -address :8081
```

#### Three configuration changes you make once

**(a) Register our team name.** For a local override, add it to `config/engine.yaml`. To make it permanent for everyone (including at competitions), add it to `defaultTeams` in `internal/app/engine/config.go` in the GC repo and open a pull request upstream. **Do the PR in September, not April.**

The name must then match *exactly* everywhere — it is case-sensitive, spaces included — because the same string authenticates our team client connection later.

**(b) Choose the time mode.** In `config/ssl-game-controller.yaml`, set `time-acquisition-mode` to one of:

| Mode | What it does | Use it when |
|---|---|---|
| `system` | Uses wall-clock time | Normal development against the ER-Force sim; real matches |
| `vision` | Uses timestamps from incoming SSL-Vision frames | You generate vision packets from your own simulator and want the GC to follow your clock |
| `ci` | You drive it over TCP: send timestamp + tracker packets, receive the resulting referee message | Integrating the GC with rSim; deterministic tests; running faster than realtime |

**`ci` is the interesting one for us.** It is explicitly recommended for integrating the GC with your own simulator: no multicast traffic, the GC does nothing asynchronously in the background, you define the time and therefore the speed, and you supply the tracking data directly. It opens TCP port **10009** and speaks `CiInput` / `CiOutput` from `proto/ssl_gc_ci.proto`.

Two constraints when using `ci` mode:
- Geometry must be supplied statically in `config/ssl-game-controller.yaml` or sent through `CiInput`.
- Ball and robot positions **must** be sent with each `CiInput`. Filling only the required fields is sufficient.
- To fully suppress multicast, also unset `network.publish-address`.

This means we can drive the **real referee logic** from rSim at 100× realtime. Use it for evaluation runs and rule-compliance tests — **not** in the skill-training inner loop, where a TCP round trip per tick would destroy throughput.

**(c) Move off the standard ports, or kill the TTL.** The GC docs recommend non-standard ports whenever possible to avoid interfering with a real field setup. Our approach is simpler and stricter: **set multicast TTL to 0 everywhere in development** (already the default in `net/multicast.py`). TTL 0 means the packet never leaves your machine, so two teammates on the same lab wifi cannot referee each other's matches.

#### What the GC needs from us

The GC has external runtime dependencies. Without them it still runs, but several features silently do nothing:

- **ssl-vision geometry packets** — for correct field dimensions. If absent, configure dimensions manually in `config/ssl-game-controller.yaml`.
- **A tracker source producing `TrackerWrapperPacket`** — for ball and robot positions. Required for: ball placement progress, checking the number of robots per team, checking whether play can resume, "no progress" detection, and validating goalkeeper changes. The TIGERs AutoRef and the ER-Force AutoRef are both tracker-source implementations.

If you run the GC with no tracker, expect ball placement and robot-count enforcement to do nothing. That is not a bug.

### 13.2 The match simulator — we use ER-Force `simulator-cli`, and only that

You asked whether three simulators is too many. It is — so let us be precise about what we actually run.

We run **two** simulators, and they do genuinely different jobs:

| | **rSim** | **ER-Force `simulator-cli`** |
|---|---|---|
| Where it runs | Inside our Python process | Separate process, UDP |
| Speed | 100×+ realtime | Realtime |
| Protocol | None — direct function calls | Real SSL simulation protocol + SSL-Vision |
| Job | **Train policies** | **Prove they work for real** |

You cannot collapse these into one. A simulator fast enough for RL cannot be talking UDP at 60 Hz; a simulator that proves protocol compliance must be. Two is the floor.

**grSim is the third one, and we are dropping it.** Here is the reasoning, so nobody re-litigates it in November:

| | ER-Force `simulator-cli` | grSim |
|---|---|---|
| Used by the official virtual tournament setup | ✅ yes | ❌ present but commented out |
| Actively maintained | ✅ | ⚠️ repo yes, published Docker image is 4+ years stale |
| Headless | ✅ — runs in Docker, CI, and on servers with no display | ❌ needs Qt5 + OpenGL + a display |
| Tunable realism profiles | ✅ `--realism RC2021` etc. | ❌ |
| 3D GUI | ❌ | ✅ |
| macOS build | ⚠️ awkward | ⚠️ awkward |

grSim's one real advantage is its 3D GUI — and **we already have a visualizer that works with any vision source.** `ssl-vision-client` renders ER-Force, rSim, and live matches identically. Once you have that, grSim's GUI is a duplicate capability that costs a Qt/OpenGL dependency, a display server, and a stale container image.

**Decision: ER-Force `simulator-cli` is our only networked simulator.** Do not install grSim. If a specific need for its 3D view ever appears, we can add it in an afternoon — but do not carry it as a dependency on the theory that we might.

#### Running it

```bash
docker pull roboticserlangen/simulatorcli:latest
docker run --rm --network host \
  -e GEOMETRY=2020B -e REALISM=RC2021 \
  roboticserlangen/simulatorcli:latest
```

Three command-line options matter:

| Option | Meaning |
|---|---|
| `-g <name>` | Initial geometry, from the defaults in `config/simulator`. **Use `2020B`** — we play Division B (9.0 x 6.0 m). `2020` is the Division A pitch (12.04 x 9.02 m) and does not match `core/geometry.py`'s `DIV_B`. |
| `--realism <name>` | Realism profile from `config/simulator-realism` (e.g. `RC2021`, or `None` for idealised physics) |
| `--localhost` | Send only to the local machine. **Always use this in development.** |

> **Do not hardcode the vision port.** `simulator-cli` multicasts vision on **10020** by default rather than the usual 10006, specifically to avoid conflicts in tournament networks. The official virtual-tournament configuration uses `224.5.23.2:10020` for vision and `224.5.23.2:10010` for tracker data. Our `configs/net/*.yaml` makes this a setting for exactly this reason — set `vision.port: 10020` when running against ER-Force.

#### Building it from source (macOS, or when you want to modify realism configs)

```bash
brew install qt@5 protobuf cmake        # macOS
cd ~/code/tritonbots/third_party
git clone --recurse-submodules https://github.com/robotics-erlangen/framework.git erforce
cd erforce && mkdir -p build && cd build
cmake .. && make simulator-cli -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
./bin/simulator-cli -g 2020B --realism RC2021 --localhost
```

Budget an hour on macOS. If it fights you, that is fine — do networked-backend work on a Linux or WSL machine instead. See the [platform matrix](#platform-support-matrix).

### 13.3 ssl-vision-client — one visualizer for everything

`ssl-vision-client` is a small Go web app that listens to the SSL-Vision multicast and draws a 2D field in your browser. **It does not care where the packets came from.** Combined with our `VisionPublisher` (Step 10.3), it renders rSim training rollouts *and* live ER-Force matches with the same tool.

```bash
cd ~/code/tritonbots/tools/bin
# Check https://github.com/RoboCup-SSL/ssl-vision-client/releases/latest
# for the current version and asset name.
curl -L -o ssl-vision-client \
  "https://github.com/RoboCup-SSL/ssl-vision-client/releases/latest/download/ssl-vision-client_linux_amd64"
chmod +x ssl-vision-client
./ssl-vision-client -address :8082 -visionAddress 224.5.23.2:10006
```

Open `http://localhost:8082`. It will show an empty field until something publishes vision packets.

> If the release asset name has changed, browse the releases page and adjust. Do not guess the URL — check it.

### 13.4 `docker-compose.yml`

Create this in the repo root. It brings up referee + simulator + visualizer together.

```yaml
# Full local match stack. `docker compose up`
#
# network_mode: host is required for multicast to reach your host Python
# process. This works on Linux and inside WSL2.
#
# *** THIS FILE DOES NOT WORK ON macOS. ***
# Docker Desktop for Mac has no host networking, so multicast cannot cross
# the VM boundary. Mac users: run the game controller and ssl-vision-client
# as native binaries (Steps 13.1 Option A and 13.3), and use a Linux/WSL
# machine for the networked simulator. See the platform matrix in Step 0.
#
# VERIFY IMAGE TAGS against hub.docker.com before trusting them. Tags move.

services:
  game-controller:
    image: robocupssl/ssl-game-controller:3.21.0
    network_mode: host
    restart: unless-stopped
    volumes:
      - ./tools/gc-config:/config
      - ./tools/gc-data:/data
    command: ["-address", ":8081"]

  simulator:
    image: roboticserlangen/simulatorcli:latest
    network_mode: host
    restart: unless-stopped
    environment:
      GEOMETRY: "2020B"      # Division B. "2020" is Division A -- see above.
      REALISM: "RC2021"
    # Reminder: this publishes vision on 10020, not 10006.
    # Set vision.port accordingly in configs/net/dev.yaml.

  vision-client:
    image: robocupssl/ssl-vision-client:latest
    network_mode: host
    restart: unless-stopped
    command: ["-address", ":8082", "-visionAddress", "224.5.23.2:10020"]

  vision-client-rsim:
    # The same client for OUR publishers (viz_rsim, eval, replay) on 10006.
    # They cannot share 10020 with the simulator: both send camera 0 and
    # the browser flickers between the two worlds. :8082 is the match,
    # :8083 is rSim.
    image: robocupssl/ssl-vision-client:latest
    network_mode: host
    restart: unless-stopped
    command: ["-address", ":8083", "-visionAddress", "224.5.23.2:10006"]

  # Uncomment once you need ball placement and robot-count enforcement.
  # The GC cannot do those without a tracker source.
  # autoref:
  #   image: <check https://github.com/TIGERs-Mannheim/AutoReferee releases>
  #   network_mode: host

  # Handles ball/robot replacement and pushes robot specs to the simulator.
  # It picks Division A or B geometry based on the max robot count in the
  # referee message, and only applies it during NORMAL_FIRST_HALF_PRE.
  # simulation-controller:
  #   image: robocupssl/ssl-simulation-controller:0.13.0
  #   network_mode: host
  #   command:
  #     - "-refereeAddress"
  #     - "224.5.23.1:10003"
  #     - "-visionAddress"
  #     - "224.5.23.2:10006"
  #     - "-trackerAddress"
  #     - "224.5.23.2:10010"
  #     - "-simControlPort"
  #     - "10300"
```

**Verify:**

```bash
docker compose up -d
sleep 5
docker compose ps
curl -sf http://localhost:8081 > /dev/null && echo "GC UI ok"
curl -sf http://localhost:8082 > /dev/null && echo "vision-client ok"
curl -sf http://localhost:8083 > /dev/null && echo "vision-client-rsim ok"
```

> The GC repo also ships its own `docker-compose.yaml` that runs the GC together with autoRefs and other common components. Read it as a reference — it is maintained by the league and yours is not.

> **Two authoritative references** for how a real tournament stack is wired: `RoboCup-SSL/ssl-simulation-setup` (the virtual tournament configuration) and the GC repo's own compose file. When our compose disagrees with theirs, theirs is right.

### 13.5 `scripts/fetch_tools.sh`

Wrap the binary downloads so nobody has to remember URLs:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tools/bin

GC_VERSION="${GC_VERSION:-v3.21.0}"

echo "Fetching ssl-game-controller ${GC_VERSION} ..."
curl -fL -o tools/bin/ssl-game-controller \
  "https://github.com/RoboCup-SSL/ssl-game-controller/releases/download/${GC_VERSION}/ssl-game-controller_${GC_VERSION}_linux_amd64"
chmod +x tools/bin/ssl-game-controller

echo "Fetching ssl-vision-client ..."
curl -fL -o tools/bin/ssl-vision-client \
  "https://github.com/RoboCup-SSL/ssl-vision-client/releases/latest/download/ssl-vision-client_linux_amd64"
chmod +x tools/bin/ssl-vision-client

echo "done. binaries in tools/bin/"
tools/bin/ssl-game-controller -h 2>&1 | head -5 || true
```

```bash
chmod +x scripts/fetch_tools.sh
git add docker-compose.yml scripts/fetch_tools.sh
git commit -m "chore: external tool stack"
```

---

## Step 14 — Configuration

### 14.1 `configs/net/dev.yaml`

```yaml
# Development networking. TTL 0 keeps every packet on this machine, so two
# people on the same wifi cannot referee each other's matches.

team_name: "TritonBots"

multicast_ttl: 0

# Linux / WSL2: "0.0.0.0" is fine.
# macOS: MUST be "127.0.0.1" -- joining on the wildcard interface is
# unreliable, and you also need the loopback multicast route from Step 1M.5.
interface: "0.0.0.0"

vision:
  group: "224.5.23.2"
  # 10006 is the classic SSL-Vision port, used by our own VisionPublisher.
  # The ER-Force simulator publishes on 10020. Whichever you are pointing
  # at right now, this is the knob.
  port: 10006

referee:
  group: "224.5.23.1"
  port: 10003

tracker:
  group: "224.5.23.2"
  port: 10010

simulator:
  host: "127.0.0.1"
  control_port: 10300
  blue_port: 10301
  yellow_port: 10302

game_controller:
  ui_port: 8081
  team_port: 10008     # TCP; TLS on 10108
  ci_port: 10009       # TCP; only open in ci time-acquisition-mode
```

### 14.2 `configs/net/competition.yaml`

```yaml
# Competition networking. Identical addresses, TTL 1 so packets actually
# reach the field network. Simulation control is NOT available -- at a
# tournament that port is locked down, so nothing in the match path may
# depend on teleporting the ball.

team_name: "TritonBots"
multicast_ttl: 1
interface: "0.0.0.0"

vision:    {group: "224.5.23.2", port: 10006}
referee:   {group: "224.5.23.1", port: 10003}
tracker:   {group: "224.5.23.2", port: 10010}

simulator: null        # deliberately unavailable

game_controller:
  team_port: 10008
  ci_port: null
```

### 14.3 `configs/env/div_b_6v6.yaml`

```yaml
division: B
n_us: 6
n_them: 6
control_hz: 60          # MUST match everywhere. See the note below.
max_episode_seconds: 120

backend:
  kind: rsim
  field_type: 1         # verified in docs/RSIM_FACTS.md -- 0 is Division A

domain_randomization:
  enabled: true
  vision_latency_ms: [20, 45]
  command_latency_ms: [10, 20]
  position_noise_m: 0.005
  angle_noise_rad: 0.02
  dropout_probability: 0.02
```

> **Control frequency is one number, used everywhere.** rSim defaults to a 25 ms timestep (40 Hz); we override it to 1/60 s to match the real stack. A policy trained at 40 Hz and deployed at 60 Hz has a silent 1.5× error in every velocity integral, and you will spend a week finding it. Set it once, in config, and read it from there.

### 14.4 `configs/reward/example.yaml`

```yaml
# Machinery smoke test only. Real reward design is fall work.
terms:
  - {name: alive, weight: 0.0}
```

### 14.5 `configs/train/curriculum_example.yaml`

The scaffolding for curriculum learning. The stages below are a **shape**, not a recommendation — designing the actual curriculum is fall work.

```yaml
# A curriculum is an ordered list of stages. The trainer advances when the
# promotion criterion is met, or when max_steps is exhausted.
#
# The observation encoder MUST produce a fixed-size vector across every
# stage, or nothing transfers and the curriculum is decorative.
# See Step 9.2a.

observation:
  encoder: set_encoder     # or: pad_and_mask
  max_robots: 6            # padding width; constant across all stages

stages:
  - name: solo
    n_us: 1
    n_them: 0
    max_steps: 2_000_000
    promote_when: {metric: success_rate, above: 0.9}

  - name: one_v_one
    n_us: 1
    n_them: 1
    max_steps: 5_000_000
    promote_when: {metric: success_rate, above: 0.7}

  - name: two_v_two
    n_us: 2
    n_them: 2
    max_steps: 10_000_000
    promote_when: {metric: goal_diff_per_episode, above: 0.3}

  - name: full
    n_us: 6
    n_them: 6
    max_steps: 50_000_000
    promote_when: null       # terminal stage
```

> **Superseded 2026-09-07 (TASK-072).** This originally read "the trainer
> calls `backend.reconfigure(stage.n_us, stage.n_them)` on promotion".
> `reconfigure()` is gone and robot counts are fixed per run, so **each stage
> is its own run**, launched with its own `n_us`/`n_them` and resuming from the
> previous stage's checkpoint; `promote_when` is not read by anything and
> promotion is a human decision. See Step 9.2a. The file stays as documentation
> of the intended progression.

Because the field geometry never changes between stages, a policy carries its spatial understanding forward.

```bash
git add configs && git commit -m "chore: configuration"
```

---

## Step 15 — Tests and CI

### 15.1 `tests/test_core.py`

```python
import math

import pytest

from tbots.core.command import RobotCommand
from tbots.core.geometry import DIV_B, dist
from tbots.core.units import angle_diff, deg_to_rad, wrap_angle


def test_wrap_angle_range():
    for a in (-10.0, -math.pi, 0.0, math.pi, 3 * math.pi, 100.0):
        w = wrap_angle(a)
        assert -math.pi < w <= math.pi + 1e-12


def test_angle_diff_takes_short_way():
    assert angle_diff(deg_to_rad(179), deg_to_rad(-179)) == pytest.approx(
        deg_to_rad(-2), abs=1e-6
    )


def test_div_b_dimensions():
    assert DIV_B.length == 9.0 and DIV_B.width == 6.0
    assert DIV_B.their_goal == (4.5, 0.0)
    assert DIV_B.our_goal == (-4.5, 0.0)


def test_defense_areas_are_on_opposite_ends():
    assert DIV_B.inside_our_defense_area(-4.2, 0.0)
    assert not DIV_B.inside_our_defense_area(4.2, 0.0)
    assert DIV_B.inside_their_defense_area(4.2, 0.0)


def test_command_clamping():
    c = RobotCommand(0, vx=99.0, kick_speed=99.0, dribbler=5.0).clamped()
    assert c.vx == 3.0
    assert c.kick_speed == 6.5
    assert c.dribbler == 1.0
```

### 15.2 `tests/test_rsim_backend.py`

```python
import pytest

from tbots.backends.base import Scenario
from tbots.backends.rsim import RSimBackend
from tbots.core.command import RobotCommand
from tbots.core.geometry import dist


@pytest.fixture
def backend():
    b = RSimBackend(n_us=6, n_them=6, dt=1.0 / 60.0)
    yield b
    b.close()


def test_reset_places_robots_where_asked(backend):
    w = backend.reset(Scenario.single_robot_at(-1.0, 0.5))
    assert dist(w.us[0].pos, (-1.0, 0.5)) < 0.05
    assert len(w.us) == 6 and len(w.them) == 6


def test_forward_command_moves_forward(backend):
    w = backend.reset(Scenario.single_robot_at(0.0, 0.0))
    x0 = w.us[0].x
    for _ in range(60):
        w = backend.step([RobotCommand(0, vx=1.0)])
    assert w.us[0].x > x0 + 0.3, "robot should have moved ~1 m in 1 s"


def test_time_advances_by_dt(backend):
    w = backend.reset(Scenario.kickoff())
    t0 = w.t
    w = backend.step([])
    assert w.t == pytest.approx(t0 + backend.dt)


def test_state_length_matches_constants(backend):
    # If this fails, BALL_STRIDE / ROBOT_STRIDE are wrong.
    # Re-run scripts/verify_rsim.py.
    backend.reset(Scenario.kickoff())
```

### 15.3 `tests/test_backend_parity.py`

The acceptance test for the whole architecture. It is skipped until the network backend exists.

```python
"""Same skill, same result, both backends.

If GoToPoint converges in rSim but not against ER-Force, we have a
sim-to-sim gap -- and we would much rather find that in September than in April.
"""

import os

import pytest

from tbots.backends.base import Scenario
from tbots.backends.rsim import RSimBackend
from tbots.core.geometry import dist
from tbots.skills.go_to_point import GoToPoint

TARGET = (2.0, 1.0)


def _run(backend):
    world = backend.reset(Scenario.single_robot_at(-1.0, -1.0))
    skill = GoToPoint(target=TARGET)
    skill.reset(world, 0)
    for _ in range(600):
        world = backend.step([skill.step(world, 0)])
        if skill.status() == "success":
            break
    return dist(world.us[0].pos, TARGET)


def test_rsim_converges():
    b = RSimBackend(n_us=1, n_them=0)
    try:
        assert _run(b) < 0.06
    finally:
        b.close()


@pytest.mark.skipif(
    os.environ.get("TBOTS_NETWORK_TESTS") != "1",
    reason="requires a running simulator; set TBOTS_NETWORK_TESTS=1",
)
def test_network_converges():
    from tbots.backends.network import NetworkBackend  # noqa
    pytest.skip("TASK-014: implement NetworkBackend.observe()")
```

**Verify:**

```bash
pytest -q
```

Core and rSim tests must pass. The network test is skipped — that is correct for now.

### 15.4 GitHub Actions

Create `.github/workflows/ci.yml`:

```yaml
name: ci
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - name: Install system deps
        run: |
          sudo apt-get update
          sudo apt-get install -y build-essential cmake autoconf automake libtool libccd-dev

      - name: Cache ODE
        id: ode-cache
        uses: actions/cache@v4
        with:
          path: /usr/local/lib/libode*
          # Bump this key whenever the ODE configure flags change, or CI will
          # restore a stale library built with different physics settings.
          key: ode-0.16.2-double-sysccd

      - name: Build ODE
        if: steps.ode-cache.outputs.cache-hit != 'true'
        run: |
          cd /tmp
          wget -q https://bitbucket.org/odedevs/ode/downloads/ode-0.16.2.tar.gz
          tar -xzf ode-0.16.2.tar.gz && cd ode-0.16.2
          ./configure --enable-double-precision --with-box-cylinder=libccd \
                      --enable-libccd --enable-shared --disable-demos
          make -j2 && sudo make install && sudo ldconfig

      - uses: astral-sh/setup-uv@v3
      - run: uv venv --python 3.11
      - run: uv pip install -e ".[dev]"
      - run: uv pip install -e third_party/rsim
      - run: make proto
      - run: make lint
      - run: make test
```

```bash
git add tests .github && git commit -m "test: core, backend, and parity tests + CI"
```

---

## Step 16 — The acceptance checklist

Work through these in order. **Each one proves a whole layer works.** If you can tick all seven, the setup is done and the team can start on tactics and reward functions.

Open several terminals. In every one: `cd ~/code/tritonbots && source .venv/bin/activate`.

### ✅ 1. Protobufs generate and import

```bash
make proto
python -c "from tbots._pb.state.ssl_gc_referee_message_pb2 import Referee; print('ok')"
```
*Proves:* submodules are checked out, codegen works, import rewriting worked.

### ✅ 2. rSim builds and steps

```bash
pytest -q tests/test_rsim_backend.py
```
*Proves:* ODE is correct, the C++ extension built, your stride constants are right.

### ✅ 3. The game controller runs

```bash
docker compose up -d game-controller          # Linux / WSL2
# macOS:  cd tools/gc-config && ../bin/ssl-game-controller -address :8081
```
Open `http://localhost:8081`. Confirm `TritonBots` appears in the team dropdown.
*Proves:* the GC binary/image works and your team name is registered.

### ✅ 4. We receive referee messages

Terminal A: GC running (step 3).
Terminal B:
```bash
python -m tbots.apps.ref_monitor --team "TritonBots"
```
Now click **Stop**, then **Force Start**, then **Halt** in the GC UI. Each click must print a new line within a second, with the counter incrementing.

*Proves:* multicast works, protobuf parsing works, the `Referee` → `GameState` mapping works, colour resolution works.

*If nothing prints:* see [Troubleshooting → No referee messages](#no-referee-messages).

### ✅ 5. The visualizer runs

```bash
docker compose up -d vision-client-rsim       # Linux / WSL2
# macOS:  tools/bin/ssl-vision-client -address :8082 -visionAddress 224.5.23.2:10006
```
Open `http://localhost:8083` (macOS: `:8082`). You should see an empty green field.
*Proves:* the client is up and listening on the right multicast group.

### ✅ 6. rSim renders in the browser — **the architecture proof**

Terminal C:
```bash
python -m tbots.apps.viz_rsim --realtime --seconds 60
```
Watch `http://localhost:8083` (macOS: `:8082`). Six robots should orbit the centre circle.

*Proves:* the in-process training simulator, our `WorldState` contract, and the league's vision serialisation all agree with each other. At this moment the same browser tab can render both backends, and nothing above `net/` knows the difference. **This is the whole design working.**

Now run it without `--realtime` and record the throughput:
```bash
python -m tbots.apps.viz_rsim --seconds 60 | tail -1
```
Put that steps/s number in the README. Every scaling decision depends on it.

### ✅ 7. Full stack starts clean

**Ubuntu / WSL2:**

```bash
docker compose down && docker compose up -d && sleep 5 && docker compose ps
```
All services `running`. No restart loops.

**macOS** — compose does not work here, so run the native binaries instead:

```bash
# Terminal 1
sudo route -n add -net 224.0.0.0/4 -interface lo0     # once per boot
cd tools/gc-config && ../bin/ssl-game-controller -address :8081

# Terminal 2
tools/bin/ssl-vision-client -address :8082 -visionAddress 224.5.23.2:10006

# Terminal 3
python -m tbots.apps.viz_rsim --realtime --seconds 30
```

Both web UIs load and the robots move. That is the macOS equivalent of a green stack — the networked simulator is the only piece you are missing, and that is expected.

---

## Step 17 — HPC image

Training runs on the cluster, not on a laptop. Three things about HPC that catch every student team:

1. **Most clusters ban Docker.** You need Apptainer (formerly Singularity).
2. **Compute nodes usually have no internet.** Weights & Biases must run in offline mode and be synced from the login node afterwards.
3. **Walltime limits will kill your run.** Checkpoint-and-resume is not optional — write it in week one, not the week you first lose a twelve-hour job.

### 17.1 `containers/train.def`

```
Bootstrap: docker
From: ubuntu:24.04

%post
    export DEBIAN_FRONTEND=noninteractive
    apt-get update && apt-get install -y \
        build-essential cmake pkg-config git curl wget ca-certificates \
        autoconf automake libtool libccd-dev

    # ODE 0.16.2, double precision, libccd
    # libccd-dev above is required: without it configure silently uses ODE's
    # bundled libccd and the container's physics diverges from developers'.
    cd /tmp
    wget -q https://bitbucket.org/odedevs/ode/downloads/ode-0.16.2.tar.gz
    tar -xzf ode-0.16.2.tar.gz && cd ode-0.16.2
    ./configure --enable-double-precision --with-box-cylinder=libccd \
                --enable-libccd --enable-shared --disable-demos
    make -j"$(nproc)" && make install && ldconfig
    cd / && rm -rf /tmp/ode-0.16.2*

    # Python 3.11 is not in the noble repositories, and 24.04 enforces
    # PEP 668 so system pip refuses to install anyway. Install uv standalone
    # and let it fetch its own CPython 3.11 -- same interpreter as dev boxes.
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
    uv python install 3.11

%environment
    export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    export WANDB_MODE=offline
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1

%runscript
    exec "$@"
```

Build it (needs a machine where you have root or fakeroot):

```bash
apptainer build --fakeroot ssl-train.sif containers/train.def
```

> `OMP_NUM_THREADS=1` is not decorative. rSim is single-threaded, and we parallelise by running many *processes*. If PyTorch and BLAS each spawn threads inside every worker, 64 workers will fight over 64 cores and throughput collapses. Pin it to 1.

### 17.2 `scripts/train.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=ssl-train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load apptainer || true

RUN_DIR="runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR" logs

apptainer exec --bind "$PWD:/work" --pwd /work ssl-train.sif \
  python -m tbots.rl.train \
    env=div_b_6v6 \
    reward="${REWARD:-example}" \
    train.num_envs=128 \
    train.run_dir="$RUN_DIR" \
    train.resume_from="${RUN_DIR}/latest.pt"
```

Submit and sync:

```bash
sbatch scripts/train.slurm
# after it finishes, from the LOGIN node (which has internet):
wandb sync wandb/offline-run-*
```

### 17.3 Throughput expectations

rSim is single-threaded C++ per instance; we scale by processes, not by GPU batching. On a 64-core node with 128 environments, expect roughly **10⁴–10⁵ environment-steps/second** for 6v6. Benchmark it in week one (checklist item 6) and write the real number down — do not trust this estimate.

That is plenty for skill-level PPO and for the tactics layer, where each macro-step covers ~30 physics ticks. It is *not* Isaac Gym territory. If we ever need 10⁶ steps/s, that is a rewrite onto a GPU-native simulator and a decision for next year, not this one.

```bash
git add containers scripts/train.slurm && git commit -m "chore: HPC image and slurm script"
```

---

## Troubleshooting

### No referee messages

Work down this list:

```bash
# 1. Is the GC actually running and publishing?
docker compose logs game-controller | tail -20

# 2. Is anything on the wire at all?
sudo tcpdump -i any -n 'udp port 10003'
# Click a button in the GC UI. If tcpdump shows nothing, the GC is not
# publishing -- check network.publish-address in its config, and confirm
# time-acquisition-mode is not `ci` with publishing disabled.

# 3. Is your machine even joining the multicast group?
ip maddr show | grep -A2 224.5.23

# 4. Firewall
sudo ufw status
sudo ufw allow 10003/udp   # if active
```

Most common causes, in order of frequency:
1. GC is in `ci` mode with `network.publish-address` unset — it is deliberately not broadcasting.
2. You are on a VPN or a virtual interface that is capturing multicast. Disconnect it.
3. WSL2 — multicast between the Windows host and the WSL VM does not work reliably. Run everything *inside* WSL, including Docker with `network_mode: host`.
4. `docker compose` without `network_mode: host` — bridge networking does not forward multicast.

### `ModuleNotFoundError: No module named 'ssl_gc_common_pb2'`

You ran `protoc` without the `protoletariat` rewrite step. Run `make proto`, which does both.

### rSim state array is the wrong length

Your `BALL_STRIDE` / `ROBOT_STRIDE` constants in `backends/rsim.py` do not match your build. Re-run `python scripts/verify_rsim.py`, update `docs/RSIM_FACTS.md`, then update the constants.

### Robots drive in the wrong direction

Almost always one of three things:
1. Angle units — check `ANGLES_IN_DEGREES` against `docs/RSIM_FACTS.md`.
2. Local vs global frame — `RobotCommand` velocities are **robot-local**. `GoToPoint` rotates the global error into the local frame; if you skipped that rotation, robots will drive at an angle that changes as they turn.
3. The colour/side flip in the network backend firing when it shouldn't, or not firing when it should. Print `we_are_yellow` and `flip_x` from `ref_monitor` and confirm they match reality.

### Everything worked yesterday, nothing works today

```bash
git submodule status         # did a submodule drift off its pin?
make clean && make proto     # stale generated code
uv pip install -e third_party/rsim --force-reinstall
```

### macOS: nothing sends or receives multicast

Almost always the missing loopback route from Step 1M.5:

```bash
netstat -rn | grep 224
# if empty:
sudo route -n add -net 224.0.0.0/4 -interface lo0
```

**This route does not survive a reboot.** If multicast worked yesterday and does not today, and you restarted your Mac, that is why.

Second most common: `interface` is still `"0.0.0.0"` in `configs/net/dev.yaml`. On macOS, joining a multicast group on the wildcard interface is unreliable — set it to `"127.0.0.1"`.

Third: you are on a corporate VPN. Most VPN clients hijack multicast routing. Disconnect and retry.

### macOS: `docker compose up` starts but nothing talks to my Python

Expected. Docker Desktop for Mac has no host networking, so multicast cannot reach your host process. Run the game controller and ssl-vision-client as native binaries — see acceptance check 7's macOS variant.

### macOS: mixed architecture errors

Symptoms: `mach-o file, but is an incompatible architecture`, or an import that fails with no useful message.

```bash
file "$(brew --prefix)/lib/libode.dylib"
python3 -c "import platform; print(platform.machine())"
lipo -info .venv/lib/python3.11/site-packages/robosim*.so
```

All three must agree (`arm64` on M-series, `x86_64` on Intel). If they do not, you have a Rosetta-contaminated toolchain: reinstall Homebrew natively, recreate the venv, and rebuild ODE and rSim.

### Ubuntu 24.04 (noble) specifics

**`pkg-config --modversion ode` says 0.16.2 but physics is wrong.**
The single most important 24.04 gotcha. Noble's packaged `libode-dev` *is*
0.16.2, so the version check passes while resolving to Debian's build rather
than ours. Diagnose with `pkg-config --variable=libdir ode` (must be
`/usr/local/lib`) and
`ldd "$(python -c 'import robosim._robosim as m; print(m.__file__)')" | grep ode`.
Fix per the warning in Step 1.4.

**`E: Unable to locate package docker-compose-plugin`.**
That package lives in Docker's own apt repo, not Ubuntu's. Use
`docker-compose-v2` from universe instead.

**`E: Package 'freeglut3-dev' has no installation candidate`, or it installs
a suspiciously tiny package.** On noble it is a transitional shim. Install
`libglut-dev`.

**`error: externally-managed-environment` from pip.**
24.04 enforces PEP 668. You should not be hitting this at all — everything
goes through `uv` inside `.venv`. If you see it, you are outside the venv:
`source .venv/bin/activate`. Do not reach for `--break-system-packages`.

**C++ compile errors mentioning `uint32_t`, `int64_t`, or `size_t` "not
declared".** GCC 13 on noble no longer leaks `<cstdint>` in transitively.
Add the include. This is TASK-002 work — fix it in our rSim fork and push.

**Claude Code's `/sandbox`, or anything using bubblewrap, fails to start.**
24.04 restricts unprivileged user namespaces by default via AppArmor. Check
with `sysctl kernel.apparmor_restrict_unprivileged_userns`. This does not
affect the build; it only affects sandboxing tools. Leave the restriction in
place and work without the sandbox rather than weakening a system-wide
security control.

### WSL2 specifics

Run **everything** inside WSL — Python, Docker, the binaries. Multicast between the Windows host and the WSL VM is unreliable and will waste your afternoon.

Use Docker installed inside WSL (`sudo apt install docker.io`) rather than Docker Desktop's Windows backend, or `network_mode: host` will not behave.

Clone the repo into the WSL filesystem (`~/code/tritonbots`), never into `/mnt/c/...`. Building C++ extensions across the 9p filesystem boundary is roughly ten times slower and occasionally corrupts build artifacts.

---

## The task board

> **Moved.** The live task board is now **`docs/TASKS.md`**.
>
> The board that used to sit here described the world *before* this guide had
> ever been run. It has been overtaken by the build: TASK-001 and TASK-002 are
> closed, twelve tasks that the build surfaced were never on it, and the
> must-do/can-wait split has been re-cut around what actually blocks a recruit's
> first day. `docs/TASKS.md` carries the current tiers, sizes, owners, and a
> "done when" gate for every item, plus a section recording exactly what changed
> from this version and why.
>
> The task IDs are unchanged — `TASK-050` still means `rl/envs/skill_env.py`.
> The `NotImplementedError("TASK-0xx")` markers throughout this guide remain
> correct; look the ID up in `docs/TASKS.md`.

Everything this setup deliberately leaves unimplemented is listed there, with
the two exceptions this guide builds itself: TASK-001 (verify rSim's
conventions) and TASK-002 (rSim on Python 3.11).

---

## `README.md`

Finally, create a short root README so a new person knows where to start:

```markdown
# TritonBots — RoboCup Small Size League, Division B

## Quick start

    git clone --recurse-submodules https://github.com/tritonbots/tritonbots.git
    cd tritonbots
    # then follow docs/SETUP.md end to end
    # already set up? go straight to docs/ONBOARDING.md

## Once set up

    make proto              # regenerate protobufs
    make test               # run tests
    docker compose up -d    # referee + simulator + visualizer
    python -m tbots.apps.viz_rsim --realtime   # watch rSim at localhost:8083

## Measured throughput

rSim, 6v6, 60 Hz, single process:  ______ steps/s   (fill this in)

## Where things live

| I want to... | Go to |
|---|---|
| write a reward function | `src/tbots/rl/rewards/` |
| write a skill | `src/tbots/skills/` |
| write a tactic | `src/tbots/tactics/` |
| change field dimensions | `src/tbots/core/geometry.py` |
| change ports | `configs/net/` |
| set up a new machine | `docs/SETUP.md` |
| get productive on day one | `docs/ONBOARDING.md` |
| understand the architecture | `docs/ARCHITECTURE.md` |

## The architectural rules

- **Rule 1.** `src/tbots/core/` imports nothing from the rest of the codebase.
- **Rule 2.** Two backends, one match contract. Match code sees only `Backend`; training takes `SimBackend`.
- **Rule 3.** We are always `us`, we always attack `+x`. The backend does the flipping.
- **Rule 4.** Units convert exactly once, at the backend boundary. Above it: meters, radians, seconds.
```

```bash
git add README.md && git commit -m "docs: readme"
git push -u origin main
```

---

## You are done when

- [ ] All seven acceptance checks pass
- [ ] `docs/RSIM_FACTS.md` contains four verified answers, not question marks
- [ ] The measured steps/s number is in the README
- [ ] CI is green on a fresh clone
- [ ] Every Tier 1 id in `docs/TASKS.md` is a GitHub issue with an owner

At that point a new recruit can clone the repo, run one command, watch six robots move in a browser, and start writing the code that actually wins matches.
