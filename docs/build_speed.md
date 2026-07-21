# Build speed & the agent build workflow

Canonical guide for making Thalos compile fast on every platform an agent (or a
human) drives it from: **Windows native**, **Windows + WSL2**, **macOS**, and a
**headless Linux cloud box**. Also the home for the **sccache** setup and the
**two build regimes** (fast local iteration vs. parallel/clean throughput).

This supersedes the per-platform "fast incremental loop" recipes that used to
live in [tooling.md](tooling.md); tooling.md now points here. The invariants it
still owns (dynamic-linking launch contract, one-Cargo-at-a-time, feature
fingerprint, env-var launch toggles) are referenced below, not duplicated.

The rule from `CLAUDE.md` stands: **platform-specific compiler/linker tuning
lives in the gitignored `.cargo/config.toml` and `.env.just`, never in committed
workspace config.** Everything this doc configures is local, per-machine, and
ignored by Git. The committed `Cargo.toml` keeps only portable profile choices
(`dev` opt-level 1, deps opt-level 3, release thin-LTO).

---

## 1. TL;DR

If you do nothing else:

- **Every platform:** keep Bevy `dynamic_linking` on (already the committed
  default for `just game`/`screenshot`/`preview`/`ui-preview`), use a **fast
  linker** (rust-lld on Windows, **mold** on Linux/WSL, the default new ld on
  macOS), keep **incremental on** for the edit→rebuild→screenshot loop, and on
  Windows **exclude `target/` from Defender**.
- **Run `scripts/setup-build-env.sh`** (Linux/macOS/WSL) or
  `scripts/setup-build-env.ps1` (Windows). They install the linker + sccache and
  drop the right local config in place.
- **For parallel agents / a cloud box / fresh worktrees:** turn on **sccache**
  and set `CARGO_INCREMENTAL=0`. sccache turns a cold Bevy dep-graph build from
  minutes into a cache hit; it is the single biggest lever for many-worktree and
  clean-build throughput.

The highest-leverage single move for **agent iteration specifically** — because
it is headless (screenshot tool, no interactive window) — is to run the loop on
**Linux (WSL2 or a cloud box) with mold + dynamic linking + sccache**, project
on the Linux filesystem. Your headless requirement removes the only real reason
Windows→Linux is painful for game dev (the display/GPU path).

---

## 2. What actually costs time

Two numbers dominate, and they respond to different levers:

**Linking.** A Bevy binary links a large amount of code on *every* change, and
the link is largely single-threaded. This is why the linker choice and Bevy
`dynamic_linking` (which stops relinking the engine into your binary) matter more
for the *iteration* loop than raw core count. Fast linker + dynamic linking is
where the incremental-loop wins are.

**The dependency graph (Bevy et al.).** Compiling Bevy + wgpu + deps from cold
is minutes of CPU. You pay it on a fresh checkout, a new git worktree, a changed
feature fingerprint, or a `cargo clean`. This is what **sccache** eliminates
(cache hit instead of recompile) and what **more cores** speed up (deps compile
in parallel across crates).

So there are two regimes, and the ideal config differs between them:

| Regime | You are… | Bottleneck | Win with |
|--------|----------|-----------|----------|
| **Iterate** | Editing one crate, rebuilding + screenshotting on one machine | Link + one-crate codegen (single-threaded) | Fast linker, dynamic linking, **incremental on** |
| **Cold / parallel** | Fresh worktree, cloud box, N agents in parallel, CI | The whole dep graph | **sccache**, cores, RAM; **incremental off** |

The awkward part: **incremental and sccache pull against each other** (§5). Pick
per regime rather than trying to run both hot at once.

---

## 3. The optimization layers

Stack these; they are orthogonal except where noted.

### 3.1 Bevy dynamic linking — already the committed default
`just game`/`screenshot`/`preview`/`ui-preview` build with
`--features bevy/dynamic_linking`. Bevy links once into a shared `bevy_dylib`;
subsequent edits relink only Thalos crates. Do **not** add it to release paths
(`just build`, `just trace`, tests, bakes) — a shipped binary with a missing
`bevy_dylib` crashes. Any tool that launches a dev renderer directly must
reproduce Cargo's dylib search-path contract (profile dir + `deps` +
`rustc --print target-libdir` onto `PATH`/`DYLD_FALLBACK_LIBRARY_PATH`/
`LD_LIBRARY_PATH`) — see INC-0008 and tooling.md.

### 3.2 A fast linker (per platform)
The default MSVC `link.exe` (Windows) and the historical `ld` are slow. Use:

- **Windows:** `rust-lld` (ships with the toolchain). Not the Rust default on
  Windows yet, so configure it explicitly (§6.1).
- **Linux / WSL:** **mold** — the fastest option, configured explicitly via
  clang.
- **macOS:** the new Apple `ld` (Xcode 15+) is already fast; usually nothing to
  do. lld on macOS is possible but historically fussy — leave the default unless
  you measure a reason.

### 3.3 Incremental + trimmed debug info
`CARGO_INCREMENTAL=1` and `[profile.dev] incremental = true` make one-crate edits
cheap. `debug = "line-tables-only"` (both `dev` and `dev.package."*"`) keeps
backtraces while cutting debuginfo generation and link size — a real chunk of
link time. This is the **Iterate** regime default in `.cargo/config.toml`.

### 3.4 opt-level split — already committed
`Cargo.toml` sets `[profile.dev] opt-level = 1` (your code stays cheap to
compile) and `[profile.dev.package."*"] opt-level = 3` (deps run fast). Tradeoff:
the *first* cold dep build is slower because deps compile optimized — which is
exactly why sccache (§5) pays for itself on the cold/parallel regime.

### 3.5 sccache — the new piece
A compiler cache that turns a repeated rustc invocation into a cache hit. See
§5 for the full setup and the incremental interaction. Headline: it does **not**
speed up your own frequently-edited crates in the iterate loop, but it makes
**cold dep-graph builds and fresh worktrees near-instant**, which is the whole
game for parallel agents and cloud boxes.

### 3.6 One Cargo at a time; one feature fingerprint
From `CLAUDE.md`/tooling.md, still load-bearing:

- Run **one Cargo command at a time** against the workspace `target/`.
  Concurrent `game`/`check`/`screenshot` invocations serialize on the target
  lock while competing for CPU and several GiB of compiler memory. Use
  `cargo check-game` while editing, then **one** linked `just game`/`just
  screenshot` when an artifact is actually needed. (For *genuinely* parallel
  agents, give each its own worktree + target dir + shared sccache — §7.2 —
  which sidesteps the single-target-lock contention.)
- Keep every dev renderer on the **same Bevy/wgpu feature fingerprint**. Adding
  a feature to one entry point forces Cargo to build a second full `bevy_dylib`
  (+ Windows PDB). wgpu's `counters` feature stays opt-in as
  `thalos_game/gpu-counters`; only enable it for a focused `mem_diag`
  investigation via a temporary `THALOS_GAME_COMMAND` override, never in normal
  iteration.

### 3.7 What NOT to do
- **No unstable `-Zthreads`.** INC-0006: `-Zthreads=8` on the 2026-07-20 Windows
  nightly hit a parallel-MIR ICE and left incremental objects that failed to
  link (missing LLVM symbols), turning a speculative speedup into a full
  recovery build. Cargo's crate-level parallelism + reliable incremental reuse
  is the faster loop.
- **No compiler-backend experiments.** LLVM is the pinned stable toolchain's
  backend on every platform. The former Windows Cranelift attempt failed with
  cross-crate undefined statics (reverted 2026-07-04); the project does not
  support a separate local backend.

### 3.8 Windows Defender exclusions (Windows only, big win)
Real-time AV scanning of every `.o`/`.rlib`/`.pdb`/`.exe` in `target/` silently
taxes every build. Excluding the build dirs is often a 20–40% incremental-build
improvement on Windows. Requires an elevated shell:

```powershell
Add-MpPreference -ExclusionPath "C:\Users\korbi\Documents\thalos\target"
Add-MpPreference -ExclusionPath "$env:USERPROFILE\.cargo"
Add-MpPreference -ExclusionPath "$env:USERPROFILE\.rustup"
Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\Mozilla\sccache"  # if sccache used
```

`scripts/setup-build-env.ps1` offers to do this when run elevated.

---

## 4. Headless rendering (no display) — required for the screenshot tool

Agent iteration verifies visually through `just screenshot`/`just preview`/`just
compare`, which boot the real game binary **winit-less** (a `ScheduleRunnerPlugin`
frame loop) and render one frame off-screen. This needs a **working Vulkan
device**, but **no X server / no `xvfb`** (there is no window).

- **Real GPU (fastest, best fidelity):** install the GPU's Vulkan driver. On a
  Linux box with an NVIDIA GPU this already works from an agent shell
  (NVIDIA/Vulkan, per `CLAUDE.md`). This is the preferred cloud-box setup.
- **Software fallback (no GPU / flaky driver):** Mesa **lavapipe** (`lvp`) gives
  a deterministic CPU Vulkan device. Slower per frame but reproducible and
  driver-independent — good for CI-style screenshot diffing. Install
  `mesa-vulkan-drivers` (Debian/Ubuntu) and select it with
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`, or force the
  GL backend via `THALOS_WGPU_BACKEND=gl`.
- **WSL2:** WSLg exposes the host GPU (`/dev/dxg` → Mesa). Real-GPU Vulkan
  usually works out of the box; lavapipe is the fallback. Either way the
  screenshot tool needs no display server.

`THALOS_WGPU_BACKEND` (`auto|dx12|vulkan|metal|gl`) selects the backend without
touching source — see tooling.md.

---

## 5. sccache

sccache caches individual rustc compilations keyed on inputs, so a repeated
compile (same crate, same flags, same sources) is a cache read instead of a
rebuild.

### 5.1 The incremental interaction — read this first
**sccache cannot cache incrementally-compiled crates.** With
`CARGO_INCREMENTAL=1`, your workspace crates are compiled incrementally and
sccache skips them; only the non-incremental dependency crates get cached. Two
consequences:

- In the **Iterate** regime (editing your crates on one machine), incremental is
  the win and sccache adds little for *your* crates — keep incremental on, leave
  sccache off (or accept it only caches deps).
- In the **Cold / parallel** regime (fresh worktree, cloud box, CI, N agents),
  set **`CARGO_INCREMENTAL=0`** and turn sccache **on**. Now the entire graph —
  Bevy included — is cacheable, and a fresh worktree reuses the cache instead of
  recompiling from scratch.

That is why sccache is **opt-in per shell/box**, not baked into
`.cargo/config.toml`. Flipping it globally would silently disable the
incremental loop.

### 5.2 Install
- **Linux/WSL/macOS:** `cargo binstall sccache` (prebuilt, fast) — or
  `brew install sccache` (macOS), or `cargo install sccache --locked`.
- **Windows:** `scoop install sccache` or `winget install Mozilla.sccache` — or
  `cargo install sccache --locked`.

The setup scripts do this for you.

### 5.3 Turn it on for a shell (Cold/parallel regime)
```bash
export RUSTC_WRAPPER=sccache
export CARGO_INCREMENTAL=0
export SCCACHE_DIR="$HOME/.cache/sccache"     # Windows: %LOCALAPPDATA%\Mozilla\sccache
export SCCACHE_CACHE_SIZE="50G"               # Bevy artifacts are large; give it room
```

Helper toggles are provided: `source scripts/sccache-on.sh` (bash) /
`scripts/sccache-on.ps1` (PowerShell) set exactly these for the current shell so
you don't pollute the iterate loop.

### 5.4 Verify it's working
```bash
sccache --show-stats     # before
cargo build --workspace  # or: just build
sccache --show-stats     # 'Compile requests' up; 'Cache hits' rising on 2nd worktree/build
```
A cold first build is all misses (it's *populating* the cache); the payoff is
the second worktree/branch/checkout, which should be mostly hits.

### 5.5 Shared cache for a cloud box (advanced)
For **multiple agents on one box**, a single local `SCCACHE_DIR` is already
shared across all their worktrees — that's the main win. To share across
*machines* (a fleet, or CI + dev), point sccache at a backend instead of the
local dir: S3 (`SCCACHE_BUCKET`), GCS, Redis, or memcached. Keep the local dir
for a single box; reach for a remote backend only when a second machine needs
the same cache.

---

## 6. Per-platform setup

Run the setup script for your platform; the manual config it writes is below for
reference. All of it lands in the **gitignored** `.cargo/config.toml` /
`.env.just`.

The portable `.cargo/config.toml` this repo ships (local, gitignored) carries a
`[target.<triple>]` table per platform. Cargo only applies the table matching the
**active** target, so one file works on all machines; the Windows absolute
linker path is inert on macOS/Linux and vice-versa.

### 6.1 Windows (native)
`rust-lld` + incremental + trimmed debug. This is your current known-good setup;
keep it:

```toml
[env]
CARGO_INCREMENTAL = "1"

[profile.dev]
incremental = true
debug = "line-tables-only"

[profile.dev.package."*"]
debug = "line-tables-only"

[target.x86_64-pc-windows-msvc]
# rust-lld.exe on PATH also works; absolute path is the toolchain copy.
linker = "C:/Users/korbi/.rustup/toolchains/1.97.0-x86_64-pc-windows-msvc/lib/rustlib/x86_64-pc-windows-msvc/bin/rust-lld.exe"

[alias]
check-game = "check -p thalos_game"
```

Then: Defender exclusions (§3.8). Optionally sccache for clean builds (§5).
`cargo check-game` while editing; one `just game`/`just screenshot` when you need
a frame.

### 6.2 Windows + WSL2 (recommended for agent iteration)
WSL2 gives you the Linux toolchain (mold) with none of the display headaches,
because iteration is headless. Setup:

1. `wsl --install` (Ubuntu). Inside WSL: install rustup; the repository selects the pinned stable release,
   `just`, `clang`, `mold`, `mesa-vulkan-drivers`.
2. **Keep the project on the Linux filesystem** — clone into `~/thalos`, **not**
   `/mnt/c/...`. Building across the `/mnt/c` 9p mount is drastically slower and
   erases most of the gains. This is the #1 WSL mistake.
3. Run `scripts/setup-build-env.sh` — it writes the Linux `.cargo/config.toml`
   (mold via clang) and installs sccache.
4. Headless rendering: WSLg usually gives real-GPU Vulkan; lavapipe is the
   fallback (§4).

You can share one Windows checkout by cloning the repo separately inside WSL, or
work entirely in WSL and push/pull. Do not build the same directory from both
Windows and WSL — the `target/` artifacts are not cross-compatible.

### 6.3 macOS
The default new linker is fast; usually just incremental + trimmed debug:

```toml
[env]
CARGO_INCREMENTAL = "1"

[profile.dev]
incremental = true
debug = "line-tables-only"

[profile.dev.package."*"]
debug = "line-tables-only"

[alias]
check-game = "check -p thalos_game"
```

If a macOS toolchain hits stale `.llvm.<hash>` anonymous-symbol references
between incremental objects, disable incremental locally (`[profile.dev]
incremental = false`) rather than touching the workspace profile. Cranelift for
local iteration is an opt-in local experiment (§8). Metal is the default wgpu
backend; the screenshot tool renders headless through it.

### 6.4 Linux cloud box (recommended for parallel agents)
The throughput setup. mold + sccache + `CARGO_INCREMENTAL=0`, real-GPU Vulkan if
the box has a GPU, lavapipe otherwise.

```toml
# .cargo/config.toml
[profile.dev]
debug = "line-tables-only"

[profile.dev.package."*"]
debug = "line-tables-only"

[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=mold"]

[alias]
check-game = "check -p thalos_game"
```

```bash
# box profile (~/.bashrc or the agent launcher) — Cold/parallel regime
export RUSTC_WRAPPER=sccache
export CARGO_INCREMENTAL=0
export SCCACHE_DIR=/var/cache/sccache        # shared across all agents/worktrees on this box
export SCCACHE_CACHE_SIZE=100G
```

Provision: `clang`, `mold`, `just`, `sccache`, the GPU Vulkan driver (or
`mesa-vulkan-drivers` for lavapipe), and rustup (the repository selects the pinned stable release). Size the box
for the dep graph: many cores help the cold build, and **RAM matters** (linking
is memory-hungry — budget several GiB per concurrent cargo; see §7.2 for how
many agents to run).

---

## 7. The ideal agent build workflow

Principles first, then the two concrete shapes.

**Principles (both shapes):**
- **Check before you link.** `cargo check-game` (or `cargo check -p <crate>`)
  catches type/borrow errors in a fraction of a link. Only build a linked
  artifact (`just game`/`just screenshot`) when you actually need to run or see
  something.
- **Verify visually, headless.** Terrain/lighting/scatter changes are `verify`,
  not `done`, until a headless capture confirms them — `just screenshot
  <preset>`, `just preview`, `just compare <preset> <axis>`. Agents run these and
  read the PNG directly; only genuine "does it feel right in motion" needs the
  user. (See visual_testing.md, terrain.md.)
- **Never run the game interactively to check compile/agent work** — that's the
  user's job (`CLAUDE.md`). The headless tools are the agent's eyes.

### 7.1 Shape A — single machine, one agent (local dev, WSL, your laptop)
**Iterate regime.** Incremental on, dynamic linking on, fast linker, **sccache
off** (or deps-only). One Cargo command at a time against the single `target/`.

Loop:
```
edit → cargo check-game            # fast, no link
     → (repeat until it type-checks)
     → just screenshot <preset>    # one linked artifact when a frame is needed
     → read the PNG → iterate
```
This is the everyday Windows/WSL/macOS developer loop. It leans entirely on
incremental + dynamic linking + the fast linker.

### 7.2 Shape B — parallel agents on a cloud box
**Cold/parallel regime.** This is the throughput setup and the one your
description points at (headless agents, screenshot-driven).

- **Each agent gets its own git worktree** (`git worktree add`) with its **own
  `target/`** — so they don't serialize on the single target lock (§3.6). The
  Agent tool's `isolation: "worktree"` fits this directly.
- **All worktrees share one sccache** (`SCCACHE_DIR` on the box). The first
  worktree populates the cache building Bevy; every subsequent worktree links
  the dep graph from cache in seconds. This is what makes N parallel agents
  affordable — without it, each worktree recompiles Bevy from cold.
- **`CARGO_INCREMENTAL=0`** box-wide so sccache caches everything (§5.1).
- **Concurrency budget:** cap parallel builds at roughly `min(cores/2,
  RAM_GiB / 4)` — a Bevy link peaks at several GiB, so memory, not cores, is
  usually the ceiling. Oversubscribing thrashes and is slower than a lower cap.
- **Dynamic linking** still applies per-worktree for the renderer entry points.

Loop, per agent:
```
git worktree add ../wt-<task>      # isolated checkout + target dir
cargo check -p <crate>             # sccache-backed; deps are cache hits
just screenshot <preset>           # linked artifact; Bevy from cache
read PNG → iterate → hand back diff
```

Result: the expensive dep graph is compiled **once per box**, shared across
every agent and worktree; each agent only pays for its own changed crates.

### 7.3 Which box?
For agent iteration, favor a **Linux cloud box or WSL2** over native Windows:
you get mold, no Defender tax, trivial sccache, and headless Vulkan — and your
workload never needs an interactive window. Native Windows stays the right choice
when a human is doing interactive play-testing on the same machine.

---

## 8. Compiler backend policy

The exact stable toolchain uses LLVM everywhere. Do not add Cargo `unstable`
configuration, `codegen-backend`, or a Cranelift component to an agent or CI
environment. A future backend change requires an explicit cross-platform
decision, not a per-machine experiment.

---

## 9. Measuring
Don't guess — measure the loop you care about.

- **Timing a build:** `cargo build --workspace --timings` writes an HTML report
  (`target/cargo-timings/`) showing per-crate wall time and parallelism — finds
  the long pole.
- **Incremental touch-rebuild:** `touch crates/game/src/main.rs && time just
  screenshot hub` measures the real iterate-loop cost (edit one file → linked
  artifact).
- **sccache payoff:** `sccache --show-stats` before/after; the hit rate on a
  *second* worktree/branch is the number that matters (§5.4).
- **Cold graph:** `cargo clean && time cargo build --workspace` (with and
  without sccache warm) sizes the cold/parallel regime.

---

## 10. Cross-references
- [tooling.md](tooling.md) — toolchain policy, the dynamic-linking launch
  contract (INC-0008), env-var launch/window/vsync toggles, artifact layout.
- [visual_testing.md](visual_testing.md) — the headless screenshot A/B workflow
  agents verify with.
- `CLAUDE.md` (root) — the operating manual; the "Toolchain" and "Fast iteration
  invariants" sections point here.
- INC-0006 (`docs/incidents/`) — the `-Zthreads` parallel-MIR ICE.
- INC-0008 — the dynamic-linking dylib-search-path contract.
