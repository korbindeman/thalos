# Build speed & the agent build workflow

Canonical guide for making Thalos compile fast on every platform an agent (or a
human) drives it from: **Windows native**, **Windows + WSL2**, **macOS**, and a
**headless Linux cloud box**. Also the home for the **two build regimes** (fast
local iteration vs. parallel/clean throughput).

This supersedes the per-platform "fast incremental loop" recipes that used to
live in [tooling.md](tooling.md); tooling.md now points here. The invariants it
still owns (dynamic-linking launch contract, one-Cargo-at-a-time, feature
fingerprint, env-var launch toggles) are referenced below, not duplicated.

The rule from `CLAUDE.md` stands: portable profile policy belongs in committed
`Cargo.toml`; machine-specific linkers and CPU budgets belong in the gitignored
`.cargo/config.toml`; launcher overrides belong in `.env.just`.
See ADR-20260721T212438Z-portable-build-policy-local-acceleration.

> **There is no compiler cache.** sccache was removed 2026-07-23
> (ADR-20260723T222214Z-abandon-sccache): it produced corrupt builds and carried
> a large silent-misactivation surface (INC-0019) for a benefit that was marginal
> on the solo iterate loop. Build speed now rests entirely on the fast linker,
> dynamic linking, incremental compilation, and per-process job budgets — §5.

---

## 1. TL;DR

If you do nothing else:

- **Every platform:** keep Bevy `dynamic_linking` on for every dev renderer —
  the normal game, cold capture, temporary preview examples, **and** the
  persistent `just screenshot` host all share the one `dev-renderer`
  fingerprint (ADR-20260724T153619Z). Use a **fast linker** (rust-lld on
  Windows, **mold** on Linux/WSL, the default new ld on macOS), keep
  **incremental on**, and on Windows **exclude `target/` from Defender**.
- **Run `scripts/setup-build-env.sh`** (Linux/macOS/WSL) or
  `scripts/setup-build-env.ps1` (Windows). They install the fast linker + agent
  tools and write a local config with the fast linker and a per-process Cargo
  job budget. Pass `--agents N` / `-AgentSlots N` for expected concurrency. On
  WSL/Linux agent boxes, create all worktrees first and add `--all-worktrees`,
  then validate with `bash scripts/check-build-env.sh --parallel`.
- **For parallel agents / a cloud box / fresh worktrees:** give every agent a
  separate worktree/`target/`. Each worktree compiles the Bevy dep graph cold
  the first time — there is no shared cache to skip it — so size the box for that
  cold build (cores + RAM) and reuse worktrees rather than recreating them.

The highest-leverage single move for **agent iteration specifically** — because
it is headless (screenshot tool, no interactive window) — is to run the loop on
**Linux (WSL2 or a cloud box) with mold + dynamic linking**, project on the
Linux filesystem. Your headless requirement removes the only real reason
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
feature fingerprint, or a `cargo clean`. Nothing skips it — there is no compiler
cache (§5); **more cores** are what speed it up (deps compile in parallel across
crates), and **incremental** reuse keeps you from paying it again for unchanged
crates within a worktree.

So there are two regimes, and the ideal config differs between them:

| Regime | You are… | Bottleneck | Win with |
|--------|----------|-----------|----------|
| **Iterate** | Editing one crate, rebuilding + screenshotting on one machine | Link + one-crate codegen (single-threaded) | Fast linker, dynamic linking, **incremental on** |
| **Cold / parallel** | Fresh worktree, cloud box, N agents in parallel, CI | The whole dep graph | Cores, RAM; **incremental on** (per-worktree reuse) |

Keep **incremental on in both regimes.** It was previously turned off for the
cold/parallel regime only to let sccache cache workspace crates across
worktrees; with no cache that trade no longer exists, and incremental reuse is a
straight win within each worktree.

---

## 3. The optimization layers

Stack these; they are orthogonal except where noted.

### 3.1 Two renderer fingerprints — committed policy
`just game`, `just screenshot-cold`, and the temporary preview examples use the
normal dynamic renderer fingerprint. Bevy links once into a shared `bevy_dylib`;
subsequent edits relink only Thalos crates. Do **not** add it to release paths
(`just build`, `just trace`, tests, bakes) — a shipped binary with a missing
`bevy_dylib` crashes. Any tool that launches a dev renderer directly must
reproduce Cargo's dylib search-path contract (profile dir + `deps` +
`rustc --print target-libdir` onto `PATH`/`DYLD_FALLBACK_LIBRARY_PATH`/
`LD_LIBRARY_PATH`) — see INC-0008 and tooling.md.

The persistent `just screenshot` / `just compare` host runs on the **same**
`dev-renderer` fingerprint, spawned as a detached
`cargo run -p thalos_capture_host --features dev-renderer` by the capture
client (ADR-20260724T153619Z-retire-hotpatch-single-stable-capture-lane).
`dev-renderer` carries `bevy/embedded_watcher`, so file-backed *and*
crate-embedded WGSL hot-reload inside the running host (~3 s from save to a
fresh PNG) — and inside the interactive game.

**There is no Rust hot-patching.** dx/subsecond was retired 2026-07-24 after
an applied patch reproducibly stack-overflowed the app
(INC-20260724T044418Z-subsecond-patch-stack-overflow; the whole repair story:
INC-20260724T030400Z, INC-20260724T040523Z). Instead the client compares
workspace `.rs`/`.toml` mtimes against the running host's launch time on every
shot: a stale host is stopped, rebuilt (dynamic relink), and relaunched
automatically (~1.5–2.5 min warm — the disk tile cache keeps the terrain
re-stream short). A Rust edit can never leave a crashed or silently stale
server behind, and `just capture-stop` is optional hygiene. Keep visual
iteration WGSL-first; that is the hot path.

Amortize the boot floor when evidence needs several framings:
`just capture <preset>...` sends the batch through one controller. Presets with
the same target body, spawn scenario, hub mode, viewport, and startup override
fingerprint reuse the booted world/GPU; only a real boot-context boundary
restarts. The controller groups an interleaved scene list by the compatibility
set reported from each booted host, so caller ordering cannot accidentally pay
the same boot twice. Comparisons are a
subcommand of the same binary and send every live-compatible variant to that
host. This is the safe startup win: fewer boots, with no new compiler backend,
runtime patcher, or feature fingerprint.

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
Committed `[profile.dev] incremental = true` makes one-crate edits cheap.
`debug = "line-tables-only"` (both `dev` and `dev.package."*"`) keeps
backtraces while cutting debuginfo generation and link size — a real chunk of
link time. These portable defaults live in `Cargo.toml`, not local config.

### 3.4 opt-level split — already committed
`Cargo.toml` sets `[profile.dev] opt-level = 1` (your code stays cheap to
compile) and `[profile.dev.package."*"] opt-level = 3` (deps run fast). Tradeoff:
the *first* cold dep build is slower because deps compile optimized. With no
compiler cache (§5) that cost is paid once per fresh worktree / `cargo clean`;
incremental reuse then keeps it from recurring within a worktree.

### 3.5 No compiler cache
Thalos does **not** use a compiler cache. sccache was removed 2026-07-23
(ADR-20260723T222214Z-abandon-sccache) after it produced corrupt builds and,
per INC-0019, proved chronically silent-fragile (directory-scoped activation,
a `SCCACHE_BASEDIRS` snapshot that decayed on every `git worktree add`, a
platform-separator footgun, a version floor). Its only real payoff was sharing
the cold dep-graph build across parallel worktrees; that regime is not how the
project runs day to day, and on the solo iterate loop incremental already makes
workspace crates non-cacheable. If cross-machine cold-build throughput becomes a
priority again, evaluate a **remote** cache backend (S3/GCS/Redis) for
correctness first under a fresh ADR — do not restore the local-daemon design.

The GitHub release workflow does cache Cargo's already-compiled **dependency
artifacts** with `Swatinem/rust-cache`; that is deliberately not a compiler
cache. There is no `RUSTC_WRAPPER`, daemon, cross-worktree path normalization,
or object-level lookup. Cargo still validates every restored artifact against
the pinned compiler, manifests, lockfile, target, flags, and feature selection,
and a miss is an ordinary clean build. The action prunes workspace crates and
incremental artifacts before saving, keeping the cache focused on the expensive
Bevy/wgpu dependency graph (ADR-20260802T232314Z-release-cache-default-branch).

### 3.6 One Cargo at a time; one feature fingerprint
From `CLAUDE.md`/tooling.md, still load-bearing:

- Run **one Cargo command at a time** against the workspace `target/`.
  Concurrent `game`/`check`/`screenshot` invocations serialize on the target
  lock while competing for CPU and several GiB of compiler memory. Use
  `just check` while editing, then **one** linked `just game`/`just
  screenshot` when an artifact is actually needed. (For *genuinely* parallel
  agents, give each its own worktree + target dir — §7.2 — which sidesteps the
  single-target-lock contention.)
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
- **No hand-rolled `cargo clean -p <subset>`.** The dev lane links Bevy
  dynamically, so `bevy_dylib` and every crate that links against it form **one
  artifact set**. Cleaning part of that set while `profile.dev`'s incremental
  caches survive leaves codegen units referencing the *old* dylib's
  LLVM-internalized symbols, and the next link dies with a wall of
  `undefined symbol: anon.<hash>.N.llvm.<hash>` (INC-20260724T182642Z). Note the
  trap: this is the failure mode of the "clear the stale artifacts" reflex, so
  it tends to strike precisely when someone is already fighting the lane.
  - **Reading it:** an undefined symbol containing `anon.` or `.llvm.`, referred
    to from a workspace rlib, is an *artifact* inconsistency — never a missing
    dependency, feature, or `extern`. Do not go looking for the "missing crate".
  - **Recovering:** the capture client self-heals this once (drops
    `target/debug/incremental`, rebuilds, retries) before it ever reaches you.
    If it reports the retry failed too, run `just build-reset`
    (`thalos_capture reset`) — stop the host, drop the incremental cache, and
    clean the dynamic-linking crate set together.
  - Note the same symptom shape appears in INC-0006 (`-Zthreads` ICE leaving
    unlinkable incremental objects): "missing LLVM symbols at link" is this
    project's standard tell for a poisoned build tree, from either cause.

### 3.8 Windows Defender exclusions (Windows only, big win)
Real-time AV scanning of every `.o`/`.rlib`/`.pdb`/`.exe` in `target/` silently
taxes every build. Excluding the build dirs is often a 20–40% incremental-build
improvement on Windows. Requires an elevated shell:

```powershell
Add-MpPreference -ExclusionPath "C:\Users\korbi\Documents\thalos\target"
Add-MpPreference -ExclusionPath "$env:USERPROFILE\.cargo"
Add-MpPreference -ExclusionPath "$env:USERPROFILE\.rustup"
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
- **WSL2:** WSLg exposes the host GPU (`/dev/dxg` → Mesa). Distros that ship
  Mesa's `dzn` Vulkan-on-D3D12 ICD get real-GPU Vulkan; Ubuntu 24.04 does
  **not** (only `lvp`/llvmpipe software Vulkan). The Mesa D3D12 **GL** bridge
  does reach the host GPU, and both dev lanes compile wgpu's GL backend for it
  (optional direct `bevy_render = { features = ["gles"] }` in
  `crates/runtime/game/Cargo.toml`, 2026-07-22 — the top-level `bevy` crate
  does not re-export that feature), **but the full renderer cannot run on GL**:
  the terrain height/albedo atlas needs `TEXTURE_FORMAT_16BIT_NORM` and the
  `VIEW_FORMATS` downlevel flag, which wgpu-GL does not expose (verified
  2026-07-22, `Device::create_texture 'height_attachment'` validation errors).
  On such distros use **llvmpipe software Vulkan** (`THALOS_WGPU_BACKEND=vulkan`
  with only the `lvp` ICD installed): conformant and visually correct, slow
  frames, and **GPU-time budget numbers are meaningless there** — measure
  budgets on the native/RTX side. Player/release builds keep the default
  backend set.

`THALOS_WGPU_BACKEND` (`auto|dx12|vulkan|metal|gl`) selects the backend without
touching source — see tooling.md.

---

## 5. Compiler cache — removed

Thalos used sccache to turn a cold Bevy dep-graph build into a cache hit. It was
**removed 2026-07-23** (ADR-20260723T222214Z-abandon-sccache) and is not coming
back in its previous form.

**Why it's gone:**

- **It produced corrupt builds** — cache-attributable build errors, not source
  errors. A cache that makes a build *wrong* is worse than one that is merely
  slow.
- **It was chronically silent-fragile.** INC-0019 documents the failure modes:
  activation was directory-scoped (worktrees outside the checkout built
  uncached), `SCCACHE_BASEDIRS` was a provisioning-time snapshot that decayed on
  every `git worktree add`, the roots separator differed by platform (`:` vs
  `;`), and a hard ≥ 0.14.0 version floor. Keeping it working meant a global
  `RUSTC_WRAPPER`, longest-first root sorting, canonical path comparison in the
  checker, resync-after-every-worktree-add — pure tax.
- **The payoff was marginal here.** On the solo iterate loop `profile.dev` sets
  `incremental = true`, so every workspace crate is non-cacheable by
  construction; sccache only cached registry deps on a cold build. The one
  regime where it genuinely paid — a cache shared across parallel worktrees — is
  not how the project runs day to day.

**What replaced it:** nothing — build speed rests on the fast linker (§3.2),
dynamic linking (§3.1), incremental + trimmed debug info (§3.3), and per-process
job budgets (§6). The setup scripts no longer install or configure sccache;
`scripts/setup-build-env.ps1` additionally **clears** any stale sccache user
environment variables (`RUSTC_WRAPPER`, `SCCACHE_*`) so re-running it fully
de-sccaches an already-provisioned Windows box. The former `scripts/sccache-on.*`
regime toggles are deleted, and `check-build-env.sh` no longer probes a cache.

**If cold/parallel throughput becomes the bottleneck again:** revisit with a
**remote** cache backend (S3/GCS/Redis), evaluated for correctness first, under a
new ADR — not a return to the local-daemon + basedirs design this section
removes.

---

## 6. Per-platform setup

Run the setup script for your platform; the manual config it writes is below for
reference. Machine-local settings land in the **gitignored**
`.cargo/config.toml`; portable profile settings remain in `Cargo.toml`.

Setup defaults to **one** concurrent Cargo process (the whole machine for the
solo case). It divides the machine's logical CPU count between the expected
slots; override the slot count when provisioning a larger agent box.

### 6.1 Windows (native)
`rust-lld` + a bounded job count, with the linker provisioned as an
**`lld-link.exe` shim** — a plain copy of the toolchain's `rust-lld.exe` at
`~/.cargo/shims/lld-link.exe` (the setup script creates and refreshes it). The
name matters: lld dispatches its driver flavor on argv[0], and any tool that
invokes the configured linker directly with raw MSVC-style args (as the
retired dx lane did) dies on bare `rust-lld.exe` with "lld is a generic
driver" (INC-20260724T030400Z-dx-linker-rust-lld-generic-driver); `rustc`'s
own invocations accept either spelling, so the shim is the safe spelling. For a 16-thread machine used by one Cargo
process at a time:

```toml
# No compiler cache (sccache removed, §5).
[build]
jobs = 16   # logical CPUs / expected concurrent Cargo processes; 16 = solo

[target.x86_64-pc-windows-msvc]
# Copy of rust-lld.exe; the lld-link.exe NAME selects the MSVC driver flavor.
linker = "C:/Users/korbi/.cargo/shims/lld-link.exe"

```

Changing the configured linker path invalidates Cargo fingerprints for the
whole graph — budget one full rebuild per lane (normal + desktop-dev) after
re-provisioning.

`jobs` is scheduling only — it never enters a fingerprint, so changing it costs
nothing. **Provisioning defaults to one Cargo process** (`jobs` = all logical
CPUs), because the common case is working alone. Pass `-AgentSlots N` /
`--agents N` when you actually intend N concurrent Cargo processes.

Then: Defender exclusions (§3.8). `just check` while editing; one
`just game`/`just screenshot` when you need a frame.

### 6.2 Windows + WSL2 (recommended for agent iteration)
WSL2 gives you the Linux toolchain (mold) with none of the display headaches,
because iteration is headless. Setup:

1. `wsl --install` (Ubuntu). Inside WSL: install rustup; the repository selects the pinned stable release,
   `just`, `clang`, `mold`, `mesa-vulkan-drivers`.
2. **Keep the project on the Linux filesystem** — clone into `~/thalos`, **not**
   `/mnt/c/...`. Building across the `/mnt/c` 9p mount is drastically slower and
   erases most of the gains. This is the #1 WSL mistake.
3. Create the complete worktree set, then run
   `scripts/setup-build-env.sh --agents <N> --all-worktrees`. It writes
   host-specific Linux Cargo config (mold via clang) and installs the
   `just` agent commands.
4. In each agent launcher, run `bash scripts/check-build-env.sh --parallel`
   before accepting work.
5. Headless rendering: WSLg usually gives real-GPU Vulkan; lavapipe is the
   fallback (§4).

You can share one Windows checkout by cloning the repo separately inside WSL, or
work entirely in WSL and push/pull. Do not build the same directory from both
Windows and WSL — the `target/` artifacts are not cross-compatible.

### 6.3 macOS
The default new linker is fast; local config only needs the job budget:

```toml
[build]
jobs = 4
```

If a macOS toolchain hits stale `.llvm.<hash>` anonymous-symbol references
between incremental objects, set `CARGO_INCREMENTAL=0` for that shell rather
than touching the workspace profile. Metal is the default wgpu backend; the
screenshot tool renders headlessly through it.

### 6.4 Linux cloud box (recommended for parallel agents)
The throughput setup. mold + a bounded job budget, real-GPU Vulkan if the box
has a GPU, lavapipe otherwise.

```toml
# .cargo/config.toml
[build]
jobs = 8

[target.<rustc-host-triple>]
linker = "/absolute/path/to/clang"
rustflags = ["-C", "link-arg=-fuse-ld=mold"]

```

Provision: `clang`, `mold`, `just`, the GPU Vulkan driver (or
`mesa-vulkan-drivers` for lavapipe), and rustup (the repository selects the
pinned stable release). Size the box for the dep graph: many cores help the cold
build, and **RAM matters** (linking is memory-hungry — budget several GiB per
concurrent cargo; see §7.2 for how many agents to run). Each worktree builds the
dep graph cold the first time — there is no shared cache — so favor reusing
worktrees over recreating them.

---

## 7. The ideal agent build workflow

Principles first, then the two concrete shapes.

**Principles (both shapes):**
- **Check before you link.** `just check` (or `cargo check -p <crate>`)
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
**Iterate regime.** Incremental on, dynamic linking on, fast linker. One Cargo
command at a time against the single `target/`.

Loop:
```
edit → just check                  # fast, no link
     → (repeat until it type-checks)
     → just screenshot <preset>    # one linked artifact when a frame is needed
     → read the PNG → iterate
```
This is the everyday Windows/WSL/macOS developer loop. It combines incremental
Thalos crates, dynamic linking, and the fast linker.

### 7.2 Shape B — parallel agents on a cloud box
**Cold/parallel regime.** This is the throughput setup and the one your
description points at (headless agents, screenshot-driven).

- **Each agent gets its own git worktree** (`git worktree add`) with its **own
  `target/`** — so they don't serialize on the single target lock (§3.6). The
  Agent tool's `isolation: "worktree"` fits this directly.
- Create the full worktree set before launching agents, then run
  `scripts/setup-build-env.sh --agents <N> --all-worktrees` once from the
  coordinating checkout. It writes the same machine-local config (fast linker +
  job budget) into every root. If a worktree is added later, rerun that command
  before assigning it work.
- **There is no shared compiler cache.** Each worktree compiles the Bevy dep
  graph cold the first time (§5), then reuses it incrementally within that
  worktree. Recreating a worktree throws that away — reuse worktrees where you
  can, and prune deliberately.
- **Keep incremental on** in each worktree — per-worktree reuse is a straight
  win now that no cache depends on it being off.
- **Concurrency budget:** cap parallel builds at roughly `min(cores/2,
  RAM_GiB / 4)` — a Bevy link peaks at several GiB, so memory, not cores, is
  usually the ceiling. Oversubscribing thrashes and is slower than a lower cap,
  and it competes for the cold dep-graph build every worktree pays independently.
- **Dynamic linking** still applies per-worktree for the renderer entry points.
- **Each worktree runs its own persistent capture host** (`just screenshot`
  spawns a per-worktree `cargo run` host; the state/log files live under that
  worktree's `artifacts/diagnostics/`). Headless off-screen renderers coexist
  on one GPU — budget roughly 1–2 GiB VRAM per live host and prefer stopping
  hosts (`just capture-stop`) in worktrees that are between tasks. The
  `lld-link.exe` shim is machine-global; the per-worktree `.cargo/config.toml`
  is what `-AllWorktrees` provisions.
- **Cold budget per worktree (measured 2026-07-24, §9):** ~7 min from empty
  lane to first PNG, then warm iteration (WGSL ~3 s, Rust rebuild ~1.5–2.5
  min). Two to three concurrent agents is the sweet spot on a 16-thread /
  single-GPU box — beyond that they contend on cold builds and VRAM.

Provision once from the coordinating checkout:
```
git worktree add ../wt-<task>      # isolated checkout + target dir
scripts/setup-build-env.sh --agents <N> --all-worktrees
```

Then, per agent:
```
cd ../wt-<task>
bash scripts/check-build-env.sh --parallel
cargo check -p <crate>             # deps compile cold once per worktree
just screenshot <preset>           # linked artifact
read PNG → iterate → hand back diff
```

Result: the expensive dep graph is compiled **once per worktree** (cold), then
reused incrementally within it. There is no cross-worktree sharing, so the first
build in each worktree is the price of admission — budget the box for it.

### 7.3 Which box?
For agent iteration, favor a **Linux cloud box or WSL2** over native Windows:
you get mold, no Defender tax, and headless Vulkan — and your workload never
needs an interactive window. Native Windows stays the right choice when a human
is doing interactive play-testing on the same machine.

### 7.4 Feature-selectable distribution builds

`.github/workflows/build-game.yml` is the release/distribution entry point. A
manual **Build game** run builds Windows x64 from the selected ref and accepts:

- `cargo_features`: comma-separated `thalos_game` feature names; canonical
  pre-alpha value `neural-terrain-default`;
- `use_default_features`: whether Cargo defaults join that explicit list.

`release_ref` and `release_sha` are internal dispatcher inputs; leave them
empty for manual builds. A `v*` tag starts
`.github/workflows/dispatch-game-release.yml`, which dispatches **Build game**
on `main` with both the immutable tag ref and its exact commit. The build checks
out that ref, rejects a SHA mismatch, and uses the verified SHA for package
provenance. Running the expensive job on `main` is load-bearing: GitHub Actions
caches are scoped to a branch or tag, and one release tag cannot restore a
previous tag's cache. The separate dispatch gives all releases the shared
default-branch cache scope while still building the tagged source.

The canonical tagged build never inherits an implicit workspace state: it uses
`--no-default-features --features neural-terrain-default`, builds with the
repository-pinned Rust 1.97.0 toolchain, and publishes a GitHub prerelease after
both Windows x64 and macOS arm64 artifacts succeed. The Windows build links the
MSVC C runtime statically, so the portable ZIP does not require a separately
installed Visual C++ redistributable. Before compiling, each platform restores
only Cargo registry data and dependency build artifacts keyed by target,
toolchain, compiler environment, manifests, lockfile, and requested features;
workspace crates are always rebuilt. Manual runs upload the Windows ZIP for 14
days and do not publish a release.

Terrain's build contract is capability-first:

| Cargo selection | Learned code/content | Session default |
|---|---|---|
| *(none, no defaults)* | no | procedural |
| `neural-terrain` | yes | procedural |
| `neural-terrain-default` | yes | neural |

`THALOS_TERRAIN=procedural` or `neural` remains a runtime A/B override, but it
cannot request a capability omitted at build time. When neural terrain becomes
the ordinary game default, `apps/game/Cargo.toml` can add
`neural-terrain-default` to its `default` list; procedural artifacts remain
reproducible with `--no-default-features`.

`scripts/package-game.ps1` copies only content supported by the built binary,
writes `BUILD_INFO.txt`, includes the licensing files, creates the ZIP and
SHA-256 file, extracts the ZIP, then launches `thalos_game.exe --verify-install`
from an unrelated empty directory. That non-rendering gate proves all runtime
content resolves beside the executable and verifies the LFS-backed neural
rasters against their sidecar dimensions and SHA-256 hashes. It does not replace
the user's Windows/GPU play test; a newly landed package stays `verify` until it
has launched and rendered on the target machine.

Windows compilation and packaging are separate jobs joined by a short-lived
executable artifact. If archive verification fails, **rerun failed jobs** repeats
only the LFS checkout/package gate; it does not rebuild the release graph. The
split was added after the first `v0.1.0` attempt spent 28 minutes compiling and
then exposed a platform-dependent terrain content key
(`INC-20260802T180726Z`).

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
- **Incremental touch-rebuild:** `touch crates/runtime/game/src/lib.rs && time just
  screenshot hub` measures the real iterate-loop cost (edit one file → linked
  artifact).
- **Cold graph:** `cargo clean && time cargo build --workspace` sizes the
  cold/parallel regime — the price each fresh worktree pays.

### Measured baseline — 2026-07-24, native Windows, 16 threads, RTX GPU

Taken immediately after the 2026-07-24 lane repair, `spaceport-aerial`
preset, warm disk tile cache. (The Rust-loop and cold rows were measured on
the since-retired static `dev-iteration` lane; the dynamic lane replaces its
11–38 s fat link with a dylib relink and loads a far smaller exe — re-measure
and update after the first sessions on ADR-20260724T153619Z's lane.)

| Loop | Measured |
|---|---|
| `cargo check -p thalos_game`, warm, no-op | ~10–12 s |
| `cargo check` after touching one crate (any layer) | 2–12 s |
| Warm no-op `just screenshot` (host alive) | 2.4–7.6 s |
| **WGSL edit → fresh PNG** (embedded_watcher) | **2.6 s** |
| **Rust edit → auto-restart → PNG** (the Rust loop) | **~95 s** on the static lane (~30 s compile + 12 s fat link + ~50 s boot/stream/capture); expected lower on the dynamic lane |
| Cold lane (full dep graph + workspace + link + boot) | **7 m 03 s** (static lane) |

Where the ~50 s boot/stream/capture goes (from the boot log): ~12 s process +
engine + GPU init (the 228 MB static exe was seconds of that; the dynamic exe
is much smaller), ~4 s cloud fill-calibration Monte-Carlo, ~4 s ocean spectra
authoring (two bodies), ~7 s terrain streaming to the settle gate (plus one
avoidable flatten-triggered terrain respawn), ~3 s warmup frames, 0.15 s
capture+encode. The LUT/ocean/flatten items are tracked as boot-time backlog
rows.

Parallel-agent budgeting: each fresh worktree pays the cold lane once, then
iterates at the warm numbers above; reuse worktrees (§7.2). The world
boot+stream floor is ~25–50 s with a warm disk tile cache and up to ~15 s
*per body surface* extra when the cache is cold (see the cold-streaming note
in CLAUDE.md).

---

## 10. Cross-references
- [tooling.md](tooling.md) — toolchain policy, the dynamic-linking launch
  contract (INC-0008), env-var launch/window/vsync toggles, artifact layout.
- [visual_testing.md](visual_testing.md) — the headless screenshot A/B workflow
  agents verify with.
- `CLAUDE.md` (root) — the operating manual; its "Build & iteration" section
  carries the load-bearing subset and points here.
- ADR-20260723T222214Z-abandon-sccache — why the compiler cache was removed.
- ADR-20260802T232314Z-release-cache-default-branch — why tagged releases
  dispatch their build on `main`, and what the CI cache may contain.
- ADR-20260721T212438Z-portable-build-policy-local-acceleration — the portable
  profile / local-config split.
- INC-0006 (`docs/incidents/`) — the `-Zthreads` parallel-MIR ICE.
- INC-0008 — the dynamic-linking dylib-search-path contract.
- INC-0019 — the sccache silent-misactivation post-mortem (historical; the tool
  it fixed is now removed).
