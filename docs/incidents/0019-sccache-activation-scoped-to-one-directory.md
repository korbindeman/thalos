# INC-0019: sccache silently inactive in worktrees; job budget halved for the solo case

- **Status:** Fixed
- **Date:** 2026-07-22 (observed) / 2026-07-22 (fixed)
- **Severity:** perf (build iteration)
- **Surface:** every Cargo invocation — `just check`, `just build`, `just screenshot`; worst in agent worktrees

## Summary

Builds felt slower with the sccache setup than without it. sccache was in fact
working in the main checkout (52% hit rate), but two independent defects meant
the machinery was attached to the wrong lanes: the **job budget throttled the
main repo to half the CPUs** for a solo developer, and **sccache activation
lived in a gitignored per-directory file that only one directory had**, so
worktrees created outside the checkout built with no cache and no fast linker.
Neither failure produced any output. Fixed by moving activation to a
machine-global `RUSTC_WRAPPER`, defaulting the job budget to the whole machine,
and making the checker verify every worktree's cache normalization.

## Symptoms

- "The sccache stuff is silently failing (on worktrees? idk), slowing down
  iteration instead of speeding it up."
- No error, no warning, no missing artifact. Builds succeed; they are just cold
  and single-threaded-ish.
- `sccache --show-stats` looks *healthy* (52% hits), which actively misleads —
  the hits come from the one provisioned directory.

## Evidence

Job budget on a 16-thread machine:

```
.cargo/config.toml:  jobs = 8 # 16 logical CPUs / 2 expected concurrent agents
nproc:               16
```

Activation reachable from nowhere but the main checkout:

```
RUSTC_WRAPPER(User)=[]        RUSTC_WRAPPER(Machine)=[]
~/.cargo/config.toml          (none)
ancestor scan from C:/tmp/thalos-plumes → no .cargo/config.toml found
```

Normalization set vs. reality — 6 roots registered, 8 worktrees present:

```
FAIL  sccache does NOT normalize worktree: .../worktrees/rebase-terrain-main-e12835
FAIL  sccache does NOT normalize worktree: .../worktrees/terrain-status-f451c4
```

sccache's own accounting of what it *cannot* cache in the iterate lane:

```
Non-cacheable reasons:
crate-type   374     (proc-macros / dylib)
incremental  250     (profile.dev sets incremental = true)
```

## Hypotheses considered

1. **sccache daemon broken / cache corrupt** — ruled out: `--show-stats`
   responds, 52% hit rate, and the checker's real-rustc probe compiles twice and
   produces a genuine hit.
2. **sccache overhead is the slowdown** — ruled out as *primary*. Non-cacheable
   calls bail early and `Average cache read hit` is 0.014 s; the per-invocation
   tax is tens of ms and cannot explain the felt cost. It is real but marginal.
3. **Cache misses from cross-worktree path differences** — **confirmed, partial
   cause.** `SCCACHE_BASEDIRS` is a provisioning-time snapshot; two current
   worktrees were absent, so they hash absolute paths and can never hit.
4. **Worktrees not using sccache at all** — **confirmed, primary cause for
   worktrees.** Activation was only `build.rustc-wrapper` in a gitignored
   `.cargo/config.toml` written solely to the repo root.
5. **Job budget starving the build** — **confirmed, primary cause for the main
   repo.** `-AgentSlots 2` default → `jobs = 8` of 16, permanently, regardless of
   whether any second agent exists.

## Root cause

Two mechanisms, both silent:

**(a) Directory-scoped activation.** Cargo discovers `.cargo/config.toml` by
walking *up* from the cwd. Worktrees under `.claude/worktrees/*` therefore
inherited the repo-root config by accident, while `C:/tmp/thalos-*` and
`~/.codex/worktrees/*` — outside the tree, with no `~/.cargo/config.toml` and no
`RUSTC_WRAPPER` in the environment — resolved *no* wrapper and *no* linker
override. They built uncached with stock `link.exe`. Compounding it,
`setup-build-env.sh` (which has `--all-worktrees`) hard-exits on MinGW, and
`setup-build-env.ps1` had no equivalent flag: on Windows there was **no supported
way to provision a worktree at all**, while `docs/build_speed.md` §7.2 instructed
the operator to use the flag they could not run.

**(b) Snapshot normalization.** `SCCACHE_BASEDIRS` is what strips each checkout
root so two worktrees hash identically. It is captured once at provisioning and
read by the *server* at startup, so every subsequent `git worktree add` leaves a
worktree permanently outside the set until someone restarts the daemon. Nothing
hooked worktree creation, so the set decayed monotonically.

A third latent defect: `sccache-on.sh` and `setup-build-env.sh` joined the roots
with `:`. sccache splits on the *platform* separator (`;` on Windows), so under
git bash every `C:/…` root would shred into `C` plus a bogus path. Inert only
because the helper declined to restart a healthy server.

## Fix

- **Activation is now machine-global on Windows.** `setup-build-env.ps1` sets
  `RUSTC_WRAPPER` as a user environment variable; the generated
  `.cargo/config.toml` keeps only what genuinely differs per checkout (job
  budget, linker). A worktree anywhere on disk now gets the cache with zero
  provisioning. Linux/macOS keep the per-worktree config form, which
  `--all-worktrees` already handled.
- **Job budget defaults to the whole machine** (`-AgentSlots`/`--agents` default
  1). `jobs` is scheduling-only and never enters a fingerprint, so this is free.
- **`-SyncOnly`** added as the cheap post-`git worktree add` resync (recompute
  roots → user env → restart server), and **`-AllWorktrees`** added to the
  PowerShell script so Windows can finally provision worktrees.
- **Roots are sorted longest-first** in all four scripts, so a nested worktree
  matches before the repo root it lives inside.
- **Platform-correct separator** in the bash scripts.
- **The silence is gone.** `check-build-env.sh` now verifies *every* worktree is
  normalized (not just the current one), compares paths canonically so git-bash
  `/c/Users/…` matches sccache's `c:/users/…`, accepts either activation form,
  and both helpers warn when the live server's root set is stale. All
  daemon-restart paths refuse to run while `cargo`/`rustc`/`dx` is active.

## Prevention & recurrence signals

- **Standing rule** (added to CLAUDE.md "Fast iteration invariants" and
  `docs/build_speed.md` §5.0/§5.3.1): *build-environment activation must not
  depend on the current directory, and `SCCACHE_BASEDIRS` must be resynced after
  every `git worktree add`.*
- **Corollary worth remembering:** a healthy-looking `--show-stats` is not
  evidence the cache is working *for the build you care about*. Hit rate is
  aggregated across every directory that ever used the daemon.
- **Tells for a recurrence:**
  - `bash scripts/check-build-env.sh` reports `sccache does NOT normalize
    worktree: …` — the direct signal.
  - A fresh worktree's first `cargo check` takes cold-build time while
    `Compile requests` climbs and `Cache hits` does not.
  - `sccache --show-stats` → `Base directories` count < `git worktree list`
    count.
  - Build wall-clock roughly doubles with no code change → check `jobs` in
    `.cargo/config.toml` against `nproc`.
