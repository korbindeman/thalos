# Local Cargo configuration

`config.toml` in this directory is intentionally local and ignored. Generate it
with `scripts/setup-build-env.ps1` on Windows or `scripts/setup-build-env.sh` on
Linux, WSL, and macOS.

The split is deliberate:

- portable profile policy lives in the root `Cargo.toml` and is inherited by
  every checkout and worktree;
- this directory contains machine-specific linker selection, the sccache
  wrapper, and a CPU budget per concurrent Cargo process;
- target-directory isolation is not configured globally. Each parallel agent
  uses its own worktree and therefore its own `target/`, while all agents share
  the same sccache store. Setup discovers every current `git worktree` root for
  path normalization; restart sccache after that set changes.

Do not commit an absolute linker path or a machine-sized job count.
