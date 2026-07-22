# Local Cargo configuration

`config.toml` in this directory is intentionally local and ignored. Generate it
with `scripts/setup-build-env.ps1` on Windows or `scripts/setup-build-env.sh` on
Linux, WSL, and macOS.

For a WSL or remote-Linux parallel-agent box, create the complete worktree set
first, then provision every worktree in one pass:

```bash
scripts/setup-build-env.sh --agents <N> --all-worktrees
source scripts/sccache-on.sh
bash scripts/check-build-env.sh --parallel
```

The split is deliberate:

- portable profile policy lives in the root `Cargo.toml` and is inherited by
  every checkout and worktree;
- this directory contains machine-specific linker selection, the sccache
  wrapper, and a CPU budget per concurrent Cargo process;
- target-directory isolation is not configured globally. Each parallel agent
  uses its own worktree and therefore its own `target/`, while all agents share
  the same sccache store. Setup discovers every current `git worktree` root for
  path normalization; `--all-worktrees` writes the local config to each root and
  setup refreshes its own generated configs and the cache server after that set
  changes. A hand-written config is preserved unless `--force` is explicit;
- Linux linker config is generated for the actual `rustc` host triple, so both
  x86-64 and ARM64 boxes use clang + mold without a checked-in architecture
  assumption. WSL checkouts must live on the Linux filesystem, not `/mnt/c`.

Do not commit an absolute linker path or a machine-sized job count.
