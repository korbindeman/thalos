# Tooling

Thalos keeps committed Cargo and Rust toolchain configuration deliberately
plain. `rust-toolchain.toml` pins nightly, and `Cargo.toml` sets only shared
profile choices that are expected to be sane across Windows, macOS, and Linux.

## Local compiler tuning

Platform-specific compiler and linker speedups belong in local Cargo config,
not in committed project config. Use either a personal Cargo config under your
home directory or the workspace-local `.cargo/config.toml`; the workspace file
is ignored by Git for this purpose. The `just game` command can also be
customized locally with `.env.just`, which is ignored by Git.

This includes:

- `CARGO_INCREMENTAL` overrides for platform-specific incremental behavior.
- Debug-info reductions for local iteration.
- Bevy dynamic-linking aliases.
- Local linker or backend experiments.
- `rustc-codegen-cranelift-preview` / `codegen-backend = "cranelift"`.

Do not commit a Cargo backend override unless the project intentionally adopts
that backend for all supported platforms. The default checked-in backend is the
Rust toolchain's normal LLVM backend.

### Windows fast incremental loop

A good Windows-local starting point is:

```toml
[env]
CARGO_INCREMENTAL = "1"

[profile.dev]
incremental = true
debug = "line-tables-only"

[profile.dev.package."*"]
debug = "line-tables-only"

[target.x86_64-pc-windows-msvc]
linker = "rust-lld.exe"

[alias]
check-game = "check -p thalos_game"
```

Then set the local `just game` command in `.env.just`:

```dotenv
THALOS_GAME_COMMAND="cargo run -p thalos_game --features bevy/dynamic_linking"
```

Use `just game` as the single app path on every platform. On this Windows
machine, `.env.just` keeps the compiler backend on LLVM, uses LLD for faster
MSVC-target linking through Cargo config, and enables Bevy's `dynamic_linking`
feature only for the local dev run. Release commands stay on the checked-in
defaults because neither `Cargo.toml` nor the default `just game` command
enables `bevy/dynamic_linking`.

Use `cargo check-game` for fast type checking when no app launch is needed.
Avoid adding a second local run alias unless the default Windows path changes
deliberately.

If `rust-lld.exe` is not on `PATH`, either install/update the Rust LLVM tools
for the active toolchain or use the absolute path to the toolchain copy in
local config. On Windows that copy usually lives under:

```text
%USERPROFILE%\.rustup\toolchains\<toolchain>\lib\rustlib\x86_64-pc-windows-msvc\bin\rust-lld.exe
```

### macOS incremental workaround

If a macOS toolchain hits stale `.llvm.<hash>` anonymous symbol references
between incremental codegen objects, disable incremental compilation locally
instead of changing the workspace profile:

```toml
[profile.dev]
incremental = false
```

macOS developers who want Cranelift for local iteration can also configure it
locally or pass one-off `cargo --config` flags. Keep that opt-in local so
Windows and Linux continue to use LLVM unless the project makes a deliberate
cross-platform backend decision.
