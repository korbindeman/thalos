# Tooling

## Rust Toolchain

`rust-toolchain.toml` pins the workspace to nightly and does not request
platform-specific components. There is no checked-in Cargo backend override,
so codegen stays on Cargo's default LLVM backend for every platform.

Cranelift is not project policy right now. If a macOS developer wants it for
local iteration, they can install `rustc-codegen-cranelift-preview` and opt in
through personal Cargo config or one-off `cargo --config` flags. Do not check
that setup into the repo unless Thalos deliberately moves back to a
cross-platform Cranelift configuration.
