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

## Terrain bakes

Full-resolution terrain bakes are developer-local build artifacts. `just bake <body>` writes the game-loadable binary to `target/bakes/<body>.bin` and PNG inspection outputs to `stage-bakes/<body>/full/`; `just bake <body> --preview` writes only fast PNG previews to `stage-bakes/<body>/preview/`.

Do not track terrain bakes in Git or Git LFS. Developers should bake their own local maps, and release/distribution assets will use a separate pipeline when that exists. Git LFS may still be useful later for actual authored assets, but not for generated terrain bakes.

## `bevy_erosion_filter` Shader Source

`thalos_bake_dump` uses `bevy_erosion_filter` from crates.io and needs the
erosion compute WGSL as a raw `&str` because the bake CLI runs `wgpu` directly,
outside Bevy's asset/shader loader.

`bevy_erosion_filter` 0.1.2 ships its WGSL source as the public
`EROSION_WGSL: &'static str` constant, available with `default-features = false`
so the bake CLI doesn't pull in Bevy. `crates/bake_dump/src/gpu.rs` imports it
directly and strips the one `#define_import_path` naga_oil directive before
handing the source to `wgpu`.
