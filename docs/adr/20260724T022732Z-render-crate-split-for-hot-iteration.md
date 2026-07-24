# ADR-20260724T022732Z — Split render/material code into leaf crates for hot visual iteration

**Status:** accepted
**Date:** 2026-07-24

## Context

The visual-iteration loop (edit a shader/material → `just screenshot` → read the
PNG) is the core inner loop of the keystone graphics work. It should be snappy
for the self-contained edits it is dominated by: tweaking a shader body, a
lighting/palette constant, or adding one uniform knob.

The persistent capture lane is **already** built for this, and this ADR corrects
an earlier misreading of it:

- `just screenshot` → `tools/capture/src/main.rs::start_server` launches the host
  via `dx serve --hot-patch … --package thalos_capture_host --features
  dev-iteration`, into a separate `target/…/desktop-dev` target dir.
- `dev-iteration = ["bevy/hotpatching", "bevy/embedded_watcher", …]`
  (`crates/runtime/game/Cargo.toml`). `bevy/hotpatching` pulls `subsecond 0.7.9`;
  `dx 0.7.9` is installed.
- The host reports reload timestamps (`record_shader_reloads`,
  `record_code_hotpatches` reading `bevy::ecs::HotPatched` in `screenshot.rs`),
  and the client blocks in `wait_for_reloads` until the reload lands before
  shooting.

So two edit classes are already fast: **pure WGSL** (embedded_watcher reload,
zero compile) and **Rust function-body tweaks** (dx hot-patch, no rebuild). The
`--features dev-renderer` seen in the `justfile` is only the `-cold` authoritative
recipes, not this lane.

The remaining slowness is **structural Rust edits** — adding a uniform field to a
material struct, a new `Material`, a new system/plugin, a type/`derive` change.
`dx`/subsecond patches function bodies only; anything structural forces a full
rebuild + host restart + cold terrain re-stream, and because `dev-iteration`
trades away dynamic linking (its cost of enabling hot-patch) that rebuild
statically links Bevy into whichever crate changed.

The crux is that **`dx`'s rebuild/patch scope is per-crate**, and the render code
lives in two monster crates:

| Visual edit | Recompiles today | rlib |
|---|---|---|
| New material uniform / new material | `thalos_body_render` (`ground/` = 9,679 lines) | **186 MB** |
| Render-driver / system change | `thalos_runtime` (`rendering/` = 14,764 lines, in a 28,504-line top level) | **286 MB** |

The edit is self-contained; the *crate* is not, so the tooling cannot exploit the
self-containment. A "add a knob" edit almost always means "add a uniform field" =
structural = drags the whole enclosing monster.

## Decision

Carve the render/material code into leaf crates so a self-contained visual edit
recompiles a small unit, narrowing what `dx` must rebuild and raising the
hot-patch hit rate. Two seams, staged:

- **Seam A — a `thalos_body_*` leaf under `thalos_body_render`.** First landing
  slice: extract **`shading`** (SceneLighting / AtmosphereBlock / StarLight /
  SkyViewLut / multi_scatter + the `thalos::{lighting,atmosphere,shadow,
  landcover,water,foliage,grass_displace}` WGSL libraries and
  `PlanetLightingPlugin`) into `thalos_body_shading`. It is the clean bottom leaf
  — everything imports it, it imports ~nothing — so lighting/atmosphere-uniform
  and shared-library edits recompile ~1.2k lines instead of 186 MB, and it is the
  layer the materials sit on. Then peel individual material types onto it where
  they are clean (starting with `body_material`'s shared blocks).
- **Seam B — extract `crates/runtime/game/src/rendering`** (14,764 lines) out of
  `thalos_runtime` into its own crate, so render-*system* edits stop dragging the
  286 MB game crate. Staged after Seam A.

The **WGSL import graph does not block this**: material shaders import the shared
libraries by their naga_oil virtual names (`thalos::lighting`, declared via
`#define_import_path` inside the WGSL) and `thalos_udlod::*` by crate name —
neither changes when a `.wgsl` moves crates. The only path that changes is a
material's own `embedded://thalos_body_render/…` load string, which lives in the
same `.rs` file being moved, so a material + its `.wgsl` relocate as one unit.

### Honest ceiling

The split shrinks compile+link. It does **not** remove the *host restart* a
non-patchable edit triggers — the preset world reboots and re-streams terrain
(~15 s for a surface preset), which is inherent to a GPU-layout change. The
mitigation for that cost is keeping edits WGSL-first / function-body-shaped so
they stay on the truly-hot path; this ADR does not try to engineer world reuse
across a restart.

## Rejected alternatives

- **Do nothing; rely on the hot lane as-is.** Rejected: the hot lane already
  covers WGSL and function-body edits, but "add a uniform" (structural) is a
  common visual edit and falls off it onto a 186/286 MB rebuild.
- **Enable dynamic linking on the screenshot lane to cut the relink.** Rejected:
  `dynamic_linking` conflicts with `hotpatching` (the reason `dev-iteration`
  drops it); you cannot have both fingerprints in one graph, and hot-patch is the
  higher-value property.
- **Extract all `ground` materials into one leaf in a single move.** Rejected:
  the materials are not a clean leaf — `sky_material` is woven into the udlod tile
  components + a custom pipeline, `rock_material` needs `vegetation::GrassParams`.
  A one-shot move drags in vegetation and the tile machinery. `shading` is the
  only true bottom leaf; materials peel incrementally on top of it.

## Consequences

- New workspace crate(s): `thalos_body_shading` (Seam A), later a render-systems
  crate (Seam B). `thalos_body_render` re-exports the extracted symbols
  (`pub use thalos_body_shading::*` behind `shading`) so existing
  `thalos_body_render::{SceneLighting, …}` and `crate::shading::` paths keep
  resolving — no churn at the ~dozens of call sites during the move.
- The `dev-iteration` graph gains a crate boundary; the first build after the
  split is a one-time cold rebuild of that graph.
- Verification is by headless capture: `just check` green, then `just screenshot
  spaceport-aerial` (and a lighting-sensitive preset) to confirm the shared
  libraries still register under their virtual names and rendering is intact.

## Cross-references

- `docs/development/build_speed.md` — the two build regimes; the persistent
  `dev-iteration` hotpatch lane (§3.1).
- `docs/development/capture.md`, `docs/development/visual_testing.md` — the
  persistent capture host + client.
- Backlog: `BL-20260724T022732Z-body-shading-leaf` (Seam A first slice),
  `BL-20260724T022732Z-render-systems-crate` (Seam B).
