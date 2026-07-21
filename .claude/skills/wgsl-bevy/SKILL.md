---
name: wgsl-bevy
description: WGSL shader gotchas for Bevy / naga in this project. Use when writing or debugging .wgsl shaders, or when a WGSL/naga compile error appears (reserved-word identifiers, type-conversion strictness, naga_oil #import issues). This is a living list — append a new entry whenever a WGSL error is worth remembering.
---

# WGSL (Bevy / naga) pitfalls

A running list of WGSL traps we've hit in Thalos shaders so we don't
rediscover them. Bevy compiles WGSL through **naga** (with `naga_oil`
preprocessing), so both standard-WGSL rules and Bevy-specific quirks
apply.

## How to maintain this skill

When you run into a WGSL error worth remembering — a keyword you couldn't
use as a variable name, a confusing naga error message, a Bevy-specific
gotcha — add an entry below. Keep each entry short: the symptom (error
text if useful), the cause, and the fix. Prefer concrete cases over
general advice.

## Pitfalls

### Reserved words can't be used as identifiers

WGSL reserves a set of words — both current keywords and a forward-looking
"reserved words" list kept for future language versions. naga rejects them
as variable / function / member names even when they read like ordinary
names. Symptom is usually an "expected identifier" or "reserved word"
parse error pointing at the declaration.

Fix: rename the identifier (e.g. add a domain prefix/suffix). When you hit
a specific one, **record it here** so the next person recognizes it
immediately:

- `patch` — hit July 2026 in `gpu_grass.wgsl` (`let patch = …` → ``name `patch` is a reserved keyword``; the whole grass pipeline silently stopped rendering). Renamed to `mottle`.
- `meta` — hit July 2026 in `body_terrain.wgsl` (struct field `meta: vec4<f32>` → ``name `meta` is a reserved keyword``; the terrain pipeline failed to build and the whole ground vanished while everything else kept rendering). Renamed to `header`. Uniform structs match the Rust mirror by declaration order, not name, so renaming only the WGSL side is safe.
- `macro` — hit July 2026 in `clouds_compute.wgsl` (`let macro = …` → ``name `macro` is a reserved keyword``; both cloud compute pipelines failed at runtime even though Rust compiled). Renamed to `macro_noise`.

### Strict numeric typing — no implicit conversions

WGSL does not implicitly convert between numeric types. `let x: f32 = 1;`
is an error — the literal `1` is an integer. Use `1.0` for floats, and
cast explicitly across types: `f32(i)`, `i32(x)`, `u32(n)`. Mixing `i32`
and `u32` in an expression also needs an explicit cast.

### AsBindGroup sampled textures default to vertex|fragment visibility

Adding a `#[texture(N)] #[sampler(M)]` field to an `AsBindGroup` struct
whose layout feeds a **compute** pipeline fails at pipeline creation
(runtime, not compile time) with:

> `Shader global ResourceBinding { group: G, binding: N } is not
> available in the pipeline layout — Visibility flags don't include the
> shader stage`

Cause: the derive's default visibility for sampled textures/samplers is
vertex|fragment, while `#[storage_texture]` defaults to compute — so a
bind group of storage textures works until the first sampled texture is
added. Fix: `#[texture(N, visibility(compute))]
#[sampler(M, visibility(compute))]`. (Hit in
`thalos_volumetric_clouds::CloudsImage`, June 2026.)

### naga_oil imports are a Bevy preprocessor feature, not WGSL

`#import`, `#define_import_path`, `#ifdef`/`#endif` are `naga_oil`
directives Bevy runs before handing WGSL to naga — they aren't standard
WGSL. A module is only importable if it declares `#define_import_path`,
and the import path string must match exactly. In Thalos, shared shader
libraries (`thalos::lighting`, `thalos::atmosphere`) are registered by
`PlanetLightingPlugin`; a shader that imports them must run in an app that
added that plugin or the import won't resolve.

### A struct *field* name that matches a naga_oil `#import`ed global breaks field access

Symptom: `error: invalid field accessor 'config'` pointing at a perfectly
valid-looking chained access like `terrain_extras.shadow.config.x`, even
though the struct clearly declares `config`. The shader fails to compile →
its whole pipeline silently doesn't render (for terrain: the ground just
*disappears* — translucent, no error except the one buried in stderr).

Cause: the shader `#import`s a global named `config`
(`#import thalos_udlod::bindings::{config, …}` in `body_terrain.wgsl`), and
naga_oil's import handling collides with a **struct field** of the same name,
corrupting the field accessor. The *imported global* wins; your field becomes
unreachable.

Fix: rename the field to something not imported (we used `config` → `gate` in
`ShadowCascadeBlock`). Field names are free — for a uniform the layout is by
declaration *order*, so the WGSL field name need not match the Rust
(`encase`/`ShaderType`) field name. Only that shader needs the rename, but
keep them consistent across shaders for sanity.

Nasty because it's **shader-specific**: the identical struct in `tree.wgsl`
compiled fine — that shader doesn't import udlod's `config`, so no collision.
A headless preview that only exercises the tree material will NOT catch a
terrain-shader-only collision; capture the game's **stderr** to see the real
naga error (`pipeline_cache: failed to process shader`). Cost a long blind
debugging loop, June 2026, adding cascaded sun shadows.

### Value-noise gradients show the cubic lattice as a grid "weave" — use gradient noise for normals

A detail *normal* derived from value noise (random scalars hashed at integer
lattice corners, trilinearly interpolated) shows a regular axis-aligned grid
"weave": value noise's gradient is strongly anisotropic along the lattice axes,
and a normal amplifies it under raking light. The value/albedo of value noise
hides this; its *derivative* does not. Fix: derive detail normals from
**gradient (Perlin) noise** instead — randomness in per-corner gradient vectors,
a far more isotropic derivative — with an analytic derivative so it stays one
evaluation, and keep the corners wrapped mod the period for floating-origin
safety. See `perlin3_periodic_grad` / `fbm3_perlin_grad` in
`crates/body_render/src/ground/body_terrain.wgsl`. (Confirmed June 2026:
switching the terrain detail normal value→gradient removed the weave entirely.)

### Don't override `descriptor.vertex.buffers` in a custom `Material::specialize` if it's drawn in the prepass

A custom `Material` with its own vertex shader still gets a **prepass** pipeline
built (depth/normal/motion) when the camera has a prepass — using the *standard*
prepass shaders, not your shader. If your `specialize` sets
`descriptor.vertex.buffers = vec![my_layout]` to force a specific attribute set,
that override also applies to the prepass pipeline and **truncates the layout the
standard prepass vertex shader needs** — symptom is a fatal
`create_render_pipeline, label = 'prepass_pipeline'` validation error like
"Location[7] Float32x4 ... is not provided by the previous stage outputs".

Fix: don't override the vertex layout. Bevy's mesh + prepass pipelines already
auto-include every attribute the mesh actually has (POSITION/NORMAL/UV_0/UV_1/
TANGENT/COLOR at their standard locations 0/1/2/3/4/5), so a custom vertex shader
can just read e.g. `@location(3)` for `UV_1` as long as the mesh carries it.
Keep `specialize` to genuinely pipeline-wide tweaks (e.g. `cull_mode`). Seen June
2026 adding `TreeMaterial` (vegetation) — `GrassMaterial` worked precisely
because it never overrode the layout.

### A custom-billboard `Material` can dodge the prepass by keeping POSITION degenerate

Corollary to the above. A `Material::vertex_shader` is used **only in the main
pass** — the prepass uses the *standard* prepass vertex shader (you didn't set
`prepass_vertex_shader`). So if your vertex shader *builds* geometry the standard
prepass can't reproduce (e.g. expands one `POSITION` into a camera-facing
billboard quad from a corner id in `UV_0`), the prepass renders the raw
`POSITION`s. Make all of a quad's corners share the same `POSITION` (the billboard
centre) and the standard prepass draws a **degenerate, zero-area** quad → nothing,
with no crash and no custom prepass shader. Keep the material `AlphaMode::Opaque`
(or `Mask`) and `discard` in the fragment: the **main** opaque pass still writes
depth where coverage passes, so it occludes correctly. Caveat: the DepthPrepass /
NormalPrepass textures then lack these draws — fine for Thalos because the
atmosphere clips against `scene_depth` (a copy of the **main-pass** depth, which
includes them), but SSAO-style prepass consumers would miss them. Used June 2026
for `TreeImpostorMaterial` (octahedral foliage impostors).

### A custom prepass shader with a displacing vertex — DO override the vertex layout

Flip side of the "don't override `vertex.buffers`" entry above. When you DO want a
displacing opaque material in the depth prepass (for early-Z), give it a
`prepass_vertex_shader()` that reproduces the displacement EXACTLY (share the
displacement via a `#define_import_path` lib so the main + prepass clip depth match
bit-for-bit — a mismatch makes the pre-populated prepass depth early-Z-reject your
own visible geometry). But the **depth-only** prepass derives a minimal,
POSITION-only vertex buffer layout, so if your prepass shader reads any other
attribute (UV_0, TANGENT, COLOR…) you get `create_render_pipeline label =
'prepass_pipeline'` → "Location[N] … is not provided by the previous stage
outputs". Fix: override `descriptor.vertex.buffers = vec![layout.0.get_layout(&[…])]`
in `specialize` to include every attribute BOTH your main and custom-prepass shaders
read. This is the override the earlier entry warns against *only when using the
standard prepass shader* — with a matching custom prepass shader it is required and
safe. Per-material prepass opt-in/out in Bevy 0.18 is `fn enable_prepass() -> bool`
(the `MaterialPlugin { prepass_enabled }` field is gone). Landed June 2026 for the
grass depth prepass (`grass_prepass.wgsl` + `thalos::grass_displace`).

### A depth prepass forces Equal depth-test → Z-fights coincident opaque surfaces

Enabling a `DepthPrepass` makes the main opaque pass use an **Equal** depth compare
(shade each pixel once). If two opaque surfaces are *coincident* (same depth — e.g.
alpha-tested leaf cards lying exactly on an opaque canopy "egg-shell"), Equal lets
**both** pass and draw order decides the winner → a Z-fight that's invisible without
a prepass (front-to-back + GreaterEqual + depth-write makes the first-drawn win
consistently). Symptom for the broadleaf tree: the canopy went **pale/white** (the
shell won over the green leaves) the instant trees joined the prepass; toggling
`enable_prepass` off restored green, and the no-prepass impostor bake was always
green. No clean shader fix — the surfaces must be physically separated (offset the
leaves off the shell). So a material with coincident internal geometry should stay
OUT of the prepass (`enable_prepass=false`) until that geometry is fixed. Grass has
no such coincidence (free-standing blades), so it prepasses fine. June 2026.

### One WGSL file with both @vertex and @fragment entries needs def-gated entries under udlod

A `Material` that supplies the SAME file as both `vertex_shader()` and
`fragment_shader()` for the thalos_udlod terrain pipeline gets composed twice —
the fragment compile with the `FRAGMENT` shader def, the vertex compile without.
udlod's `Coordinate` struct (types.wgsl) gains `uv_dx`/`uv_dy` fields under
`FRAGMENT`, so any `Coordinate(...)` *constructor* is only valid under one def
state: udlod's `vertex.wgsl` builds 4-arg Coordinates (invalid under FRAGMENT),
`fragment.wgsl`'s `fragment_info` builds 6-arg ones + calls `dpdx` (invalid
without). naga_oil prunes unused *imported* functions, so imports are safe — but
it always keeps the top file's **entry points**, which drag the off-stage
constructors into the final module. Symptom: `failed to build a valid final
module: Function 'thalos_udlod::fragment::fragment_info' is invalid` on the
vertex pipeline. Fix: wrap the fragment entry in `#ifdef FRAGMENT … #endif` and
the vertex entry in `#ifndef FRAGMENT … #endif` (naga_oil supports `#ifndef`).
Top-file helper functions shared by both are fine ungated as long as they don't
construct `Coordinate` or use derivatives. Hit July 2026 adding the analytic
pad-flatten vertex stage to `body_terrain.wgsl`.
