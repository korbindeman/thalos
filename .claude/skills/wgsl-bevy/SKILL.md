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
- `partition` — hit July 2026 in `cloud_composite.wgsl` (`let partition = …` → ``name `partition` is a reserved keyword``). Renamed to `ramp`. Note the failure shape: the capture still wrote a plausible-looking PNG (BL-20), and only the pipeline-cache ERROR in stderr plus the capture client's non-zero exit distinguished it from a good run — read stderr before trusting the image.

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

### naga_oil resolves `const` imports but never emits them — export values as functions

Symptom: a pipeline dies at creation with

```
error: no definition in scope for identifier: `thalos::landcover::C_ROCK_LO`
750 │ ground = mix(ground, mix(C_ROCK_LO, C_ROCK_HI, alpine), rock_t);
```

and — because a failed terrain pipeline still lets the process run — the
visible result is **ground rendered black** while everything else lights
normally. The capture host exits 3, and the real error only reaches
`artifacts/diagnostics/visual_capture_server.log`, never the client's stdout,
so `just screenshot` reports a bare exit code 1. Read that log first (BL-20's
cousin: PNGs are still written).

Cause: `#import thalos::landcover::{C_ROCK_LO}` is accepted by the
preprocessor, and every *use* is rewritten to the mangled module-qualified
name — but naga_oil only composes **functions** (and types) from the imported
module into the final naga module. The `const` definition is never carried
across, so the mangled reference dangles.

Fix: wrap the value in a function and import that.

```wgsl
const LC_ROCK_LO: vec3<f32> = vec3<f32>(0.108, 0.104, 0.098);
fn substrate_rock_color(alpine: f32) -> vec3<f32> {
    return mix(LC_ROCK_LO, LC_ROCK_HI, clamp(alpine, 0.0, 1.0));
}
```

This is why every shared value in this repo already travels as a function
(`vegetation_color`, `macro_variation`, `shade_surface`) — the shared-library
rule and naga_oil's capabilities happen to agree. A shared *palette anchor*
therefore needs an accessor, not a `const`. Hit July 2026 giving the tile
renderer udlod's substrate palette.

Hit again July 2026 extracting the cloud radiance model into
`thalos::volumetrics`: the constants moved verbatim from a module where they were
local (and therefore legal untyped) into a library, and every importer died at
pipeline creation. Adding an explicit type (`const X: f32 = 0.8;`) does **not**
help — it is not a typing problem, naga_oil simply never emits the const. The
accessor (`fn water_cloud_albedo() -> f32`) is the fix. Watch for this whenever a
constant is *promoted* into a shared library; it is invisible until then.


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

### A prepass vertex shader that reads the material bind group must NOT be `AlphaMode::Opaque`

Bevy 0.19 skips the material bind group for a **depth-only opaque** pass: the
pipeline gets `empty_layout` at group 3 and the draw function
(`PrepassOpaqueDepthOnlyDrawFunction`, or `ShadowsDepthOnlyDrawFunction` for
Bevy's shadow maps) never binds it. A `prepass_vertex_shader()` that reads
`@group(#{MATERIAL_BIND_GROUP})` — a displacement atlas, a per-instance table —
therefore builds a pipeline referencing a binding its own layout omits, and the
**fatal** wgpu error takes the process down:

```text
create_render_pipeline, label = 'pbr_prepass_pipeline'
  Shader global ResourceBinding { group: 3, binding: 111 } is not available in
  the pipeline layout → Binding is missing from the pipeline layout
```

Fix: `AlphaMode::Mask(_)` — it sets `MeshPipelineKey::MAY_DISCARD`, which is the
only user-reachable bit that disqualifies the depth-only path in **both** the
prepass and the shadow queue. (`PREPASS_READS_MATERIAL` looks like the intended
opt-in and is read by `light.rs`, but 0.19.0 never sets it and no `Material` impl
can — neither queue ORs `MaterialProperties::mesh_pipeline_key_bits` into the mesh
key. Recheck on the next Bevy bump.) Override it at the *pipeline* level only —
`MaterialExtension::alpha_mode() -> Option<AlphaMode>` — and leave the base
`StandardMaterial` opaque, so the GPU-side flags still say opaque, `alpha_discard`
forces alpha to 1.0, and nothing is actually masked. The main-pass pipeline is then
identical apart from a `MAY_DISCARD` def (blend/depth come from the blend bits).

Two debugging notes. `pbr_prepass_pipeline` exists nowhere in Bevy or this repo:
the label is Bevy's `prepass_pipeline` with `StandardMaterial::specialize`'s `pbr_`
prefix, so it only tells you the culprit is some `ExtendedMaterial<StandardMaterial, _>`
— and it covers the camera prepass *and* the shadow pass. And "the binding works in
the main pass" is not evidence against this: the main pass always binds group 3.
Cost of the fix: the depth prepass compiles a fragment stage (Bevy's void
`prepass_alpha_discard`), losing early-Z depth writes on that draw. NTR-X1 tile
terrain, August 2026 — INC-20260826T124420Z.

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

### Naga requires an explicit tail return after a returning `loop`

Naga does not prove that an unconditional WGSL `loop` is exhaustive even when
every apparent exit path returns a value. A function such as `fn f() -> T` whose
body ends immediately after that loop fails validation with `Returning None
where Some([...]) is expected`. Add an explicit, type-correct fallback `return`
after the loop (it may be unreachable in practice). Hit July 2026 while adding
resident-ancestor fallback to the udlod tile-tree lookup.

### Procedural normal detail needs footprint filtering, not only distance fade

A camera-distance fade does not bound a procedural octave's screen frequency:
a grazing surface can compress metre-scale noise into subpixel fragments while
remaining well inside the distance band. Compute a world/body-space footprint
from `dpdx`/`dpdy` in uniform fragment control flow and fade each colour and
normal octave as the footprint approaches its wavelength. Otherwise a strong
BRDF response can turn unresolved normal flips into bright/dark stipple even on
fully opaque geometry. Hit July 2026 on Mira's Hapke regolith (INC-0009).

### Never project a large position onto a per-fragment direction to phase a noise

`noise(dot(p, across) / L, dot(p, fall) / L2)` is the obvious way to stretch a
procedural texture along a slope frame, and it is a moiré generator whenever
`|p|` is large. The phase derivative with respect to surface orientation is
`|p| / L`: with the tile path's wrapped body position (up to `TILE_WRAP_M`,
8192 m) and `L = 24 m`, tilting the surface **0.2°** slides the pattern a full
stripe. The texture then tracks the normal field instead of the ground and
renders as contour-following whorls — an agate / topographic-map look, densest
where slope turns, and worst up close where no footprint fade retires it.

The tell: swirls that follow *shading* rather than *ground*. Suspect this before
precision, LOD, or the height source — `p` being bounded means f32 is innocent.

Fix: keep the coordinate isotropic in body space (so phase depends on position
only) and get the anisotropy from a directional **filter** — N taps of the same
field displaced along the direction. Sensitivity drops to `span/2 × angle`. The
slope frame may *orient* a pattern; it may never *phase* one. Same rule applies
to any varying frame (flow fields, tangents, view vectors). Hit July 2026 on
`tile_terrain.wgsl`'s rock gully striation (INC-20260727T004856Z).

### A mid-edit hot-reload can kill a pipeline for the rest of the boot — check the log before judging any capture

Bevy's `embedded_watcher` reloads a WGSL file the moment it is saved. With
several agents (or one agent probing) editing shaders live, a reload can catch
a file — or any naga_oil module it imports — in a transient broken state. The
pipeline then fails to build and its pass silently no-ops; PNGs keep coming out
and the process exits zero (the BL-20 gap). Two tells, both hit July 2026 while
bisecting a cloud-march change (INC-20260729T012803Z follow-up work):

- **A near-zero GPU timing for a pass that should cost milliseconds** (the
  cloud probe report showed 0.07 ms for the whole march) means the pass did not
  run, not that it got fast.
- **`no definition in scope for identifier: X` where X verifiably exists in
  the source module** means the *import chain* failed on a stale/mid-edit
  snapshot, not that the symbol is missing. Do not "fix" the import.

The poison spreads: every subsequent hot-reload probe on the same boot can
inherit the dead pipeline, so an A/B bisect returns noise — five successive
probes "confirmed" an innocent change. Discipline: grep the capture host log
for `failed to process shader` / `Validation Error` **per shot** before
treating its PNG as evidence, and when a bisect behaves inconsistently,
cold-restart the host (`just capture-stop`) before trusting another variant.
