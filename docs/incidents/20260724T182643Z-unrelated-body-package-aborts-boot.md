# INC-20260724T182643Z — One stale terrain package aborts the whole app (a stale `Mira.bin` blocks every Thalos capture)

**Status:** fixed (per-body surface failures degrade that body only).

## Symptom

Capture runs died at boot, twice in one session, with:

```
thread 'main' panicked at crates\runtime\game\src\lib.rs:224:10:
Failed to load body terrain surfaces: "Mira requires an offline terrain package:
stale terrain package at assets/terrain_packages\Mira.bin:
content key 5620985c9a0e3475, expected a3ea8530fd58c78a. Run `just bake Mira`"
error: process didn't exit successfully: `target\debug\thalos_capture_host.exe` (exit code: 101)
```

The runs in question were **Thalos** diffusion captures (`massif-*`,
`THALOS_TERRAIN=diffusion`) — scenes that never reference Mira. Mira's package
goes stale on any terrain-generator change, so ordinary terrain work
self-blocks every unrelated capture until an ~offline bake is re-run.

## Evidence

- `BodySurfaceRegistry::load` used `?` on the per-body package load, so the
  first failing body aborted construction of the registry for **all** bodies.
- `AppBuilder::build` turned that `Err` into `.expect(…)` — a process-wide
  panic before any window/preset logic runs.
- The same function already degrades gracefully three lines above: a failed
  diffusion-surface load logs a warning and falls back to the procedural
  surface. The strict path was inconsistent with its own neighbour.
- Consumers already read surfaces through `Option` accessors
  (`surfaces.surface(body_id)?`, `let … else`), i.e. "this body has no
  surface" was already representable; only two call sites in
  `rendering/spawn.rs` asserted otherwise via
  `.expect("terrain body has a canonical surface")`.

## Hypotheses considered

- **Mira's package genuinely required by these presets** — ruled out: the
  presets target Thalos; Mira contributes nothing to the frame.
- **Content-key check too strict** — rejected as the fix: the key is doing its
  job (the package *is* stale). The defect is the blast radius, not the check.

## Root cause

Surface construction treated a **per-body** failure as a **global** one. One
body's unusable offline artifact took down the process, so unrelated work
inherited an unrelated body's maintenance state.

## Fix

- `BodySurfaceRegistry` records failures in `degraded: HashMap<BodyId,
  DegradedSurface>` and keeps loading every other body; the airless package
  construction moved into `build_airless_package_surface` so a failure is an
  ordinary `Err` instead of a `?` that aborts the loop.
- `AppBuilder::build` prints each degraded body in the startup banner; the
  `expect` now covers only registry-global failure.
- `rendering/spawn.rs` degrades instead of asserting: no impostor albedo bake
  (blank cube + `albedo.w = 0` → flat-colour path), no coast atlas, and no
  height-source registration for a surface-less body.
- **Evidence integrity is preserved at the one place it matters**:
  `poll_capture_requests` refuses a capture whose *target* body is degraded,
  responding with the body name, the reason, and the `just bake <Body>`
  command — so a broken world can never be photographed and filed as a valid
  PNG.

## Prevention

- **Per-body failures stay per-body.** New body-scoped resources (surfaces,
  packages, weather fields) record a degraded entry; they do not abort the
  registry or the process. Where the failure would corrupt *evidence*, refuse
  at the request boundary (as the capture server now does) rather than at boot.

## Recurrence tells

- A panic at `AppBuilder::build` naming a body the current scene does not use.
- "Run `just bake <Body>`" appearing in a run that targets a different body.
