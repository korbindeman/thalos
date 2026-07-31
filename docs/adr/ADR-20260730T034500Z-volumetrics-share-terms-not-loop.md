# ADR-20260730T034500Z — Volumetrics share radiance terms, never the march

**Status:** accepted (2026-07-30)

## Context

Thalos is growing participating media beyond clouds: a transonic vapour cone
landed 2026-07-30, fog and lofted dust are wanted, and contrails/trails are
queued behind the ribbon primitive. Each is condensed or suspended matter
scattering sunlight — the same physics with a different density field.

The cloud system already contains a hard-won radiance model: multi-scatter
octaves, dual-lobe phase, a diffusion-limit reservoir, and a powder/silver-lining
term, with a written record of two rounds of getting it wrong and measured
luminance numbers from `cloud_cruise`. The vapour cone's first implementation
ignored all of it and hand-rolled a single Henyey–Greenstein lobe. It rendered
the exact defect that record describes — "no brighter than the sky ambient
filling it — grey-blue mud" — because single scattering *cannot* reach the
brightness of a real cloud.

The obvious reaction is "make a volumetrics foundation and have clouds be one
user of it". The question this record settles is **where the seam goes**.

## Decision

**Share the radiance terms. Keep the march per-effect.**

`thalos::volumetrics` (`crates/rendering/shading/src/shaders/volumetrics.wgsl`)
owns the medium-agnostic radiance model: phase lobes, the multi-scatter octaves
and their extinction weights, `volumetric_scattering`, `powder_term`,
`ambient_occlusion`, and the canonical diffusion albedos. The scattering albedo
is a **parameter**, not a constant — water, ice and dust differ, and that is the
generalization that makes it a volumetrics library rather than a cloud library.

Every medium keeps its own march, bounds and density field.

## Why not a generic march

It is the tempting next step and it is wrong here, for three separate reasons:

- **The loops are not variants of each other.** The cloud march walks a planetary
  shell with distance-driven tiering, a coverage-calibrated fill LUT and temporal
  amortization across frames. The vapour cone marches a twenty-metre analytic
  envelope attached to a vehicle moving at Mach 1. Fog will be a fullscreen
  composite. Only the per-sample radiance is common.
- **WGSL has no dynamic dispatch.** A `march_volume(field, bounds)` abstraction
  becomes either one mega-shader with dead branches or `naga_oil` template
  gymnastics, and it puts the single heaviest pass in the frame at risk for no
  visual gain.
- **The cloud march's calibration is not transferable.** Its fill LUT is an
  *expected-opacity* curve fitted to statistical cloud coverage, and its temporal
  reconstruction assumes a body-fixed field. A small, fast, attached volume is
  the worst case for both — the step sizes step over it, and hit-aware history on
  a translucent mover is what smeared in INC-0016.

## Why clouds were not moved into a new crate

Extracting the foundation *out of* clouds is a small, verifiable change; moving
the cloud system into a new crate is a large one that buys nothing the extraction
does not. Clouds stay where they are and import the library.

Whether the CPU-side conventions (proxy hulls, uniform layout, the attached-volume
plugin pattern the three flow effects currently duplicate) eventually justify
their own crate is deliberately left open — see `docs/roadmap/volumetrics.md`.
That decision wants two *unlike* users exercising the seam first, on the same
reasoning BIO-7 is the falsification test for the biome abstraction: generalize
off clouds alone and the abstraction comes out cloud-shaped.

## Consequences

- The diffusion albedo now has **one** definition. It used to be written twice —
  `CLOUD_MS_ALBEDO` in the near march, `FAR_CLOUD_ALBEDO` in the composite — each
  commented "MUST equal" the other. That drift hazard is gone.
- Values cross the module boundary **as functions**, not `const`: naga_oil accepts
  a `const` import and rewrites uses to a mangled name but never composes the
  definition, so the reference dangles at pipeline creation. Recorded in the
  `wgsl-bevy` skill; hit again during this extraction.
- The extraction was verified neutral by matched `cloud-cruise` captures against
  a measured run-to-run noise floor: floor max Δ6 / mean 0.246, before-vs-after
  max Δ6 / mean 0.236 — i.e. below the floor.
- Any new medium that renders with a single scattering lobe is now a review
  finding, not a taste call.
