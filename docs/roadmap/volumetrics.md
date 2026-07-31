# Volumetrics — a shared foundation, not a shared renderer

**Background sprint.** Strategy and sequencing live here; status lives in
`docs/backlog.md`. The seam is fixed by ADR-20260730T034500Z: **share the
radiance terms, keep the march per-effect.**

## The idea

Clouds, fog, vapour cones, dust, contrails and rocket smoke are one family:
condensed or suspended matter scattering sunlight. They differ in density field,
extent and lifetime — not in how light behaves inside them. Today clouds carry
the only good radiance model in the tree, and every new medium is tempted to
re-derive it badly (the vapour cone did, on its first pass, and reproduced a
defect the cloud shader had already diagnosed in writing).

The foundation is therefore **smaller and more valuable than "a volumetrics
renderer"**: it is the physics of the source term, plus the conventions around
it. The loops stay local.

## Three kinds of volume

The taxonomy that actually decides code. `flow_effects.md` splits attached
effects by *memory*; adding fog and clouds shows a second axis:

| kind | examples | geometry | ordering |
|---|---|---|---|
| **Fullscreen composite** | clouds ✅, fog, atmosphere ✅, ocean ✅ | planetary shell / whole frame | **must claim a `composite_order` slot** |
| **Attached volume** | engine plume ✅, reentry shock layer ✅, vapour cone ✅, dust puffs, heat haze | analytic proxy hull on a vehicle | `Transparent3d`, sorts by depth |
| **Shed volume (has memory)** | contrails, smoke trails, ablation wakes, vortices | ring buffer of aged samples → swept tube | `Transparent3d` |

Fog is the entry that proves the axis matters: it is a *cloud-shaped* user
(fullscreen, needs a composite slot), not a *cone-shaped* one, even though people
group it with "small effects".

## What is built

**`thalos::volumetrics`** — the shared radiance model, extracted from the cloud
march 2026-07-30 and verified neutral by matched captures against a measured
noise floor. Owns: phase lobes, multi-scatter octaves and their extinction
weights, `volumetric_scattering`, `powder_term`, `ambient_occlusion`, and the
canonical diffusion albedos (water, dust). **Albedo is a parameter** — that is
what makes it a volumetrics library rather than a cloud library.

Users: the cloud near-march, the cloud composite, and the vapour cone.

## What comes next, and in what order

1. **Fog** — the second *unlike* user, and the one that tests the seam properly.
   A fullscreen composite like clouds but with a trivial density field, so it
   exercises the library without the cloud march's machinery. Needs a
   `composite_order` slot.
2. **`RibbonTrail`** — the shed-volume primitive (see `flow_effects.md` for the
   three constraints already known: body-fixed samples, warp gating, ambient
   ice-supersaturation lifetime). Contrails, smoke and wakes are then emitter
   configs. These are scattering media, so they shade through the library.
3. **Dust** — lofted mineral dust at engine ignition and on rough landings. First
   user of a **non-white** albedo, which is the check that the parameterization
   was real and not water-shaped.
4. **CPU-side conventions.** The three attached effects (`plume`, `reentry`,
   `vapor_cone`) currently duplicate ~80 near-identical lines each: mesh
   resource, material, spawn system, update/visibility system, cull mode,
   `NoFrustumCulling` + `NotShadowCaster` + `FlowProxyMesh`. That is a "one
   canonical path" violation forming in real time. Fold it into one
   `AttachedVolume` plugin pattern **after** fog and dust have shown what varies.

## The open question, deliberately left open

Whether this becomes a `thalos_volumetrics` **crate** — owning the CPU
conventions, proxy geometry and uniform types — or stays a WGSL library plus a
module in `rendering`.

Do not decide it before items 1 and 3 above. Generalizing off clouds alone
produces a cloud-shaped abstraction; the second and third users are what falsify
it. Same discipline as BIO-7 for biomes.

## Rules that already have scars behind them

- **Never render a medium with a single scattering lobe.** It cannot reach the
  brightness of a real cloud, and the result is grey-blue mud that takes its
  chroma from the ambient. Measured, twice.
- **Never build a generic march.** ADR-20260730T034500Z.
- **Values cross a WGSL module boundary as functions, not `const`.** naga_oil
  accepts the import and never composes the definition.
- **A fullscreen volumetric must claim a `composite_order` slot.** Bias 0 leaves
  its position in the stack depending on where the camera points.
- **Any refactor touching the cloud march is proven neutral by matched captures
  against the run-to-run noise floor** — clouds are the heaviest pass in the
  frame and the most tuned thing in the tree.
