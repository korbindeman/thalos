# INC-20260724T232256Z-cloud-multiscatter-octaves-carried-no-energy: the multi-scatter octave sum was normalised into an average, so clouds rendered at single-scatter radiance

- **Status:** Fixed (agent-side); **capture + user verification pending** — the working tree
  could not build during this session (unrelated in-progress work in
  `crates/simulation/physics_canonical/src/simulation.rs`)
- **Date:** 2026-07-24
- **Severity:** visual
- **Surface:** every near-volume cloud view — user's live low-altitude screenshots, and
  `just screenshot cloud-cruise` / `cloud-interior` / `cloud-sunset`

## Summary

Clouds read muddy grey-blue rather than white, at every sun angle, across seven rounds of
morphology and transition work. The cause was not morphology, coverage, tonemapping, or
ambient tuning: the near march's "multiple-scattering octaves" divided their weighted sum by
`Σw`, which turns an energy **sum** into a weighted **average** of phase values. The octaves
therefore reshaped the phase lobe and added none of the energy they are named for, leaving
cloud radiance at *single-scattering* magnitude — roughly a sixth of the diffusion limit, and
below the sky ambient filling the same cell. A cloud lit mostly by blue sky ambient is a
grey-blue cloud.

## Symptom

User verdict against MSFS/Blackrack references: "the clouds always look muddy, and not pure
white." Independently measurable in the existing `artifacts/visual/latest/cloud_cruise.png`
(display luminance, sRGB-decoded):

| region | p99 luminance | top-200 mean RGB |
|---|---|---|
| near-tier puffs | **0.30** | 0.271 / 0.294 / **0.347** |
| far-tier deck | 0.73 | 0.705 / 0.745 / 0.736 |
| sky, zenith | 0.49 | 0.388 / 0.517 / 0.673 |
| sky, horizon | 0.73 | 0.658 / 0.745 / 0.759 |

Two tells in one table. The near tier is **darker than the sky it is drawn against** — no
sunlit cloud can be. And its brightest pixels are blue-dominant (B/R = 1.28), i.e. its chroma
comes from the ambient term, not the sun. The far tier renders the same weather field 2.4×
brighter.

## Root cause

`raymarch` in `clouds_compute.wgsl`:

```wgsl
let scattering = dot(MS_OCTAVE_WEIGHTS * ms_lobes, octave_shadow)
    / (MS_OCTAVE_WEIGHTS.x + MS_OCTAVE_WEIGHTS.y + MS_OCTAVE_WEIGHTS.z);
```

`ms_lobes[i]` is a Henyey–Greenstein value already normalised by `1/4π`, so it integrates to
1 over the sphere: each octave is a *probability density*, carrying exactly one scattering
event's worth of energy. Averaging three of them yields one event's worth of energy with a
slightly wider lobe. At side-scattering that is `p(90°) ≈ 0.051`, so a fully sunlit optically
thick cell returned `0.051 · E`.

The diffusion limit says what it should be. A water cloud scatters conservatively
(ϖ ≈ 0.9999); an optically thick sunlit cell returns ~0.8 of the incident flux, leaving close
to isotropically, so its radiance is `A · E/π ≈ 0.25 · E`. **A factor of five, and the
missing part is exactly the part called "multiple scattering".**

Everything downstream then reads as a separate defect and invites a separate local fix, which
is how the error survived so long:

- Sky ambient (physically bound to F3/F4 `SkyAmbient` since round 6) is ~0.2–0.35 in the same
  units, so it *dominated* the 0.17 direct term. Hence blue-grey chroma and near-zero
  lit/shadow contrast — clouds with correct sculpted geometry still read as flat sheets,
  because nothing shaded them.
- The composite's far-tier prefactor had been hand-raised **twice** (0.55 → 0.68) "because far
  cells read grey next to the near volume's sunlit white at the handoff". It was compensating
  for the near tier being 4× dark, and ended up ~1.7× brighter than a white Lambertian
  surface under the same sun.
- The near march's Reinhard peak white point (2.2) was set against single-scatter-magnitude
  radiance. Left alone it would have eaten a third of every corrected cloud top.

A third, independent contributor to "not pure white": both tiers multiplied the authored
climate albedo by a *second* near-1 per-channel factor kept as "headroom for phase peaks" —
`(0.94, 0.96, 0.99)` in the near tier, `(0.90, 0.93, 0.97)` in the far. Stacked on Thalos's
authored `(0.94, 0.96, 1.0)`, sunlit cloud carried a 12–15% blue bias by construction.

### Ruled out

- **Morphology** (rounds 5–7). Sculpted dome tops landed the day before and are real, but a
  shape only reads through a lit/shadow gradient the shading was not producing.
- **Tonemapping / exposure.** The defect is a *ratio* — clouds darker than the sky beside
  them in the same frame — which no global curve creates.
- **Fill / opacity parity** (`fill_lut`, round 5). That contract governs how much of the
  frame clouds cover, and it is derived. This is radiance per covered pixel; the two are
  independent, and the calibration does not need re-deriving.

## Fix

The source term is split the way the physics is, in `clouds_compute.wgsl`:

```wgsl
let single = ms_lobes.x * octave_shadow.x;                       // exact normalised phase
let ms_depth = dot(MS_OCTAVE_WEIGHTS.yz, octave_shadow.yz)
    / (MS_OCTAVE_WEIGHTS.y + MS_OCTAVE_WEIGHTS.z);               // 0..1 depth response
let ms_aniso = mix(1.0, ms_lobes.z / INV_FOUR_PI, MS_ANISO);
let multi = CLOUD_MS_ALBEDO * INV_PI * ms_depth * ms_aniso;      // diffusion reservoir
let scattering = single + multi;
```

Single scattering keeps the exact normalised phase and hard shadow attenuation, so it still
owns the forward glare and the silver lining. Multiple scattering becomes an
isotropic-equivalent reservoir whose magnitude is the medium's diffusion albedo
(`CLOUD_MS_ALBEDO = 0.80`) — a property, not a brightness knob. The wider octaves keep the one
job of theirs that was ever physical: supplying the reservoir's **depth response**, so light
leaks around occluders rather than multiplying cores to charcoal, plus a gentle residual
anisotropy (`MS_ANISO`) that keeps the sun side brighter.

Resulting behaviour at a fully lit thick cell, side view: `0.288 · E` (was `0.051 · E`),
against the diffusion-limit target of `0.80/π = 0.255 · E`. Sun-facing rises to `2.5 · E` for
the glare; a τ_sun = 4 shadow side falls to 41% of lit and a τ_sun = 12 core to 15% — the
contrast that makes a sculpted dome read as a volume.

Three consequences of that anchor, applied in the same change so no tier is left compensating:

1. **The near tier is now anchored to the renderer's own photometry.** A lit cell lands at
   `A · flux · SCENE_FLUX_SCALE · SURFACE_DIRECT_SCALE` — exactly a Lambertian surface of
   albedo `A` facing the same sun, the anchor every spine surface already uses.
2. **The far prefactor is derived from that anchor, not eyeballed at the handoff:**
   `K = A · SURFACE_DIRECT_SCALE / (tint.g · FAR_SHADE_LIT) = 0.302` (was 0.68).
3. **The Reinhard white point moves 2.2 → 10.0**, above ordinary sunlit cloud, so it bounds
   the forward-scatter spike instead of dimming every white top.

Both per-channel "headroom" tints are deleted; chroma is the authored climate albedo alone,
and peak headroom lives in the achromatic white point.

## Recurrence signal

**Cloud pixels darker than the sky behind them in the same frame.** That is the one-glance
tell, and it is measurable without a reference image: sample the brightest near-tier cloud
pixels and the sky beside them in any daylit capture. A sunlit cloud top should land near a
white Lambertian surface under the same sun; if it lands near the sky's own radiance, the
scattering integral has lost its multiple-scattering energy again.

Secondary tell: **blue-dominant "white" clouds** (B/R > ~1.1 in the brightest cloud pixels).
Either the ambient term is out-competing the direct term, or a per-channel tint has been
stacked on the authored albedo again.

## Standing rules

Recorded in `docs/rendering/clouds.md` §3.4:

- **A normalised phase function carries one scattering event.** Any octave series meant to
  approximate multiple scattering must **sum** energy; dividing by `Σw` silently reduces it to
  a lobe-widening filter with a physical-sounding name.
- **Both tiers are anchored to `CLOUD_MS_ALBEDO`, never to each other.** Matching the far tier
  to the near tier by eye is what let a 4× error hide inside a "handoff continuity" fix. If
  either side's photometry moves, re-derive the far prefactor from the anchor.
