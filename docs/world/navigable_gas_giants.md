# Navigable Gas Giants

**Status:** direction captured 2026-07-29; no implementation has landed.
This document owns the product and architecture direction for ships entering
gas- and ice-giant atmospheres. The execution queue is `docs/backlog.jsonl`.

Gas giants should be places, not only distant pictures. A ship can approach
from orbit, descend through haze and cloud systems, fly inside the visible
weather, and continue into increasingly hostile air until pressure, heating, or
dynamic load destroys it. There is no terrain impact and no hidden solid
surface at the body's authored radius.

Rendering detail remains in `docs/rendering/atmosphere.md` and
`docs/rendering/clouds.md`; aerodynamic force and regime mechanics remain in
`docs/simulation/aerodynamics.md` and `docs/simulation/regimes.md`. This
document defines the cross-system authority those projections share.

## 1. Player experience

An entry should be one continuous event:

1. **Orbit.** The existing procedural disc carries zonal bands, vortices, haze,
   limb darkening, rings, and the planet-scale weather read.
2. **Approach.** The disc becomes a vast curved weather horizon. Resolved
   systems gain height and parallax without changing identity or position.
3. **Cloud top.** The ship flies above and between immense decks, towers, gaps,
   and storm walls. The local sky and cloud light agree with the exterior
   planet.
4. **Interior.** Visibility collapses inside dense cloud. Scattered light,
   extinction, precipitation, lightning, and nearby cloud structure replace
   the orbital band read.
5. **Below the visible deck.** Overhead cloud becomes a dim roof; clear regions
   remain hazy rather than empty. Deeper condensate layers and storms may
   appear. Sunlight progressively gives way to darkness and diffuse thermal
   colour.
6. **Terminal descent.** The craft fails from its own pressure, temperature,
   heating, or dynamic-pressure limits. Failure depth is a vehicle property,
   not a planetary collision sphere.

The intent is a navigable **upper atmosphere**, not an implausible trip to a
rocky core. A gas giant has no sharp gas-to-surface boundary: pressure and
temperature rise continuously until the atmosphere becomes a dense fluid.

## 2. Physical and gameplay model

### 2.1 Reference radius is a datum

For a gas or ice giant, `BodyDefinition::radius_m` is a named pressure datum,
recommended to be the conventional **1-bar radius**. It is useful for gravity,
altitude displays, cloud authoring, orbit planning, and exterior scale, but it
is not collidable.

Every body exposes collision semantics explicitly:

```text
SolidSurface
    collision radius = reference radius + SurfaceQuery elevation

AtmosphericEnvelope
    reference radius = named pressure datum
    no terrain collision
    terminal conditions = craft limits sampled from the atmosphere
```

Code must not infer solidity from `radius_m`. Prediction, impact markers,
surface-frame selection, camera floors, and debug placement all consume the
explicit body boundary kind.

### 2.2 One atmospheric state query

Rendering and gameplay require different projections, but they share one
authored vertical atmosphere. The pure-world seam should answer:

```rust,ignore
pub struct AtmosphereSample {
    pub pressure_pa: f64,
    pub density_kg_m3: f64,
    pub temperature_k: f64,
    pub speed_of_sound_m_s: f64,
    pub wind_body_m_s: DVec3,
    pub composition: AtmosphereComposition,
}

pub trait AtmosphericEnvelope {
    fn top_altitude_m(&self) -> f64;
    fn sample(&self, altitude_from_datum_m: f64, latitude_rad: f64)
        -> AtmosphereSample;
}
```

The exact Rust shape is an implementation decision; the invariant is not.
Regime selection, aero, HUD, damage, sky optics, and cloud placement must not
carry parallel pressure or density profiles.

The first model may be a hydrostatic, piecewise temperature/scale-height
profile. It must support negative altitude relative to the 1-bar datum and
remain monotonic and numerically stable across the navigable range. A later
model may add latitude, storm, and composition variation behind the same seam.

### 2.3 Entry, flight, and failure

- Crossing `atmosphere_top_m` enters `Medium::Atmosphere`, clamps warp to 1×,
  and hands translation to the local rigid-body backend exactly as terrestrial
  atmospheric flight does today.
- The persistent surface-local frame is reused as a **body-local tangent
  frame**. It does not require a terrain collider; its small coordinates,
  co-rotation, gravity, and Coriolis terms remain useful in free atmosphere.
- Air velocity is body rotation plus the sampled zonal/meridional wind. A
  globally static co-rotating airmass is an acceptable first rung, but the
  authored differential rotation already visible in the clouds must eventually
  affect air-relative velocity.
- On-rails trajectory prediction terminates at an `AtmosphericEntry` event for
  the first slice. It must not emit `SurfaceImpact` at the reference radius.
  Drag-aware prediction can follow behind the canonical atmospheric sampler.
- Craft failure evaluates static pressure, temperature, convective heating,
  and dynamic pressure against craft capabilities. Until those capabilities
  are fully authored, one explicit per-craft envelope is preferable to magic
  planet depths.
- A gas giant never enters a landed or surface-contact regime. The useful
  velocity frame near it is atmospheric/body-relative, not a fictional ground
  frame.

## 3. Rendering architecture

### 3.1 What exists

Thalos already has most of the generic mechanism:

- `GasGiantMaterial` synthesizes the exterior cloud deck from a latitude
  palette, band field, differential rotation, turbulence, vortices, haze,
  limb response, and rings.
- `thalos_body_render::clouds` performs a body-fixed, planet-centred spherical
  shell march and explicitly handles cameras below, inside, and above a cloud
  layer.
- The cloud path already owns 3-D density, vertical profiles, self-shadow,
  multiple scattering, foreground extinction, true cloud hit distance,
  temporal reconstruction, a near/far projection, and view-anchored cloud-sun
  transmittance.
- `big_space`, `ViewAnchor`, body-fixed weather coordinates, and the persistent
  local physics bubble already operate at planetary scale.

These are reusable mechanisms, not a shipping gas-giant interior. The current
cloud driver selects only `TerrestrialAtmosphere.clouds`; its density shapes,
vertical slab, and atmospheric coefficients are terrestrial. Conversely, the
gas-giant shader is a camera-exterior billboard and its 2-D optically thick
deck is not a volume that can be entered.

### 3.2 One weather field, several projections

The exterior disc and local volume must be projections of one gas-weather
authority:

```text
AtmosphereParams + GiantWeatherState
                  |
          shared macro field
     bands / jets / vortices / layer state
          /          |           \
         v           v            v
 exterior disc   local volume   light occlusion
 projection      raymarch       / environment
```

The current gas shader's band, edge-wave, and vortex math should move into a
shared shader library that returns physical or normalized weather signals, not
only final colour. The exterior projection integrates those signals into the
familiar disc; the near projection uses them to condition local 3-D density,
layer altitude, cloud type, wind, and albedo.

This is the load-bearing continuity rule: refinement adds resolved bandwidth
and parallax to the weather already seen from orbit. It must not generate a new
unrelated cloudscape when the camera crosses a distance threshold.

### 3.3 Bounded local volume

Do not raymarch the entire giant. The existing range- and footprint-banded
cloud path is the right shape:

- a camera-local detailed march covers resolved weather;
- density bandwidth and step cadence fall together with footprint;
- a reduced representation carries the unresolved tail;
- the exterior disc owns whole-body scale;
- the regimes overlap and crossfade by projected footprint and confidence,
  never a single altitude switch.

The present single shell can prove entry through one visible deck. Production
gas giants need a list of condensate/aerosol layers, each with altitude or
pressure bounds, optical properties, morphology, and coupling to the shared
macro weather. Jupiter-like authoring suggests an upper ammonia deck, a deeper
ammonium-hydrosulfide layer, and a deeper water-cloud/storm layer; ice giants
need different composition and colour. The first playable slice should ship
one primary deck plus continuous haze before attempting all layers.

### 3.4 Continuous atmosphere

Clouds alone cannot render the interior. A gas/ice giant also needs a continuous
participating atmosphere:

- molecular and aerosol scattering above and between decks;
- sample-to-camera extinction;
- sun-to-sample transmittance and planet occlusion;
- pressure/composition-dependent absorption with depth;
- shared sky/environment light for the ship and clouds;
- a deep-light response that trends toward darkness or bounded thermal
  emission rather than exposing space below a finite cloud shell.

This can reuse the shared atmosphere integration and future froxel machinery,
but coefficients and vertical structure are per body. The hard-coded
Earth-like cloud-lighting fallback is not a gas-giant model.

### 3.5 Exterior/interior handoff

The current gas billboard cannot be rendered around an interior camera: its
near ray-sphere root lies behind the viewer and the opaque deck becomes a false
enclosing surface. Visibility therefore becomes regime-owned:

- far/orbit: exterior `GasGiantMaterial`;
- overlap: exterior projection plus local atmosphere/cloud contribution;
- near/interior: local atmosphere and volume only;
- map: exterior projection only.

Rings remain real sibling geometry. Their visibility through the atmosphere,
their shadow on local cloud layers, and ringshine should use the same
transmittance model, but they are polish after a clean entry transition.

## 4. Authored data

`AtmosphereParams` currently describes exterior optics. It should grow—or be
superseded by a sibling schema that contains—four distinct concerns:

1. **Datum and envelope:** reference pressure, atmosphere top, vertical
   pressure/temperature/density profile, composition.
2. **Macro weather:** band/jet profile, differential rotation, vortices, storm
   potential, large-scale coverage.
3. **Cloud and haze layers:** pressure/altitude range, optical properties,
   palette/albedo, morphology, precipitation/lightning potential.
4. **Terminal environment:** the atmosphere supplies state; craft data supplies
   survivable pressure, temperature, heating, and load limits.

Exterior-only palette fields may remain as derived art controls during
migration, but they cannot remain the sole authority once a local volume exists.
The data model must keep gas and ice giants meaningfully different rather than
making both tinted versions of terrestrial water clouds.

## 5. Initial vertical slice

**Auron** is the recommended first target: its strong band palette, haze, ring
system, and differential motion make exterior/interior continuity easy to
judge. The slice is deliberately narrow:

- treat `radius_m` as the 1-bar datum and remove solid collision for Auron;
- author one navigable atmospheric envelope and one primary cloud/haze deck;
- derive the local macro weather from the same band/vortex field as the disc;
- render orbit → approach → cloud-top → cloud-interior → deep-haze without a
  hard swap;
- enable 1× local aerodynamic flight with pressure/density readouts;
- destroy the craft at an explicit provisional pressure/thermal envelope;
- keep multiple deep decks, lightning, precipitation, and drag-aware
  prediction for later phases.

Nereus follows as the falsification target. If the abstraction cannot produce
an ice giant without copying Auron's Saturn-like layer stack, the schema is too
Jupiter-shaped.

## 6. Implementation phases

| ID | Phase | Outcome | Est. |
|----|-------|---------|------|
| GG-0 | Boundary and atmosphere authority | Explicit solid-vs-atmospheric boundary semantics; reference-pressure datum; shared atmospheric sampler; prediction emits entry rather than impact | M |
| GG-1 | Shared weather and near renderer | Gas macro field shared by disc and volume; one primary volumetric layer; gas sky/haze; footprint-based exterior/interior handoff | L |
| GG-2 | Atmospheric flight and failure | Regime/aero/HUD consume the gas sampler; body-relative wind seam; pressure/temperature/q/heating telemetry and craft terminal envelope | M |
| GG-3 | Auron playable slice | Deterministic approach/top/interior/deep presets, ring-aware framing, performance and continuity tuning, user fly-through | M–L |
| GG-4 | Giant-world fidelity | Multiple decks, storms, precipitation/lightning, deep absorption/emission, ringshine/transmittance, Nereus differentiation, drag-aware prediction | L–XL |

A visual-only prototype can be built inside GG-1, but it is not completion:
without GG-0 the reference radius still behaves as hidden ground, and without
GG-2 the ship flies through vacuum until an arbitrary death sphere.

Rough planning scale:

- visual prototype: 3–6 focused days;
- credible Auron gameplay MVP (GG-0 through GG-3): 2–4 focused weeks;
- polished multi-giant feature including GG-4: 1–2 months.

These are scope estimates, not schedule commitments. First-build integration
and visual calibration dominate the uncertainty, not the raymarch itself.

## 7. Acceptance

### Agent-verifiable

- Deterministic `gas-approach`, `gas-cloud-top`, `gas-interior`, and
  `gas-deep` screenshot presets plus a cold exterior/interior comparison.
- No shader, pipeline, missing-layer, or capture-receipt failure in any preset.
- A typed comparison axis isolates the exterior/local ownership transition.
- Weather landmarks retain body-fixed identity and position across the
  handoff.
- No `SurfaceImpact` event occurs at the pressure datum; prediction produces a
  typed atmospheric-entry event.
- Runtime diagnostics record body, atmosphere regime, altitude from datum,
  pressure, density, temperature, air-relative speed, dynamic pressure,
  heating proxy, optical depth, active render tier, and terminal cause.
- Cloud/atmosphere GPU time and persistent memory are recorded at 1080p and
  1440p. The current terrestrial cloud budget is a reference, not an assumed
  gas-giant budget.

### User-verifiable

- Orbit-to-interior flight has no visible shell pop, scale jump, or change to
  the weather system being approached.
- Cloud-top flight reads as kilometre-scale weather, not a planet texture
  enlarged around the camera.
- Entering a cloud produces continuous extinction and loss of visibility;
  leaving it restores the same surrounding formation.
- Controls, drag, wind, and loss of warp make the atmosphere feel occupied.
- Descent ends from a legible environmental limit, never from striking an
  invisible surface.
- Auron and Nereus feel like different worlds, not recoloured presets.

## 8. Non-goals

- Navigating to a rocky core or simulating the metallic-hydrogen interior.
- A full fluid-dynamics weather solver in the first playable slice.
- Raymarching a whole planetary diameter at local-cloud resolution.
- Giving a gas giant terrain, a landing state, a camera ground floor, or a
  surface height source.
- Making the exterior impostor support an interior camera. It remains the
  correct far projection and stands down when the local projection owns the
  view.
- Spectrally exact chemistry. Composition must shape authoring and optics, but
  real-time RGB approximations remain acceptable.

## 9. Decisions to resolve at pickup

These are not blockers to recording the direction; they become ADR candidates
when GG-0/GG-1 is pulled into active work:

1. Whether `AtmosphereParams` grows into the shared physical schema or a new
   `GiantAtmosphere` replaces the exterior-only shape.
2. The named pressure datum and authored radius convention for every existing
   giant. One bar is the recommended default.
3. Whether the first terminal envelope is per craft, per part, or an aggregate
   derived from part capabilities. The long-term answer must be craft-owned.
4. Whether gas-weather state reuses `CloudWeatherField` with richer channels or
   introduces a composition/layer-aware sibling behind shared sampling
   helpers.
5. How oblateness enters local shell intersection and altitude. Giant bodies
   should eventually use an ellipsoidal datum; a spherical Auron MVP is
   acceptable only if the seam does not bake sphere-only assumptions into the
   atmospheric authority.

## References

- `docs/rendering/atmosphere.md` — shipping giant impostor and atmospheric
  rendering.
- `docs/rendering/clouds.md` — spherical volume, weather authority, lighting,
  temporal reconstruction, and near/orbit handoff.
- `docs/simulation/aerodynamics.md` — force model and atmospheric sampling.
- `docs/simulation/regimes.md` — translation ownership and atmosphere warp
  policy.
- [NASA: Jupiter facts](https://science.nasa.gov/jupiter/jupiter-facts/) —
  cloud layers and the continuous gas-to-fluid interior.
- [NASA: Galileo Jupiter Atmospheric Probe](https://science.nasa.gov/mission/galileo-jupiter-atmospheric-probe/)
  — measured entry through the Jovian atmosphere.
