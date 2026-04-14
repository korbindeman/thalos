# Gas Giant Rendering — Implementation Reference

A technique guide for rendering realistic gas giants as flat impostors (no volumetrics), with a migration path toward volumetric atmospheres later. Targets Bevy/Rust but the pipeline is engine-agnostic.

## Goals and constraints

- **Visual target:** Hubble/Juno-style imagery — zonal banding, sheared filaments, vortex chains, turbulent poles.
- **Budget:** Fragment shader only, single draw per planet, no simulation state, no render targets beyond the impostor itself.
- **Per-planet authoring:** ~5 parameters should describe anything from Jupiter to Neptune to a brown dwarf.
- **Forward-compatible:** Work done here must not be thrown away when volumetrics come online.

## Core idea

Gas giants are **zonal bands** (latitude-parallel jets) that get turbulently sheared at their boundaries, with long-lived vortices embedded in the flow. We fake all three with 2D noise techniques on a sphere, keyed on latitude.

The pipeline, start to finish:

1. Sphere UV from surface normal
2. Differential rotation (longitudinal scroll per latitude)
3. Primary domain warp — large-scale turbulence
4. Shear-driven edge wave — Kelvin–Helmholtz signature at band boundaries
5. Edge vortex chain — budding-off eddies along shear layers
6. Analytic large vortices — Great Red Spot and named features
7. Curl noise with boundary sign flip — counter-rotating eddies
8. Band palette lookup — 1D LUT sampled by warped latitude
9. Limb darkening, terminator warmth, Fresnel rim

Steps 4, 5, and 7 are what make the planet feel alive rather than a scrolling texture.

---

## Pipeline stages

### 1. Sphere UV

Impostor fragment has an interpolated sphere normal. Convert to equirectangular UV:

```glsl
vec2 sphereUV = vec2(
    atan(n.z, n.x) / (2.0 * PI) + 0.5,
    asin(n.y) / PI + 0.5
);
```

Use `n.y` directly (not the UV `v`) as `lat ∈ [-1, 1]` for all latitude-keyed lookups — it avoids the equirectangular pinch at the poles.

### 2. Differential rotation

Scroll longitude per latitude from a 1D speed LUT:

```glsl
uv.x += time * texture(speedLUT, lat * 0.5 + 0.5).r;
```

The LUT should be signed — retrograde belts are visually important on Jupiter. Keep speeds small (~0.001–0.01 UV/sec) or the planet looks like a washing machine.

### 3. Primary domain warp

Two levels of fBm warping, per Inigo Quilez's domain warping article. This produces the large-scale "dragged taffy" turbulence:

```glsl
vec2 q = vec2(fbm(uv * 3.0), fbm(uv * 3.0 + 5.2));
vec2 r = vec2(fbm(uv * 3.0 + 4.0*q + 1.7),
              fbm(uv * 3.0 + 4.0*q + 9.2));
float warpedLat = lat + warpAmp(lat) * fbm(uv * 6.0 + 4.0*r);
```

`warpAmp(lat)` scales with latitude — low at the equator, high at mid-latitudes and poles. A single `turbulence(lat)` LUT should drive this and later stages.

### 4. Shear-driven edge wave

Band edges are Kelvin–Helmholtz shear layers. Compute shear as the derivative of the speed LUT:

```glsl
float shear = abs(dFdy(speed(lat)));
```

Drive a secondary latitude displacement whose amplitude scales with shear:

```glsl
float edgeWave = sin(uv.x * k + time * omega + fbm(uv * 8.0) * TAU);
warpedLat += shear * 0.04 * edgeWave;
```

The fBm inside the phase randomizes wavelength so it doesn't read as a pure sine. Time scale: `omega ≈ 0.05–0.2 rad/s`. Faster than that reads as water.

### 5. Edge vortex chain

Along high-shear latitudes, spawn a chain of weak vortex primitives that bud off, drift, and fade. Stateless via hashing:

```glsl
// for each edge slot near current pixel:
float seed = hash(bandIndex, edgeSlot, floor(time / lifetime));
vec2 center = vortexCenter(bandIndex, edgeSlot, time, seed);
float age = fract(time / lifetime);
float radius = baseR * sin(age * PI);        // grow then shrink
float strength = vortexStrength * sin(age * PI);
// apply swirl to local UV (see step 6)
```

Key points:

- **Stateless:** hash `(band, slot, epoch)` so spawn/drift/death is deterministic without buffers.
- **Spatial culling:** bin vortices into a coarse lat/lon grid; each pixel considers only 2–3 nearest edge-vortices.
- **Spacing:** roughly 1/8 to 1/4 of the band width between slots.
- **Lifetime:** 30–120 seconds. Shorter looks frantic.

### 6. Analytic large vortices

Named features (GRS, white ovals) as a small uniform array: `(center_uv, radius, strength, color_tint)`. For each, apply a swirl:

```glsl
vec2 d = uv - center;
float dist = length(d);
float angle = strength * smoothstep(radius, 0.0, dist);
vec2 swirled = rotate(d, angle) + center;
// sample warped noise at `swirled` and blend tint by smoothstep falloff
```

10–30 vortices total is plenty. They tint the band lookup rather than replacing it so they sit *in* the flow.

### 7. Curl noise with boundary sign flip

Single curl noise field applied as an additional small warp. Multiply its contribution by `sign(speed(lat) - speed(lat + ε))` so eddies on either side of a band edge counter-rotate. This one line of code is what makes the interface read as physically turbulent rather than noisy.

Crank amplitude with `turbulence(lat)` so poles go fully chaotic (Juno-style).

### 8. Band palette lookup

1D LUT texture (64–256 px) indexed by `warpedLat`. This is the single biggest art-direction lever — swap LUTs to go from Jupiter to Neptune to brown dwarf. Store as gradient stops in the planet's KDL definition and bake to texture at load.

### 9. Lighting and limb

On the flat impostor, cheap but high-impact:

- **Limb darkening:** `color *= pow(NdotV, 0.4..0.6)`
- **Terminator warmth:** thin warm rim where `NdotL ≈ 0` on the lit side
- **Fresnel haze:** faint desaturated blue/white rim on the lit limb, stand-in for Rayleigh scattering

When volumetrics come online, stage 9 is the hook point — the real atmosphere shader replaces these fakes.

---

## Authoring parameters per planet

Minimum viable set, all storable in KDL:

| Parameter | Type | Notes |
|---|---|---|
| `band_palette` | gradient stops | Baked to 1D LUT |
| `band_speeds` | per-latitude signed array | Baked to 1D LUT |
| `turbulence_profile` | per-latitude scalar | Drives warp amp, curl amp, edge wave amp |
| `named_vortices` | array of `(uv, radius, strength, tint)` | GRS etc. |
| `overall_tint` / `contrast` | vec3 / float | Final color tweak |

Neptune, Saturn, Uranus are all the same shader with different entries in this table.

---

## Fidelity notes

**What makes it look real:**

- **Latitude-dependent everything.** Real gas giants are laminar at the equator and chaotic at the poles. A single `turbulence(lat)` curve should modulate warp amplitude, curl strength, edge wave amplitude, and noise octave count. This single observation elevates output more than any other tweak.
- **Slow time scales.** Edge waves over tens of seconds, vortices over minutes, GRS effectively static. If it animates fast enough to notice casually, it's wrong.
- **Counter-rotation across boundaries.** The sign-flip trick in stage 7 is the difference between "noise" and "fluid."
- **Limb darkening is non-optional.** A flat-lit gas giant looks like a decal. `pow(NdotV, 0.5)` alone is a huge step up.
- **Two-layer parallax cheat.** Sample the band pattern twice with slightly different warps, offset by `viewDir.xy * 0.01`, blend. Hints at depth before real volumetrics exist.

**What to avoid:**

- **Tiling noise.** Use noise functions that wrap in longitude or mask the seam at the back of the planet. A visible seam kills everything.
- **Equirectangular pole pinch.** Don't sample textures with `v` directly near poles — use the 3D normal for noise inputs and reserve `v` for the band LUT only.
- **Over-animating.** Scrolling noise too fast is the single most common failure mode. When in doubt, halve the speed.
- **Uniform detail.** Constant-amplitude noise across all latitudes looks like marble, not a planet.
- **Fighting the shader with textures.** Resist the urge to paint a Jupiter texture and project it. You lose animation, parallax, and per-planet variation, and it won't match volumetrics later.
- **Simulating anything.** KH instabilities, vortex dynamics, band formation — all tempting, all wrong at this stage. The fakes are indistinguishable at impostor scale and cost 1000× less.
- **Premature volumetrics.** Don't try to fake ring shadows, cloud self-shadowing, or multi-layer parallax properly in 2D. Stub them and do them right when the volumetric pass lands.

---

## Performance

- **Half-res impostor.** Gas giant detail is low-frequency; rendering the impostor at half resolution and upsampling is usually indistinguishable from full-res and is a clean lever when many planets are on screen.
- **Octave budget.** 4–5 fBm octaves is plenty. Beyond that you're paying for detail the band LUT smears out anyway.
- **Vortex culling.** Edge vortex chains must be spatially binned. Per-pixel iteration over all vortices is a trap.
- **LOD by angular size.** Below ~20 px, drop edge waves and vortex chains entirely — nobody will see them. Below ~5 px, drop to a static cubemap bake.

---

## Migration to volumetrics

When the volumetric pass comes online, the stages map forward cleanly:

- Stages 2–7 become **displacement and density inputs** for cloud layers instead of 2D warps. Same noise, same LUTs, sampled in 3D.
- Stage 8 becomes **absorption coefficient lookup** along the view ray.
- Stage 9 is replaced by real single/multi-scatter integration.
- Edge waves and vortex chains become **height displacements** on cloud decks, which is where they were always trying to be.

Nothing in this doc is throwaway work. The authoring parameters and LUTs carry forward unchanged.

---

## References

- Inigo Quilez — [domain warping](https://iquilezles.org/articles/warp/) and [fBm](https://iquilezles.org/articles/fbm/)
- Sebastian Lague — procedural planets series (YouTube), gas giant episode
- `bevy_terrain` patterns for LUT-driven material authoring
- Juno mission imagery (NASA/SwRI/MSSS) for pole and high-latitude reference
- Hubble OPAL program imagery for equatorial band reference
