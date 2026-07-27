# INC-20260724T204059Z-standard-path-never-set-camera-exposure: the Bevy universe ran at Blender's default exposure

- **Status:** Fixed
- **Date:** 2026-07-24 (observed) / 2026-07-24 (fixed)
- **Severity:** visual
- **Surface:** every ship-view scene with `StandardMaterial` surfaces — hull, gear,
  structures, runway/apron, and (loudest) the NTR-X1 tile terrain. Found on
  `massif-valley` under `THALOS_TERRAIN=diffusion THALOS_TILE_RENDERER=1`.

## Summary

Nothing ever inserted a Bevy [`Exposure`] component on the ship camera, so the
entire `StandardMaterial` half of the scene rendered at Bevy's
`EV100_BLENDER` default (9.7) while the `thalos::lighting` spine converted the
same scene flux to radiance through its own constants. The two disagreed by a
factor of 2.77 (~1.5 stops). Terrain on the tile renderer therefore came out
about a stop hot and visibly desaturated next to identical terrain rendered
through udlod, which flattened every mountain-texturing decision made on top of
it (NTR-X4). The fix derives the camera exposure from the spine's own constants
so the two universes agree by construction, and re-bridges the sky-fill term
that had been tuned against the wrong exposure.

## Symptoms

- Tile-rendered mountains read as pale, low-contrast, milky grey — "washed out"
  — with weak colour and no depth in gullies, at every distance including the
  near field.
- The *same terrain content, framing, sun and epoch* through udlod looked
  saturated and contrasty.
- On the standard path, runway/apron asphalt rendered **brighter** than the
  sunlit grass beside it — backwards for a ~0.08-albedo material.

Repro (before the fix):

```bash
THALOS_TERRAIN=diffusion THALOS_TILE_RENDERER=1 just screenshot massif-valley
THALOS_TERRAIN=diffusion THALOS_TILE_RENDERER=0 just screenshot massif-valley
```

## Evidence

Ground-region statistics over the two captures (`artifacts/visual/runs/ntr-x4-debug/`,
`udlod_valley.png` vs `floor006_valley.png`) — the decisive numbers:

```
udlod (spine)        lum mean 0.232  p05 0.125  p95 0.342  sat 0.250
tiles (Bevy PBR)     lum mean 0.450  p05 0.116  p95 0.574  sat 0.161
```

Twice as bright, a quarter less saturated, in the near field as much as the far
— so not aerial perspective. Spaceport patch ratios (`spaceport-aerial`, where
udlod ground and Bevy structures share a frame) gave the second tell:

```
before   apron/grass 1.12   roof/grass 3.83      (asphalt brighter than grass)
after    apron/grass 0.71   roof/grass 3.17      (grass unchanged: 0.201 → 0.194)
```

## Hypotheses considered

Each ruled out by a capture before moving on:

- **Volumetric clouds compositing over the terrain** — `THALOS_SCREENSHOT_CLOUD_COVERAGE=0`
  produced an identical frame.
- **Detail normals aliasing into a bright haze** — zeroing `normal_offset`:
  identical.
- **Grazing-angle Fresnel specular** — forcing `roughness = 1.0`: identical.
- **`SHADOW_FLOOR` filling shadows to 40 %** — 0.4 → 0.06 barely moved the
  frame. That is its own finding: **terrain receives no cast shadow at showcase
  distances** (`shadow_f ≈ 1`), so the floor was never the binding term.
- **The material layer stack being too bright** — the layer-weight
  visualization (`layers_valley.png`) showed the pale regions are ordinary rock
  at 0.082 linear, *darker* than the vegetation it was out-brightening. Albedo
  could not explain it.
- **Aerial perspective / `BodySky` compositing against tile depth** — plausible
  (and the reason the tile depth path was checked), but the near-field bands
  are hot too, and both paths show the same far-field gradient.

## Root cause

Both paths are handed the same heliocentric scene flux and each turns one unit
of it into display radiance its own way. For a Lambert-ish surface of albedo `a`
at incidence `cos θ`:

- spine: `a · cos θ · flux · SCENE_FLUX_SCALE · SURFACE_DIRECT_SCALE`
  — its BRDFs return bare radiance factors, the 1/π family living in
  `SCENE_FLUX_SCALE` (0.5 × 0.23 = 0.115 per unit flux);
- Bevy: `a/π · cos θ · (flux · LUX_PER_SPINE_FLUX) · view.exposure`
  — 1000 lux per unit flux, times whatever `view.exposure` is.

Equating them leaves exposure as the only free term:
`exposure = SCENE_FLUX_SCALE · SURFACE_DIRECT_SCALE · π / LUX_PER_SPINE_FLUX`
= 3.61e-4, i.e. EV100 ≈ 11.17. **No `Exposure` was ever inserted**, so Bevy used
`EV100_BLENDER` = 9.7 → 1.00e-3, a factor of 2.77 too much light on every
`StandardMaterial`.

The error hid for as long as it did because the Bevy universe had only ever been
compared *against itself*: `LUX_PER_SPINE_FLUX` was eyeball-tuned "until the hull
reads at the same brightness as the terrain beside it", and only the **product**
`LUX_PER_SPINE_FLUX × view.exposure` reaches a Bevy-lit fragment — so a hull
albedo chosen under the wrong exposure absorbs the error invisibly. Putting
*terrain* on the standard path finally placed the two universes side by side on
the same material at the same instant, where a 1.5-stop offset is unmissable.

A second term was hiding behind the first: the env cubemap
(`GeneratedEnvironmentMapLight`) is painted in **scene-flux units** (~0.1–1.3)
while Bevy consumes it in the directional light's photometric space, so at
`PROBE_INTENSITY = 1.0` its diffuse contribution is ~three orders of magnitude
short of the sky it depicts — effectively zero. `AMBIENT_SKY_LUX_GAIN` had been
held at 0.2 on the theory that the env map carried the rest, which made the flat
ambient the *entire* sky fill at a fifth of its proper strength. Invisible while
everything was 2.77× hot; with the exposure corrected it crushed gullies to
black and turned asphalt into a hole in the ground.

## Fix

- `thalos_body_shading::spine_parity_exposure(lux_per_spine_flux)` derives the
  parity `Exposure` from the spine's own mirrored constants (`SCENE_FLUX_SCALE`,
  `SURFACE_DIRECT_SCALE`), and `camera::spawn_camera` inserts it on the ship
  camera. Structural: retuning either side keeps parity as long as it flows
  through that function.
- `AMBIENT_SKY_LUX_GAIN` 0.2 → 0.7, the sky fill now bridged by the same
  flux→lux constant. 1.0 (the whole sky irradiance) overfills against the
  spine, whose own sky term carries an artistic `SURFACE_SKY_SCALE`; 0.7 is
  where the two shadow fills meet by measurement. **If the env map is ever put
  on physical units, this must come back down** or the sky is double-counted.
- The space-regime ambient stand-ins (`AMBIENT_DAY_BRIGHTNESS`,
  `AMBIENT_NIGHT_BRIGHTNESS`) scaled ×2.77 — they were eyeball-tuned at the old
  exposure, so carrying the ratio through leaves the space regime unchanged.

Result on the probe frame: shadow fill p05 0.134 vs udlod's 0.125, saturation
0.259 vs 0.250. The residual lit-level difference (p95 0.429 vs 0.342) is the
two paths' *material models* disagreeing at this site — the tile path paints the
canonical alpine rock/snow bands, udlod paints its own dry-grass tan — not a
remaining exposure error.

## Prevention & recurrence signals

- **Standing rule:** a Bevy-lit surface and a spine-lit surface in the same
  frame must be reconciled through `spine_parity_exposure`, not by tuning a
  material's albedo until it looks right. Only the product
  `LUX_PER_SPINE_FLUX × view.exposure` reaches the fragment, so eyeball-tuning
  either one alone can absorb an arbitrary error — and will, silently.
- **Never rely on Bevy's default `Exposure`** in a scene with a custom
  photometric spine. It is `EV100_BLENDER` (9.7), an interior-lighting value
  with no relation to this project's flux units.
- **Tell:** standard-path surfaces reading hot, flat, and desaturated next to
  spine-lit ground; low-albedo materials (asphalt, dark hull) rendering
  *brighter* than sunlit vegetation. Measure it as ground-region mean luminance
  + mean saturation on matched `THALOS_TILE_RENDERER=1` / `=0` captures rather
  than by eye — the wash reads as haze and invites chasing the atmosphere.
- **Related open item:** the env cubemap's unit mismatch is untouched here (it
  only contributes specular flavour today). Fixing it is `gfx` W7/F7 and must be
  paired with lowering `AMBIENT_SKY_LUX_GAIN`.
