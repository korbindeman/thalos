# INC-20260727T004856Z — slope-projected noise draws contour moiré, not striation

## Symptom

Close views of the massif (`massif-valley` at ~1.8 km, `THALOS_TERRAIN=diffusion`)
showed rock faces covered in a fine swirling "agate" / wood-grain pattern: dense
thin contour lines closing into rings and eyes, densest where the slope turns.
It sat on top of the terrain shape rather than in it, and it was worst up close
— nothing retired it. Reported as "post-diffusion terrain sharpening makes
mountains look weird up close".

## The tell

The pattern follows the *normal field*, not the ground. Contour-like closed
loops around slope extrema, wood grain along ridges — the signature of a texture
whose **phase** is a function of surface orientation.

## Mechanism

`tile_terrain.wgsl`'s rock layer built its fall-line gully striation as

```wgsl
let gully_uv = vec3<f32>(dot(p, across) / 24.0, dot(p, fall) / 150.0, 0.0);
```

`across` / `fall` are per-fragment directions derived from the geometric normal,
and `p` is the vertex-carried body-fixed position wrapped to `TILE_WRAP_M`
(8192 m) — so `|p|` runs to several kilometres from its anchor. The projection
therefore has

```
∂(phase) / ∂(slope angle) = |p| / wavelength
```

i.e. tilting the surface by `24 / 8192 ≈ 0.003 rad ≈ 0.2°` slides the pattern a
whole stripe. Real terrain turns orders of magnitude faster than that across a
face, so the striation stopped tracking position and became an interference
pattern between the noise lattice and the mesh's normal field.

This is **not** a precision problem — `p` is bounded and f32 is ample. The
sensitivity is exact arithmetic doing what it was asked to.

## Diagnosis path (what ruled what out)

1. **The diffusion height field is innocent.** A scratch probe walked a 60 m
   ground transect at 5 cm spacing through `DiffusionSurface::sample_height_m`
   on that face: every sample distinct, median riser 0.026 m, no staircase, no
   ripple. The erosion and fine bands were the first suspects and were wrong.
2. **`THALOS_TERRAIN_INSPECTION=1/2` produced captures byte-comparable to the
   lit one** on the tile path, so the albedo-vs-normal split it exists to
   provide was unavailable. That comparison was rejected rather than read
   (BL-20 discipline); the switch appears not to reach the tile material.
3. **Zeroing the gully term** (WGSL hot-reload, ~3 s) removed the entire
   pattern and left clean rock — attribution to one term, not one band.

## Fix

Replace the anisotropic *coordinate transform* with a directional *filter*: an
isotropic body-space value noise at a 32 m wavelength, low-passed with 5 taps
walking ±48 m down the fall line. Same anisotropy, bounded sensitivity — a tap
moves by `span/2 × angle`, ~170× less than the projection moved the phase. The
slope frame now **orients** the filter instead of setting its phase.

Matched A/B in one host lifetime (`artifacts/visual/runs/sharpen-probe/`,
`ab_before.png` / `ab_after.png`): high-frequency energy over the face falls
8.00 → 4.77 mean |Laplacian|, and the whorls are visually gone.

## Recurrence tell

Any procedural term of the form `noise(dot(p, <per-fragment direction>))` where
`p` is a large position. If a texture's swirls follow shading rather than
ground, check for a projection onto a varying frame before suspecting precision,
LOD, or the height source. Wavelengths must also divide `TILE_WRAP_M` exactly —
the old term used 24 m, which does not.

## Note on capture comparisons

Frames from *different host lifetimes* differ globally in haze/cloud state
(measured: 20–29 mean |Δ| per channel on an unchanged scene, vs 3.3 for a
matched pair). Any tile-shader A/B must be two hot-reloads inside **one** host
run, or the pixel metrics are noise.
