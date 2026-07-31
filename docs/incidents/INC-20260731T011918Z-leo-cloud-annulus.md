# INC-20260731T011918Z-leo-cloud-annulus: clouds covered only an annulus of the disc from low orbit

- **Date:** 2026-07-31 · **Surface:** any LEO view (~200–1900 km altitude); `just screenshot cloud-leo`

## Symptom

From a ~404 km orbit the cloud field did not span the visible planet: the player saw
volumetric clouds in a patch around the sub-craft region dissolving into bare ocean,
while `cloud-limb` (200 km) showed the complement — clouds near the horizon, bare
near-field. `THALOS_SCREENSHOT_CLOUD_TIER=near` at 404 km rendered **zero clouds
anywhere**; `=far` was identical to the composite. The two tiers flipped regimes on a
per-ray knife edge that moved with resolution/FOV, so different sessions saw the hole
in different places.

## Root cause

Two mirrored-integrator defects in the near-march reach law
(`thalos::atmosphere::cloud_march_stop_m`):

1. **The budget frontier was clamped to an absolute camera distance.** The final
   cap-phase return clamped the stop to `CLOUD_MARCH_REACH_M` (300 km) — a constant
   documented, and used everywhere else (`get_ray`), as a **segment length from the
   shell entry**. Any camera more than 300 km from the deck got a frontier *behind*
   its own shell entry: the marcher broke on its first iteration and the near tier
   rendered nothing. Only rays whose probe budget spilled into the cap phase hit the
   clamp, so the failure switched on per-ray, mid-disc.
2. **The composite evaluated the marcher's budget law with the wrong pixel angle.**
   The cloud compute target renders at `resolution_scale` (default ⅔) of the
   viewport; the composite's ownership partition (`far_ownership`) used its own
   full-resolution pixel angle — 1.5× finer. Across the band where the two angles
   put the same ray in different reach regimes, the composite kept blocking the far
   tier over rays the marcher had already abandoned: neither tier drew, leaving the
   bare annulus.

## Fix

`cloud_march_stop_m` clamps every phase exit to `t_entry + CLOUD_MARCH_REACH_M`
(entry-relative, matching `get_ray`). The game driver publishes the cloud target
width in `BodySkyExtra::cloud_time.y`, and the composite derives its ownership
pixel angle from that width, so both integrators run the identical budget law.
Verified by `cloud-leo` (new preset: 404 km oblique, the player's framing):
near-only now covers the whole disc, composite is continuous to the limb;
`cloud-limb` / `cloud-cruise` / `cloud-planet` unchanged or improved.

## Recurrence signal

`just screenshot cloud-leo` with `THALOS_SCREENSHOT_CLOUD_TIER=near` — a bare disc
(or any camera-centred bare annulus in the composite) means the reach law or the
ownership mirror has drifted again. When touching either side, grep for the pairing:
`cloud_march_stop_m` callers must feed the SAME pixel angle the marcher steps with.
