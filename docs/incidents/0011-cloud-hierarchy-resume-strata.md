# INC-0011: Cloud hierarchy posterized volumes into horizontal strata

- **Status:** Fixed
- **Date:** 2026-07-21 (observed) / 2026-07-21 (fixed)
- **Severity:** visual
- **Surface:** `just screenshot cloud-cruise`, `cloud-runway`, and `cloud-sunset`

## Summary

The initial CLOUD-3 empty-space hierarchy replaced organic cauliflower clouds
with stable horizontal shelves. Its weather, local base/top, and broad-shape
heuristics advanced a shallow view ray by nonuniform kilometre-scale leaps,
then resumed full density evaluation near similar normalized heights. Across
neighbouring pixels those resume points formed repeated height isosurfaces,
posterizing a continuous 3-D field. Temporal reconstruction and CLOUD-4
lighting changed the noise or colour of the shelves but did not create them.

The fix deletes the heuristic leaps and the density remaps introduced beside
them. The marcher again samples continuous typed 3-D density at its ordinary
cadence. Safe range savings come from fading only sub-pixel fine erosion and
reusing the deliberately low-frequency macro modulation once per ray; neither
optimization changes density sample positions.

## Symptoms

- Cruise clouds contained obvious, evenly spaced horizontal tonal shelves.
- Runway clouds acquired flat nested contours and posterized interiors.
- The pattern was stable with temporal history disabled.
- The earlier `9d91467` captures had rounded, irregular silhouettes under the
  same camera and weather seed.

## Evidence

Matched cruise captures changed one factor at a time:

- `cloud3_ab_no_cloud4_cruise.png`: removing analytic CLOUD-4 atmosphere,
  powder, multi-scatter, and the output soft clip left the shelves in the same
  positions; lighting was ruled out.
- A temporal-off capture retained the same strata; CLOUD-2 reconstruction was
  ruled out.
- `cloud3_ab_no_hierarchy_cruise.png`: disabling all three hierarchy leaps
  removed the stable shelves while keeping the current density and lighting.
- `cloud3_ab_envelope_only_cruise.png`: restoring only the base/top leap was
  sufficient to reproduce the bad structure. Restoring the weather/macro
  gates also reintroduced weaker aligned shelves, so the hierarchy contract as
  a whole was unsafe rather than one threshold merely needing retuning.
- `cloud3_ab_checkpoint_density_cruise.png`: the last organic typed-density
  function, sampled continuously inside the current CLOUD-2 reconstruction,
  restored rounded cauliflower structure.

The corrected 2560×1440 High sunset probe uses a 1712×960 cloud target and
measures 2.471 ms mean / 2.476 ms p95, below the 3.5 ms program target without
heuristic empty-space jumps.

## Hypotheses considered

- **Temporal reprojection / neighborhood clamp:** ruled out because temporal-off
  retained the same spatial shelves.
- **Analytic atmosphere, powder, or output compression:** ruled out by removing
  the complete CLOUD-4 light-transport slice with negligible geometric change.
- **Individual height warp, mid-scale shape, view-step stretch, weather tuple,
  or shadow count:** each changed emphasis but did not eliminate the strata.
- **Heuristic hierarchy resume positions:** confirmed by the all-off and
  envelope-only matched captures.
- **CLOUD-3 density additions:** not the primary shelf generator, but they made
  the hierarchy-free result visibly softer and less organic; they were removed
  from this checkpoint instead of preserving unverified complexity.

## Root cause

The hierarchy was described as conservative, but none of its levels stored a
true upper bound on density over the skipped interval. A weather maximum did
not bound local typed density, a broad-shape proxy did not bound eroded mass,
and a radial base/top estimate was invalid for shallow curved-shell crossings.
The resulting data-dependent leaps synchronized resume locations around
similar shell heights. Temporal accumulation then stabilized those biased
samples into visible horizontal strata.

## Fix

- Delete weather, base/top, and broad-occupancy leaps from the view marcher.
- Restore the proven typed Perlin/Worley density and vertical profiles.
- Sample weather continuously at every ordinary view step.
- Fade only fine boundary erosion from 10–22 km, where its authored 450 m
  features become sub-pixel.
- Evaluate the 21.6 km anti-tiling modulation once per short view segment and
  reuse it for view and shadow density.

## Prevention & recurrence signals

- Empty-space skipping may leap only from a conservative max-density bound over
  the entire skipped interval (for example a max mip / occupancy volume), not
  from a correlated proxy or an estimated profile boundary.
- Performance work that changes march positions requires matched grazing-angle
  cruise and runway captures, not only timing and top-down images.
- `docs/rendering/clouds.md` records the conservative-skip invariant.
- A recurrence presents as stable shelves or contours that survive temporal-off
  and lighting A/B tests but disappear when nonuniform ray advances are removed.
