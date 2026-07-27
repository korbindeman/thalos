# INC-20260726T225008Z-defaulted-seam-method-zeroed-by-decorator: a defaulted `SurfaceQuery` method the flatten decorator forgot to forward zeroed canopy planet-wide

- **Date:** 2026-07-26 · **Surface:** `just screenshot forest-stand` (any Thalos surface view)

## Symptom

Immediately after adding `SurfaceQuery::canopy_coverage` and routing vegetation
placement through it, **every tree on the planet vanished**. The ground rendered
correctly (its canopy came through a different route — per-vertex
`MaterialBands`), so the scene looked like plausible treeless terrain rather than
a failure.

Nothing logged an error. The tell was in `artifacts/diagnostics/runtime.jsonl`:

```
"event":"forest_stand_site","lat_deg":0.0,"stand_coverage":0.0
```

`stand_coverage: 0.0` at `lat_deg: 0.0` — the capture's forest-site search had
scored every candidate on the daylight hemisphere as zero and fallen back to the
sub-stellar point. A field that is zero *everywhere* is a plumbing failure, not a
tuning one.

## Root cause

`canopy_coverage` was added as a **defaulted** trait method returning `0.0`, and
implemented on `ProceduralSurface` and `DiffusionSurface`. But Thalos's surface is
not either of those directly: it is wrapped in `FlattenedSurface`, the decorator
that overlays the spaceport pad. That decorator forwards each seam method
explicitly (`landcover_moisture`, `sample_bands_d`, `radius_m`, …) and had no
arm for the new one — so every call landed on the trait default and returned
`0.0` for the whole body.

The failure is silent by construction: a defaulted method plus an explicit
forwarding decorator means *forgetting* is indistinguishable from *declining*.
There is no unimplemented-method error, and the default is a legitimate value for
backings that genuinely have no landcover model (airless bodies, plain oceans).

A wrong hypothesis worth recording: the first suspicion was cost, because the
same change had just timed the capture client out (see below) — so the instinct
was "it's still too slow". It was not; the client reached `GameContext -> Flight`
and the process was healthy. Checking the actual *value* in the diagnostic, not
the timing, is what separated the two.

Two adjacent cost traps surfaced in the same change and share the diagnosis
"a seam method's cost must match its call rate":

1. Routing `canopy_coverage` through the full band evaluation re-composed terrain
   **height** per call. Evaluated per vegetation candidate (grass alone runs
   thousands per tile) this stalled terrain streaming enough that the capture
   client declared the host unhealthy.
2. Even with height supplied by the caller, the moisture + orogeny terms cost a
   domain warp plus several fBm octaves — far too much per candidate, and
   pointless, since they vary over ~100 km.

## Fix

- `FlattenedSurface` forwards `canopy_climate`, under a comment block that names
  the trap and says **add every new `SurfaceQuery` method here**.
- `canopy::tests::flatten_decorator_passes_canopy_through` pins it, using an
  *empty* flatten handle — the transparent case, which is exactly the one that
  must not zero the field. The test asserts it saw non-zero canopy somewhere
  before comparing, so an all-zero sweep cannot pass it trivially.
- The cost split is structural, not a tuning: `SurfaceQuery::canopy_climate`
  returns the slowly-varying climate half (hoisted **once per tile**), and
  `CanopyClimate::coverage(dir, height_m)` does the cheap per-candidate half (the
  altitude chain off the candidate's own height, plus the stand noise). The
  altitude chain deliberately stays per-candidate: hoisting it would staircase
  the treeline and the shoreline to the tile grid.
- `canopy::tests::point_query_matches_per_vertex_band` asserts the point query and
  the per-vertex band return bit-equal values, so the cheap route cannot drift
  from the one the albedo bake used.

## Recurrence signal

**A landcover-derived field that reads zero or constant across an entire body is a
forwarding failure, not a tuning problem.** Check `FlattenedSurface` (and any
other `SurfaceQuery` decorator) before touching thresholds. The `forest_stand_site`
diagnostic is the cheapest probe: `stand_coverage: 0.0` with `lat_deg: 0.0` means
the site search found nothing anywhere and fell back to the sub-stellar point.

For the cost half: if terrain streaming slows sharply after a seam addition, check
whether the new method is being called per *candidate* when its inputs vary per
*tile*. The rule now lives with the contract — see `CanopyClimate` in
`crates/domain/terrain/src/canopy.rs` and `docs/world/vegetation.md` §4.4.
