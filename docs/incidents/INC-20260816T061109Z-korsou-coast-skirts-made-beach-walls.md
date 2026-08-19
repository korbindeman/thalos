# INC-20260816T061109Z: Kòrsou coast skirts made beach walls

> Follow-up: local 30 m DEM profiles still invented the visible waterline.
> The mapped-polyline replacement is documented in
> `INC-20260817T062355Z-korsou-sdf-profiles-ignored-mapped-shore.md`.

## Symptom

After the first coastline-facet fix, `beach-close` showed a flat beach ending
in a dark faceted retaining wall. Its waterline still changed direction in
visible straight chords. Rough coasts were denser than before but not more
natural.

## Mechanism

The fix in `INC-20260814T041958Z` classified only the height of each clipped
coast vertex, then emitted the same ten-metre-deep vertical skirt for every
coast segment. A beach vertex could correctly sit at sea level and still own a
deep wall below it; displaced translucent water exposed that wall. The land
sheet also kept the uncorrected GLO-30 height until its final clipped triangle,
so there was no cross-shore beach or bluff profile to make smooth.

The attempted replacement then sampled terrain as far as 120 m inland and
rebuilt a 90–150 m coastal band. That promoted ordinary inland hills into coast
cliffs, flattened real beaches into slabs, and added a visible underwater
apron. Its green tests asserted the shape of that implementation rather than
the visual requirement.

## Fix and recurrence tell

The terrain mesher now derives each coast profile strictly from the first 30 m
GLO-30 cell inland, with a median of three alongshore samples. Low local relief
puts the lip at sea level and eases into the unchanged source surface within
that cell. Only a near-shore cell that is already high and steep raises the lip
into a bluff or cliff. Terrain beyond 30 m is never reconstructed.

The unconditional skirt and the generic underwater apron are gone. A separate
face is emitted only for a terrain-classified raised lip, and its depth scales
continuously to zero at beach transitions. The shoreline field receives a
small spatial filter bounded to one authored 15 m sample; boundary edges remain
three metres or shorter.

Doubling the filter footprint from 7.5 m to the full 15 m source spacing was
tested and rejected. It slightly rounded cliff-coast corners but merged nearby
Caracasbaai beach contours into a new pointed cusp. The half-cell footprint is
therefore a recurrence constraint, not an arbitrary quality setting.

The recurrence checks are:

- `coast_profile_uses_only_the_nearest_dem_cell`;
- `coastal_reconstruction_stops_after_one_dem_cell`;
- `coast_filter_stays_within_half_an_authored_sample`;
- `boka_tabla_nearshore_terrain_produces_a_cliff_profile`;
- `caracasbaai_beach_mesh_has_no_vertical_shore_wall`;
- `inland_hills_do_not_create_a_vertical_coast_face`;
- `real_coastline_boundary_is_watertight_and_subdivided`.
