# INC-20260817T062355Z: Kòrsou reconstructed the shore instead of using it

## Symptom

`fucked-coast`, `beach-close`, and `rough-coast` still showed faceted
waterlines, fake beach walls, and smooth invented curves after the local
30 m DEM profile fix. Analytic mesh and profile tweaks did not match the
real Curaçao shore.

## Mechanism

The baker already had OSM `natural=coastline`. It rasterized those vectors
to a 15 m signed-distance field. The mesher then:

1. reconstructed the visible waterline from that field (box filter plus
   gradient projection), not from the source polylines;
2. invented beach and cliff profiles from the nearest 30 m GLO-30 cell.

Around Caracasbaai, OSM already contained 180 m chords. SDF reconstruction
turned those chords into faceted zero-contours, and the DEM profile draped
a fake cliff or retaining wall onto beaches. The first-looking hypothesis —
"the coastline data is missing" — was wrong. The mapped shore was present
and then discarded.

## Fix and recurrence tell

The baker now keeps the densified OSM+Sentinel-2 rings as `KSH1` polylines
and still rasterizes an SDF for land/sea and ocean foam. Runtime clips mixed
triangles with the SDF, then snaps and walks the polylines for the visible
waterline. Waterline vertices sit at sea level. Inland vertices keep GLO-30.
There is no reconstructed beach/cliff profile.

Recurrence tells: a waterline vertex whose `distance_to_coast_line_m` is
larger than a few centimetres, a coast edge longer than 3 m, or a new
`CoastProfile`-style height function that samples GLO-30 to invent a lip.
The checks are `real_coastline_boundary_is_watertight_and_subdivided`,
`caracasbaai_beach_meets_water_without_a_retaining_wall`, and
`inland_hills_do_not_create_a_vertical_coast_face`.
