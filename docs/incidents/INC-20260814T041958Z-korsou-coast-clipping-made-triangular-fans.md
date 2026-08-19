# INC-20260814T041958Z: Kòrsou coast clipping made triangular fans

> Follow-up: the dedicated face described here was emitted as a deep skirt on
> every coast and regressed beaches. Its replacement is documented in
> `INC-20260816T061109Z-korsou-coast-skirts-made-beach-walls.md`.

## Symptom

At the Caracasbaai waterline, long straight coast edges and large triangular
faces were visible from roughly 6 to 22 m above the water. The same geometry
appeared around the island. The earlier 7.5 m coastal grid reduced the size of
the facets but did not remove their characteristic fan shape.

## Mechanism

Every mixed land/water triangle was clipped at the signed-distance zero
crossing, then every crossing vertex was forced to sea level. A high DEM or
synthetic cliff vertex therefore reached the water through a sloped triangle.
The representation made a cliff out of a fan in the land sheet rather than a
coast face. Procedural rock displacement also remained fully active at the
authored lip, so adjacent crossings inherited metre-scale height differences
and formed a sawtooth silhouette.

Increasing the grid density could make those fans smaller, but could not remove
the mechanism. Applying one vertical wall to every coast also failed visually:
gentle beaches became retaining walls.

## Fix and recurrence tell

Clipped zero-contour edges are projected back onto the same authored field and
subdivided to at most 3 m. A local profile connects the waterline to the first
30 m DEM cell. Gentle shores meet sea level without a separate face; an
already elevated near-shore cell may produce a dedicated cliff face. Synthetic
displacement is zero at the coast and fades in from 8 to 36 m inland, so it can
add rock character without moving the waterline. The failed unconditional
skirt and wide inland reconstruction are detailed in the follow-up incident.

The recurrence checks are:

- `coast_profile_uses_only_the_nearest_dem_cell`;
- `caracasbaai_beach_meets_water_without_a_retaining_wall`;
- `inland_hills_do_not_create_a_vertical_coast_face`;
- `real_coastline_boundary_is_watertight_and_subdivided`;
- `synthetic_detail_is_deterministic_and_coast_safe`.
