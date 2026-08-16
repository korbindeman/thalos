# INC-20260810T204743Z: Kòrsou cliffs needed character and density

## Symptom

At the `caracasbaai-mid` waterline view, the coast alternated between a rounded
sheet and visible diagonal facets. The foreground beach carried the same broad
surface bands even though its measured grade was nearly flat.

## Mechanism

Three independent limits compounded:

- the closest terrain mesh had the same 15 m interior spacing as the shoreline
  field, too coarse for rock-scale silhouette changes;
- synthetic displacement received elevation only, so it could not distinguish
  the 1–2% beach from the ~58% cliff;
- RTIN constrained only vertices immediately crossing the coast, allowing
  oblique diamonds to expose shoreline edges longer than one authored cell.

The broad tangent-space normal field then made the remaining directional bands
more visible on sand.

## Fix and recurrence tell

The finest mesh now samples at 7.5 m, the deterministic relief field uses DEM
grade plus shore distance to keep beaches smooth and activate multi-scale rock
fractures, and a 60 m RTIN refinement band keeps exposed coast edges below the
15 m shoreline spacing. Broad normal relief was reduced to fine surface grain.

The recurrence checks are:

- `caracasbaai_cliff_mesh_has_finer_geometry_than_the_shoreline_grid`;
- `caracasbaai_height_profile_separates_smooth_beach_from_broken_cliff`;
- `real_coastline_boundary_is_watertight_and_cell_sized`.
