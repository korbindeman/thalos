# ADR-0002: Planet water is an analytic ray-traced sphere — never a mesh, at any scale

- **Status:** Accepted
- **Date:** 2026-07-01 (recorded 2026-07-18 from the session notes)

## Context

Ocean bodies (Thalos) need visible water in the ship view and the map view. The obvious approach
— a sea-level icosphere shell mesh (`BodyWaterMaterial` / `body_water.wgsl` existed for it) — was
tried twice and failed twice, for scale reasons that will not go away:

- **Facet sag:** a polyhedron's flat triangles are chords that dip `R·θ²/8` below the true
  sphere. At planet radius that is *tens of metres* at each face centre even at subdiv 7, so the
  seabed and coast punched through the water everywhere except the vertices. Tessellation can't
  fix it (~20M+ tris and still a faceted limb from orbit).
- **Map view:** the same mesh at map scale z-fought the seabed into dot moiré; a lift+subdiv hack
  only made the dots hexagonal.

## Decision

Water is always an **analytic ray-traced sphere in a shader**, never geometry:

- **Ship view:** the ocean is intersected inside the `BodySky` fullscreen pass (`thalos::water`,
  `shade_ocean`), drawn wherever the ocean hit is nearer than scene depth — the same
  fold-into-body-sky pattern clouds use (separate transparent quads sort unreliably under
  big_space). Water-column thickness (`scene_t − t_ocean`) gives shallow/deep colour for free.
- **f32 stability at planet radius** requires the cancellation-free form: pass the camera's exact
  f64-computed height above sea level in, form `c_sea = h·(2r+h)`, and take the near root via
  Vieta (`t_near = c_sea / t_far`). The naive `−b−√disc` wobbles by metres per frame.
- **Map view:** a fullscreen billboard (`MapOceanMaterial`) ray-traces the sphere and writes real
  `@builtin(frag_depth)`, so the hardware depth buffer sorts it against map terrain exactly — no
  z-fight, no facets, no extra pass.

## Alternatives

- **Meshed icosphere shell** — rejected twice (ship + map) for the facet-sag and z-fight failures
  above. The mesh path is dormant dead code slated for deletion.
- **Tessellating the mesh harder** — rejected: cost explodes and the limb stays faceted.
- **A separate transparent water quad/pass** — rejected: transparent-quad sorting is unreliable
  under big_space; layered sky effects belong inside `BodySky`.

## Consequences

- Perfectly smooth waterline at every altitude; shallows/depth grading comes free from the depth
  buffer.
- Water shading lives in shared WGSL (`thalos::water`) consumed by both views — F9 later folds it
  into `shade_surface` proper.
- Constraint on future work: any new water feature (waves, foam, wakes) must work in the analytic
  formulation or as a local detail layer — do not reintroduce a planet-scale water mesh. Only
  bodies with a `BodySky` (terrestrial atmosphere) currently get ship-view water; revisit if an
  airless ocean body is ever authored.
