# ADR-20260810T194324Z-woody-vegetation-is-impostor-first: woody vegetation is a shared impostor-first representation cascade

- **Status:** Accepted
- **Date:** 2026-08-10
- **Extends:** ADR-20260808T205119Z-korsou-second-application-render-kit

## Context

Kòrsou proved the wrong cost curve for dense foliage. Every accepted shrub and
tree was expanded into a copy of an authored alpha-masked mesh, batched by 128 m
cell, and submitted to Bevy's cascaded shadow passes. Thick coverage therefore
multiplied source-mesh vertices, leaf overdraw, CPU mesh construction, upload
volume, and shadow rendering together. On the target Mac, enabling foliage could
reduce the explorer to unusable frame rates. The high-detail meshes were also a
poor visual fit for the mostly aerial camera.

Thalos already had the better far representation: hemisphere-octahedral
impostors selected from projected size and aerial clearance, plus forest colour
in terrain for the unresolved tail. But the atlas types, bake shader, bake rig,
material, and four-vertex batch builder were split between the planetary renderer
and game runtime. Kòrsou could not use the mechanism without copying it.

The problem is representation, not insufficient thinning. Reducing density would
trade away the requested thick canopy while retaining the same per-plant cost
curve.

## Decision

Woody vegetation is **impostor-first** whenever an individual plant does not
justify authored geometry on screen.

`thalos_vegetation` owns the topology-independent mechanism:

- the species mesh payload used only as an atlas source;
- hemisphere-octahedral atlas layout, HDR targets, capture shader, and one-shot
  bake rig;
- the generic standard-path impostor material;
- the stable instance payload and batch builder: exactly four degenerate
  vertices and six indices per accepted root;
- bounds helpers and bake-readiness lifecycle helpers.

Adapters own only what genuinely differs:

- deterministic placement and the canopy/landcover field;
- planar, ellipsoidal, or cube-sphere coordinates and streaming topology;
- visible range, projected-size policy, and terrain-colour handoff;
- lighting extensions and bounded shadow policy.

Kòrsou uses impostors for all streamed woody plants. Its density, placement,
species mix, and 128 m cells are unchanged. Foliage is excluded from Bevy's
cascaded shadow caster set; the aerial environment and terrain proxy provide the
unresolved grounding instead. Thalos retains real mesh LODs only where projected
size and low AGL justify them, then uses the same shared impostor batches. Its
planetary material may add cloud and custom sun-shadow bindings, but may not own
a second atlas or geometry implementation. Coarse planetary impostor shadows
remain bounded to the near custom-shadow rings.

The permanent cost gate is structural: tests assert four vertices and six
indices per root in both the shared mechanism and the Kòrsou adapter. Kòrsou's
F3 extension reports root and impostor-vertex counts plus atlas readiness, and
headless capture waits for the atlas before judging scene readiness.

GPU compute compaction and indirect drawing are not part of this decision. They
remain an escalation only if measurements show draw submission or CPU tile
generation is the next ceiling after removing authored-mesh multiplication and
foliage shadow passes.

## Alternatives

- **Keep full meshes and tune LOD/density.** Rejected: it preserves the
  multiplicative alpha/shadow cost and makes coverage the performance knob.
- **Use one fixed crossed billboard.** Rejected: cheap, but visibly flat from
  aerial and oblique angles. Hemisphere-octahedral views preserve canopy volume
  from every supported above-ground perspective.
- **Give every plant an entity or a custom instance draw immediately.** Rejected:
  per-cell meshes already bound entity/draw count, while the measured structural
  waste was vertices, overdraw, upload, and shadows. A custom render pipeline is
  justified only by a remaining measured ceiling.
- **Keep the planetary implementation and copy it into Kòrsou.** Rejected: atlas
  conventions and material behaviour would drift, and a fix in one application
  would not repair the other.
- **Use meshlets or virtualized geometry.** Rejected for foliage: disconnected
  alpha-tested cards are a poor simplification substrate, and the aerial view
  does not benefit from preserving authored triangles it cannot resolve.

## Consequences

- Thick coverage no longer scales with authored tree complexity: a root costs
  four streamed vertices in both applications.
- Kòrsou pays a one-shot startup atlas capture, then streams only compact cards;
  its capture and diagnostics surfaces expose that readiness explicitly.
- Thalos and Kòrsou share one atlas/bake/batch contract while keeping honest
  spatial and lighting adapters.
- The terrain canopy colour remains the planet-scale representation. Instances
  are a resolvable-detail layer, not the authority for whether distant terrain
  is vegetated.
- Full visual and frame-time acceptance still requires a GPU run on the target
  Mac; compile and structural tests cannot validate shader pipelines or the
  resulting canopy read.
