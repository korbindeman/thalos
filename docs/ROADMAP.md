# Roadmap

Thalos is a spaceflight simulation game targeting a physically grounded
take on the genre — realistic scaling, physically plausible simulation
that remains playable, and a solar system whose nature reveals itself
through exploration. The work is organized as two main phases of
engineering deliverables, plus a deferred tail.

This doc is the entry point. Each milestone references a system spec
that owns the depth.

## Where we are

Working today:

- Patched-conic orbital simulation with a single propagator
  (`KeplerianPropagator`) used for both live stepping and prediction.
- Maneuver nodes with synchronous in-frame trajectory rebuild.
- Time warp with ghost-body preview.
- Canonical `CraftState` + `AuthorityMode` shell; `Simulation` owns one
  canonical craft and one authority book. `OnRails`, `WarpIntegrated`,
  the first `LocalRigidBody` bubble, and `BodyFixed` landed-state
  evaluation are wired.
- Map view decoupled: `MapSnapshot` + `MapProjection` are the only
  inputs to map rendering, ghost bodies, and maneuver UI.
- Real-space scene under a single `BigSpace` root; ship camera carries
  `FloatingOrigin`, bodies are placed via `Grid::translation_to_grid`.
- Feature-compiler v1: `AirlessImpactMoon` (Mira) and
  `ColdDesertFormerlyWet` (Vaelen).
- Flat impostor planet renderer (`planet_impostor.wgsl`) with baked
  cubemaps and analytic crater SSBO.
- Gas-giant impostors and ring rendering.
- Celestial sky v1 (stars + galaxies via `thalos_celestial`).
- Planet editor with live rebake, sketch tool, and MVP feature spec.
- CPU-painted reflection probe (interim).

Hard limits today:

- Thalos and Pelagos render via the flat-water `Ocean` placeholder.
- No rocky-atmosphere scattering.
- M5 first slice is aggregate-only and Thalos-only: no aero, part-level
  joints, docking, debris, rover wheels, or persistence of live Avian
  state beyond canonical readback/collapse.
- `WorldPreset::Realistic` (N-body ephemeris backend) is unwired and
  rejected by `validate_supported` — deferred to M7.

## Phase 1 — Architectural & Rendering

The shape of Phase 1: a system you can fly through at any scale, on
every body, with the renderer telling a coherent visual story from
orbit down to surface scale. The player isn't yet landing; the world
is being built.

### M1 — Canonical state + big_space + map decoupling ✅

Every craft has one canonical `f64` state and one authority mode. The
map view is a presentation client of trajectory snapshots rather than
the active scene. The real-space scene runs under a `BigSpace`
hierarchy with the camera as floating origin.

- **Status:** Done. Phases 1-3 of the simulation spec are implemented
  (`crates/physics/src/canonical.rs`, `crates/game/src/map_view.rs`,
  `crates/game/src/rendering/real_space.rs`).
- **Spec:** [simulation.md](simulation.md), Implementation Phases 1-3.
- **Gates:** M3 needs the big_space hierarchy. M5 needs the authority
  model.

### M2 — Terrain pipeline revamp

General overhaul of the feature compiler. Lands the v2 backlog from
the terrestrial-pipeline research (first-class hydrology features,
layered material columns under each biome, climate-field inputs to
`BiomeMaskPlan`, provenance API + seed-promotion state machine), and
adds the missing archetypes (`AgingOceanicHomeworld`,
`GenericTerrestrial`) so Thalos and Pelagos stop falling back to the
flat-water placeholder. Goal of M2: Mira, Vaelen, Thalos, and Pelagos
all render through the revamped pipeline at the right visual quality
from orbit.

- **Spec:** [terrain.md](terrain.md), generation half + "v2 backlog".
- **Deps:** feature compiler v1 (done; Mira and Vaelen wired).
- **Gates:** M3 needs the revamped pipeline producing tile-friendly
  output; M4 needs ocean topology to render against.

### M3 — Ground LOD rendering

Surface-scale rendering of the four main bodies. The `thalos_udlod`
fork is already done (Bevy 0.18 port + `TileProvider` trait added,
lives at `~/dev/bevy_terrain`); M3 is wiring it into the Thalos
workspace, implementing `PipelineTileProvider` against the revamped
feature compiler, and onboarding Mira, Vaelen, Thalos, Pelagos.
Seamless camera traversal from orbit to ~cm features.

- **Spec:** [terrain.md](terrain.md), "Ground LOD rendering" section.
- **Deps:** M1 (big_space hierarchy), M2 (synthesis pipeline producing
  tile-friendly output).

### M4 — Rocky atmospheres + ocean rendering

Bruneton precomputed scattering for rocky bodies, parameterized per
body. Soft terminators, limb glow, twilight wedge. Microfacet ocean
with sun-glint streak. Cloud shells with shadows on terrain.

- **Spec:** [atmosphere.md](atmosphere.md).
- **Deps:** M3 (atmosphere is read against the surface; ocean is
  rendered against terrain).

**End of Phase 1:** Mira, Vaelen, Thalos, and Pelagos render as real
worlds from orbit down to surface scale, with no LOD popping or
precision artifacts. No landings yet. Other Pyros bodies follow the
same pipeline; onboarding them is incremental work, not a separate
milestone.

## Phase 2 — Gameplay (sketch)

Phase 2 starts only after Phase 1 ends; the breakdown here is
indicative and will be refined.

### M5 — Avian local bubble + landing handoffs

Avian f64 rigidbodies in a local bubble centered on the active craft.
Hydration, sampling, collapse. Body-fixed sleep for landed craft.
Authority transitions between rails / warp-integrated /
local-rigidbody / body-fixed / docked.

- **First slice status:** Implemented for Thalos landing validation.
  The game hydrates `ships/apollo.ron` as one aggregate Avian
  rigidbody using blueprint-derived primitive compound colliders and
  existing aggregate mass/inertia. It builds one terrain collider patch
  from the same rendered R16 cubemap height path that
  `PipelineTileProvider` uses for UDLOD, reads Avian state back into
  canonical state, and collapses stable landed craft to `BodyFixed`.
- **Current thresholds:** enter in ship view, dominant body Thalos,
  any explicit target body Thalos, surface available, warp 0x/1x, AGL
  below 20 km. Patch half extent 4096 m, resolution 129 x 129, rebuild
  after 1024 m lateral drift.
  Collapse after 2.0 s terrain contact, linear speed <0.5 m/s,
  angular speed <0.05 rad/s, throttle zero.
- **Out of scope for first slice:** aero/drag/lift/heating, part-level
  joints, docking, debris, rover wheels, and save/load persistence of
  live local rigidbody state.
- **Spec preview:** [simulation.md](simulation.md), Implementation
  Phases 4, 5, 7. On-foot / EVA surface gameplay is specced in
  [surface_gameplay.md](surface_gameplay.md), which extends M5's
  authority + height-source machinery to the player-character case.

### M6 — Aero regime, warp force integration, advanced ships

Atmospheric flight regime (full aero, lift, heating, control surfaces)
with warp clamping. Warp force integration for finite burns + drag +
attitude. Plus ship complexity: staging, radial attachment,
surface-to-orbit ascent vehicles, possibly atmospheric planes.

- **Spec preview:** [simulation.md](simulation.md), Implementation
  Phase 6 + atmospheric regime section. Ship-parts spec spun out when
  the design is concrete.

## Beyond Phase 2 (deferred)

Not currently scoped. Listed so dependencies stay legible.

### M7 — Realistic policy / N-body ephemeris baker

Offline N-body baker producing Chebyshev segments, runtime
`NBodyEphemeris` provider, `RealisticPolicyV1`.

### M8 — Surface gameplay scaffolding

Resource economy, colonies, biosecurity protocols, narrative
milestones from `lore/civilization.md`.

## Dependency graph

```
M1 (sim foundation: state + big_space + map)
 ├── M3 (ground LOD: needs big_space)
 └── M5 (avian bubble: needs authority model)

M2 (terrestrial pipeline)
 └── M3 (ground LOD: needs real terrain to show)
      └── M4 (atmospheres + ocean: read against the surface)
           └── M5 (landing visuals depend on M4)

M5 ── M6 (aero) ── M7/M8
```

M1 and M2 can run in parallel; both feed M3.

## System specs

| System | Spec | Phase 1 milestones | Phase 2 |
|---|---|---|---|
| Simulation, physics, authority, time | [simulation.md](simulation.md) | M1 | M5, M6 |
| Terrain generation + ground LOD | [terrain.md](terrain.md) | M2, M3 | — |
| Atmospheres, oceans, IBL | [atmosphere.md](atmosphere.md) | M4 | — |
| Celestial sky | [celestial.md](celestial.md) | (done) | — |
| Surface gameplay (on-foot, BodyFixed, height source) | [surface_gameplay.md](surface_gameplay.md) | M5 | M6, M8 |

Lore: [lore/solar_system.md](lore/solar_system.md),
[lore/civilization.md](lore/civilization.md).

Research and process notes (kept as standalone references):
[gen/terrestrial_pipeline_research.md](gen/terrestrial_pipeline_research.md),
[gen/planet_aesthetics.md](gen/planet_aesthetics.md),
[gen/dunes.md](gen/dunes.md),
[gen/vaelen_processes.md](gen/vaelen_processes.md).
