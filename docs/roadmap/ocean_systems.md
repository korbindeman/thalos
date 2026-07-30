# Ocean systems

**Status: future world/gameplay program (2026-07-29).** The rendering foundation
exists, but this program stays behind the neural-terrain × standard-path renderer
keystone until explicitly pulled forward. `docs/backlog.md` is the only status
authority.

This roadmap turns the ocean from an attractive surface into a traversable world.
It joins the current rendering work in
[`rendering/ocean.md`](../rendering/ocean.md), the authored coastline and
bathymetry in [`world/terrain_macro.md`](../world/terrain_macro.md), the weather
authority in [`rendering/clouds.md`](../rendering/clouds.md), and the player
experience in [`gameplay.md`](../gameplay.md).

## 1. Vision

The ocean is a second terrain system. A player reads its shape, chooses a route
through it, shelters from it, and sometimes has to respect it more than the
vehicle they brought.

The target has two deliberately contrasting faces:

- **Heavenly coasts:** clear shallows, coherent breaker sets, wet sand, sheltered
  water, reefs, dunes and vegetation arranged by coastal process rather than a
  universal sand strip.
- **Awe-inspiring open water:** long swell trains, changing sea states, distant
  weather, foam and spray, and storms large enough to turn an ordinary crossing
  into an event.

Pirate games are the experiential reference, not the visual style to copy. Their
useful lesson is that the sea must act on the vehicle and the player's decisions:
waves hide the horizon, change steering and speed, make weather legible, and
create stories between authored destinations. Physical plausibility supports
that readability; it does not require every wave to be understated.

Thalos is the proving ground. Pelagos is the culmination. The homeworld provides
near-term coastlines, infrastructure, vehicles, weather and iteration without
making the late narrative destination carry an untested technology stack.

## 2. Product contract

An ocean feature earns its place by improving at least one of these experiences:

1. **Traversal:** the player can read wave direction, swell, shelter, current and
   hazards and use that information to choose a route or operating window.
2. **Place:** open ocean, sheltered bay, reef, beach, cliff coast and storm sea
   feel materially different rather than sharing one water preset.
3. **Weather:** a storm approaches, develops, passes and leaves a changing sea
   behind. It is a spatial event with a lifecycle, not a screen-space effect.
4. **Vehicle response:** the same surface that is rendered drives buoyancy,
   attitude, impacts, wakes and control difficulty.
5. **Discovery:** on Pelagos, the transition from dangerous surface weather
   through blue water into a living reef is a mechanical and visual climax.

Storms should create decisions rather than unavoidable attrition: delay a launch,
route around a cell, seek a lee shore, ride a following sea, conduct risky
science, or rescue something already caught offshore. The program does not make
constant danger the definition of ocean gameplay; calm water is valuable because
the contrast is real.

## 3. Authorities and scale model

ADR-20260729T060720Z fixes the ownership:

- The authored per-body climate and ocean configuration describe what a body can
  produce.
- One evolving per-body weather authority owns storm identity, wind,
  precipitation potential and other surface forcing. The existing
  `CloudWeatherField` evolves behind that seam rather than the ocean inventing a
  second storm map.
- One dynamic sea-state field is the ocean's response to that forcing and to
  bathymetry. Rendering, buoyancy and gameplay query the same wave state.
- The analytic planet-scale water sphere remains the exact global surface.
  Camera-relative displacement is a bounded local detail layer over it.
- The one signed sea-height field remains the sole coastline and bathymetry
  authority. Weather and waves may flood, surge, break and refract locally; they
  do not author a second permanent shoreline.

The system spans three scales:

| Scale | Owns | Examples |
|---|---|---|
| Planetary / basin | durable geography and slowly evolving forcing | coastline, bathymetry, prevailing climate, storm tracks, basin currents |
| Regional | coherent ocean state over tens to hundreds of kilometres | fetch, wind sea, travelled swell, storm surge, sheltered lee, current fronts |
| Local | geometry and interaction around relevant views and actors | displaced waves, breakers, swash, buoyancy, wakes, spray, local shallow-water tiles |

Unresolved energy crosses scale boundaries. Local geometry may disappear with
distance, but its slope variance remains in the BRDF; a shallow-water tile
receives spectral boundary conditions rather than starting a private sea; a
storm's swell can outlive and outrun the rain cell that generated it.

## 4. Coast character

A heavenly beach is a coupled land-water-biome result:

- a process-typed depositional coast with a shallow foreshore and offshore bar;
- water colour derived from the real seabed, column depth and atmosphere;
- coherent breaker sets that shoal, refract and spill over that bathymetry;
- swash, persistent foam residue and a wet-sand drying band;
- dunes, vegetation and reef or rock structure placed from exposure, substrate,
  climate and storm reach;
- sheltered water that is visibly calmer than the exposed side of the same
  island.

Not every coast becomes a beach. The landform-province and coastal-morphology
programs (`NTR-X2f`/`NTR-X2h`) should produce depositional strands, volcanic
islands, cliffs, rias, barrier systems and reefs. Ocean simulation consumes that
character; it does not flatten it into one shoreline effect.

Pelagos reuses the same contracts with different authored content: volcanic
island arcs, cold blue-green water, dense living shelves, humid haze and severe
ocean weather. Its biosphere may alter roughness, colour, foam persistence or
underwater visibility, but never through a separate Pelagos-only renderer.

## 5. Thalos proving slice

Before Pelagos depends on the program, one bounded Thalos region proves the whole
chain:

- one showcase depositional beach with a sheltered and exposed side;
- one small test vessel using the canonical vessel/control path;
- calm, moderate and storm sea states through the same simulation;
- one moving offshore storm that is visible before arrival and leaves swell
  after it passes;
- a short beach-to-open-water route through which wave reading, shelter and
  operating-window choice matter.

This is not a naval-combat commitment. The vessel is an integration instrument
and an early gameplay surface: it validates buoyancy, controls, camera motion,
wakes, coast interaction and weather consequences. Weapon systems and pirate
economy are outside scope.

## 6. Program

The stable work IDs are owned here; their status lives only in
`docs/backlog.md`.

### OCEAN-2 — shared dynamic sea foundation

Replace the current representative two-packet tracer with measured
JONSWAP/TMA-derived fields behind the existing filtered-slope seam. Produce
height/displacement, slope and Jacobian cascades; define the deterministic
simulation-time contract; expose one bounded wave-query surface for physics; add
GPU timings, spectrum/energy diagnostics and sea-state captures before selecting
final grid sizes.

The gate is not merely a prettier still: calm/moderate/storm authored inputs must
produce distinct, energy-accounted fields, and render and query samples must
agree at known body-fixed points.

### OCEAN-3 — displaced sea and vessel response

Add a snapped camera-relative projected grid or clipmap over the analytic ocean,
transfer omitted energy into roughness, and make one test vessel respond through
the canonical physics/control path. Persistent foam begins from Jacobian
compression and vertical motion; a cheap far Kelvin wake plus bounded local
impulses proves vessel coupling.

The player gate is a calm-to-rough-water run in which the vessel feels supported
by the visible crests, remains controllable, and never reveals the local/global
handoff.

### OCEAN-4 — heavenly beach vertical

Choose one deterministic Thalos coast after the landform-province/coastal
morphology work. Add coherent breaker wavefronts, shallow-water coupling, swash,
foam age/advection, wet-sand drying, seabed optics, exposure-aware vegetation and
reef/rock placement. Capture both exposed and sheltered sides under matched
weather.

The gate is a continuous offshore-to-dry-sand approach: no authority seam,
floating foam texture, universal breaker ring, translucent shelf or vegetation
through the strand.

### OCEAN-5 — regional storm lifecycle

Broaden the existing cloud-weather authority into the shared weather forcing
seam. Add authored/procedural moving fronts or cyclone cells with birth,
intensification, translation and decay. Wind and pressure force the sea;
clouds/rain/lightning/visibility project the same identity; swell grows with
fetch and decays or travels after the cell moves on. Add deterministic storm
tracks, forecast/debug views and event telemetry.

The gate is one storm observed before, during and after passage: cloud structure,
surface wind, wave spectrum, rain/spray and gameplay hazard agree spatially and
temporally.

### OCEAN-6 — Thalos proving voyage

Integrate the test vessel, showcase beach and moving storm into the §5 route.
Add forecast/operating-window information, shelter, launch/recovery and one
science or rescue objective. Tune readability and consequence so the event is
exciting without turning ocean travel into constant interruption.

The agent gate is a deterministic headless sequence with machine-readable
weather/sea/vehicle state. The user gate is the complete playable route.

### OCEAN-7 — Pelagos ocean-world slice

Author Pelagos's climate, bathymetry, island arc and living shelf on the shared
contracts. Add ocean-world entry/landing constraints, surface support,
submersible deployment, underwater light/visibility, biological sampling and
the first ecological consequence hooks. Reuse the Thalos vehicle, weather and
coast mechanisms with Pelagos data.

The gate is the narrative sequence already promised in `docs/gameplay.md`:
arrival through severe weather, safe deployment, descent through blue haze and
first sight of a living reef.

## 7. Observability and verification

Every dynamic layer needs a reader, not merely a debug texture:

- sea-state energy by cascade, significant height, dominant period/direction and
  unresolved-variance budget;
- render/query agreement at body-fixed probes;
- projected-grid extent, handoff weight and GPU time;
- foam source, resident mass/age and decay balance;
- storm id/lifecycle, forcing, translation, fetch and swell lag;
- vessel heave/roll/pitch response, water contact and control saturation;
- shallow-tile count, boundary-energy error and coast interaction cost.

Headless evidence needs deterministic `ocean-calm`, `ocean-storm`,
`beach-exposed`, `beach-sheltered` and integrated-route framings. Moving
acceptance requires scripted sequences plus user play because wave timing,
camera motion and control feel cannot be judged from a still.

The first implementation phase must measure GPU and memory cost before locking
cascade counts or resolutions. Heavy probes remain `THALOS_*` opt-in; standing
health summaries join `just diag` only where a threshold can identify a real
defect.

## 8. Deliberate limits and open decisions

- No planet-wide shallow-water or CFD solver. High-cost interaction is bounded
  to relevant coasts, actors and authored zones.
- No second visual-only storm preset and no physics-only wave function.
- No tides until orbital forcing or gameplay demonstrates value; the permanent
  coastline datum remains 0 m.
- No naval combat commitment in the proving slice.
- Exact test-vessel form, storm severity distribution and Pelagos underwater
  vehicle remain content decisions made at their phase gates.
- Reflection techniques beyond the stable atmosphere/sun fallback are judged by
  measured value; SSR may improve local confidence but cannot become the only
  reflection source.

## References

- [Ocean rendering](../rendering/ocean.md)
- [Clouds and weather](../rendering/clouds.md)
- [Gameplay](../gameplay.md)
- [Pelagos lore](../lore/solar_system.md#pelagos)
- [ADR-20260720T185954Z: analytic planet water](../adr/20260720T185954Z-analytic-planet-water-never-meshed.md)
- [ADR-20260720T185958Z: one signed sea field](../adr/20260720T185958Z-water-projects-one-signed-sea-field.md)
- [ADR-20260720T212214Z: one cloud weather field](../adr/20260720T212214Z-one-weather-field-many-cloud-projections.md)
- [ADR-20260729T060720Z: one coupled ocean world system](../adr/20260729T060720Z-ocean-is-one-coupled-world-system.md)
