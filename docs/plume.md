# Rocket engine plumes

Liquid-engine exhaust is rendered as a **data-driven, pressure-responsive
billboard effect** evaluated from engine state and local atmosphere. This is the
consumer-side visual layer over the shipyard's `Engine` parts; it owns *how a
firing engine looks*, not *what the engine is* (that stays in `thalos_shipyard`).

The full design direction (propellant families, layered effects, the volumetric
path for solids) is captured in the uploaded concept note; this doc tracks what
is **built** and the seams the later phases extend.

## Status

**Phase 1 (built).** One mesh-based emissive plume per liquid rocket-bell engine,
driven from a typed signal boundary, with propellant-family presets and a
pressure-ratio → shape response. Replaces the old placeholder engine-mesh tint
(which, post-F6, was silently dead — it mutated `StandardMaterial` while the hull
had migrated to `ShadowedStandardMaterial`).

Not yet built (later phases, in design-note order): secondary GPU particles
(sparks/ice/wisps), heat-haze distortion, clustered engine lights, ground
interaction (dust/steam), jet-nacelle afterburners, and the solid-motor
`CloudEmissionProfile` path into the volumetric cloud system. Per-engine `p_exit`
is still approximated by a design constant (see *Pressure response*).

Verify: `just screenshot plume` (agent-servable). Live feel (throttle sweeps,
startup/shutdown transients, in-flight pressure sweep on a real ascent) is a user
play session.

## Architecture

The pipeline mirrors the design note's "separate physics inputs from visual
mappings" principle. Everything lives in `crates/runtime/game/src/rendering/plume.rs`
plus the shader `assets/shaders/plume.wgsl`.

```
Engine sim (EngineThrust + ThrottleState)   local atmosphere (ambient pressure)
                    \                              /
             update_plume_signals  ── PlumeSignals (typed boundary, on the engine)
                                          │            ▲
                    PlumeDebugOverride ───┘            │ propellant preset + curves
                                                       │
                            update_plume_visuals ──► PlumeParams (flat uniform)
                                                       │
                                              plume.wgsl (billboard + volumetric)
```

- **`PlumeSignals`** (component on the engine entity) is the single typed
  boundary. Visual code reads this, never the gameplay components directly — so
  future particle/light layers share one signal and the effect can be previewed
  without a live vehicle. Fields: `throttle`, smoothed `ignition` (retains
  startup/shutdown transients), `ambient_pressure_pa`, `pressure_ratio`.
- **`update_plume_signals`** publishes those from each engine's live
  `EngineThrust` (the `crate::engine` plumbing — fuel-out / gating already folded
  in) plus the craft's ambient pressure, resolved once per frame from the craft's
  altitude over its dominant body's `terrestrial_atmosphere`.
- **`PlumeDebugOverride`** (resource) drives the signals directly with frozen
  values — the design note's authoring workflow, and how the headless `plume`
  screenshot preset lights an engine at a chosen back-pressure without fighting
  the fuel/warp gating.
- **`update_plume_visuals`** is the "curve evaluator": it maps signals + the
  engine's propellant preset into `PlumeParams` (the flat shader uniform) over a
  `log2(pressure_ratio)` domain, and toggles plume visibility.
- **`PlumeMaterial`** is one additive HDR material per engine, so each carries its
  own resolved params. `AlphaMode::Add` (Bevy renders this as
  premultiplied-alpha blending; the shader emits premultiplied colour with alpha
  0 for a pure-additive glow the post-stack bloom haloes).

## The billboard + volumetric fragment

`plume.wgsl` renders a unit quad (built once, shared) as a **cylindrical
billboard**: the vertex stage locks it to the engine's exhaust axis (part-local
`-Y`, opposite thrust) but rotates it about that axis to face the camera, so a
flat strip reads as a round plume from any side view.

The fragment stage integrates a radially-symmetric density through that round
cross-section (analytic chord through a cylinder of radius `R(t)`), so the plume
is bright and thick on-axis and feathers to nothing at the silhouette — no hard
mesh edge. On that envelope it layers the emission temperature field: a hot
near-nozzle core, shock-diamond (Mach-disk) compression nodes that fade
downstream, a cooler mixing-layer sheath, and animated turbulent breakup. Colour
comes from a three-stop propellant palette (edge → mid → core) indexed by
temperature.

The mesh uses **normalized axial coordinates** (`v` = axial 0→1, `x` = lateral
−1→1); the vertex shader scales and orients it from `PlumeParams`, so shape and
pressure response change with no runtime mesh regeneration (design note decision
#1: procedural radial profile in the shader, not authored blend targets).

## Propellant families

`PropellantFamily` is derived from the engine's `reactants` (methalox / kerolox /
hydrolox / generic fallback) and supplies the starting colour palette, HDR core
intensity, and base opacity — the defaults an engine profile then tunes. Only the
palette differs today; all families share the one shape/response model.

- **Methalox** — pale blue-white core, blue plume, blue-violet sheath.
- **Kerolox** — warm white core, orange plume, sooty amber sheath, denser.
- **Hydrolox** — faint blue-white, near-invisible; legibility from the core glow.
- **Generic** — restrained translucent jet (hypergolic / mono / unknown).

## Pressure response

`r = p_exit / p_ambient` drives the shape over a compressed `log2` domain. A
single `vac = smoothstep(log2 r)` lever expands the plume from a compact,
shock-celled sea-level jet toward a broad, feathered, cell-free vacuum plume:
radial `expansion`, axial `length`, `core_decay`, and `edge_softness` all
interpolate on it. Shock-diamond contrast additionally gates on real ambient
pressure (no diamonds in vacuum) and fades as the plume goes underexpanded.

Raw altitude is deliberately **not** an input — the same altitude has different
pressure on different worlds, and the system must work across the Pyros system.

`p_exit` is currently a first-slice design constant (~45 kPa: overexpanded and
shock-celled near sea level, strongly underexpanded toward vacuum). Threading a
real per-engine exit pressure / nozzle design from the propulsion + catalog layer
(`EngineOptimization`) is the natural next refinement.

## Screenshot preset

`just screenshot plume` boots a plain orbit (space/planet backdrop), forces the
engine to full throttle via `PlumeDebugOverride`, and frames the craft
three-quarter on the engine (`craft_context` in `screenshot.rs`: focus centered a
few metres down the stack, up = the ship's nose axis). `THALOS_PLUME_PRESSURE`
(Pa) scrubs the ambient pressure so the sea-level ↔ vacuum look is reproducible
regardless of the craft's real orbit altitude; the usual
`THALOS_SCREENSHOT_{AZIMUTH,ELEVATION,DISTANCE}` reframe it.
