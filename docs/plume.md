# Rocket engine plumes

Liquid-engine exhaust is rendered as a **data-driven, pressure-responsive
billboard effect** evaluated from engine state and local atmosphere. This is the
consumer-side visual layer over the shipyard's `Engine` parts; it owns *how a
firing engine looks*, not *what the engine is* (that stays in `thalos_shipyard`).

The full design direction (propellant families, layered effects, the volumetric
path for solids) is captured in the uploaded concept note; this doc tracks what
is **built** and the seams the later phases extend.

## Status

**Phase 2 (built, 2026-07-22).** The emission model is now *physical*: shape and
brightness come from one thermodynamic chain rather than two independent authored
fade curves, so the pad → orbit envelope is covered with no regime switch. See
*The emission model* below. Screenshot-verified at vacuum / 18 kPa / sea level;
live throttle and ignition transients are still unverified (user play session).

**Phase 1 (built).** One mesh-based emissive plume per liquid rocket-bell engine,
driven from a typed signal boundary, with propellant-family presets and a
pressure-ratio → shape response. Replaces the old placeholder engine-mesh tint
(which, post-F6, was silently dead — it mutated `StandardMaterial` while the hull
had migrated to `ShadowedStandardMaterial`).

Not yet built (later phases, in design-note order): secondary GPU particles
(sparks/ice/wisps), heat-haze distortion, engine lights, ground interaction
(dust/steam), jet-nacelle afterburners, and the solid-motor
`CloudEmissionProfile` path into the volumetric cloud system.

> **Engine lights are not a `PointLight`.** Bevy's clustered forward lighting is
> degenerate at the flight camera's 0.5 m → 1e11 m depth range, so a `PointLight`
> on a craft contributes *exactly zero* at any intensity — measured, see
> INC-0021. The plume light has to be an analytic term in the lighting spine
> alongside sun/moonlight (BL-40). Do not re-attempt the `PointLight` route.

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

## The emission model

The plume is an axisymmetric emitting gas column. Its brightness is **derived
from its shape**, not authored alongside it:

```
R(s)  = R0·lip + tan(theta)·s   free expansion off the nozzle lip, × barrel(s)
rho   ∝ (R0/R)²                 mass conservation along the column
T     ∝ (R0/R)^(2(gamma-1))     adiabatic (expansion) cooling
T    ×= exp(-e·s/R0)            entrainment cooling (atmosphere only)
S     = exp(-W·(1/T - 1))       visible-band emission (Wien side)
tau   ∝ rho · chord             optical depth across the line of sight
L     = S · (1 - exp(-tau))     emission through an absorbing column
```

Three properties of this chain are load-bearing, and each replaced an authored
hack that had a visible failure mode:

- **Two cooling mechanisms, one law.** Expansion cooling dominates in vacuum;
  entrainment cooling dominates at sea level, where the column barely widens at
  all. With only the former, a sea-level plume stays uniformly incandescent for
  its whole length and reads as a featureless white sausage.
- **Wien-side emission, not grey-body `T⁴`.** A plume radiates in the visible
  from the Wien side of the Planck curve, where output collapses exponentially
  in `1/T`. A polynomial falloff leaves the plume still bright where the
  geometry ends, producing a **hard lit disc at the tip**; the exponential dies
  on its own, so the mesh can simply stop.
- **Emission through an absorbing column, not a coverage mask.** `1 - exp(-tau)`
  saturates the dense near-nozzle core to a flat blinding white (the sea-level
  look) while the thin outer plume stays translucent and feathers to nothing at
  the silhouette, with no edge mask.

The visible length falls out of the same model: `VISIBLE_RADIUS_GROWTH` is the
expansion factor at which emission has died, so the billboard is exactly as long
as the visible plume. At sea level a separate mixing-limited length caps it,
because entrainment destroys the jet after a few tens of diameters however
slowly it expands.

**Packed-uniform warning.** `PlumeParams` addresses unrelated scalars
positionally (`anim.w`, `shock.z`, …). Repurposing a lane is a rename, not an
edit: audit every reader on both sides first. Getting this wrong erased the
vacuum plume entirely — see INC-0020.

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

`p_exit` comes from the engine's authored design point —
`Engine::optimized_for` (`EngineOptimization`), threaded onto the component from
the catalog: `Atmosphere` 55 kPa, `Balanced` 25 kPa, `Vacuum` 7 kPa. Exit
pressure scales with throttle (chamber pressure does), so a throttled-down
engine near the pad is *more* overexpanded — shorter, with a harder shock train.

A vacuum-optimised bell fired at sea level is therefore strongly overexpanded and
shows a pronounced waist and shock train, while a sea-level bell at the same
altitude runs nearly perfectly expanded — from the same code, differing only by
the authored design point.

## Screenshot preset

`just screenshot plume` boots a plain orbit (space/planet backdrop), forces the
engine to full throttle via `PlumeDebugOverride`, and frames the craft
three-quarter on the engine (`craft_context` in `screenshot.rs`: focus centered a
few metres down the stack, up = the ship's nose axis). `THALOS_PLUME_PRESSURE`
(Pa) scrubs the ambient pressure so the sea-level ↔ vacuum look is reproducible
regardless of the craft's real orbit altitude; the usual
`THALOS_SCREENSHOT_{AZIMUTH,ELEVATION,DISTANCE}` reframe it.
