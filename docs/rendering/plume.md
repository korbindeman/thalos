# Rocket engine plumes

Liquid-engine exhaust is rendered as a **data-driven, pressure-responsive
volumetric effect** evaluated from engine state and local atmosphere. This is the
consumer-side visual layer over the shipyard's `Engine` parts; it owns *how a
firing engine looks*, not *what the engine is* (that stays in `thalos_shipyard`).

The full design direction (propellant families, layered effects, the volumetric
path for solids) is captured in the uploaded concept note; this doc tracks what
is **built** and the seams the later phases extend.

## Status

**Phase 4 (built, 2026-07-29) — the plume became a volume.** The billboard is
gone: the fragment stage now integrates the same emission model along the
**actual view ray** through an axisymmetric envelope, rasterized by a proxy
prism that is never seen. This fixed a failure that was structural, not a tuning
miss — see *The view-ray integral* below. Screenshot-verified across an
elevation sweep (side-on → straight down the exhaust). Motion and the
orbiting-camera cue remain unverifiable from stills.

**Phase 3 (built, 2026-07-25) — turbulent motion + a single length authority.**
The column now convects like a turbulent jet rather than sliding as one rigid
noise pattern (see *Motion*), and it ends by going dark instead of by running out
of mesh (see *One length authority*). Screenshot-verified at sea level and in
vacuum: the tail fades to black over ~15 % of its length with no rim, and the
silhouette feathers. **Motion itself is unverified** — a still cannot show it;
that needs a play session or a video capture.

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

Verify: `just screenshot plume` for the column itself, and
`just screenshot plume-skyline` for how it *composites* — a firing engine seen
from below against both sky and terrain, with the camera pitched above the
horizontal. The second exists because the plume is a `Transparent3d` emitter and
therefore invisible to the fullscreen composites' depth clip: when the
atmosphere sorted in front of it, the column was erased on every sky pixel while
looking perfectly correct against ground
(INC-20260725T185440Z-plume-erased-by-the-sky). Any change to plume ordering,
blending, or the composite stack must be judged on that framing, not on the
black-sky hero shot, which cannot show the failure.

Live feel (throttle sweeps, startup/shutdown transients, in-flight pressure
sweep on a real ascent) is a user play session.

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
                                        plume.wgsl (proxy prism + ray integral)
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

## One length authority

**The visible length falls out of the model, and nothing else may touch it.**
`visible_length_m` bisects the CPU twin of the fragment's own chain — both layers,
gain included — for the station where the rendered radiance drops below
`VISIBLE_RADIANCE`, and the march is cut exactly there. Every input that
should shorten a plume feeds that chain instead of trimming its result:

| input | acts through |
|---|---|
| throttle | mass flow → `κ`, and the mixing length |
| ignition transient | exit temperature `T_exit` |
| back-pressure | entrainment rate, expansion angle |
| propellant | `κ` (opacity), radiance |

Two rules keep this honest, both learned the expensive way
(INC-20260724T235437Z-plume-ended-on-a-lit-rim):

- **No cap the shader cannot see.** The mixing-limited length used to be a
  `min()` on the mesh while the fragment's entrainment rate was an unrelated
  authored constant — two numbers for one physical process, so the geometry ended
  where the column was still at 12 % of exit radiance. The entrainment rate is
  now *derived* from the mixing length, so emission genuinely dies within it.
- **The visibility floor is absolute, not a fraction of peak.** The core
  saturates at an HDR radiance of order 10; 0.3 % of that is still clearly
  visible after tonemapping. Relative floors are meaningless downstream of a
  tonemapper.

Both layers also use radial density kernels with **compact support** —
`(1-(r/R)²)^½` for the core, `(1-(r/R)²)²` for the shear layer, with chord
integrals `(π/2)·R·(1-(p/R)²)` and `(16/15)·R·(1-(p/R)²)^{5/2}`. A saturated
top-hat has a razor silhouette in every direction, because the only thing that
can end it is the chord going to zero; compact kernels reach exactly zero at the
mesh boundary however optically thick the column is, so the edges feather on
their own.

**Packed-uniform warning.** `PlumeParams` addresses unrelated scalars
positionally (`anim.w`, `shock.z`, …). Repurposing a lane is a rename, not an
edit: audit every reader on both sides first. Getting this wrong erased the
vacuum plume entirely — see INC-0020.

## The view-ray integral

**The plume is integrated along the real view ray, not across an assumed
perpendicular chord.** This is the same principle that earlier replaced a
coverage mask with `1 − exp(−τ)`, taken one step further, and it is what makes
the column hold up from every direction.

The mesh is a closed **proxy prism** around the exhaust axis, sized by
`bound_radius(s)` — a strictly conservative, noise-free over-estimate of the
envelope. It is a rasterization bound and nothing else: its silhouette is never
seen, because the visible edge comes from the density integral reaching zero.
Front faces are kept (back faces culled) so each ray shades exactly once and the
plume keeps depth-testing at the surface where the column starts.

The fragment reconstructs the world-space ray, clips it to a bounding cylinder,
and marches, accumulating the *same* chain per sample: the compact-support
radial profiles are now **local densities** rather than pre-integrated chords,
so a side-on ray still accumulates exactly the optical depth the look was tuned
against (`∫(1−(r/R)²)^½` along a perpendicular chord *is* `(π/2)·R·(1−(p/R)²)`,
the closed form it replaced). Core and sheath keep independent transmittances,
preserving the previous "neither layer absorbs the other" balance.

### Why the billboard had to go

It was a **cylindrical billboard**: a flat strip locked to the exhaust axis and
rotated about it to face the camera. That construction degenerates as the view
swings onto the axis — and the degenerate case is not a rare grazing angle, it is
looking up the exhaust, where a real plume is at its **brightest**, because the
line of sight runs the entire length of the column. Measured on the `plume`
preset at fixed azimuth: correct at 1°, a flat triangular flap with hard straight
edges by 80°, and at 90° **the plume vanished completely**. No "stable
perpendicular" fallback can help; a quad seen edge-on covers no pixels whatever
its orientation. The old code's comment that this degenerates "to a sliver, which
is correct end-on" was the bug, stated out loud.

Two lesser angle defects went with it: at oblique angles the perpendicular-chord
assumption **under-estimated** the path length by roughly `1/sin(view-to-axis)`,
so the column thinned exactly as it should have thickened; and the turbulence
azimuth was recovered from the billboard's own lateral coordinate (`asin(x)`), so
the entire eddy field counter-rotated with the camera — the cue that reads as
"sticker" rather than "object".

### Consequences to respect

- **`bound_radius` must stay an over-estimate.** It is not a second shape
  authority. If it ever cut inside the envelope it would clip a still-emitting
  column — the class of defect in
  INC-20260724T235437Z-plume-ended-on-a-lit-rim.
- **Azimuth is engine-fixed, and its noise wraps.** `turbulence` takes the true
  azimuth in turns and samples `value_noise_wrapped`, whose period *must* be a
  whole number or a stationary seam appears down one meridian. The per-layer
  periods (44/94/22) reproduce the pre-raymarch angular frequencies.
- **Per-sample weighting dilutes what per-chord weighting did not.** The
  Mach-disk concentration `pow(fc, 3)` was authored against an `fc` evaluated
  once per chord; averaged inside the integral its on-axis peak lands at 0.547
  of that, which showed up as a visibly softer shock train. `SHOCK_NODE_MARCH_GAIN`
  restores the authored peak analytically rather than by eye. Any future term
  that concentrates toward the axis inherits this correction.

Shape and pressure response still change with no runtime mesh regeneration
(design note decision #1: procedural radial profile in the shader, not authored
blend targets).

**Known gap:** a camera *inside* the bounding prism (inside the exhaust) loses
the part of the volume behind the near plane. Fixing it properly means clamping
the march against scene depth, which would also make the plume intersect hull and
terrain volumetrically instead of at a flat depth-test.

## Motion

The reference point is KSP's Waterfall, which gets its life from four cheap
things: a noise texture **scrolled along the flow**, **several layers scrolling at
different rates** so nothing reads as one repeating pattern, an edge falloff, and
**controllers** (throttle, atmospheric density, and a `random` controller) driving
material properties through curves. It simulates nothing, and neither do we.

What this shader does, and why each piece is there:

- **Advection is a rate, in eddies per second** — not a scroll in normalized
  axial coordinates. Under the old normalized advection a *longer* plume moved its
  own structure more slowly, which is backwards.
- **Structures grow as they travel.** Noise is sampled on the **eddy coordinate**
  `ξ(s) = ∫ds/eddy_size(s)`, which has the closed form `ln(1 + g·s/e₀)/g` for a
  shear layer whose eddies grow linearly. A uniform grid in `ξ` is a self-similar
  grid of turbulent structures, so fine striations at the lip coarsen into large
  puffs downstream. This is the single strongest cue that the column is a
  turbulent jet and not a scrolling texture.
- **Three layers, three convection rates** (1.00 / 0.62 / 0.33) and three
  azimuthal drift rates. The shear layer genuinely does convect slower than the
  core, and the large structures slower still; the composite never repeats. The
  slow layer's weight ramps up toward the tail, where the jet has broken down.
- **The silhouette boils.** `radius_wobble` perturbs the envelope radius as eddies
  pass. It is a function of `s` alone so the vertex and fragment stages agree
  exactly. (Pre-Phase-4 this also kept the mesh edge *on* the envelope; the
  proxy prism now merely bounds it, but the two stages must still agree or the
  bound can clip the column.)
- **Laminar where it should be.** Turbulence amplitude is gated by `breakup(s)`,
  which is zero inside the potential core (the un-mixed cone that survives until
  the shear layer reaches the axis) and one where the jet has fully broken down.
  Past the core, sheath growth accelerates and the tail disperses.
- **Flicker.** Low-frequency combustion roughness on gain and exit temperature,
  worse at low throttle, damped in vacuum. It only ever *dims and shortens*, so
  the visible column always stays inside the span the CPU sized for the
  unflickered state.
- **Shock cells lengthen downstream.** A constant wavenumber produces an evenly
  spaced ladder of identical rungs; the phase is now `k·ln(1+g·s)/g`. Compression
  nodes are weighted hard toward the axis (`fc³`) so they read as lenses rather
  than flat rungs across a saturated column.

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
