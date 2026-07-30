# Vehicle flow effects

Contrails, rocket exhaust, reentry plasma, vapour cones, heat haze and pad dust
are usually treated as six unrelated features. They are not. **Every one of them
is a participating medium attached to a vehicle, driven by the same handful of
freestream numbers**, and they split cleanly on exactly one axis: whether they
have *memory*.

| | zero-memory (a function of the current state) | memory (shed into the air, ages) |
|---|---|---|
| effects | engine plume ✅, **reentry shock layer** ✅, vapour/sonic cone, afterburner, heat haze | contrails, rocket smoke trail, reentry ablation wake, wingtip vortices, pad dust |
| representation | analytic proxy geometry + physics chain in the fragment shader | ring buffer of aged samples → swept tube |
| status | plume and shock layer built | not built |

That split, not the effect names, is what decides the code. **No general particle
system**: the plume is the existence proof that an analytic emission model plus a
ray-marched proxy beats particles here, and it already composites, sorts and
screenshots correctly — things a GPU particle layer would have to re-earn against
`composite_order` and BigSpace.

## The signal boundary

`rendering::flow::FlowSignals` is the single typed answer to *what air is this
vehicle flying through, how fast, and how hard is it being heated*. Visual code
reads it; nothing reaches into the simulation itself.

Fields: `in_atmosphere`, `altitude_m`, `ambient_pressure_pa`, `density_kg_m3`,
`static_temp_k`, `speed_of_sound_m_s`, `airspeed_m_s`, `mach`,
`dynamic_pressure_pa`, `stagnation_temp_k`, `heat_flux_w_m2`, `flow_from_dir`
(render space) and `flow_from_local` (craft axes), `nose_radius_m`,
`craft_radius_m`, `craft_half_extents_m`.

**Single writer: `update_flow_signals`.** `PlumeSignals` stays as it is —
throttle, ignition and nozzle pressure ratio are genuinely *per engine*, while
everything above is per vehicle. The plume now takes its ambient half from here
rather than resolving atmosphere itself, so the plume and the shock layer cannot
disagree about the air they share.

Three properties are load-bearing:

- **It is not derived from `AeroReadout`.** That readout only exists inside the
  Avian bubble, which is precisely where reentry *is not*: a vehicle entering
  from orbit is on the canonical propagator, far above any bubble. The flow state
  is resolved from canonical ship state against the dominant body, so it is valid
  from orbit to touchdown with no regime switch.
- **Airspeed is relative to the co-rotating airmass.** Body spin is a few percent
  of entry speed, and heat flux goes as `v³`, so dropping it is a ~15 % error in
  exactly the regime that matters.
- **Heating is Sutton–Graves, `q = K·sqrt(ρ/R_n)·v³`.** This is the only quantity
  that separates *fast in thin air* from *fast in thick air* — i.e. a benign
  orbital pass from a fireball. Any brightness that rides on speed alone lights
  up every orbiting vehicle; there is a test pinning that.

`FlowDebugOverride` drives the signals from an authoring surface, and overrides go
on the **inputs** (density, airspeed, temperature) so the derived quantities stay
mutually consistent. Forcing `heat_flux` directly while airspeed sat at zero would
render a state no atmosphere can produce, and the probe would stop being evidence
about the real envelope.

## Shared proxy geometry

`flow::axial_proxy_prism_mesh` is one closed, outward-wound prism template
(`xy` = unit circle, `z` = axial fraction, cap centres at `xy = 0`) that every
ray-marched flow effect maps onto its own bounding surface — the plume scales its
rings by the envelope bound, the shock layer wraps them onto an ellipsoid. It is
**never seen**: it exists so the rasterizer visits every pixel the volume can
touch, and the silhouette comes from the density integral. Back faces are culled
so each ray shades exactly once.

## Reentry shock layer

Built 2026-07-29. See `reentry.md`.

## Not built

In the order they are worth doing:

1. **`RibbonTrail`** — one primitive, N emitters: a ring buffer of aged samples
   rendered as a camera-facing swept tube using the plume's chord integral.
   Contrails, soot trails, ablation wakes and vortices are then emitter configs
   plus a lifetime law. Three constraints are already known: samples must live in
   the **body-fixed frame** (a trail stored in render coordinates disintegrates
   the moment `RenderOrigin` shifts); emission must be **gated on time warp**
   rather than faked; and lifetime must come from **ambient ice supersaturation**,
   which is the whole difference between a stub that dies in seconds and the
   minutes-long contrails of a real sky.
   **A contrail is a *scattering* medium, not an emissive one** — it must take sun,
   the `thalos::shadow` cascade and `object_aerial_recession`, or it reads as a
   white decal. Same primitive as the plume, different source term.
2. **Vapour/sonic cone** — cheapest of the remaining set. Cone half-angle is
   `asin(1/M)`; the visible collar is a condensation shell where the local
   pressure drop takes static temperature below the dew point. Gated on Mach *and*
   humidity, and not emissive.
3. **Hull glow** — not an effect: an emissive term on the hull material driven by
   an integrated per-part heat state. Belongs in shading.
4. **Heat haze / schlieren** — screen-space refraction, and therefore a
   *fullscreen* pass that **must claim a slot in `composite_order`**. Bias 0 is
   the documented failure.
5. **Injecting contrails into the volumetric cloud density field** (the
   `CloudEmissionProfile` seam) — the correct long-term answer for contrails seen
   from orbit, self-shadowing and multi-scatter lighting. Deferred because the
   cloud march is calibrated by a fill LUT built around statistical coverage, and
   threading a thin line source through it is a project, not a phase.

## Verification

Every effect needs a `FlowDebugOverride`-driven preset, because none of these
states is reachable headlessly by flying to it: `plume`, `plume-skyline`,
`reentry`.

**All of them are `Transparent3d`, so all of them must be judged on the
`plume-skyline` framing**, not on a hero shot. INC-20260725T185440Z is exactly the
failure where a transparent looks perfect against terrain while being erased on
every sky pixel, and it only reproduces with the camera pitched above the local
horizontal.
