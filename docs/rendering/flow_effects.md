# Vehicle flow effects

Contrails, rocket exhaust, reentry plasma, vapour cones, heat haze and pad dust
are usually treated as six unrelated features. They are not. **Every one of them
is a participating medium attached to a vehicle, driven by the same handful of
freestream numbers**, and they split cleanly on exactly one axis: whether they
have *memory*.

| | zero-memory (a function of the current state) | memory (shed into the air, ages) |
|---|---|---|
| effects | engine plume ✅, **reentry shock layer** ✅, **vapour/sonic cone** ✅, afterburner, heat haze | contrails, rocket smoke trail, reentry ablation wake, wingtip vortices, pad dust |
| representation | analytic proxy geometry + physics chain in the fragment shader | ring buffer of aged samples → swept tube |
| status | plume, shock layer and vapour cone built | not built |

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
`relative_humidity_frac`, `craft_radius_m`, `craft_half_extents_m`,
`craft_bounds_centre_m`, `craft_rotation`.

**Craft bounds are an AABB with a centre, not half-extents about the origin.** A
craft's origin is not its centre — a rocket's sits near the engine end — so a body
fitted about the origin is half empty space at one end, and an attached shell
wraps that phantom volume. `craft_rotation` is published for the same reason
`flow_from_local` is: so an effect can map world-space quantities (the sun) into
the craft frame **without a second query on the craft's `Transform`**, which
paired with an effect's own `&mut Transform` is a B0001 boot panic.

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
about the real envelope. **It cannot override environmental presence**:
`in_atmosphere` always comes from the real body atmosphere at the craft's real
altitude. A probe authors a point *inside* an atmosphere; it cannot manufacture
air around an orbiting craft.

`flow_from_dir` and `flow_from_local` point to the side the freestream arrives
**from** (upstream), not the direction the air moves in craft space. The latter is
their negative. Keeping that convention at the producer matters because both the
reentry cap and the vapour bell use the former to find the leading end; a second
negation puts the effect ahead of a live vehicle even though a hand-authored
capture can still look correct.

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

## Transonic vapour cone

Built 2026-07-30 — `rendering::vapor_cone` + `assets/shaders/vapor_cone.wgsl`.

**The one that is not an emitter.** The plume and the shock layer are hot gas and
only ever add light; the vapour cone is condensed water droplets *scattering
sunlight*, so its source term is illumination with a Henyey–Greenstein
forward-scatter lobe, and its material blends rather than adds — a cloud is not
transparent to what it covers. Shading it like the plume gives a self-lit white
cone that looks identical at midnight and reads as a decal.

It is also **not the shock**. The shock is invisible; the visible cloud is
condensation in the low-pressure region behind the expansion. That is why the
collar lives in a *window* around Mach 1 (0.75 → 1.35, peaking 0.92–1.05) rather
than everywhere supersonic, and why it needs humidity at all.

**Three gates, all required at once** — and this is the whole reason the effect
reads as a photographed moment rather than as permanent decoration:

| gate | why |
|---|---|
| transonic Mach window | above ~1.35 the low-pressure region has left the body |
| relative humidity ≥ 0.35 | the same aircraft shows a collar over the sea and nothing over a desert |
| dynamic pressure ≥ 8 kPa | a transonic pass in the stratosphere carries no visible moisture |

Humidity is the new signal on `FlowSignals`, and it is **a named stand-in**: the
atmosphere model carries no water vapour, so it is currently an exponential
altitude profile. The cloud system already models weather; when it publishes a
per-column moisture field, that is what replaces it. Contrail persistence will
want the same field, which is why it lives on the shared boundary rather than
inside this module.

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
2. **Hull glow** — not an effect: an emissive term on the hull material driven by
   an integrated per-part heat state. Belongs in shading.
3. **Heat haze / schlieren** — screen-space refraction, and therefore a
   *fullscreen* pass that **must claim a slot in `composite_order`**. Bias 0 is
   the documented failure.
4. **Injecting contrails into the volumetric cloud density field** (the
   `CloudEmissionProfile` seam) — the correct long-term answer for contrails seen
   from orbit, self-shadowing and multi-scatter lighting. Deferred because the
   cloud march is calibrated by a fill LUT built around statistical coverage, and
   threading a thin line source through it is a project, not a phase.

## Verification

Every effect needs a `FlowDebugOverride`-driven preset, because none of these
states is reachable headlessly by flying to it: `plume`, `plume-skyline`,
`reentry`, `vapor-cone`. The reentry and vapour probes boot real atmospheric
scenarios; overrides choose their flow point but never bypass the atmosphere
boundary. The vapour probe also authors its nose-side arrival direction because a
cold host can wait minutes for terrain and lose the cruise scenario's initial
velocity. Live direction is pinned at the `FlowSignals` producer instead.

**All of them are `Transparent3d`, so all of them must be judged on the
`plume-skyline` framing**, not on a hero shot. INC-20260725T185440Z is exactly the
failure where a transparent looks perfect against terrain while being erased on
every sky pixel, and it only reproduces with the camera pitched above the local
horizontal.
