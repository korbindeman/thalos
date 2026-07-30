// Rocket-engine exhaust plume — an axisymmetric emitting gas column whose shape
// is set by the nozzle/ambient pressure ratio and whose brightness *follows from
// that shape* rather than from an authored fade curve.
//
// The model, in one place:
//
//   R(s)  = R0·lip + tan(theta)·s          free-expansion cone off the nozzle lip
//           × barrel(s)                    shock-cell pinching (needs back-pressure)
//   rho   ∝ (R0/R)²                        mass conservation along the column
//   T     ∝ T_exit · (R0/R)^(2(gamma-1))   adiabatic (expansion) cooling
//   T    ×= exp(-e·s/R0)                   entrainment cooling (atmosphere only)
//   S     = exp(-W·(1/T − 1))              visible-band emission, Wien side
//   tau   ∝ rho · chord                    optical depth across the line of sight
//   L     = S · (1 − exp(−tau))            emission through an absorbing column
//
// Three consequences make this cover pad-to-orbit with no regime switch:
//
//  * Near the nozzle tau >> 1, so the core saturates to a flat, blindingly bright
//    column — the sea-level look (dense, shock-celled).
//  * In vacuum the column cools by expanding, and the exponential Wien term
//    collapses, so the cone dissolves on its own with no tip taper.
//  * At sea level it barely expands, so *entrainment* of ambient air is what
//    cools it along its length. Without that term a sea-level plume stays
//    uniformly incandescent end to end.
//
// ONE LENGTH AUTHORITY. The billboard is cut exactly where `L` above — summed
// over *both* layers and scaled by the gain — falls below an absolute visibility
// floor; the CPU solves this same chain for that station (`visible_length_m` in
// `plume.rs`). Never cap the mesh length by anything the fragment stage cannot
// see: a limit the emission model doesn't know about ends the geometry while the
// column is still incandescent, which is a flat lit rim hanging in mid-air. See
// INC-20260724T235437Z-plume-ended-on-a-lit-rim.
//
// A turbulent shear layer (the sheath) wraps the core: cool and violet in vacuum,
// hot and orange in atmosphere where the fuel-rich exhaust afterburns with
// entrained air. Everything about the response lives CPU-side in
// `crate::rendering::plume`; this shader renders the resolved profile.

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}

// Resolved per-engine plume profile. Mirrors `PlumeParams` (Rust, ShaderType);
// packed as vec4s so the std140 layout is unambiguous.
struct PlumeParams {
    // rgb = hot core colour (near-nozzle / shock nodes), a = HDR radiance scale.
    core_color: vec4<f32>,
    // rgb = expanded-plume colour, a = nozzle exit radius R0 (m).
    mid_color: vec4<f32>,
    // rgb = shear-layer / sheath colour, a = visible axial length L (m).
    edge_color: vec4<f32>,
    // x = lip radius scale, y = tan(core half-angle),
    // z = shear-layer spread rate, w = adiabatic exponent 2(gamma-1).
    shape: vec4<f32>,
    // x = core opacity kappa, y = sheath opacity kappa,
    // z = shock-cell wavenumber (rad/m), w = shock strength 0..1.
    shock: vec4<f32>,
    // x = shock decay length (m), y = afterburn 0..1,
    // z = turbulence amplitude, w = RESERVED (throttle; no longer read here —
    // the CPU folds throttle into kappa, entrainment and the flicker rate).
    mixing: vec4<f32>,
    // x = time (s), y = per-engine seed, z = ignition 0..1,
    // w = entrainment cooling rate (per nozzle radius; 0 in vacuum).
    anim: vec4<f32>,
    // Turbulent motion. x = eddy growth per axial metre,
    // y = convection rate (eddies/s), z = azimuthal swirl (rad/s),
    // w = radial wobble amplitude.
    flow: vec4<f32>,
    // x = tail dispersal growth, y = potential-core length (m),
    // z = flicker amplitude, w = flicker rate (Hz).
    tail: vec4<f32>,
    // x = exit temperature (normalized; ignition transient),
    // y = core turbulence weight, z = tail turbulence boost,
    // w = shock-cell lengthening per axial metre.
    therm: vec4<f32>,
}

@group(3) @binding(0) var<uniform> plume: PlumeParams;

// -- cheap hash / value noise for the turbulent shear layer ------------------

fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash2(i);
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Value noise whose x axis is *periodic* with an integer period.
//
// The turbulence field is now sampled on the column's true azimuth (see
// `turbulence`), which wraps at a full turn. Plain `value_noise` would put a
// visible seam along one meridian — a permanent stationary scar down the side of
// the plume, which is worse than the camera-locked field it replaced. Wrapping
// the integer lattice makes the field genuinely periodic around the column, so
// there is no seam to find from any angle.
//
// `period` MUST be a whole number: the lattice is integer-spaced, so a
// fractional period puts the wrap partway through a cell and reintroduces the
// discontinuity it exists to remove.
fn hash2_wrapped(p: vec2<f32>, period: f32) -> f32 {
    let x = p.x - period * floor(p.x / period);
    return hash2(vec2<f32>(x, p.y));
}

fn value_noise_wrapped(p: vec2<f32>, period: f32) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash2_wrapped(i, period);
    let b = hash2_wrapped(i + vec2<f32>(1.0, 0.0), period);
    let c = hash2_wrapped(i + vec2<f32>(0.0, 1.0), period);
    let d = hash2_wrapped(i + vec2<f32>(1.0, 1.0), period);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// -- emission ---------------------------------------------------------------

// Dimensionless Wien parameter hc/(lambda·k·T_exit) for the visible band at
// combustion-chamber temperature. Larger = the plume darkens faster as it cools.
// MIRRORED in `plume.rs` (`WIEN`) — the CPU solves this same curve for the mesh
// length, so the two constants must stay equal.
const WIEN: f32 = 3.0;

// Restores the authored strength of the Mach-disk compression nodes now that
// `pow(fc, 3)` is applied per march sample instead of once per chord.
//
// That concentration exponent was tuned against a fragment that evaluated `fc`
// ONCE, at the chord's impact parameter, and applied the result along the whole
// chord. Inside a real integral the same expression is averaged over the path,
// where `fc` falls away from the axis, so the on-axis peak lands at
//
//   ∫(1−w²)^3.5 dw / ∫(1−w²)^0.5 dw = 0.859 / 1.571 = 0.547
//
// of what it used to be (numerator = node weight × core density `fc^0.5`,
// denominator = the density alone). Measured as a visibly softer shock train in
// the matched side-on capture. The radial *shape* is now the more physical of
// the two — a Mach disk is a volume near the axis, not a flat mask — so the fix
// is to keep the shape and put the authored peak back, not to re-tune the
// exponent by eye.
const SHOCK_NODE_MARCH_GAIN: f32 = 1.829;

// Visible-band emission from a cooling gas, normalised to 1 at chamber
// temperature.
//
// This is deliberately *not* a grey-body T^4. A rocket plume radiates in the
// visible from the Wien side of the Planck curve, where output falls off
// exponentially in 1/T rather than polynomially — which is why a real plume goes
// dark over a few nozzle diameters instead of trailing off forever. It is also
// what lets the geometry end without a lit disc at the tip: by the time the
// column has cooled enough, this term has already collapsed.
fn band_emission(temp_norm: f32) -> f32 {
    return exp(-WIEN * (1.0 / max(temp_norm, 1e-3) - 1.0));
}

// -- turbulent structure ----------------------------------------------------

// Dimensionless "eddy count" from the exit plane: turbulent structures in a jet
// grow linearly with distance, so integrating ds/eddy_size(s) gives a coordinate
// in which they are all the *same* size. Sampling noise on this coordinate
// instead of on raw distance is what makes structures coarsen as they travel
// downstream — the strongest single cue that this is a turbulent column and not
// a scrolling texture.
fn eddy_coord(s: f32) -> f32 {
    let g = max(plume.flow.x, 1e-3);
    let e0 = max(plume.mid_color.a * plume.shape.x, 1e-3);
    return log(1.0 + g * s / e0) / g;
}

// Low-frequency combustion roughness. Only ever *dims* and *shortens* the
// column, never lengthens it, so the visible plume always stays inside the mesh
// the CPU sized for the unflickered state.
fn flicker() -> f32 {
    let n = value_noise(vec2<f32>(plume.anim.x * plume.tail.w, plume.anim.y * 53.0));
    return 1.0 - plume.tail.z * n;
}

// Progress from the end of the potential core to the visible tip. 0 inside the
// coherent near field, 1 where the jet has fully broken down.
fn breakup(s: f32) -> f32 {
    let core_len = max(plume.tail.y, 1e-3);
    let len = max(plume.edge_color.a, core_len * 1.001);
    return smoothstep(core_len, len, s);
}

// Three independently-advected noise layers. Real exhaust does not slide as one
// rigid pattern: the shear layer convects slower than the core and the large
// downstream structures slower still, so the composite never reads as a single
// repeating scroll. Each layer also drifts azimuthally at its own rate, so the
// column rotates as well as flows.
// Cells around the full circumference, per layer. These reproduce the angular
// frequencies the pre-raymarch shader had (it sampled `value_noise` at `phi·K`
// for K = 7 / 15 / 3.5 with `phi` in radians, i.e. 2πK cells per turn), so the
// texture reads the same at the side-on framing the look was tuned against.
// Whole numbers, because `value_noise_wrapped` needs an integer period.
const TURB_PERIOD_FINE: f32 = 44.0;
const TURB_PERIOD_MID: f32 = 94.0;
const TURB_PERIOD_PUFF: f32 = 22.0;

const TAU: f32 = 6.2831855;

// `phi01` is the azimuth **in turns** around the column, measured in the
// engine's own frame — NOT derived from the view. The old billboard recovered a
// pseudo-azimuth from the quad's lateral coordinate (`asin(x)`), which meant the
// field was re-derived in whatever plane faced the camera that frame: a given
// eddy did not stay at a fixed place on the column, so the whole turbulent
// structure counter-rotated as the camera orbited. That is the cue that reads as
// "flat sticker" rather than "object", and it is why azimuth is now a genuine
// world-space quantity.
//
// Each layer's rotation and per-engine seed are applied *in turns, before* the
// scale to cell counts, so both stay exact rotations of a periodic field and the
// wrap survives them.
fn turbulence(phi01: f32, s: f32, b: f32) -> f32 {
    let xi = eddy_coord(s);
    let now = plume.anim.x;
    let adv = plume.flow.y;
    // Azimuthal drift, converted from rad/s to turns/s.
    let swirl = plume.flow.z / TAU;
    let seed = plume.anim.y;

    // Fine striations riding the core, the dominant texture of a dense jet.
    let fine = value_noise_wrapped(
        vec2<f32>(
            (phi01 + seed + swirl * now) * TURB_PERIOD_FINE,
            xi * 1.00 - adv * now,
        ),
        TURB_PERIOD_FINE,
    );
    // Shear-layer billows: finer around the circumference, convecting slower
    // because the mixing layer is subsonic relative to the core.
    let mid = value_noise_wrapped(
        vec2<f32>(
            (phi01 + seed * 1.7 - swirl * 1.9 * now) * TURB_PERIOD_MID,
            xi * 2.30 - adv * 0.62 * now,
        ),
        TURB_PERIOD_MID,
    );
    // Large, slow puffs — what a broken-down jet disperses into.
    let puff = value_noise_wrapped(
        vec2<f32>(
            (phi01 + seed * 0.6 + swirl * 0.5 * now) * TURB_PERIOD_PUFF,
            xi * 0.45 - adv * 0.33 * now,
        ),
        TURB_PERIOD_PUFF,
    );

    let near = 0.62 * fine + 0.38 * mid;
    return mix(near, puff, clamp(0.18 + 0.34 * b, 0.0, 1.0));
}

// -- plume geometry ---------------------------------------------------------
//
// `sheath_radius` is the true envelope: the fragment stage marches inside it and
// density reaches exactly zero on it. The proxy mesh is sized by `bound_radius`,
// a strictly larger, noise-free *bound* — it exists only to rasterize the pixels
// the volume can touch, and its silhouette is never seen.

// Phase along the shock train. Cells *lengthen* downstream as the train weakens
// — a constant wavenumber gives an evenly-spaced ladder of identical rungs,
// which is the one thing a real Mach-diamond train never looks like.
fn shock_phase(s: f32) -> f32 {
    let g = max(plume.therm.w, 1e-6);
    return plume.shock.z * log(1.0 + g * s) / g;
}

// Shock-cell phase modulation, decaying downstream. 1 = unmodulated.
fn barrel(s: f32) -> f32 {
    let strength = plume.shock.w;
    let decay = exp(-s / max(plume.mixing.x, 1e-3));
    return 1.0 + 0.10 * strength * decay * cos(shock_phase(s));
}

// Luminous core radius at axial station s (metres from the exit plane).
fn core_radius(s: f32) -> f32 {
    let r0 = plume.mid_color.a;
    let cone = r0 * plume.shape.x + plume.shape.y * s;
    return max(cone * barrel(s), 1e-4);
}

// Travelling constriction/bulge: eddies passing down the shear layer pinch and
// swell the column, so the silhouette boils instead of being a perfect analytic
// curve. A function of `s` alone, so the vertex and fragment stages agree
// exactly and the mesh edge stays *on* the envelope.
fn radius_wobble(s: f32, b: f32) -> f32 {
    let amp = plume.flow.w * (0.25 + 0.75 * b);
    let xi = eddy_coord(s);
    let n = value_noise(vec2<f32>(
        plume.anim.y * 19.0,
        xi * 0.8 - plume.anim.x * plume.flow.y * 0.8,
    ));
    return 1.0 + amp * (n - 0.5) * 2.0;
}

// Outer edge of the turbulent shear layer. Grows linearly with entrainment while
// the potential core survives; once the shear layer has eaten the core the jet
// *disperses* — growth accelerates and the column flares open, so it dissolves
// into widening structure rather than ending on a rim. Plus a thin lip so the
// sheath is never zero-width at the nozzle.
fn sheath_radius(s: f32) -> f32 {
    let r0 = plume.mid_color.a;
    let b = breakup(s);
    let spread = plume.shape.z * s * (1.0 + plume.tail.x * b * b);
    return (core_radius(s) + spread + r0 * 0.05) * radius_wobble(s, b);
}

// Conservative, noise-free upper bound on `sheath_radius(s)`, used to build the
// rasterization proxy and to clip the march. Every term is taken at its extreme:
// `barrel` peaks at 1 + 0.10·strength, `breakup` at 1, `radius_wobble` at
// 1 + amplitude. Monotonic in `s`, so `bound_radius(L)` bounds the whole column.
//
// This MUST stay an over-estimate. It is not a second length/shape authority —
// if it ever cut inside the envelope it would clip a still-emitting column, the
// same class of defect as INC-20260724T235437Z-plume-ended-on-a-lit-rim.
fn bound_radius(s: f32) -> f32 {
    let r0 = plume.mid_color.a;
    let cone = (r0 * plume.shape.x + plume.shape.y * s) * (1.0 + 0.10 * plume.shock.w);
    let spread = plume.shape.z * s * (1.0 + plume.tail.x);
    let wobble = 1.0 + plume.flow.w;
    return (cone + spread + r0 * 0.05) * wobble * 1.05;
}

// Compression-node brightness: peaks exactly where `barrel` pinches the column.
fn shock_node(s: f32) -> f32 {
    let strength = plume.shock.w;
    let decay = exp(-s / max(plume.mixing.x, 1e-3));
    return strength * decay * pow(max(0.0, -cos(shock_phase(s))), 8.0);
}

// -- vertex: a proxy hull that only has to *bound* the volume ----------------
//
// The mesh is a closed prism around the exhaust axis, sized by `bound_radius`.
// Nothing about the plume's appearance comes from it: it exists so the
// rasterizer visits every pixel the column can cover, and the fragment stage
// does the rest by integrating along the real view ray.
//
// This replaced a camera-facing quad, which failed by construction as the view
// swung onto the exhaust axis: the strip narrowed to a sliver and then to
// nothing, so a plume viewed end-on — the angle at which a real one is at its
// *brightest*, because the line of sight runs the whole length of the column —
// disappeared entirely. No choice of "stable perpendicular" fixes that; a flat
// quad seen edge-on covers no pixels whatever its orientation.

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    // Prism template: xy = unit circle direction, z = axial fraction 0..1.
    // Cap centres carry xy = 0.
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    // The plume frame, constant across the primitive. `axis` is the exhaust
    // direction; `e1`/`e2` span the cross-section and are engine-fixed, which is
    // what anchors the turbulent structure to the column instead of to the view.
    @location(1) @interpolate(flat) origin: vec3<f32>,
    @location(2) @interpolate(flat) axis: vec3<f32>,
    @location(3) @interpolate(flat) e1: vec3<f32>,
    @location(4) @interpolate(flat) e2: vec3<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    // Nozzle exit (plume-local origin) and exhaust axis (local -Y) in world space.
    let origin = world_from_local[3].xyz;
    let axis = normalize((world_from_local * vec4<f32>(0.0, -1.0, 0.0, 0.0)).xyz);

    // Engine-fixed cross-section basis, orthonormalised against the axis so a
    // scaled or slightly non-orthogonal model matrix cannot skew the azimuth.
    let raw_e1 = (world_from_local * vec4<f32>(1.0, 0.0, 0.0, 0.0)).xyz;
    var e1 = raw_e1 - axis * dot(raw_e1, axis);
    if (dot(e1, e1) < 1e-12) {
        let alt = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(axis.x) > 0.9);
        e1 = alt - axis * dot(alt, axis);
    }
    e1 = normalize(e1);
    let e2 = cross(axis, e1);

    let s = in.position.z * plume.edge_color.a;
    let world_pos = origin
        + axis * s
        + (e1 * in.position.x + e2 * in.position.y) * bound_radius(s);

    var out: VertexOutput;
    out.clip_position = position_world_to_clip(world_pos);
    out.world_pos = world_pos;
    out.origin = origin;
    out.axis = axis;
    out.e1 = e1;
    out.e2 = e2;
    return out;
}

// -- fragment: emission integrated along the actual view ray ----------------

// Longest march, and the target step in nozzle radii. The finest structure the
// column carries (shock cells, near-lip striations) is of order R0, so a quarter
// of that resolves it.
//
// The cap matters because the early-out below does *not* save the worst case: a
// dense sea-level column saturates and retires in a few steps, but a thin vacuum
// plume keeps high transmittance for its whole length and runs the full march
// while covering a lot of screen. 64 was compared against 128 on matched
// captures (vacuum side-on and end-on, sea-level side-on) and is
// indistinguishable, shock train included — so this is half the worst-case cost
// for no visible difference. Re-measure before raising it.
const MAX_STEPS: i32 = 64;
const STEP_PER_R0: f32 = 0.25;
// Transmittance below which the remaining column cannot change the pixel.
const TRANS_EPS: f32 = 0.003;

// Ray/capped-cylinder overlap in plume-local coordinates, as [near, far] along
// the ray. Returns far <= near when the ray misses. This is only a *bound* on
// the march — the true envelope is tested per step — so a loose fit costs steps,
// never correctness.
fn march_bounds(o: vec3<f32>, d: vec3<f32>, axis: vec3<f32>, radius: f32, length_m: f32) -> vec2<f32> {
    let oz = dot(o, axis);
    let dz = dot(d, axis);

    // Axial slab 0 <= s <= L.
    var t_near = 0.0;
    var t_far = 1.0e30;
    if (abs(dz) < 1e-6) {
        if (oz < 0.0 || oz > length_m) {
            return vec2<f32>(1.0, 0.0);
        }
    } else {
        let ta = (0.0 - oz) / dz;
        let tb = (length_m - oz) / dz;
        t_near = max(t_near, min(ta, tb));
        t_far = min(t_far, max(ta, tb));
    }

    // Infinite cylinder of the given radius about the axis.
    let o_perp = o - axis * oz;
    let d_perp = d - axis * dz;
    let a = dot(d_perp, d_perp);
    let c = dot(o_perp, o_perp) - radius * radius;
    if (a < 1e-12) {
        // Ray parallel to the axis: inside or outside for its whole length.
        if (c > 0.0) {
            return vec2<f32>(1.0, 0.0);
        }
    } else {
        let b = 2.0 * dot(o_perp, d_perp);
        let disc = b * b - 4.0 * a * c;
        if (disc < 0.0) {
            return vec2<f32>(1.0, 0.0);
        }
        let sq = sqrt(disc);
        t_near = max(t_near, (-b - sq) / (2.0 * a));
        t_far = min(t_far, (-b + sq) / (2.0 * a));
    }

    return vec2<f32>(max(t_near, 0.0), t_far);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let r0 = plume.mid_color.a;
    let length_m = plume.edge_color.a;

    let ray_origin = view.world_position;
    let to_frag = in.world_pos - ray_origin;
    if (dot(to_frag, to_frag) < 1e-12) {
        discard;
    }
    let ray_dir = normalize(to_frag);

    let bounds = march_bounds(
        ray_origin - in.origin,
        ray_dir,
        in.axis,
        bound_radius(length_m),
        length_m,
    );
    if (bounds.y <= bounds.x) {
        discard;
    }

    let span = bounds.y - bounds.x;
    let steps = i32(clamp(ceil(span / (STEP_PER_R0 * max(r0, 1e-3))), 8.0, f32(MAX_STEPS)));
    let ds = span / f32(steps);

    // The ignition transient and combustion roughness are properties of the
    // chamber, not of position, so they are resolved once for the whole ray.
    let flick = flicker();
    let temp_exit = plume.therm.x * flick;
    let afterburn = plume.mixing.y;

    // Two independent transmittances, one per layer. The pre-raymarch shader
    // integrated the core and the sheath separately and *summed* their emitted
    // radiance — i.e. neither layer absorbs the other. Keeping them separate
    // reproduces that exactly rather than quietly changing the balance the
    // propellant palettes were tuned against.
    var trans_core = 1.0;
    var trans_sheath = 1.0;
    var radiance = vec3<f32>(0.0);

    for (var i = 0; i < steps; i = i + 1) {
        let t = bounds.x + (f32(i) + 0.5) * ds;
        let p = (ray_origin - in.origin) + ray_dir * t;

        let s = dot(p, in.axis);
        if (s < 0.0 || s > length_m) {
            continue;
        }
        let radial = p - in.axis * s;
        let r = length(radial);

        // The envelope, evaluated where the ray actually is.
        let rs = sheath_radius(s);
        let rc = min(core_radius(s), rs * 0.94);
        let fs = max(1.0 - (r * r) / (rs * rs), 0.0);
        if (fs <= 0.0) {
            continue;
        }
        let fc = max(1.0 - (r * r) / (rc * rc), 0.0);

        // LOCAL radial densities — the same compact-support profiles as before,
        // but as densities at a point rather than pre-integrated chords. Their
        // perpendicular chord integrals are (pi/2)*R*(1-(p/R)^2) and
        // (16/15)*R*(1-(p/R)^2)^(5/2), the exact expressions this shader used to
        // evaluate in closed form, so a side-on ray still accumulates precisely
        // the optical depth the look was tuned against. Compact support is what
        // feathers the silhouette; it now feathers in every direction, including
        // along the axis, instead of only across the billboard.
        let rho_core = sqrt(fc);
        let rho_sheath = fs * fs;

        let frac = s / max(length_m, 1e-3);
        let b = breakup(s);

        // Thermodynamics at this station (unchanged chain).
        let er = r0 / rc;
        let node = shock_node(s) * pow(fc, 3.0) * SHOCK_NODE_MARCH_GAIN;
        let entrained = exp(-plume.anim.w * s / r0);
        let temp = min(temp_exit * pow(er, plume.shape.w) * entrained * (1.0 + 0.8 * node), 1.6);
        let density = er * er;

        // True azimuth about the column in the engine's own frame.
        let phi01 = fract(atan2(dot(radial, in.e2), dot(radial, in.e1)) / TAU + 1.0);
        let turb = turbulence(phi01, s, b);

        let amp = plume.mixing.z * mix(0.12, 1.0, b) * smoothstep(0.0, 0.06, frac);
        let amp_sheath = clamp(amp * (1.0 + plume.therm.z * b), 0.0, 1.0);
        let amp_core = clamp(amp * plume.therm.y, 0.0, 1.0);
        let mod_sheath = mix(1.0, 0.28 + 1.5 * turb, amp_sheath);
        let mod_core = mix(1.0, 0.45 + 1.1 * turb, amp_core);

        // Optical depth over this step. The `/(2*r0)` normalisation is carried
        // over verbatim from the closed-form version.
        let dtau_core = plume.shock.x * density * (rho_core * ds / (2.0 * r0)) * mod_core;
        let dtau_sheath = plume.shock.y * density * (rho_sheath * ds / (2.0 * r0)) * mod_sheath;

        let temp_sheath = temp * mix(0.62, 0.88, afterburn);
        let src_core = band_emission(temp) * mix(1.0, 1.8, afterburn);
        let src_sheath = band_emission(temp_sheath) * mix(1.0, 2.4, afterburn);

        let col_core = mix(plume.mid_color.rgb, plume.core_color.rgb, smoothstep(0.80, 1.0, temp));
        let col_sheath = mix(plume.edge_color.rgb, plume.mid_color.rgb, smoothstep(0.22, 0.78, temp_sheath));

        // Emission through an absorbing column, accumulated front to back. With
        // the source constant along the path this telescopes to the old
        // `S*(1 - exp(-tau))` exactly; it stops being equivalent only where the
        // ray genuinely crosses a temperature gradient, which is the whole point.
        let a_core = 1.0 - exp(-dtau_core);
        let a_sheath = 1.0 - exp(-dtau_sheath);
        radiance += trans_core * col_core * src_core * a_core;
        radiance += trans_sheath * col_sheath * src_sheath * a_sheath;
        trans_core *= 1.0 - a_core;
        trans_sheath *= 1.0 - a_sheath;

        if (trans_core < TRANS_EPS && trans_sheath < TRANS_EPS) {
            break;
        }
    }

    // NB: `anim.w` is the entrainment rate (consumed above), *not* a radiance
    // trim — it must not appear here. It multiplied this product until
    // 2026-07-22, which zeroed the whole plume in vacuum (rate 0) and dimmed it
    // ~60x at sea level. See docs/incidents/0020.
    radiance *= plume.core_color.a * plume.anim.z * mix(1.0, flick, 0.6);

    // Bevy's AlphaMode::Add uses PREMULTIPLIED_ALPHA_BLENDING
    // (out = src.rgb + dst.rgb * (1 - src.a)); a pure-additive glow emits the
    // premultiplied radiance with alpha 0 so the background is preserved.
    return vec4<f32>(radiance, 0.0);
}
