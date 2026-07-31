// Reentry shock layer — the shock-heated air standing off a vehicle's windward
// side, rendered as an emitting shell integrated along the view ray.
//
// This is the plume's emission model applied to a different geometry, which is
// the point: both are hot gas whose brightness must *follow from* the physics
// rather than from an authored fade. What changes is where the gas is and what
// heats it.
//
//   d          distance from the craft origin
//   cos_t      how windward a point is (1 = stagnation point, 0 = the flank)
//   delta(t)   shock standoff, smallest at the stagnation point and sweeping
//              back as the shock goes oblique
//   u          position across the shell, 0 at the wall, 1 at the shock front
//   rho  = 4u(1-u)                 compact support at BOTH ends
//   T    = T_stag · aft · front     hottest at the stagnation point, near the shock
//   S    = exp(-W·(1/T - 1))       visible-band emission, Wien side (as the plume)
//   L    = S · (1 - exp(-tau))     emission through an absorbing shell
//
// Two properties are load-bearing:
//
//  * Compact support at both ends of the shell. A top-hat across the shell shows
//    a hard bright line at the wall and another at the shock front — two razor
//    edges. `4u(1-u)` reaches exactly zero on both surfaces, so the layer
//    feathers into the hull and into the freestream on its own.
//  * The march STOPS AT THE BODY. The shell is a spherical annulus, so a ray
//    aimed at the craft crosses it twice; integrating the far crossing would
//    paint the leeward shock over the hull that should hide it. Clamping the
//    march at the body sphere both fixes that and makes the marched interval
//    tight, because the annulus crossing *is* the interval.

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}

// Resolved shock-layer profile. Mirrors `ReentryParams` (Rust, ShaderType).
//
// Lanes are addressed positionally, so repurposing one is a rename, not an edit
// — audit every reader on both sides first (the plume learned this the hard way,
// see docs/incidents/0020).
struct ReentryParams {
    // rgb = hottest colour stop (blue-white plasma), a = HDR radiance gain.
    hot_color: vec4<f32>,
    // rgb = mid stop (orange-white), a = shell opacity kappa.
    mid_color: vec4<f32>,
    // rgb = coolest stop (deep orange), a = normalized stagnation temperature.
    cool_color: vec4<f32>,
    // xyz = craft-local body half-extents (m), w = unused.
    body: vec4<f32>,
    // x = luminous envelope thickness, y = shock standoff — both as fractions of
    // the body surface in normalized space.
    envelope: vec4<f32>,
    // xyz = freestream arrival direction in craft-local axes (unit),
    // w = standoff growth with obliqueness.
    flow: vec4<f32>,
    // x = time (s), y = seed, z = shimmer amplitude, w = supersonic ramp 0..1.
    anim: vec4<f32>,
}

@group(3) @binding(0) var<uniform> params: ReentryParams;

// Windward cutoff. Emission survives slightly *past* 90° from the flow, where a
// real bow shock has swept back into the wake, so the layer wraps the flank
// instead of stopping on a terminator line. The upper edge is well short of 1 so
// full brightness is concentrated near the stagnation point — at a wider setting
// the whole windward hemisphere lights evenly and reads as a glowing ball rather
// than a shock cap.
const WRAP_LO: f32 = -0.15;
const WRAP_HI: f32 = 0.75;

// Wien parameter for the visible band, against the reference temperature the CPU
// normalises the stagnation temperature by. Larger = emission collapses faster as
// the layer cools, which is what lets the flanks go dark without a mask.
const WIEN: f32 = 3.5;

const STEP_PER_SHELL: f32 = 0.2;
const MAX_STEPS: i32 = 64;
const TRANS_EPS: f32 = 0.004;

fn hash3(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
}

fn value_noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let w = f * f * (3.0 - 2.0 * f);
    let x00 = mix(hash3(i), hash3(i + vec3<f32>(1.0, 0.0, 0.0)), w.x);
    let x10 = mix(hash3(i + vec3<f32>(0.0, 1.0, 0.0)), hash3(i + vec3<f32>(1.0, 1.0, 0.0)), w.x);
    let x01 = mix(hash3(i + vec3<f32>(0.0, 0.0, 1.0)), hash3(i + vec3<f32>(1.0, 0.0, 1.0)), w.x);
    let x11 = mix(hash3(i + vec3<f32>(0.0, 1.0, 1.0)), hash3(i + vec3<f32>(1.0, 1.0, 1.0)), w.x);
    return mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z);
}

// Visible-band emission from hot gas, normalised to 1 at the reference
// temperature. Same Wien-side law as the plume, and for the same reason: a
// polynomial falloff leaves the flanks lit where the geometry ends.
fn band_emission(temp_norm: f32) -> f32 {
    return exp(-WIEN * (1.0 / max(temp_norm, 1e-3) - 1.0));
}

// Thickness of the LUMINOUS envelope at obliqueness `cos_t`, in normalized
// (body-surface) units. Smallest at the stagnation point and growing as the layer
// lies over, which is what gives it a swept teardrop rather than a concentric
// bubble.
//
// This is deliberately NOT the shock standoff. The shock sits a few percent of the
// nose radius out; the radiating region is the compressed layer plus the thermal
// boundary layer and the hot afterbody gas, several times thicker. Drawing only
// the standoff renders a centimetres-thin sheath that is invisible on a real
// vehicle — measured.
fn envelope_thickness(cos_t: f32) -> f32 {
    return params.envelope.x * (1.0 + params.flow.w * (1.0 - cos_t));
}

// Where the shock front sits across the envelope, 0 at the wall and 1 at the
// outer edge. The layer is brightest here — a bow shock has a sharply defined
// luminous leading edge, and a profile that peaks mid-envelope instead reads as a
// soft halo.
fn shock_station() -> f32 {
    return clamp(params.envelope.y / max(params.envelope.x, 1e-5), 0.05, 0.95);
}

// Scale from the body surface out to the proxy hull. Taken at the worst case
// (`cos_t = -1`) so the hull bounds the layer in every direction — it must stay an
// over-estimate, or the bound clips a still-emitting shell, the defect class of
// INC-20260724T235437Z-plume-ended-on-a-lit-rim.
fn hull_scale() -> f32 {
    return 1.0 + envelope_thickness(-1.0);
}

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    // Shared flow-effect prism template: xy = unit circle, z = axial fraction.
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    // The craft frame, constant across the primitive. The shell is hull-fitted, so
    // the fragment stage works in craft-local axes and the freestream arrives as a
    // uniform (`flow.xyz`) rather than as a baked-in orientation.
    @location(1) @interpolate(flat) origin: vec3<f32>,
    @location(2) @interpolate(flat) axis_x: vec3<f32>,
    @location(3) @interpolate(flat) axis_y: vec3<f32>,
    @location(4) @interpolate(flat) axis_z: vec3<f32>,
}

// The proxy hull is the craft's bounding ellipsoid grown by `hull_scale()`.
// Closed and convex, so culling back faces leaves exactly one fragment per ray.
@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    let axis_x = world_from_local[0].xyz;
    let axis_y = world_from_local[1].xyz;
    let axis_z = world_from_local[2].xyz;

    // Rings sweep the whole ellipsoid; the poles stay just short of degenerate so
    // the closing caps are real triangle fans.
    let u = mix(-0.999, 0.999, in.position.z);
    let ring = sqrt(max(1.0 - u * u, 0.0));
    let unit = vec3<f32>(in.position.x * ring, in.position.y * ring, u);
    let local = unit * params.body.xyz * hull_scale();

    let world_pos = world_from_local * vec4<f32>(local, 1.0);

    var out: VertexOutput;
    out.clip_position = position_world_to_clip(world_pos.xyz);
    out.world_pos = world_pos.xyz;
    out.origin = world_from_local[3].xyz;
    out.axis_x = axis_x;
    out.axis_y = axis_y;
    out.axis_z = axis_z;
    return out;
}

// Ray/unit-sphere overlap in normalized space, as [near, far] along the ray.
// `far <= near` means a miss. `d` is NOT unit here — the per-axis divide by the
// body half-extents rescales it — so the quadratic keeps its `a` term.
fn shell_hit(o: vec3<f32>, d: vec3<f32>, radius: f32) -> vec2<f32> {
    let a = dot(d, d);
    if (a < 1e-18) {
        return vec2<f32>(1.0, 0.0);
    }
    let b = 2.0 * dot(o, d);
    let c = dot(o, o) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if (disc < 0.0) {
        return vec2<f32>(1.0, 0.0);
    }
    let sq = sqrt(disc);
    return vec2<f32>((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let ramp = params.anim.w;
    if (ramp <= 0.0) {
        discard;
    }

    let half_extents = max(params.body.xyz, vec3<f32>(1e-3));
    let flow_local = params.flow.xyz;

    let to_frag = in.world_pos - view.world_position;
    if (dot(to_frag, to_frag) < 1e-12) {
        discard;
    }
    let ray_world = normalize(to_frag);

    // Into craft-local axes. The craft transform is rigid, so the inverse
    // rotation is the transpose — three dots against the basis columns.
    let rel = view.world_position - in.origin;
    let o_local = vec3<f32>(dot(rel, in.axis_x), dot(rel, in.axis_y), dot(rel, in.axis_z));
    let d_local = vec3<f32>(
        dot(ray_world, in.axis_x),
        dot(ray_world, in.axis_y),
        dot(ray_world, in.axis_z),
    );

    // Normalized space: the body is the unit sphere, the shell an annulus above
    // it. `t` stays a world-space distance, so step sizes remain metric.
    let o_n = o_local / half_extents;
    let d_n = d_local / half_extents;

    let outer = shell_hit(o_n, d_n, hull_scale());
    if (outer.y <= outer.x) {
        discard;
    }
    var t_start = max(outer.x, 0.0);
    var t_end = outer.y;

    // Stop at the body. The shell is an annulus, so a ray aimed at the craft
    // crosses it twice; integrating the far crossing would paint the leeward
    // shock over the hull that should hide it.
    let body_hit = shell_hit(o_n, d_n, 1.0);
    if (body_hit.y > body_hit.x && body_hit.x > t_start) {
        t_end = min(t_end, body_hit.x);
    }
    if (t_end <= t_start) {
        discard;
    }

    // Step to the shell's THINNEST metric thickness, so the narrowest
    // cross-section is still resolved. This is a step size only — optical depth is
    // normalised by the *local* thickness inside the loop, which is a different
    // quantity on an elongated body.
    let thinnest_shell_m = min(half_extents.x, min(half_extents.y, half_extents.z))
        * max(params.envelope.x, 1e-4);
    let span = t_end - t_start;
    let steps = i32(clamp(
        ceil(span / (STEP_PER_SHELL * thinnest_shell_m)),
        6.0,
        f32(MAX_STEPS),
    ));
    let ds = span / f32(steps);

    let temp_stag = params.cool_color.a;
    let kappa = params.mid_color.a;
    let now = params.anim.x;
    let shimmer_amp = params.anim.z;
    let seed = params.anim.y;

    var trans = 1.0;
    var radiance = vec3<f32>(0.0);

    for (var i = 0; i < steps; i = i + 1) {
        let t = t_start + (f32(i) + 0.5) * ds;
        let q = o_n + d_n * t;
        let dist = length(q);
        if (dist < 1e-4) {
            continue;
        }

        // Windward weight uses the *metric* direction from the craft centre, so an
        // elongated vehicle's flank is judged by where it really points.
        let p_local = q * half_extents;
        let cos_t = dot(normalize(p_local), flow_local);
        let w = smoothstep(WRAP_LO, WRAP_HI, cos_t);
        if (w <= 0.0) {
            continue;
        }

        let across = (dist - 1.0) / max(envelope_thickness(cos_t), 1e-5);
        if (across <= 0.0 || across >= 1.0) {
            continue;
        }

        // Peaks AT THE SHOCK and reaches exactly zero at both the wall and the
        // outer edge, so the layer feathers into the hull and into the freestream
        // with no mask. A symmetric bump would put the brightest gas halfway out
        // and lose the sharp luminous leading edge entry footage shows.
        let peak = shock_station();
        var rho = smoothstep(0.0, peak, across) * (1.0 - smoothstep(peak, 1.0, across));

        // Plasma shimmer. Deliberately small: a shock layer is a smooth continuum,
        // and heavy noise reads as fire rather than as compressed air.
        if (shimmer_amp > 0.0) {
            let n = value_noise3(q * 6.0 + vec3<f32>(0.0, 0.0, now * 1.7 + seed * 31.0));
            rho *= 1.0 + shimmer_amp * (n - 0.5) * 2.0;
        }
        rho = max(rho, 0.0);

        // Hottest at the stagnation point (the shock is normal there, so the air
        // is brought fully to rest) and just behind the shock front; coolest on the
        // swept flanks and in the wall boundary layer.
        // Hottest at the stagnation point and at the shock front, cooling inward
        // through the boundary layer and outward into the wake.
        let temp = temp_stag * mix(0.30, 1.0, w) * mix(0.72, 1.0, rho);
        let src = band_emission(temp);

        var col = mix(params.cool_color.rgb, params.mid_color.rgb, smoothstep(0.18, 0.55, temp));
        col = mix(col, params.hot_color.rgb, smoothstep(0.55, 0.95, temp));

        // Optical depth must be normalised by the shell's thickness WHERE THE RAY
        // IS, not by a single scalar.
        //
        // In normalized space the shell is a uniform annulus, but the metric map
        // back through `half_extents` stretches it: on a body 10x longer than it is
        // wide, the shell is 10x thicker along the long axis. Dividing by one
        // global thickness therefore over-counts tau by that ratio wherever the
        // body is wide, and the excess saturates. Measured: with the *minimum*
        // half-extent as the scale, a rocket rendered a bright teardrop streaming
        // off its nose — the layer along the long axis was ~10x too opaque — while
        // the windward belly it should have hugged stayed faint.
        //
        // The metric distance from wall to shock along the local radial direction
        // is `standoff * |normalize(q) * half_extents|`, so kappa means "optical
        // depth across the shell" uniformly (the rho profile integrates to 2/3 of
        // the span).
        let local_shell_m = envelope_thickness(cos_t) * length(normalize(q) * half_extents);
        let dtau = kappa * rho * (ds / max(local_shell_m, 1e-5));
        let a = 1.0 - exp(-dtau);
        radiance += trans * col * src * a;
        trans *= 1.0 - a;

        if (trans < TRANS_EPS) {
            break;
        }
    }

    radiance *= params.hot_color.a * ramp;

    // Premultiplied additive, matching the plume: AlphaMode::Add resolves to
    // PREMULTIPLIED_ALPHA_BLENDING, so alpha 0 preserves the background.
    return vec4<f32>(radiance, 0.0);
}
