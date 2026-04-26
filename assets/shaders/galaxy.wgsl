//! Forward-rendered galaxy shader.
//!
//! Two rendering paths, selected per-galaxy from the Sérsic index:
//!
//! * `n ≥ 2` — **elliptical**. Single Sérsic profile, smooth falloff,
//!   warm old-population colour.
//! * `n < 2` — **spiral**. Composited from a central bulge (Sérsic
//!   n = 4), an exponential disk (n = 1), and log-spiral arms with
//!   FBM dust-lane modulation. Two-tone colour (warm bulge, blue disk).
//!
//! The apparent magnitude flux is spread across the on-screen pixel
//! area via `surface_flux` so a big spiral and a small distant
//! elliptical of the same magnitude add the same total light.

#import bevy_pbr::mesh_view_bindings::view

struct GalaxyParams {
    pixel_radius_scale: f32,
    min_pixel_radius:   f32,
    brightness:         f32,
    _pad0:              f32,
}

@group(3) @binding(0) var<uniform> params: GalaxyParams;

struct VertexInput {
    @location(0) direction: vec3<f32>,
    @location(1) corner:    vec2<f32>,
    @location(2) shape:     vec3<f32>,
    @location(3) orient:    vec4<f32>,
    @location(4) color:     vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) shape_uv: vec2<f32>,
    @location(1) color:    vec4<f32>,
    @location(2) sersic_n: f32,
    @location(3) surface_flux: f32,
    @location(4) axis_ratio: f32,
}

const INFINITY_DISTANCE: f32 = 1.0e10;
const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;
// Screen quad extends past the half-light radius so the outer halo
// and arm sweep have room. The shader's `evaluate_spiral` discards
// at `r > 1.45`, so 1.5 gives a small safety margin.
const QUAD_EXTENT: f32 = 1.5;

// Cheap blackbody approximation good over ~3000–10000 K. Not
// photometrically accurate but matches the warm-to-cool stellar
// gradient real spiral galaxies show from bulge to arms.
fn blackbody_color(temperature_k: f32) -> vec3<f32> {
    let t01 = clamp((temperature_k - 3000.0) / 7000.0, 0.0, 1.0);
    return vec3<f32>(
        1.0,
        0.55 + 0.45 * t01,
        0.25 + 0.75 * smoothstep(0.0, 1.0, t01),
    );
}

// ── Hash / noise (u32-only maths for portability) ──────────────────

fn pcg(x: u32) -> u32 {
    let state = x * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash_f32(x: u32) -> f32 {
    return f32(pcg(x)) / 4294967295.0;
}

fn hash2d(ix: i32, iy: i32, seed: u32) -> f32 {
    // Two arbitrary large primes. u32 multiplication wraps by spec,
    // so this is well-defined on every backend.
    let ux = u32(ix) * 73856093u;
    let uy = u32(iy) * 19349663u;
    return hash_f32(pcg(ux ^ uy ^ seed));
}

fn value_noise(p: vec2<f32>, seed: u32) -> f32 {
    let ix = i32(floor(p.x));
    let iy = i32(floor(p.y));
    let f = p - vec2<f32>(f32(ix), f32(iy));
    let a = hash2d(ix,     iy,     seed);
    let b = hash2d(ix + 1, iy,     seed);
    let c = hash2d(ix,     iy + 1, seed);
    let d = hash2d(ix + 1, iy + 1, seed);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>, seed: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0u; i < 5u; i = i + 1u) {
        sum = sum + amp * value_noise(q, seed + i * 1013u);
        q = q * 2.03;
        amp = amp * 0.5;
    }
    return sum;
}

// ── Vertex ─────────────────────────────────────────────────────────

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let world_pos = view.world_position + in.direction * INFINITY_DISTANCE;
    let center_clip = view.clip_from_world * vec4<f32>(world_pos, 1.0);

    let angular_radius = max(in.shape.x, 1.0e-7);
    let radius_px = max(
        angular_radius * params.pixel_radius_scale,
        params.min_pixel_radius,
    );

    let cos_pa = in.orient.y;
    let sin_pa = in.orient.z;
    let axis_ratio = in.orient.x;

    // Pre-expand the corner so the quad covers the whole galaxy
    // including halo and warped arms. Keeps `shape_uv` values up to
    // ±QUAD_EXTENT along the major axis.
    let corner_ext = in.corner * QUAD_EXTENT;
    let corner_px = vec2<f32>(
        corner_ext.x * cos_pa - corner_ext.y * sin_pa,
        corner_ext.x * sin_pa + corner_ext.y * cos_pa,
    );

    let viewport = view.viewport.zw;
    let ndc_per_pixel = vec2<f32>(2.0 / viewport.x, 2.0 / viewport.y);
    let offset_clip = corner_px * radius_px * ndc_per_pixel * center_clip.w;

    // NDC z = 0 → reverse-Z far plane (see stars.wgsl for rationale).
    // Depth compare is `GreaterEqual` via `GalaxyMaterial::specialize`.
    var out: VertexOutput;
    out.clip_position = vec4<f32>(
        center_clip.xy + offset_clip,
        0.0,
        center_clip.w,
    );
    out.shape_uv = vec2<f32>(corner_ext.x, corner_ext.y / max(axis_ratio, 0.15));
    out.color = in.color;
    out.sersic_n = in.shape.y;
    out.axis_ratio = axis_ratio;
    // Effective flux-integral area. Real Sérsic profiles concentrate
    // most of their light in the core, so dividing the magnitude flux
    // by the full quad area leaves the peak too dim. We divide by a
    // much smaller characteristic area so the bulge can spike above
    // the tonemap threshold without blowing out the whole disk.
    let char_area = max(PI * radius_px * radius_px * axis_ratio * 0.08, 1.0);
    out.surface_flux = in.color.a / char_area;
    return out;
}

// ── Small helpers ──────────────────────────────────────────────────

// Unresolved-star overlay. Hashes each cell, accepts the top fraction,
// then `pow(mag, 12)` concentrates flux into a few very-bright stars
// that survive the bloom threshold. Samples three octaves so both
// dense fields and individual pop-out stars appear.
fn star_field(uv: vec2<f32>, scale: f32, density: f32, seed: u32) -> f32 {
    let p = uv * scale;
    let ix = i32(floor(p.x));
    let iy = i32(floor(p.y));
    let f = p - vec2<f32>(f32(ix), f32(iy));
    let h = hash2d(ix, iy, seed);
    if h < 1.0 - density {
        return 0.0;
    }
    let jx = hash2d(ix, iy, seed + 17u);
    let jy = hash2d(ix, iy, seed + 31u);
    let d = length(f - vec2<f32>(jx, jy));
    let spark = pow(max(0.0, 1.0 - d * 4.0), 8.0);
    let mag = hash2d(ix, iy, seed + 53u);
    return spark * pow(mag, 12.0);
}

// ── Profiles ───────────────────────────────────────────────────────

// Sérsic with peak normalised to 1 at r = 0.
fn sersic(r: f32, n: f32) -> f32 {
    let b_n = 2.0 * n - 0.33;
    return exp(-b_n * (pow(max(r, 1.0e-6), 1.0 / n) - 1.0) - b_n);
}

// Smooth soft-edge helper (accepts `hi > lo` even though the usual
// use is "fall off from inside toward outside").
fn soft_edge(inner: f32, outer: f32, r: f32) -> f32 {
    return 1.0 - smoothstep(inner, outer, r);
}

fn evaluate_elliptical(shape_uv: vec2<f32>, base_color: vec3<f32>, n: f32) -> vec3<f32> {
    let r = length(shape_uv);
    if r > 1.05 {
        return vec3<f32>(0.0);
    }
    let profile = sersic(r, n);
    let edge = soft_edge(0.80, 1.05, r);
    let tint = vec3<f32>(1.05, 0.95, 0.80);
    return base_color * tint * (profile * edge);
}

fn evaluate_spiral(
    shape_uv: vec2<f32>,
    base_color: vec3<f32>,
    axis_ratio: f32,
) -> vec3<f32> {
    let r = length(shape_uv);
    if r > 1.45 {
        return vec3<f32>(0.0);
    }

    // Central bulge — Sérsic n=2, wide soft nucleus.
    let bulge_scale = 0.35;
    let bulge_flat = sersic(r / bulge_scale, 2.0);
    // Chord-length trick: treat the bulge as a 3D sphere seen through
    // itself. The line-of-sight chord through a unit sphere at
    // projected radius r is `sqrt(1 - r²)`; integrated emission is
    // proportional to chord². Multiplying the 2D profile by chord²
    // converts a flat hot disc into a volumetric glow.
    let chord = sqrt(max(0.0, 1.0 - min(r, 1.0) * min(r, 1.0)));
    let bulge = bulge_flat * (0.4 + 0.6 * chord * chord);

    // Disk — exponential, concentrated so mid-disk drops off before
    // the arm structure runs out.
    let disk = sersic(r, 1.4);

    // Extended warm halo past r_eff. Modulated by a large-scale
    // FBM so it doesn't read as a perfect elliptical disc, and
    // falls off fast so it's barely perceptible past r_eff — the
    // arm/disk exponential already carries the outer material.
    let halo_fbm = fbm(shape_uv * 1.3, 4141u);
    let halo = exp(-2.6 * r) * (0.55 + 0.9 * halo_fbm);

    // Warp the polar angle so arms aren't perfect log spirals.
    let theta_raw = atan2(shape_uv.y, shape_uv.x);
    let warp = fbm(vec2<f32>(r * 3.2, theta_raw * 0.8), 2222u) - 0.5;
    let theta = theta_raw + warp * 1.1;

    // Density-wave arms: find the angular distance to the nearest
    // arm ridge and feed it through a Gaussian. The ridge angular
    // position spirals through `pitch * log(r)` so inner material
    // leads outer material (differential rotation). Gaussian width
    // widens toward the centre — same trick itinerantgames uses to
    // flare arms near the bulge.
    let num_arms = 3.0;
    let pitch = 4.4;
    let arm_period = TAU / num_arms;
    let arm_phase = theta * num_arms - pitch * log(max(r, 0.04));
    // `arm_phase` increases monotonically; reducing mod 2π and then
    // centring gives a signed distance to the nearest ridge.
    let phase_wrapped = arm_phase - floor(arm_phase / TAU) * TAU - PI;
    let arm_width = 0.55 + 0.35 / max(r + 0.15, 0.2);
    let arm_gauss = exp(-(phase_wrapped * phase_wrapped) / (arm_width * arm_width));
    // Secondary feathery arms at double frequency for detail.
    let arm_phase2 = theta * (num_arms * 2.0)
        - pitch * 1.05 * log(max(r, 0.04)) + 1.7;
    let phase2_wrapped = arm_phase2 - floor(arm_phase2 / TAU) * TAU - PI;
    let arm2 = 0.25 * exp(-(phase2_wrapped * phase2_wrapped) / 0.35);
    let arm_structure = arm_gauss + arm2;

    // Clumpiness noise along the arm direction. 1D FBM of the arm
    // phase keeps clumps streaming along arm ridges.
    let clumps = fbm(
        vec2<f32>(arm_phase * 0.45, r * 4.0),
        7777u,
    );
    let arms_modulated = arm_structure * (0.40 + 1.25 * clumps);

    // Arm envelope.
    let arm_radial = smoothstep(0.08, 0.28, r) * soft_edge(0.50, 1.40, r);
    let arms = arms_modulated * arm_radial;

    // Dust lane extinction with wavelength-dependent optical depth.
    // Red passes, green attenuates moderately, blue absorbs most —
    // classic interstellar reddening.
    let dust_fbm = fbm(shape_uv * 4.2, 9999u);
    let dust_mask = smoothstep(0.12, 0.40, r) * soft_edge(0.45, 1.30, r);
    var tau = 1.2 * smoothstep(0.35, 0.68, dust_fbm) * dust_mask;
    // Mid-plane dust lane for inclined galaxies. In face-on view the
    // dust is distributed over the whole disk; in edge-on view all
    // the dust piles along the mid-plane (shape_uv.y ≈ 0 here after
    // the projection-undo). Inclination factor `(1 - axis_ratio)`
    // turns this on smoothly for highly inclined spirals.
    let inclination = 1.0 - axis_ratio;
    let mid_plane_width = 0.12 + 0.18 * axis_ratio;
    let mid_plane = exp(-(shape_uv.y * shape_uv.y) / (mid_plane_width * mid_plane_width));
    let mid_plane_mask = smoothstep(0.05, 0.25, r) * soft_edge(0.40, 1.30, r);
    tau = tau + 1.8 * inclination * mid_plane * mid_plane_mask;
    let extinction = exp(-tau * vec3<f32>(0.55, 1.05, 1.80));

    // Unresolved star overlay (HII regions + bright supergiants).
    // Three scales: a dense sprinkle of tiny background sparks, a
    // medium layer for star clusters, and a sparse layer of very
    // bright HII blobs. Multiplied by the disk density so they only
    // appear where the galaxy has material.
    let stars_a = star_field(shape_uv, 120.0, 0.06, 11u);
    let stars_b = star_field(shape_uv, 40.0, 0.04, 29u);
    let stars_c = star_field(shape_uv, 12.0, 0.10, 53u);
    let star_density = (disk * 0.8 + arms * 1.6) * arm_radial;
    let stars_value = (stars_a + 0.8 * stars_b + 1.2 * stars_c) * star_density;

    // Radius-driven stellar temperature → blackbody colour. Inner
    // region is old-pop warm (~4000 K), outer disk/arms are young
    // and blue (~7000 K). Arm ridges get a further blue nudge.
    let temp_k = mix(3900.0, 6800.0, smoothstep(0.0, 0.9, r)) + arms * 1200.0;
    let stellar_rgb = blackbody_color(temp_k);

    // Composition.
    let bulge_rgb = stellar_rgb * (bulge * 5.2);
    let disk_rgb = stellar_rgb * (disk * (0.30 + arms * 1.5));
    let star_rgb = stellar_rgb * (stars_value * 3.0);
    let halo_rgb = stellar_rgb * (halo * 0.20);

    // Soft outer envelope. Wide transition so no sharp disc boundary.
    let edge = soft_edge(0.75, 1.45, r);

    // Apply extinction to disk + stars (dust sits in the thin disk),
    // not the bulge or halo (they're in front of / above the plane).
    let attenuated = (disk_rgb + star_rgb) * extinction;
    return (bulge_rgb + attenuated + halo_rgb) * edge;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    var rgb: vec3<f32>;
    if in.sersic_n >= 2.0 {
        rgb = evaluate_elliptical(in.shape_uv, in.color.rgb, in.sersic_n);
    } else {
        rgb = evaluate_spiral(in.shape_uv, in.color.rgb, in.axis_ratio);
    }
    let out_rgb = rgb * in.surface_flux * params.brightness;
    return vec4<f32>(out_rgb, 0.0);
}
