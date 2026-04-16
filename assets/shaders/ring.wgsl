//! Saturn-style ring shader.
//!
//! Runs on a flat annulus mesh laid out in the body-local XZ plane.
//! The vertex stage passes through world positions; the fragment stage
//! recovers the normalised radial coordinate from the local mesh UV,
//! looks up the authored palette, modulates with multi-octave radial
//! noise for ringlet detail, then composites the planet shadow and
//! simple forward/back-scatter lighting.
//!
//! ## Lighting model
//!
//! Rings are dust + ice chunks, not a solid surface. The visible
//! brightness at a point is approximately the sum of:
//!
//!   - **Diffuse reflection** — standard Lambert with the ring-plane
//!     normal. Dominates on the lit side.
//!   - **Forward scatter** — Mie-like glow when the viewer looks
//!     roughly toward the sun through the ring dust. Dominates on the
//!     unlit side.
//!
//! We approximate both with `clamp(n·l, 0, 1)` plus a Henyey-Greenstein
//! forward-scatter lobe controlled by the sun-view angle. The forward
//! lobe is the reason Saturn's unlit side lights up at high phase.
//!
//! ## Planet shadow
//!
//! Every fragment ray-casts a shadow ray from its world position
//! toward the sun (`params.light_dir`) and tests it against the
//! planet sphere (`params.planet_center_radius`). If the ray hits the
//! sphere before "escaping", the fragment is in shadow — we scale the
//! direct term to the ambient floor, preserving only forward-scatter
//! (which doesn't require direct sunlight on the ring particle).

#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::mesh_functions::get_world_from_local
#import thalos::lighting::{SceneLighting, eclipse_factor}

const MAX_RING_STOPS: u32 = 8u;
const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;

struct RingParams {
    // xyz = planet center in world space; w = planet render radius.
    planet_center_radius: vec4<f32>,
    inner_radius:         f32,
    outer_radius:         f32,
    _pad0:                f32,
    _pad1:                f32,
    // Stars, ambient, eclipse occluders (shared scene lighting).
    scene:                SceneLighting,
}

struct RingLayers {
    palette_color:    array<vec4<f32>, 8>,
    palette_opacity:  array<vec4<f32>, 8>,
    stop_count:       u32,
    opacity:          f32,
    ringlet_noise:    f32,
    ringlet_octaves:  u32,
    seed_lo:          u32,
    seed_hi:          u32,
    _pad0:            u32,
    _pad1:            u32,
}

@group(3) @binding(0) var<uniform> params: RingParams;
@group(3) @binding(1) var<uniform> layers: RingLayers;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) uv:       vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal:   vec3<f32>,
    // x = radial u in [0, 1] (0 = inner rim, 1 = outer rim),
    // y = angular position in [0, 1].
    @location(2) ring_uv:        vec2<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    // Manual model-matrix transform. `gas_giant.wgsl` uses this same
    // pattern rather than the `mesh_functions` helpers so the shader
    // stays free of binding-layout assumptions that change between
    // Bevy versions.
    let model = get_world_from_local(in.instance_index);
    let world_pos = model * vec4(in.position, 1.0);
    // Ring mesh has no non-uniform scale, so the direct 3x3 is a
    // safe normal transform (skips the inverse-transpose cost).
    let world_normal = normalize((model * vec4(in.normal, 0.0)).xyz);

    var out: VertexOutput;
    out.world_position = world_pos.xyz;
    out.world_normal = world_normal;
    out.ring_uv = in.uv;
    out.clip_position = view.clip_from_world * world_pos;
    return out;
}

// ── Hash / noise helpers ────────────────────────────────────────────

fn pcg(x: u32) -> u32 {
    let state = x * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash_f32(x: u32) -> f32 {
    return f32(pcg(x)) / 4294967295.0;
}

fn value_noise_1d(x: f32, seed: u32) -> f32 {
    let xf = floor(x);
    let xu = u32(xf);
    let f = x - xf;
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash_f32(pcg(xu ^ seed));
    let b = hash_f32(pcg((xu + 1u) ^ seed));
    return mix(a, b, u) * 2.0 - 1.0;
}

fn fbm_1d(x: f32, seed: u32, octaves: u32, pixel_width: f32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        if i >= octaves { break; }
        // Stop before this octave aliases: if one pixel spans more
        // than half a noise cell, the octave is above Nyquist.
        if freq * pixel_width > 0.5 { break; }
        sum = sum + amp * value_noise_1d(x * freq, seed + i * 1013u);
        freq = freq * 2.0;
        amp = amp * 0.5;
    }
    return sum;
}

// ── Palette lookup ──────────────────────────────────────────────────

struct RingSample {
    color:   vec3<f32>,
    opacity: f32,
}

fn palette_lookup(u: f32) -> RingSample {
    let count = layers.stop_count;
    if count == 0u {
        return RingSample(vec3(0.0), 0.0);
    }
    if count == 1u {
        return RingSample(layers.palette_color[0].xyz, layers.palette_opacity[0].x);
    }

    let last = count - 1u;
    if u <= layers.palette_color[0].w {
        return RingSample(layers.palette_color[0].xyz, layers.palette_opacity[0].x);
    }
    if u >= layers.palette_color[last].w {
        return RingSample(layers.palette_color[last].xyz, layers.palette_opacity[last].x);
    }

    for (var i = 0u; i < last; i = i + 1u) {
        let a = layers.palette_color[i];
        let b = layers.palette_color[i + 1u];
        if u >= a.w && u <= b.w {
            let span = max(b.w - a.w, 1e-6);
            let t = clamp((u - a.w) / span, 0.0, 1.0);
            let ts = t * t * (3.0 - 2.0 * t);
            let oa = layers.palette_opacity[i].x;
            let ob = layers.palette_opacity[i + 1u].x;
            return RingSample(mix(a.xyz, b.xyz, ts), mix(oa, ob, ts));
        }
    }
    return RingSample(layers.palette_color[last].xyz, layers.palette_opacity[last].x);
}

// ── Planet shadow ───────────────────────────────────────────────────

// Returns 1.0 if the point is directly lit, 0.0 if fully shadowed.
// The shadow geometry is a single ray-sphere intersection along the
// sun direction — the rings are thin enough that the "height" of a
// ring fragment above the ring plane is negligible.
fn planet_shadow_attenuation(world_pos: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let center = params.planet_center_radius.xyz;
    let radius = params.planet_center_radius.w;
    if radius <= 0.0 {
        return 1.0;
    }
    let oc = center - world_pos;
    // Shadow ray: p(t) = world_pos + t * light_dir, t >= 0 points
    // toward the sun. For the sun to occlude the ring point, the
    // planet must lie between the point and the sun — i.e. the
    // closest-approach parameter `t_c = dot(oc, light_dir)` is
    // positive and the perpendicular distance is below the planet
    // radius.
    let t_c = dot(oc, light_dir);
    if t_c <= 0.0 {
        return 1.0;
    }
    let perp_sq = dot(oc, oc) - t_c * t_c;
    let r_sq = radius * radius;
    if perp_sq >= r_sq {
        return 1.0;
    }
    // Soften the edge slightly so the shadow terminator isn't a
    // single-pixel step. 0.02 = ~2% of the planet radius feather.
    let d = sqrt(max(r_sq - perp_sq, 0.0));
    let feather = radius * 0.02;
    return 1.0 - smoothstep(-feather, feather, d);
}

// ── Fragment ────────────────────────────────────────────────────────

struct FragOutput {
    @location(0) color: vec4<f32>,
}

@fragment
fn fragment(in: VertexOutput) -> FragOutput {
    // Angular position is irrelevant for the radial palette — ring
    // geometry is rotationally symmetric. All variation comes from u
    // and the shared noise seed.
    let u = clamp(in.ring_uv.x, 0.0, 1.0);
    let seed = layers.seed_lo ^ 0xA55A5AA5u;

    // ── Palette lookup ──────────────────────────────────────────
    let sample = palette_lookup(u);
    var color = sample.color;
    var opacity = sample.opacity * layers.opacity;

    // ── Ringlet noise — bright/dark radial striations ───────────
    //
    // Expensive multi-octave FBM in 1D is cheap at fragment rate,
    // and it's what makes the ring read as "thousands of ringlets"
    // rather than "eight palette stops." Amplitude is split
    // between a luminance modulation and an opacity modulation so
    // dark rings can actually show gaps instead of just darker
    // bands.
    //
    // The noise input is `u * 140.0`, so one noise cell = 1/140
    // of the ring width. Screen-space derivatives tell us how many
    // noise cells one pixel spans; octaves whose frequency would
    // exceed the Nyquist limit are culled to prevent aliasing
    // (the "flashing dots" artifact).
    if layers.ringlet_noise > 0.0 && layers.ringlet_octaves > 0u {
        let noise_coord = u * 140.0;
        let pixel_width = max(abs(dpdx(noise_coord)), abs(dpdy(noise_coord)));
        let n = fbm_1d(noise_coord, seed, layers.ringlet_octaves, pixel_width);
        let amp = layers.ringlet_noise;
        color = color * clamp(1.0 + n * amp * 0.8, 0.0, 2.0);
        opacity = clamp(opacity * (1.0 + n * amp * 0.6), 0.0, 1.0);
    }

    // ── Geometry ────────────────────────────────────────────────
    let normal = normalize(in.world_normal);
    let to_cam = normalize(view.world_position - in.world_position);
    // Primary star — see the gas giant shader for the multi-star note.
    let primary_star = params.scene.stars[0];
    let light_dir = normalize(primary_star.dir_flux.xyz);
    let star_flux = primary_star.dir_flux.w;

    // The ring faces both ways — pick whichever normal is toward the
    // viewer so Lambert/forward-scatter both make sense.
    let facing = sign(dot(normal, to_cam));
    let nrm = normal * facing;

    // ── Lighting: diffuse + forward-scatter ─────────────────────
    let n_dot_l = dot(nrm, light_dir);
    let lit_side = clamp(n_dot_l, 0.0, 1.0);

    // Henyey-Greenstein forward lobe for unlit-side glow. `g=0.6`
    // gives a broad forward peak without collapsing to a point —
    // rings read bright for a few tens of degrees around the
    // anti-sun line.
    let g = 0.6;
    let cos_theta = dot(-light_dir, to_cam);
    let hg_denom = pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5);
    let forward_scatter = (1.0 - g * g) / (4.0 * PI * max(hg_denom, 1e-4));

    // ── Planet shadow + cross-body eclipse ─────────────────────
    let shadow = planet_shadow_attenuation(in.world_position, light_dir);
    let eclipse = eclipse_factor(params.scene, in.world_position, light_dir);

    // Combine lighting terms. Forward scatter doesn't care about
    // which face you're looking at, but it is still extinguished by
    // planet shadow — a ring particle behind the planet isn't
    // picking up sunlight to forward-scatter.
    let lambert = lit_side * (1.0 / PI);
    let direct = (lambert + forward_scatter * 0.8) * shadow * eclipse;
    let lit = color * star_flux * direct
            + color * params.scene.ambient_intensity;

    // Opacity drops toward the edges of each ringlet so the feathered
    // border reads as dust, not a cookie-cutter stamp.
    if opacity <= 0.004 {
        discard;
    }

    var out: FragOutput;
    out.color = vec4<f32>(lit, opacity);
    return out;
}
