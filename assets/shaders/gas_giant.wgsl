//! Gas / ice giant fragment shader.
//!
//! Renders a gas giant as an impostor billboard with a ray-sphere
//! intersection on the cloud-deck sphere, then composites several
//! atmosphere layers on top. There is NO baked cubemap — the entire
//! visible disk is synthesised per fragment from the uniform blocks in
//! `thalos_planet_rendering::gas_giant`.
//!
//! ## Layer pipeline
//!
//! 1. **Ray-sphere** hit test against the cloud deck at `params.radius`.
//!    - Hit: evaluate the cloud deck at the surface point and light it.
//!    - Miss: check whether the ray grazes the rim halo shell; if so,
//!      emit the halo contribution and write a far depth so the rest of
//!      the scene composites correctly.
//!
//! 2. **Cloud deck** (`cloud_deck_color`): latitude palette lookup +
//!    zonal band structure + fine turbulence. Applies the body-local
//!    orientation quaternion so bands rotate with the planet.
//!
//! 3. **Haze layer** (`apply_haze`): optional multiplicative tint with a
//!    view-angle-dependent bias that strengthens near the terminator.
//!
//! 4. **Rim halo** (`rim_halo_contribution`): exponential-density ring
//!    just outside the cloud deck. Contributes on both hit and miss
//!    paths so the glow wraps the silhouette.
//!
//! Later layers (storms, auroras) plug in between steps 2 and 3; the
//! uniform layout already has space reserved for their data.

#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::mesh_functions::get_world_from_local

// ── Uniforms ────────────────────────────────────────────────────────────────

const MAX_PALETTE_STOPS: u32 = 10u;
const PROFILE_N: u32 = 16u;
const MAX_VORTICES: u32 = 16u;
const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;

// Matches `GasGiantParams` in `gas_giant.rs`. Field order is load-bearing.
// `light_dir.w` carries `elapsed_seconds` — raw sim time wrapped to a
// day so f32 precision stays tight. Used by differential rotation, edge
// wave phase, and edge vortex chain epoch hashing.
struct GasGiantParams {
    radius:            f32,
    light_intensity:   f32,
    ambient_intensity: f32,
    rotation_phase:    f32,
    light_dir:         vec4<f32>,
    orientation:       vec4<f32>,
}

// Matches `GasGiantLayers` in `gas_giant.rs`. Field order is load-bearing.
struct GasGiantLayers {
    palette:              array<vec4<f32>, 10>,
    stop_count:           u32,
    band_frequency:       f32,
    band_warp:            f32,
    turbulence:           f32,
    // xyz = cloud tint multiplier, w = band_contrast luminance swing.
    tint:                 vec4<f32>,
    // x = band_sharpness exponent.
    band_shape_params:    vec4<f32>,
    haze_tint_thickness:  vec4<f32>,
    haze_params:          vec4<f32>,
    rim_color_intensity:  vec4<f32>,
    rim_shape:            vec4<f32>,
    speed_profile:        array<vec4<f32>, 4>,
    turbulence_profile:   array<vec4<f32>, 4>,
    // x = differential_rotation_rate, y = edge_wave_amp,
    // z = curl_amp, w = parallax_amp.
    dynamics:             vec4<f32>,
    // x = vortex_count, y = edge_chain_slots, z = edge_chain_enabled,
    // w = speed_profile_valid.
    counts:               vec4<u32>,
    // x = base_radius, y = strength, z = lifetime_s.
    edge_chain:           vec4<f32>,
    terminator_warmth:    vec4<f32>,
    fresnel_rim:          vec4<f32>,
    // xyz = Rayleigh blue-gap colour, w = strength.
    rayleigh_color:       vec4<f32>,
    // x = haze_scale, y = clearing_threshold, z = latitude_bias.
    rayleigh_params:      vec4<f32>,
    // xyz = per-channel Minnaert exponents (r, g, b), w = strength.
    limb_exponents:       vec4<f32>,
    // x = inner radius (render units), y = outer radius (render units),
    // z = ringlet noise amp, w = enabled flag.
    ring_shadow:          vec4<f32>,
    vortex_pos:           array<vec4<f32>, 16>,
    vortex_tint:          array<vec4<f32>, 16>,
    seed_lo:              u32,
    seed_hi:              u32,
    _pad0:                u32,
    _pad1:                u32,
}

@group(3) @binding(0) var<uniform> params: GasGiantParams;
@group(3) @binding(1) var<uniform> layers: GasGiantLayers;

// ── Vertex stage ────────────────────────────────────────────────────────────
//
// Uses the same camera-aligned billboard trick as `planet_impostor.wgsl`:
// expand a unit quad so it fully encloses the sphere from the camera's
// point of view, then let the fragment shader do the ray/sphere work.

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) sphere_center: vec3<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let model = get_world_from_local(in.instance_index);
    let sphere_center = (model * vec4(0.0, 0.0, 0.0, 1.0)).xyz;

    let cam_pos = view.world_position;
    let to_cam  = normalize(cam_pos - sphere_center);

    let ref_up = select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), abs(to_cam.y) > 0.99);
    let right  = normalize(cross(ref_up, to_cam));
    let up     = normalize(cross(to_cam, right));

    // Grow the billboard slightly beyond the cloud-deck silhouette so
    // the rim halo shell has pixels to draw into. The halo is clamped
    // by `rim_shape.y` but we budget 1.15× unconditionally — cheap in
    // geometry and keeps the shell from being scissored off.
    let effective_radius = params.radius * 1.15;
    let d = length(cam_pos - sphere_center);
    let d_safe = max(d, effective_radius * 1.0001);
    let billboard_radius = effective_radius * d_safe
        / sqrt(d_safe * d_safe - effective_radius * effective_radius);

    let world_pos = sphere_center
        + in.position.x * right * billboard_radius
        + in.position.y * up    * billboard_radius;

    var out: VertexOutput;
    out.clip_position  = view.clip_from_world * vec4(world_pos, 1.0);
    out.world_position = world_pos;
    out.sphere_center  = sphere_center;
    return out;
}

struct FragOutput {
    @location(0)         color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn rotate_quat(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u
         + (s * s - dot(u, u)) * v
         + 2.0 * s * cross(u, v);
}

// Hash helpers — same shape as the impostor shader's PCG so both layers
// produce compatible noise if we ever want to share a feature buffer.
fn pcg(x: u32) -> u32 {
    let state = x * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash_f32(x: u32) -> f32 {
    return f32(pcg(x)) / 4294967295.0;
}

fn hash_cell(ix: f32, iy: f32, k: u32) -> f32 {
    // Bitcast the lattice coordinates so negative floats produce
    // deterministic integer values. WGSL's `u32(f)` is undefined for
    // negative inputs, and `floor(p)` can easily be negative on a
    // full-sphere sample field — bitcasting sidesteps the whole mess.
    let xu = bitcast<u32>(ix);
    let yu = bitcast<u32>(iy);
    return hash_f32(xu ^ (yu * 374761393u) ^ k);
}

fn value_noise_2d(p: vec2<f32>, seed: u32) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash_cell(i.x,       i.y,       seed);
    let b = hash_cell(i.x + 1.0, i.y,       seed);
    let c = hash_cell(i.x,       i.y + 1.0, seed);
    let d = hash_cell(i.x + 1.0, i.y + 1.0, seed);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 2.0 - 1.0;
}

fn fbm_2d(p: vec2<f32>, seed: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0u; i < 4u; i = i + 1u) {
        sum = sum + amp * value_noise_2d(q, seed + i * 1013u);
        q = q * 2.0;
        amp = amp * 0.5;
    }
    return sum;
}

// ── 3D value noise ──────────────────────────────────────────────────────────
//
// Used everywhere the cloud deck pipeline touches the sphere surface.
// Sampling on (x, y, z) avoids the degenerate (x, z)-only projection
// that made the previous pipeline kill all warp near the poles: the
// lattice is finite at `y = ±1`, so turbulence can go fully chaotic at
// high latitudes without streaking toward the pole.

fn hash_cell_3d(ix: f32, iy: f32, iz: f32, k: u32) -> f32 {
    let xu = bitcast<u32>(ix);
    let yu = bitcast<u32>(iy);
    let zu = bitcast<u32>(iz);
    return hash_f32(xu ^ (yu * 374761393u) ^ (zu * 2246822519u) ^ k);
}

fn value_noise_3d(p: vec3<f32>, seed: u32) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let c000 = hash_cell_3d(i.x,       i.y,       i.z,       seed);
    let c100 = hash_cell_3d(i.x + 1.0, i.y,       i.z,       seed);
    let c010 = hash_cell_3d(i.x,       i.y + 1.0, i.z,       seed);
    let c110 = hash_cell_3d(i.x + 1.0, i.y + 1.0, i.z,       seed);
    let c001 = hash_cell_3d(i.x,       i.y,       i.z + 1.0, seed);
    let c101 = hash_cell_3d(i.x + 1.0, i.y,       i.z + 1.0, seed);
    let c011 = hash_cell_3d(i.x,       i.y + 1.0, i.z + 1.0, seed);
    let c111 = hash_cell_3d(i.x + 1.0, i.y + 1.0, i.z + 1.0, seed);
    let x00 = mix(c000, c100, u.x);
    let x10 = mix(c010, c110, u.x);
    let x01 = mix(c001, c101, u.x);
    let x11 = mix(c011, c111, u.x);
    let y0 = mix(x00, x10, u.y);
    let y1 = mix(x01, x11, u.y);
    return mix(y0, y1, u.z) * 2.0 - 1.0;
}

fn fbm_3d(p: vec3<f32>, seed: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0u; i < 5u; i = i + 1u) {
        sum = sum + amp * value_noise_3d(q, seed + i * 1013u);
        q = q * 2.0;
        amp = amp * 0.5;
    }
    return sum;
}

// Vector fBm: three decorrelated scalar fBm fields packed as a vec3.
// Used as the `q` and `r` fields in Inigo Quilez's domain-warp recipe.
fn fbm_3d_vec3(p: vec3<f32>, seed: u32) -> vec3<f32> {
    return vec3<f32>(
        fbm_3d(p,                                 seed),
        fbm_3d(p + vec3<f32>(5.2, 1.3, 2.7),      seed ^ 0xA3u),
        fbm_3d(p + vec3<f32>(1.7, 9.2, 4.1),      seed ^ 0x5Bu),
    );
}

// ── Latitude profile lookups ────────────────────────────────────────────────
//
// Both profiles are stored as 16 signed scalars packed four per vec4.
// `sample_speed` falls back to zero when the author didn't provide a
// profile; `sample_turb` falls back to 1.0 (fully uniform turbulence,
// the scalar `turbulence` then acts as the one and only amplitude).

fn profile_fetch(profile: array<vec4<f32>, 4>, lat: f32) -> f32 {
    let x = clamp((lat * 0.5 + 0.5) * 15.0, 0.0, 15.0);
    let i0 = u32(floor(x));
    let i1 = min(i0 + 1u, 15u);
    let f = x - floor(x);
    let v0 = profile[i0 / 4u][i0 % 4u];
    let v1 = profile[i1 / 4u][i1 % 4u];
    return mix(v0, v1, f);
}

fn sample_speed(lat: f32) -> f32 {
    if layers.counts.w == 0u {
        return 0.0;
    }
    return profile_fetch(layers.speed_profile, lat);
}

fn sample_turb(lat: f32) -> f32 {
    return profile_fetch(layers.turbulence_profile, lat);
}

// Rodrigues' rotation formula.
fn rotate_axis(v: vec3<f32>, axis: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

// Convert `(lat, lon)` in the [-1, 1] × [-π, π] body-local frame to a
// unit sphere normal.
fn latlon_to_normal(lat: f32, lon: f32) -> vec3<f32> {
    let cl = sqrt(max(1.0 - lat * lat, 0.0));
    return vec3<f32>(cl * cos(lon), lat, cl * sin(lon));
}

// Curl-like divergence-free warp built from three finite differences of
// a scalar noise field. Single-octave to keep the cost sane — the
// downstream cloud deck already stacks multiple fbm samples.
fn curl_warp(p: vec3<f32>, seed: u32) -> vec3<f32> {
    let eps = 0.15;
    let a1 = value_noise_3d(p + vec3<f32>(0.0, eps, 0.0), seed);
    let a2 = value_noise_3d(p - vec3<f32>(0.0, eps, 0.0), seed);
    let b1 = value_noise_3d(p + vec3<f32>(0.0, 0.0, eps), seed);
    let b2 = value_noise_3d(p - vec3<f32>(0.0, 0.0, eps), seed);
    let c1 = value_noise_3d(p + vec3<f32>(eps, 0.0, 0.0), seed);
    let c2 = value_noise_3d(p - vec3<f32>(eps, 0.0, 0.0), seed);
    let da = (a1 - a2) * 0.5;
    let db = (b1 - b2) * 0.5;
    let dc = (c1 - c2) * 0.5;
    return vec3<f32>(db - dc, dc - da, da - db) / eps;
}

// ── Layer 2: cloud deck ─────────────────────────────────────────────────────
//
// Evaluates the cloud-deck colour at a point on the sphere, in body-local
// coordinates. The palette lookup is signed-latitude keyed; band
// structure is introduced by warping the sample latitude with a noise
// field before the palette fetch, which makes the lerp between stops
// run along a non-straight contour and read as zonal banding.

fn palette_lookup(lat: f32) -> vec3<f32> {
    let count = layers.stop_count;
    if count == 0u {
        return vec3<f32>(0.5);
    }
    // Single-stop degenerate case.
    if count == 1u {
        return layers.palette[0].xyz;
    }

    // Below first stop / above last stop → clamp.
    if lat <= layers.palette[0].w {
        return layers.palette[0].xyz;
    }
    let last = count - 1u;
    if lat >= layers.palette[last].w {
        return layers.palette[last].xyz;
    }

    // Locate the pair we lerp between.
    for (var i = 0u; i < last; i = i + 1u) {
        let a = layers.palette[i];
        let b = layers.palette[i + 1u];
        if lat >= a.w && lat <= b.w {
            let span = max(b.w - a.w, 1e-6);
            let t = clamp((lat - a.w) / span, 0.0, 1.0);
            // Smoothstep across the palette span. Hue is intentionally
            // *smooth* between palette stops — the visible band edges
            // come from a per-band luminance step layered on top in
            // `cloud_deck_color`, not from sharpening this lookup. (If
            // you sharpen here, the disc collapses into one flat zone
            // per palette stop instead of showing every band.)
            let ts = t * t * (3.0 - 2.0 * t);
            return mix(a.xyz, b.xyz, ts);
        }
    }
    return layers.palette[last].xyz;
}

// Result of applying the named vortex pass: the body-local sample
// position is swirled around each vortex's axis and an accumulated tint
// multiplier is returned so the palette lookup can be recoloured inside
// long-lived features like the Great Red Spot.
struct VortexPass {
    pos:  vec3<f32>,
    tint: vec3<f32>,
}

fn apply_named_vortices(p_in: vec3<f32>) -> VortexPass {
    var p = p_in;
    var tint = vec3<f32>(1.0);
    let count = layers.counts.x;
    for (var i = 0u; i < MAX_VORTICES; i = i + 1u) {
        if i >= count { break; }
        let v = layers.vortex_pos[i];
        let c = latlon_to_normal(v.x, v.y);
        let radius = v.z;
        let strength = v.w;
        let cosang = clamp(dot(normalize(p), c), -1.0, 1.0);
        let ang = acos(cosang);
        if ang < radius && radius > 1e-4 {
            let falloff = smoothstep(radius, 0.0, ang);
            p = rotate_axis(p, c, strength * falloff);
            let tv = layers.vortex_tint[i].xyz;
            tint = mix(tint, tv, falloff);
        }
    }
    return VortexPass(p, tint);
}

// Result of applying the edge vortex chain pass: an optionally
// swirled sample position plus a per-channel colour multiplier that
// shapes each eddy as a zonally elongated oval with a slightly warm
// core and a faint darker rim — matches the look of real Jovian/
// Saturnian band-shear vortices without reading as paint dots.
struct EdgePass {
    pos:      vec3<f32>,
    spot_mul: vec3<f32>,
}

// Hashed stateless edge vortex chain. Primitives spawn, grow, drift,
// and fade along the nearest band boundary. Each eddy has its own
// lifecycle phase, size, latitude offset, and density gate so the
// population reads as a natural scatter rather than a synced grid.
fn apply_edge_vortices(p_in: vec3<f32>, t: f32) -> EdgePass {
    var p = p_in;
    var mul = vec3<f32>(1.0);
    if layers.counts.z == 0u { return EdgePass(p, mul); }
    let slots = layers.counts.y;
    if slots == 0u { return EdgePass(p, mul); }
    let bands = layers.band_frequency;
    if bands <= 0.0 { return EdgePass(p, mul); }

    let lifetime = max(layers.edge_chain.z, 1.0);
    let base_r = layers.edge_chain.x;
    let strength = layers.edge_chain.y;

    let n = normalize(p);
    let lat = n.y;
    let b_idx = round(lat * bands);
    let b_lat = b_idx / bands;
    // Meridional reach includes lat jitter (≤0.7·base_r) plus the
    // maximum meridional ellipse radius (~1.15·base_r). 3× covers it.
    if abs(lat - b_lat) > base_r * 3.0 { return EdgePass(p, mul); }

    // Per-band activity mask — only a fraction of boundaries host a
    // chain, and each active band picks its own density.
    let b_cell = u32(i32(b_idx) + 1024);
    let band_seed = b_cell * 747796405u ^ layers.seed_lo;
    if hash_f32(pcg(band_seed)) < 0.55 { return EdgePass(p, mul); }
    let band_density = 0.25 + 0.55 * hash_f32(pcg(band_seed ^ 0xA5A5A5A5u));

    let lon = atan2(n.z, n.x);
    let lon_norm = lon / TAU + 0.5;
    let slot_f = lon_norm * f32(slots);
    let slot_center = i32(floor(slot_f));
    let i_slots = i32(slots);

    for (var ds = -1; ds <= 1; ds = ds + 1) {
        var s = (slot_center + ds) % i_slots;
        if s < 0 { s = s + i_slots; }
        let epoch = u32(floor(t / lifetime));
        let seed_raw = (u32(s) * 2654435761u)
                     ^ (b_cell * 40503u)
                     ^ (epoch * 374761393u)
                     ^ layers.seed_lo;
        let h0 = pcg(seed_raw);
        let h1 = pcg(h0 ^ 0x9e3779b9u);
        let h2 = pcg(h1 ^ 0x85ebca6bu);
        let h3 = pcg(h2 ^ 0xc2b2ae35u);
        let h4 = pcg(h3 ^ 0x27d4eb2fu);
        let h5 = pcg(h4 ^ 0x165667b1u);

        // Density gate — most slots empty, population broken up.
        if hash_f32(h0) > band_density { continue; }

        let slot_jitter = hash_f32(h1);
        let slot_lon = (f32(s) + slot_jitter) / f32(slots) * TAU - PI;
        let lat_jitter = (hash_f32(h2) - 0.5) * base_r * 1.4;
        let c_lat = clamp(b_lat + lat_jitter, -0.999, 0.999);
        let c = latlon_to_normal(c_lat, slot_lon);

        // Per-spot size and lifecycle phase — independent birth time
        // kills the synced pulse across the population.
        let size_mul = 0.35 + 1.30 * hash_f32(h3);
        let phase_off = hash_f32(h4);
        let age = fract(t / lifetime + phase_off);
        let pulse = sin(age * PI);
        if pulse < 1e-3 { continue; }
        // Signed spin — half the vortices rotate the other way so
        // neighbours don't all read as the same handedness.
        let spin_sign = select(-1.0, 1.0, hash_f32(h5) > 0.5);

        // Zonally elongated ellipse in the local tangent frame at the
        // spot centre. Band-shear vortices stretch ~3:1 east-west;
        // 2.2 / 0.7 is a readable approximation. Compare against the
        // current (possibly already warped) sample position so
        // overlapping vortices compose.
        let rx = base_r * size_mul * 2.2 * pulse;
        let ry = base_r * size_mul * 0.7 * pulse;
        let east = normalize(vec3<f32>(-sin(slot_lon), 0.0, cos(slot_lon)));
        let north = normalize(cross(east, c));
        let n_cur = normalize(p);
        let d_vec = n_cur - c;
        let dx = dot(d_vec, east);
        let dy = dot(d_vec, north);
        let ed = sqrt((dx / rx) * (dx / rx) + (dy / ry) * (dy / ry));
        if ed > 1.0 { continue; }

        // Spiral swirl: inner rings rotate far more than outer, so
        // the band colours shear into a visible cyclone. Angle drops
        // to zero at the rim so the eddy blends cleanly into the
        // surrounding flow.
        let inner = 1.0 - smoothstep(0.0, 1.0, ed);
        let rot_amt = strength * 3.0 * size_mul * pulse * inner * spin_sign;
        let cs = cos(rot_amt);
        let sn = sin(rot_amt);
        let dx_r = cs * dx - sn * dy;
        let dy_r = sn * dx + cs * dy;

        // Meridional pinch: pulls the nearest band boundary inward
        // so it visibly bows around the centre. Zonal axis barely
        // compressed so the oval keeps its east-west shape.
        let pinch = 0.55 * inner * inner * pulse;
        let dx_p = dx_r * (1.0 - pinch * 0.12);
        let dy_p = dy_r * (1.0 - pinch);

        let warped = c + east * dx_p + north * dy_p;
        p = normalize(warped);

        // Very light core shade so the eddy is still legible when it
        // sits inside a uniform band. No warm rim — the band
        // distortion is the primary read.
        let shade = mix(1.0, 0.90, inner * 0.6 * pulse);
        mul = mul * vec3<f32>(shade);
    }
    return EdgePass(p, mul);
}

struct CloudDeckResult {
    // Surface-reflectance RGB in linear space.
    color: vec3<f32>,
    // Independent upper-haze density field, normalised to roughly
    // [0, 1]. Drives the Rayleigh blue-gap composite in the fragment:
    // where this is low the cloud deck reads its "real" colour directly,
    // where it is high the haze caps that colour and scattered blue
    // can leak through.
    haze_density: f32,
}

fn cloud_deck_color(p_local: vec3<f32>, t: f32) -> CloudDeckResult {
    var n = normalize(p_local);
    let seed = layers.seed_lo;

    // ── Bulk rotation ────────────────────────────────────────────
    //
    // Rigid spin of the whole cloud deck around the body's Y axis.
    // Driven by `params.rotation_phase`, which the game's per-frame
    // update system advances at the body's real rotation rate. This
    // is the "planet spinning" rotation — it stays in [0, 2π) and
    // keeps the noise field coherent. The differential rotation
    // block below layers a small per-latitude offset on top.
    {
        let cs = cos(params.rotation_phase);
        let sn = sin(params.rotation_phase);
        let px =  n.x * cs + n.z * sn;
        let pz = -n.x * sn + n.z * cs;
        n = vec3<f32>(px, n.y, pz);
    }

    // ── Differential rotation ────────────────────────────────────
    //
    // `sample_speed(lat)` is a signed per-latitude scroll rate. With
    // `dynamics.x` as an overall gain, each latitude band spins around
    // the Y axis at its own rate — retrograde belts go the other way
    // relative to bulk rotation. The bulk rotation itself is still
    // baked into `params.orientation`.
    //
    // The rate lookup is *quantized* to the band lattice: all pixels
    // inside one band share the same `phi`, so the noise field rotates
    // as a rigid strip with no intra-band shear. A continuous lat
    // profile would otherwise shear the noise by `d(sample_speed)/dlat
    // * diff_rate * t` per radian of latitude, growing without bound
    // and aliasing into visible blur within minutes of sim time. The
    // hard rate step at each band boundary coincides with the band
    // staircase colour step downstream, so the seam is hidden.
    let diff_rate = layers.dynamics.x;
    if diff_rate != 0.0 && layers.counts.w == 1u {
        let lat0 = clamp(n.y, -1.0, 1.0);
        let bands_for_rate = max(layers.band_frequency, 1.0);
        let band_lat = clamp(round(lat0 * bands_for_rate) / bands_for_rate, -1.0, 1.0);
        let phi = sample_speed(band_lat) * diff_rate * t;
        let cs = cos(phi);
        let sn = sin(phi);
        let px =  n.x * cs + n.z * sn;
        let pz = -n.x * sn + n.z * cs;
        n = vec3<f32>(px, n.y, pz);
    }

    // ── Named vortices (fixed in body frame) ─────────────────────
    let vortex = apply_named_vortices(n);
    n = normalize(vortex.pos);

    // ── Edge vortex chain ────────────────────────────────────────
    let edge = apply_edge_vortices(n, t);
    n = normalize(edge.pos);

    let lat = clamp(n.y, -1.0, 1.0);
    let turb = sample_turb(lat);

    // ── IQ two-level domain warp ─────────────────────────────────
    //
    // Stage 3 of the doc. Producing the "dragged taffy" turbulence
    // via two successive fBm domain warps. Base frequency raised so
    // warp features are planet-scale small (handful of degrees, not
    // planet-spanning) — Saturn-scale turbulence lives at this tier.
    let q = fbm_3d_vec3(n * 8.0, seed);
    let r = fbm_3d_vec3(n * 8.0 + 4.0 * q, seed ^ 0x1u);
    let warp_field = fbm_3d(n * 18.0 + 4.0 * r, seed ^ 0x2u);
    // `band_warp` is authored as a fraction of a band's width. A
    // band's latitude span is `1 / band_frequency`, so multiplying
    // by that here gives the author-facing units the obvious
    // meaning: `band_warp = 0.3` shifts band boundaries by 30% of a
    // band. Previously this was scaled by a hidden `0.25`, which
    // made authored values silently near-invisible.
    let band_span = 1.0 / max(layers.band_frequency, 1e-3);
    var warped_lat = lat + layers.band_warp * warp_field * turb * band_span;

    // ── Curl noise with boundary sign flip ───────────────────────
    //
    // Stage 7. A single curl noise field applied as a small extra warp
    // to the sample position. Sign is flipped across shear boundaries
    // using the derivative of `sample_speed`, so eddies on either side
    // of a band edge counter-rotate — the doc calls this the single
    // "noise vs. fluid" tell.
    if layers.dynamics.z > 0.0 {
        let eps_lat = 0.015;
        let sh = sample_speed(lat + eps_lat) - sample_speed(lat - eps_lat);
        let sgn = select(-1.0, 1.0, sh >= 0.0);
        let cw = curl_warp(n * 4.0, seed ^ 0x3u);
        n = normalize(n + cw * layers.dynamics.z * turb * sgn * 0.05);
    }

    // ── Shear-driven edge wave (Kelvin–Helmholtz) ────────────────
    //
    // Stage 4. A lat displacement whose amplitude scales with the
    // absolute shear so it only fires at band boundaries. Phase is
    // perturbed by the existing warp field to break the pure sine.
    // Faded at high latitudes where the longitude coord pinches.
    if layers.dynamics.y > 0.0 {
        let eps_lat = 0.015;
        let shear = abs(sample_speed(lat + eps_lat) - sample_speed(lat - eps_lat));
        let lon = atan2(n.z, n.x);
        let pole_fade_w = 1.0 - smoothstep(0.75, 0.95, abs(lat));
        let phase = lon * layers.band_frequency * 4.0 + t * 0.1 + warp_field * TAU;
        warped_lat = warped_lat
            + layers.dynamics.y * shear * sin(phase) * 0.05 * pole_fade_w;
    }

    warped_lat = clamp(warped_lat, -1.0, 1.0);

    // ── Latitude quantization → monotone band staircase ──────────
    //
    // Saturn's bands are a monotone staircase along latitude: each
    // jet locks to a slightly different tone, and the overall drift
    // across the disk is continuous from pole to pole. There is NO
    // ± swing around a mean — that would produce alternating
    // bright/dark belts, which Saturn does not have.
    //
    // Implementation: quantize warped latitude into `band_frequency`
    // discrete plateaus. Each band samples ONE palette position, so
    // adjacent bands differ only by the direction the palette is
    // heading. Palette is authored monotone → bands are monotone.
    //
    // Non-uniform widths: warp the latitude by a small fraction of a
    // band span using a three-sine harmonic sum before quantization.
    // This shifts band boundaries away from the regular lattice
    // without re-introducing a ± envelope.
    let bands = max(layers.band_frequency, 1.0);
    let quant_span = 1.0 / bands;
    let k1 = bands * PI;
    let k2 = k1 * 1.618;
    let k3 = k1 * 2.414;
    let warp_phase = warp_field * 0.8;
    let harm =
          sin(warped_lat * k1 + warp_phase)
        + 0.35 * sin(warped_lat * k2 + warp_phase * 1.3)
        + 0.18 * sin(warped_lat * k3 + warp_phase * 0.7);
    let lat_for_quant = warped_lat + 0.22 * quant_span * harm;
    let quant_lat = clamp(round(lat_for_quant * bands) / bands, -1.0, 1.0);

    // ── Anisotropic streak detail ────────────────────────────────
    //
    // Sampling fbm at `n * vec3(1, N, 1)` compresses the lattice in
    // latitude while leaving longitude free, producing narrow
    // cross-latitude filaments that flow around the body.
    let streak_mid  = fbm_3d(n * vec3<f32>(3.0, 32.0, 3.0), seed ^ 0xCAFEu);
    let streak_fine = fbm_3d(n * vec3<f32>(5.0, 90.0, 5.0), seed ^ 0xF1BEu);

    // Low-frequency organic drift — gentle hue wander across the
    // disk so the staircase picks up subtle longitudinal variation.
    let hue_drift = fbm_3d(n * 4.0, seed ^ 0x51DE1u);

    // Hue: smooth palette lookup across the *un*-quantized warped
    // latitude. The base hue gradient varies continuously pole-to-pole
    // — band edges show up via the luminance step below, not via a
    // hue staircase. Decorative noise perturbations add per-pixel
    // wander without crossing band boundaries.
    let hue_lat = clamp(
        warped_lat
            + 0.020 * hue_drift
            + 0.012 * streak_fine * turb
            + 0.010 * streak_mid  * turb,
        -1.0, 1.0,
    );
    var col = palette_lookup(hue_lat);

    // ── Per-band luminance step ──────────────────────────────────
    //
    // Each quantized band gets a unique hashed luminance offset, so
    // adjacent bands sit on different brightness plateaus and the
    // edges read as crisp steps — but the offsets are random rather
    // than ± alternating, which avoids the zebra pattern that strict
    // alternation produces. `tint.w` (band_contrast) is the swing
    // amplitude. Saturn's true-colour bands look like this: every jet
    // has its own slightly different brightness, no obvious belt/zone
    // metronome.
    let band_idx_i = i32(round(lat_for_quant * bands));
    let band_seed = u32(band_idx_i + 1024) * 747796405u ^ layers.seed_lo;
    let band_rand = hash_f32(pcg(band_seed)) * 2.0 - 1.0;
    col = col * (1.0 + layers.tint.w * band_rand);

    // Luminance pinstripe — the streak layers drive a multiplicative
    // swing on top of the palette so bright/dark fibres read across
    // every band. Fine layer carries the strongest swing because
    // that is the scale Cassini resolves.
    col = col * (1.0
        + layers.turbulence * turb * streak_fine * 3.4
        + layers.turbulence * turb * streak_mid  * 2.0);

    // ── Fine-scale turbulence ────────────────────────────────────
    if layers.turbulence > 0.0 {
        let mid = fbm_3d(n * 8.0,  seed ^ 0x9E3779B9u);
        let hi  = fbm_3d(n * 22.0, seed ^ 0xDEADBEEFu);
        let detail = mid * 0.7 + hi * 0.45;
        col = col * (1.0 + layers.turbulence * turb * detail * 1.2);
    }

    col = col * vortex.tint;
    col = col * edge.spot_mul;
    col = col * layers.tint.xyz;

    // ── Upper-haze density field ─────────────────────────────────
    //
    // Independent low-frequency fBm evaluated on the same rotated
    // sample position so the field rotates with the body but is
    // decorrelated from the cloud-band field. A latitude bias shifts
    // the mean toward / away from one hemisphere so "blue gap"
    // clearings can concentrate (Saturn's Cassini-era north).
    let haze_scale = layers.rayleigh_params.x;
    let haze_bias = layers.rayleigh_params.z;
    let haze_raw =
        fbm_3d(n * haze_scale, seed ^ 0x13371337u) * 0.5 + 0.5;
    let haze_biased = haze_raw + haze_bias * 0.5 * lat;
    let haze_density = clamp(haze_biased, 0.0, 1.0);

    return CloudDeckResult(col, haze_density);
}

// ── Layer 3: haze ───────────────────────────────────────────────────────────
//
// Multiplicative tint applied on top of the cloud deck. The view-angle
// bias concentrates the tint near the terminator, mimicking the longer
// optical path through the haze layer at grazing angles.

fn apply_haze(base: vec3<f32>, n_dot_v: f32) -> vec3<f32> {
    let thickness = layers.haze_tint_thickness.w;
    if thickness <= 0.0 {
        return base;
    }
    let tint = layers.haze_tint_thickness.xyz;
    let terminator_bias = layers.haze_params.x;

    // Angular weight: pure thickness at nadir, grows toward the rim.
    let rim_weight = 1.0 - clamp(n_dot_v, 0.0, 1.0);
    let w = thickness * mix(1.0, rim_weight, terminator_bias);
    return mix(base, base * tint, clamp(w, 0.0, 1.0));
}

// ── Layer 4: rim halo ───────────────────────────────────────────────────────
//
// Approximates the upper-atmosphere glow visible just outside the
// silhouette of a gas giant. Integrates an exponential-density column
// along the viewing ray between the cloud-deck radius `R` and an outer
// cutoff `R + outer_altitude`. A ray that misses the cloud deck but
// crosses the shell picks up a contribution; a ray that hits the cloud
// deck picks up a smaller contribution from the near-side shell above
// the hit.
//
// Simplified first pass: treat the column as a trapezoid in altitude
// and weight by a single exponential factor. Good enough to see a soft
// halo; ray-marched scattering comes later.

struct RimHit {
    contribution: vec3<f32>,
    opacity: f32,
}

fn no_rim() -> RimHit {
    return RimHit(vec3<f32>(0.0), 0.0);
}

fn rim_halo_contribution(
    cam_pos: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    light_dir_ws: vec3<f32>,
) -> RimHit {
    let intensity = layers.rim_color_intensity.w;
    if intensity <= 0.0 {
        return no_rim();
    }
    let outer_alt = layers.rim_shape.y;
    if outer_alt <= 0.0 {
        return no_rim();
    }

    let r_inner = params.radius;
    let r_outer = params.radius + outer_alt;

    // Ray-sphere intersection against the outer shell.
    let oc = cam_pos - center;
    let half_b = dot(oc, ray_dir);
    let c_outer = dot(oc, oc) - r_outer * r_outer;
    let disc_o = half_b * half_b - c_outer;
    if disc_o < 0.0 {
        return no_rim();
    }
    let sq_o = sqrt(max(disc_o, 0.0));
    let t0 = -half_b - sq_o;
    let t1 = -half_b + sq_o;
    let t_entry = max(t0, 0.0);
    if t1 <= t_entry {
        return no_rim();
    }

    // Closest approach distance from the center.
    let closest = oc + ray_dir * half_b * (-1.0);
    let closest_d = length(closest);
    // Normalised altitude of the closest approach, 0 at cloud deck,
    // 1 at outer shell. Values below zero mean the ray enters the cloud
    // deck (we still want a rim contribution from the front shell).
    let closest_alt = clamp((closest_d - r_inner) / max(r_outer - r_inner, 1e-6), -1.0, 1.0);

    // Exponential falloff with altitude — stand-in for atmospheric
    // density. Scale height is in render units.
    let scale_h = max(layers.rim_shape.x, 1e-4);
    let rel_h = (closest_alt * max(r_outer - r_inner, 1e-6)) / scale_h;
    let density = exp(-max(rel_h, 0.0));

    // Column length along the ray between entry and exit of the shell,
    // clamped at the cloud-deck sphere if the ray hits it.
    let c_inner = dot(oc, oc) - r_inner * r_inner;
    let disc_i = half_b * half_b - c_inner;
    var t_exit = t1;
    if disc_i >= 0.0 {
        let ti = -half_b - sqrt(max(disc_i, 0.0));
        if ti > t_entry {
            t_exit = min(t_exit, ti);
        }
    }
    let column = max(t_exit - t_entry, 0.0);

    // Sunlit factor — halo only appears on the lit side. `closest` is
    // the closest-approach vector relative to the planet center, so its
    // projection onto the light direction tells us which face the
    // column is on.
    let closest_dir = normalize(oc + ray_dir * (-half_b));
    let sun_factor = clamp(dot(closest_dir, light_dir_ws) * 0.5 + 0.5, 0.0, 1.0);

    let color = layers.rim_color_intensity.xyz;
    // Normalise the column length by the outer shell thickness so the
    // `intensity` parameter behaves consistently as `outer_altitude_m`
    // changes. A grazing tangent ray can see many shell thicknesses of
    // atmosphere, so clamp the saturating path factor into [0, 1]
    // rather than letting it climb unboundedly — otherwise the
    // rim halo outruns the tonemapper and pins to white on the sunlit
    // silhouette. Replace with a proper along-ray scattering integral
    // when fidelity goes up.
    let column_norm = column / max(r_outer - r_inner, 1e-6);
    let path_factor = column_norm / (1.0 + column_norm);
    let strength = clamp(intensity * density * sun_factor * path_factor, 0.0, intensity);
    return RimHit(color * strength, min(strength, 1.0));
}

// ── Fragment ────────────────────────────────────────────────────────────────

@fragment
fn fragment(in: VertexOutput) -> FragOutput {
    let cam_pos = view.world_position;
    let ray_dir = normalize(in.world_position - cam_pos);
    let center = in.sphere_center;
    let light_dir_ws = params.light_dir.xyz;

    // Ray-sphere intersection against the cloud deck.
    let oc = cam_pos - center;
    let half_b = dot(oc, ray_dir);
    let c_deck = dot(oc, oc) - params.radius * params.radius;
    let disc_deck = half_b * half_b - c_deck;

    var cloud_color = vec3<f32>(0.0);
    var cloud_opacity = 0.0;
    var depth_out = 0.0;

    if disc_deck >= 0.0 {
        let t_hit = -half_b - sqrt(max(disc_deck, 0.0));
        if t_hit > 0.0 {
            let hit = cam_pos + t_hit * ray_dir;
            let normal = normalize(hit - center);
            let p_local = rotate_quat(params.orientation, normal);

            let elapsed = params.light_dir.w;

            let deck = cloud_deck_color(p_local, elapsed);
            var base = deck.color;

            // ── Lighting ──
            //
            // Gas-giant atmospheres forward-scatter sunlight well past
            // the geometric terminator, so a hard Lambert cosine edge
            // reads as "solid rock painted cream", not "planet with
            // atmosphere". Use a wrap-lit squared falloff: the `wrap`
            // parameter shifts the zero of illumination to
            // `n·l = -wrap`, broadening the terminator over ~20°, and
            // the square softens the mid-range so the crescent fades
            // smoothly. Beyond `n·l < -wrap` the sun term is still
            // pure zero, so the night hemisphere stays physically dark.
            //
            // No blanket limb-darkening factor. A proper gas-giant
            // limb darkening comes out of the column radiance
            // transfer integral; a `pow(n·v, k)` multiplier here
            // produced a thin colored seam at the silhouette that
            // read as a rendering bug. Re-add via column integration
            // when fidelity goes up.
            let n_dot_l = dot(normal, light_dir_ws);
            let n_dot_v = clamp(dot(normal, -ray_dir), 0.0, 1.0);

            let wrap = 0.18;
            let wrapped = max((n_dot_l + wrap) / (1.0 + wrap), 0.0);
            let soft_nl = wrapped * wrapped;
            let lambert = soft_nl * (1.0 / (4.0 * 3.14159265));

            // ── Ring shadow on the cloud deck ────────────────────
            //
            // When a ring system is present, project the surface
            // point toward the sun in body-local space and test
            // whether the projection crosses the ring plane inside
            // the inner/outer annulus. Because the ring plane is
            // the body's equatorial plane (y = 0 in body-local),
            // the projection closes in a single division.
            //
            // The final `ring_shadow_t` is applied to the entire lit
            // composite at the end of this block — not just
            // `sun_term` — so terminator warmth and Fresnel rim get
            // shadowed too. Multiplying only `sun_term` produced a
            // visible seam at the terminator because the warmth /
            // fresnel contributions leaked bright through the dim
            // shadowed strip.
            var ring_shadow_t = 1.0;
            if layers.ring_shadow.w > 0.5 {
                let l_local = rotate_quat(params.orientation, light_dir_ws);
                if l_local.y != 0.0 {
                    // Signed distance along the sun ray to reach y=0.
                    // Positive = travels toward the sun, which is the
                    // direction we want (shadow is cast sunward from
                    // the surface point). `t_plane <= 0` means the
                    // ring plane is behind the point relative to the
                    // sun, so it can't occlude.
                    let t_plane = -p_local.y / l_local.y;
                    if t_plane > 0.0 {
                        let hit_xz = p_local.xz + l_local.xz * t_plane;
                        let r_hit = length(hit_xz) * params.radius;
                        let r_in = layers.ring_shadow.x;
                        let r_out = layers.ring_shadow.y;
                        if r_hit >= r_in && r_hit <= r_out {
                            // Procedural ring opacity lookup. Smooth
                            // ramp at edges so the shadow falloff
                            // isn't a hard cookie-cutter; coarse
                            // radial noise breaks the single ring
                            // into bright/dark bands (Cassini gap,
                            // Encke, ringlets) using the same seed
                            // that drives the ring shader.
                            let u = (r_hit - r_in) / max(r_out - r_in, 1e-6);
                            let edge = smoothstep(0.0, 0.05, u)
                                     * smoothstep(0.0, 0.05, 1.0 - u);
                            // Cassini-style gap at u ≈ 0.42
                            let gap = smoothstep(0.03, 0.0, abs(u - 0.42));
                            let noise_amp = layers.ring_shadow.z;
                            let n_ring = fbm_2d(vec2<f32>(u * 80.0, 0.0), layers.seed_lo ^ 0x7A5u);
                            let dens = clamp(edge - gap * 0.9
                                + noise_amp * n_ring * 0.35, 0.0, 1.0);
                            ring_shadow_t = 1.0 - 0.85 * dens;
                        }
                    }
                }
            }

            let sun_term = base * params.light_intensity * lambert;
            var lit = sun_term + base * params.ambient_intensity;

            // ── Per-channel Minnaert limb darkening ──────────────
            //
            // Apply BEFORE terminator warmth / Fresnel rim so those
            // tints lie on top of an already-rounded disk. A blanket
            // `pow(n·v, k)` with k≈0.3 gives a physically-plausible
            // limb roll-off; per-channel exponents reproduce the
            // wavelength-dependent limb reddening seen in Cassini
            // photometry (red shrinks slower than blue).
            let ld_strength = layers.limb_exponents.w;
            if ld_strength > 0.0 {
                let mu = max(n_dot_v, 1e-3);
                let kr = layers.limb_exponents.x;
                let kg = layers.limb_exponents.y;
                let kb = layers.limb_exponents.z;
                let darken = vec3<f32>(pow(mu, kr), pow(mu, kg), pow(mu, kb));
                lit = mix(lit, lit * darken, ld_strength);
            }

            // ── Rayleigh "blue gap" composite ────────────────────
            //
            // Where the upper-haze density thins below the authored
            // threshold, the cloud deck reads as bluer because
            // molecular scattering is no longer masked. Multiplies
            // by sun-lit factor so the gap only shows on the day side.
            let ray_strength = layers.rayleigh_color.w;
            if ray_strength > 0.0 {
                let threshold = layers.rayleigh_params.y;
                // `1 - smoothstep(threshold-w, threshold, density)`
                // keeps the gap hard at its core but rolls off at the
                // edges so it doesn't clip to black/blue.
                let gap_w = 0.15;
                let gap = 1.0 - smoothstep(threshold - gap_w, threshold,
                    deck.haze_density);
                // Grazing boost — Rayleigh path length grows near the
                // limb, so make the gap brighter there.
                let path = 1.0 + pow(1.0 - n_dot_v, 1.5) * 1.4;
                let rayleigh_contrib = layers.rayleigh_color.xyz
                    * ray_strength * gap * soft_nl * path
                    * params.light_intensity * (1.0 / (4.0 * 3.14159265));
                // Additive composite — leaks on top of the darkened
                // cloud deck, can't turn the disk black.
                lit = lit + rayleigh_contrib;
            }

            // Haze modulates the lit colour. Multiplicative so dark
            // side stays dark (haze fades to zero when `lit` is zero).
            lit = apply_haze(lit, n_dot_v);

            // ── Terminator warmth ────────────────────────────────
            //
            // Warm chromatic shift in the already-lit region near the
            // terminator. Applied as a multiplicative mix against
            // `lit` rather than an additive contribution so it can
            // never overpower the base colour and cannot leak
            // brightness into the dark hemisphere. The previous
            // additive version produced a scalloped bright ring.
            let tw_strength = layers.terminator_warmth.w;
            if tw_strength > 0.0 {
                let lit_side = smoothstep(-0.05, 0.10, n_dot_l);
                let near_zero = 1.0 - smoothstep(0.05, 0.30, n_dot_l);
                let w = lit_side * near_zero * tw_strength;
                lit = mix(lit, lit * layers.terminator_warmth.xyz, clamp(w, 0.0, 1.0));
            }

            // ── Fresnel rim ──────────────────────────────────────
            //
            // Cool desaturated rim on the lit limb as a cheap
            // Rayleigh stand-in. Multiplicative tint instead of
            // additive so bright hemispheres don't clip, and tightly
            // gated by `n_dot_l^2` so the unlit limb stays dark.
            let fr_strength = layers.fresnel_rim.w;
            if fr_strength > 0.0 {
                let fresnel = pow(1.0 - n_dot_v, 4.0);
                let lit_gate = clamp(n_dot_l, 0.0, 1.0);
                let w = fresnel * lit_gate * lit_gate * fr_strength;
                lit = mix(lit, lit * layers.fresnel_rim.xyz, clamp(w, 0.0, 1.0));
            }

            // ── Forward-scatter crescent ─────────────────────────
            //
            // Mie-like forward scatter from the atmosphere when the
            // sun sits behind the planet relative to the camera.
            // Paints a thin warm crescent around the unlit limb so
            // the terminator does not cliff straight into black.
            // Additive: the base `lit` is already ~0 on the night
            // side, so there is nothing to blend against.
            let fs_intensity = layers.rim_color_intensity.w;
            if fs_intensity > 0.0 {
                let phase = pow(max(dot(ray_dir, light_dir_ws), 0.0), 8.0);
                let unlit_gate = smoothstep(0.15, -0.10, n_dot_l);
                let rim_gate = pow(1.0 - clamp(n_dot_v, 0.0, 1.0), 3.0);
                let w = phase * unlit_gate * rim_gate * fs_intensity
                    * params.light_intensity * (1.0 / (4.0 * 3.14159265));
                lit = lit + layers.rim_color_intensity.xyz * w;
            }

            // Apply ring shadow to the whole lit composite — sun
            // term + warmth + fresnel + haze — so the shadow stripe
            // reads consistently through the terminator instead of
            // leaving an unshadowed rim.
            lit = lit * ring_shadow_t;

            cloud_color = lit;
            cloud_opacity = 1.0;

            let clip = view.clip_from_world * vec4(hit, 1.0);
            depth_out = clip.z / clip.w;
        }
    }

    // Rim halo: computed once, but suppressed for fragments that hit
    // the cloud deck. For hit fragments the cloud-deck colour already
    // represents the optically-thick surface and adding the halo on
    // top would bleed the bright rim across the dark hemisphere. A
    // proper in-atmosphere absorption integral can reintroduce a
    // "near-side atmospheric glow" contribution later without
    // disturbing the cloud-deck shading.
    //
    // `rim.contribution` already folds the authored `intensity` scalar,
    // so we normalise by the same `1/(4π)` factor the cloud deck uses
    // and multiply by `light_intensity` once.
    let rim_hidden = cloud_opacity;
    let rim = rim_halo_contribution(cam_pos, ray_dir, center, light_dir_ws);
    let rim_contrib = rim.contribution
        * params.light_intensity
        * (1.0 / (4.0 * 3.14159265))
        * (1.0 - rim_hidden);

    var final_rgb = cloud_color + rim_contrib;
    var final_a = max(cloud_opacity, rim.opacity * (1.0 - rim_hidden));

    // Miss path: discard if the pixel is completely transparent so
    // downstream compositing sees through the billboard.
    if final_a <= 0.001 {
        discard;
    }

    // Miss path: depth at closest approach along the ray. `-half_b`
    // is the parametric distance from cam_pos to the foot of the
    // perpendicular dropped from the sphere center onto the ray — a
    // reasonable stand-in for the silhouette depth for rim-only
    // fragments.
    if cloud_opacity <= 0.0 {
        let closest_point = cam_pos + ray_dir * max(-half_b, 0.0);
        let clip = view.clip_from_world * vec4(closest_point, 1.0);
        depth_out = clip.z / clip.w;
    }

    return FragOutput(vec4<f32>(final_rgb, final_a), depth_out);
}
