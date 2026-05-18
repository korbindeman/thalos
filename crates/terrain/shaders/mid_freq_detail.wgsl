// Mid-frequency detail kernel.
//
// Reads + writes an R32Float 2D-array storage texture with 6 layers
// (cubemap faces). Each invocation maps its texel to a 3D direction on
// the body's sphere, evaluates fBm of value-noise at the corresponding
// world-space position, and additively perturbs the stored height.
//
// Workgroup size 8×8×1: each workgroup covers a 64-texel tile of one
// face. Caller dispatches `(ceil(res/8), ceil(res/8), 6)` workgroups
// per pass.
//
// Layer / face convention: layer N == `CubemapFace` with discriminant N,
// i.e. 0=+X, 1=−X, 2=+Y, 3=−Y, 4=+Z, 5=−Z. The `face_uv_to_dir` below
// mirrors the Rust function of the same name in
// `crates/terrain/src/cubemap.rs` — if either side moves, both
// must move together or the kernel writes to the wrong texels.
//
// The fBm here is the **canonical** Thalos noise: byte-equivalent to
// `crates/planet_rendering/src/shaders/noise.wgsl` (the impostor's
// `thalos::noise::fbm3`) and to the Rust port in
// `crates/terrain/src/noise.rs`. If you edit any of:
//   - the u32 PCG mixer (`pcg_u32`)
//   - the three-coord hash fold (`hash3_u32`)
//   - the 24-bit-precision f32 conversion (`hash3_f32`)
//   - the quintic fade
//   - the trilinear interpolation
//   - the per-octave sub-seed (`pcg_u32(seed + o)`)
// keep all three mirrors in lockstep. The impostor's surface and the
// bake's surface only agree because all three sites evaluate the
// **same arithmetic at the same point**.
//
// Long-term, naga_oil joins `bake_dump` and these get `#import`-shared
// from one file. Today the duplication is hand-maintained; the
// equivalent runs in `noise.rs` tests pin the f32 results.

struct Params {
    body_radius_m: f32,
    base_wl_m: f32,
    noise_amp_m: f32,
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
    seed: u32,
    _pad: u32,
}

@group(0) @binding(0) var height: texture_storage_2d_array<r32float, read_write>;
@group(0) @binding(1) var<uniform> params: Params;

// ---------------------------------------------------------------------------
// Canonical fBm — byte-mirror of `planet_rendering/src/shaders/noise.wgsl`
//                              and `terrain/src/noise.rs`
// ---------------------------------------------------------------------------

// One step of a u32 PCG mixer (Jarzynski, "Hash Functions for GPU
// Rendering"). Constants must match `pcg_u32` in noise.rs.
fn pcg_u32(state: u32) -> u32 {
    let s = state * 747796405u + 2891336453u;
    let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (word >> 22u) ^ word;
}

// Hash three integer lattice coords + a seed to a u32. Repeated PCG
// folding decorrelates the output across coordinates and seed.
fn hash3_u32(ix: i32, iy: i32, iz: i32, seed: u32) -> u32 {
    var h = pcg_u32(seed);
    h = pcg_u32(h ^ bitcast<u32>(ix));
    h = pcg_u32(h ^ bitcast<u32>(iy));
    h = pcg_u32(h ^ bitcast<u32>(iz));
    return h;
}

// Hash three integer lattice coords + a seed to a f32 in [-1, 1).
// 24-bit mantissa precision; `(h >> 8u) / 16777216.0` is exact in f32.
fn hash3_f32(ix: i32, iy: i32, iz: i32, seed: u32) -> f32 {
    let h = hash3_u32(ix, iy, iz, seed);
    let u = f32(h >> 8u) / 16777216.0;
    return u * 2.0 - 1.0;
}

// Perlin's quintic fade, `6t⁵ − 15t⁴ + 10t³`.
fn noise_fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// 3D value noise at a point, seeded. Returns roughly in [-1, 1].
fn value_noise_3d(x: f32, y: f32, z: f32, seed: u32) -> f32 {
    let xi = i32(floor(x));
    let yi = i32(floor(y));
    let zi = i32(floor(z));
    let fx = noise_fade(x - f32(xi));
    let fy = noise_fade(y - f32(yi));
    let fz = noise_fade(z - f32(zi));

    let c000 = hash3_f32(xi,     yi,     zi,     seed);
    let c100 = hash3_f32(xi + 1, yi,     zi,     seed);
    let c010 = hash3_f32(xi,     yi + 1, zi,     seed);
    let c110 = hash3_f32(xi + 1, yi + 1, zi,     seed);
    let c001 = hash3_f32(xi,     yi,     zi + 1, seed);
    let c101 = hash3_f32(xi + 1, yi,     zi + 1, seed);
    let c011 = hash3_f32(xi,     yi + 1, zi + 1, seed);
    let c111 = hash3_f32(xi + 1, yi + 1, zi + 1, seed);

    let x00 = c000 + (c100 - c000) * fx;
    let x10 = c010 + (c110 - c010) * fx;
    let x01 = c001 + (c101 - c001) * fx;
    let x11 = c011 + (c111 - c011) * fx;

    let y0 = x00 + (x10 - x00) * fy;
    let y1 = x01 + (x11 - x01) * fy;

    return y0 + (y1 - y0) * fz;
}

// fBm stacker. Argument order matches `thalos::noise::fbm3` in the
// impostor's noise.wgsl: `(p, seed, octaves, persistence, lacunarity)`.
// Bounded at 8 octaves for safety; the per-octave sub-seed is
// `pcg_u32(seed + o)`, matching noise.rs exactly.
fn fbm3(p: vec3<f32>, seed: u32, octaves: u32, persistence: f32, lacunarity: f32) -> f32 {
    var sum: f32 = 0.0;
    var amp: f32 = 1.0;
    var freq: f32 = 1.0;
    var norm: f32 = 0.0;
    let n = min(octaves, 8u);
    for (var o: u32 = 0u; o < n; o = o + 1u) {
        let osubseed = pcg_u32(seed + o);
        sum = sum + amp * value_noise_3d(p.x * freq, p.y * freq, p.z * freq, osubseed);
        norm = norm + amp;
        amp = amp * persistence;
        freq = freq * lacunarity;
    }
    return sum / norm;
}

// Derivative of the quintic fade: 30t²(t − 1)².
fn noise_fade_derivative(t: f32) -> f32 {
    return 30.0 * t * t * (t - 1.0) * (t - 1.0);
}

struct NoiseDerivative3 {
    value: f32,
    derivative: vec3<f32>,
}

// 3D value noise with analytic derivatives. Inigo Quilez's expansion of
// the trilinear interpolation lets us evaluate value + ∂/∂x ∂/∂y ∂/∂z
// from the same 8 corner samples — no finite differences. Bit-mirror of
// `value_noise_3d_derivative` in `terrain/src/noise.rs`.
fn value_noise_3d_derivative(x: f32, y: f32, z: f32, seed: u32) -> NoiseDerivative3 {
    let xi = i32(floor(x));
    let yi = i32(floor(y));
    let zi = i32(floor(z));

    let wx = x - f32(xi);
    let wy = y - f32(yi);
    let wz = z - f32(zi);

    let ux = noise_fade(wx);
    let uy = noise_fade(wy);
    let uz = noise_fade(wz);
    let dux = noise_fade_derivative(wx);
    let duy = noise_fade_derivative(wy);
    let duz = noise_fade_derivative(wz);

    let a = hash3_f32(xi,     yi,     zi,     seed);
    let b = hash3_f32(xi + 1, yi,     zi,     seed);
    let c = hash3_f32(xi,     yi + 1, zi,     seed);
    let d = hash3_f32(xi + 1, yi + 1, zi,     seed);
    let e = hash3_f32(xi,     yi,     zi + 1, seed);
    let f = hash3_f32(xi + 1, yi,     zi + 1, seed);
    let g = hash3_f32(xi,     yi + 1, zi + 1, seed);
    let h = hash3_f32(xi + 1, yi + 1, zi + 1, seed);

    let k0 = a;
    let k1 = b - a;
    let k2 = c - a;
    let k3 = e - a;
    let k4 = a - b - c + d;
    let k5 = a - c - e + g;
    let k6 = a - b - e + f;
    let k7 = -a + b + c - d + e - f - g + h;

    let value = k0
        + k1 * ux
        + k2 * uy
        + k3 * uz
        + k4 * ux * uy
        + k5 * uy * uz
        + k6 * uz * ux
        + k7 * ux * uy * uz;
    let derivative = vec3<f32>(
        (k1 + k4 * uy + k6 * uz + k7 * uy * uz) * dux,
        (k2 + k5 * uz + k4 * ux + k7 * uz * ux) * duy,
        (k3 + k6 * ux + k5 * uy + k7 * ux * uy) * duz,
    );

    return NoiseDerivative3(value, derivative);
}

// fBm with analytic derivatives. Stacks `value_noise_3d_derivative`
// across octaves and applies the chain rule for the per-octave
// frequency scaling: ∇(f(αp)) = α∇f(αp). Scalar `value` is
// bit-identical to `fbm3` (same hash, same sub-seeding, same f32
// ordering). Bit-mirror of `fbm3_derivative` in noise.rs.
fn fbm3_derivative(p: vec3<f32>, seed: u32, octaves: u32, persistence: f32, lacunarity: f32) -> NoiseDerivative3 {
    var sum_value: f32 = 0.0;
    var sum_grad: vec3<f32> = vec3<f32>(0.0);
    var amp: f32 = 1.0;
    var freq: f32 = 1.0;
    var norm: f32 = 0.0;
    let n = min(octaves, 8u);
    for (var o: u32 = 0u; o < n; o = o + 1u) {
        let osubseed = pcg_u32(seed + o);
        let nd = value_noise_3d_derivative(p.x * freq, p.y * freq, p.z * freq, osubseed);
        sum_value = sum_value + amp * nd.value;
        sum_grad = sum_grad + (amp * freq) * nd.derivative;
        norm = norm + amp;
        amp = amp * persistence;
        freq = freq * lacunarity;
    }
    return NoiseDerivative3(sum_value / norm, sum_grad / norm);
}

// ---------------------------------------------------------------------------
// Cubemap face → direction
// ---------------------------------------------------------------------------
//
// Mirrors `face_uv_to_dir` in `crates/terrain/src/cubemap.rs`. The
// inverse (`dir_to_face_uv`) is the one the CPU side calls when sampling
// the resulting cubemap, so this function must match its convention
// exactly — otherwise the GPU writes to texel (face, x, y) but the
// CPU reads it back from a different direction.

fn face_uv_to_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let s = uv.x * 2.0 - 1.0;
    let t = uv.y * 2.0 - 1.0;
    var d: vec3<f32>;
    switch face {
        case 0u: { d = vec3<f32>( 1.0, -t,   -s); }    // +X
        case 1u: { d = vec3<f32>(-1.0, -t,    s); }    // −X
        case 2u: { d = vec3<f32>(   s,  1.0,  t); }    // +Y
        case 3u: { d = vec3<f32>(   s, -1.0, -t); }    // −Y
        case 4u: { d = vec3<f32>(   s, -t,   1.0); }   // +Z
        default: { d = vec3<f32>(  -s, -t,  -1.0); }   // −Z (5)
    }
    return normalize(d);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

// Per-face constant tangent basis. Mirrors `face_tangent_basis` in
// `crates/terrain/src/pipeline.rs` — both render paths must project the
// 3D noise gradient onto the same basis or the eroded surface differs
// between bake and runtime detail cascade.
fn face_tangent_basis(face: u32) -> array<vec3<f32>, 2> {
    var tx: vec3<f32>;
    var ty: vec3<f32>;
    switch face {
        case 0u: { tx = vec3<f32>( 0.0, 0.0, -1.0); ty = vec3<f32>(0.0, -1.0,  0.0); } // +X
        case 1u: { tx = vec3<f32>( 0.0, 0.0,  1.0); ty = vec3<f32>(0.0, -1.0,  0.0); } // −X
        case 2u: { tx = vec3<f32>( 1.0, 0.0,  0.0); ty = vec3<f32>(0.0,  0.0,  1.0); } // +Y
        case 3u: { tx = vec3<f32>( 1.0, 0.0,  0.0); ty = vec3<f32>(0.0,  0.0, -1.0); } // −Y
        case 4u: { tx = vec3<f32>( 1.0, 0.0,  0.0); ty = vec3<f32>(0.0, -1.0,  0.0); } // +Z
        default: { tx = vec3<f32>(-1.0, 0.0,  0.0); ty = vec3<f32>(0.0, -1.0,  0.0); } // −Z (5)
    }
    return array<vec3<f32>, 2>(tx, ty);
}

// Bake-time erosion params. Hardcoded for now; if a body needs custom
// tuning the natural extension is to add an `ErosionFilterParams`
// block to the host UBO and bind it. Values mirror
// `pipeline.rs::erosion_params_base()` (the runtime cascade defaults)
// with `scale` rescaled to the bake's coarser base wavelength so the
// gully pattern reads at mid-freq sizes, not local-cascade sizes.
fn bake_erosion_params(base_wl_m: f32) -> ErosionFilterParams {
    return ErosionFilterParams(
        base_wl_m / 6.0,                    // scale: ~5 km gullies on 30 km features
        0.22,                               // strength
        0.5,                                // gully_weight
        1.5,                                // detail
        vec4<f32>(0.1, 0.0, 0.1, 2.0),      // rounding
        vec4<f32>(1.25, 1.25, 2.8, 1.5),    // onset
        vec2<f32>(0.7, 1.0),                // assumed_slope
        0.7,                                // cell_scale
        0.5,                                // normalization
        4,                                  // octaves
        2.0,                                // lacunarity
        0.5,                                // gain
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(height);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let res_f = f32(dims.x);
    let face = gid.z;

    // Pixel center → uv → direction → world-space surface point.
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / res_f;
    let dir = face_uv_to_dir(face, uv);
    let pos = dir * params.body_radius_m;

    // fBm with analytic 3D gradient. The gradient feeds erosion's
    // slope-onset / drainage logic — finite-differencing across the
    // cubemap would introduce face-seam artifacts the analytic form
    // sidesteps. Frequency scale: `1 / base_wl_m`.
    let inv_wl = 1.0 / params.base_wl_m;
    let nd = fbm3_derivative(
        pos * inv_wl,
        params.seed,
        params.octaves,
        params.persistence,
        params.lacunarity,
    );
    let noise_h = nd.value * params.noise_amp_m;
    // Chain rule: ∇(noise(p * inv_wl) * amp) = amp * inv_wl * ∇noise(p * inv_wl).
    let noise_grad = nd.derivative * (params.noise_amp_m * inv_wl);

    // Existing accumulated height + fbm contribution. Erosion sees the
    // combined height + gradient so it can shape drainage networks
    // that respect both the continental base and the mid-freq band.
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let base_h = textureLoad(height, coord, i32(face)).r;
    let combined_h = base_h + noise_h;

    // Per-face constant tangent basis — same projection as the runtime
    // detail cascade in `crates/terrain/src/pipeline.rs`. The face_uv
    // scaled to arc-length metres becomes the 2D `p` coordinate; the
    // 3D noise gradient projects onto the basis to become 2D slope.
    let basis = face_tangent_basis(face);
    let tangent_x = basis[0];
    let tangent_y = basis[1];
    let noise_grad_2d = vec2<f32>(
        dot(noise_grad, tangent_x),
        dot(noise_grad, tangent_y),
    );

    // face_size_m = arc length of a face edge = (π/2) * radius. Matches
    // `pipeline.rs:289` so the bake and runtime cascade sample erosion
    // at the same physical scale.
    let face_size_m = 1.57079632679 * params.body_radius_m;
    let p_2d = uv * face_size_m;

    // Fade target keeps erosion contributions bounded as height grows;
    // ~ ±max_relief_m maps to ±1. Continental highs sit in ±3.5 km
    // today, so 5 km is a forgiving cap.
    let max_relief_m = 5000.0;
    let fade_target = clamp(combined_h / max_relief_m, -1.0, 1.0);

    let erosion_params = bake_erosion_params(params.base_wl_m);
    let erosion = erosion_filter(
        p_2d,
        vec3<f32>(combined_h, noise_grad_2d.x, noise_grad_2d.y),
        fade_target,
        erosion_params,
    );

    let final_h = combined_h + erosion.delta.x;
    textureStore(
        height,
        coord,
        i32(face),
        vec4<f32>(final_h, 0.0, 0.0, 0.0),
    );
}
