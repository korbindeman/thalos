// Canonical 3D value noise + fBm shader library.
//
// Mirrors `crates/terrain_gen/src/noise.rs` bit-for-bit. Every operation
// — the u32 PCG mixer, the quintic fade, the trilinear interpolation,
// the per-octave sub-seed — has the same form on both sides so the
// impostor's high-frequency terrain band agrees with anything the
// future 3D-mesher synthesises from the same canonical function.
//
// If you change anything in here, change `noise.rs` in lockstep (and
// re-pin the parity reference values in that file's tests).
//
// Imports:
//
//   #import thalos::noise::{value_noise_3d, fbm3}
//
// Both work in unitless coordinates — multiply the input direction or
// world-position by the desired frequency (cycles per unit) before
// passing in.

#define_import_path thalos::noise

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

// Perlin's quintic fade, `6t⁵ − 15t⁴ + 10t³`. C² continuous so the
// resulting noise has continuous gradients (matters for shading
// normal perturbation downstream).
fn noise_fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// 3D value noise at a point, seeded. Returns a value in roughly [-1, 1].
fn value_noise_3d(x: f32, y: f32, z: f32, seed: u32) -> f32 {
    let xi = i32(floor(x));
    let yi = i32(floor(y));
    let zi = i32(floor(z));
    let fx = noise_fade(x - f32(xi));
    let fy = noise_fade(y - f32(yi));
    let fz = noise_fade(z - f32(zi));

    let c000 = hash3_f32(xi,        yi,        zi,        seed);
    let c100 = hash3_f32(xi + 1,    yi,        zi,        seed);
    let c010 = hash3_f32(xi,        yi + 1,    zi,        seed);
    let c110 = hash3_f32(xi + 1,    yi + 1,    zi,        seed);
    let c001 = hash3_f32(xi,        yi,        zi + 1,    seed);
    let c101 = hash3_f32(xi + 1,    yi,        zi + 1,    seed);
    let c011 = hash3_f32(xi,        yi + 1,    zi + 1,    seed);
    let c111 = hash3_f32(xi + 1,    yi + 1,    zi + 1,    seed);

    let x00 = c000 + (c100 - c000) * fx;
    let x10 = c010 + (c110 - c010) * fx;
    let x01 = c001 + (c101 - c001) * fx;
    let x11 = c011 + (c111 - c011) * fx;

    let y0 = x00 + (x10 - x00) * fy;
    let y1 = x01 + (x11 - x01) * fy;

    return y0 + (y1 - y0) * fz;
}

// Fractal Brownian motion stacker over `value_noise_3d`. Roughly [-1, 1].
//
// `octaves` is bounded at compile time inside the shader (WGSL has no
// f64 / loop bound on dynamic uniforms beyond what the compiler accepts);
// a max of 8 is enforced for safety. Match the per-octave sub-seeding
// scheme in noise.rs exactly: `osubseed = pcg_u32(seed + o)`.
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
