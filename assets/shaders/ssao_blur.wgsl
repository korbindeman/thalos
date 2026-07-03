// Depth-aware 3×3 blur for the SSAO target (graphics F5).
//
// Standard SSAO noise resolve: the raw pass dithers its sample kernel per pixel
// (IGN rotation), which trades banding for high-frequency noise — this pass
// averages that noise away. Weights are gaussian × depth-similarity, so the blur
// never bleeds occlusion across depth discontinuities (object silhouettes).
//
// Input:  raw AO (RG16Float — R = visibility, G = view-space distance).
// Output: resolved AO (R16Float — what the terrain samples).

@group(0) @binding(0) var ao_raw: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Fullscreen triangle: (-1,-1), (3,-1), (-1,3).
    let uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(ao_raw));
    let cc = vec2<i32>(in.position.xy);
    let center = textureLoad(ao_raw, clamp(cc, vec2<i32>(0), dims - 1), 0);
    let center_dist = max(center.g, 1e-3);

    // Gaussian 5×5 ((1,4,6,4,1) outer product) × depth-similarity. 5-wide (10 px
    // at full res) because the raw pass's residual noise showed row-scale
    // structure a 3×3 couldn't fully resolve.
    var sum = 0.0;
    var wsum = 0.0;
    for (var dy: i32 = -2; dy <= 2; dy = dy + 1) {
        for (var dx: i32 = -2; dx <= 2; dx = dx + 1) {
            let coord = clamp(cc + vec2<i32>(dx, dy), vec2<i32>(0), dims - 1);
            let s = textureLoad(ao_raw, coord, 0);
            let gx = gauss5(dx);
            let gy = gauss5(dy);
            // Relative depth similarity: reject neighbours whose view distance
            // differs by more than a few % of the centre's (a depth edge).
            let w_d = exp(-abs(s.g - center_dist) / (0.03 * center_dist + 0.05));
            let w = gx * gy * w_d;
            sum = sum + s.r * w;
            wsum = wsum + w;
        }
    }

    let resolved = sum / max(wsum, 1e-4);
    return vec4<f32>(resolved, resolved, resolved, 1.0);
}

/// Binomial (1,4,6,4,1) weight for offset `d` ∈ [-2, 2].
fn gauss5(d: i32) -> f32 {
    let a = abs(d);
    if a == 0 { return 6.0; }
    if a == 1 { return 4.0; }
    return 1.0;
}
