// Screen-space ambient occlusion (graphics-fidelity F5).
//
// Half-resolution hemisphere SSAO computed from the copied scene depth
// (`rendering::scene_depth`'s `SceneDepthImage`), which — unlike Bevy's depth
// prepass — sees the forked-udlod terrain. Runs as a fullscreen pass AFTER the
// opaque main pass (once the depth copy is populated); the result is sampled by
// the terrain material one frame later (1-frame latency, invisible at planet-cam
// speeds) and multiplied into its AMBIENT occlusion only. See `rendering::ssao`.
//
// Everything is in VIEW space (camera-relative render metres), so it is f32-safe
// under big_space's floating origin — no planet-centric magnitudes enter here.
// Bevy uses reverse-Z: NDC depth 1 = near, 0 = far; a sky/far pixel reads depth
// ≤ 0 and is left unoccluded.

struct AoUniform {
    // clip → view (inverse projection): reconstruct a view-space position from
    // a screen UV + sampled NDC depth.
    view_from_clip: mat4x4<f32>,
    // view → clip (projection): re-project a view-space sample point to a screen
    // UV to look up the scene depth there.
    clip_from_view: mat4x4<f32>,
    // AO target (half-res) size in pixels; xy used, zw padding.
    target_res: vec4<f32>,
    // x = radius (view/render units ≈ metres), y = depth bias, z = intensity,
    // w = contrast power.
    params: vec4<f32>,
}

@group(0) @binding(0) var scene_depth: texture_depth_2d;
@group(0) @binding(1) var<uniform> ao: AoUniform;

const SAMPLE_COUNT: u32 = 16u;
const GOLDEN_ANGLE: f32 = 2.399963229728653;
const TAU: f32 = 6.283185307179586;

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

/// Reconstruct a view-space position from a screen UV by sampling the (full-res)
/// scene depth there. Returns `w = 0` for sky/far pixels (depth ≤ 0).
fn view_pos_at(uv: vec2<f32>) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(scene_depth));
    let coord = vec2<i32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(0.99999)) * dims);
    let d = textureLoad(scene_depth, coord, 0);
    if d <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0); // sky / far plane
    }
    // NDC: uv.y is top-down, NDC y is bottom-up → flip.
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d);
    let v = ao.view_from_clip * vec4<f32>(ndc, 1.0);
    return vec4<f32>(v.xyz / v.w, 1.0);
}

/// The `i`-th of `n` tangent-space hemisphere directions (+z = normal), lengths
/// biased toward the origin so near-field contact dominates the occlusion.
fn hemisphere_sample(i: u32, n: u32) -> vec3<f32> {
    let fi = f32(i);
    let t = (fi + 0.5) / f32(n);
    let phi = fi * GOLDEN_ANGLE;
    let cos_t = 1.0 - t;                      // z ∈ (0, 1], cosine-ish
    let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
    let dir = vec3<f32>(cos(phi) * sin_t, sin(phi) * sin_t, cos_t);
    let scale = mix(0.1, 1.0, t * t);         // cluster near the centre
    return dir * scale;
}

/// Interleaved gradient noise for a per-pixel rotation that breaks up the fixed
/// kernel's banding (in lieu of a separate blur pass — a noted follow-up).
fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.position.xy / ao.target_res.xy;

    let center = view_pos_at(uv);
    if center.w == 0.0 {
        return vec4<f32>(1.0, 1.0e9, 0.0, 1.0); // sky → no occlusion, far depth
    }
    let p = center.xyz;

    // Contact AO only: a ~1 m occlusion radius is sub-pixel past a couple hundred
    // metres, so fade AO out with distance — removes far-field reconstruction
    // noise AND skips the sample loop for most of a landscape view.
    let dist = -p.z;
    let far_fade = smoothstep(350.0, 200.0, dist); // 1 near → 0 far
    if far_fade <= 0.0 {
        return vec4<f32>(1.0, dist, 0.0, 1.0);
    }

    // Reconstruct the view-space normal from symmetric central differences.
    // (An earlier version picked the nearer neighbour per axis with `select` to
    // avoid silhouette bleed, but that branch flips row-to-row on grazing ground
    // and imprinted horizontal stripes; symmetric differences are smooth on
    // planes, and the ambient-only application tolerates the rare silhouette
    // texel — the blur pass evens it out.)
    let full = vec2<f32>(textureDimensions(scene_depth));
    let texel = 1.0 / full;
    let pr = view_pos_at(uv + vec2<f32>(texel.x, 0.0));
    let pl = view_pos_at(uv - vec2<f32>(texel.x, 0.0));
    let pu = view_pos_at(uv + vec2<f32>(0.0, texel.y));
    let pd = view_pos_at(uv - vec2<f32>(0.0, texel.y));
    if pr.w == 0.0 || pl.w == 0.0 || pu.w == 0.0 || pd.w == 0.0 {
        return vec4<f32>(1.0, dist, 0.0, 1.0); // silhouette against sky
    }
    let ddx = pr.xyz - pl.xyz;
    let ddy = pu.xyz - pd.xyz;
    var n = normalize(cross(ddx, ddy));
    // Face the camera (view dir toward origin = -p).
    if dot(n, normalize(-p)) < 0.0 {
        n = -n;
    }

    // Tangent frame rotated per-pixel.
    let angle = ign(in.position.xy) * TAU;
    let rvec = vec3<f32>(cos(angle), sin(angle), 0.0);
    let t = normalize(rvec - n * dot(rvec, n));
    let b = cross(n, t);
    let tbn = mat3x3<f32>(t, b, n);

    let radius = ao.params.x;
    // Depth-slope-relative bias: on a grazing surface the plane recedes fast in
    // screen space, so a fixed bias lets the plane self-occlude (a smooth AO
    // gradient that reads as banding on flat ground). Widen the bias by the local
    // view-depth change per texel so a flat plane reads unoccluded regardless of
    // viewing angle, while a surface facing the camera (slope ≈ 0) keeps full AO.
    // The distance-relative term floors the bias above the depth-reconstruction
    // noise (which grows with distance): top-down flat ground has near-zero slope
    // AND near-zero real occlusion deltas, so without it the comparison operates
    // at the f32 noise floor and imprints faint bands.
    let slope = max(abs(pr.z - pl.z), abs(pu.z - pd.z));
    let bias = ao.params.y + slope + 2.0e-4 * dist;

    var occ = 0.0;
    for (var i: u32 = 0u; i < SAMPLE_COUNT; i = i + 1u) {
        let sv = p + (tbn * hemisphere_sample(i, SAMPLE_COUNT)) * radius;
        // Project the view-space sample point to a screen UV.
        let clip = ao.clip_from_view * vec4<f32>(sv, 1.0);
        if clip.w <= 0.0 {
            continue;
        }
        let sndc = clip.xyz / clip.w;
        let suv = vec2<f32>(sndc.x * 0.5 + 0.5, 1.0 - (sndc.y * 0.5 + 0.5));
        if suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0 {
            continue;
        }
        let scene = view_pos_at(suv);
        if scene.w == 0.0 {
            continue; // sky behind the sample → not an occluder
        }
        // View z is negative forward; the scene surface occludes the sample when
        // it is CLOSER to the camera (greater z) than the sample, beyond bias.
        let occluded = scene.z >= sv.z + bias;
        // Fade occlusion by distance so far geometry doesn't over-darken.
        let range = smoothstep(0.0, 1.0, radius / max(abs(p.z - scene.z), 1e-4));
        occ += select(0.0, range, occluded);
    }

    let intensity = ao.params.z;
    let power = ao.params.w;
    var visibility = 1.0 - (occ / f32(SAMPLE_COUNT)) * intensity;
    visibility = pow(clamp(visibility, 0.0, 1.0), power);
    // Fade to unoccluded with distance (see far_fade above).
    visibility = mix(1.0, visibility, far_fade);
    // G carries view-space distance for the depth-aware blur's edge weights.
    return vec4<f32>(visibility, dist, 0.0, 1.0);
}
