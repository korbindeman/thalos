// Screen-space contact shadows (graphics-fidelity W18a).
//
// The CONTACT tier of the three-tier shadow split
// (ADR-20260722T111848Z-shadows-three-tier-not-virtual-shadow-maps): cascade 0
// is a 400 m half-extent at 4096² ≈ 0.2 m/texel, which is coarser than a landing
// gear strut, a tree trunk, or a building's ground seam. The near field is a
// regime the cascade rig structurally cannot serve, so it gets its own
// mechanism: a short march through the copied scene depth toward the sun.
//
// Reads the current frame's Bevy depth prepass and runs before opaque shading.
// Everything runs in VIEW space (camera-relative render metres), so it is
// f32-safe under big_space's floating origin.
//
// FULL RESOLUTION, unlike SSAO's half-res target. AO is a low-frequency field
// and tolerates upsampling; a contact shadow is inherently high-frequency — the
// casters that matter (gear struts, trunks) are a few pixels wide, so half-res
// would alias away exactly the detail this pass exists to produce.
//
// Bevy uses reverse-Z: NDC depth 1 = near, 0 = far; a sky/far pixel reads
// depth ≤ 0 and is left fully lit.

struct ContactUniform {
    // clip → view (inverse projection): reconstruct a view-space position from
    // a screen UV + sampled NDC depth.
    view_from_clip: mat4x4<f32>,
    // view → clip (projection): re-project a marched point to a screen UV.
    clip_from_view: mat4x4<f32>,
    // Target size in pixels (xy); zw padding.
    target_res: vec4<f32>,
    // x = march reach (view metres), y = occluder thickness (view metres),
    // z = shadow strength in [0,1], w = receiver normal bias (view metres).
    params: vec4<f32>,
    // xyz = normalized VIEW-space direction toward the sun,
    // w = distance (view metres) at which the effect has fully faded out.
    sun_view: vec4<f32>,
}

#ifdef CONTACT_DEPTH_MSAA
@group(0) @binding(0) var scene_depth: texture_depth_multisampled_2d;
#else
@group(0) @binding(0) var scene_depth: texture_depth_2d;
#endif
@group(0) @binding(1) var<uniform> cs: ContactUniform;

// Fixed step count. Short marches with few steps are the whole point — this is a
// contact term, not a screen-space shadow solver.
const STEP_COUNT: u32 = 12u;

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

/// Reconstruct a view-space position from a screen UV by sampling the scene
/// depth there. Returns `w = 0` for sky / far-plane pixels.
fn view_pos_at(uv: vec2<f32>) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(scene_depth));
    let coord = vec2<i32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(0.99999)) * dims);
    let d = textureLoad(scene_depth, coord, 0);
    if d <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0); // sky / far plane
    }
    // NDC: uv.y is top-down, NDC y is bottom-up → flip.
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d);
    let v = cs.view_from_clip * vec4<f32>(ndc, 1.0);
    return vec4<f32>(v.xyz / v.w, 1.0);
}

/// Interleaved gradient noise — jitters each pixel's first step so the fixed
/// step count doesn't band into visible rings around a caster's contact point.
fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.position.xy / cs.target_res.xy;

    let center = view_pos_at(uv);
    if center.w == 0.0 {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0); // sky → lit
    }
    let p = center.xyz;
    let dist = -p.z;

    // A sub-metre reach is sub-pixel past a short distance, and beyond that the
    // cascades own the shadow anyway. Fading out here also skips the march for
    // most of a landscape view.
    let fade_end = cs.sun_view.w;
    let far_fade = smoothstep(fade_end, fade_end * 0.6, dist); // 1 near → 0 far
    if far_fade <= 0.0 {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }

    // Reconstruct the view-space normal from symmetric central differences (the
    // same estimator `ssao.wgsl` settled on — asymmetric neighbour picking flips
    // row-to-row on grazing ground and imprints stripes).
    let full = vec2<f32>(textureDimensions(scene_depth));
    let texel = 1.0 / full;
    let pr = view_pos_at(uv + vec2<f32>(texel.x, 0.0));
    let pl = view_pos_at(uv - vec2<f32>(texel.x, 0.0));
    let pu = view_pos_at(uv + vec2<f32>(0.0, texel.y));
    let pd = view_pos_at(uv - vec2<f32>(0.0, texel.y));
    if pr.w == 0.0 || pl.w == 0.0 || pu.w == 0.0 || pd.w == 0.0 {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0); // silhouette against sky
    }
    var n = normalize(cross(pr.xyz - pl.xyz, pu.xyz - pd.xyz));
    if dot(n, normalize(-p)) < 0.0 {
        n = -n;
    }

    let sun = cs.sun_view.xyz;
    let ndl = dot(n, sun);
    // Facing away from the sun: the BRDF's own n·l already unlits this fragment,
    // and marching from a back-facing surface only produces self-occlusion.
    if ndl <= 0.0 {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }

    let reach = cs.params.x;
    let thickness = cs.params.y;
    let strength = cs.params.z;
    let normal_bias = cs.params.w;

    // Lift the ray off the receiver along its normal, scaled up as the sun
    // grazes: at low n·l the depth-reconstructed surface and the true surface
    // diverge most, which is exactly when self-occlusion appears.
    let origin = p + n * (normal_bias * (1.0 + 2.0 * (1.0 - ndl)));
    let jitter = ign(in.position.xy);
    let step = reach / f32(STEP_COUNT);

    var occluded = 0.0;
    for (var i: u32 = 0u; i < STEP_COUNT; i = i + 1u) {
        let t = (f32(i) + jitter) * step;
        let ray_p = origin + sun * t;
        let clip = cs.clip_from_view * vec4<f32>(ray_p, 1.0);
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
            continue; // sky behind the ray → not an occluder
        }
        // View z is negative forward, so the scene surface is IN FRONT of the
        // ray sample when its z is greater. `depth` is how far in front.
        let depth = scene.z - ray_p.z;
        // Thickness test: an occluder is only real if the ray passes just behind
        // it. Without this, distant background geometry (a mountain far beyond
        // the receiver) shadows everything drawn in front of it.
        if depth > 0.0 && depth < thickness {
            // Soften with march distance so the shadow hardens at the contact
            // point and dissipates toward the reach limit — a cheap stand-in for
            // the real penumbra W18c will compute.
            occluded = 1.0 - smoothstep(0.0, reach, t);
            break;
        }
    }

    let visibility = 1.0 - occluded * strength * far_fade;
    return vec4<f32>(clamp(visibility, 0.0, 1.0), 0.0, 0.0, 1.0);
}
