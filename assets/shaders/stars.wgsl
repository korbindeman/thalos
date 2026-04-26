//! Forward-rendered star shader.
//!
//! Each star is a small camera-facing quad. The vertex stage places
//! the center at a virtual "infinity" distance along the star's
//! direction unit vector and offsets each corner in screen space so
//! the quad occupies a fixed pixel radius regardless of how far away
//! the star "is". Z is forced to the far plane so stars never fight
//! with solar-system bodies in the depth buffer.
//!
//! The fragment stage synthesises a cheap but layered point spread
//! function: Gaussian core + radial tail + two diffraction spikes
//! crossed at the horizontal and vertical axes. The output is
//! additively blended into the HDR scene — no tonemapping happens
//! here; that belongs to the post pipeline.

#import bevy_pbr::mesh_view_bindings::view

struct StarsParams {
    // Half-width of a magnitude-0 star's quad in pixels.
    pixel_radius: f32,
    // Overall flux multiplier applied to every star.
    brightness:   f32,
    // Per-star size exponent: larger → bright stars grow more.
    size_gamma:   f32,
    _pad0:        f32,
}

@group(3) @binding(0) var<uniform> params: StarsParams;

struct VertexInput {
    @location(0) direction: vec3<f32>,   // star direction, unit length
    @location(1) corner:    vec2<f32>,   // quad corner, in {-1, +1}²
    @location(2) color:     vec4<f32>,   // linear rgb, a = magnitude flux
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Quad-local coordinates in the scaled frame the PSF is drawn in.
    @location(0) quad_uv: vec2<f32>,
    @location(1) color:   vec4<f32>,
    // Per-star angular scale ratio so the fragment PSF can shrink
    // spikes / tails for dim stars (makes them look more point-like).
    @location(2) size_scale: f32,
}

const INFINITY_DISTANCE: f32 = 1.0e10;

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    // Star sits at a large fixed distance along its direction. The
    // magnitude doesn't affect position — all 4 quad corners share
    // the same center and are billboarded in screen space below.
    let world_pos = view.world_position + in.direction * INFINITY_DISTANCE;
    let center_clip = view.clip_from_world * vec4<f32>(world_pos, 1.0);

    // Per-star radius in pixels: the alpha channel carries magnitude
    // flux (10^(-0.4 m)). Take a soft log so the brightest few stars
    // grow larger than the Pogson ratio would otherwise imply without
    // pushing the faintest into sub-pixel oblivion.
    let flux = max(in.color.a, 1.0e-4);
    let scale = pow(clamp(flux, 0.001, 100.0), params.size_gamma * 0.5);
    let radius_px = params.pixel_radius * max(scale, 0.20);

    // Screen-space offset expressed in clip units. Multiplying by the
    // center clip w pre-compensates the perspective divide the GPU
    // will do on the final position.
    let viewport = view.viewport.zw;
    let ndc_per_pixel = vec2<f32>(2.0 / viewport.x, 2.0 / viewport.y);
    let offset_clip = in.corner * radius_px * ndc_per_pixel * center_clip.w;

    // NDC z = 0 is the reverse-Z far plane — the projective expression
    // of "at infinity." Paired with `CompareFunction::GreaterEqual` in
    // `StarsMaterial::specialize`, fragments pass against the cleared
    // depth buffer (also 0) but fail against any real body, whose NDC z
    // is strictly positive. A previous `1e-7 * w` offset worked under
    // map-view scale but broke in ship view, where planet NDC z can dip
    // below 1e-7 and let stars bleed through.
    var out: VertexOutput;
    out.clip_position = vec4<f32>(
        center_clip.xy + offset_clip,
        0.0,
        center_clip.w,
    );
    out.quad_uv = in.corner;
    out.color = in.color;
    out.size_scale = max(scale, 0.20);
    return out;
}

// Procedural HDR point-spread function. Units: the quad frame is
// [-1, 1]², centre at origin. The combination of a tight Gaussian
// core, a radial tail, and two thin diffraction bars produces the
// "bright point with subtle glare" look without needing a texture.
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = in.quad_uv;
    let r = length(p);

    // Gaussian core. Wider than a single pixel so post-process CAS
    // sharpening can't re-crisp the spot into a pixel-perfect dot.
    let core_sigma = 0.33;
    let core = exp(-(r * r) / (2.0 * core_sigma * core_sigma));

    // Radial 1/r² tail, falling to zero at the quad edge.
    let edge_falloff = smoothstep(1.05, 0.2, r);
    let tail = edge_falloff * 0.02 / (0.01 + r * r);

    // Two crossed diffraction spikes. Each spike is a very thin
    // Gaussian bar in one axis modulated by a smooth radial mask.
    let spike_width = 0.035;
    let spike_mask = smoothstep(1.1, 0.0, r);
    let hspike = exp(-(p.y * p.y) / (2.0 * spike_width * spike_width)) * spike_mask;
    let vspike = exp(-(p.x * p.x) / (2.0 * spike_width * spike_width)) * spike_mask;
    // Spikes fade out for dim stars — only the brightest produce glare.
    let spike_gain = smoothstep(0.6, 1.4, in.size_scale);
    let spikes = (hspike + vspike) * 0.22 * spike_gain;

    let intensity = core + tail + spikes;
    let flux = in.color.a;
    let rgb = in.color.rgb * intensity * flux * params.brightness;
    return vec4<f32>(rgb, 0.0);
}
