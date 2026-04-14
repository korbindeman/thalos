#import bevy_ui::ui_vertex_output::UiVertexOutput

struct LensFlareUniform {
    tint: vec4<f32>,
};

@group(1) @binding(0) var<uniform> material: LensFlareUniform;
@group(1) @binding(1) var flare_tex: texture_2d<f32>;
@group(1) @binding(2) var flare_sampler: sampler;

@fragment
fn fragment(in: UiVertexOutput) -> @location(0) vec4<f32> {
    let tex = textureSample(flare_tex, flare_sampler, in.uv);
    // tex.rgb carries per-channel shape (enables baked chromatic aberration).
    let rgb = tex.rgb * material.tint.rgb * material.tint.a;
    return vec4<f32>(rgb, 0.0);
}
