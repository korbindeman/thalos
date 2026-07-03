// Generic sun-shadow-receiving StandardMaterial (thalos_body_render::craft::
// ShadowedStandardMaterial): stock Bevy PBR, then attenuate by the shared
// cascaded sun-shadow rig — so structures, the runway, plain craft parts, and
// the EVA capsule receive the SAME shadow world the terrain / trees / craft
// cast into (graphics-fidelity F6). The stock Bevy CSM on the sun light is
// disabled; this receive path is the only shadowing StandardMaterial surfaces
// get.
//
// A trimmed copy of `ship_part.wgsl` with the procedural panel/rivet layer
// removed. Like the hull, `apply_pbr_lighting` returns direct + ambient
// combined, so we can't gate only the sun term yet (F8 ports these surfaces
// onto `shade_surface` for that) — instead attenuate toward a floor so a
// shadowed surface keeps its ambient / IBL fill rather than going black.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}

// Shared cascaded sun-shadow sampler (registered by
// `body_render::shading::PlanetLightingPlugin`).
#import thalos::shadow::{ShadowCascadeBlock, sun_shadow_factor_nrm}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

// Sun-shadow cascade — the RECEIVING side. `gate.x == 0` (off-surface) makes
// the sampler skip entirely. Bindings mirror `ShadowReceiveExtension` in
// `body_render::craft` (uniform 100, depth maps 101–103).
@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> recv_shadow: ShadowCascadeBlock;
@group(#{MATERIAL_BIND_GROUP}) @binding(101)
var recv_shadow_map_0: texture_depth_2d;
@group(#{MATERIAL_BIND_GROUP}) @binding(102)
var recv_shadow_map_1: texture_depth_2d;
@group(#{MATERIAL_BIND_GROUP}) @binding(103)
var recv_shadow_map_2: texture_depth_2d;

// How dark a fully-shadowed surface gets — keeps the ambient / IBL fill (see
// header). Matches `CRAFT_SHADOW_FLOOR` in `ship_part.wgsl` so hull and
// structures darken identically.
const SHADOW_FLOOR: f32 = 0.4;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    // Receive the shared sun-shadow cascade. The geometric world normal drives
    // the stable-CSM receiver offset + slope-scaled bias (normal-mapped N would
    // wobble the offset).
    let shadow_f = sun_shadow_factor_nrm(
        pbr_input.world_position.xyz,
        normalize(pbr_input.world_normal),
        recv_shadow,
        recv_shadow_map_0,
        recv_shadow_map_1,
        recv_shadow_map_2,
    );
    out.color = vec4<f32>(
        out.color.rgb * max(shadow_f, SHADOW_FLOOR),
        out.color.a,
    );
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

    return out;
}
