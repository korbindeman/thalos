// Generic sun-shadow-receiving StandardMaterial (thalos_body_render::craft::
// ShadowedStandardMaterial): stock Bevy PBR, then attenuate by the shared
// cascaded sun-shadow rig — so structures, the runway, plain craft parts, and
// the EVA capsule receive the SAME shadow world the terrain / trees / craft
// cast into (graphics-fidelity F6). The stock Bevy CSM on the sun light is
// disabled; this receive path is the only shadowing StandardMaterial surfaces
// get.
//
// A trimmed copy of `ship_part.wgsl` with the procedural panel/rivet layer
// removed. `apply_pbr_lighting` returns direct + indirect combined, so the
// direct sun term is isolated by a SECOND evaluation with the indirect
// occlusions zeroed and emissive removed — pure `exposure·direct` — and the
// shadow gates only that part (see `fragment`). The old whole-colour multiply
// needed a 0.4 floor to keep ambient readable, which made every shadow a pale
// grey wash; with the split, shadow kills the sun outright and the surface
// keeps its full ambient / env-map sky fill — deep, correctly-tinted shadows
// with no floor.

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
    // ── Direct/indirect split (exact, by linearity) ──────────────────────────
    // `apply_pbr_lighting` = exposure·(direct + indirect) + emissive, and the
    // indirect terms (flat ambient, env map, irradiance) are the ONLY ones
    // scaled by the occlusion inputs. So a second evaluation with occlusions
    // zeroed and emissive removed returns exactly exposure·direct — whatever
    // indirect stack the view carries, no reconstruction. The shadow then
    // subtracts only the direct share: full − (1 − s)·direct.
    var pbr_direct = pbr_input;
    pbr_direct.diffuse_occlusion = vec3<f32>(0.0);
    pbr_direct.specular_occlusion = 0.0;
    pbr_direct.material.emissive = vec4<f32>(0.0);
    let direct = apply_pbr_lighting(pbr_direct);
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
        max(out.color.rgb - (1.0 - shadow_f) * direct.rgb, vec3<f32>(0.0)),
        out.color.a,
    );
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

    return out;
}
