// NTR-X1 tile-terrain surface material (ExtendedMaterial fragment).
//
// The keystone rule ("one lighting universe = Bevy's") shapes this shader:
// both branches below are lit by Bevy's own directional light + per-camera
// ambient + view exposure — no SceneLighting binding, no custom flux units.
// Only the *BRDF* changes per body style:
//
//   style 0 — stock PBR (`apply_pbr_lighting`), for vegetated / generic
//             bodies once they move onto the tile path.
//   style 1 — Hapke regolith: the shared `thalos::lighting` Hapke lobe
//             (opposition surge + backscatter + Chandrasekhar H), driven by
//             Bevy's sun. This is what keeps airless ground from reading as
//             waxy plastic under the standard path, and lets tiles
//             reconverge with the impostor's Hapke look across the swap.
//
// Both branches receive the shared `thalos::shadow` cascade (bindings mirror
// `ShadowReceiveExtension` / `shadowed_standard.wgsl`) so craft / structure
// shadows land on tile ground exactly as they do on udlod ground.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}
#import bevy_pbr::mesh_view_bindings::{view, lights}
#import thalos::shadow::{ShadowCascadeBlock, sun_shadow_factor_nrm}
#import thalos::lighting::{hapke_brdf_rgb, hapke_w_from_albedo}

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

struct TileShadingParams {
    // 1 = Hapke regolith (airless), 0 = stock PBR. u32 for tight mirroring
    // with the Rust `TileShadingParams`.
    style: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> recv_shadow: ShadowCascadeBlock;
@group(#{MATERIAL_BIND_GROUP}) @binding(101)
var recv_shadow_map_0: texture_depth_2d;
@group(#{MATERIAL_BIND_GROUP}) @binding(102)
var recv_shadow_map_1: texture_depth_2d;
@group(#{MATERIAL_BIND_GROUP}) @binding(103)
var recv_shadow_map_2: texture_depth_2d;
@group(#{MATERIAL_BIND_GROUP}) @binding(104)
var<uniform> tile_params: TileShadingParams;

// Same floor as `shadowed_standard.wgsl` / the hull, so every standard-path
// surface darkens identically in shadow.
const SHADOW_FLOOR: f32 = 0.4;

// Hapke → photometric bridge. `hapke_brdf_rgb` returns a radiance factor with
// the 1/(4π)-family normalisation deliberately left to the call site (see the
// library note); dividing by π puts a nadir-lit, w≈albedo surface in the same
// brightness family as Bevy's Lambert diffuse (albedo/π · n·l), so the sun's
// authored illuminance + camera exposure keep meaning what they mean. The
// remaining deviation from Lambert IS the regolith look (opposition surge,
// limb flattening), not a brightness error. Calibrate against the impostor by
// capture, not by re-tuning the light.
const HAPKE_PHOTOMETRIC_SCALE: f32 = 0.3183098862; // 1/π

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
    // Geometric world normal drives the stable-CSM receiver offset (a
    // perturbed N would wobble it) — same rule as every other receiver.
    let n = normalize(pbr_input.world_normal);
    let shadow_f = sun_shadow_factor_nrm(
        pbr_input.world_position.xyz,
        n,
        recv_shadow,
        recv_shadow_map_0,
        recv_shadow_map_1,
        recv_shadow_map_2,
    );

    if tile_params.style == 1u && lights.n_directional_lights > 0u {
        let albedo = pbr_input.material.base_color.rgb;
        let v = pbr_input.V;
        let n_dot_v = max(dot(n, v), 1.0e-3);
        let w = hapke_w_from_albedo(albedo);
        // Sum ALL directional lights (the scene carries sun + moonlight; the
        // sun's slot index is not guaranteed).
        var direct = vec3<f32>(0.0);
        let n_lights = min(lights.n_directional_lights, 4u);
        for (var i = 0u; i < n_lights; i = i + 1u) {
            let light = lights.directional_lights[i];
            let l = light.direction_to_light;
            let r = hapke_brdf_rgb(
                max(dot(n, l), 0.0),
                n_dot_v,
                dot(v, l),
                pbr_input.material.perceptual_roughness,
                w,
            );
            direct += r * HAPKE_PHOTOMETRIC_SCALE * light.color.rgb;
        }
        // Ambient stays keyed to the flat albedo (it stands in for fill the
        // lobe doesn't model — same rule as `shade_hapke_surface`). Airless
        // bodies have no atmosphere fill, so this is the scene's authored
        // space/starlight term.
        direct *= max(shadow_f, SHADOW_FLOOR);
        let ambient = albedo * lights.ambient_color.rgb;
        out.color = vec4<f32>(
            (direct + ambient) * view.exposure,
            pbr_input.material.base_color.a,
        );
        out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    } else {
        out.color = apply_pbr_lighting(pbr_input);
        out.color = vec4<f32>(
            out.color.rgb * max(shadow_f, SHADOW_FLOOR),
            out.color.a,
        );
        out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    }
#endif

    return out;
}
