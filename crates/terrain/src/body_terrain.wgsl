#import bevy_terrain::types::AtlasTile
#import bevy_terrain::bindings::config
#import bevy_terrain::attachments::{sample_height, sample_normal, sample_attachment1}
#import bevy_terrain::fragment::{FragmentInput, FragmentOutput, fragment_info}
#import bevy_terrain::functions::lookup_tile

fn sample_color(tile: AtlasTile) -> vec4<f32> {
    // Attachment 1 is RGBA8 albedo encoded sRGB on the CPU side; the texture
    // is uploaded as Rgba8UnormSrgb so the sampler returns linear values.
    return sample_attachment1(tile);
}

// M3-era body terrain: write the cubemap albedo directly to the framebuffer.
//
// We intentionally skip `bevy_terrain::fragment::fragment_output` (which runs
// PBR under `#ifdef LIGHTING`) and `fragment_debug` (which darkens near
// fragments by 70% as a precision-debug overlay) — the fork's `queue_terrain`
// forces `LIGHTING` and the debug helper on by default, and at this stage
// our terrain attachments are not authored to support PBR (normal-map encoding
// is placeholder, there's no proper sun-light setup for SHIP_LAYER yet).
// Writing the raw albedo gives a visually correct flat surface that matches
// what the impostor's baked albedo cubemap shows from orbit.
//
// PBR + sun direction + atmospheric scattering on the terrain land in M4
// alongside the impostor↔terrain opacity crossfade; at that point the
// shader either grows a real `apply_pbr_lighting` path or hooks into the
// shared Hapke BRDF the impostor already uses.
@fragment
fn fragment(input: FragmentInput) -> FragmentOutput {
    var info  = fragment_info(input);
    let tile  = lookup_tile(info.coordinate, info.blend, 0u);
    var color = sample_color(tile);

    if (info.blend.ratio > 0.0) {
        let tile2 = lookup_tile(info.coordinate, info.blend, 1u);
        color = mix(color, sample_color(tile2), info.blend.ratio);
    }

    var output: FragmentOutput;
    output.color = color;
    return output;
}
