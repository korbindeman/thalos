#import bevy_terrain::types::AtlasTile
#import bevy_terrain::bindings::config
#import bevy_terrain::attachments::{sample_height, sample_normal, sample_attachment1}
#import bevy_terrain::fragment::{FragmentInput, FragmentOutput, fragment_info, fragment_output, fragment_debug}
#import bevy_terrain::functions::lookup_tile

fn sample_color(tile: AtlasTile) -> vec4<f32> {
    // Attachment 1 is RGBA8 albedo encoded sRGB on the CPU side; the texture
    // is uploaded as Rgba8UnormSrgb so the sampler returns linear values.
    return sample_attachment1(tile);
}

@fragment
fn fragment(input: FragmentInput) -> FragmentOutput {
    var info   = fragment_info(input);
    let tile   = lookup_tile(info.coordinate, info.blend, 0u);
    var color  = sample_color(tile);
    var normal = sample_normal(tile, info.world_normal);

    if (info.blend.ratio > 0.0) {
        let tile2 = lookup_tile(info.coordinate, info.blend, 1u);
        color     = mix(color,  sample_color(tile2),                     info.blend.ratio);
        normal    = mix(normal, sample_normal(tile2, info.world_normal), info.blend.ratio);
    }

    var output: FragmentOutput;
    fragment_output(&info, &output, color, normal);
    fragment_debug(&info, &output, tile, normal);
    return output;
}
