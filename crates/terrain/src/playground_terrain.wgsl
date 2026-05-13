#import bevy_terrain::types::AtlasTile
#import bevy_terrain::bindings::config
#import bevy_terrain::attachments::{sample_height, sample_normal}
#import bevy_terrain::fragment::{FragmentInput, FragmentOutput, fragment_info, fragment_output, fragment_debug}
#import bevy_terrain::functions::lookup_tile

fn sample_color(tile: AtlasTile) -> vec4<f32> {
    let h = sample_height(tile);
    let span = max(config.max_height - config.min_height, 1.0);
    let t = clamp((h - config.min_height) / span, 0.0, 1.0);
    let gray = 0.18 + 0.78 * t;
    return vec4<f32>(gray, gray, gray, 1.0);
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
