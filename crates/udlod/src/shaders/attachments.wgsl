#define_import_path thalos_udlod::attachments

#import thalos_udlod::types::AtlasTile
#import thalos_udlod::bindings::{config, atlas_sampler, attachments, attachment0_atlas, attachment1_atlas, attachment2_atlas}
#import thalos_udlod::functions::tile_count

fn attachment_uv(uv: vec2<f32>, attachment_index: u32) -> vec2<f32> {
    let attachment = attachments[attachment_index];
    return uv * attachment.scale + attachment.offset;
}

fn sample_attachment0(tile: AtlasTile) -> vec4<f32> {
    let uv = attachment_uv(tile.coordinate.uv, 0u);

#ifdef FRAGMENT
#ifdef SAMPLE_GRAD
    return textureSampleGrad(attachment0_atlas, atlas_sampler, uv, tile.index, tile.coordinate.uv_dx, tile.coordinate.uv_dy);
#else
    return textureSampleLevel(attachment0_atlas, atlas_sampler, uv, tile.index, 0.0);
#endif
#else
    return textureSampleLevel(attachment0_atlas, atlas_sampler, uv, tile.index, 0.0);
#endif
}

fn sample_attachment1(tile: AtlasTile) -> vec4<f32> {
    let uv = attachment_uv(tile.coordinate.uv, 1u);

#ifdef FRAGMENT
#ifdef SAMPLE_GRAD
    return textureSampleGrad(attachment1_atlas, atlas_sampler, uv, tile.index, tile.coordinate.uv_dx, tile.coordinate.uv_dy);
#else
    return textureSampleLevel(attachment1_atlas, atlas_sampler, uv, tile.index, 0.0);
#endif
#else
    return textureSampleLevel(attachment1_atlas, atlas_sampler, uv, tile.index, 0.0);
#endif
}

fn sample_attachment1_gather0(tile: AtlasTile) -> vec4<f32> {
    let uv = attachment_uv(tile.coordinate.uv, 1u);
    return textureGather(0, attachment1_atlas, atlas_sampler, uv, tile.index);
}

fn decode_height_unit(height_sample: vec4<f32>) -> f32 {
    // R16 stores normalized height in x. Packed RG16 height attachments store
    // a filtered residual in y. R32Float height also works because y is zero.
    return height_sample.x + height_sample.y / 65535.0;
}

fn decode_height_m(height_sample: vec4<f32>) -> f32 {
    return mix(config.min_height, config.max_height, decode_height_unit(height_sample));
}

fn load_attachment0_texel_lod(layer: u32, texel: vec2<i32>, mip: i32) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(attachment0_atlas, mip));
    let p = clamp(texel, vec2<i32>(0), dims - vec2<i32>(1));
    return textureLoad(attachment0_atlas, p, i32(layer), mip);
}

fn sample_height_unit_atlas_uv_lod(
    tile: AtlasTile,
    atlas_uv: vec2<f32>,
    mip: i32,
) -> f32 {
    // RG16 height is a packed fixed-point value: x is the coarse UNORM16 and
    // y is a sub-LSB residual. Hardware linear filtering blends x and y before
    // decoding, which is invalid because y wraps from ~1 to 0 whenever x steps.
    // That creates false height terraces that become dark normal/shadow bands.
    // Load texels explicitly, decode each complete fixed-point value, then
    // bilinear-filter the decoded scalar height.
    let dims = vec2<f32>(textureDimensions(attachment0_atlas, mip));
    let p = atlas_uv * dims - vec2<f32>(0.5);
    let base = vec2<i32>(floor(p));
    let f = fract(p);

    let h00 = decode_height_unit(load_attachment0_texel_lod(tile.index, base + vec2<i32>(0, 0), mip));
    let h10 = decode_height_unit(load_attachment0_texel_lod(tile.index, base + vec2<i32>(1, 0), mip));
    let h01 = decode_height_unit(load_attachment0_texel_lod(tile.index, base + vec2<i32>(0, 1), mip));
    let h11 = decode_height_unit(load_attachment0_texel_lod(tile.index, base + vec2<i32>(1, 1), mip));

    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

fn sample_height_unit_atlas_uv(tile: AtlasTile, atlas_uv: vec2<f32>) -> f32 {
    return sample_height_unit_atlas_uv_lod(tile, atlas_uv, 0);
}

fn sample_height_atlas_uv_lod_m(tile: AtlasTile, atlas_uv: vec2<f32>, mip: i32) -> f32 {
    return mix(config.min_height, config.max_height, sample_height_unit_atlas_uv_lod(tile, atlas_uv, mip));
}

fn sample_height_atlas_uv_m(tile: AtlasTile, atlas_uv: vec2<f32>) -> f32 {
    return mix(config.min_height, config.max_height, sample_height_unit_atlas_uv(tile, atlas_uv));
}

fn sample_height(tile: AtlasTile) -> f32 {
    return sample_height_atlas_uv_m(tile, attachment_uv(tile.coordinate.uv, 0u));
}

fn sample_normal(tile: AtlasTile, vertex_normal: vec3<f32>) -> vec3<f32> {
    let uv = attachment_uv(tile.coordinate.uv, 0u);

#ifdef SPHERICAL
    var FACE_UP = array(
        vec3( 0.0, 1.0,  0.0),
        vec3( 0.0, 1.0,  0.0),
        vec3( 0.0, 0.0, -1.0),
        vec3( 0.0, 0.0, -1.0),
        vec3(-1.0, 0.0,  0.0),
        vec3(-1.0, 0.0,  0.0),
    );

    let face_up = FACE_UP[tile.coordinate.side];

    let normal = normalize(vertex_normal);
    var tangent = cross(face_up, normal);
    if (dot(tangent, tangent) < 1.0e-8) {
        var fallback_axis = vec3<f32>(0.0, 1.0, 0.0);
        if (abs(normal.y) > 0.9) {
            fallback_axis = vec3<f32>(1.0, 0.0, 0.0);
        }
        tangent = cross(fallback_axis, normal);
    }
    tangent = normalize(tangent);
    let bitangent = normalize(cross(normal, tangent));
    let TBN       = mat3x3(tangent, bitangent, normal);

    let side_length = 3.14159265359 / 4.0 * config.scale;
#else
    let TBN = mat3x3(1.0, 0.0, 0.0,
                     0.0, 0.0, 1.0,
                     0.0, 1.0, 0.0);

    let side_length = config.scale;
#endif

    // Todo: this is only an approximation of the S2 distance (pixels are not spaced evenly and they are not perpendicular)
    let pixels_per_side = attachments[0u].size * tile_count(tile.coordinate.lod);
    // A grazing ridge can compress many height texels into one screen pixel.
    // Differentiating mip 0 there aliases the relief normal into bright/dark
    // single-pixel facets (especially visible through Hapke on airless bodies).
    // Choose the normal's height mip from the atlas-UV derivatives, while the
    // vertex stage continues to sample mip 0 for full-resolution geometry.
#ifdef FRAGMENT
    let base_dims = vec2<f32>(textureDimensions(attachment0_atlas, 0));
    let dx_texels = dpdx(uv) * base_dims;
    let dy_texels = dpdy(uv) * base_dims;
    let footprint_texels = max(length(dx_texels), length(dy_texels));
    let max_mip = f32(textureNumLevels(attachment0_atlas) - 1u);
    let mip_f = clamp(floor(log2(max(footprint_texels, 1.0))), 0.0, max_mip);
    let mip = i32(mip_f);
    let mip_scale = exp2(mip_f);
#else
    let mip = 0;
    let mip_scale = 1.0;
#endif

    let distance_between_samples = side_length / pixels_per_side * mip_scale;
    let offset = 0.5 * mip_scale / attachments[0u].size;

    let left  = sample_height_atlas_uv_lod_m(tile, uv + vec2<f32>(-offset,     0.0), mip);
    let up    = sample_height_atlas_uv_lod_m(tile, uv + vec2<f32>(    0.0, -offset), mip);
    let right = sample_height_atlas_uv_lod_m(tile, uv + vec2<f32>( offset,     0.0), mip);
    let down  = sample_height_atlas_uv_lod_m(tile, uv + vec2<f32>(    0.0,  offset), mip);

    let surface_normal = normalize(vec3<f32>(left - right, down - up, distance_between_samples));

    return normalize(TBN * surface_normal);
}

fn sample_color(tile: AtlasTile) -> vec4<f32> {
    let height = sample_height_unit_atlas_uv(tile, attachment_uv(tile.coordinate.uv, 0u));

    return vec4<f32>(height * 0.5);
}
