// Batched grass-blade shader.
//
// Vertex: standard mesh transform plus a world-space wind sway displacement,
// weighted by UV.x (0 at the root, 1 at the tip) and phase-shifted per blade
// by UV.y so the field doesn't move as one sheet.
//
// Fragment: wrap-diffuse direct sun plus the SAME hemisphere sky model the
// ground uses, both pulled from `thalos::lighting` (`compute_surface_sky` /
// `sky_ambient_irradiance`) so blades and the ground they grow from can't drift.
// The driver hands the blades the sun flux, radial up, and Rayleigh τ_v the
// model needs (see `GrassParams`). Distance fade is a screen-space-dithered
// `discard` in the opaque pass — no sorting, no blend state.

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}
#import thalos::lighting::{compute_surface_sky, SurfaceSky, FoliageSurface, shade_foliage, object_aerial_recession}
#import thalos::shadow::{ShadowCascadeBlock, sun_shadow_factor_vert}
#import thalos::grass_displace::grass_blade_world_pos

struct GrassParams {
    // xyz = unit direction toward the star (world render space), w = sun flux
    // (lux × exposure gain — the same value the terrain `SceneLighting` carries).
    sun_dir: vec4<f32>,
    // xyz = wind direction (world render space), w = tip sway amplitude (m).
    wind: vec4<f32>,
    // x = time (s), y = fade start (m), z = fade end (m), w unused.
    time_fade: vec4<f32>,
    // xyz = local radial up (world render space) for the sky hemisphere, w unused.
    sky_up: vec4<f32>,
    // xyz = Rayleigh vertical optical depth τ_v, w = atmosphere strength.
    sky_tau: vec4<f32>,
    // xyz = vegetation focus offset = (player craft − camera) in render space;
    // w = 1 valid / 0 = fade around the camera. The fade reference is rebuilt as
    // `view.world_position + offset`, i.e. the craft expressed in the CURRENT
    // frame's render origin. Passing an *offset* (not an absolute anchor) makes
    // it robust to big_space floating-origin recentres — see `vertex`.
    anchor: vec4<f32>,
}

// Standard MaterialPlugin bind group in Bevy 0.18: group 3.
@group(3) @binding(0) var<uniform> grass: GrassParams;

// Cascaded sun-shadow maps — the SAME depth maps the terrain + trees sample, so a
// tree's shadow falls on the grass beneath it. `ShadowCascadeBlock` + `sun_shadow_factor`
// are shared from `thalos::shadow`; sampled per-vertex (cheap on this
// overdraw-heavy material), so the depth bindings carry `vertex` visibility.
@group(3) @binding(1) var<uniform> grass_shadow: ShadowCascadeBlock;
@group(3) @binding(2) var sun_shadow_map_0: texture_depth_2d;
@group(3) @binding(3) var sun_shadow_map_1: texture_depth_2d;
@group(3) @binding(4) var sun_shadow_map_2: texture_depth_2d;

// Baked grass clump-card atlas (thalos_texgen::grass_card_atlas): variant cells
// side by side; A = coverage, RGB = tint modulation (linear, encoded ÷ the
// range below). Sampled only on CARD quads (far/mid rings).
@group(3) @binding(5) var grass_card_atlas: texture_2d<f32>;
@group(3) @binding(6) var grass_card_sampler: sampler;

// Mirrors thalos_texgen::GRASS_CARD_VARIANTS / GRASS_CARD_RGB_SCALE.
const GRASS_CARD_VARIANTS: f32 = 4.0;
const GRASS_CARD_RGB_SCALE: f32 = 1.35;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Blade ROOT in tile-local space (xyz; w unused), carried in the standard
    // tangent slot. Shared by every vertex of a blade — the clipmap height-fade
    // shrinks the blade uniformly toward this point (see `vertex`).
    @location(4) root: vec4<f32>,
    @location(5) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    // rgb = blade tint (linear); a unused.
    @location(2) color: vec4<f32>,
    // Per-vertex sun-shadow factor (sampled at this vertex's world position).
    @location(3) shadow: f32,
    // The sky/sun environment is a per-draw CONSTANT (built only from uniforms),
    // so `compute_surface_sky` — the one expensive call (exp/airmass integration)
    // — runs once per vertex and is flat-interpolated to the fragment instead of
    // recomputed per overdrawn pixel. The cheap part (the dot-product BRDF) stays
    // per-FRAGMENT on the interpolated normal, so sub-pixel blades shade smoothly
    // and don't alias to black the way full per-vertex (Gouraud) shading did.
    @location(4) @interpolate(flat) sky0: vec4<f32>, // sun_color.rgb, sun_scale
    @location(5) @interpolate(flat) sky1: vec4<f32>, // sky_radiance.rgb
    @location(6) @interpolate(flat) sky2: vec4<f32>, // ground_radiance.rgb
    // Clump-card data: x = across-card fraction, y = height fraction, z = 1 +
    // atlas variant on a CARD (far/mid rings) → the fragment samples the baked
    // card atlas and discards the gaps; 0 on a solid blade. (`root.w` carries
    // the card flag + variant.)
    @location(7) card: vec3<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    let world_pos_in =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(in.position, 1.0))
            .xyz;
    // Blade root in world space — the fixed point the height-fade shrinks toward.
    let root_pos =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(in.root.xyz, 1.0))
            .xyz;
    let world_normal = normalize(mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index));

    // Clipmap height-fade + altitude-collapse + wind sway. SHARED with the depth
    // prepass (`grass_prepass.wgsl`) through `thalos::grass_displace` so the two
    // passes produce identical clip depth — otherwise the prepass's pre-populated
    // depth early-Z-rejects visible blades. The fade is referenced to the craft
    // (anchor = craft−camera offset, origin-invariant across big_space recentres),
    // and a 0-height blade is a degenerate invisible sliver (seamless ring fade).
    let world_pos = grass_blade_world_pos(
        world_pos_in,
        root_pos,
        in.uv,
        grass.wind,
        grass.time_fade,
        grass.sky_up.w,
        grass.anchor,
        view.world_position,
    );

    // Build the sky/sun environment once here (it's all uniforms → constant across
    // the draw) and flat-pass it down, keeping the one expensive integration out of
    // the overdraw-heavy fragment path. The BRDF itself runs per-fragment (see
    // `fragment`) on the SAME atmosphere-derived environment the ground builds, so
    // grass tracks the ground through the day and gets the same blue-sky fill.
    let sun_dir = grass.sun_dir.xyz;
    let up = grass.sky_up.xyz;
    let sky = compute_surface_sky(grass.sky_tau.xyz, grass.sky_tau.w, up, sun_dir, grass.sun_dir.w);

    var out: VertexOutput;
    out.clip_position = position_world_to_clip(world_pos);
    out.world_position = world_pos;
    out.world_normal = world_normal;
    out.color = in.color;
    // Per-vertex sun-shadow: tree (and self) shadows on the grass, sampled at the
    // blade's final (swayed) world position and interpolated up the blade.
    out.shadow = sun_shadow_factor_vert(
        world_pos, grass_shadow, sun_shadow_map_0, sun_shadow_map_1, sun_shadow_map_2,
    );
    out.sky0 = vec4<f32>(sky.sun_color, sky.sun_scale);
    out.sky1 = vec4<f32>(sky.sky_radiance, 0.0);
    out.sky2 = vec4<f32>(sky.ground_radiance, 0.0);
    // For a card: uv.x = height, uv.y = across; root.w = 1 + variant marks it.
    // Blades carry root.w = 0 → card.z = 0 → fragment treats them as solid (no
    // atlas discard).
    out.card = vec3<f32>(in.uv.y, in.uv.x, in.root.w);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Clump cards (far/mid rings) sample the baked card atlas — a painted cluster
    // of layered blades — discarding the gaps so a single quad reads as a slice of
    // meadow, and modulating the per-clump tint by the texel (root-dark → bright
    // tips, per-blade hue drift — the same ramp the blade vertex colours carry).
    // Blades (card.z = 0) are solid and skip this. `textureSampleLevel` (the atlas
    // is mip-free) keeps the sample legal in non-uniform control flow.
    var albedo = in.color.rgb;
    if in.card.z > 0.5 {
        let variant = clamp(floor(in.card.z - 0.5), 0.0, GRASS_CARD_VARIANTS - 1.0);
        let uv = vec2<f32>(
            (variant + clamp(in.card.x, 0.0, 1.0)) / GRASS_CARD_VARIANTS,
            1.0 - clamp(in.card.y, 0.0, 1.0),
        );
        let texel = textureSampleLevel(grass_card_atlas, grass_card_sampler, uv, 0.0);
        if texel.a < 0.5 {
            discard;
        }
        albedo *= texel.rgb * GRASS_CARD_RGB_SCALE;
    }

    // Reconstruct the per-draw-constant sky the vertex shader integrated once.
    var sky: SurfaceSky;
    sky.sun_color = in.sky0.rgb;
    sky.sun_scale = in.sky0.a;
    sky.sky_radiance = in.sky1.rgb;
    sky.ground_radiance = in.sky2.rgb;

    // Blades carry the *terrain* normal (rounded across the width), so they light
    // like the ground they grow from. The BRDF is cheap (dot products) and runs
    // per-FRAGMENT on the interpolated normal — smooth across the blade, so
    // sub-pixel cards don't Gouraud-alias to black — while the costly sky model
    // was already paid once per vertex. Ground-matching hemisphere fill
    // (ambient_scale 1.0), no leaf-transmit term. No edge discard — the seamless
    // ring cross-fade is the vertex scale-fade, so there's nothing to cut here.
    let n = normalize(in.world_normal);
    let up = grass.sky_up.xyz;
    let sun_dir = grass.sun_dir.xyz;
    var s: FoliageSurface;
    s.albedo = albedo;
    s.normal_ws = n;
    s.translucency = 0.0;
    s.ambient_scale = 1.0;
    s.ambient_bleed = 0.0;
    let view_dir = normalize(view.world_position - in.world_position);
    var lit = shade_foliage(s, view_dir, up, sun_dir, sky, clamp(in.shadow, 0.0, 1.0));
    // Same distance recession as the trees (a no-op within the blade reach, but
    // keeps every object layer on one fade model).
    lit = object_aerial_recession(lit, sky, in.world_position, view.world_position);
    return vec4<f32>(lit, 1.0);
}
