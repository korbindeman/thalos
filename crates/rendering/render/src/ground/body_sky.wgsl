// Unified atmosphere fullscreen pass per body.
//
// Renders one fullscreen quad per body (per camera) that integrates
// single-scattering Rayleigh + Mie atmospheric scattering for every view
// ray. The integration interval is clipped by both the body's atmosphere
// shell and the scene depth from `scene_depth_texture` (a per-frame copy
// of the main pass's depth attachment maintained by
// `thalos_render_foundation`). `BodySkyMaterial` compiles this source with
// `ATMOSPHERE_ONLY`; `BodyOceanMaterial` compiles it with `OCEAN_ONLY`.
// Atmosphere, ocean, and clouds therefore have separate render ownership while
// retaining one optical/signed-field implementation.
//
//   * Mid  — camera outside the shell but terrain is visible. The pass
//     produces in-front haze/clouds on terrain pixels and halo on rim pixels.
//   * Near — camera inside the shell. The integral runs from cam to terrain
//     depth (aerial perspective) or to the shell exit on sky pixels.
//
// Depth-compare is disabled (`Always` in `sky_material.rs::specialize`), so
// the quad rasterizes on every pixel, including terrain. The integration
// length comes from scene_depth, not from the depth attachment.

#import bevy_pbr::mesh_view_bindings::view
#import thalos::atmosphere::{
    AtmosphereBlock,
    atmosphere_jitter,
    integrate_atmosphere_multiscatter_occluded,
}
#import thalos::cloud_shadow::CloudShadowBlock
#import thalos::lighting::SCENE_FLUX_SCALE
#import thalos::ocean_waves::{
    ocean_coastal_wave_scale,
    ocean_sample_slope_field,
    ocean_sample_surface_wave,
    ocean_sphere_hit_distance_m,
}
#import thalos::water::shade_ocean_detailed

// Standard MaterialPlugin bind group in Bevy 0.18: group 3 (group 2 is the
// material-indices storage buffer used by the bindless material allocator).
@group(3) @binding(0) var<uniform> sky_atmos: AtmosphereBlock;

struct SkyAtmosExtra {
    sun_dir_flux:              vec4<f32>,  // xyz = sun dir (normalized), w = flux
    planet_center_radius:      vec4<f32>,  // xyz = planet center (render-space), w = radius
    world_to_body_orientation: vec4<f32>,  // render-space direction -> body-local cubemap direction
    cloud_band_radii:          vec4<f32>,  // z = atmosphere airlight ratio; x/y/w are shared with the separate cloud composite
    ocean:                     vec4<f32>,  // x = ocean sphere radius (render units = m), y = enable (>=0.5), z = shore-wave time reduced to its repeat period, w = camera height above sea (m, CPU f64-precise)
    ocean_color_depth:         vec4<f32>,  // xyz = deep-water linear-RGB tint, w = min optical-depth scale
    ocean_camera_phase:        vec4<f32>,  // xy = f64 camera wind/crosswind coordinate modulo 8192 m
    ocean_low_phase:           vec4<f32>,  // low-frequency packet phase cycles for 8192/1024/128/16 m domains
    ocean_high_phase:          vec4<f32>,  // high-frequency packet phase cycles for 8192/1024/128/16 m domains
    ocean_slope_amplitudes:    vec4<f32>,  // resolved slope amplitudes for those four domains
    ocean_surface_wavelengths_m: vec4<f32>, // resolved swell/wind/cross wavelengths in metres
    ocean_surface_amplitudes_m:  vec4<f32>, // resolved swell/wind/cross amplitudes in metres
    ocean_surface_phases_rad:    vec4<f32>, // f64-reduced angular phase for those waves
    ocean_spectrum:            vec4<f32>,  // x = swell angle, y = swell energy, z = foam slope onset, w = slope debug
    ocean_wind_basis:          vec4<f32>,  // xyz = body-local dominant-wave tangent
    ocean_crosswind_basis:     vec4<f32>,  // xyz = body-local crosswind tangent
    tile_lookup:               vec4<f32>,  // x = height-tile lookup enable, y = lod_count, z = tree_size, w = attachment-0 center size (texels/tile edge)
    tile_atlas_uv:             vec4<f32>,  // x = atlas-UV scale (center/texture), y = offset (border/texture), z = height min (m), w = height max (m)
    cloud_march:               vec4<f32>,  // x = cloud view raymarch step count (composite partition contract); y = tier diagnostic, z/w = far filter/aggregation modes
    fill_response:             array<vec4<f32>, 4>,  // far-tier opacity response LUT (cloud composite only; unused here)
    fill_cell_edge:            array<vec4<f32>, 4>,  // resolved far cell-edge LUT (cloud composite only; unused here)
    fill_cell_solid:           array<vec4<f32>, 4>,  // resolved far solid-opacity LUT (cloud composite only; unused here)
    cloud_ambient_top:         vec4<f32>,            // far-tier sky ambient (cloud composite only; unused here)
    cloud_ambient_bottom:      vec4<f32>,
    // x = cell-scale cloud evolution, sim seconds; yzw reserved.
    cloud_time:                vec4<f32>,            // x = cell-evolution sim time, y = cloud-target width px (cloud composite only; unused here)
}
@group(3) @binding(1) var<uniform> sky_atmos_extra: SkyAtmosExtra;

// Scene-depth copy: contains the main pass's depth attachment at the
// moment the copy node runs (between `Opaque3d` and `Transparent3d`).
// `texture_depth_2d` is sampled with `textureLoad` (no sampler) for
// unfiltered exact texel reads at fragment coordinates.
@group(3) @binding(2) var scene_depth_texture: texture_depth_2d;

// Precomputed multi-scatter LUT (Rgba16Float, 32×32). Each cell stores the
// average single-scattered radiance arriving at a point from every direction
// (per unit sun flux × strength), indexed by (u = (sun·zenith + 1) / 2,
// v = altitude / atmos_top). `integrate_atmosphere_multiscatter` samples it at
// every view step and adds the second bounce, which is what gives the daytime
// sky its blue luminance and lets the alpha boost below crush stars at noon.
@group(3) @binding(3) var ms_lut_tex: texture_2d<f32>;
@group(3) @binding(4) var ms_lut_sampler: sampler;

// Coast/bathymetry cube (ADR-20260720T185957Z-coastline-as-authored-data): signed terrain height about sea level,
// R16Unorm-encoded over ±COAST_ATLAS_HEIGHT_RANGE_M, indexed by body-fixed
// direction. Baked once at spawn from the same generator the tiles bake from.
// The analytic-ocean branch samples it at range for water coverage + colour so
// the far-field coastline is independent of tile LOD / depth-buffer error.
// No-ocean bodies bind a 1×1 blank and never sample it (`ocean.y` gate).
@group(3) @binding(5) var coast_atlas_tex: texture_cube<f32>;
@group(3) @binding(6) var coast_atlas_sampler: sampler;

// ── Resident-height-tile lookup (ADR-20260720T185958Z-water-projects-one-signed-sea-field) ─────────────────────────────────
// The analytic-ocean branch samples signed sea height straight from the udlod
// height atlas — the exact texels the visible terrain mesh is displaced from —
// so water coverage/colour are a projection of the one terrain field, never a
// depth-buffer comparison. The tile tree + atlas bindings mirror what the
// terrain's own shaders bind through `thalos_udlod::bindings`; the lookup
// functions below are a byte-faithful port of
// `thalos_udlod::functions::{lookup_best, lookup_tile_tree_entry}` +
// `thalos_udlod::math::Coordinate::from_world_position`, capped at the pixel's
// footprint LOD (naga_oil binds udlod's own functions to udlod's bind groups,
// so they can't be imported here directly — keep both sides in sync).
@group(3) @binding(7) var tile_height_atlas: texture_2d_array<f32>;
@group(3) @binding(8) var tile_height_sampler: sampler;
struct SkyTileTreeEntry {
    atlas_index: u32,
    atlas_lod: u32,
}
@group(3) @binding(9) var<storage, read> sky_tile_tree: array<SkyTileTreeEntry>;
@group(3) @binding(10) var<storage, read> sky_tile_origins: array<vec2<u32>>;

// Low- and high-frequency directional slope packets (RG and BA), with a full
// CPU-authored mip chain. Sampling both packets at separate dispersion phases
// prevents the old broadband cascade from translating as one rigid sheet.
// Unlike the old scalar-noise derivative this has no isolated extrema whose
// specular contours become closed "worms". Explicit texture gradients below
// preserve cross-view detail at grazing angles while filtering only the
// severely foreshortened view direction.
@group(3) @binding(11) var ocean_slope_tex: texture_2d<f32>;
@group(3) @binding(12) var ocean_slope_sampler: sampler;

// Cloud sun-transmittance cascade (CLOUD-5 §3.5 atmosphere shafts): the SAME
// field every surface receiver samples (`thalos::cloud_shadow`), here gating
// the raymarch's per-sample sun term so cloud gaps become bright crepuscular
// shafts in the air and shadowed columns lose their airlight. A zeroed block
// (no active cascade / shafts disabled) stands the term down before any fetch.
@group(3) @binding(13) var<uniform> sky_cloud_shadow: CloudShadowBlock;
@group(3) @binding(14) var cloud_shadow_tex: texture_2d<f32>;
@group(3) @binding(15) var cloud_shadow_samp: sampler;

// Mirror of `thalos_body_render::COAST_ATLAS_HEIGHT_RANGE_M` — change together.
const COAST_ATLAS_HEIGHT_RANGE_M: f32 = 8000.0;

// Mirror of `thalos_udlod`'s cube-sphere warp constant `C_SQR` (0.87²).
const CUBE_C_SQR: f32 = 0.7569;
// `INVALID_ATLAS_INDEX` / `INVALID_LOD` sentinels (u32::MAX).
const TILE_INVALID: u32 = 0xffffffffu;

// Physical wet-edge half-band (m of signed sea height) — the see-through
// shoreline sliver at beach scale. The coverage band never gets narrower than
// this, and widens only with the MEASURED local slope × sampled texel (the
// real height spread inside the pixel footprint — see the ocean branch),
// never with a range-scaled error model or an assumed representative slope.
const WET_EDGE_BAND_M: f32 = 0.75;

// The cube-face coordinate of a body-local direction: side index + face UV in
// [0, 1]. Port of `thalos_udlod::math::Coordinate::from_world_position`
// (spherical branch) — the side table and warp must match exactly or the
// lookup reads the wrong tile.
struct CubeCoord {
    side: u32,
    uv: vec2<f32>,
}

fn cube_coord_from_dir(n: vec3<f32>) -> CubeCoord {
    let a = abs(n);
    var side: u32;
    var uv: vec2<f32>;
    if a.x > a.y && a.x > a.z {
        if n.x < 0.0 {
            side = 0u;
            uv = vec2(-n.z / n.x, n.y / n.x);
        } else {
            side = 3u;
            uv = vec2(-n.y / n.x, n.z / n.x);
        }
    } else if a.z > a.y {
        if n.z > 0.0 {
            side = 1u;
            uv = vec2(n.x / n.z, -n.y / n.z);
        } else {
            side = 4u;
            uv = vec2(n.y / n.z, -n.x / n.z);
        }
    } else {
        if n.y > 0.0 {
            side = 2u;
            uv = vec2(n.x / n.y, n.z / n.y);
        } else {
            side = 5u;
            uv = vec2(-n.z / n.y, -n.x / n.y);
        }
    }
    let w = uv * sqrt((1.0 + CUBE_C_SQR) / (1.0 + CUBE_C_SQR * uv * uv));
    // Clamp just inside [0, 1) so `floor(uv * tiles)` never lands one tile out
    // of range on exact face edges.
    return CubeCoord(side, clamp(0.5 * w + 0.5, vec2(0.0), vec2(0.999999)));
}

// Terrain height (m about the reference radius) sampled from the resident
// height tile best matching `footprint_rad` (the surface arc one screen pixel
// subtends at the hit). Returns vec2(height_m, sampled_texel_m), or
// x >= 1e29 when no resident tile covers this direction (lookup disabled,
// cold streaming, terrain despawned) — the caller falls back to the coast
// atlas. At mip 0 the packed Rg16 height is manually decoded per texel
// (full precision — the coarse channel alone terraces gentle shores in
// ~0.23 m steps); at coarser mips only the monotone coarse channel is
// hardware-filtered, where the quantization is irrelevant.
fn sample_tile_sea_height(dir_local: vec3<f32>, footprint_rad: f32) -> vec2<f32> {
    let lookup = sky_atmos_extra.tile_lookup;
    if lookup.x < 0.5 {
        return vec2(1.0e30, 0.0);
    }
    let lod_count = u32(lookup.y + 0.5);
    let tree_size = u32(lookup.z + 0.5);
    let center_size = lookup.w;

    let cc = cube_coord_from_dir(dir_local);

    // Target LOD from the footprint: texel arc at LOD L is ~(π/2)/(2^L·center).
    let want = log2(1.5707963 / max(footprint_rad * center_size, 1.0e-9));
    let lod_target = u32(clamp(floor(want), 0.0, f32(lod_count) - 1.0));

    // Walk the tree root→target, keeping the deepest LOD whose wandering
    // window contains this point (lookup_best, capped at the footprint LOD —
    // LOD 0 is always in-window).
    var best_lod: u32 = 0u;
    var best_xy: vec2<u32> = vec2(0u);
    for (var lod: u32 = 0u; lod <= lod_target; lod = lod + 1u) {
        let tiles = f32(1u << lod);
        let scaled = cc.uv * tiles;
        let xy = vec2<u32>(scaled);
        let tuv = scaled - floor(scaled);
        if lod > 0u {
            let origin = sky_tile_origins[cc.side * lod_count + lod];
            let window = min(f32(tree_size), tiles);
            let rel = (vec2<f32>(vec2<i32>(xy) - vec2<i32>(origin)) + tuv) / window;
            if any(rel <= vec2(0.0)) || any(rel >= vec2(1.0)) {
                break;
            }
        }
        best_lod = lod;
        best_xy = xy;
    }

    let tree_xy = best_xy % vec2(tree_size);
    let entry_index = ((cc.side * lod_count + best_lod) * tree_size + tree_xy.x) * tree_size
        + tree_xy.y;
    let entry = sky_tile_tree[entry_index];
    if entry.atlas_index == TILE_INVALID || entry.atlas_lod == TILE_INVALID
        || entry.atlas_lod >= lod_count {
        return vec2(1.0e30, 0.0);
    }

    // The entry redirects to its best resident ancestor: re-derive the in-tile
    // UV at the entry's own LOD.
    let atlas_tiles = f32(1u << entry.atlas_lod);
    let scaled_at = cc.uv * atlas_tiles;
    let tile_uv = scaled_at - floor(scaled_at);
    let atlas_uv = tile_uv * sky_atmos_extra.tile_atlas_uv.x + sky_atmos_extra.tile_atlas_uv.y;

    // Footprint-matched mip inside the tile (providers bake full mip chains),
    // so grazing-angle anisotropy reads the mean height over the footprint
    // instead of point-sampling a foreshortened coast into moiré.
    let texel_arc = 1.5707963 / (atlas_tiles * center_size);
    let max_mip = f32(textureNumLevels(tile_height_atlas) - 1u);
    let mip = clamp(log2(max(footprint_rad / texel_arc, 1.0)), 0.0, max_mip);
    var h_unit: f32;
    if (mip < 0.5) {
        // Near field (mip 0): decode the FULL packed Rg16 fixed-point height —
        // coarse UNORM16 in x plus the sub-LSB residual in y, each texel
        // decoded before bilinear filtering (hardware-filtering the packed
        // pair is invalid: y wraps at every x step — see
        // `thalos_udlod::attachments`). The coarse channel alone quantizes
        // height in ~0.23 m steps, which terraced the shore water into
        // ~20 m depth/phase bands on gentle beaches.
        let dims = vec2<f32>(textureDimensions(tile_height_atlas));
        let p = atlas_uv * dims - vec2(0.5);
        let base = floor(p);
        let f = p - base;
        let bi = vec2<i32>(base);
        let dmax = vec2<i32>(dims) - vec2(1);
        let layer = i32(entry.atlas_index);
        let t00 = textureLoad(tile_height_atlas, clamp(bi, vec2(0), dmax), layer, 0);
        let t10 = textureLoad(
            tile_height_atlas, clamp(bi + vec2(1, 0), vec2(0), dmax), layer, 0);
        let t01 = textureLoad(
            tile_height_atlas, clamp(bi + vec2(0, 1), vec2(0), dmax), layer, 0);
        let t11 = textureLoad(
            tile_height_atlas, clamp(bi + vec2(1, 1), vec2(0), dmax), layer, 0);
        let h00 = t00.x + t00.y / 65535.0;
        let h10 = t10.x + t10.y / 65535.0;
        let h01 = t01.x + t01.y / 65535.0;
        let h11 = t11.x + t11.y / 65535.0;
        h_unit = mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
    } else {
        // At range the 0.23 m coarse quantization is far below the coverage
        // band; hardware-filter the monotone coarse channel with the mips.
        h_unit = textureSampleLevel(tile_height_atlas, tile_height_sampler, atlas_uv,
                                    entry.atlas_index, mip).x;
    }
    let h = mix(sky_atmos_extra.tile_atlas_uv.z, sky_atmos_extra.tile_atlas_uv.w, h_unit);
    let texel_m = texel_arc * sky_atmos_extra.ocean.x * exp2(mip);
    return vec2(h, texel_m);
}

// Signed sea height (m about sea level) of the terrain field at a body-local
// direction — THE water authority (ADR-20260720T185958Z-water-projects-one-signed-sea-field). One field, two resolutions:
// the resident height tile when one covers this direction, else the baked
// coast atlas mip chain (cold streaming, beyond terrain despawn, no-terrain
// bodies). Both are projections of the same `SurfaceQuery` surface, so the
// handoff is a resolution change, not an authority change — there is no seam
// to tune and the waterline cannot move with camera distance or streaming
// state. Returns vec2(signed_height_m, sampled_texel_m).
fn sample_sea_field(dir_local: vec3<f32>, footprint_rad: f32, sea_level_m: f32) -> vec2<f32> {
    let tile = sample_tile_sea_height(dir_local, footprint_rad);
    if tile.x < 1.0e29 {
        return vec2(tile.x - sea_level_m, tile.y);
    }
    let atlas_res = f32(textureDimensions(coast_atlas_tex).x);
    let texel_rad = 1.5707963 / atlas_res;
    let atlas_lod = clamp(log2(max(footprint_rad / texel_rad, 1.0)), 0.0, 12.0);
    let h = (textureSampleLevel(coast_atlas_tex, coast_atlas_sampler, dir_local, atlas_lod).r
        - 0.5) * (2.0 * COAST_ATLAS_HEIGHT_RANGE_M);
    let texel_m = texel_rad * sky_atmos_extra.ocean.x * exp2(atlas_lod);
    return vec2(h, texel_m);
}

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    // Mesh = Rectangle::new(2.0, 2.0), corners at ±1 in local x/y. Pass
    // through unchanged to cover the entire viewport in NDC regardless of
    // where the entity is parented. `z = 1.0` is the near plane in
    // reverse-Z; since depth_compare = Always the value doesn't matter
    // beyond keeping clip-space valid.
    var out: VertexOutput;
    out.clip_position = vec4(in.position.x, in.position.y, 1.0, 1.0);
    return out;
}

fn rotate_quat(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Reconstruct world-space view ray from fragment screen position.
    // Camera-basis form (vs. `world_from_clip * ndc`) keeps everything in
    // small numbers — the matrix-inverse form loses precision at orbital
    // distances when subtracted with `view.world_position`.
    let cam_right = view.world_from_view[0].xyz;
    let cam_up    = view.world_from_view[1].xyz;
    let cam_fwd   = -view.world_from_view[2].xyz;
    let ndc_x = (in.clip_position.x / view.viewport.z) * 2.0 - 1.0;
    let ndc_y = 1.0 - (in.clip_position.y / view.viewport.w) * 2.0;
    let tan_fov_y = 1.0 / view.clip_from_view[1][1];
    let tan_fov_x = 1.0 / view.clip_from_view[0][0];
    let ray_dir = normalize(
        cam_right * (ndc_x * tan_fov_x)
        + cam_up * (ndc_y * tan_fov_y)
        + cam_fwd
    );

    let cam_pos       = view.world_position;
    let planet_center = sky_atmos_extra.planet_center_radius.xyz;
    let planet_radius = sky_atmos_extra.planet_center_radius.w;
    let atmos_top_r   = planet_radius + sky_atmos.atmos_geom.x;

    // Atmosphere-shell intersection: t_enter to t_exit defines the segment
    // of the view ray that lies inside the atmosphere. When the camera is
    // already inside, t_enter clamps to 0.
    let oc        = cam_pos - planet_center;
    let oc_len_sq = dot(oc, oc);
    let b         = dot(oc, ray_dir);
    let c_atmos   = oc_len_sq - atmos_top_r * atmos_top_r;
    let disc      = b * b - c_atmos;
    if disc < 0.0 {
        discard;
    }
    let sqrt_disc = sqrt(disc);
    var t_enter   = max(-b - sqrt_disc, 0.0);
    var t_exit    = -b + sqrt_disc;
    if t_exit <= 0.0 {
        discard;
    }

    // Fallback solid-sphere hit. Scene depth is authoritative when present:
    // ground LOD terrain can sit above the mean-radius sphere, especially at
    // low grazing angles where mountains peek over the geometric horizon. If
    // the fallback sphere clips first, those depth-visible terrain pixels get
    // composited as though the ray ended at the hidden reference sphere, which
    // crushes the horizon into a dark band.
    let c_planet    = oc_len_sq - planet_radius * planet_radius;
    let disc_planet = b * b - c_planet;
    var fallback_t_surface: f32 = 1.0e30;
    var fallback_surface_fade: f32 = 0.0;
    var surface_fade: f32 = 0.0;
    // Camera→surface distance on a geometry/fallback hit, for aerial perspective.
    var surface_dist: f32 = 0.0;
    // Distance to opaque geometry at this pixel (ship hull, terrain), or a large
    // sentinel when the pixel is sky. Used below to keep the cloud layer from
    // painting over geometry that sits in front of the cloud band.
    var scene_t: f32 = 1.0e30;
    if disc_planet > 0.0 {
        let sqrt_disc_planet = sqrt(disc_planet);
        let t_planet = -b - sqrt_disc_planet;
        if t_planet > 0.0 {
            fallback_t_surface = t_planet;
            // Fade cloud compositing in across the geometric horizon.
            // Without this, an observer above the fixed cloud deck sees a
            // hard tangent band where sky-only rays begin hitting the cloud
            // shell. The fade is in metres because ship space is 1 unit = 1 m.
            fallback_surface_fade = smoothstep(0.0, 20000.0, sqrt_disc_planet);
        }
    }

    // Clip at scene depth too: if there is opaque geometry in this pixel
    // (terrain, ship hull, impostor body), terminate the raymarch there.
    // `depth_sample == 0` means "cleared / no geometry at this pixel" in
    // reverse-Z; skip the clip in that case.
    let depth_sample = textureLoad(scene_depth_texture, vec2<i32>(in.clip_position.xy), 0);
    if depth_sample > 0.0 {
        // Reconstruct view-space position at the sampled depth, then take
        // its length as the world-space distance from the camera (the
        // view-from-world basis preserves distances).
        let view_pos_h = view.view_from_clip * vec4<f32>(ndc_x, ndc_y, depth_sample, 1.0);
        let view_pos   = view_pos_h.xyz / view_pos_h.w;
        let t_scene    = length(view_pos);
        scene_t = t_scene;
        // Distinguish NEAR geometry (terrain, ship hull) from a FAR background
        // celestial body (a moon/planet impostor seen through this atmosphere,
        // millions of metres away). For near geometry, clip the raymarch at it
        // and run the surface aerial-perspective path. For a far body, leave
        // `t_exit` at the shell exit and `surface_fade = 0` so this pixel is
        // treated exactly like a SKY pixel: the in-scatter integrates the full
        // air column and the perceptual sky-luminance opacity boost below
        // CRUSHES the body by day (the same way it crushes stars), while a dim
        // night sky lets it show. Without this the impostor was veiled as if it
        // were distant *terrain* 190,000 km away, leaving Mira far too
        // prominent in daylight.
        //
        // "Near" means the hit lies INSIDE this ray's atmosphere-shell segment
        // (`t_exit` is still the shell far exit here). This is scale-free: the
        // body's own terrain always qualifies at any camera distance, while a
        // background body behind the shell never does. A fixed distance cutoff
        // (the old `atmos_top_r * 4`) misclassified the planet's OWN terrain
        // as a far body once the camera flew ~4 shell radii out — every land
        // pixel got the star-crushing sky treatment and the continents went
        // black while the analytic ocean stayed lit. The 0.1% margin absorbs
        // f32 depth-reconstruction error at planet-scale distances.
        if t_scene <= t_exit * 1.001 {
            t_exit = min(t_exit, t_scene);
            surface_fade = 1.0;
            surface_dist = t_scene;
        }
    } else if fallback_t_surface < 1.0e29 {
        t_exit = min(t_exit, fallback_t_surface);
        surface_fade = fallback_surface_fade;
        surface_dist = fallback_t_surface;
    }

    // ── Analytic ocean ────────────────────────────────────────────────────
    // Ray-trace a math sphere at sea level (ADR-20260720T185954Z-analytic-planet-water-never-meshed: never a mesh). WHERE it
    // is water and HOW DEEP it looks are both direct samples of the one
    // signed sea-height field (ADR-20260720T185958Z-water-projects-one-signed-sea-field): the resident udlod height tiles —
    // the exact texels the visible terrain mesh is displaced from — with the
    // baked coast atlas as the coarse tail. The depth buffer NEVER decides
    // coverage or colour; its one remaining job is occluding water behind
    // geometry that resolvably stands in front of it. Because the field's
    // sea-level crossings are LOD-invariant (INC-0003), the waterline cannot
    // move with camera distance, tile LOD, or streaming state.
    //
    // Numerical stability at planet radius is the whole ballgame here. The naive
    // `oc·oc − r_sea²` (two ~R² terms) and the near root `−b − √disc` (two ~R
    // terms) both catastrophically cancel in f32, so the surface jitters by
    // metres as the camera moves. Instead we take the camera's EXACT height
    // above the sea `h` from the CPU (f64-computed), form `c_sea = h·(2r+h)`
    // with no cancellation, and recover the near root from `t_near·t_far = c_sea`
    // (Vieta) using the well-conditioned far root `t_far = −b + √disc`.
    var water_here = false;
    var t_ocean = 0.0;
    // Water coverage [0, 1] and the colour-driving column (m along the ray),
    // both resolved from the sea field below (ADR-20260720T185958Z-water-projects-one-signed-sea-field).
    var ocean_cov = 0.0;
    var ocean_color_column_m = 0.0;
    // Shore-interaction inputs for `shade_ocean` (BL-10): vertical depth at
    // the hit, distance to the waterline, and the tangent direction toward
    // shore — all from the same signed sea field. Far sentinels disable the
    // shore path (open ocean, or outside the near-field FX range).
    var ocean_depth_m = 1.0e6;
    var ocean_shore_dist_m = 1.0e9;
    var ocean_shore_dir = vec3<f32>(0.0, 0.0, 1.0);
    var ocean_fp_m = 1.0e9;
    var ocean_fp_minor_m = 1.0e9;
    var ocean_fp_major_dir = vec2<f32>(1.0, 0.0);
    var ocean_enabled = sky_atmos_extra.ocean.y >= 0.5;
#ifdef ATMOSPHERE_ONLY
    ocean_enabled = false;
#endif
    if ocean_enabled {
        let r_sea = sky_atmos_extra.ocean.x;
        let h = sky_atmos_extra.ocean.w;              // camera height above sea (m)
        let up = normalize(oc);                        // planet-centre → camera, unit
        let mu = dot(up, ray_dir);
        t_ocean = ocean_sphere_hit_distance_m(mu, r_sea, h);
        if t_ocean < 1.0e29 {
            // The tile height datum is the reference radius; the sea
            // sphere may sit `sea_level_m` above it (0 for runtime
            // procedural oceans).
            let sea_level_m = r_sea - sky_atmos_extra.planet_center_radius.w;
            // Direction of the sphere hit, body-local. The f32
            // planet-centre quantization (±0.25 m at Thalos radius)
            // bounds the waterline's absolute placement error to
            // sub-texel; it shifts only with floating-origin cell
            // crossings, not per frame.
            let hit_dir_w = normalize(cam_pos + t_ocean * ray_dir - planet_center);
            let hit_dir_l = rotate_quat(
                sky_atmos_extra.world_to_body_orientation, hit_dir_w);
            let mu_hit = abs(dot(hit_dir_w, ray_dir));
            // Surface arc one screen pixel sweeps at this hit —
            // `pixel_angle · t / (R·|μ|)` — grows without bound at
            // grazing incidence. Every field sample below is
            // mip-filtered to this footprint, so foreshortened
            // coastlines average instead of shredding into moiré.
            let pixel_angle = 2.0 * tan_fov_y / max(view.viewport.w, 1.0);
            let footprint_rad =
                pixel_angle * t_ocean / (r_sea * max(mu_hit, 1.0e-3));
            ocean_fp_m = footprint_rad * r_sea;
            ocean_fp_minor_m = pixel_angle * t_ocean;
            let ray_dir_l = rotate_quat(
                sky_atmos_extra.world_to_body_orientation, ray_dir);
            let view_tangent_l = ray_dir_l - hit_dir_l * dot(ray_dir_l, hit_dir_l);
            let view_tangent_len = length(view_tangent_l);
            if view_tangent_len > 1.0e-5 {
                let view_tangent = view_tangent_l / view_tangent_len;
                let projected_major = vec2<f32>(
                    dot(view_tangent, sky_atmos_extra.ocean_wind_basis.xyz),
                    dot(view_tangent, sky_atmos_extra.ocean_crosswind_basis.xyz),
                );
                if length(projected_major) > 1.0e-5 {
                    ocean_fp_major_dir = normalize(projected_major);
                }
            }

            // ── The one authority: the signed sea field ─────────────
            let field = sample_sea_field(hit_dir_l, footprint_rad, sea_level_m);
            // Colour column: exact bathymetry over the slant path.
            ocean_color_column_m = max(-field.x, 0.0) / max(mu_hit, 0.05);
            ocean_depth_m = max(-field.x, 0.0);

            // Local field gradient (two extra taps), wherever the
            // height is inside the coastal decision band. It feeds
            // BOTH the coverage antialiasing band below (the real
            // height spread inside this pixel's footprint) and the
            // near-field shore-wave geometry (shore distance ≈
            // |h| / |∇h|, uphill = toward land).
            var slope_local = 0.0;
            if (abs(field.x) < 60.0) {
                let axis_ref = select(
                    vec3<f32>(0.0, 1.0, 0.0),
                    vec3<f32>(1.0, 0.0, 0.0),
                    abs(hit_dir_l.y) > 0.99,
                );
                let ta = normalize(cross(axis_ref, hit_dir_l));
                let tb = cross(hit_dir_l, ta);
                // Step ~¾ of the sampled texel: differences stay
                // resolved at every cascade resolution, and at a
                // mip-filtered shoreline the measured ramp slope IS
                // the footprint-scale height spread we want.
                let eps_m = max(field.y * 0.75, 3.0);
                let eps_rad = eps_m / r_sea;
                let ha = sample_sea_field(
                    normalize(hit_dir_l + ta * eps_rad), footprint_rad, sea_level_m).x;
                let hb = sample_sea_field(
                    normalize(hit_dir_l + tb * eps_rad), footprint_rad, sea_level_m).x;
                let g = vec2<f32>(ha - field.x, hb - field.x) / eps_m;
                let g_len = length(g);
                slope_local = g_len;
                if (field.x < 0.0 && g_len > 1.0e-5) {
                    ocean_shore_dist_m = clamp(-field.x / g_len, 0.0, 1.0e8);
                    // Uphill in the local tangent plane = toward land;
                    // rotate back to render space (conjugate of the
                    // world→body quaternion).
                    let shore_l = (ta * g.x + tb * g.y) / g_len;
                    let q = sky_atmos_extra.world_to_body_orientation;
                    let q_conj = vec4<f32>(-q.xyz, q.w);
                    ocean_shore_dir = normalize(rotate_quat(q_conj, shore_l));
                }
            }

            // Coverage: a band around the field's zero crossing sized
            // by the MEASURED local height spread in the footprint
            // (slope × sampled texel), floored at the physical wet
            // edge. Using a fixed representative coastal slope here
            // painted flat −10 m shoal fields as ~40 % land at coarse
            // footprints (±40 m bands — the island "halo" speckle);
            // the real slope gives shorelines their antialiasing while
            // flat shallows stay fully covered water.
            let band = clamp(slope_local * field.y, WET_EDGE_BAND_M, 40.0);
            var cov = 1.0 - smoothstep(-band, band, field.x);

            // ── Occlusion: geometry resolvably in front ─────────────
            // Two terms, both footprint-gated so unresolvable
            // coarse-mesh slivers at the limb defer to the filtered
            // field (the BL-5 lesson) while everything genuinely in
            // front still hides the water:
            if scene_t < 1.0e29 && scene_t < t_ocean {
                let fp_m = footprint_rad * r_sea;
                let scene_dir_w =
                    normalize(cam_pos + scene_t * ray_dir - planet_center);
                let scene_dir_l = rotate_quat(
                    sky_atmos_extra.world_to_body_orientation, scene_dir_w);
                // (a) Terrain: the FIELD's height at the blocker's
                // direction (exact data — the old radial depth
                // reconstruction was ±metres of f32 noise at range).
                // Thresholds scale with footprint: at flight ranges a
                // few metres of dune ridge occludes the lagoon behind
                // it; at limb anisotropy only mountain-scale relief
                // may override the filtered mask.
                let blocker = sample_sea_field(
                    scene_dir_l, footprint_rad, sea_level_m);
                let occ_land = smoothstep(
                    1.5 + 2.0e-3 * fp_m, 6.0 + 6.0e-3 * fp_m, blocker.x);
                // (b) Non-terrain geometry (craft, structures) isn't
                // in the field: occlude when it stands in front of
                // the water surface by more than a footprint-scaled
                // margin. The margin swallows depth-reconstruction
                // noise (≤ metres at orbital range, far below fp_m
                // there) so seabed ties can never flicker this on.
                let front_margin = max(4.0, fp_m);
                let occ_front = smoothstep(
                    front_margin, 2.0 * front_margin, t_ocean - scene_t);
                cov = cov * (1.0 - max(occ_land, occ_front));
            }

            ocean_cov = cov;
            water_here = ocean_cov > 0.002;
            if ocean_cov >= 0.5 {
                // Mostly water: integrate the air column to the WATER
                // surface, not the seabed behind it, so aerial
                // perspective lands on the water.
                t_exit = min(t_exit, t_ocean);
                surface_fade = 1.0;
                surface_dist = t_ocean;
            }
        }
    }

#ifdef OCEAN_ONLY
    // Ocean owns a separate transparent projection. Sky pixels must be truly
    // transparent so the canonical atmosphere
    // remains the sole sky renderer, and we avoid paying for its integration.
    if !water_here {
        return vec4<f32>(0.0);
    }
#endif

    if t_exit <= t_enter {
        discard;
    }

    let jitter = atmosphere_jitter(in.clip_position.xy);
    var scatter = integrate_atmosphere_multiscatter_occluded(
        cam_pos, ray_dir, planet_center,
        sky_atmos_extra.sun_dir_flux.xyz,
        sky_atmos_extra.sun_dir_flux.w * SCENE_FLUX_SCALE,
        t_enter, t_exit, planet_radius, sky_atmos, jitter,
        ms_lut_tex, ms_lut_sampler,
        sky_cloud_shadow, cloud_shadow_tex, cloud_shadow_samp,
    );

    // Aerial-perspective decoupling. The authored in-scatter strength is tuned
    // so the SKY DOME reads bright and crushes stars, but the same in-scatter is
    // also the airlight added on top of terrain — at that strength it over-fogs
    // the ground at any altitude. `cloud_band_radii.z` is the CPU-computed
    // ratio `aerial_perspective_strength / sky_strength` (see `AtmosphereTuning`):
    // it scales the in-scatter on surface/geometry-hit pixels only down to an
    // absolute clear-weather airlight, blended by `surface_fade` so it eases to
    // full sky-dome strength across the horizon (sky pixels are untouched).
    // Extinction (transmittance) is left physical — it already matches
    // Earth-clear-day visibility — so this dims the additive haze veil without
    // changing how distance fades contrast. `0` is unset (airless / pre-first-
    // update); treat it as full strength so we never blank the in-scatter.
    let airlight_ratio = sky_atmos_extra.cloud_band_radii.z;

    // Aerial perspective. The clear-weather airlight above keeps NEAR ground
    // crisp, but real distant terrain desaturates and tints toward the sky as
    // the air column between camera and surface grows. We drive an artistic
    // veil from camera→surface distance and fade the surface toward the
    // atmospheric in-scatter (haze) colour: it is folded into BOTH the additive
    // in-scatter strength (here) and the dst-attenuation opacity (below), so the
    // composite reduces to a clean mix(terrain, haze, veil) at range while
    // leaving near ground at its tuned look. Physical extinction is untouched.
    // Gated to bodies with an atmosphere (airlight_ratio > 0) so airless
    // surfaces (Mira) are never veiled toward a black/near-zero in-scatter.
    // Air-mass driver. The veil must scale with how much air the view ray
    // actually traverses to reach the surface, NOT the Euclidean camera→surface
    // distance. `view_tau` is the mean optical depth the integrator already
    // accumulated over `[t_enter, t_exit]` (recovered from its transmittance):
    // a thin vertical column at nadir-from-orbit, a long slant column at the
    // limb or along a low horizontal flight path. Keying on distance instead
    // saturated the ramp for the WHOLE disc the moment the camera left the
    // atmosphere — veiling even the crisp nadir uniformly (the "washed out from
    // orbit" bug). The tau thresholds are calibrated to the old distance ramp at
    // sea level (`view_tau ≈ 0.30` at the 8 km onset, `≈ 2.40` at the 70 km full
    // veil), so the on-surface look is unchanged and only altitude re-grades it.
    let mean_trans = (scatter.transmittance.x + scatter.transmittance.y + scatter.transmittance.z) / 3.0;
    let view_tau = -log(clamp(mean_trans, 1.0e-4, 1.0));
    let aerial_tau_near = 0.30;
    let aerial_tau_far = 2.40;
    let aerial_max = 0.72;
    // Deep-tau extension: past the `aerial_max` plateau the veil keeps
    // climbing toward near-total at extreme optical depths (tangent-zone /
    // limb rays, whose air column is many times the vertical one). Physically
    // the limb should melt into airlight; this ramp moves that way while
    // leaving every view up to `aerial_tau_far` exactly as calibrated.
    // (The INC-0003 "limb streak" residual was ultimately fixed elsewhere —
    // the footprint-scaled occlusion in the ocean branch above — but this
    // ramp remains the physically-right limb behaviour and softens whatever
    // grazing detail survives.)
    let aerial_deep_max = 0.96;
    let aerial_deep = (aerial_deep_max - aerial_max) * smoothstep(aerial_tau_far, 7.0, view_tau);
    let aerial = select(
        0.0,
        (smoothstep(aerial_tau_near, aerial_tau_far, view_tau) * aerial_max + aerial_deep)
            * clamp(surface_fade, 0.0, 1.0),
        airlight_ratio > 0.0,
    );
    // `surface_dist` is retained for readability of the hit-classification branches
    // above but no longer drives the veil (air mass does). Phony-assign so naga
    // doesn't flag it unused.
    _ = surface_dist;

    // Column consistency (the low-τ half of aerial perspective).
    //
    // `airlight_ratio` scales the ADDITIVE in-scatter only; the dst-attenuation
    // `physical_opacity` below stays physical (`1 − T`). Those two must agree or
    // the composite is not energy-consistent: on any surface pixel it removes
    // `1 − T` of the terrain's own radiance and hands back only `airlight_ratio`
    // of the airlight that is supposed to replace it. Straight down from orbit
    // through a full Thalos column that is a ~15% dim with ~1/30th of the light
    // returned, so orbital land came out DARKER and MORE saturated rather than
    // veiled — the "too clear at high altitude" report. The `aerial` ramp above
    // already keeps the pair consistent at the thick end (it feeds both the
    // in-scatter scale and the opacity); only this floor was left constant.
    //
    // So lift the airlight from the ground-calibrated clear-weather value toward
    // the strength the in-scatter would carry at `strength: 1` — `atmos_geom.z`
    // is the artistic sky-dome inflation, which exists only to crush stars, so
    // `1 / atmos_geom.z` undoes it and lands on the physically consistent
    // airlight. `max` so an authored hazier-than-physical weather is never
    // pulled back down.
    //
    // Driven by RAYLEIGH air mass, not total τ. The sea-level calibration is
    // aerosol-dominated (8 km horizontal crosses ~6.7 Mie scale heights but only
    // one Rayleigh one), while an orbital ray barely crosses the 1.2 km Mie layer
    // at all, so keying the lift on total τ would re-haze the ground distance the
    // Mie cut was made to keep crisp — and with GREY haze, which is what washed
    // it to a grey-tan band before. The two columns separate out of the
    // CHROMATIC spread of the transmittance the integrator already returned: Mie
    // is spectrally flat, so it cancels exactly in a channel difference and what
    // survives is pure Rayleigh. `1.0` = one vertical column = nadir from orbit.
    // (Assumes `mie_beta_g.xyz` is grey, which every authored body is; a body that
    // authored coloured aerosols would leave a small residue in the difference and
    // read a slightly high air mass — degrades gracefully, never fails.)
    let tau_chan = -log(clamp(scatter.transmittance, vec3<f32>(1.0e-4), vec3<f32>(1.0)));
    let tau_zenith_rayleigh = sky_atmos.rayleigh_beta_h.xyz * sky_atmos.rayleigh_beta_h.w;
    let rayleigh_chroma_span = max(tau_zenith_rayleigh.z - tau_zenith_rayleigh.x, 1.0e-4);
    let rayleigh_air_mass = max((tau_chan.z - tau_chan.x) / rayleigh_chroma_span, 0.0);
    // Onset at a sixth of a vertical column so the near field a ground observer
    // sees (≤ ~2 km at sea level) keeps its tuned look untouched. The far end is
    // deliberately well past the one column a nadir orbital ray crosses: an
    // oblique view from orbit spans roughly 1–4 columns across a single frame, so
    // a nearer ceiling pins most of the frame at one value and reads as a flat
    // wash rather than aerial perspective that develops with depth.
    let column_lift_near = 0.15;
    let column_lift_far = 4.00;
    // Fraction of the physically consistent airlight actually applied. Surface
    // radiance and airlight are not in a common exposure/flux scale, so the
    // in-scatter over-contributes even at `strength: 1` (docs/rendering/
    // atmosphere.md). The ground calibration folds a 0.10 correction into
    // `aerial_perspective_strength`; that is far too aggressive once the column
    // is thick — applying it here is what produced no veil at all — but the
    // uncorrected value over-veils orbital land to a flat blue-grey with no
    // biome colour left. Screenshot-calibrated against ISS reference framings.
    let column_airlight_exposure = 0.50;
    let physical_airlight =
        column_airlight_exposure / max(sky_atmos.atmos_geom.z, 1.0e-3);
    let column_airlight = mix(
        airlight_ratio,
        max(airlight_ratio, physical_airlight),
        smoothstep(column_lift_near, column_lift_far, rayleigh_air_mass),
    );

    let base_surface_airlight = mix(1.0, column_airlight, clamp(surface_fade, 0.0, 1.0));
    let surface_airlight = max(base_surface_airlight, aerial);
    let airlight_scale = select(surface_airlight, 1.0, airlight_ratio <= 0.0);
    scatter.in_scatter = scatter.in_scatter * airlight_scale;

    // Premultiplied: `rgb` is already weighted by sun flux and β coefficients
    // inside `integrate_atmosphere`. Alpha is the mean opacity over the three
    // channels — the standard `Premultiplied` blend dims what was drawn
    // behind (terrain albedo, impostor surface, stars).
    // `mean_trans` computed above (drives the air-mass aerial veil).
    let physical_opacity = clamp(1.0 - mean_trans, 0.0, 1.0);

    // Perceptual sky-luminance opacity boost. The dst-attenuation factor of
    // a premultiplied blend is `1 − α`, so to make a bright daytime sky drown
    // out stars the alpha has to approach 1.0 even when extinction (and
    // therefore physical opacity) is small — Earth's midday sky has τ_v ≈ 0.2
    // for blue, so the physically correct factor only dims background stars
    // by ~20%, far too little against star peak values in the hundreds. Stars
    // are calibrated to be visible against a black sky, so the perceptual fix
    // is to crush them whenever the local in-scatter radiance is high enough
    // that a real observer's eye would adapt away from them. Restricted to
    // sky pixels (no opaque hit) so terrain aerial perspective stays driven
    // by physical transmittance only. The analytic planet-sphere fallback
    // also counts as a surface hit; otherwise the horizon flips between
    // boosted sky opacity and physical surface transmittance as a hard band.
    var opacity = physical_opacity;
    if surface_fade <= 0.0 {
        let sky_lum = max(scatter.in_scatter.r,
                          max(scatter.in_scatter.g, scatter.in_scatter.b));
        let lum_opacity = smoothstep(0.03, 0.20, sky_lum);
        opacity = max(opacity, lum_opacity);
    } else {
        // Surface aerial veil: lift the dst-attenuation to match the `aerial`
        // in-scatter strength added above, so distant terrain's own colour is
        // replaced by the haze colour (desaturate + tint) rather than merely
        // brightened. Near ground keeps `physical_opacity` (aerial ≈ 0).
        opacity = max(opacity, aerial);
    }
    let combined_opacity = clamp(opacity, 0.0, 1.0);
    let sky_rgb = scatter.in_scatter;

    // Analytic ocean composite. Water occludes the seabed already in the
    // framebuffer, so we supply its radiance here and output fully opaque
    // (alpha = 1, the framebuffer seabed contributes 0). The surface is dimmed
    // by air transmittance `(1 − opacity)` (physical + aerial veil, mirroring
    // how terrain in the framebuffer is attenuated) and by clouds in front of
    // it. The in-scatter / clouds were already integrated to the water surface
    // (`t_exit = t_ocean`).
    if water_here {
        let hit_ws = cam_pos + t_ocean * ray_dir;
        let geo_n = normalize(hit_ws - planet_center);
        let sun_flux_scaled = sky_atmos_extra.sun_dir_flux.w * SCENE_FLUX_SCALE;
        let q_world_to_body = sky_atmos_extra.world_to_body_orientation;
        let q_body_to_world = vec4<f32>(-q_world_to_body.xyz, q_world_to_body.w);
        let rel_body_m = rotate_quat(q_world_to_body, hit_ws - cam_pos);
        let geo_n_body = rotate_quat(q_world_to_body, geo_n);
        let local_m = sky_atmos_extra.ocean_camera_phase.xy + vec2<f32>(
            dot(rel_body_m, sky_atmos_extra.ocean_wind_basis.xyz),
            dot(rel_body_m, sky_atmos_extra.ocean_crosswind_basis.xyz),
        );
        let major_gradient_m = ocean_fp_major_dir * ocean_fp_m;
        let minor_gradient_m = vec2<f32>(
            -ocean_fp_major_dir.y,
            ocean_fp_major_dir.x,
        ) * ocean_fp_minor_m;
        let detail = ocean_sample_slope_field(
            ocean_slope_tex,
            ocean_slope_sampler,
            local_m,
            major_gradient_m,
            minor_gradient_m,
            sky_atmos_extra.ocean_low_phase,
            sky_atmos_extra.ocean_high_phase,
            sky_atmos_extra.ocean_slope_amplitudes,
            sky_atmos_extra.ocean_spectrum.x,
            sky_atmos_extra.ocean_spectrum.y,
        );
        let surface = ocean_sample_surface_wave(
            local_m,
            sqrt(max(ocean_fp_m * ocean_fp_minor_m, 1.0e-6)),
            vec2<f32>(1.0, 0.0),
            sky_atmos_extra.ocean_surface_wavelengths_m,
            sky_atmos_extra.ocean_surface_amplitudes_m,
            sky_atmos_extra.ocean_surface_phases_rad,
            ocean_coastal_wave_scale(ocean_shore_dist_m, 116.0, 1.0),
        );
        let resolved_slope = detail.slope + surface.slope;
        let resolved_alpha_ggx = clamp(
            sqrt(detail.alpha_ggx * detail.alpha_ggx + 2.0 * surface.omitted_variance),
            0.06,
            0.22,
        );
        let resolved_breakup = max(detail.breakup, surface.crest);
        let sky_tau_zenith = sky_atmos.rayleigh_beta_h.xyz * sky_atmos.rayleigh_beta_h.w;
        var water: vec3<f32>;
        if sky_atmos_extra.ocean_spectrum.w >= 0.5 {
            // Diagnostic separates resolved field topology (RG) from the
            // mip→GGX variance handoff (B), bypassing the sun road and BRDF.
            water = vec3<f32>(
                clamp(0.5 + resolved_slope.x * 1.8, 0.0, 1.0),
                clamp(0.5 + resolved_slope.y * 1.8, 0.0, 1.0),
                clamp(resolved_alpha_ggx / 0.15, 0.0, 1.0),
            );
        } else {
            water = shade_ocean_detailed(
                geo_n_body,
                q_body_to_world,
                geo_n,
                -ray_dir,
                t_ocean,
                sky_atmos_extra.ocean.z,
                sky_atmos_extra.sun_dir_flux.xyz,
                sun_flux_scaled,
                sky_atmos_extra.ocean_color_depth,
                ocean_color_column_m,
                ocean_depth_m,
                ocean_shore_dist_m,
                ocean_shore_dir,
                ocean_fp_m,
                resolved_slope,
                resolved_alpha_ggx,
                resolved_breakup,
                sky_atmos_extra.ocean_spectrum.z,
                sky_atmos_extra.ocean_wind_basis.xyz,
                sky_atmos_extra.ocean_crosswind_basis.xyz,
                sky_tau_zenith,
                sky_atmos.atmos_geom.z,
            );
        }
        let surf_trans = 1.0 - opacity;
        // `ocean_cov` was resolved in the ocean block from the signed sea
        // field (ADR-20260720T185958Z-water-projects-one-signed-sea-field: resident height tiles → coast-atlas tail). Partial
        // coverage lets the framebuffer seabed show through for the wet
        // shoreline sliver / clear shallows.
        var out_rgb = sky_rgb + water * surf_trans * ocean_cov;
        var out_a = mix(combined_opacity, 1.0, ocean_cov);
#ifdef OCEAN_ONLY
        // This pass sits over an already-rendered atmosphere. Replace only the
        // signed-field water fraction, carrying its own foreground air column
        // and water radiance as premultiplied colour. The untouched fraction
        // leaves the canonical sky/terrain behind it intact.
        out_rgb = (sky_rgb + water * surf_trans) * ocean_cov;
        out_a = ocean_cov;
#endif
        return vec4(out_rgb, out_a);
    }

#ifdef OCEAN_ONLY
    return vec4<f32>(0.0);
#else
    return vec4(sky_rgb, combined_opacity);
#endif
}
