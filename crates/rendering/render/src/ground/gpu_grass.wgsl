// GPU-generated grass field (see gpu_grass.rs).
//
// Vertex: SYNTHESIZES each blade from its template slot — no geometry comes
// in. The template `POSITION` packs (cell dx, cell dy, band·2048 + blade·8 +
// corner); everything else — the clump's world spot, the placement gates, the
// grass type, the tint, the blade curve — is derived here per frame from
// body-global cell hashes, the CPU-filled height/aux control window, and the
// shared `thalos::landcover` field. Rejected blades collapse to their root (a
// degenerate zero-area strip): drawn, never shaded.
//
// The look is the Ghost-of-Tsushima recipe mapped onto Thalos:
// - quadratic-Bézier blades bent from the ROOT by a scrolling gust noise
//   field (rolling wind waves), plus the shared `thalos::grass_displace`
//   micro-flutter;
// - view-dependent widening so edge-on blades never vanish (the sparse-when-
//   moving read);
// - a rounded cross-blade normal blended toward the terrain normal with
//   distance (near = individual lit blades, far = one cohesive lawn);
// - specular sheen + tip translucency on the shared foliage BRDF;
// - per-clump coherence (facing, height, hue) over per-blade jitter, with a
//   moisture-driven dry-straw blade mix;
// - root darkening + root ambient occlusion that FADE OUT with band so far
//   blades converge on the terrain's own vegetation colour (the MSFS rule:
//   distant grass IS the terrain colour, so thinning coverage never reads as
//   polka dots).
//
// Grass types are configured from the Rust side via the `style` uniform table
// (gpu_grass_style_table) — no shader edit to add/tune a type.

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}
#import thalos::lighting::{compute_surface_sky, SurfaceSky, FoliageSurface, shade_foliage, object_aerial_recession, sun_daylight}
#import thalos::shadow::{ShadowCascadeBlock, sun_shadow_factor_vert}
#import thalos::grass_displace::grass_blade_world_pos
#import thalos::landcover::{moisture_detail, macro_variation, vegetation_color, climate_warmth}

// Mirror of gpu_grass.rs `GpuGrassParams` — field order is load-bearing.
struct GpuGrassParams {
    sun_dir: vec4<f32>,
    wind: vec4<f32>,
    // x = time (s), y = altitude collapse, z = sea level (m), w = enable.
    time_fade: vec4<f32>,
    sky_up: vec4<f32>,
    sky_tau: vec4<f32>,
    anchor: vec4<f32>,
    // Body-fixed (= entity-local) anchor tangent frame; east.w = anchor height,
    // north.w / up.w = the anchor's offset from the WINDOW centre along
    // east/north (the anchor re-registers every few metres, the window every
    // ~25 m — window lookups add this offset).
    frame_east: vec4<f32>,
    frame_north: vec4<f32>,
    frame_up: vec4<f32>,
    // x = texel (m), y = size (px), z = half extent (m), w = climate cold
    // lift at the anchor (m — shifts the treeline fade / veg palette).
    window_meta: vec4<f32>,
    // xyz = anchor surface point mod the landcover period (body-fixed metres),
    // w = macro landcover moisture at the anchor ([-1, 1]).
    phase: vec4<f32>,
    band_cell: array<vec4<u32>, 5>,
    band_geom: array<vec4<f32>, 5>,
    // Style table (gpu_grass_style_table): 2 rows per style (dry/lush/lawn) —
    // (height, width, radius, droop) then (dome, dry_mix, sheen, stiffness).
    style: array<vec4<f32>, 6>,
}

@group(3) @binding(0) var<uniform> gg: GpuGrassParams;
@group(3) @binding(1) var<uniform> gg_shadow: ShadowCascadeBlock;
@group(3) @binding(2) var sun_shadow_map_0: texture_depth_2d;
@group(3) @binding(3) var sun_shadow_map_1: texture_depth_2d;
@group(3) @binding(4) var sun_shadow_map_2: texture_depth_2d;
// Control window: heights (R32Float, textureLoad only) + aux masks (Rgba8:
// grass weight, terrain normal xz biased, scatter treatment).
@group(3) @binding(5) var gg_height: texture_2d<f32>;
@group(3) @binding(6) var gg_aux: texture_2d<f32>;
@group(3) @binding(7) var gg_aux_sampler: sampler;

// ── Band table — mirror of gpu_grass.rs `GPU_GRASS_BANDS` ───────────────────
const GG_BAND_INNER: array<f32, 5> = array<f32, 5>(0.0, 10.0, 30.0, 80.0, 170.0);
const GG_BAND_OUTER: array<f32, 5> = array<f32, 5>(10.0, 30.0, 80.0, 170.0, 340.0);
const GG_BAND_FADE: array<f32, 5> = array<f32, 5>(3.0, 4.0, 8.0, 14.0, 28.0);
const GG_BAND_BLADES: array<f32, 5> = array<f32, 5>(26.0, 18.0, 10.0, 6.0, 4.0);
// Constant-coverage: blades widen as per-area count falls outward — far-band
// blades are card-scale tufts, not literal blades. The first table alone can
// NOT hold coverage (count/m² falls ~60× band 0→4); the far field also leans
// on the screen-space minimum width in the vertex (GG_MIN_WIDTH_RAD).
const GG_BAND_WIDTH_MUL: array<f32, 5> = array<f32, 5>(1.35, 2.2, 4.0, 7.0, 12.0);
const GG_BAND_HEIGHT_MUL: array<f32, 5> = array<f32, 5>(1.0, 1.0, 1.1, 1.25, 1.5);

// Blade strip corner layout: height fraction + side, 7 verts (see the
// template's BLADE_INDICES).
const GG_CORNER_T: array<f32, 7> = array<f32, 7>(0.0, 0.0, 0.45, 0.45, 0.8, 0.8, 1.0);
const GG_CORNER_SIDE: array<f32, 7> = array<f32, 7>(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0.0);

const GG_TAU: f32 = 6.28318530717958647;
const GG_GOLDEN_ANGLE: f32 = 2.3999634;
// Clump footprint spread over the profile's radius: >1 interleaves
// neighbouring tufts into a continuous carpet instead of discrete polka-dot
// tussocks (the near-sparseness read), without more template blades.
const GG_CLUMP_RADIUS_MUL: f32 = 1.45;
// How strongly blades in a clump share the clump's facing direction (0 = all
// random, 1 = a combed clump). Coherent clumps catch the sun as patches —
// the field stops reading as uniform noise.
const GG_CLUMP_FACING: f32 = 0.4;

// ── Wind gust field ──────────────────────────────────────────────────────────
// A scrolling two-octave value-noise field bends whole blades from the root —
// visible rolling waves — while `grass_displace` keeps a small per-blade
// flutter on top.
const GG_GUST_WAVELEN_M: f32 = 14.0;   // primary rolling-wave length
const GG_GUST_SPEED_M_S: f32 = 5.5;    // how fast waves travel downwind
const GG_GUST_BASE: f32 = 0.10;        // ever-present lean (fraction of height)
const GG_GUST_AMP: f32 = 0.42;         // gust crest bend (fraction of height)
// Fraction of the lib's tip-flutter amplitude kept under the gust bend.
const GG_FLUTTER_KEEP: f32 = 0.5;

// ── Blade shading knobs ──────────────────────────────────────────────────────
// Edge-on widening: max extra width when the blade plane is parallel to the
// view ray (GoT's view-space thickening).
const GG_VIEW_WIDEN: f32 = 0.9;
// Screen-space minimum blade width, radians of view angle (~1.5 px at a 60°
// FOV / 1080p ≈ 0.0015). A blade thinner than a pixel doesn't render darker —
// it stochastically VANISHES, which read in-game as "grass just stops" at
// ~50 m. Clamping width to a fixed on-screen angle keeps the far field a
// continuous sward; the band tint already converges to the terrain colour so
// the widened blades never read as fat paddles.
const GG_MIN_WIDTH_RAD: f32 = 0.0016;
// Cross-blade normal rounding (cylinder read) and its near/far blend range:
// near blades shade individually, far blades inherit the terrain normal and
// fuse into one lawn.
const GG_NORMAL_ROUND: f32 = 0.65;
const GG_NORMAL_BLEND_NEAR: f32 = 0.75;
const GG_NORMAL_BLEND_FAR: f32 = 0.10;
const GG_NORMAL_BLEND_LO_M: f32 = 8.0;
const GG_NORMAL_BLEND_HI_M: f32 = 70.0;
// Specular sheen lobe (scaled per style by style.sheen).
const GG_SHEEN_POWER: f32 = 24.0;
const GG_SHEEN_GAIN: f32 = 0.55;
// Tip translucency (t-weighted into the foliage BRDF's transmit lobe).
const GG_TRANSLUCENCY: f32 = 0.42;
// Saturation lift on the blade tint over the raw landcover colour (>1 =
// richer green). The terrain keeps the raw colour; blades read lusher
// because live blades ARE more saturated than the ground they stand on.
const GG_SATURATION: f32 = 1.18;
// Root ambient occlusion floor (ambient_scale at t = 0, band 0).
const GG_ROOT_AO: f32 = 0.45;
// Straw colour dry blades mix toward.
const GG_STRAW: vec3<f32> = vec3<f32>(0.88, 0.70, 0.35);

// ── Hashing (PCG) — the GPU analogue of the CPU `blade_hash` stream ─────────
fn gg_pcg(v_in: u32) -> u32 {
    var v = v_in * 747796405u + 2891336453u;
    let w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (w >> 22u) ^ w;
}

// One hash stream per (global cell, band, blade, salt); all draws of a cell at
// any frame/anchor hash identically, so placement is body-fixed-stable.
fn gg_hash01(cell: vec2<u32>, face: u32, band: u32, blade: u32, salt: u32) -> f32 {
    var h = cell.x * 0x9E3779B9u;
    h = h ^ (cell.y * 0x85EBCA6Bu);
    h = h ^ (face * 0xC2B2AE35u);
    h = h ^ (band * 0x27D4EB2Fu);
    h = h ^ (blade * 0x165667B1u);
    h = h ^ (salt * 0xD6E8FEB8u);
    return f32(gg_pcg(h)) * 2.3283064365386963e-10; // / 2^32
}

// Value noise over metres (for the gust field): integer-lattice PCG hash,
// smoothstep-interpolated. Isotropy doesn't matter here (wind waves are
// directional anyway), so value noise is fine and cheap.
fn gg_lattice01(i: vec2<i32>) -> f32 {
    var h = bitcast<u32>(i.x) * 0x9E3779B9u;
    h = h ^ (bitcast<u32>(i.y) * 0x85EBCA6Bu);
    return f32(gg_pcg(h)) * 2.3283064365386963e-10;
}

fn gg_vnoise(p: vec2<f32>) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = p - floor(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = gg_lattice01(i);
    let b = gg_lattice01(i + vec2<i32>(1, 0));
    let c = gg_lattice01(i + vec2<i32>(0, 1));
    let d = gg_lattice01(i + vec2<i32>(1, 1));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Gust strength [0,1] at a stable tangent-plane point (metres), scrolled
// downwind over time. Two octaves: the rolling wave + a finer chop.
fn gg_gust(p_t: vec2<f32>, wind_t: vec2<f32>, t: f32) -> f32 {
    let q = p_t - wind_t * (t * GG_GUST_SPEED_M_S);
    let wave = gg_vnoise(q / GG_GUST_WAVELEN_M);
    let chop = gg_vnoise(q / (GG_GUST_WAVELEN_M * 0.31) + vec2<f32>(37.7, 11.3));
    let g = wave * 0.72 + chop * 0.28;
    // Sharpen crests a touch so waves read as waves, not haze.
    return g * g * (3.0 - 2.0 * g);
}

// ── Window sampling ──────────────────────────────────────────────────────────
// `local` = metres from the anchor along (east, north). Returns height via
// manual bilinear over textureLoad (R32Float is non-filterable). A missing
// probe was written as f32::MIN — treat any absurdly low corner as a miss.
fn gg_window_uv(local: vec2<f32>) -> vec2<f32> {
    return (local + vec2<f32>(gg.window_meta.z)) / gg.window_meta.x;
}

fn gg_window_height(local: vec2<f32>) -> f32 {
    let size = gg.window_meta.y;
    let p = gg_window_uv(local) - 0.5;
    let p0 = floor(p);
    let f = p - p0;
    let i0 = vec2<i32>(clamp(p0, vec2<f32>(0.0), vec2<f32>(size - 1.0)));
    let i1 = vec2<i32>(clamp(p0 + 1.0, vec2<f32>(0.0), vec2<f32>(size - 1.0)));
    let h00 = textureLoad(gg_height, vec2<i32>(i0.x, i0.y), 0).x;
    let h10 = textureLoad(gg_height, vec2<i32>(i1.x, i0.y), 0).x;
    let h01 = textureLoad(gg_height, vec2<i32>(i0.x, i1.y), 0).x;
    let h11 = textureLoad(gg_height, vec2<i32>(i1.x, i1.y), 0).x;
    let lo = min(min(h00, h10), min(h01, h11));
    if (lo < -1.0e9) {
        return -1.0e10; // missing probe → caller kills the blade
    }
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    // x = cell dx, y = cell dy, z = band·2048 + blade·8 + corner.
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    // Terrain (cohesion) normal — the far-field shading normal.
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) shadow: f32,
    @location(4) @interpolate(flat) sky0: vec4<f32>,
    @location(5) @interpolate(flat) sky1: vec4<f32>,
    @location(6) @interpolate(flat) sky2: vec4<f32>,
    // xyz = blade facing normal (world), w = cross-blade coordinate −1..1.
    @location(7) blade_n: vec4<f32>,
    // xyz = blade side axis (world), w = height fraction t.
    @location(8) blade_s: vec4<f32>,
    // x = sheen, y = translucency, z = band fraction 0..1, w unused.
    @location(9) extra: vec4<f32>,
}

// Finish a vertex: world transform, scale-fade + flutter, per-vertex shadow,
// the per-draw sky environment, and the blade-frame varyings the fragment's
// rounded-normal / sheen path reads.
fn gg_emit(
    in: VertexInput,
    local_pos: vec3<f32>,
    local_root: vec3<f32>,
    local_normal: vec3<f32>,
    blade_n_local: vec3<f32>,
    blade_s_local: vec3<f32>,
    side01: f32,
    tint: vec3<f32>,
    uv: vec2<f32>,
    band: u32,
    extra: vec4<f32>,
) -> VertexOutput {
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    let world_pos_in =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(local_pos, 1.0)).xyz;
    let root_pos =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(local_root, 1.0)).xyz;
    let world_normal =
        normalize(mesh_functions::mesh_normal_local_to_world(local_normal, in.instance_index));
    let blade_n_w =
        normalize(mesh_functions::mesh_normal_local_to_world(blade_n_local, in.instance_index));
    let blade_s_w =
        normalize(mesh_functions::mesh_normal_local_to_world(blade_s_local, in.instance_index));

    var band_inner = GG_BAND_INNER;
    var band_outer = GG_BAND_OUTER;
    var band_fade = GG_BAND_FADE;
    // Band 0 never fades in (inner edge at the player's feet).
    var near_edge = band_inner[band];
    if (band == 0u) {
        near_edge = -1.0e6;
    }
    let fade = vec4<f32>(gg.time_fade.x, near_edge, band_outer[band], band_fade[band]);

    // Residual per-blade tip flutter (the gust bend already moved the Bézier).
    let flutter = vec4<f32>(gg.wind.xyz, gg.wind.w * GG_FLUTTER_KEEP);
    let world_pos = grass_blade_world_pos(
        world_pos_in,
        root_pos,
        uv,
        flutter,
        fade,
        gg.time_fade.y,
        gg.anchor,
        view.world_position,
    );

    let sun_dir = gg.sun_dir.xyz;
    let up = gg.sky_up.xyz;
    let sky = compute_surface_sky(gg.sky_tau.xyz, gg.sky_tau.w, up, sun_dir, gg.sun_dir.w);

    var out: VertexOutput;
    out.clip_position = position_world_to_clip(world_pos);
    out.world_position = world_pos;
    out.world_normal = world_normal;
    out.color = vec4<f32>(tint, 1.0);
    out.shadow = sun_shadow_factor_vert(
        world_pos, gg_shadow, sun_shadow_map_0, sun_shadow_map_1, sun_shadow_map_2,
    );
    out.sky0 = vec4<f32>(sky.sun_color, sky.sun_scale);
    out.sky1 = vec4<f32>(sky.sky_radiance, 0.0);
    out.sky2 = vec4<f32>(sky.ground_radiance, 0.0);
    out.blade_n = vec4<f32>(blade_n_w, side01);
    out.blade_s = vec4<f32>(blade_s_w, uv.x);
    out.extra = extra;
    return out;
}

// A rejected blade: all 7 corners at the clump root → zero-area, never shaded.
fn gg_kill(in: VertexInput, local_root: vec3<f32>) -> VertexOutput {
    return gg_emit(
        in,
        local_root,
        local_root,
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        0.0,
        vec3<f32>(0.02, 0.04, 0.02),
        vec2<f32>(0.0),
        0u,
        vec4<f32>(0.0),
    );
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    // ── Decode the template slot ────────────────────────────────────────────
    let packed = u32(in.position.z);
    let band = packed / 2048u;
    let blade = (packed % 2048u) / 8u;
    let corner = packed % 8u;
    let cell_d = vec2<i32>(i32(in.position.x), i32(in.position.y));

    var band_geom = gg.band_geom;
    var band_cell = gg.band_cell;
    let geom = band_geom[band];
    let cellp = band_cell[band];
    let face = cellp.z;

    // Global cell id (body-fixed permanent) — the hash key.
    let gx = u32(i32(cellp.x) + cell_d.x);
    let gy = u32(i32(cellp.y) + cell_d.y);
    let gcell = vec2<u32>(gx, gy);

    // ── Clump root in anchor-tangent metres ─────────────────────────────────
    let jx = 0.05 + 0.9 * gg_hash01(gcell, face, band, 0u, 0u);
    let jy = 0.05 + 0.9 * gg_hash01(gcell, face, band, 0u, 1u);
    let clump_2d = vec2<f32>(
        (f32(cell_d.x) + jx - geom.z) * geom.x,
        (f32(cell_d.y) + jy - geom.w) * geom.y,
    );

    let east = gg.frame_east.xyz;
    let north = gg.frame_north.xyz;
    let r_up = gg.frame_up.xyz;
    let anchor_h = gg.frame_east.w;

    // Window-space position: the anchor drifts between window rebuilds, so
    // window lookups add the anchor→window-centre offset.
    let win2d = clump_2d + vec2<f32>(gg.frame_north.w, gg.frame_up.w);

    // Fallback root (used by every reject path too): on the terrain if the
    // window has it, else at the anchor plane.
    var h = gg_window_height(win2d);
    let have_h = h > -1.0e9;
    var h_rel = select(0.0, h - anchor_h, have_h);
    let local_root = east * clump_2d.x + north * clump_2d.y + r_up * h_rel;

    // ── Placement gates (the CPU builder's, from the control window) ────────
    if (gg.time_fade.w < 0.5 || !have_h) {
        return gg_kill(in, local_root);
    }
    let half = gg.window_meta.z;
    if (max(abs(win2d.x), abs(win2d.y)) > half - 2.0 * gg.window_meta.x) {
        return gg_kill(in, local_root);
    }
    let aux = textureSampleLevel(gg_aux, gg_aux_sampler, gg_window_uv(win2d) / gg.window_meta.y, 0.0);
    let treatment = aux.w;
    if (treatment > 0.75) { // cleared (paving / building footprint)
        return gg_kill(in, local_root);
    }
    let lawn = treatment > 0.25;
    // Sea level + beach clearance — mirror of `scatter::VEG_BEACH_CLEAR_M`
    // (the strand stays bare sand; change together).
    if (h <= gg.time_fade.z + 4.0) {
        return gg_kill(in, local_root);
    }
    // Terrain normal in the tangent frame; slope = |∇h| = tan(tilt).
    let nx = aux.y * 2.0 - 1.0;
    let nz = aux.z * 2.0 - 1.0;
    let ny = sqrt(max(1.0 - nx * nx - nz * nz, 1.0e-4));
    if (length(vec2<f32>(nx, nz)) / ny > 0.45) {
        return gg_kill(in, local_root);
    }
    let grass_w = aux.x;
    // Climate-shifted (ecological) altitude: the treeline fade descends with
    // latitude (window_meta.w = cold lift at the anchor), matching the
    // terrain paint + CPU blades.
    let eco_h = h + gg.window_meta.w;
    // Loose ramp: real landcover grass weights sit ~0.3–0.6, and the old
    // (0.20, 0.50) ramp culled up to two-thirds of those blades — the
    // in-game "bald" read. Sub-0.3 weights now still grow a thinning sward.
    var accept = smoothstep(0.06, 0.30, grass_w)
        * (1.0 - smoothstep(2000.0, 2700.0, eco_h));
    if (lawn) {
        accept = 1.0;
    }
    if (gg_hash01(gcell, face, band, 0u, 2u) >= accept) {
        return gg_kill(in, local_root);
    }

    // ── Grass type + tint from the shared landcover field ───────────────────
    // Macro moisture rides `phase.w` (per-window, sampled at the anchor from
    // the planet-scale f64 field — docs/terrain_macro.md); the wrapped fine
    // tier is added per blade so blades match the terrain's per-pixel field.
    let p_body = gg.phase.xyz + east * clump_2d.x + north * clump_2d.y + r_up * h_rel;
    let moisture = clamp(gg.phase.w + moisture_detail(p_body), -1.0, 1.0);
    let macro_var = macro_variation(p_body);
    let veg = vegetation_color(eco_h, moisture, macro_var, climate_warmth(gg.window_meta.w));

    // Style blend: moisture picks along dry→lush; lawn treatment overrides.
    var styles = gg.style;
    let type_mix = smoothstep(-0.55, 0.55, moisture);
    var style_a = mix(styles[0], styles[2], type_mix); // height,width,radius,droop
    var style_b = mix(styles[1], styles[3], type_mix); // dome,dry_mix,sheen,stiffness
    if (lawn) {
        style_a = styles[4];
        style_b = styles[5];
    }
    var band_width_mul = GG_BAND_WIDTH_MUL;
    var band_height_mul = GG_BAND_HEIGHT_MUL;
    let band_f = f32(band) * 0.25; // 0 (nearest) → 1 (band 4)

    // Per-clump coherence: shared height factor, facing direction, hue lean.
    let clump_hmul = 0.72 + 0.56 * gg_hash01(gcell, face, band, 0u, 4u);
    let clump_face_ang = gg_hash01(gcell, face, band, 0u, 5u) * GG_TAU;
    let clump_hue = gg_hash01(gcell, face, band, 0u, 6u);

    let profile_h = style_a.x * band_height_mul[band] * clump_hmul;
    let profile_w = style_a.y * band_width_mul[band];
    let profile_r = style_a.z;
    let droop_p = style_a.w;
    let dome_p = style_b.x;

    // ── Blade within the clump (sunflower fountain — the CPU emitter) ───────
    var band_blades = GG_BAND_BLADES;
    let n_blades = band_blades[band];
    let t01 = (f32(blade) + 0.5) / n_blades;
    let rf = sqrt(t01);
    let rng2 = gg_hash01(gcell, face, band, blade + 1u, 12u);
    // Clump footprint: at least the style's tuft radius, but never below
    // ~45 % of the band's cell size — coarser bands must interleave their
    // neighbours' blades or the cell lattice reads as polka-dot rows.
    let spread = max(profile_r * GG_CLUMP_RADIUS_MUL, geom.x * 0.45);
    let rad = spread * rf * (0.86 + rng2 * 0.26);
    let ang = f32(blade) * GG_GOLDEN_ANGLE
        + (gg_hash01(gcell, face, band, blade + 1u, 13u) - 0.5) * 0.7
        + gg_hash01(gcell, face, band, 0u, 3u) * GG_TAU; // per-clump spin

    // Clump tangent frame around the terrain normal.
    let up_t = normalize(east * nx + r_up * ny + north * nz);
    var right = east - up_t * dot(east, up_t);
    right = normalize(right);
    let fwd = cross(up_t, right);
    let outward = right * cos(ang) + fwd * sin(ang);
    let root = local_root + outward * rad;

    // Arch: a per-blade RANDOM azimuth pulled toward the clump's shared
    // facing. Deliberately decoupled from the blade's radial position angle —
    // tying them made every tuft a radial starburst fan, which reads as
    // planted plugs instead of a wild sward.
    let blade_az = gg_hash01(gcell, face, band, blade + 1u, 2u) * GG_TAU;
    let arch_rand = right * cos(blade_az) + fwd * sin(blade_az);
    let arch_clump = right * cos(clump_face_ang) + fwd * sin(clump_face_ang);
    let arch = normalize(mix(arch_rand, arch_clump, GG_CLUMP_FACING));

    let rng3 = gg_hash01(gcell, face, band, blade + 1u, 3u);
    let blade_h = profile_h * (0.82 + rng3 * 0.36) * (1.0 - dome_p * rf * 0.30);
    // Droop is mostly per-blade character, only slightly rim-graded (a strong
    // rim grade is the other half of the fountain-fan read).
    let d = droop_p * (0.40 + 0.45 * gg_hash01(gcell, face, band, blade + 1u, 17u) + 0.15 * rf);

    // Quadratic-Bézier centreline (push_grass_blade's arch).
    let tip_out = blade_h * (0.10 + d * 0.95);
    let tip_up = blade_h * (1.0 - d * 0.60);
    let ctrl_up = blade_h * (0.62 - d * 0.12);
    let ctrl_out = blade_h * (d * 0.40);
    let p0 = root;
    var p1 = root + up_t * ctrl_up + arch * ctrl_out;
    var p2 = root + up_t * tip_up + arch * tip_out;

    // ── Wind: gust field bends the whole Bézier from the root ───────────────
    // Wind in entity-local axes (rigid transform: local = Rᵀ · world).
    let wfl = mesh_functions::get_world_from_local(in.instance_index);
    let wind_local = vec3<f32>(
        dot(gg.wind.xyz, wfl[0].xyz),
        dot(gg.wind.xyz, wfl[1].xyz),
        dot(gg.wind.xyz, wfl[2].xyz),
    );
    let wind_t = vec2<f32>(dot(wind_local, east), dot(wind_local, north));
    let wind_t_len = max(length(wind_t), 1.0e-4);
    let wind_dir_t = wind_t / wind_t_len;
    // Stable tangent-plane metres for the gust phase (continuous across
    // re-anchors: phase + clump offset ≡ the body-fixed surface point).
    let gust_uv = vec2<f32>(
        dot(gg.phase.xyz, east) + clump_2d.x,
        dot(gg.phase.xyz, north) + clump_2d.y,
    );
    let gust = gg_gust(gust_uv, wind_dir_t, gg.time_fade.x);
    // Per-blade suppleness: style stiffness ± a little blade jitter.
    let stiff = clamp(
        style_b.w + (gg_hash01(gcell, face, band, blade + 1u, 14u) - 0.5) * 0.25,
        0.0, 1.0,
    );
    let bend = (GG_GUST_BASE + GG_GUST_AMP * gust) * (1.0 - stiff * 0.8);
    let wdir3 = normalize(east * wind_dir_t.x + north * wind_dir_t.y);
    // Hinge from the root: control point leans, tip leans harder and drops so
    // arc length is roughly preserved (a bent blade, not a stretched one).
    p1 = p1 + wdir3 * (bend * 0.35 * blade_h);
    p2 = p2 + wdir3 * (bend * blade_h) - up_t * (bend * bend * 0.55 * blade_h);

    var corner_t = GG_CORNER_T;
    var corner_side = GG_CORNER_SIDE;
    let t = corner_t[corner];
    let side_sign = corner_side[corner];

    // Bézier eval + tapered width along a slightly up-twisted side axis.
    let omt = 1.0 - t;
    let center = p0 * (omt * omt) + p1 * (2.0 * omt * t) + p2 * (t * t);
    let perp = normalize(cross(up_t, arch));
    let twist = (gg_hash01(gcell, face, band, blade + 1u, 5u) - 0.5) * 0.7;
    let side_axis = normalize(perp + up_t * (twist * 0.30));

    // Blade facing normal (⊥ side axis and the chord) — for the fragment's
    // rounded normal and the view widening below.
    let chord = p2 - p0;
    var blade_n = cross(side_axis, chord);
    let bn_len = length(blade_n);
    if (bn_len > 1.0e-5) {
        blade_n = blade_n / bn_len;
    } else {
        blade_n = up_t;
    }

    // View-dependent widening: an edge-on blade (view ray ∥ blade plane) gets
    // up to GG_VIEW_WIDEN extra width so it never thins to nothing.
    let local_view = vec3<f32>(
        dot(view.world_position - wfl[3].xyz, wfl[0].xyz),
        dot(view.world_position - wfl[3].xyz, wfl[1].xyz),
        dot(view.world_position - wfl[3].xyz, wfl[2].xyz),
    );
    let to_view = normalize(local_view - root);
    let facing = abs(dot(to_view, blade_n));
    // Distance-gated: a blade 2 m away subtends plenty of pixels edge-on —
    // widening there makes fat paddles. The widen exists for the mid/far
    // field where edge-on blades alias away.
    let view_dist = length(local_view - root);
    let widen = 1.0
        + GG_VIEW_WIDEN
            * (1.0 - smoothstep(0.0, 0.45, facing))
            * smoothstep(3.0, 18.0, view_dist);

    // Fuller mid-blade taper than the old linear needle: constant-ish lower
    // half, quadratic pinch to the tip.
    let taper = 1.0 - t * t;
    var width = profile_w
        * (0.78 + gg_hash01(gcell, face, band, blade + 1u, 6u) * 0.50)
        * widen;
    // Screen-space floor: never let a blade fall below ~a pixel of width.
    width = max(width, view_dist * GG_MIN_WIDTH_RAD);
    let local_pos = center + side_axis * (side_sign * 0.5 * width * taper);

    // ── Colour ───────────────────────────────────────────────────────────────
    // Landcover green + clump hue lean + per-blade drift; root darkening and
    // the value spread both COLLAPSE with band so far blades converge on the
    // terrain's own vegetation colour (no far-field dark speckle).
    let hue = (clump_hue - 0.5) * 0.12 + (gg_hash01(gcell, face, band, blade + 1u, 11u) - 0.5) * 0.06;
    var tint = vec3<f32>(veg.x * (1.0 + hue), veg.y, veg.z * (1.0 - hue));
    // Meadow patches: medium-scale (~9 m) mottling — value drift plus a warm
    // cast on the high patches — so the sward reads as living ground cover,
    // not one continuous tone. Static (not the scrolled gust field). Also
    // gates the dry-straw mix below into patchy stands.
    // (`patch` itself is a WGSL reserved word — see the wgsl-bevy skill.)
    let mottle = gg_vnoise(gust_uv * 0.11 + vec2<f32>(53.1, 71.7));
    tint = tint * mix(0.88, 1.12, mottle);
    tint = mix(tint, tint * vec3<f32>(1.10, 1.04, 0.74), mottle * mottle * 0.30);
    let value_spread = mix(0.24, 0.08, band_f);
    tint = tint * (1.0 - value_spread * 0.5 + value_spread * gg_hash01(gcell, face, band, blade + 1u, 7u));

    // Dry-straw blades: style dry_mix × landcover dryness picks whole blades;
    // every blade also dries a little toward the tip.
    let dryness = 1.0 - type_mix;
    // Mottle-gated so dry blades cluster into sun-scorched stands instead of
    // uniform salt-and-pepper.
    let dry_frac = style_b.y * (0.30 + 0.70 * dryness)
        * smoothstep(0.35, 0.75, gg_vnoise(gust_uv * 0.16 + vec2<f32>(9.2, 43.9))) * 2.2;
    // Straw is a HUE — renormalized to ~1.5× the local field luminance so dry
    // blades read sun-bleached against the green, not self-luminous.
    let veg_luma = dot(veg, vec3<f32>(0.30, 0.59, 0.11));
    let straw_luma = dot(GG_STRAW, vec3<f32>(0.30, 0.59, 0.11));
    let straw = GG_STRAW * (veg_luma * 1.5 / max(straw_luma, 1.0e-3));
    let is_dry = step(gg_hash01(gcell, face, band, blade + 1u, 15u), dry_frac);
    tint = mix(tint, straw, is_dry * 0.85);
    // Warm tip cast on green blades (sun-cured tips) — subtle, or the whole
    // field reads as ripe wheat.
    tint = mix(tint, tint * vec3<f32>(1.18, 1.06, 0.62), t * t * 0.22 * (1.0 - is_dry));

    // Lush saturation lift (band-faded so far blades stay terrain-matched).
    let tint_luma = dot(tint, vec3<f32>(0.30, 0.59, 0.11));
    tint = max(mix(vec3<f32>(tint_luma), tint, mix(GG_SATURATION, 1.0, band_f)), vec3<f32>(0.0));

    // Root→tip ramp, flattening with band (MSFS rule: far grass ≡ terrain).
    let root_floor = mix(0.64, 0.92, band_f);
    tint = tint * mix(root_floor, 1.0, t);

    let sheen = style_b.z * GG_SHEEN_GAIN;
    let transl = GG_TRANSLUCENCY * (0.6 + 0.4 * (1.0 - stiff));
    let phase_j = gg_hash01(gcell, face, band, blade + 1u, 9u);
    return gg_emit(
        in, local_pos, root, up_t,
        blade_n, side_axis, side_sign,
        tint, vec2<f32>(t, phase_j), band,
        vec4<f32>(sheen, transl, band_f, 0.0),
    );
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    var sky: SurfaceSky;
    sky.sun_color = in.sky0.rgb;
    sky.sun_scale = in.sky0.a;
    sky.sky_radiance = in.sky1.rgb;
    sky.ground_radiance = in.sky2.rgb;

    let up = gg.sky_up.xyz;
    let sun_dir = gg.sun_dir.xyz;
    let view_dir = normalize(view.world_position - in.world_position);
    let dist = distance(view.world_position, in.world_position);

    // ── Shading normal: rounded blade near, terrain lawn far ────────────────
    let n_terrain = normalize(in.world_normal);
    var blade_n = normalize(in.blade_n.xyz);
    // Two-sided strip: face the viewer.
    if (dot(blade_n, view_dir) < 0.0) {
        blade_n = -blade_n;
    }
    // Cylinder rounding across the width (side01 interpolates −1..1).
    let rounded = normalize(blade_n + normalize(in.blade_s.xyz) * (in.blade_n.w * GG_NORMAL_ROUND));
    let nb = mix(
        GG_NORMAL_BLEND_NEAR, GG_NORMAL_BLEND_FAR,
        smoothstep(GG_NORMAL_BLEND_LO_M, GG_NORMAL_BLEND_HI_M, dist),
    );
    let n = normalize(mix(n_terrain, rounded, nb));

    let t = in.blade_s.w;
    var s: FoliageSurface;
    s.albedo = in.color.rgb;
    s.normal_ws = n;
    // Tips transmit; roots don't (they're buried in the sward).
    s.translucency = in.extra.y * t;
    // Root ambient occlusion — the depth cue that makes a sward read 3-D.
    // Fades with band: far blades are the terrain, which carries its own AO.
    s.ambient_scale = mix(mix(GG_ROOT_AO, 1.0, t), 1.0, in.extra.z * 0.7);
    s.ambient_bleed = 0.0;
    let shadow = clamp(in.shadow, 0.0, 1.0);
    var lit = shade_foliage(s, view_dir, up, sun_dir, sky, shadow);

    // ── Specular sheen — the rolling highlight (per-style strength) ─────────
    let daylight = sun_daylight(dot(up, sun_dir));
    let hvec = normalize(sun_dir + view_dir);
    let ndh = clamp(dot(n, hvec), 0.0, 1.0);
    let sheen = pow(ndh, GG_SHEEN_POWER) * in.extra.x * (0.25 + 0.75 * t);
    lit += sheen * sky.sun_color * (sky.sun_scale * shadow * daylight);

    lit = object_aerial_recession(lit, sky, in.world_position, view.world_position);
    return vec4<f32>(lit, 1.0);
}
