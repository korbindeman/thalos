// Octahedral tree impostor shader (far band of the tree LOD cascade).
//
// Vertex: billboards one quad per tree (4 verts share the tree base; the corner
// id is in UV_0) to face the camera, oriented to the captured view basis so the
// card matches the atlas projection, sized from the per-species bounding sphere,
// and scale-faded about the base from the craft anchor (grow-from-zero — the
// same seamless, zoom-independent fade the mesh trees use).
//
// Fragment: maps the camera→tree view direction (in the tree's terrain frame)
// to hemisphere-octahedral atlas coords, bilinearly blends the 4 surrounding
// captured views (coverage-weighted, so silhouettes don't ghost), alpha-tests on
// blended coverage, rotates the blended OBJECT-frame normal into world, and
// lights it through the SAME `thalos::lighting` sky model the mesh trees and
// ground use — so the forest reads continuous across the mesh→impostor handoff.

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}
#import thalos::lighting::{compute_surface_sky, FoliageSurface, shade_foliage}
#import thalos::foliage::foliage_hue_tint

// Mirror of GrassParams (shared with grass.wgsl / tree.wgsl) — field order
// load-bearing.
struct TreeParams {
    sun_dir: vec4<f32>,   // xyz = toward star, w = sun flux (lux × exposure)
    wind: vec4<f32>,      // xyz = wind dir, w = sway amplitude (m)
    time_fade: vec4<f32>, // x = time, y = fade start, z = fade end
    sky_up: vec4<f32>,    // xyz = local radial up
    sky_tau: vec4<f32>,   // xyz = Rayleigh τ_v, w = atmosphere strength
    anchor: vec4<f32>,    // xyz = (craft − camera) offset (render space), w = 1 valid / 0 camera; ref = view.world_position + offset
}

struct ImpostorParams {
    grid: vec4<f32>,  // x = cells N, y = species count, z = alpha cutoff, w = v-flip
    atlas: vec4<f32>, // x = cell fill fraction
    species_geo: array<vec4<f32>, 4>, // per species: x = radius, y = centre height
}

@group(3) @binding(0) var<uniform> tree: TreeParams;
@group(3) @binding(1) var<uniform> imp: ImpostorParams;
@group(3) @binding(2) var albedo_tex: texture_2d<f32>;
@group(3) @binding(3) var albedo_smp: sampler;
@group(3) @binding(4) var normal_tex: texture_2d<f32>;
@group(3) @binding(5) var normal_smp: sampler;

const TAU: f32 = 6.28318530717958647;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>, // tree base (tile-centre-relative)
    @location(1) normal: vec3<f32>,   // terrain up (body-fixed)
    @location(2) uv0: vec2<f32>,      // card corner 0..1
    @location(3) uv1: vec2<f32>,      // x = instance scale, y = species index
    @location(5) color: vec4<f32>,    // rgb = tint, a = yaw / TAU
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) d_local: vec3<f32>,
    @location(2) card_uv: vec2<f32>,
    @location(3) frame_t: vec3<f32>,
    @location(4) frame_u: vec3<f32>,
    @location(5) frame_b: vec3<f32>,
    @location(6) tint_seed: vec4<f32>, // rgb = tint, a = seed
    @location(7) species: f32,
}

fn hash1(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
}

// Card basis for a view direction (object→camera), matching `impostor_bake_rotation`.
fn view_basis_right_up(fwd: vec3<f32>) -> mat2x3<f32> {
    let f = normalize(fwd);
    var up_ref = vec3<f32>(0.0, 1.0, 0.0);
    if abs(f.y) > 0.999 {
        up_ref = vec3<f32>(0.0, 0.0, 1.0);
    }
    let r = normalize(cross(up_ref, f));
    let u = cross(f, r);
    return mat2x3<f32>(r, u);
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    let base = in.position;
    let base_w =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(base, 1.0)).xyz;
    let up_w = normalize(mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index));

    // Tree terrain frame in world: up = terrain normal, tangent/bitangent rotated
    // by the per-tree yaw so impostors don't all face-align identically.
    var ref_axis = vec3<f32>(0.0, 0.0, 1.0);
    if abs(up_w.y) > 0.99 {
        ref_axis = vec3<f32>(1.0, 0.0, 0.0);
    }
    let t0 = normalize(cross(ref_axis, up_w));
    let b0 = cross(up_w, t0);
    let yaw = in.color.a * TAU;
    let cw = cos(yaw);
    let sw = sin(yaw);
    let tangent_w = t0 * cw + b0 * sw;
    let bitangent_w = -t0 * sw + b0 * cw;

    // View direction (object→camera) in the tree frame.
    let view_w = normalize(view.world_position - base_w);
    let d_local = vec3<f32>(
        dot(view_w, tangent_w),
        dot(view_w, up_w),
        dot(view_w, bitangent_w),
    );

    // Card axes from the view direction (object-local) → world.
    let basis = view_basis_right_up(d_local);
    let right_l = basis[0];
    let up_l = basis[1];
    let right_w = right_l.x * tangent_w + right_l.y * up_w + right_l.z * bitangent_w;
    let cup_w = up_l.x * tangent_w + up_l.y * up_w + up_l.z * bitangent_w;

    // Per-species geometry → card size + centre.
    let sp = u32(in.uv1.y + 0.5);
    let geo = imp.species_geo[sp];
    let scale = in.uv1.x;
    var half = geo.x * scale;
    let center_w = base_w + up_w * (geo.y * scale);

    // Clipmap scale-fade (grow from zero across the ring's near/far edges),
    // measured from the craft anchor so adjacent rings cross-fade seamlessly.
    // Anchor is a camera-relative OFFSET (ship − camera), rebuilt in the current
    // frame's render origin so it survives big_space floating-origin recentres
    // (an absolute anchor jumps a cell and pops impostors in/out while moving —
    // see `rendering::grass`). offset 0 → camera.
    let ref_pos = view.world_position + tree.anchor.xyz;
    let inst_dist = distance(ref_pos, base_w);
    let near_edge = tree.time_fade.y;
    let far_edge = tree.time_fade.z;
    let band = max(tree.time_fade.w, 1.0);
    let fade_in = smoothstep(near_edge - band, near_edge + band, inst_dist);
    let fade_out = 1.0 - smoothstep(far_edge - band, far_edge + band, inst_dist);
    let grow = fade_in * fade_out;
    half = half * grow;

    let c = in.uv0 * 2.0 - 1.0;
    let world_pos = center_w + c.x * half * right_w + c.y * half * cup_w;

    var out: VertexOutput;
    out.world_position = world_pos;
    out.clip_position = position_world_to_clip(world_pos);
    out.d_local = d_local;
    out.card_uv = in.uv0;
    out.frame_t = tangent_w;
    out.frame_u = up_w;
    out.frame_b = bitangent_w;
    out.tint_seed = vec4<f32>(in.color.rgb, hash1(base));
    out.species = in.uv1.y;
    return out;
}

// Hemisphere-octahedral encode (y up), inverse of the Rust `hemioct_decode`.
fn hemioct_encode(dir: vec3<f32>) -> vec2<f32> {
    var d = dir;
    d.y = max(d.y, 0.0);
    let l1 = abs(d.x) + abs(d.y) + abs(d.z);
    let p = d / max(l1, 1.0e-5);
    return vec2<f32>(p.x + p.z, p.x - p.z) * 0.5 + 0.5;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let n_cells = imp.grid.x;
    let species_count = max(imp.grid.y, 1.0);
    let cutoff = imp.grid.z;
    let v_flip = imp.grid.w;
    let fill = imp.atlas.x;
    let sp = floor(in.species + 0.5);

    let enc = hemioct_encode(in.d_local);
    let cf = enc * n_cells - 0.5;
    let i0 = floor(cf);
    let fr = cf - i0;

    // Card UV (billboard ±radius) → cell UV: the bounding sphere occupies `fill`
    // of the cell, centred, so the rest is the anti-bleed gutter.
    var cell_uv = vec2<f32>(0.5) + (in.card_uv - vec2<f32>(0.5)) * fill;
    if v_flip > 0.5 {
        cell_uv.y = 1.0 - cell_uv.y;
    }

    let v_total = n_cells * species_count;
    var acc_alb = vec3<f32>(0.0);
    var acc_n = vec3<f32>(0.0);
    var acc_cov = 0.0;

    for (var dj = 0; dj < 2; dj = dj + 1) {
        for (var di = 0; di < 2; di = di + 1) {
            let ci = clamp(i0.x + f32(di), 0.0, n_cells - 1.0);
            let cj = clamp(i0.y + f32(dj), 0.0, n_cells - 1.0);
            let wx = select(1.0 - fr.x, fr.x, di == 1);
            let wy = select(1.0 - fr.y, fr.y, dj == 1);
            let w = wx * wy;

            let u = (ci + cell_uv.x) / n_cells;
            let vv = 1.0 - (sp * n_cells + cj + cell_uv.y) / v_total;
            let auv = vec2<f32>(u, vv);

            let a = textureSampleLevel(albedo_tex, albedo_smp, auv, 0.0);
            let nd = textureSampleLevel(normal_tex, normal_smp, auv, 0.0);
            // `a.rgb` is ALREADY premultiplied by coverage: the atlas is cleared to
            // transparent black, so the bilinear filter blends covered (rgb, a=1)
            // with cleared (0, a=0) → rgb = colour × coverage near the silhouette.
            // Accumulate it straight (÷ acc_cov below un-premultiplies); multiplying
            // by `a.a` again would darken the silhouette edge to black (the toon
            // rim). Normals aren't premultiplied that way, so they keep the `a.a`
            // coverage weight to down-weight the garbage edge normals.
            acc_alb += a.rgb * w;
            acc_n += (nd.rgb * 2.0 - 1.0) * a.a * w;
            acc_cov += a.a * w;
        }
    }

    if acc_cov < cutoff {
        discard;
    }

    let albedo = acc_alb / max(acc_cov, 1.0e-4);
    let n_local = normalize(acc_n / max(acc_cov, 1.0e-4));
    let n_world =
        normalize(n_local.x * in.frame_t + n_local.y * in.frame_u + n_local.z * in.frame_b);

    // Same hemisphere sky/sun environment the grass + ground + mesh trees build.
    let sun_dir = tree.sun_dir.xyz;
    let up = tree.sky_up.xyz;
    let sky = compute_surface_sky(tree.sky_tau.xyz, tree.sky_tau.w, up, sun_dir, tree.sun_dir.w);

    // Per-instance hue via the SHARED `foliage_hue_tint` — the SAME jitter the
    // mesh trees apply, so a stand varies identically across the mesh→impostor
    // handoff. The baked albedo is already the near-tree colour (the bake calls
    // the same `foliage_base_albedo`), so impostor and mesh now read continuous.
    let tint = albedo * in.tint_seed.rgb * foliage_hue_tint(in.tint_seed.a);

    // Shade through the shared `shade_foliage` with the SAME canopy parameters as
    // the mesh trees (0.8 ambient, leaf transmit, 0.40 wrap), so the forest reads
    // continuous across the mesh→impostor handoff. The card is canopy foliage, so
    // it transmits; sun-shadow on impostors lands in Phase 2b (fully lit for now).
    var s: FoliageSurface;
    s.albedo = tint;
    s.normal_ws = n_world;
    s.translucency = 1.0;
    s.ambient_scale = 0.8;
    s.ambient_bleed = 0.5;
    let view_dir = normalize(view.world_position - in.world_position);
    let lit = shade_foliage(s, view_dir, up, sun_dir, sky, 1.0);
    return vec4<f32>(lit, 1.0);
}
