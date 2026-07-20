// Solid-color planet placeholder.
//
// Fullscreen-quad ray-traced sphere. The vertex shader emits a fullscreen
// clip-space quad, and the fragment shader reconstructs each pixel's
// world-space ray direction from screen position, intersects the sphere,
// and shades it with simple Lambertian + ambient + planetshine. Used for
// bodies that don't have a terrain pipeline yet.
//
// Why fullscreen instead of a tightly-fit billboard: any 3D billboard
// sized to enclose the sphere's silhouette has failure modes when the
// camera is close and the body off-axis (corners behind near plane, or
// quad center far enough off-axis that the in-plane offsets don't move
// the corners onto the visible silhouette). Fullscreen sidesteps the
// whole class of bugs — every screen pixel gets a fragment, ray-sphere
// decides what to shade vs. discard.

#import bevy_pbr::mesh_view_bindings::view
#import bevy_pbr::mesh_functions::get_world_from_local
#import thalos::lighting::{SceneLighting, shade_hapke_surface, SCENE_FLUX_SCALE}
#import thalos::water::shade_ocean
#import thalos::atmosphere::{
    AtmosphereBlock,
    integrate_atmosphere,
    integrate_atmosphere_multiscatter,
    atmosphere_jitter,
    atmosphere_scattering_active,
}

const PI: f32 = 3.14159265358979323846;

// Fraction of the physically-integrated atmospheric in-scatter kept as on-disc
// airlight for the distant billboard. The full sky-dome `strength` (atmos_geom.z,
// tuned bright for the rim halo / star-crush) washes the whole disc milky white;
// but zeroing it makes the planet read airless from space — a real planet carries
// a soft blue veil across its WHOLE disc (thickening toward the limb where the
// view skims the air edge-on), not just a hairline rim.
//
// The billboard now runs the full multi-scatter integral (matching the ground
// `BodySky` path), so the *diffuse* second-order blue fill — most of what makes a
// planet-from-space look properly atmospheric — is physical, not faked. That fill
// (scaled by `multi_gain`, atmos_geom.w ≈ 3) is substantial, so this dial sits
// well below 1: it is the single overall airlight knob (raise = hazier/bluer
// planet, lower = crisper). The in-scatter is already air-mass-graded by chord
// length, so the sub-observer point stays subtle while the limb glows.
// Screenshot-tuned: washy/milky → lower toward 0.08; airless → raise toward 0.3.
const DISC_AIRLIGHT_FRACTION: f32 = 0.15;

struct SolidPlanetParams {
    radius:  f32,
    albedo:  vec4<f32>,      // xyz = flat colour, w = use albedo_cube (>= 0.5)
    orientation: vec4<f32>,  // quaternion (xyzw): render-space dir -> body-fixed
    scene:   SceneLighting,
    atmosphere: AtmosphereBlock,
}

@group(3) @binding(0) var<uniform> params: SolidPlanetParams;

// Baked impostor albedo cube (continents + oceans), sampled by the body-fixed
// normal. Only the body pass declares/uses it; the halo pass has no albedo.
#ifndef HALO_PASS
@group(3) @binding(1) var albedo_cube_tex: texture_cube<f32>;
@group(3) @binding(2) var albedo_cube_sampler: sampler;
// Multi-scatter LUT (the same one `BodySkyMaterial` binds). Body pass only —
// the diffuse second-order fill that gives the disc its pervasive blue haze
// (single scattering alone leaves it looking airless from space).
@group(3) @binding(3) var ms_lut_tex: texture_2d<f32>;
@group(3) @binding(4) var ms_lut_sampler: sampler;
#endif

// Rotate vector `v` by unit quaternion `q` (xyz = axis·sin, w = cos).
fn rotate_quat(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) sphere_center:  vec3<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let model = get_world_from_local(in.instance_index);
    let sphere_center = (model * vec4(0.0, 0.0, 0.0, 1.0)).xyz;

    // Fullscreen clip-space quad. The mesh is `Rectangle::new(2.0, 2.0)`
    // with corners at ±1 in local x/y, so passing the raw in.position
    // through covers the entire viewport in NDC, regardless of where the
    // body is. The model transform is intentionally ignored.
    //
    // z = 1.0 is the near plane in Bevy's reverse-Z, so the rasterizer
    // accepts every fragment by the early depth test. The fragment
    // shader then writes the real per-pixel hit depth via
    // @builtin(frag_depth) on hit, or discards on miss (no depth or
    // color write).
    var out: VertexOutput;
    out.clip_position = vec4(in.position.x, in.position.y, 1.0, 1.0);
    out.sphere_center = sphere_center;
    return out;
}

struct FragOutput {
    @location(0)         color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

/// View-ray entry/exit distances through the atmosphere shell (`radius +
/// atmos_geom.x`). Used by both the halo and body passes to bound the
/// scattering raymarch. Mirrors `atmosphere_shell_hit` in `planet_impostor.wgsl`.
struct ShellHit {
    valid: bool,
    t_enter: f32,
    t_exit: f32,
}

fn atmosphere_shell_hit(cam_pos: vec3<f32>, ray_dir: vec3<f32>, center: vec3<f32>) -> ShellHit {
    let alt = params.atmosphere.atmos_geom.x;
    if alt <= 0.0 {
        return ShellHit(false, 0.0, 0.0);
    }
    let r_outer = params.radius + alt;
    let oc = cam_pos - center;
    let half_b = dot(oc, ray_dir);
    let c_o = dot(oc, oc) - r_outer * r_outer;
    let disc_o = half_b * half_b - c_o;
    if disc_o < 0.0 {
        return ShellHit(false, 0.0, 0.0);
    }
    let sq = sqrt(disc_o);
    let t_far = -half_b + sq;
    if t_far <= 0.0 {
        return ShellHit(false, 0.0, 0.0);
    }
    return ShellHit(true, max(-half_b - sq, 0.0), t_far);
}

@fragment
fn fragment(in: VertexOutput) -> FragOutput {
    let cam_pos = view.world_position;

    // Reconstruct the world-space ray through this fragment from screen
    // position. Avoid `world_from_clip` and large-magnitude world-space
    // arithmetic — at orbital distances both `world_from_clip * ndc`
    // and `cam_pos` are millions of meters and the subtraction loses
    // f32 precision, which collapses `dpdx/dpdy` of any downstream
    // texture coords across adjacent pixels. The camera-basis form
    // works in small numbers (basis vectors are unit, ndc and tan_fov
    // are O(1)) so adjacent-pixel `ray_dir` differs by a real angular
    // delta rather than precision noise.
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

    let center = in.sphere_center;

    // Ray-sphere intersection against the solid body radius.
    let oc     = cam_pos - center;
    let half_b = dot(oc, ray_dir);
    let c      = dot(oc, oc) - params.radius * params.radius;
    let disc   = half_b * half_b - c;
    let t      = -half_b - sqrt(max(disc, 0.0));
    // `disc < 0` → ray never reaches the sphere; `t < 0` → sphere fully behind
    // the camera. Either way the ray sees only sky/atmosphere here.
    let is_miss = disc < 0.0 || t < 0.0;

    let star     = params.scene.stars[0];
    let sun_dir  = star.dir_flux.xyz;
    let sun_flux = star.dir_flux.w;

#ifdef HALO_PASS
    // Rim glow: keep only the atmospheric in-scatter on rays that miss the
    // solid disc, integrated along the full atmosphere chord. The body pass
    // (the opaque `SolidPlanetMaterial`) owns every surface-hit fragment.
    if !is_miss {
        discard;
    }
    if !atmosphere_scattering_active(params.atmosphere) {
        discard;
    }
    let shell = atmosphere_shell_hit(cam_pos, ray_dir, center);
    if !shell.valid {
        discard;
    }
    let jitter = atmosphere_jitter(in.clip_position.xy);
    let scatter = integrate_atmosphere(
        cam_pos, ray_dir, center, sun_dir,
        sun_flux * SCENE_FLUX_SCALE,
        shell.t_enter, shell.t_exit,
        params.radius, params.atmosphere, jitter,
    );
    // Premultiplied alpha = how much of the background the column occludes.
    // Rec.709 luminance of the per-channel transmittance gives one coherent
    // alpha fading between vacuum (T=1, α=0) and an opaque chord (T=0, α=1).
    let alpha = clamp(1.0 - dot(scatter.transmittance, vec3<f32>(0.2126, 0.7152, 0.0722)),
                      0.0, 1.0);
    let lum = dot(scatter.in_scatter, vec3<f32>(0.2126, 0.7152, 0.0722));
    if alpha < 0.002 && lum < 0.0005 {
        discard;
    }
    // Closest-approach depth gives a sensible silhouette depth so opaque
    // bodies in front of the halo (and nearer halos) depth-test correctly.
    let closest = cam_pos + ray_dir * max(-half_b, 0.0);
    let clip_halo = view.clip_from_world * vec4(closest, 1.0);
    return FragOutput(vec4(scatter.in_scatter, alpha), clip_halo.z / clip_halo.w);
#else
    if is_miss {
        discard;
    }

    let hit    = cam_pos + t * ray_dir;
    let normal = normalize(hit - center);

    // Baked-impostor albedo: sample the continents/oceans cube by the body-fixed
    // normal (so it co-rotates with the planet). `albedo.w < 0.5` = solid-colour
    // body → use the flat colour and never touch the (blank) cube. The cube's
    // alpha is the water mask (baked: 1 = ocean, 0 = land).
    var surface_albedo = params.albedo.xyz;
    var is_water = false;
    if params.albedo.w >= 0.5 {
        let n_body = rotate_quat(params.orientation, normal);
        let cube = textureSampleLevel(albedo_cube_tex, albedo_cube_sampler, n_body, 0.0);
        surface_albedo = cube.rgb;
        is_water = cube.a >= 0.5;
    }

    // Shade through the shared Hapke regolith BRDF — the SAME routine the ground
    // LOD uses for airless bodies (`body_terrain.wgsl`, SURFACE_REGOLITH) and the
    // procedural-body impostor. A moon's distant disc then reads as a real lit
    // sphere (opposition surge, limb behaviour, correct terminator so phases show)
    // and reconverges with its own ground across the LOD swap, instead of the old
    // flat Lambert disc. The billboard has no relief, so the shading and geometric
    // normals are both the sphere normal; planetshine (earthshine from the parent)
    // and eclipse occlusion are folded in by the helper. Roughness ~0.85 ≈ lunar
    // regolith.
    let view_dir = normalize(cam_pos - hit);
    var lit = vec3<f32>(0.0);
    if is_water {
        // Distant ocean: the shared water BRDF (Fresnel + GGX sun glint), the
        // SAME `thalos::water::shade_ocean` the ship-surface ocean uses. view_dist
        // huge → waves off (smooth sphere, so the glint is a crisp specular spot);
        // the baked depth-graded blue is the water subsurface colour.
        let water_cd = vec4(surface_albedo, 120.0);
        lit = shade_ocean(
            hit,
            normal,
            view_dir,
            1.0e9,
            0.0,
            sun_dir,
            sun_flux * SCENE_FLUX_SCALE,
            water_cd,
            1.0e6,
            // Far shore sentinels: no shore interaction at impostor scale.
            1.0e6,
            1.0e9,
            vec3<f32>(0.0, 0.0, 1.0),
            1.0e9,
        );
    } else {
        // Land: shared Hapke regolith BRDF — the SAME routine the ground
        lit = shade_hapke_surface(
            surface_albedo,
            0.85,
            normal,
            normal,
            view_dir,
            hit,
            sun_dir,
            sun_flux,
            params.scene,
            1.0,
        );
    }

    // Aerial perspective + daylight haze across the lit disc: integrate the
    // atmosphere from the shell entry to the surface hit, dim the surface by
    // the view transmittance, and add the in-scatter. Vacuum bodies early-out
    // in `atmosphere_scattering_active`, so this is a no-op for airless solids.
    if atmosphere_scattering_active(params.atmosphere) {
        let shell = atmosphere_shell_hit(cam_pos, ray_dir, center);
        if shell.valid {
            let jitter = atmosphere_jitter(in.clip_position.xy);
            // Multi-scatter integral (matches the ground `BodySky` path): adds the
            // diffuse second-order blue fill that single scattering omits, so the
            // disc reads as a real atmosphere-veiled planet from space, not a rim.
            let scatter = integrate_atmosphere_multiscatter(
                cam_pos, ray_dir, center, sun_dir,
                sun_flux * SCENE_FLUX_SCALE,
                shell.t_enter, t,
                params.radius, params.atmosphere, jitter,
                ms_lut_tex, ms_lut_sampler,
            );
            // Additive airlight, kept at a fraction of the full in-scatter so the
            // disc shows a real blue veil (strongest at the limb — longer chord)
            // without the sky-dome `strength` washing it to milky white. Physical
            // transmittance is left untouched (it dims/reddens the surface along
            // the same path). See `DISC_AIRLIGHT_FRACTION`.
            lit = lit * scatter.transmittance + scatter.in_scatter * DISC_AIRLIGHT_FRACTION;
        }
    }

    let clip = view.clip_from_world * vec4(hit, 1.0);
    return FragOutput(vec4(lit, 1.0), clip.z / clip.w);
#endif
}
