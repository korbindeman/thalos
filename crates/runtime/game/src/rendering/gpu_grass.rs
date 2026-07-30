//! GPU grass field driver — anchoring, control-window rebuilds, shading params.
//!
//! The engine side (`thalos_body_render::ground::gpu_grass`) owns the template
//! mesh, the material, and the window math; this driver owns the lifecycle:
//!
//! - **One field entity** on the active vegetated body (the CPU grass driver's
//!   body-selection rule), anchored with the runway/grass f64 pattern: a
//!   root-grid big_space child re-posed every frame from the body state.
//! - **Anchor registration** (cheap, every ~3 m of ground movement): re-derive
//!   the tangent frame + per-band lattice registration and push them into the
//!   material params. Blade placement hashes off *global* cells, so blades
//!   stay body-fixed across re-registrations.
//! - **Control-window rebuilds** (expensive, every ~25 m or on terrain
//!   revision/structure change): fill height + aux mask textures off-thread on
//!   `AsyncComputeTaskPool`, swap on completion. The old window keeps
//!   rendering while the new one builds — no gap, no churn.
//! - **Per-frame params**: sun/wind/sky/anchor/time mirroring the CPU grass
//!   material update, plus the live sun-shadow cascade rebind, so the GPU
//!   field and the remaining CPU card ring shade identically.
//!
//! Gated by `GraphicsSettings::{grass, gpu_grass}`; when active, the CPU
//! driver parks its blade rings (0–1) and keeps only the far card ring.

use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::camera::primitives::Aabb;
use bevy::camera::visibility::RenderLayers;
use bevy::image::Image;
use bevy::light::NotShadowCaster;
use bevy::math::{DVec3, UVec4, Vec3, Vec4};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use big_space::prelude::{BigSpace, CellCoord, Grid};

use thalos_body_render::{
    AU_M, GPU_GRASS_BAND_COUNT, GPU_GRASS_WINDOW_HALF_M, GPU_GRASS_WINDOW_SIZE_PX, GpuGrassAnchor,
    GpuGrassMaterial, GpuGrassWindow, GpuGrassWindowInput, LIGHT_AT_1AU, TerrainShadingStyle,
    build_gpu_grass_template, build_gpu_grass_window, fallback_shadow_map, gpu_grass_anchor,
    gpu_grass_style_table,
};
use thalos_physics_local::HeightSourceRegistry;
use thalos_world::BodyId;

use crate::SimStage;
use crate::coords::SHIP_LAYER;
use crate::graphics_settings::GraphicsSettings;
use crate::rendering::grass::grass_scatter_regions;
use crate::rendering::ground_terrain::terrain_shading_style_for;
use crate::rendering::real_space::{RealSpaceRoot, real_space_grid};
use crate::rendering::types::CameraExposure;
use crate::rendering::view_anchor::ViewAnchor;
use crate::solar_system_state::{SimulationState, SolarSystemState, sync_solar_system_state};
use crate::structures::StructureRegistry;

// ── Tuning ────────────────────────────────────────────────────────────────────
/// Re-register the anchor (frame + lattice params — cheap) when the player's
/// ground point drifts this far from it. Must stay under the engine's
/// `GPU_GRASS_SNAP_SLACK_M` so the template annuli keep covering their bands.
const ANCHOR_STEP_M: f64 = 3.0;
/// Rebuild the control window (expensive, async) when the anchor drifts this
/// far from the window centre. `WINDOW_HALF (420) − reach+fade+slack (~372)`
/// leaves ~48 m of margin, so 25 m keeps blades on valid data with slack for
/// the build time.
const WINDOW_REFILL_M: f64 = 25.0;
/// Staleness scan interval (terrain revision / structure changes), seconds.
const WINDOW_CHECK_S: f32 = 2.0;
/// Rebuild on revision change only when the anchor height actually moved
/// (mirrors the CPU driver's `GRASS_REBUILD_DELTA_M` noise-floor reasoning).
const WINDOW_REBUILD_DELTA_M: f32 = 0.5;
/// Altitude collapse band. Deliberately higher than the old CPU rings' 150 to 300 m:
/// the GPU field is O(1) memory at any reach, and the low-aerial view (climb-out,
/// pattern altitude) is exactly where bald terrain used to snap in. Beyond this
/// the band-2 terrain-shading grass in `body_terrain.wgsl` carries the field.
const FADE_LO_AGL_M: f64 = 250.0;
const FADE_HI_AGL_M: f64 = 500.0;
/// Above this AGL the field hides entirely (blades are long collapsed).
const HIDE_AGL_M: f64 = 550.0;
/// Wind sway amplitude at the blade tip, metres (the CPU driver's value).
const WIND_SWAY_M: f32 = 0.06;
/// Landcover coordinate period (m) the sampling phase folds on — must match
/// `thalos::landcover` / `body_terrain.wgsl`'s `DETAIL_COORD_PERIOD_M`.
const LANDCOVER_PERIOD_M: f64 = 4000.0;

/// Marker + body-fixed pose for the field entity (the grass-tile visual
/// pattern: re-posed in f64 every frame).
#[derive(Component)]
struct GpuGrassVisual {
    body_id: BodyId,
    anchor_surface_body: DVec3,
}

/// Driver state. **Sole writer:** the systems in this module.
#[derive(Resource, Default)]
struct GpuGrassState {
    body: Option<BodyId>,
    entity: Option<Entity>,
    material: Option<Handle<GpuGrassMaterial>>,
    template: Option<Handle<Mesh>>,
    height_img: Option<Handle<Image>>,
    aux_img: Option<Handle<Image>>,
    /// The live registration (None until the first window lands).
    anchor: Option<GpuGrassAnchor>,
    /// The window's own anchor (its texel-grid origin) + entity height.
    window_anchor: Option<GpuGrassAnchor>,
    anchor_height_m: f32,
    /// Macro landcover moisture at the window anchor (see
    /// `GpuGrassWindow::anchor_moisture`), forwarded to `params.phase.w`.
    anchor_moisture: f32,
    built_revision: u64,
    built_region_count: usize,
    agl_m: f64,
    in_flight: Option<(Task<GpuGrassWindow>, GpuGrassAnchor, usize)>,
    check_timer: f32,
}

pub struct GpuGrassPlugin;

impl Plugin for GpuGrassPlugin {
    fn build(&self, app: &mut App) {
        // Strictly chained: anchor re-registration (drive) must land in the
        // SAME frame's transform re-pose AND material params, or the entity
        // pose and the shader's lattice registration disagree for a frame —
        // which strobed every ~3 m of camera travel.
        app.init_resource::<GpuGrassState>().add_systems(
            Update,
            (
                drive_gpu_grass,
                finalize_gpu_grass,
                update_gpu_grass_transform,
                update_gpu_grass_material,
            )
                .chain()
                .in_set(SimStage::Sync)
                .after(sync_solar_system_state),
        );
    }
}

/// Park everything: despawn the field and forget the window.
fn park(state: &mut GpuGrassState, commands: &mut Commands) {
    if let Some(entity) = state.entity.take() {
        commands.entity(entity).despawn();
    }
    state.body = None;
    state.anchor = None;
    state.window_anchor = None;
    state.in_flight = None;
}

/// Body pick, AGL gate, anchor drift tracking, and window-rebuild dispatch.
#[allow(clippy::too_many_arguments)]
fn drive_gpu_grass(
    mut state: ResMut<GpuGrassState>,
    solar: Res<SolarSystemState>,
    sim: Res<SimulationState>,
    height_sources: Res<HeightSourceRegistry>,
    structures: Res<StructureRegistry>,
    paved: Res<crate::base_editor::PavedFootprints>,
    graphics: Res<GraphicsSettings>,
    view_anchor: Res<ViewAnchor>,
    time: Res<Time>,
    mut visuals: Query<&mut GpuGrassVisual>,
    mut commands: Commands,
) {
    if solar.states.is_none() {
        return;
    }
    if !graphics.grass || !graphics.gpu_grass {
        if state.body.is_some() {
            park(&mut state, &mut commands);
        }
        return;
    }

    // Active body: the view anchor's (nearest terrain-backed) body, when it can
    // grow grass. The field follows the VIEW — see `rendering::view_anchor`.
    let anchored = view_anchor.resolved.filter(|a| {
        sim.system
            .bodies
            .get(a.body)
            .is_some_and(|b| terrain_shading_style_for(b) == TerrainShadingStyle::Vegetated)
    });
    let Some(view) = anchored else {
        if state.body.is_some() {
            park(&mut state, &mut commands);
        }
        return;
    };
    let body_id = view.body;
    if state.body != Some(body_id) {
        park(&mut state, &mut commands);
        state.body = Some(body_id);
    }

    let radius_m = view.radius_m;
    let Some(height_source) = height_sources.get(body_id) else {
        return;
    };

    let ground_dir = view.cam_dir;
    state.agl_m = view.agl_m;

    // High above the field: keep the entity (cheap when hidden by the material
    // update below) but don't chase the ground with window rebuilds.
    if state.agl_m > HIDE_AGL_M {
        return;
    }

    // ── Anchor re-registration (cheap, every ~3 m) ──────────────────────────
    // Re-derive the tangent frame + lattice registration under the camera.
    // Runs HERE — first in the chained flow — so this frame's transform
    // re-pose and material params both see the new anchor together (doing it
    // in the material update raced the transform and flickered every step).
    // Blade placement hashes off global cells, so blades stay body-fixed.
    if state.window_anchor.is_some() {
        let drift_m = state
            .anchor
            .map(|a| a.dir.angle_between(ground_dir) * radius_m)
            .unwrap_or(f64::MAX);
        if drift_m > ANCHOR_STEP_M {
            let anchor = gpu_grass_anchor(ground_dir, radius_m);
            state.anchor = Some(anchor);
            // The entity stays seated at the WINDOW's anchor height — blade
            // heights are window-relative to it (`frame_east.w`), so absolute
            // blade positions are invariant across re-registrations.
            if let Some(entity) = state.entity
                && let Ok(mut visual) = visuals.get_mut(entity)
            {
                visual.anchor_surface_body = anchor.dir * (radius_m + state.anchor_height_m as f64);
            }
        }
    }

    // Supersede a stale in-flight window: at speed the camera outruns a
    // build — landing it would seat the field on data already ~stale and
    // immediately re-trigger another rebuild. Drop it and refill from here.
    if let Some((_, pending, _)) = &state.in_flight
        && pending.dir.angle_between(ground_dir) * radius_m > 2.0 * WINDOW_REFILL_M
    {
        state.in_flight = None;
    }

    // Window rebuild triggers: none yet, drifted too far, or stale data.
    state.check_timer += time.delta_secs();
    let regions = grass_scatter_regions(&structures, &paved, body_id, radius_m);
    let mut want_refill = false;
    match state.window_anchor {
        None => want_refill = true,
        Some(window_anchor) => {
            let drift_m = window_anchor.dir.angle_between(ground_dir) * radius_m;
            if drift_m > WINDOW_REFILL_M {
                want_refill = true;
            } else if state.check_timer >= WINDOW_CHECK_S {
                state.check_timer = 0.0;
                if regions.len() != state.built_region_count {
                    want_refill = true;
                } else if height_source.revision() != state.built_revision {
                    let h = height_source
                        .sample_height_m(window_anchor.dir.as_vec3(), 0.5)
                        .unwrap_or(state.anchor_height_m);
                    if (h - state.anchor_height_m).abs() > WINDOW_REBUILD_DELTA_M {
                        want_refill = true;
                    } else {
                        // Same ground — adopt the new revision without a rebuild.
                        state.built_revision = height_source.revision();
                    }
                }
            }
        }
    }

    if want_refill && state.in_flight.is_none() {
        let anchor = gpu_grass_anchor(ground_dir, radius_m);
        let input = GpuGrassWindowInput {
            height_source: Arc::clone(&height_source),
            radius_m,
            anchor,
            scatter_regions: Arc::new(regions.clone()),
            size_px: GPU_GRASS_WINDOW_SIZE_PX,
            half_m: GPU_GRASS_WINDOW_HALF_M,
        };
        let task = AsyncComputeTaskPool::get().spawn(async move { build_gpu_grass_window(&input) });
        state.in_flight = Some((task, anchor, regions.len()));
    }
}

/// Poll the window build; on completion upload the textures, seat the entity,
/// and register the anchor.
#[allow(clippy::too_many_arguments)]
fn finalize_gpu_grass(
    mut state: ResMut<GpuGrassState>,
    solar: Res<SolarSystemState>,
    root: Option<Res<RealSpaceRoot>>,
    sim: Res<SimulationState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GpuGrassMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut commands: Commands,
) {
    let Some((task, ..)) = state.in_flight.as_mut() else {
        return;
    };
    let Some(window) = block_on(poll_once(task)) else {
        return;
    };
    let Some((_, anchor, region_count)) = state.in_flight.take() else {
        return;
    };
    let (Some(states), Some(root), Some(body_id)) = (solar.states.as_deref(), root, state.body)
    else {
        return;
    };
    let (Some(body_state), Some(body)) = (states.get(body_id), sim.system.bodies.get(body_id))
    else {
        return;
    };

    let size = GPU_GRASS_WINDOW_SIZE_PX;
    let extent = Extent3d {
        width: size,
        height: size,
        depth_or_array_layers: 1,
    };
    let height_img = Image::new(
        extent,
        TextureDimension::D2,
        window
            .heights
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect(),
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    let aux_img = Image::new(
        extent,
        TextureDimension::D2,
        window.aux.clone(),
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    let height_handle = match &state.height_img {
        Some(h) => {
            let _ = images.insert(h, height_img);
            h.clone()
        }
        None => {
            let h = images.add(height_img);
            state.height_img = Some(h.clone());
            h
        }
    };
    let aux_handle = match &state.aux_img {
        Some(h) => {
            let _ = images.insert(h, aux_img);
            h.clone()
        }
        None => {
            let h = images.add(aux_img);
            state.aux_img = Some(h.clone());
            h
        }
    };

    // First window: create the material + template + entity.
    if state.material.is_none() {
        let fb = images.add(fallback_shadow_map());
        let maps: [Handle<Image>; 3] = [fb.clone(), fb.clone(), fb];
        state.material = Some(materials.add(GpuGrassMaterial {
            sun_shadow_map_0: maps[0].clone(),
            sun_shadow_map_1: maps[1].clone(),
            sun_shadow_map_2: maps[2].clone(),
            height_window: height_handle.clone(),
            aux_window: aux_handle.clone(),
            ..default()
        }));
    }
    if state.template.is_none() {
        state.template = Some(meshes.add(build_gpu_grass_template()));
    }

    state.anchor = Some(anchor);
    state.window_anchor = Some(anchor);
    state.anchor_height_m = window.anchor_height_m;
    state.anchor_moisture = window.anchor_moisture;
    state.built_revision = window.built_revision;
    state.built_region_count = region_count;

    let anchor_surface_body = anchor.dir * (body.radius_m + window.anchor_height_m as f64);
    if let Some(entity) = state.entity {
        if let Ok(mut ec) = commands.get_entity(entity) {
            ec.insert(GpuGrassVisual {
                body_id,
                anchor_surface_body,
            });
        }
    } else {
        let orientation = body_state.orientation.normalize();
        let center_world = body_state.position + orientation * anchor_surface_body;
        let (cell, local) = real_space_grid().translation_to_grid(center_world);
        let entity = commands
            .spawn((
                Mesh3d(state.template.clone().unwrap()),
                MeshMaterial3d(state.material.clone().unwrap()),
                Transform {
                    translation: local,
                    rotation: orientation.as_quat(),
                    scale: Vec3::ONE,
                },
                cell,
                Visibility::Inherited,
                RenderLayers::layer(SHIP_LAYER),
                NotShadowCaster,
                // Generous local bound: the field surrounds the camera and the
                // window's terrain relief rides in the vertex shader, so give
                // the culler the full envelope rather than a tight box.
                Aabb::from_min_max(
                    Vec3::new(-220.0, -800.0, -220.0),
                    Vec3::new(220.0, 800.0, 220.0),
                ),
                ChildOf(root.entity),
                GpuGrassVisual {
                    body_id,
                    anchor_surface_body,
                },
                Name::new("GPU Grass Field"),
            ))
            .id();
        state.entity = Some(entity);
    }
}

/// Re-pose the field in f64 each frame (the grass-tile/runway anchor math).
fn update_gpu_grass_transform(
    solar: Res<SolarSystemState>,
    root_grid: Query<&Grid, With<BigSpace>>,
    mut fields: Query<(&GpuGrassVisual, &mut CellCoord, &mut Transform)>,
) {
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let Ok(grid) = root_grid.single() else {
        return;
    };
    for (field, mut cell, mut transform) in &mut fields {
        let Some(body_state) = states.get(field.body_id) else {
            continue;
        };
        let orientation = body_state.orientation.normalize();
        let center_world = body_state.position + orientation * field.anchor_surface_body;
        let (next_cell, local) = grid.translation_to_grid(center_world);
        *cell = next_cell;
        transform.translation = local;
        transform.rotation = orientation.as_quat();
    }
}

/// Per-frame material params: sun / wind / sky / fade mirroring the CPU grass
/// update and the window/lattice registration upload. The frame-coherent
/// shadow payload is fanned out once in `sun_shadow`'s `Last` pass.
/// Read-only on the driver state — anchor re-registration happens in
/// [`drive_gpu_grass`], first in the chain, so pose and params stay coherent.
#[allow(clippy::too_many_arguments)]
fn update_gpu_grass_material(
    state: Res<GpuGrassState>,
    solar: Res<SolarSystemState>,
    sim: Res<SimulationState>,
    time: Res<Time>,
    exposure: Res<CameraExposure>,
    view_anchor: Res<ViewAnchor>,
    height_sources: Res<HeightSourceRegistry>,
    mut materials: ResMut<Assets<GpuGrassMaterial>>,
) {
    let (Some(body_id), Some(states), Some(material_handle)) =
        (state.body, solar.states.as_deref(), state.material.clone())
    else {
        return;
    };
    let (Some(body_state), Some(body)) = (states.get(body_id), sim.system.bodies.get(body_id))
    else {
        return;
    };
    let Some(window_anchor) = state.window_anchor else {
        return;
    };
    let radius_m = body.radius_m;

    let Some(anchor) = state.anchor else {
        return;
    };
    let Some(mut material) = materials.get_mut(&material_handle) else {
        return;
    };

    // ── Shared shading params (the CPU grass update's math) ─────────────────
    let star_pos = states.first().map(|s| s.position).unwrap_or(DVec3::ZERO);
    let offset = star_pos - body_state.position;
    let sun_dir = offset.normalize_or_zero().as_vec3();
    let au_over_d = (AU_M / offset.length().max(1.0)) as f32;
    let flux = LIGHT_AT_1AU * au_over_d * au_over_d * exposure.gain;
    material.params.sun_dir = Vec4::new(sun_dir.x, sun_dir.y, sun_dir.z, flux);

    let t = time.elapsed_secs();
    // Local vertical at the VIEW (the field exists around the view anchor).
    let up = view_anchor
        .resolved
        .filter(|a| a.body == body_id)
        .map(|a| (body_state.orientation * a.cam_dir).as_vec3())
        .unwrap_or_else(|| {
            (sim.simulation.ship_state().position - body_state.position)
                .normalize_or_zero()
                .as_vec3()
        });
    let seed = if up.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let east = seed.cross(up).normalize_or_zero();
    let north = up.cross(east);
    let veer = t * 0.03;
    let wind_dir = (east * veer.cos() + north * veer.sin()).normalize_or_zero();
    material.params.wind = Vec4::new(wind_dir.x, wind_dir.y, wind_dir.z, WIND_SWAY_M);

    let ramp = ((state.agl_m - FADE_LO_AGL_M) / (FADE_HI_AGL_M - FADE_LO_AGL_M)).clamp(0.0, 1.0);
    let altitude_collapse = (ramp * ramp * (3.0 - 2.0 * ramp)) as f32;
    // Sea level is the project datum (0 m); w = enable.
    material.params.time_fade = Vec4::new(t, altitude_collapse, 0.0, 1.0);
    material.params.sky_up = Vec4::new(up.x, up.y, up.z, 0.0);

    let (tau, strength) = body
        .terrestrial_atmosphere
        .as_ref()
        .and_then(|a| a.scattering.as_ref())
        .map(|s| (Vec3::from_array(s.vertical_optical_depth), s.strength))
        .unwrap_or((Vec3::ZERO, 0.0));
    material.params.sky_tau = Vec4::new(tau.x, tau.y, tau.z, strength);

    // Fade reference = the VIEW (`view.world_position` in the shader, offset 0):
    // blade density is a per-instance LOD keyed by distance from the eye. Offset
    // 0 is origin-invariant and this-frame-exact — the former craft-anchored
    // offset worked around the main-world camera transform lagging a frame,
    // which the shader's own view position doesn't.
    material.params.anchor = Vec4::ZERO;

    // ── Window / lattice registration ────────────────────────────────────────
    material.params.frame_east = Vec4::new(
        anchor.east.x as f32,
        anchor.east.y as f32,
        anchor.east.z as f32,
        state.anchor_height_m,
    );
    // The anchor's offset from the window centre, in the anchor's tangent frame
    // (frame_north.w / frame_up.w — see the WGSL `gg_window_uv`).
    let delta = (anchor.dir - window_anchor.dir) * radius_m;
    let off_u = delta.dot(anchor.east) as f32;
    let off_v = delta.dot(anchor.north) as f32;
    material.params.frame_north = Vec4::new(
        anchor.north.x as f32,
        anchor.north.y as f32,
        anchor.north.z as f32,
        off_u,
    );
    material.params.frame_up = Vec4::new(
        anchor.dir.x as f32,
        anchor.dir.y as f32,
        anchor.dir.z as f32,
        off_v,
    );
    let texel_m = (2.0 * GPU_GRASS_WINDOW_HALF_M) / GPU_GRASS_WINDOW_SIZE_PX as f64;
    material.params.window_meta = Vec4::new(
        texel_m as f32,
        GPU_GRASS_WINDOW_SIZE_PX as f32,
        GPU_GRASS_WINDOW_HALF_M as f32,
        // Climate cold lift at the anchor (m): shifts the blade treeline fade
        // and the veg palette with latitude, matching the terrain shader.
        thalos_terrain::climate_cold_lift_m(anchor.dir.y.abs()) as f32,
    );

    let anchor_point = anchor.dir * (radius_m + state.anchor_height_m as f64);
    let phase = anchor_point.map(|c| c.rem_euclid(LANDCOVER_PERIOD_M));
    material.params.phase = Vec4::new(
        phase.x as f32,
        phase.y as f32,
        phase.z as f32,
        state.anchor_moisture,
    );

    for i in 0..GPU_GRASS_BAND_COUNT {
        let (cx, cy) = anchor.band_cell[i];
        material.params.band_cell[i] = UVec4::new(cx as u32, cy as u32, anchor.face as u32, 0);
        let (cu, cv) = anchor.band_cell_m[i];
        let (fx, fy) = anchor.band_frac[i];
        material.params.band_geom[i] = Vec4::new(cu as f32, cv as f32, fx as f32, fy as f32);
    }
    // Grass style table (dry/lush/lawn) — authored Rust-side; an all-zero
    // table renders zero-size blades.
    material.params.style = gpu_grass_style_table();

    // Freshness guard: if the height source vanished (body teardown), park.
    if !height_sources.contains(body_id) {
        material.params.time_fade.w = 0.0;
    }
}
