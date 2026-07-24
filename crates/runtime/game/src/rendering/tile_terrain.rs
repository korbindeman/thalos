//! NTR-X1 driver — the probe-extracted standard-path tile renderer, in-game.
//!
//! Behind `THALOS_TILE_RENDERER=1` (default off; udlod remains the production
//! path until parity). When enabled:
//!
//! - udlod ground terrain stands down (`terrain_residency::try_spawn` gate),
//! - the first `ViewAnchor`-resolved terrain body gets a
//!   [`TileTerrainRoot`] on its `RealSpaceBody` grid entity — tiles stream as
//!   ordinary `Mesh` + `StandardMaterial` children of the rotating body grid,
//!   fed by the body's canonical `Arc<dyn SurfaceQuery>`,
//! - the selection eye is republished every frame from `ViewAnchor`
//!   (body-fixed camera position — no per-mode camera plumbing).
//!
//! Slice-1 limitations (tracked in NTR-X1's backlog row): plain
//! `StandardMaterial` (no `thalos::shadow`/Hapke), impostor visibility is
//! left to the existing udlod-coupled swap (without resident udlod terrain
//! the impostor stays visible; tiles win depth where they cover — a sliver
//! of billboard may peek past the tile limb), single body (the first anchor
//! body wins), no GPU height mirror (CPU `HeightSource` fallback serves
//! colliders/HUD).

use std::sync::{Arc, OnceLock};

use bevy::prelude::*;
use thalos_body_render::tiles::material::{TileShadingParams, TileTerrainMaterial, tile_material};
use thalos_body_render::tiles::{
    SurfaceQueryProvider, TileEye, TileEyeTarget, TileTerrainRoot,
};
use thalos_world::BodyId;

use super::terrain_residency::TerrainRebuildRequest;
use super::types::RealSpaceBody;
use super::view_anchor::ViewAnchor;
use crate::terrain_registry::BodySurfaceRegistry;
use std::sync::Mutex;

/// Bodies currently owned by the tile renderer — read by
/// `terrain_residency::try_spawn` (which may run before this plugin's
/// systems), hence a process-global rather than a Bevy resource.
static TILE_RENDERED: Mutex<Vec<thalos_world::BodyId>> = Mutex::new(Vec::new());

/// Is `body_id` rendered by the tile path (udlod stands down for it)?
pub fn tile_rendered(body_id: BodyId) -> bool {
    tile_renderer_enabled() && TILE_RENDERED.lock().is_ok_and(|v| v.contains(&body_id))
}

/// One env check, cached: `THALOS_TILE_RENDERER=1|true|on|yes`.
pub fn tile_renderer_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("THALOS_TILE_RENDERER")
            .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on" | "yes"))
            .unwrap_or(false)
    })
}

/// Driver-side tag pairing the tile root with its body id.
#[derive(Component)]
pub struct TileTerrainBody {
    pub body_id: BodyId,
}

pub struct TileTerrainDriverPlugin;

impl Plugin for TileTerrainDriverPlugin {
    fn build(&self, app: &mut App) {
        if !tile_renderer_enabled() {
            return;
        }
        info!("THALOS_TILE_RENDERER=1 — standard-path tile terrain active; udlod ground terrain gated off");
        app.add_systems(Update, (ensure_tile_root, update_tile_eye).chain());
    }
}

/// Lazily install the tile terrain on the first anchor-resolved body that has
/// a canonical surface. Slice 1: one body per session.
fn ensure_tile_root(
    anchor: Res<ViewAnchor>,
    surfaces: Res<BodySurfaceRegistry>,
    sim: Res<crate::solar_system_state::SimulationState>,
    bodies: Query<(Entity, &RealSpaceBody)>,
    existing: Query<(), With<TileTerrainRoot>>,
    mut materials: ResMut<Assets<TileTerrainMaterial>>,
    mut rebuild: ResMut<TerrainRebuildRequest>,
    mut commands: Commands,
) {
    if !existing.is_empty() {
        return;
    }
    let Some(resolved) = anchor.resolved else {
        return;
    };
    let Some(surface) = surfaces.surface(resolved.body) else {
        return;
    };
    let Some((entity, _)) = bodies.iter().find(|(_, rsb)| rsb.body_id == resolved.body) else {
        return;
    };
    // Vertex colors carry the surface's linear albedo; the base material
    // stays neutral. Airless bodies shade through the Hapke regolith branch
    // (tile_terrain.wgsl) so ground reconverges with the impostor's Hapke
    // look; atmosphere-bearing bodies keep stock PBR. Both branches receive
    // the shared `thalos::shadow` cascade via `apply_craft_shadow`.
    let airless = sim
        .system
        .bodies
        .get(resolved.body)
        .is_none_or(|body| body.terrestrial_atmosphere.is_none());
    let params = if airless { TileShadingParams::hapke() } else { TileShadingParams::pbr() };
    let material = materials.add(tile_material(
        StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.97,
            metallic: 0.0,
            ..default()
        },
        params,
    ));
    let radius_m = resolved.radius_m;
    let root = TileTerrainRoot::new(
        radius_m,
        Arc::new(SurfaceQueryProvider { surface }),
        material,
    );
    info!(
        "tile terrain: installing on body {:?} (radius {:.0} m, max level {})",
        resolved.body, radius_m, root.max_level
    );
    commands
        .entity(entity)
        .insert((root, TileTerrainBody { body_id: resolved.body }));
    if let Ok(mut list) = TILE_RENDERED.lock() {
        list.push(resolved.body);
    }
    // Boot race: residency may have spawned udlod for this body before the
    // anchor resolved. A rebuild request despawns it; the respawn declines
    // via the `tile_rendered` gate, leaving the tile path sole owner.
    rebuild.request(resolved.body);
}

/// Republish the selection eye from `ViewAnchor` each frame.
fn update_tile_eye(
    anchor: Res<ViewAnchor>,
    roots: Query<(Entity, &TileTerrainBody), With<TileTerrainRoot>>,
    mut eye: ResMut<TileEye>,
) {
    eye.target = None;
    let Some(resolved) = anchor.resolved else {
        return;
    };
    for (entity, body) in &roots {
        if body.body_id == resolved.body {
            eye.target = Some(TileEyeTarget { root: entity, cam_body: resolved.cam_body });
            return;
        }
    }
}
