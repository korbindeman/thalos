//! Editor visualization for the tectonic structural prior.
//!
//! When the preview body has a `TectonicSystem`, this module draws plate
//! boundaries (color-coded by kind) and per-plate motion arrows as Bevy
//! gizmos overlaid on the impostor sphere. The toggles live in
//! [`TectonicOverlayState`] and are exposed in the editor's Tectonics panel.
//!
//! Gizmo positions are computed in world space at radius `RENDER_RADIUS`
//! (slightly inflated to avoid z-fighting with the impostor billboard).
//! Cell directions are rotated through the planet's orientation quaternion
//! so the overlay tracks the visually rendered orientation rather than the
//! body-local frame.

use bevy::prelude::*;
use thalos_terrain_gen::tectonics::surface_velocity;
use thalos_terrain_gen::{BoundaryKind, PlateKind, TectonicActivity, TectonicSystem};

/// Edit-time component carrying the body's tectonic graph for overlay
/// rendering. Inserted by `finalize_terrain_bake` when the bake task
/// returns a `PlanetSurface` with `tectonics: Some(_)`.
#[derive(Component, Clone)]
pub struct PreviewTectonics {
    pub system: TectonicSystem,
}

/// Toggleable overlay layers. Defaults: boundaries off, arrows on, plate
/// colors off (debug material swap not yet wired).
#[derive(Resource)]
pub struct TectonicOverlayState {
    pub show_boundaries: bool,
    pub show_motion_arrows: bool,
    pub show_plate_centroids: bool,
}

impl Default for TectonicOverlayState {
    fn default() -> Self {
        Self {
            show_boundaries: false,
            show_motion_arrows: true,
            show_plate_centroids: false,
        }
    }
}

/// Plugin: registers the overlay resource and the gizmo-draw system.
pub struct TectonicOverlayPlugin;

impl Plugin for TectonicOverlayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TectonicOverlayState>()
            .add_systems(Update, draw_tectonic_overlay);
    }
}

/// Inflate gizmo radii slightly past the impostor surface so the lines
/// don't z-fight with the billboard. 2% past gives a comfortable margin
/// without visibly floating off the surface at typical zoom levels.
const GIZMO_LIFT: f32 = 1.02;

/// World-space length of a plate motion arrow at maximum live magnitude.
/// Tuned visually so the longest arrows reach roughly `0.4 * RENDER_RADIUS`,
/// which is readable but doesn't crowd neighbors.
const ARROW_MAX_LENGTH: f32 = 0.6;

/// Draw boundary line segments and per-plate motion arrows. Runs every
/// frame; skip when the toggles are off or the body has no tectonics.
pub fn draw_tectonic_overlay(
    mut gizmos: Gizmos,
    state: Res<TectonicOverlayState>,
    render_radius: Res<TectonicRenderRadius>,
    orientation: Res<TectonicOverlayOrientation>,
    query: Query<&PreviewTectonics>,
) {
    let Ok(preview) = query.single() else {
        return;
    };
    let sys = &preview.system;
    let r = render_radius.0 * GIZMO_LIFT;
    let q = orientation.0;

    if state.show_boundaries {
        for boundary in &sys.boundaries {
            let a = sys.mesh.cells[boundary.cell_a as usize];
            let b = sys.mesh.cells[boundary.cell_b as usize];
            let p_a = (q * a) * r;
            let p_b = (q * b) * r;
            let color = boundary_color(boundary.kind);
            gizmos.line(p_a, p_b, color);
        }
    }

    if state.show_motion_arrows && sys.config.activity.live_velocity() {
        // Magnitude normalization: scale the longest plate's surface
        // velocity at its centroid to ARROW_MAX_LENGTH and apply that
        // scale uniformly to all plates. This keeps relative motion
        // legible without committing to a real physical ratio.
        let activity = sys.config.activity;
        let max_speed = sys
            .plates
            .iter()
            .map(|p| surface_velocity(p, p.centroid_dir, sys.body_radius_m, activity).length())
            .fold(0.0_f32, f32::max);

        if max_speed > 0.0 {
            for plate in &sys.plates {
                let dir = plate.centroid_dir;
                let v = surface_velocity(plate, dir, sys.body_radius_m, activity);
                let scale = ARROW_MAX_LENGTH / max_speed;
                let v_world = q * v * scale;
                let origin = (q * dir) * r;
                let tip = origin + v_world;
                gizmos.arrow(origin, tip, plate_color(plate.kind, plate.id.0));
            }
        }
    }

    if state.show_plate_centroids {
        for plate in &sys.plates {
            let p = (q * plate.centroid_dir) * r;
            gizmos.sphere(
                Isometry3d::from_translation(p),
                0.025,
                plate_color(plate.kind, plate.id.0),
            );
        }
    }
}

/// Resource carrying the planet's display radius. Owned by the editor
/// (set once at startup); broken out as a resource so the overlay
/// doesn't have to import the editor's private constants.
#[derive(Resource)]
pub struct TectonicRenderRadius(pub f32);

/// Resource carrying the planet's display orientation quaternion. The
/// editor updates this every frame from `body_orientation(&planet)`.
#[derive(Resource, Default)]
pub struct TectonicOverlayOrientation(pub Quat);

fn boundary_color(kind: BoundaryKind) -> Color {
    match kind {
        // Vivid, distinguishable on dark and light planet colors. Keep
        // the same hues we used in the bake_dump equirect so the editor
        // and the equirect read consistently.
        BoundaryKind::Convergent => Color::srgb(1.00, 0.18, 0.18),
        BoundaryKind::Divergent => Color::srgb(0.20, 0.55, 1.00),
        BoundaryKind::Transform => Color::srgb(1.00, 0.85, 0.20),
    }
}

fn plate_color(kind: PlateKind, plate_id: u32) -> Color {
    let h = thalos_terrain_gen::seeding::splitmix64(plate_id as u64 ^ 0xB1ADE0FF);
    let hue_unit = ((h & 0xFFFF) as f32) / 65535.0;
    let (hue_deg, sat, val) = match kind {
        PlateKind::Continental => {
            let hue = if hue_unit < 0.5 {
                hue_unit * 120.0
            } else {
                300.0 + (hue_unit - 0.5) * 120.0
            };
            (hue, 0.65, 0.85)
        }
        PlateKind::Oceanic => (180.0 + hue_unit * 60.0, 0.70, 0.55),
    };
    Color::hsv(hue_deg, sat, val)
}

/// Activity-mode label for the editor sidebar. Doesn't strictly belong
/// here, but it's the only place that needs to map an enum to a string.
pub fn activity_label(activity: TectonicActivity) -> &'static str {
    match activity {
        TectonicActivity::Active => "Active",
        TectonicActivity::StagnantLid => "Stagnant lid",
        TectonicActivity::Frozen { .. } => "Frozen",
    }
}
