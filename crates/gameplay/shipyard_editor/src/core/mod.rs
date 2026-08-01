//! UI-framework-agnostic ship-editor core.
//!
//! Everything an interactive ship editor needs *except its 2D UI and its
//! camera*: the [`EditorState`] command/state hub, part placement (attach
//! nodes + surface mounts + KSP linked symmetry), live mesh rebuilds, the
//! build-frame transform solve, selection/hover highlighting, the
//! tank-resize handle, placement-preview ghost, and blueprint save/load
//! against `ships/*.ron`.
//!
//! Interstage shrouds are **not** here: they are derived from the attach graph
//! for the flight craft too, so they live in the runtime's `shrouds` module as one shared
//! reconcile pass over both worlds.
//!
//! This is the editor logic proper; its 2D UI + camera front-end is the
//! sibling [`crate`] module (native Bevy UI). It reads the
//! [`thalos_shipyard`] construction model and is intentionally kept separate
//! from it — the crate owns *what a craft is*, this owns *editing one*.
//!
//! A front-end's contract:
//! - add [`ShipEditorCorePlugin`] (plus `MeshPickingPlugin`, a
//!   `PartCatalog` resource, and `thalos_input`'s `ShipyardInputPlugin` —
//!   the core reads `ShipyardInputIntent`);
//! - mark its editing camera with [`EditorViewCamera`];
//! - keep [`EditorUiGate`] in sync with its 2D UI hover/capture state;
//! - read and write [`EditorState`] (and the toggles: [`SymmetryMode`],
//!   [`PlacementSnap`], [`BuildOrientation`]) to drive everything else.
//!
//! **World partition:** every entity the editor owns carries [`EditorPart`],
//! and every iterating core query filters on it. A host application that
//! assembles other ships from the same part components (the game's flight
//! ship) must filter its own part aggregations `Without<EditorPart>`.

// Editor systems are query-heavy Bevy systems; the arg-count and
// type-complexity ceilings fight the natural shape.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

pub mod commands;
pub mod debug_log;
pub mod files;
pub mod format;
pub mod placement;
pub mod state;
pub mod visuals;

use bevy::picking::mesh_picking::MeshPickingPlugin;
use bevy::prelude::*;

use thalos_shipyard::ShipyardPlugin;
use thalos_shipyard::sizing::propagate_node_sizes;

pub use commands::{CollectQuery, collect_blueprint};
pub use files::SavedShip;
pub use format::{
    kind_order, palette_category_label, palette_category_order, palette_part_summary,
};
pub use placement::symmetry_edit_target;
pub use state::{
    BuildOrientation, CLICK_THRESHOLD_PX, DeselectTracker, EditorPart, EditorState, EditorUiGate,
    EditorViewCamera, NextSymmetryId, PendingPart, PlacementPreview, PlacementSnap, PreviewGhost,
    SymmetryMode, TankResizeArrow, TankResizeDrag,
};

/// The current [`thalos_shipyard::PartParams`] of a selected part, read back from its
/// kind components. Front-end inspectors use this to compute live resource
/// capacities (`thalos_shipyard::resource_capacity_for`) for the selection.
pub fn inspector_params(
    dec: Option<&thalos_shipyard::Decoupler>,
    adapter: Option<&thalos_shipyard::Adapter>,
    tank: Option<&thalos_shipyard::FuelTank>,
    fuselage: Option<&thalos_shipyard::Fuselage>,
    wing: Option<&thalos_shipyard::Wing>,
    gear: Option<&thalos_shipyard::Gear>,
) -> thalos_shipyard::PartParams {
    if let Some(d) = dec {
        thalos_shipyard::PartParams::Decoupler {
            diameter: d.diameter,
        }
    } else if let Some(a) = adapter {
        thalos_shipyard::PartParams::Adapter {
            diameter: a.diameter,
            target_diameter: a.target_diameter,
        }
    } else if let Some(t) = tank {
        thalos_shipyard::PartParams::Tank {
            diameter: t.diameter,
            length: t.length,
        }
    } else if let Some(f) = fuselage {
        thalos_shipyard::PartParams::Fuselage {
            length: f.length,
            max_width: f.max_width,
            max_height: f.max_height,
            roundness: f.roundness,
            nose_fraction: f.nose_fraction,
            nose_bluntness: f.nose_bluntness,
            tail_fraction: f.tail_fraction,
            nose_droop: f.nose_droop,
            tail_upsweep: f.tail_upsweep,
            tail_tip_diameter: f.tail_tip_diameter,
            tail_bluntness: f.tail_bluntness,
        }
    } else if let Some(w) = wing {
        thalos_shipyard::PartParams::Wing {
            span: w.span,
            root_chord: w.root_chord,
            tip_chord: w.tip_chord,
            sweep: w.sweep,
            dihedral: w.dihedral,
            thickness: w.thickness,
            incidence: w.incidence,
            control_surfaces: w.control_surfaces.clone(),
        }
    } else if let Some(g) = gear {
        thalos_shipyard::PartParams::Gear {
            strut_length: g.strut_length,
            wheel_radius: g.wheel_radius,
        }
    } else {
        thalos_shipyard::PartParams::None
    }
}

/// Editor-core systems + resources. See the module docs for the front-end
/// contract. Defensively adds [`ShipyardPlugin`] (sizing + recompute) and
/// `MeshPickingPlugin` when the host app hasn't already.
pub struct ShipEditorCorePlugin;

impl Plugin for ShipEditorCorePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<ShipyardPlugin>() {
            app.add_plugins(ShipyardPlugin);
        }
        if !app.is_plugin_added::<MeshPickingPlugin>() {
            app.add_plugins(MeshPickingPlugin);
        }

        app.insert_resource(debug_log::SelectionLog::from_env())
            .init_resource::<EditorState>()
            .init_resource::<TankResizeDrag>()
            .init_resource::<DeselectTracker>()
            .init_resource::<BuildOrientation>()
            .init_resource::<SymmetryMode>()
            .init_resource::<NextSymmetryId>()
            .init_resource::<PlacementSnap>()
            .init_resource::<PlacementPreview>()
            .init_resource::<EditorUiGate>()
            .add_systems(
                Startup,
                (visuals::init_editor_assets, commands::init_editor_state),
            )
            .add_systems(
                Update,
                (
                    commands::process_commands,
                    visuals::rebuild_visuals,
                    commands::sync_symmetry_groups
                        .before(visuals::rebuild_wing_visuals)
                        .before(visuals::rebuild_nacelle_visuals),
                    visuals::rebuild_wing_visuals,
                    visuals::rebuild_nacelle_visuals,
                    visuals::rebuild_gear_visuals,
                    visuals::update_part_transforms.after(propagate_node_sizes),
                    visuals::update_placement_preview.after(visuals::update_part_transforms),
                    visuals::update_node_pin_style,
                    visuals::sync_self_nodes,
                ),
            )
            .add_systems(
                Update,
                (
                    visuals::spawn_tank_resize_arrow,
                    visuals::update_tank_resize_arrow.after(visuals::update_part_transforms),
                    visuals::update_tank_resize_drag,
                    visuals::update_selection_highlight
                        .after(visuals::rebuild_visuals)
                        .after(visuals::rebuild_wing_visuals)
                        .after(visuals::rebuild_nacelle_visuals)
                        .after(visuals::rebuild_gear_visuals),
                    visuals::update_part_shader_params.after(visuals::rebuild_visuals),
                    visuals::update_part_shader_highlight.after(visuals::rebuild_visuals),
                    placement::deselect_on_empty_click,
                    debug_log::log_selection_changes.after(placement::deselect_on_empty_click),
                    visuals::propagate_coupled_material.after(visuals::rebuild_visuals),
                ),
            );
    }
}
