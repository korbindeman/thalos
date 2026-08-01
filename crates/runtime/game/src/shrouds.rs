//! Auto-generated interstage shrouds: cones wrapping a [`thalos_shipyard::Shroudable`]
//! engine above a [`thalos_shipyard::ShroudProvider`] (decoupler).
//!
//! **One canonical path for both worlds.** A shroud is a property of the
//! decoupler's attachment, not of the editor, so [`sync_shrouds`] reconciles
//! *every* provider — the VAB's [`EditorPart`] stack and the flight craft
//! alike. The editor adds interaction on top (hover transparency, click-selects
//! the provider); flight shrouds are plain opaque hull geometry. Before this
//! module was hoisted out of `shipyard_editor::core`, the query filtered
//! `With<EditorPart>` and the shroud existed only in the VAB — the craft flew
//! with a bare engine hanging under the interstage.
//!
//! Shrouds are **not** part of the persisted blueprint: they are derived from
//! the attach graph on both sides, so a saved ship never carries one.
//!
//! Ownership through staging: the shroud is a child of the provider, so firing
//! the decoupler carries it down with the jettisoned stage (the KSP interstage
//! convention). Separation strips the decoupler's `Attachment`, which would
//! make it stop qualifying here, so `staging` stamps [`ShroudFired`] on the
//! shroud at the cut and this module then leaves it alone forever.

use bevy::camera::visibility::NoFrustumCulling;
use bevy::picking::Pickable;
use bevy::picking::events::{Click, Pointer};
use bevy::picking::hover::HoverMap;
use bevy::prelude::*;
use std::collections::{HashMap, HashSet};

use thalos_body_render::{
    ShipPartExtension, ShipPartMaterial, ShipPartParams, stainless_steel_base,
};
use thalos_shipyard::sizing::propagate_node_sizes;
use thalos_shipyard::{AttachNodes, Attachment, Engine, ShroudProvider, Shroudable};

use crate::shipyard_editor::core::state::{EditorPart, EditorState, EditorUiGate, PART_RESOLUTION};
use crate::shipyard_editor::core::visuals::engine_visual_profile;

/// Owns shroud reconciliation for the whole app — the VAB build and the flight
/// craft are two sets of parts under one system, not two implementations.
pub struct ShroudPlugin;

impl Plugin for ShroudPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                // Geometry reads `AttachNodes` diameters, which sizing writes.
                sync_shrouds.after(propagate_node_sizes),
                update_shroud_transparency.after(sync_shrouds),
            ),
        );
    }
}

/// Attached to a shroud entity — a mesh child of the provider
/// (e.g. a decoupler) that wraps the shrouded part above. Spawned and
/// reconciled by [`sync_shrouds`]; not part of the persisted blueprint
/// and not user-spawnable.
#[derive(Component, Debug, Clone, Copy)]
pub struct Shroud {
    pub provider: Entity,
    pub shrouded: Entity,
    /// Axial length of the shroud, metres — also the distance the shrouded part
    /// must travel to clear it, which `staging` reads to size the separation
    /// impulse.
    pub height: f32,
    // Cached spec, compared each frame so we only rebuild the mesh /
    // material when geometry actually changed.
    bottom_radius: f32,
    top_radius: f32,
}

// Moved to `thalos_game_state::scene` (Phase 5b) — the editor tags and
// reads it too; the reconcile pass stays here.
pub use thalos_game_state::scene::ShroudBody;

/// Stamped on a shroud when its provider fires, in the same transaction that
/// cuts the attach graph. A fired decoupler has no `Attachment` left, so it can
/// never re-qualify in [`sync_shrouds`]; without this marker the reconcile pass
/// would read that as "provider no longer qualifies" and despawn the interstage
/// off the jettisoned stage one frame after separation.
#[derive(Component, Debug, Clone, Copy)]
pub struct ShroudFired;

/// Expected geometry for a shroud covering a given attachment. `None`
/// when no shroud should exist for this pair (misconfigured attachment,
/// shrouded part missing [`Shroudable`], or provider not wider than the
/// shrouded's top — the cone would degenerate).
struct ShroudSpec {
    bottom_radius: f32,
    top_radius: f32,
    height: f32,
    shrouded: Entity,
}

fn compute_shroud_spec(
    attachment: &Attachment,
    provider_nodes: &AttachNodes,
    shroudables: &Query<(&Engine, Has<Shroudable>)>,
) -> Option<ShroudSpec> {
    // Only the canonical "provider sits below shroudable" orientation
    // gets a shroud: provider's `top` mates with shroudable's `bottom`.
    if attachment.my_node != "top" || attachment.parent_node != "bottom" {
        return None;
    }
    let (engine, is_shroudable) = shroudables.get(attachment.parent).ok()?;
    if !is_shroudable {
        return None;
    }
    let provider_top_d = provider_nodes.get("top")?.diameter;
    // Shroud top matches the shrouded part's *attach* diameter — the
    // interface the stage above would mate with. That sits outside the
    // engine's narrowing visual silhouette, so the shroud stays clear
    // of the engine body instead of hugging (and z-fighting) it.
    let bottom_r = provider_top_d * 0.5;
    let top_r = engine.diameter * 0.5;
    let (_, _, height) = engine_visual_profile(engine.diameter);
    // Only generate when the provider is at least as wide as the
    // shrouded part at its top — a narrower provider would invert the
    // cone. Equal diameter gives a clean cylindrical interstage.
    if bottom_r + 1.0e-4 < top_r {
        return None;
    }
    Some(ShroudSpec {
        bottom_radius: bottom_r,
        top_radius: top_r,
        height,
        shrouded: attachment.parent,
    })
}

fn spec_matches(s: &Shroud, spec: &ShroudSpec) -> bool {
    s.shrouded == spec.shrouded
        && (s.bottom_radius - spec.bottom_radius).abs() < 1.0e-4
        && (s.top_radius - spec.top_radius).abs() < 1.0e-4
        && (s.height - spec.height).abs() < 1.0e-4
}

/// Reconcile shroud entities against current attachment state: spawn
/// missing shrouds, update ones whose geometry changed, and despawn
/// orphans. Idempotent per frame; cheap when attachment is stable.
///
/// Runs over editor *and* flight providers. The only difference between the two
/// is presentation: an editor shroud is pickable and alpha-blended so hover can
/// reveal the engine inside; a flight shroud is opaque hull geometry with
/// frustum culling off, like every other flight part visual.
pub fn sync_shrouds(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    providers: Query<(Entity, &Attachment, &AttachNodes, Has<EditorPart>), With<ShroudProvider>>,
    shroudables: Query<(&Engine, Has<Shroudable>)>,
    existing: Query<(Entity, &Shroud), Without<ShroudFired>>,
) {
    // Map provider -> (shroud_entity, current Shroud component).
    let mut current_by_provider: HashMap<Entity, (Entity, Shroud)> = HashMap::new();
    for (entity, shroud) in existing.iter() {
        current_by_provider.insert(shroud.provider, (entity, *shroud));
    }

    let mut kept: HashSet<Entity> = HashSet::new();

    for (provider, attachment, provider_nodes, is_editor) in providers.iter() {
        let Some(spec) = compute_shroud_spec(attachment, provider_nodes, &shroudables) else {
            continue;
        };
        kept.insert(provider);

        // Reuse in-place if the cached spec still matches.
        if let Some((_, current)) = current_by_provider.get(&provider)
            && spec_matches(current, &spec)
        {
            continue;
        }
        if let Some((old, _)) = current_by_provider.get(&provider) {
            commands.entity(*old).despawn();
        }

        let shroud_mesh: Mesh = ConicalFrustum {
            radius_top: spec.top_radius,
            radius_bottom: spec.bottom_radius,
            height: spec.height,
        }
        .mesh()
        .resolution(PART_RESOLUTION)
        .into();
        let mesh_handle = meshes.add(shroud_mesh);

        // Slant length — the actual surface distance v = 0 → v = 1.
        // Matches the vertical height only when the two radii agree.
        let dr = spec.bottom_radius - spec.top_radius;
        let slant_length = (spec.height * spec.height + dr * dr).sqrt();
        // The editor's blend mode is set once here; we only vary base-color
        // alpha from the hover system so the pipeline stays hot. Flight shrouds
        // stay opaque — nothing hovers them, and a blended hull surface would
        // sort against the craft it belongs to.
        let base = if is_editor {
            StandardMaterial {
                alpha_mode: AlphaMode::Blend,
                ..stainless_steel_base()
            }
        } else {
            stainless_steel_base()
        };
        let material = ship_materials.add(ShipPartMaterial {
            base,
            extension: ShipPartExtension {
                params: ShipPartParams {
                    length: slant_length,
                    radius_top: spec.top_radius,
                    radius_bottom: spec.bottom_radius,
                    // Mix provider index with a fixed mask so shroud
                    // detail doesn't look identical to the decoupler's.
                    seed: provider.index_u32() ^ 0x5A5A_5A5A,
                    ..default()
                },
                ..Default::default()
            },
        });

        // Shroud mesh center sits at +height/2 in the provider's local
        // frame, since the provider's "top" node (y = 0) meets the
        // shrouded's base and the shroud extends upward from there.
        let mut shroud_entity = commands.spawn((
            Mesh3d(mesh_handle),
            MeshMaterial3d(material),
            Transform::from_xyz(0.0, spec.height * 0.5, 0.0),
            Visibility::default(),
            Shroud {
                provider,
                shrouded: spec.shrouded,
                bottom_radius: spec.bottom_radius,
                top_radius: spec.top_radius,
                height: spec.height,
            },
        ));
        if is_editor {
            shroud_entity.insert((ShroudBody, Pickable::default()));
            shroud_entity.observe(on_shroud_click);
        } else {
            // Flight part visuals opt out of frustum culling: the craft sits
            // deep in a BigSpace grid and the culling AABBs are unreliable
            // there. Match `ship_view::rebuild_ship_visuals`.
            shroud_entity.insert(NoFrustumCulling);
        }
        let shroud_entity = shroud_entity.id();
        commands.entity(provider).add_child(shroud_entity);
    }

    // Despawn shrouds whose provider no longer qualifies (detachment,
    // geometry change below threshold, shrouded part removed, etc.).
    // Fired shrouds are excluded from `existing` and never reach here.
    for (provider, (entity, _)) in &current_by_provider {
        if !kept.contains(provider) {
            commands.entity(*entity).despawn();
        }
    }
}

/// Drive the shroud's base-color alpha from hover: opaque by default
/// (engine hidden inside), partial transparency while hovered so the
/// shrouded silhouette reads through. Editor-only — [`ShroudBody`] is not
/// present on flight shrouds.
pub fn update_shroud_transparency(
    hover_map: Res<HoverMap>,
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    shrouds: Query<(Entity, &MeshMaterial3d<ShipPartMaterial>), With<ShroudBody>>,
) {
    let hovered: HashSet<Entity> = hover_map
        .0
        .values()
        .flat_map(|hovers| hovers.keys().copied())
        .collect();

    for (entity, mesh_mat) in shrouds.iter() {
        let target_alpha: f32 = if hovered.contains(&entity) { 0.18 } else { 1.0 };
        let Some(mut mat) = ship_materials.get_mut(&mesh_mat.0) else {
            continue;
        };
        let srgba = mat.base.base_color.to_srgba();
        if (srgba.alpha - target_alpha).abs() > 1.0e-3 {
            mat.base.base_color = Color::srgba(srgba.red, srgba.green, srgba.blue, target_alpha);
        }
    }
}

/// Click on a shroud selects the provider that owns it — the shroud is
/// a visual extension of the decoupler, not an independent part.
fn on_shroud_click(
    click: On<Pointer<Click>>,
    shrouds: Query<&Shroud>,
    ui_gate: Res<EditorUiGate>,
    mut state: ResMut<EditorState>,
) {
    if ui_gate.pointer_busy {
        return;
    }
    if let Ok(shroud) = shrouds.get(click.entity) {
        state.selected = Some(shroud.provider);
    }
}
