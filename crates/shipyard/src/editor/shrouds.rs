//! Auto-generated interstage shrouds: cones wrapping a [`crate::Shroudable`]
//! engine above a [`crate::ShroudProvider`] (decoupler). Editor-only visuals —
//! they are not part of the persisted blueprint.

use bevy::picking::Pickable;
use bevy::picking::events::{Click, Pointer};
use bevy::picking::hover::HoverMap;
use bevy::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::material::{ShipPartExtension, ShipPartMaterial, ShipPartParams};
use crate::{
    AttachNodes, Attachment, Engine, ShroudProvider, Shroudable, stainless_steel_base,
};

use super::state::{EditorPart, EditorState, EditorUiGate, PART_RESOLUTION};
use super::visuals::engine_visual_profile;

/// Attached to a shroud entity — a mesh child of the provider
/// (e.g. a decoupler) that wraps the shrouded part above. Spawned and
/// reconciled by [`sync_shrouds`]; not part of the persisted blueprint
/// and not user-spawnable.
#[derive(Component, Debug, Clone, Copy)]
pub struct Shroud {
    pub provider: Entity,
    pub shrouded: Entity,
    // Cached spec, compared each frame so we only rebuild the mesh /
    // material when geometry actually changed.
    bottom_radius: f32,
    top_radius: f32,
    height: f32,
}

/// Marker on the shroud entity's body. Kept distinct from
/// [`super::state::PartBody`] so part-level highlight systems (tint, material
/// swap) don't fire on hovered shrouds — the shroud manages its own hover
/// feedback (transparency) in [`update_shroud_transparency`].
#[derive(Component, Debug, Clone, Copy)]
pub struct ShroudBody;

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
pub(super) fn sync_shrouds(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    providers: Query<(Entity, &Attachment, &AttachNodes), (With<ShroudProvider>, With<EditorPart>)>,
    shroudables: Query<(&Engine, Has<Shroudable>)>,
    existing: Query<(Entity, &Shroud)>,
) {
    // Map provider -> (shroud_entity, current Shroud component).
    let mut current_by_provider: HashMap<Entity, (Entity, Shroud)> = HashMap::new();
    for (entity, shroud) in existing.iter() {
        current_by_provider.insert(shroud.provider, (entity, *shroud));
    }

    let mut kept: HashSet<Entity> = HashSet::new();

    for (provider, attachment, provider_nodes) in providers.iter() {
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
        // Blend mode is set once here; we only vary base-color alpha
        // from the hover system so the pipeline stays hot.
        let material = ship_materials.add(ShipPartMaterial {
            base: StandardMaterial {
                alpha_mode: AlphaMode::Blend,
                ..stainless_steel_base()
            },
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
        let shroud_entity = commands
            .spawn((
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
                ShroudBody,
                Pickable::default(),
            ))
            .observe(on_shroud_click)
            .id();
        commands.entity(provider).add_child(shroud_entity);
    }

    // Despawn shrouds whose provider no longer qualifies (detachment,
    // geometry change below threshold, shrouded part removed, etc.).
    for (provider, (entity, _)) in &current_by_provider {
        if !kept.contains(provider) {
            commands.entity(*entity).despawn();
        }
    }
}

/// Drive the shroud's base-color alpha from hover: opaque by default
/// (engine hidden inside), partial transparency while hovered so the
/// shrouded silhouette reads through.
pub(super) fn update_shroud_transparency(
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
