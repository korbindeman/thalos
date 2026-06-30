//! Editor visual systems: part mesh (re)builds, attach-node pins, the build
//! transform solve, selection/hover highlighting, the tank-resize handle,
//! and the live placement-preview ghost.
//!
//! Every iterating query is scoped `With<EditorPart>` so these systems only
//! ever touch the editor's build world — never another ship assembled from
//! the same part components elsewhere in the `World` (the game's flight
//! ship).

use bevy::picking::Pickable;
use bevy::picking::events::{DragEnd, DragStart, Pointer};
use bevy::picking::hover::HoverMap;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use std::collections::{HashMap, HashSet, VecDeque};

use thalos_input::shipyard::ShipyardInputIntent;

use crate::material::{ShipPartExtension, ShipPartMaterial};
use crate::{
    Adapter, AirIntake, AttachNodes, Attachment, CatalogEntry, CommandPod, Decoupler, Engine,
    EngineGeometry, FuelTank, Fuselage, Gear, JetNacelleMount, MaterialKind, Part, PartCatalog,
    PartMaterial, PodGeometry, Ship, SurfaceMount, SurfaceMountKind, Wing,
    build_cockpit_mesh, build_control_surface_mesh, build_fuselage_mesh, build_gear_bay_mesh,
    build_gear_mesh, build_jet_nacelle_body_mesh, build_jet_nacelle_pylon_mesh, build_wing_mesh,
    host_mount_geometry, jet_nacelle_length, landing_gear_base, pod_visual_profile,
    stainless_steel_base,
};

use super::placement::{body_skin_mount, on_body_click, on_pin_click};
use super::state::*;

pub(super) fn init_editor_assets(
    mut commands: Commands,
    mut mats: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    commands.insert_resource(EditorAssets {
        // Double-sided so a wing's thin slab (and its reflected panel) reads
        // from both faces; closed bodies-of-revolution are unaffected.
        part_material: mats.add(StandardMaterial {
            double_sided: true,
            cull_mode: None,
            ..stainless_steel_base()
        }),
        gear_material: mats.add(StandardMaterial {
            double_sided: true,
            cull_mode: None,
            ..landing_gear_base()
        }),
        hover_material: mats.add(StandardMaterial {
            base_color: Color::srgb(0.82, 0.85, 0.88),
            perceptual_roughness: 0.4,
            metallic: 0.6,
            emissive: LinearRgba::rgb(0.08, 0.08, 0.08),
            double_sided: true,
            cull_mode: None,
            ..default()
        }),
        selected_material: mats.add(StandardMaterial {
            base_color: Color::srgb(0.85, 0.9, 1.0),
            perceptual_roughness: 0.4,
            metallic: 0.6,
            emissive: LinearRgba::rgb(0.15, 0.35, 0.7),
            double_sided: true,
            cull_mode: None,
            ..default()
        }),
        pending_node_material: mats.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.9, 1.0),
            emissive: LinearRgba::rgb(0.1, 0.6, 0.9),
            ..default()
        }),
        node_mesh: meshes.add(Sphere::new(0.5).mesh()),
        resize_arrow_mesh: meshes.add(Cone::new(0.3, 0.8).mesh()),
        resize_arrow_material: mats.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.75, 0.2),
            emissive: LinearRgba::rgb(0.9, 0.5, 0.05),
            perceptual_roughness: 0.5,
            unlit: false,
            ..default()
        }),
        preview_material: mats.add(StandardMaterial {
            base_color: Color::srgba(0.4, 1.0, 0.5, 0.45),
            emissive: LinearRgba::rgb(0.05, 0.3, 0.1),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            double_sided: true,
            cull_mode: None,
            ..default()
        }),
        gear_bay_material: mats.add(StandardMaterial {
            base_color: Color::srgba(0.2, 0.85, 1.0, 0.22),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            double_sided: true,
            cull_mode: None,
            // Large bias forces the ghost in front of the opaque hull so the
            // reserved bay volume is visible through the fuselage (x-ray).
            depth_bias: 1.0e9,
            ..default()
        }),
    });
}

pub struct VisualSpec {
    pub mesh: Mesh,
    pub height: f32,
}

/// `top` node diameter of a host part, or a sensible default. Single source
/// for the surface-mount radius lookups so they stay consistent.
pub fn host_top_diameter(nodes: &Query<&AttachNodes>, host: Entity) -> f32 {
    nodes
        .get(host)
        .ok()
        .and_then(|n| n.get("top").map(|nd| nd.diameter))
        .unwrap_or(2.0)
}

pub fn visual_spec(
    nodes: &AttachNodes,
    pod: Option<&CommandPod>,
    dec: Option<&Decoupler>,
    adapter: Option<&Adapter>,
    tank: Option<&FuelTank>,
    fuselage: Option<&Fuselage>,
    engine: Option<&Engine>,
    intake: Option<&AirIntake>,
) -> Option<VisualSpec> {
    if let Some(p) = pod {
        // Inline cockpit: no body mesh (the fuselage nose is the nose).
        if matches!(p.geometry, PodGeometry::Inline) {
            return None;
        }
        let (radius_top, radius_bottom, h) = pod_visual_profile(p.diameter, p.geometry);
        let mesh = match p.geometry {
            // Rounded ogive nose (airliner radome) vs the plain capsule cone.
            PodGeometry::AircraftCockpit => build_cockpit_mesh(p.diameter, h),
            PodGeometry::Inline => unreachable!("handled above"),
            PodGeometry::Capsule => ConicalFrustum {
                radius_top,
                radius_bottom,
                height: h,
            }
            .mesh()
            .resolution(PART_RESOLUTION)
            .into(),
        };
        Some(VisualSpec { mesh, height: h })
    } else if dec.is_some() {
        let d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let h = 0.2;
        Some(VisualSpec {
            mesh: Cylinder::new(d * 0.5, h)
                .mesh()
                .resolution(PART_RESOLUTION)
                .into(),
            height: h,
        })
    } else if let Some(a) = adapter {
        let top_d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let bot_d = a.target_diameter;
        let h = ((top_d + bot_d) * 0.5).max(0.4);
        Some(VisualSpec {
            mesh: ConicalFrustum {
                radius_top: top_d * 0.5,
                radius_bottom: bot_d * 0.5,
                height: h,
            }
            .mesh()
            .resolution(PART_RESOLUTION)
            .into(),
            height: h,
        })
    } else if let Some(t) = tank {
        let d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let h = t.length;
        Some(VisualSpec {
            mesh: Cylinder::new(d * 0.5, h)
                .mesh()
                .resolution(PART_RESOLUTION)
                .into(),
            height: h,
        })
    } else if let Some(f) = fuselage {
        // Barrel diameter inherits from the `top` node (parent-driven), like
        // a tank; the loft generator scales the rest to it.
        let d = nodes.get("top").map(|n| n.diameter).unwrap_or(f.max_width);
        Some(VisualSpec {
            mesh: build_fuselage_mesh(f, d),
            height: f.length,
        })
    } else if let Some(e) = engine {
        match e.geometry {
            EngineGeometry::RocketBell => {
                let (r_top, r_bot, h) = engine_visual_profile(e.diameter);
                Some(VisualSpec {
                    mesh: ConicalFrustum {
                        radius_top: r_top,
                        radius_bottom: r_bot,
                        height: h,
                    }
                    .mesh()
                    .resolution(PART_RESOLUTION)
                    .into(),
                    height: h,
                })
            }
            EngineGeometry::JetNacelle => Some(VisualSpec {
                mesh: build_jet_nacelle_body_mesh(e),
                height: jet_nacelle_length(e),
            }),
        }
    } else {
        intake.map(|i| VisualSpec {
            mesh: Cylinder::new(i.diameter * 0.5, i.length)
                .mesh()
                .resolution(PART_RESOLUTION)
                .into(),
            height: i.length,
        })
    }
}

/// Engine body silhouette: `(radius_top, radius_bottom, height)` for a
/// given engine diameter. Single source for both the engine mesh and the
/// matching shroud geometry — drift between the two would leave the
/// shroud edge either floating off the engine or clipping into it.
pub fn engine_visual_profile(diameter: f32) -> (f32, f32, f32) {
    (diameter * 0.35, diameter * 0.5, diameter * 0.9)
}

// `ship_part_params` (part dims → material uniform) is shared with the flight
// view; it now lives in `crate::appearance` (re-exported at the crate root) so
// the two no longer drift. Re-exported here so `editor::ship_part_params` (the
// existing path consumers use) keeps resolving.
pub use crate::ship_part_params;

type VisualQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static AttachNodes,
        Option<&'static CommandPod>,
        Option<&'static Decoupler>,
        Option<&'static Adapter>,
        Option<&'static FuelTank>,
        Option<&'static Fuselage>,
        Option<&'static Engine>,
        Option<&'static AirIntake>,
        Option<&'static SurfaceMount>,
        Option<&'static Children>,
        Option<&'static PartShaderHandle>,
        Has<PartMaterial>,
    ),
    (Or<(Added<Part>, Changed<AttachNodes>)>, With<EditorPart>),
>;

pub(super) fn rebuild_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    assets: Res<EditorAssets>,
    state: Res<EditorState>,
    parts: VisualQuery,
    stale: Query<(), Or<(With<PartVisual>, With<AttachNodePin>)>>,
) {
    for (
        e,
        nodes,
        pod,
        dec,
        adapter,
        tank,
        fuselage,
        engine,
        intake,
        surface,
        children,
        part_shader,
        has_part_mat,
    ) in parts.iter()
    {
        if let Some(ch) = children {
            for c in ch.into_iter() {
                if stale.get(*c).is_ok() {
                    commands.entity(*c).despawn();
                }
            }
        }

        if engine.is_some_and(|e| e.geometry == EngineGeometry::JetNacelle)
            && surface.is_some_and(|m| m.kind == SurfaceMountKind::WingPylon)
        {
            continue;
        }

        // ---- Body visual --------------------------------------------------
        if let Some(spec) = visual_spec(nodes, pod, dec, adapter, tank, fuselage, engine, intake) {
            let mesh = meshes.add(spec.mesh);

            // Parts carrying `PartMaterial` render with `ShipPartMaterial`
            // (procedural stainless); others use the shared
            // `StandardMaterial`. The ship-material asset is created lazily
            // on first rebuild and cached on the part entity so resizing
            // doesn't churn assets or drop per-part state (seed/tint).
            let body_id = if has_part_mat {
                let params = ship_part_params(nodes, tank, fuselage, dec, adapter, e.index_u32());
                let handle = match part_shader {
                    Some(h) => h.0.clone(),
                    None => {
                        let h = ship_materials.add(ShipPartMaterial {
                            base: stainless_steel_base(),
                            extension: ShipPartExtension {
                                params,
                                ..Default::default()
                            },
                        });
                        commands.entity(e).insert(PartShaderHandle(h.clone()));
                        h
                    }
                };
                commands
                    .spawn((
                        Mesh3d(mesh),
                        MeshMaterial3d(handle),
                        Transform::from_xyz(0.0, -spec.height * 0.5, 0.0),
                        Visibility::default(),
                        PartVisual,
                        PartBody(e),
                        Pickable::default(),
                    ))
                    .observe(on_body_click)
                    .id()
            } else {
                let initial_material = if Some(e) == state.selected {
                    assets.selected_material.clone()
                } else {
                    assets.part_material.clone()
                };
                commands
                    .spawn((
                        Mesh3d(mesh),
                        MeshMaterial3d(initial_material),
                        Transform::from_xyz(0.0, -spec.height * 0.5, 0.0),
                        Visibility::default(),
                        PartVisual,
                        PartBody(e),
                        Pickable::default(),
                    ))
                    .observe(on_body_click)
                    .id()
            };
            commands.entity(e).add_child(body_id);
        }

        // ---- Attach node pins --------------------------------------------
        for (id, node) in &nodes.nodes {
            let pin = commands
                .spawn((
                    Mesh3d(assets.node_mesh.clone()),
                    MeshMaterial3d(assets.pending_node_material.clone()),
                    Transform::from_translation(node.offset),
                    Visibility::Hidden,
                    AttachNodePin {
                        part: e,
                        node_id: id.clone(),
                    },
                    Pickable::default(),
                ))
                .observe(on_pin_click)
                .id();
            commands.entity(e).add_child(pin);
        }
    }
}

/// Build (or rebuild) the mesh child for each wing whose shape, mount, or
/// host diameter just changed. Wings are surface-mounted lifting surfaces,
/// not bodies of revolution, so they live outside `rebuild_visuals`: the
/// mesh is generated in the host-local frame by [`build_wing_mesh`] and the
/// wing entity's transform (set in `update_part_transforms`) places it on
/// the hull. Uses the shared part materials so selection / hover highlight
/// flows through `update_selection_highlight` like any other part.
pub(super) fn rebuild_wing_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    assets: Res<EditorAssets>,
    state: Res<EditorState>,
    wings: Query<
        (Entity, &Wing, &SurfaceMount, Option<&Children>),
        (
            Or<(Added<Wing>, Changed<Wing>, Changed<SurfaceMount>)>,
            With<EditorPart>,
        ),
    >,
    host_nodes: Query<&AttachNodes>,
    hosts: Query<&Fuselage>,
    stale: Query<(), With<WingVisual>>,
) {
    for (e, wing, mount, children) in wings.iter() {
        if let Some(ch) = children {
            for c in ch.into_iter() {
                if stale.get(*c).is_ok() {
                    commands.entity(*c).despawn();
                }
            }
        }
        let top_d = host_top_diameter(&host_nodes, mount.parent);
        let (parent_radius, _) =
            host_mount_geometry(hosts.get(mount.parent).ok(), top_d, mount.station, mount.angle);
        let mesh = meshes.add(build_wing_mesh(wing, mount.angle, parent_radius));
        let material = if Some(e) == state.selected {
            assets.selected_material.clone()
        } else {
            assets.part_material.clone()
        };
        let body = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material.clone()),
                Transform::IDENTITY,
                Visibility::default(),
                WingVisual,
                PartBody(e),
                Pickable::default(),
            ))
            .observe(on_body_click)
            .id();
        let parent = e;
        commands.queue(move |world: &mut World| {
            if world.get_entity(parent).is_ok() {
                world.entity_mut(body).insert(ChildOf(parent));
            } else {
                world.entity_mut(body).despawn();
            }
        });

        // Control surfaces, shown deflected to a small angle so they read as
        // distinct hinged panels in the editor (no flight sim here).
        for surface in &wing.control_surfaces {
            let built = build_control_surface_mesh(wing, surface, mount.angle, parent_radius);
            let preview =
                Quat::from_axis_angle(built.geometry.hinge_axis, surface.max_deflection * 0.4);
            let cs = commands
                .spawn((
                    Mesh3d(meshes.add(built.mesh)),
                    MeshMaterial3d(material.clone()),
                    Transform::from_translation(built.geometry.hinge_anchor).with_rotation(preview),
                    Visibility::default(),
                    WingVisual,
                    PartBody(e),
                    Pickable::default(),
                ))
                .observe(on_body_click)
                .id();
            commands.queue(move |world: &mut World| {
                if world.get_entity(parent).is_ok() {
                    world.entity_mut(cs).insert(ChildOf(parent));
                } else {
                    world.entity_mut(cs).despawn();
                }
            });
        }
    }
}

pub(super) fn rebuild_nacelle_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    assets: Res<EditorAssets>,
    state: Res<EditorState>,
    engines: Query<
        (Entity, &Engine, &SurfaceMount, Option<&Children>),
        (
            Or<(Added<Engine>, Changed<SurfaceMount>)>,
            With<EditorPart>,
        ),
    >,
    wings: Query<&Wing>,
    surface_mounts: Query<&SurfaceMount>,
    host_nodes: Query<&AttachNodes>,
    hosts: Query<&Fuselage>,
    stale: Query<(), With<NacelleVisual>>,
) {
    for (e, engine, mount, children) in engines.iter() {
        if engine.geometry != EngineGeometry::JetNacelle
            || mount.kind != SurfaceMountKind::WingPylon
        {
            continue;
        }
        if let Some(ch) = children {
            for c in ch.into_iter() {
                if stale.get(*c).is_ok() {
                    commands.entity(*c).despawn();
                }
            }
        }

        let Ok(wing) = wings.get(mount.parent) else {
            continue;
        };
        let Ok(wing_mount) = surface_mounts.get(mount.parent) else {
            continue;
        };
        let top_d = host_top_diameter(&host_nodes, wing_mount.parent);
        let (parent_radius, _) = host_mount_geometry(
            hosts.get(wing_mount.parent).ok(),
            top_d,
            wing_mount.station,
            wing_mount.angle,
        );
        let mesh = meshes.add(build_jet_nacelle_pylon_mesh(
            engine,
            JetNacelleMount {
                wing,
                wing_mount_angle: wing_mount.angle,
                parent_radius,
                span_fraction: mount.station,
                chord_fraction: mount.angle,
            },
        ));
        let material = if Some(e) == state.selected {
            assets.selected_material.clone()
        } else {
            assets.part_material.clone()
        };
        let body = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                Transform::IDENTITY,
                Visibility::default(),
                NacelleVisual,
                PartBody(e),
                Pickable::default(),
            ))
            .observe(on_body_click)
            .id();
        let parent = e;
        commands.queue(move |world: &mut World| {
            if world.get_entity(parent).is_ok() {
                world.entity_mut(body).insert(ChildOf(parent));
            } else {
                world.entity_mut(body).despawn();
            }
        });
    }
}

/// Build (or rebuild) the mesh child for each gearbox whose dimensions, mount,
/// or host diameter just changed. A gearbox is a single footprint part that
/// draws all of its legs ([`build_gear_mesh`]); the mesh is in the host-local
/// frame and the gear entity's transform (set in `update_part_transforms`)
/// places it on the belly. Mirrors `rebuild_wing_visuals` — no symmetry.
pub(super) fn rebuild_gear_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    assets: Res<EditorAssets>,
    state: Res<EditorState>,
    gears: Query<
        (Entity, &Gear, &SurfaceMount, Option<&Children>),
        (
            Or<(Added<Gear>, Changed<Gear>, Changed<SurfaceMount>)>,
            With<EditorPart>,
        ),
    >,
    host_nodes: Query<&AttachNodes>,
    hosts: Query<&Fuselage>,
    stale: Query<(), Or<(With<GearVisual>, With<GearBayVisual>)>>,
) {
    for (e, gear, mount, children) in gears.iter() {
        if let Some(ch) = children {
            for c in ch.into_iter() {
                if stale.get(*c).is_ok() {
                    commands.entity(*c).despawn();
                }
            }
        }
        let top_d = host_top_diameter(&host_nodes, mount.parent);
        let (parent_radius, _) =
            host_mount_geometry(hosts.get(mount.parent).ok(), top_d, mount.station, mount.angle);
        let mesh = meshes.add(build_gear_mesh(gear, mount.angle, parent_radius));
        let material = if Some(e) == state.selected {
            assets.selected_material.clone()
        } else {
            assets.gear_material.clone()
        };
        let body = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                Transform::IDENTITY,
                Visibility::default(),
                GearVisual,
                PartBody(e),
                Pickable::default(),
            ))
            .observe(on_body_click)
            .id();
        // Stow-bay ghost: an x-ray translucent box inside the fuselage. Not
        // pickable, no `PartBody` (so the selection-highlight system leaves its
        // ghost material alone), no click observer.
        let bay = commands
            .spawn((
                Mesh3d(meshes.add(build_gear_bay_mesh(gear, mount.angle, parent_radius))),
                MeshMaterial3d(assets.gear_bay_material.clone()),
                Transform::IDENTITY,
                Visibility::default(),
                GearBayVisual,
                Pickable::IGNORE,
            ))
            .id();
        let parent = e;
        commands.queue(move |world: &mut World| {
            if world.get_entity(parent).is_ok() {
                world.entity_mut(body).insert(ChildOf(parent));
                world.entity_mut(bay).insert(ChildOf(parent));
            } else {
                world.entity_mut(body).despawn();
                world.entity_mut(bay).despawn();
            }
        });
    }
}

pub(super) fn update_part_transforms(
    ships: Query<&Ship, With<EditorPart>>,
    attachments: Query<(Entity, &Attachment), With<EditorPart>>,
    surface_mounts: Query<(Entity, &SurfaceMount), With<EditorPart>>,
    nodes: Query<&AttachNodes>,
    hosts: Query<&Fuselage>,
    orientation: Res<BuildOrientation>,
    mut transforms: Query<&mut Transform, (With<Part>, With<EditorPart>)>,
) {
    let mut children_map: HashMap<Entity, Vec<(Entity, Attachment)>> = HashMap::new();
    for (e, att) in attachments.iter() {
        children_map
            .entry(att.parent)
            .or_default()
            .push((e, att.clone()));
    }

    for ship in ships.iter() {
        if let Ok(mut t) = transforms.get_mut(ship.root) {
            t.translation = Vec3::ZERO;
            t.rotation = Quat::IDENTITY;
        }
        let mut queue: VecDeque<Entity> = VecDeque::from([ship.root]);
        while let Some(parent) = queue.pop_front() {
            let parent_pos = transforms
                .get(parent)
                .map(|t| t.translation)
                .unwrap_or(Vec3::ZERO);
            let Ok(parent_nodes) = nodes.get(parent) else {
                continue;
            };
            let parent_pos_and_nodes: Vec<(Entity, Vec3)> = children_map
                .get(&parent)
                .map(|kids| {
                    kids.iter()
                        .filter_map(|(c, att)| {
                            let pn = parent_nodes.get(&att.parent_node)?;
                            let child_offset = nodes
                                .get(*c)
                                .ok()
                                .and_then(|cn| cn.get(&att.my_node))
                                .map(|n| n.offset)
                                .unwrap_or(Vec3::ZERO);
                            Some((*c, parent_pos + pn.offset - child_offset))
                        })
                        .collect()
                })
                .unwrap_or_default();
            for (child, pos) in parent_pos_and_nodes {
                if let Ok(mut ct) = transforms.get_mut(child) {
                    ct.translation = pos;
                    ct.rotation = Quat::IDENTITY;
                }
                queue.push_back(child);
            }
        }
    }

    // Surface-mounted parts sit in their host-local frame. Body-skin mounts
    // (wings) move down the host body axis; wing-pylon mounts (nacelles)
    // inherit the wing origin because the pylon mesh carries the offsets.
    //
    // Process in dependency order — BodySkin first, then WingPylon — because a
    // nacelle's parent is a wing that is itself a surface mount. Reading a
    // parent before it has been positioned this frame would pull its stale
    // (already rigid-rotated) translation from the previous frame, which the
    // rigid rotation below would then rotate a second time. Two passes keep
    // every parent upright and freshly positioned before its child reads it.
    let position_mount = |transforms: &mut Query<&mut Transform, (With<Part>, With<EditorPart>)>,
                          part: Entity,
                          mount: &SurfaceMount| {
        let Ok(parent_t) = transforms.get(mount.parent).map(|t| t.translation) else {
            return;
        };
        let local_offset = match mount.kind {
            SurfaceMountKind::BodySkin => {
                let host_height = nodes
                    .get(mount.parent)
                    .ok()
                    .and_then(|n| n.get("bottom").map(|nd| -nd.offset.y))
                    .unwrap_or(0.0);
                // On a loft host the centerline rises toward the tail
                // (upsweep) / drops at the nose (droop); the mount must follow
                // it along +Z. Flat (zero) for a plain cylinder host.
                let top_d = host_top_diameter(&nodes, mount.parent);
                let (_, v_offset) =
                    host_mount_geometry(hosts.get(mount.parent).ok(), top_d, mount.station, 0.0);
                Vec3::new(0.0, -mount.station * host_height, v_offset)
            }
            SurfaceMountKind::WingPylon => Vec3::ZERO,
        };
        if let Ok(mut pt) = transforms.get_mut(part) {
            pt.translation = parent_t + local_offset;
            pt.rotation = Quat::IDENTITY;
        }
    };
    for (part, mount) in surface_mounts.iter() {
        if mount.kind == SurfaceMountKind::BodySkin {
            position_mount(&mut transforms, part, mount);
        }
    }
    for (part, mount) in surface_mounts.iter() {
        if mount.kind == SurfaceMountKind::WingPylon {
            position_mount(&mut transforms, part, mount);
        }
    }

    // Build-layout: everything above is computed in the upright build frame;
    // lay the whole assembly down rigidly for the horizontal (aircraft)
    // layout. Identity in vertical mode, so this is a no-op for rockets.
    let r = orientation.rotation();
    if r != Quat::IDENTITY {
        for mut t in transforms.iter_mut() {
            t.translation = r * t.translation;
            t.rotation = r;
        }
    }
}

pub(super) fn update_node_pin_style(
    state: Res<EditorState>,
    catalog: Res<PartCatalog>,
    assets: Res<EditorAssets>,
    attachments: Query<&Attachment, With<EditorPart>>,
    mut pins: Query<(
        &AttachNodePin,
        &mut MeshMaterial3d<StandardMaterial>,
        &mut Visibility,
    )>,
) {
    let occupied: HashSet<(Entity, String)> = attachments
        .iter()
        .map(|a| (a.parent, a.parent_node.clone()))
        .collect();
    let pending_uses_nodes = state.pending.as_ref().is_some_and(|p| {
        !matches!(p.params, crate::PartParams::Wing { .. })
            && !catalog.resolve(&p.catalog_id).is_ok_and(|entry| {
                matches!(
                    entry,
                    CatalogEntry::Engine(e) if e.geometry == EngineGeometry::JetNacelle
                )
            })
    });

    for (pin, mut mat, mut vis) in pins.iter_mut() {
        let is_occupied = occupied.contains(&(pin.part, pin.node_id.clone()));

        // Pins only appear while a part is pending, and only on unoccupied
        // nodes.
        *vis = if pending_uses_nodes && !is_occupied {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };

        mat.0 = assets.pending_node_material.clone();
    }
}

/// Propagate a part's own kind-component values (e.g. `CommandPod.diameter`,
/// `FuelTank.length`, `Engine.diameter`) into its own AttachNodes, only
/// touching the component when a value actually differs. This way editor
/// sliders drive AttachNodes → rebuild_visuals deterministically, without
/// the kind component's spurious Changed signals causing per-frame respawns.
///
/// For parametric radius parts (Decoupler/Adapter/FuelTank) the sync is
/// bidirectional by root state:
/// - **Root**: `self.diameter → nodes.top` so the Diameter slider drives
///   the part's visual size.
/// - **Child**: `nodes.top → self.diameter` so the diameter inherited via
///   `sizing::propagate_node_sizes` is mirrored onto the component. This
///   way a later re-root starts from the displayed size instead of
///   snapping back to the palette's placeholder.
pub(super) fn sync_self_nodes(
    mut q: Query<
        (
            &mut AttachNodes,
            Option<&Attachment>,
            Option<&CommandPod>,
            Option<&mut Decoupler>,
            Option<&mut Adapter>,
            Option<&mut FuelTank>,
            Option<&Engine>,
        ),
        With<EditorPart>,
    >,
) {
    for (mut nodes, attachment, pod, mut dec, mut adapter, mut tank, engine) in q.iter_mut() {
        let is_root = attachment.is_none();
        let mut targets: Vec<(String, f32, Vec3)> = Vec::new();
        if let Some(p) = pod {
            let d = p.diameter;
            targets.push((
                "bottom".into(),
                d,
                Vec3::new(0.0, -d * p.geometry.length_factor(), 0.0),
            ));
        }
        // Read kind-component fields through `as_ref()` so the borrow only
        // goes through Bevy's `Mut::deref` (no Changed trigger). The write
        // path below reaches for `as_mut()` only when the value actually
        // needs to change.
        if let Some(d) = dec.as_ref() {
            let self_d = d.diameter;
            let top_d = if is_root {
                targets.push(("top".into(), self_d, Vec3::ZERO));
                self_d
            } else {
                let inherited = nodes.get("top").map(|n| n.diameter).unwrap_or(self_d);
                if (self_d - inherited).abs() > f32::EPSILON
                    && let Some(m) = dec.as_mut()
                {
                    m.diameter = inherited;
                }
                inherited
            };
            targets.push(("bottom".into(), top_d, Vec3::new(0.0, -0.2, 0.0)));
        }
        if let Some(a) = adapter.as_ref() {
            let self_d = a.diameter;
            let target_d = a.target_diameter;
            let top_d = if is_root {
                targets.push(("top".into(), self_d, Vec3::ZERO));
                self_d
            } else {
                let inherited = nodes.get("top").map(|n| n.diameter).unwrap_or(self_d);
                if (self_d - inherited).abs() > f32::EPSILON
                    && let Some(m) = adapter.as_mut()
                {
                    m.diameter = inherited;
                }
                inherited
            };
            let h = ((top_d + target_d) * 0.5).max(0.4);
            targets.push(("bottom".into(), target_d, Vec3::new(0.0, -h, 0.0)));
        }
        if let Some(t) = tank.as_ref() {
            let self_d = t.diameter;
            let length = t.length;
            let top_d = if is_root {
                targets.push(("top".into(), self_d, Vec3::ZERO));
                self_d
            } else {
                let inherited = nodes.get("top").map(|n| n.diameter).unwrap_or(self_d);
                if (self_d - inherited).abs() > f32::EPSILON
                    && let Some(m) = tank.as_mut()
                {
                    m.diameter = inherited;
                }
                inherited
            };
            targets.push(("bottom".into(), top_d, Vec3::new(0.0, -length, 0.0)));
        }
        if let Some(e) = engine {
            targets.push(("top".into(), e.diameter, Vec3::ZERO));
        }
        let needs_update = targets.iter().any(|(id, d, off)| {
            nodes
                .get(id)
                .map(|n| {
                    (n.diameter - *d).abs() > f32::EPSILON
                        || n.offset.distance_squared(*off) > f32::EPSILON
                })
                .unwrap_or(false)
        });
        if !needs_update {
            continue;
        }
        for (id, d, off) in &targets {
            if let Some(n) = nodes.nodes.get_mut(id) {
                n.diameter = *d;
                n.offset = *off;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tank resize arrow (parametric handle)
// ---------------------------------------------------------------------------

/// Spawn a single resize arrow per fuel tank on creation. The arrow is a
/// child of the tank entity, hidden until the tank becomes the current
/// selection, and positioned each frame by `update_tank_resize_arrow`.
pub(super) fn spawn_tank_resize_arrow(
    mut commands: Commands,
    assets: Res<EditorAssets>,
    new_tanks: Query<Entity, (Added<FuelTank>, With<EditorPart>)>,
) {
    for tank in new_tanks.iter() {
        let arrow = commands
            .spawn((
                Mesh3d(assets.resize_arrow_mesh.clone()),
                MeshMaterial3d(assets.resize_arrow_material.clone()),
                Transform::default(),
                Visibility::Hidden,
                TankResizeArrow { tank },
                Pickable::default(),
            ))
            .observe(on_arrow_drag_start)
            .observe(on_arrow_drag_end)
            .id();
        commands.entity(tank).add_child(arrow);
    }
}

/// Show the arrow only while the owning tank is selected; each frame, place
/// it on the camera-facing side of the tank at mid-height with the tip
/// pointing down along the tank's growth axis.
pub(super) fn update_tank_resize_arrow(
    state: Res<EditorState>,
    orientation: Res<BuildOrientation>,
    tanks: Query<(&FuelTank, &AttachNodes), Without<TankResizeArrow>>,
    cameras: Query<
        &Transform,
        (
            With<EditorViewCamera>,
            Without<TankResizeArrow>,
            Without<FuelTank>,
        ),
    >,
    mut arrows: Query<(&TankResizeArrow, &mut Transform, &mut Visibility)>,
) {
    let Ok(cam_transform) = cameras.single() else {
        return;
    };

    for (arrow, mut transform, mut vis) in arrows.iter_mut() {
        let is_selected = state.selected == Some(arrow.tank);
        let Ok((tank, nodes)) = tanks.get(arrow.tank) else {
            if *vis != Visibility::Hidden {
                *vis = Visibility::Hidden;
            }
            continue;
        };

        if !is_selected {
            if *vis != Visibility::Hidden {
                *vis = Visibility::Hidden;
            }
            continue;
        }
        if *vis != Visibility::Inherited {
            *vis = Visibility::Inherited;
        }

        // Place the arrow on the camera's right so it doesn't occlude the
        // tank body. The arrow is a child of the (possibly laid-down) tank,
        // so convert the world camera-right into the tank's build frame
        // before using it as a local offset; the length axis stays local −Y.
        let cam_right = orientation.rotation().inverse() * cam_transform.right().as_vec3();
        let right_xz = Vec2::new(cam_right.x, cam_right.z)
            .try_normalize()
            .unwrap_or(Vec2::X);
        let radius = nodes.get("top").map(|n| n.diameter * 0.5).unwrap_or(0.5);
        let offset_r = radius + 0.55;
        transform.translation = Vec3::new(
            right_xz.x * offset_r,
            -tank.length * 0.5,
            right_xz.y * offset_r,
        );
        // Bevy's Cone has its tip at +Y and base at -Y; rotate PI around X
        // to point the tip down (i.e., the direction the tank grows). Local,
        // so it composes with the tank's build-layout rotation.
        transform.rotation = Quat::from_rotation_x(std::f32::consts::PI);
    }
}

/// On drag start: snapshot the tank's current length, the cursor origin,
/// and project the world growth axis (-Y) into screen space. Subsequent
/// cursor motion is decomposed along that axis and rescaled to world units.
pub(super) fn on_arrow_drag_start(
    trigger: On<Pointer<DragStart>>,
    arrows: Query<&TankResizeArrow>,
    tanks: Query<(&FuelTank, &Transform)>,
    camera_q: Query<(&Camera, &GlobalTransform), With<EditorViewCamera>>,
    orientation: Res<BuildOrientation>,
    mut drag: ResMut<TankResizeDrag>,
) {
    let event = trigger.event();
    let Ok(arrow) = arrows.get(event.entity) else {
        return;
    };
    let Ok((tank, tank_transform)) = tanks.get(arrow.tank) else {
        return;
    };
    let Ok((camera, cam_transform)) = camera_q.single() else {
        return;
    };

    let origin_world = tank_transform.translation;
    // Tanks grow along the body −Y axis, which the build layout may have
    // laid down — use the rotated axis so the drag tracks the visible length.
    let grow_world = origin_world + orientation.rotation() * Vec3::NEG_Y;
    let Ok(origin_screen) = camera.world_to_viewport(cam_transform, origin_world) else {
        return;
    };
    let Ok(grow_screen) = camera.world_to_viewport(cam_transform, grow_world) else {
        return;
    };

    let axis = grow_screen - origin_screen;
    let axis_len = axis.length();
    if axis_len < 1e-3 {
        return;
    }

    drag.active = Some(TankDragState {
        tank: arrow.tank,
        start_length: tank.length,
        start_cursor: event.pointer_location.position,
        screen_axis: axis / axis_len,
        world_per_pixel: 1.0 / axis_len,
    });
}

pub(super) fn on_arrow_drag_end(_trigger: On<Pointer<DragEnd>>, mut drag: ResMut<TankResizeDrag>) {
    drag.active = None;
}

/// Apply the active drag to the tank's length each frame. Bails (and
/// clears) if the button was released without a DragEnd — can happen when
/// the pointer leaves the window mid-drag.
pub(super) fn update_tank_resize_drag(
    mut drag: ResMut<TankResizeDrag>,
    windows: Query<&Window, With<PrimaryWindow>>,
    input: Res<ShipyardInputIntent>,
    mut tanks: Query<(&mut FuelTank, &AttachNodes)>,
) {
    let Some(state) = drag.active.as_ref() else {
        return;
    };
    if !input.primary_pressed {
        drag.active = None;
        return;
    }
    let Ok(window) = windows.single() else { return };
    let Some(cursor) = window.cursor_position() else {
        return;
    };

    let cursor_delta = cursor - state.start_cursor;
    let pixels_along = cursor_delta.dot(state.screen_axis);
    let world_growth = pixels_along * state.world_per_pixel;
    let raw_length = state.start_length + world_growth;
    // Magnetic snap: smooth drag in-between, stick to nearest 0.5 within
    // a small neighborhood so users can dial in round values without
    // losing fine control.
    const SNAP_GRID: f32 = 0.5;
    const SNAP_THRESHOLD: f32 = 0.06;
    const MAX_LENGTH_OVER_DIAMETER: f32 = 8.0;
    let nearest = (raw_length / SNAP_GRID).round() * SNAP_GRID;
    let length = if (raw_length - nearest).abs() < SNAP_THRESHOLD {
        nearest
    } else {
        raw_length
    };

    if let Ok((mut tank, nodes)) = tanks.get_mut(state.tank) {
        let diameter = nodes
            .get("top")
            .map(|n| n.diameter)
            .unwrap_or(tank.diameter);
        let new_length = length.clamp(0.5, MAX_LENGTH_OVER_DIAMETER * diameter);
        if (tank.length - new_length).abs() > f32::EPSILON {
            tank.length = new_length;
        }
    }
}

// ---------------------------------------------------------------------------
// Selection / hover highlight
// ---------------------------------------------------------------------------

/// Keep `ShipPartMaterial` uniforms in sync with the part's dimensions
/// (tank length, decoupler/tank radius). Triggered whenever the
/// kind-component or attach nodes change, so slider and resize-drag
/// updates flow through to the panel / rivet layout live.
pub(super) fn update_part_shader_params(
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    parts: Query<
        (
            &AttachNodes,
            &PartShaderHandle,
            Option<&FuelTank>,
            Option<&Fuselage>,
            Option<&Decoupler>,
            Option<&Adapter>,
        ),
        (
            Or<(
                Changed<FuelTank>,
                Changed<Fuselage>,
                Changed<Decoupler>,
                Changed<Adapter>,
                Changed<AttachNodes>,
            )>,
            With<EditorPart>,
        ),
    >,
) {
    for (nodes, handle, tank, fuselage, dec, adapter) in parts.iter() {
        let Some(mat) = ship_materials.get_mut(&handle.0) else {
            continue;
        };
        let params =
            ship_part_params(nodes, tank, fuselage, dec, adapter, mat.extension.params.seed);
        mat.extension.params.length = params.length;
        mat.extension.params.radius_top = params.radius_top;
        mat.extension.params.radius_bottom = params.radius_bottom;
    }
}

/// Selection / hover tint for parts rendering through `ShipPartMaterial`
/// (tanks, decouplers). Writes into the material's tint uniform rather
/// than swapping handles so each part keeps its procedural detail.
/// Shrouds are excluded — they manage their own hover feedback via
/// `update_shroud_transparency`.
pub(super) fn update_part_shader_highlight(
    state: Res<EditorState>,
    hover_map: Res<HoverMap>,
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    bodies: Query<
        (Entity, &PartBody, &MeshMaterial3d<ShipPartMaterial>),
        Without<super::shrouds::ShroudBody>,
    >,
) {
    let hovered: HashSet<Entity> = hover_map
        .0
        .values()
        .flat_map(|hovers| hovers.keys().copied())
        .collect();

    for (body_entity, body, mesh_mat) in bodies.iter() {
        let target = if Some(body.0) == state.selected {
            Vec3::new(0.88, 1.0, 1.35)
        } else if hovered.contains(&body_entity) {
            Vec3::new(1.08, 1.08, 1.12)
        } else {
            Vec3::ONE
        };
        if let Some(mat) = ship_materials.get_mut(&mesh_mat.0)
            && (mat.extension.params.tint - target).length_squared() > 1.0e-6
        {
            mat.extension.params.tint = target;
        }
    }
}

/// Swap each part body's material based on selection and hover state.
/// Priority: selected > hovered > default.
pub(super) fn update_selection_highlight(
    state: Res<EditorState>,
    assets: Res<EditorAssets>,
    hover_map: Res<HoverMap>,
    mut bodies: Query<(
        Entity,
        &PartBody,
        Has<GearVisual>,
        &mut MeshMaterial3d<StandardMaterial>,
    )>,
) {
    let hovered: HashSet<Entity> = hover_map
        .0
        .values()
        .flat_map(|hovers| hovers.keys().copied())
        .collect();

    for (body_entity, body, is_gear, mut mat) in bodies.iter_mut() {
        let target = if Some(body.0) == state.selected {
            &assets.selected_material
        } else if hovered.contains(&body_entity) {
            &assets.hover_material
        } else if is_gear {
            &assets.gear_material
        } else {
            &assets.part_material
        };
        if mat.0.id() != target.id() {
            mat.0 = target.clone();
        }
    }
}

/// Propagate the coupled neighbor's [`MaterialKind`] onto parts that
/// visually continue with whatever is attached to their `bottom` node —
/// currently [`Decoupler`] (so the decoupler + its shroud read as part
/// of the stage below on staging) and [`Adapter`] (so a diameter
/// transition inherits from the narrower stage it feeds into). Parts
/// with nothing attached below keep their default [`MaterialKind`].
pub(super) fn propagate_coupled_material(
    attachments: Query<(Entity, &Attachment), With<EditorPart>>,
    mut params: ParamSet<(
        Query<(Entity, &PartMaterial), With<EditorPart>>,
        Query<(Entity, &mut PartMaterial), (Or<(With<Decoupler>, With<Adapter>)>, With<EditorPart>)>,
    )>,
) {
    // Build parent → bottom-attached-child entity map.
    let mut coupled: HashMap<Entity, Entity> = HashMap::new();
    for (child, att) in attachments.iter() {
        if att.parent_node == "bottom" {
            coupled.insert(att.parent, child);
        }
    }

    // Snapshot every part's current MaterialKind so read + write on
    // PartMaterial can both run in this system without conflicting
    // mutable borrows.
    let kinds: HashMap<Entity, MaterialKind> =
        params.p0().iter().map(|(e, m)| (e, m.kind)).collect();

    for (entity, mut my_mat) in params.p1().iter_mut() {
        let Some(coupled_entity) = coupled.get(&entity).copied() else {
            continue;
        };
        let Some(&kind) = kinds.get(&coupled_entity) else {
            continue;
        };
        if my_mat.kind != kind {
            my_mat.kind = kind;
        }
    }
}

// ---------------------------------------------------------------------------
// Placement preview ghost
// ---------------------------------------------------------------------------

/// Live placement preview. While a body-skin footprint part (gear / wing) is
/// pending and the cursor hovers a compatible hull, draw a translucent ghost of
/// it at the snapped `(station, angle)` it would land on — so you aim instead of
/// clicking and praying. The ghost is one reused entity; its (small) mesh is
/// rebuilt only when the snapped angle / host / params change. WingPylon
/// (nacelle-on-wing) mounts aren't cylinder mounts, so they get no preview yet.
pub(super) fn update_placement_preview(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    ui_gate: Res<EditorUiGate>,
    state: Res<EditorState>,
    catalog: Res<PartCatalog>,
    snap: Res<PlacementSnap>,
    orientation: Res<BuildOrientation>,
    assets: Res<EditorAssets>,
    hover_map: Res<HoverMap>,
    mut preview: ResMut<PlacementPreview>,
    bodies: Query<&PartBody>,
    wing_marker: Query<(), With<Wing>>,
    part_transforms: Query<&Transform, With<Part>>,
    host_nodes: Query<&AttachNodes>,
    hosts: Query<&Fuselage>,
    mut ghosts: Query<
        (&mut Transform, &mut Visibility, &mut Mesh3d),
        (With<PreviewGhost>, Without<Part>),
    >,
) {
    // Compute the desired ghost placement (None ⇒ hide it). An IIFE so the many
    // early-outs read cleanly; its borrows end before we mutate `preview`.
    let placement: Option<(Vec3, Quat, PreviewSig, Option<Handle<Mesh>>)> = (|| {
        let pending = state.pending.as_ref()?;
        // Only body-skin footprint parts get a preview (gear, wings).
        if !matches!(
            pending.params,
            crate::PartParams::Wing { .. } | crate::PartParams::Gear { .. }
        ) {
            return None;
        }
        if ui_gate.pointer_busy {
            return None;
        }

        // Hovered host = a non-wing body under the cursor, with its hit point.
        let mut found: Option<(Entity, Vec3)> = None;
        'outer: for hits in hover_map.0.values() {
            for (hovered, data) in hits.iter() {
                let Ok(pb) = bodies.get(*hovered) else {
                    continue;
                };
                if wing_marker.get(pb.0).is_ok() {
                    continue; // gear/wing mount on a hull, not on a wing
                }
                let Some(pos) = data.position else {
                    continue;
                };
                found = Some((pb.0, pos));
                break 'outer;
            }
        }
        let (host, hit) = found?;

        let (station, angle) = body_skin_mount(
            host,
            hit,
            &part_transforms,
            &host_nodes,
            &orientation,
            snap.enabled,
        );
        let host_n = host_nodes.get(host).ok();
        let top_d = host_top_diameter(&host_nodes, host);
        let (parent_radius, v_offset) =
            host_mount_geometry(hosts.get(host).ok(), top_d, station, angle);
        let host_height = host_n
            .and_then(|n| n.get("bottom").map(|nd| -nd.offset.y))
            .unwrap_or(0.0);
        let host_t = part_transforms
            .get(host)
            .map(|t| t.translation)
            .unwrap_or(Vec3::ZERO);

        // Match where `update_part_transforms` puts a body-skin part: the host
        // (already post-layout-rotation) plus the rotated local station offset
        // (including the loft's centerline upsweep along +Z).
        let r = orientation.rotation();
        let translation = host_t + r * Vec3::new(0.0, -station * host_height, v_offset);

        let sig = PreviewSig {
            host,
            angle,
            parent_radius,
            params: pending.params.clone(),
        };
        // Rebuild the mesh only when the silhouette actually changed (or there's
        // no ghost entity to carry it yet).
        let mesh = if preview.sig.as_ref() != Some(&sig) || preview.entity.is_none() {
            let m = match &pending.params {
                crate::PartParams::Gear {
                    strut_length,
                    wheel_radius,
                } => {
                    let (track_fraction, wheels_per_leg) = catalog
                        .resolve(&pending.catalog_id)
                        .ok()
                        .and_then(|e| match e {
                            CatalogEntry::Gear(g) => Some((g.track_fraction, g.wheels_per_leg)),
                            _ => None,
                        })
                        .unwrap_or((0.0, 1));
                    let g = Gear {
                        strut_length: *strut_length,
                        wheel_radius: *wheel_radius,
                        track_fraction,
                        wheels_per_leg,
                        dry_mass: 0.0,
                    };
                    build_gear_mesh(&g, angle, parent_radius)
                }
                crate::PartParams::Wing {
                    span,
                    root_chord,
                    tip_chord,
                    sweep,
                    dihedral,
                    thickness,
                    incidence,
                    control_surfaces,
                } => {
                    let w = Wing {
                        span: *span,
                        root_chord: *root_chord,
                        tip_chord: *tip_chord,
                        sweep: *sweep,
                        dihedral: *dihedral,
                        thickness: *thickness,
                        incidence: *incidence,
                        dry_mass: 0.0,
                        control_surfaces: control_surfaces.clone(),
                    };
                    build_wing_mesh(&w, angle, parent_radius)
                }
                _ => return None,
            };
            Some(meshes.add(m))
        } else {
            None
        };

        Some((translation, r, sig, mesh))
    })();

    match placement {
        None => {
            if let Some(prev) = preview.entity
                && let Ok((_, mut vis, _)) = ghosts.get_mut(prev)
            {
                *vis = Visibility::Hidden;
            }
            preview.sig = None;
        }
        Some((translation, rotation, sig, mesh)) => {
            let mut updated = false;
            if let Some(prev) = preview.entity
                && let Ok((mut t, mut vis, mut mesh3d)) = ghosts.get_mut(prev)
            {
                t.translation = translation;
                t.rotation = rotation;
                *vis = Visibility::Visible;
                if let Some(h) = &mesh {
                    mesh3d.0 = h.clone();
                }
                updated = true;
            }
            if !updated && let Some(h) = mesh {
                let id = commands
                    .spawn((
                        Mesh3d(h),
                        MeshMaterial3d(assets.preview_material.clone()),
                        Transform::from_translation(translation).with_rotation(rotation),
                        Visibility::Visible,
                        PreviewGhost,
                        Pickable::IGNORE,
                    ))
                    .id();
                preview.entity = Some(id);
            }
            preview.sig = Some(sig);
        }
    }
}
