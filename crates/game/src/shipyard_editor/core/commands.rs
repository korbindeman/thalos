//! Editor command processing: save / load / delete / re-root / place, the
//! blueprint collection walk, and the KSP linked-symmetry sync.

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use std::collections::{HashMap, VecDeque};

use thalos_shipyard::{
    Adapter, AttachNodes, Attachment, BuildLayout, CatalogEntry, CatalogRef, Connection, Decoupler,
    EngineGeometry, FuelTank, Fuselage, Gear, Part, PartBlueprint, PartCatalog, PartParams,
    PartResources, Ship, ShipBlueprint, SurfaceConnection, SurfaceMount, SurfaceMountKind,
    SymmetryGroup, SymmetryRole, Wing, default_control_surfaces,
};

use super::files::{SHIPS_DIR, list_ships, schema_ship_name, ship_path_for_name, ship_path_for_slug};
use super::placement::{host_group_members, surface_mount_from_hit};
use super::state::{
    BuildOrientation, EditorPart, EditorState, NextSymmetryId, PlacementSnap, SymmetryMode,
};

/// Seed the editor state once at startup: default ship name, the saved-ship
/// listing, and the initial status hint.
pub(super) fn init_editor_state(mut state: ResMut<EditorState>) {
    state.ship_name = "New Ship".into();
    state.ship_list = list_ships();
    state.status = "Click a part to begin".into();
}

/// Placement-mode toggles bundled into one [`SystemParam`] so `process_commands`
/// stays under Bevy's 16-argument system limit.
#[derive(SystemParam)]
pub(super) struct PlacementModes<'w> {
    symmetry: Res<'w, SymmetryMode>,
    snap: Res<'w, PlacementSnap>,
}

pub type CollectQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static CatalogRef,
        &'static PartResources,
        Option<&'static Decoupler>,
        Option<&'static Adapter>,
        Option<&'static FuelTank>,
        Option<&'static Fuselage>,
        Option<&'static Wing>,
        Option<&'static Gear>,
    ),
    With<EditorPart>,
>;

pub fn collect_blueprint(
    ship: &Ship,
    parts: &CollectQuery,
    attachments: &Query<(Entity, &Attachment), With<EditorPart>>,
    surface_mounts: &Query<(Entity, &SurfaceMount), With<EditorPart>>,
    groups: &Query<(Entity, &SymmetryGroup), With<EditorPart>>,
) -> Option<ShipBlueprint> {
    // Child graph unions both placement kinds so wings are ordered and
    // indexed alongside the node-stacked hull.
    let mut child_map: HashMap<Entity, Vec<Entity>> = HashMap::new();
    for (e, att) in attachments.iter() {
        child_map.entry(att.parent).or_default().push(e);
    }
    for (e, sm) in surface_mounts.iter() {
        child_map.entry(sm.parent).or_default().push(e);
    }

    let mut ordered: Vec<Entity> = Vec::new();
    let mut queue: VecDeque<Entity> = VecDeque::from([ship.root]);
    while let Some(e) = queue.pop_front() {
        ordered.push(e);
        if let Some(kids) = child_map.get(&e) {
            for c in kids {
                queue.push_back(*c);
            }
        }
    }

    let idx: HashMap<Entity, usize> = ordered.iter().enumerate().map(|(i, e)| (*e, i)).collect();

    let mut part_blueprints = Vec::with_capacity(ordered.len());
    for e in &ordered {
        let (_, cat_ref, res, dec, adapter, tank, fuselage, wing, gear) = parts.get(*e).ok()?;
        let params = if let Some(d) = dec {
            PartParams::Decoupler {
                diameter: d.diameter,
            }
        } else if let Some(a) = adapter {
            PartParams::Adapter {
                diameter: a.diameter,
                target_diameter: a.target_diameter,
            }
        } else if let Some(t) = tank {
            PartParams::Tank {
                diameter: t.diameter,
                length: t.length,
            }
        } else if let Some(f) = fuselage {
            PartParams::Fuselage {
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
            PartParams::Wing {
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
            PartParams::Gear {
                strut_length: g.strut_length,
                wheel_radius: g.wheel_radius,
            }
        } else {
            // Pods and engines carry no per-instance params.
            PartParams::None
        };
        // Persist amounts only — capacities are recomputed from the
        // catalog at load time.
        let resources: HashMap<thalos_shipyard::Resource, f32> =
            res.pools.iter().map(|(r, p)| (*r, p.amount)).collect();
        part_blueprints.push(PartBlueprint {
            catalog_id: cat_ref.id.clone(),
            params,
            resources: Some(resources),
        });
    }

    let mut connections = Vec::new();
    for (e, att) in attachments.iter() {
        if let (Some(&ci), Some(&pi)) = (idx.get(&e), idx.get(&att.parent)) {
            connections.push(Connection {
                parent: pi,
                parent_node: att.parent_node.clone(),
                child: ci,
                child_node: att.my_node.clone(),
            });
        }
    }

    let mut surface = Vec::new();
    for (e, sm) in surface_mounts.iter() {
        if let (Some(&ci), Some(&pi)) = (idx.get(&e), idx.get(&sm.parent)) {
            surface.push(SurfaceConnection {
                parent: pi,
                child: ci,
                kind: sm.kind,
                station: sm.station,
                angle: sm.angle,
                symmetry_group: groups.get(e).ok().map(|(_, g)| g.id),
            });
        }
    }

    Some(ShipBlueprint {
        name: schema_ship_name(&ship.name),
        root: 0,
        parts: part_blueprints,
        connections,
        surface_mounts: surface,
        // The save path stamps the editor's current layout; collection alone
        // (stats / Launch) leaves it unset.
        layout: None,
    })
}

/// KSP linked symmetry: keep every mirror counterpart in lockstep with its
/// group's primary — params copied (handed fields negated) and mount
/// reflected across the host X = 0 plane — so editing or moving the primary
/// updates the whole group. Writes are change-guarded so they don't
/// re-trigger the rebuild systems every frame.
pub(super) fn sync_symmetry_groups(
    groups: Query<(Entity, &SymmetryGroup), With<EditorPart>>,
    mut mounts: Query<&mut SurfaceMount>,
    mut wings: Query<&mut Wing>,
) {
    let mut primary_of_group: HashMap<u32, Entity> = HashMap::new();
    let mut members: HashMap<u32, Vec<Entity>> = HashMap::new();
    let mut role_of: HashMap<Entity, (u32, SymmetryRole)> = HashMap::new();
    for (e, g) in groups.iter() {
        members.entry(g.id).or_default().push(e);
        role_of.insert(e, (g.id, g.role));
        if g.role == SymmetryRole::Primary {
            primary_of_group.insert(g.id, e);
        }
    }
    // A host's mirror counterpart (for nested WingPylon parents): the member
    // of the host's own group with the opposite role.
    let host_mirror = |host: Entity| -> Option<Entity> {
        let (gid, role) = role_of.get(&host)?;
        let want = match role {
            SymmetryRole::Primary => SymmetryRole::Mirror,
            SymmetryRole::Mirror => SymmetryRole::Primary,
        };
        members
            .get(gid)?
            .iter()
            .copied()
            .find(|m| role_of.get(m).map(|(_, r)| *r) == Some(want))
    };

    for (gid, mems) in &members {
        let Some(&primary) = primary_of_group.get(gid) else {
            continue;
        };
        let Some(p_mount) = mounts.get(primary).ok().copied() else {
            continue;
        };
        let p_wing = wings.get(primary).ok().cloned();
        for &m in mems {
            if m == primary {
                continue;
            }
            let (parent, angle) = match p_mount.kind {
                // Same host, reflected azimuth.
                SurfaceMountKind::BodySkin => (p_mount.parent, -p_mount.angle),
                // The host wing is itself mirrored; mount on its counterpart
                // at the same local coords — the host reflection does the work.
                SurfaceMountKind::WingPylon => (
                    host_mirror(p_mount.parent).unwrap_or(p_mount.parent),
                    p_mount.angle,
                ),
            };
            let target = SurfaceMount {
                parent,
                kind: p_mount.kind,
                station: p_mount.station,
                angle,
            };
            if let Ok(mut mm) = mounts.get_mut(m)
                && *mm != target
            {
                *mm = target;
            }
            if let Some(w) = &p_wing
                && let Ok(mut mw) = wings.get_mut(m)
            {
                let mut tw = w.clone();
                tw.incidence = -w.incidence; // incidence is the one handed param
                if *mw != tw {
                    *mw = tw;
                }
            }
        }
    }
}

#[allow(clippy::collapsible_if)]
pub(super) fn process_commands(
    mut commands: Commands,
    mut state: ResMut<EditorState>,
    mut ships: Query<&mut Ship, With<EditorPart>>,
    parts_q: CollectQuery,
    attachments: Query<(Entity, &Attachment), With<EditorPart>>,
    surface_mounts: Query<(Entity, &SurfaceMount), With<EditorPart>>,
    part_transforms: Query<&Transform, With<Part>>,
    host_nodes: Query<&AttachNodes>,
    wings: Query<&Wing>,
    groups: Query<(Entity, &SymmetryGroup), With<EditorPart>>,
    orientation: Res<BuildOrientation>,
    modes: PlacementModes,
    mut next_sym_id: ResMut<NextSymmetryId>,
    all_parts: Query<Entity, (With<Part>, With<EditorPart>)>,
    all_ships: Query<Entity, (With<Ship>, With<EditorPart>)>,
    catalog: Res<PartCatalog>,
) {
    // ---- Save ---------------------------------------------------------
    if state.save_requested {
        state.save_requested = false;
        if let Some(ship_entity) = state.ship_entity
            && let Ok(ship) = ships.get(ship_entity)
        {
            match collect_blueprint(ship, &parts_q, &attachments, &surface_mounts, &groups) {
                Some(mut bp) => {
                    // Persist the editor's current build layout so the ship
                    // reopens the way it was authored (planes horizontal).
                    bp.layout = Some(if orientation.horizontal {
                        BuildLayout::Horizontal
                    } else {
                        BuildLayout::Vertical
                    });
                    match bp.to_ron() {
                    Ok(text) => {
                        let path = ship_path_for_name(&bp.name);
                        if let Err(e) = std::fs::create_dir_all(SHIPS_DIR) {
                            state.status = format!("mkdir failed: {e}");
                        } else {
                            match std::fs::write(&path, text) {
                                Ok(()) => {
                                    state.status = format!("Saved {}", path.display());
                                    state.refresh_list = true;
                                }
                                Err(e) => state.status = format!("Save failed: {e}"),
                            }
                        }
                    }
                    Err(e) => state.status = format!("Serialize failed: {e}"),
                    }
                }
                None => state.status = "Failed to collect blueprint".into(),
            }
        }
    }

    // ---- Load ---------------------------------------------------------
    if let Some(slug) = state.load_target.take() {
        let path = ship_path_for_slug(&slug);
        match std::fs::read_to_string(&path) {
            Ok(text) => match ShipBlueprint::from_ron(&text) {
                Ok(bp) => {
                    for e in all_parts.iter() {
                        commands.entity(e).despawn();
                    }
                    for e in all_ships.iter() {
                        commands.entity(e).despawn();
                    }
                    state.ship_root = None;
                    state.ship_entity = None;
                    state.selected = None;

                    let path_disp = path.display().to_string();
                    commands.queue(move |world: &mut World| {
                        let catalog = world.resource::<PartCatalog>().clone();
                        let mut cmds = world.commands();
                        let ship_entity = match bp.spawn(&mut cmds, &catalog) {
                            Ok(e) => e,
                            Err(err) => {
                                let mut st = world.resource_mut::<EditorState>();
                                st.status = format!("Spawn failed: {err}");
                                return;
                            }
                        };
                        world.flush();
                        let (root, name) = world
                            .get::<Ship>(ship_entity)
                            .map(|s| (Some(s.root), s.name.clone()))
                            .unwrap_or((None, String::new()));
                        // Tag the freshly spawned tree as editor-owned. The
                        // walk goes root→leaves through both placement kinds —
                        // exactly the entities `bp.spawn` created — so a
                        // coexisting flight ship in the same world is never
                        // touched.
                        world.entity_mut(ship_entity).insert(EditorPart);
                        if let Some(root) = root {
                            let mut child_map: HashMap<Entity, Vec<Entity>> = HashMap::new();
                            let mut att_q = world.query::<(Entity, &Attachment)>();
                            for (e, att) in att_q.iter(world) {
                                child_map.entry(att.parent).or_default().push(e);
                            }
                            let mut sm_q = world.query::<(Entity, &SurfaceMount)>();
                            for (e, sm) in sm_q.iter(world) {
                                child_map.entry(sm.parent).or_default().push(e);
                            }
                            let mut stack = vec![root];
                            while let Some(e) = stack.pop() {
                                world.entity_mut(e).insert(EditorPart);
                                if let Some(kids) = child_map.get(&e) {
                                    stack.extend(kids.iter().copied());
                                }
                            }
                        }
                        // Restore the saved build layout, or infer it for a
                        // save written before the field existed: horizontal for
                        // a winged craft (a plane), vertical otherwise.
                        let horizontal = bp
                            .layout
                            .map(|l| l == BuildLayout::Horizontal)
                            .unwrap_or_else(|| {
                                bp.parts
                                    .iter()
                                    .any(|p| matches!(p.params, PartParams::Wing { .. }))
                            });
                        world.resource_mut::<BuildOrientation>().horizontal = horizontal;

                        let mut st = world.resource_mut::<EditorState>();
                        st.ship_entity = Some(ship_entity);
                        st.ship_root = root;
                        st.selected = root;
                        st.ship_name = name;
                        st.status = format!("Loaded {path_disp}");
                    });
                }
                Err(e) => state.status = format!("Parse failed: {e}"),
            },
            Err(e) => state.status = format!("Read failed: {e}"),
        }
    }

    // ---- Delete file --------------------------------------------------
    if let Some(slug) = state.delete_file.take() {
        let path = ship_path_for_slug(&slug);
        match std::fs::remove_file(&path) {
            Ok(()) => {
                state.status = format!("Deleted {}", path.display());
                state.refresh_list = true;
            }
            Err(e) => state.status = format!("Delete failed: {e}"),
        }
    }

    // ---- Refresh list -------------------------------------------------
    if state.refresh_list {
        state.refresh_list = false;
        state.ship_list = list_ships();
    }

    // ---- Delete selected ---------------------------------------------
    // Deleting the root clears the whole canvas (despawns ship + all
    // parts). Deleting a non-root part despawns its subtree.
    if state.delete_selected {
        state.delete_selected = false;
        if let Some(sel) = state.selected {
            if Some(sel) == state.ship_root {
                if let Some(se) = state.ship_entity
                    && let Ok(ship) = ships.get(se)
                {
                    state.ship_name = ship.name.clone();
                }
                for e in all_parts.iter() {
                    commands.entity(e).despawn();
                }
                for e in all_ships.iter() {
                    commands.entity(e).despawn();
                }
                state.ship_root = None;
                state.ship_entity = None;
                state.selected = None;
                state.status = "Cleared canvas".into();
            } else {
                let mut child_map: HashMap<Entity, Vec<Entity>> = HashMap::new();
                for (e, att) in attachments.iter() {
                    child_map.entry(att.parent).or_default().push(e);
                }
                // Surface-mounted wings ride with their host: deleting a
                // fuselage must take its wings too.
                for (e, sm) in surface_mounts.iter() {
                    child_map.entry(sm.parent).or_default().push(e);
                }
                let mut to_remove: Vec<Entity> = Vec::new();
                // Deleting any symmetry-group member deletes the whole group
                // (KSP) — seed the walk with all counterparts of the selection.
                let mut stack = host_group_members(sel, &groups);
                while let Some(e) = stack.pop() {
                    to_remove.push(e);
                    if let Some(kids) = child_map.get(&e) {
                        stack.extend(kids.iter().copied());
                    }
                }
                for e in to_remove {
                    commands.entity(e).despawn();
                }
                state.selected = state.ship_root;
                state.status = "Deleted selection".into();
            }
        }
    }

    // ---- Set selection as root ---------------------------------------
    // Walk from the selection up through Attachment components to the
    // current root; reverse each link by inserting an Attachment on the
    // former parent pointing at the former child, with parent_node /
    // my_node swapped. Parts off the chain keep their attachments, so
    // branches follow their original subtree.
    if state.set_as_root {
        state.set_as_root = false;
        if let Some(sel) = state.selected
            && Some(sel) != state.ship_root
        {
            let att_map: HashMap<Entity, Attachment> =
                attachments.iter().map(|(e, a)| (e, a.clone())).collect();
            let mut chain: Vec<(Entity, Attachment)> = Vec::new();
            let mut current = sel;
            while let Some(att) = att_map.get(&current) {
                chain.push((current, att.clone()));
                current = att.parent;
            }
            commands.entity(sel).remove::<Attachment>();
            for (entity, att) in chain {
                commands.entity(att.parent).insert(Attachment {
                    parent: entity,
                    parent_node: att.my_node,
                    my_node: att.parent_node,
                });
            }
            if let Some(ship_entity) = state.ship_entity
                && let Ok(mut ship) = ships.get_mut(ship_entity)
            {
                ship.root = sel;
            }
            state.ship_root = Some(sel);
            state.status = "Re-rooted ship".into();
        }
    }

    // ---- Place pending part at a given (parent, node) -----------------
    if let Some((parent, node)) = state.place_at.take() {
        let Some(pending) = state.pending.take() else {
            return;
        };
        match ShipBlueprint::spawn_part(
            &mut commands,
            &catalog,
            &pending.catalog_id,
            pending.params,
            None,
        ) {
            Ok(child) => {
                commands.entity(child).insert((
                    Attachment {
                        parent,
                        parent_node: node,
                        my_node: "top".into(),
                    },
                    EditorPart,
                ));
                state.selected = Some(child);
                state.status = "Placed part".into();
            }
            Err(e) => state.status = format!("Spawn failed: {e}"),
        }
    }

    // ---- Mount pending footprint part on a surface -------------------
    // Body-skin mounts (wings) derive station/azimuth from a hull hit.
    // Wing-pylon mounts (jet nacelles) derive span/chord position from
    // the host wing hit.
    if let Some((parent, world_pos, kind)) = state.place_surface_at.take() {
        if let Some(pending) = state.pending.take() {
            let Some((station, angle, status)) = surface_mount_from_hit(
                kind,
                parent,
                world_pos,
                &part_transforms,
                &host_nodes,
                &surface_mounts,
                &wings,
                &orientation,
                modes.snap.enabled,
            ) else {
                state.status = "Surface placement failed".into();
                state.pending = Some(pending);
                return;
            };

            // Fill role-appropriate default control surfaces for a freshly
            // placed wing now that the mount azimuth is known (it decides
            // elevator vs rudder for a stabilizer). Leave a user-authored set
            // untouched.
            let mut pending = pending;
            if let PartParams::Wing {
                control_surfaces, ..
            } = &mut pending.params
                && control_surfaces.is_empty()
                && let Ok(CatalogEntry::Wing(spec)) = catalog.resolve(&pending.catalog_id)
            {
                *control_surfaces = default_control_surfaces(spec.role, angle);
            }

            // Landing gear is a self-contained gearbox — it draws its own legs,
            // so it is *always* a single mount regardless of the Mirror toggle
            // or a (hypothetically) symmetric host. Special-cased before the
            // wing/nacelle symmetry path below.
            let is_gear = matches!(pending.params, PartParams::Gear { .. });

            // KSP symmetry stamping. If the clicked host is itself a mirrored
            // pair (e.g. a nacelle onto a wing), stamp one part per host
            // member — nested symmetry. Otherwise, if mirror mode is on and
            // this is an off-centreline body-skin mount, stamp the reflected
            // pair on the same host. Else a single part.
            let host_members = host_group_members(parent, &groups);
            let stamps: Vec<(Entity, f32, f32)> = if is_gear {
                vec![(parent, station, angle)]
            } else if host_members.len() > 1 {
                host_members.iter().map(|&h| (h, station, angle)).collect()
            } else if modes.symmetry.mirror
                && kind == SurfaceMountKind::BodySkin
                && angle.sin().abs() > 0.3
            {
                vec![(parent, station, angle), (parent, station, -angle)]
            } else {
                vec![(parent, station, angle)]
            };

            let group_id = (stamps.len() > 1).then(|| next_sym_id.allocate());
            let mut first: Option<Entity> = None;
            for (i, (host, st, ang)) in stamps.iter().enumerate() {
                match ShipBlueprint::spawn_part(
                    &mut commands,
                    &catalog,
                    &pending.catalog_id,
                    pending.params.clone(),
                    None,
                ) {
                    Ok(child) => {
                        let mut ec = commands.entity(child);
                        ec.insert((
                            SurfaceMount {
                                parent: *host,
                                kind,
                                station: *st,
                                angle: *ang,
                            },
                            EditorPart,
                        ));
                        if let Some(gid) = group_id {
                            let role = if i == 0 {
                                SymmetryRole::Primary
                            } else {
                                SymmetryRole::Mirror
                            };
                            ec.insert(SymmetryGroup { id: gid, role });
                        }
                        first.get_or_insert(child);
                    }
                    Err(e) => state.status = format!("Spawn failed: {e}"),
                }
            }
            if let Some(sel) = first {
                state.selected = Some(sel);
                // `surface_mount_from_hit` labels every body-skin hit "Mounted
                // wing"; gear shares that path but isn't a wing.
                state.status = if is_gear {
                    "Mounted landing gear".into()
                } else {
                    status
                };
            }
        }
    }
    // ---- Auto-place pending as root on empty canvas ------------------
    if state.ship_root.is_none() && state.pending.is_some() {
        // Footprint parts need a host — they can't be roots. Keep them
        // pending and nudge the user to add structure first.
        let needs_surface_host = state.pending.as_ref().is_some_and(|p| {
            matches!(p.params, PartParams::Wing { .. } | PartParams::Gear { .. })
                || catalog.resolve(&p.catalog_id).is_ok_and(|entry| {
                    matches!(
                        entry,
                        CatalogEntry::Engine(e) if e.geometry == EngineGeometry::JetNacelle
                    )
                })
        });
        if needs_surface_host {
            state.status = "Add a hull first, then click a compatible surface".into();
            return;
        }
        let pending = state.pending.take().unwrap();
        match ShipBlueprint::spawn_part(
            &mut commands,
            &catalog,
            &pending.catalog_id,
            pending.params,
            None,
        ) {
            Ok(part) => {
                commands.entity(part).insert(EditorPart);
                let ship = commands
                    .spawn((
                        Ship {
                            name: state.ship_name.clone(),
                            root: part,
                        },
                        EditorPart,
                    ))
                    .id();
                state.ship_root = Some(part);
                state.ship_entity = Some(ship);
                state.selected = Some(part);
                state.status = "Placed root".into();
            }
            Err(e) => state.status = format!("Spawn failed: {e}"),
        }
    }
}
