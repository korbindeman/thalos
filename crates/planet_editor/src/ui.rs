#![allow(clippy::too_many_arguments)]

use super::*;

// Body switching — tear down old preview and spawn new one

#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_body_switch(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut gas_giant_materials: ResMut<Assets<GasGiantMaterial>>,
    mut ring_materials: ResMut<Assets<RingMaterial>>,
    mut planet: ResMut<EditedPlanet>,
    mut status: ResMut<TerrainGenStatus>,
    mut active_surface: ResMut<ActivePreviewSurface>,
    mut tile_viewer: ResMut<TileViewerState>,
    mut equirect_viewer: ResMut<EquirectViewerState>,
    billboard: Res<BillboardMesh>,
    preview_q: Query<Entity, With<PreviewPlanet>>,
) {
    if !planet.body_changed {
        return;
    }
    planet.body_changed = false;
    planet.terrain_dirty = false;
    active_surface.body_name.clear();
    active_surface.surface = None;
    active_surface.dynamic_state = None;
    tile_viewer.dirty = true;
    equirect_viewer.dirty = true;
    equirect_viewer.last_body_name.clear();

    for entity in &preview_q {
        commands.entity(entity).despawn();
    }

    spawn_preview(
        &mut commands,
        &mut meshes,
        &mut std_materials,
        &mut gas_giant_materials,
        &mut ring_materials,
        &billboard,
        &mut planet,
        &mut status,
    );
}

// Editor UI (egui)

pub(crate) fn render_body_tree_ui(
    ui: &mut egui::Ui,
    system: &SolarSystemDefinition,
    selected_body: Option<BodyId>,
) -> Option<BodyId> {
    let mut children_of: HashMap<BodyId, Vec<&BodyDefinition>> = HashMap::new();
    for body in &system.bodies {
        if let Some(parent) = body.parent {
            children_of.entry(parent).or_default().push(body);
        }
    }
    // Stable order: the file's listing order.
    for kids in children_of.values_mut() {
        kids.sort_by_key(|b| b.id);
    }

    let root = system.bodies.iter().find(|b| b.parent.is_none())?;
    let mut clicked: Option<BodyId> = None;

    // Major tree: star and its non-minor descendants.
    render_body_tree_row(ui, root, selected_body, &mut clicked, 0);
    if let Some(kids) = children_of.get(&root.id) {
        for child in kids.iter().filter(|b| !is_minor(b.kind)) {
            render_body_subtree(ui, child, &children_of, selected_body, &mut clicked, 1);
        }
    }

    // Minor bodies: collapsing group of dwarf planets / centaurs /
    // comets that orbit the star, with their own descendants nested.
    let minor: Vec<&BodyDefinition> = children_of
        .get(&root.id)
        .map(|kids| kids.iter().copied().filter(|b| is_minor(b.kind)).collect())
        .unwrap_or_default();
    if !minor.is_empty() {
        ui.collapsing("Minor bodies", |ui| {
            for body in minor {
                render_body_subtree(ui, body, &children_of, selected_body, &mut clicked, 0);
            }
        });
    }

    clicked
}

pub(crate) fn is_minor(kind: BodyKind) -> bool {
    matches!(
        kind,
        BodyKind::DwarfPlanet | BodyKind::Centaur | BodyKind::Comet
    )
}

pub(crate) fn render_body_subtree(
    ui: &mut egui::Ui,
    body: &BodyDefinition,
    children_of: &HashMap<BodyId, Vec<&BodyDefinition>>,
    selected_body: Option<BodyId>,
    clicked: &mut Option<BodyId>,
    depth: u32,
) {
    render_body_tree_row(ui, body, selected_body, clicked, depth);
    if let Some(kids) = children_of.get(&body.id) {
        for child in kids {
            render_body_subtree(ui, child, children_of, selected_body, clicked, depth + 1);
        }
    }
}

pub(crate) fn render_body_tree_row(
    ui: &mut egui::Ui,
    body: &BodyDefinition,
    selected_body: Option<BodyId>,
    clicked: &mut Option<BodyId>,
    depth: u32,
) {
    let is_selected = selected_body == Some(body.id);

    ui.horizontal(|ui| {
        ui.add_space(depth as f32 * 14.0);

        let [r, g, b] = body.color;
        let dot_color = egui::Color32::from_rgb(
            (r.clamp(0.0, 1.0) * 255.0) as u8,
            (g.clamp(0.0, 1.0) * 255.0) as u8,
            (b.clamp(0.0, 1.0) * 255.0) as u8,
        );
        let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
        ui.painter().circle_filled(rect.center(), 4.0, dot_color);
        ui.add_space(4.0);

        let label = ui.add(egui::Button::selectable(is_selected, &body.name).frame(false));
        if label.clicked() {
            *clicked = Some(body.id);
        }
    });
}

pub(crate) fn select_body(
    planet: &mut EditedPlanet,
    system: &SolarSystemDefinition,
    body_id: BodyId,
) {
    let body = &system.bodies[body_id];
    if planet.selected_body == body.name {
        return;
    }

    let resolved = build_params_for_body(system, body);
    planet.radius_m = resolved.radius_m;
    planet.gravity_m_s2 = resolved.gravity_m_s2;
    planet.axial_tilt_rad = resolved.axial_tilt_rad;
    planet.mode = resolved.mode;
    planet.rings = resolved.rings;
    planet.atmosphere = resolved.atmosphere;
    planet.heliocentric_distance_m = resolved.heliocentric_distance_m;
    planet.light_intensity = light_intensity_at(resolved.heliocentric_distance_m);
    planet.sun_orbital_elevation = resolved.sun_orbital_elevation;
    planet.selected_body = body.name.clone();
    planet.body_changed = true;
    planet.uniforms_dirty = true;
    planet.terrain_dirty = false;
    planet.last_edit = None;
    planet.requested_bake = None;
    planet.last_bake_mode = TerrainBakeMode::Preview;
    planet.selected_feature_id = None;
    planet.tool = ToolMode::default();
}

pub(crate) fn draw_physical_body_params(
    ui: &mut egui::Ui,
    planet: &mut EditedPlanet,
) -> (bool, bool) {
    let mut terrain_changed = false;
    let mut uniforms_changed = false;

    ui.heading("Physical body");
    ui.label(egui::RichText::new(&planet.selected_body).strong());

    egui::Grid::new("physical_body_params_grid")
        .num_columns(2)
        .spacing(egui::vec2(8.0, 4.0))
        .show(ui, |ui| {
            ui.label("Radius");
            let mut radius_km = planet.radius_m / 1000.0;
            if ui
                .add(
                    egui::DragValue::new(&mut radius_km)
                        .speed(1.0)
                        .range(1.0..=1_000_000.0)
                        .suffix(" km"),
                )
                .changed()
            {
                planet.radius_m = radius_km * 1000.0;
                terrain_changed = matches!(&planet.mode, BodyMode::Terrain { .. });
                uniforms_changed = true;
            }
            ui.end_row();

            ui.label("Surface gravity");
            if ui
                .add(
                    egui::DragValue::new(&mut planet.gravity_m_s2)
                        .speed(0.01)
                        .range(0.0..=500.0)
                        .suffix(" m/s²"),
                )
                .changed()
            {
                terrain_changed = matches!(&planet.mode, BodyMode::Terrain { .. });
            }
            ui.end_row();

            ui.label("Axial tilt");
            let mut tilt_deg = planet.axial_tilt_rad.to_degrees();
            if ui
                .add(
                    egui::DragValue::new(&mut tilt_deg)
                        .speed(0.25)
                        .range(-180.0..=180.0)
                        .suffix("°"),
                )
                .changed()
            {
                planet.axial_tilt_rad = tilt_deg.to_radians();
                terrain_changed = matches!(&planet.mode, BodyMode::Terrain { .. });
                uniforms_changed = true;
            }
            ui.end_row();

            ui.label("Heliocentric distance");
            let mut heliocentric_au = planet.heliocentric_distance_m / AU_M;
            if ui
                .add(
                    egui::DragValue::new(&mut heliocentric_au)
                        .speed(0.001)
                        .range(0.001..=10_000.0)
                        .suffix(" AU"),
                )
                .changed()
            {
                planet.heliocentric_distance_m = heliocentric_au * AU_M;
                planet.light_intensity = light_intensity_at(planet.heliocentric_distance_m);
                uniforms_changed = true;
            }
            ui.end_row();

            ui.label("Light intensity");
            if ui
                .add(
                    egui::DragValue::new(&mut planet.light_intensity)
                        .speed(0.05)
                        .range(0.0..=10_000.0),
                )
                .changed()
            {
                uniforms_changed = true;
            }
            ui.end_row();
        });

    (terrain_changed, uniforms_changed)
}

pub(crate) fn draw_planet_definition_panel(
    ui: &mut egui::Ui,
    planet: &mut EditedPlanet,
) -> (bool, bool) {
    let mut rebuild_requested = false;
    let mut uniforms_changed = false;

    match &mut planet.mode {
        BodyMode::Terrain { terrain, .. } => {
            rebuild_requested |= draw_terrain_definition(ui, terrain);
        }
        BodyMode::GasGiant { .. } => {
            ui.heading("Planet type");
            ui.label("Gas / ice giant");
            ui.small("Gas giant cloud-deck authoring will move here when the gas schema gets the same editor treatment as terrestrial terrain.");
        }
        BodyMode::Star => {
            ui.heading("Body type");
            ui.label("Star");
        }
    }

    ui.separator();
    uniforms_changed |= draw_terrestrial_atmosphere_definition(ui, planet);

    ui.separator();
    ui.small("Live-only for now; switching bodies reloads authored values from assets. A later pass should write these sections back to the per-body RON files.");

    (rebuild_requested, uniforms_changed)
}

pub(crate) fn draw_terrain_definition(ui: &mut egui::Ui, terrain: &mut TerrainConfig) -> bool {
    let mut edited = false;

    ui.heading("Generation identity");
    ui.label(format!("Route: {}", terrain.route_label()));

    match terrain {
        TerrainConfig::Feature(config) => {
            ui.horizontal(|ui| {
                ui.label("Archetype");
                egui::ComboBox::from_id_salt("body_archetype_combo")
                    .selected_text(format!("{:?}", config.archetype))
                    .show_ui(ui, |ui| {
                        for archetype in [
                            BodyArchetype::AirlessImpactMoon,
                            BodyArchetype::ColdDesertFormerlyWet,
                            BodyArchetype::AgingOceanicHomeworld,
                            BodyArchetype::GenericTerrestrial,
                            BodyArchetype::OceanicTerrestrial,
                        ] {
                            edited |= ui
                                .selectable_value(
                                    &mut config.archetype,
                                    archetype,
                                    format!("{archetype:?}"),
                                )
                                .changed();
                        }
                    });
            });

            ui.horizontal(|ui| {
                ui.label("Composition");
                egui::ComboBox::from_id_salt("composition_class_combo")
                    .selected_text(format!("{:?}", config.composition))
                    .show_ui(ui, |ui| {
                        for composition in [
                            CompositionClass::SilicateDominated,
                            CompositionClass::BasalticSilicate,
                            CompositionClass::IronRichSilicate,
                            CompositionClass::IcySilicate,
                        ] {
                            edited |= ui
                                .selectable_value(
                                    &mut config.composition,
                                    composition,
                                    format!("{composition:?}"),
                                )
                                .changed();
                        }
                    });
            });

            edited |= fires(
                &ui.add(egui::Slider::new(&mut config.body_age_gyr, 0.0..=10.0).text("Age (Gyr)")),
            );
            edited |= fires(
                &ui.add(
                    egui::Slider::new(&mut config.environment.stellar_flux_earth, 0.0..=4.0)
                        .text("Stellar flux × Earth"),
                ),
            );

            ui.collapsing("Environment prior", |ui| {
                edited |= draw_atmosphere_spec_control(ui, &mut config.environment.atmosphere);
                edited |= draw_hydrosphere_spec_control(ui, &mut config.environment.hydrosphere);

                ui.horizontal(|ui| {
                    ui.label("Ice inventory");
                    egui::ComboBox::from_id_salt("ice_inventory_combo")
                        .selected_text(format!("{:?}", config.environment.ice_inventory))
                        .show_ui(ui, |ui| {
                            for ice in [
                                IceInventory::None,
                                IceInventory::Trace,
                                IceInventory::Moderate,
                                IceInventory::High,
                            ] {
                                edited |= ui
                                    .selectable_value(
                                        &mut config.environment.ice_inventory,
                                        ice,
                                        format!("{ice:?}"),
                                    )
                                    .changed();
                            }
                        });
                });
            });

            ui.collapsing("Generation intent", |ui| {
                for intent in [
                    TerrainIntent::ReadAsMoon,
                    TerrainIntent::DistinctNearSideFace,
                    TerrainIntent::DifferentFarSide,
                    TerrainIntent::FirstLandingWorld,
                    TerrainIntent::ReadAsFirstInterplanetarySurfaceWorld,
                    TerrainIntent::ForgivingLandingTerrain,
                    TerrainIntent::VisibleAncientWaterStory,
                    TerrainIntent::RustDustAndEvaporites,
                    TerrainIntent::HomeworldIdentity,
                ] {
                    let mut enabled = config.intent.contains(&intent);
                    if ui.checkbox(&mut enabled, format!("{intent:?}")).changed() {
                        if enabled {
                            config.intent.push(intent);
                        } else {
                            config.intent.retain(|i| *i != intent);
                        }
                        edited = true;
                    }
                }
            });
        }
        TerrainConfig::Ocean(ocean) => {
            ui.label("Ocean placeholder");
            ui.small("A flat-water route: useful for bodies whose surface is not through the feature compiler yet.");
            edited |= fires(&ui.add(egui::Slider::new(&mut ocean.seed, 0..=9999).text("Seed")));
            edited |=
                fires(&ui.add(
                    egui::Slider::new(&mut ocean.sea_level_m, 0.0..=10.0).text("Sea level (m)"),
                ));
            edited |= fires(&ui.add(
                egui::Slider::new(&mut ocean.water_roughness, 0.0..=0.3).text("Water roughness"),
            ));
        }
        TerrainConfig::None => {
            ui.label("No authored terrain");
            ui.small(
                "This body currently renders through its fallback color or a non-terrain material.",
            );
        }
    }

    if edited {
        ui.small("Definition edited. Preview is not rebuilt automatically so broad pipeline changes stay responsive.");
    }
    ui.add(egui::Button::new("Rebuild preview from definition"))
        .clicked()
}

pub(crate) fn draw_atmosphere_spec_control(
    ui: &mut egui::Ui,
    atmosphere: &mut AtmosphereSpec,
) -> bool {
    let mut changed = false;
    let mut kind = match atmosphere {
        AtmosphereSpec::None => 0,
        AtmosphereSpec::ThinCo2 { .. } => 1,
        AtmosphereSpec::Breathable { .. } => 2,
        AtmosphereSpec::Other { .. } => 3,
    };

    ui.horizontal(|ui| {
        ui.label("Atmosphere prior");
        egui::ComboBox::from_id_salt("atmosphere_spec_combo")
            .selected_text(match kind {
                0 => "None",
                1 => "Thin CO₂",
                2 => "Breathable",
                _ => "Other",
            })
            .show_ui(ui, |ui| {
                changed |= ui.selectable_value(&mut kind, 0, "None").changed();
                changed |= ui.selectable_value(&mut kind, 1, "Thin CO₂").changed();
                changed |= ui.selectable_value(&mut kind, 2, "Breathable").changed();
                changed |= ui.selectable_value(&mut kind, 3, "Other").changed();
            });
    });

    if changed {
        let pressure_bar = match atmosphere {
            AtmosphereSpec::None => 1.0,
            AtmosphereSpec::ThinCo2 { pressure_bar }
            | AtmosphereSpec::Breathable { pressure_bar }
            | AtmosphereSpec::Other { pressure_bar } => *pressure_bar,
        };
        *atmosphere = match kind {
            0 => AtmosphereSpec::None,
            1 => AtmosphereSpec::ThinCo2 { pressure_bar },
            2 => AtmosphereSpec::Breathable { pressure_bar },
            _ => AtmosphereSpec::Other { pressure_bar },
        };
    }

    match atmosphere {
        AtmosphereSpec::ThinCo2 { pressure_bar }
        | AtmosphereSpec::Breathable { pressure_bar }
        | AtmosphereSpec::Other { pressure_bar } => {
            changed |=
                fires(&ui.add(egui::Slider::new(pressure_bar, 0.0..=100.0).text("Pressure (bar)")));
        }
        AtmosphereSpec::None => {}
    }

    changed
}

pub(crate) fn draw_hydrosphere_spec_control(
    ui: &mut egui::Ui,
    hydrosphere: &mut HydrosphereSpec,
) -> bool {
    let mut changed = false;
    let mut kind = match hydrosphere {
        HydrosphereSpec::None => 0,
        HydrosphereSpec::Trace => 1,
        HydrosphereSpec::AncientLost => 2,
        HydrosphereSpec::OceanFraction(_) => 3,
    };

    ui.horizontal(|ui| {
        ui.label("Hydrosphere prior");
        egui::ComboBox::from_id_salt("hydrosphere_spec_combo")
            .selected_text(match kind {
                0 => "None",
                1 => "Trace",
                2 => "Ancient lost",
                _ => "Ocean fraction",
            })
            .show_ui(ui, |ui| {
                changed |= ui.selectable_value(&mut kind, 0, "None").changed();
                changed |= ui.selectable_value(&mut kind, 1, "Trace").changed();
                changed |= ui.selectable_value(&mut kind, 2, "Ancient lost").changed();
                changed |= ui
                    .selectable_value(&mut kind, 3, "Ocean fraction")
                    .changed();
            });
    });

    if changed {
        let fraction = match hydrosphere {
            HydrosphereSpec::OceanFraction(f) => *f,
            _ => 0.5,
        };
        *hydrosphere = match kind {
            0 => HydrosphereSpec::None,
            1 => HydrosphereSpec::Trace,
            2 => HydrosphereSpec::AncientLost,
            _ => HydrosphereSpec::OceanFraction(fraction),
        };
    }

    if let HydrosphereSpec::OceanFraction(fraction) = hydrosphere {
        changed |= fires(&ui.add(egui::Slider::new(fraction, 0.0..=1.0).text("Ocean fraction")));
    }

    changed
}

pub(crate) fn draw_terrestrial_atmosphere_definition(
    ui: &mut egui::Ui,
    planet: &mut EditedPlanet,
) -> bool {
    let mut changed = false;
    ui.heading("Atmosphere");

    let meters_per_ru = (planet.radius_m as f32 / RENDER_RADIUS).max(1.0);
    let has_atmosphere = planet.atmosphere.is_some();
    let mut enabled = has_atmosphere;
    if ui
        .checkbox(&mut enabled, "Terrestrial atmosphere")
        .changed()
    {
        if enabled && planet.atmosphere.is_none() {
            planet.atmosphere = Some(EditorAtmosphere {
                block: default_editor_atmosphere_block(meters_per_ru),
            });
            planet.atmosphere_enabled = true;
        } else if !enabled {
            planet.atmosphere = None;
            planet.atmosphere_enabled = false;
        }
        changed = true;
    }

    let Some(atmosphere) = &mut planet.atmosphere else {
        ui.small("No thin atmosphere shell. Gas giants use a separate cloud-deck schema.");
        return changed;
    };

    changed |= ui
        .checkbox(&mut planet.atmosphere_enabled, "Render atmosphere")
        .changed();

    let block = &mut atmosphere.block;
    let mut karman_line_m = block.atmos_geom.x * meters_per_ru;
    if ui
        .add(
            egui::DragValue::new(&mut karman_line_m)
                .speed(100.0)
                .range(0.0..=5_000_000.0)
                .suffix(" m")
                .prefix("Kármán line: "),
        )
        .changed()
    {
        block.atmos_geom.x = karman_line_m.max(0.0) / meters_per_ru;
        changed = true;
    }

    ui.collapsing("Scattering", |ui| {
        let mut strength = block.atmos_geom.z;
        if ui
            .add(egui::Slider::new(&mut strength, 0.0..=4.0).text("Strength"))
            .changed()
        {
            block.atmos_geom.z = strength;
            changed = true;
        }

        let mut rayleigh_h_m = block.rayleigh_beta_h.w * meters_per_ru;
        if ui
            .add(
                egui::DragValue::new(&mut rayleigh_h_m)
                    .speed(100.0)
                    .range(1.0..=100_000.0)
                    .suffix(" m")
                    .prefix("Rayleigh H: "),
            )
            .changed()
        {
            let old_h_ru = block.rayleigh_beta_h.w.max(1.0e-6);
            let tau = block.rayleigh_beta_h.truncate() * old_h_ru;
            let new_h_ru = rayleigh_h_m.max(1.0) / meters_per_ru;
            block.rayleigh_beta_h =
                tau.extend(new_h_ru) / Vec4::new(new_h_ru, new_h_ru, new_h_ru, 1.0);
            changed = true;
        }

        let mut rayleigh_tau = block.rayleigh_beta_h.truncate() * block.rayleigh_beta_h.w;
        ui.label("Rayleigh vertical optical depth");
        changed |= ui
            .add(egui::Slider::new(&mut rayleigh_tau.x, 0.0..=1.0).text("R"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut rayleigh_tau.y, 0.0..=1.0).text("G"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut rayleigh_tau.z, 0.0..=1.0).text("B"))
            .changed();
        let h_ru = block.rayleigh_beta_h.w.max(1.0e-6);
        block.rayleigh_beta_h.x = rayleigh_tau.x / h_ru;
        block.rayleigh_beta_h.y = rayleigh_tau.y / h_ru;
        block.rayleigh_beta_h.z = rayleigh_tau.z / h_ru;

        let mut mie_h_m = block.atmos_geom.y * meters_per_ru;
        if ui
            .add(
                egui::DragValue::new(&mut mie_h_m)
                    .speed(50.0)
                    .range(1.0..=100_000.0)
                    .suffix(" m")
                    .prefix("Mie H: "),
            )
            .changed()
        {
            let old_h_ru = block.atmos_geom.y.max(1.0e-6);
            let tau = block.mie_beta_g.x * old_h_ru;
            let new_h_ru = mie_h_m.max(1.0) / meters_per_ru;
            block.atmos_geom.y = new_h_ru;
            block.mie_beta_g.x = tau / new_h_ru;
            block.mie_beta_g.y = block.mie_beta_g.x;
            block.mie_beta_g.z = block.mie_beta_g.x;
            changed = true;
        }

        let mut mie_tau = block.mie_beta_g.x * block.atmos_geom.y;
        if ui
            .add(egui::Slider::new(&mut mie_tau, 0.0..=1.0).text("Mie optical depth"))
            .changed()
        {
            let beta = mie_tau / block.atmos_geom.y.max(1.0e-6);
            block.mie_beta_g.x = beta;
            block.mie_beta_g.y = beta;
            block.mie_beta_g.z = beta;
            changed = true;
        }

        changed |= ui
            .add(egui::Slider::new(&mut block.mie_beta_g.w, -0.99..=0.99).text("Mie asymmetry"))
            .changed();
    });

    ui.collapsing("Cloud layer", |ui| {
        changed |= ui
            .add(egui::Slider::new(&mut block.cloud_albedo_coverage.w, 0.0..=1.0).text("Coverage"))
            .changed();
        let mut base_altitude_m = block.cloud_shape.x * meters_per_ru;
        if ui
            .add(
                egui::DragValue::new(&mut base_altitude_m)
                    .speed(50.0)
                    .range(0.0..=100_000.0)
                    .suffix(" m")
                    .prefix("Base altitude: "),
            )
            .changed()
        {
            block.cloud_shape.x = base_altitude_m / meters_per_ru;
            changed = true;
        }
        let mut thickness_m = block.cloud_shape.y * meters_per_ru;
        if ui
            .add(
                egui::DragValue::new(&mut thickness_m)
                    .speed(50.0)
                    .range(0.0..=100_000.0)
                    .suffix(" m")
                    .prefix("Thickness: "),
            )
            .changed()
        {
            block.cloud_shape.y = thickness_m / meters_per_ru;
            changed = true;
        }
        changed |= ui
            .add(egui::Slider::new(&mut block.cloud_shape.z, 0.0..=10.0).text("Density"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut block.cloud_shape.w, 0.0..=1.0)
                    .text("Differential rotation"),
            )
            .changed();
        changed |= ui
            .add(
                egui::DragValue::new(&mut block.cloud_dynamics.x)
                    .speed(1.0e-6)
                    .prefix("Scroll rad/s: "),
            )
            .changed();
    });

    changed
}

pub(crate) fn default_editor_atmosphere_block(meters_per_ru: f32) -> AtmosphereBlock {
    let h_r_m = 8000.0;
    let h_m_m = 1200.0;
    let h_r_ru = h_r_m / meters_per_ru;
    let h_m_ru = h_m_m / meters_per_ru;
    AtmosphereBlock {
        rayleigh_beta_h: Vec4::new(0.046 / h_r_ru, 0.108 / h_r_ru, 0.264 / h_r_ru, h_r_ru),
        mie_beta_g: Vec4::new(0.021 / h_m_ru, 0.021 / h_m_ru, 0.021 / h_m_ru, 0.76),
        atmos_geom: Vec4::new(80_000.0 / meters_per_ru, h_m_ru, 1.0, 0.0),
        cloud_albedo_coverage: Vec4::new(0.92, 0.94, 0.97, 0.0),
        cloud_shape: Vec4::new(1500.0 / meters_per_ru, 5000.0 / meters_per_ru, 1.0, 0.35),
        cloud_dynamics: Vec4::new(4.7e-6, 0.0, 0.0, 0.0),
        ..default()
    }
}

pub(crate) fn fires(r: &egui::Response) -> bool {
    r.drag_stopped() || (r.changed() && !r.dragged())
}

pub(crate) fn draw_airless_projection_controls(
    ui: &mut egui::Ui,
    projection: &mut AirlessImpactProjectionConfig,
) -> bool {
    let mut changed = false;
    ui.collapsing("Projection", |ui| {
        changed |= fires(&ui.add(
            egui::Slider::new(&mut projection.base_crater_count, 0..=500_000).text("Base craters"),
        ));
        changed |= fires(
            &ui.add(
                egui::Slider::new(&mut projection.min_crater_radius_m, 100.0..=5_000.0)
                    .text("Min crater m"),
            ),
        );
        changed |= fires(
            &ui.add(
                egui::Slider::new(&mut projection.max_crater_radius_m, 10_000.0..=180_000.0)
                    .text("Max crater m"),
            ),
        );
        changed |= fires(
            &ui.add(
                egui::Slider::new(&mut projection.cubemap_bake_threshold_m, 250.0..=5_000.0)
                    .text("Bake threshold m"),
            ),
        );
        changed |= fires(&ui.add(
            egui::Slider::new(&mut projection.mare_fill_fraction, 0.0..=1.0).text("Mare fill"),
        ));
        changed |= fires(
            &ui.add(
                egui::Slider::new(
                    &mut projection.mare_boundary_noise_amplitude_m,
                    0.0..=2_500.0,
                )
                .text("Mare edge noise m"),
            ),
        );
        changed |= fires(
            &ui.add(
                egui::Slider::new(&mut projection.regolith_bake_d_min_m, 100.0..=2_000.0)
                    .text("Regolith bake min m"),
            ),
        );
    });
    changed
}

pub(crate) fn draw_cold_desert_projection_controls(
    ui: &mut egui::Ui,
    projection: &mut ColdDesertProjectionConfig,
) -> bool {
    let mut changed = false;
    ui.collapsing("Projection", |ui| {
        changed |= fires(&ui.add(
            egui::Slider::new(&mut projection.relief_scale_m, 0.25..=2.0).text("Relief scale"),
        ));
        changed |= fires(
            &ui.add(
                egui::Slider::new(&mut projection.volcanic_dark_strength, 0.0..=2.0)
                    .text("Dark regions"),
            ),
        );
        changed |= fires(&ui.add(
            egui::Slider::new(&mut projection.pale_basin_strength, 0.0..=2.0).text("Pale basins"),
        ));
        changed |=
            fires(&ui.add(
                egui::Slider::new(&mut projection.channel_strength, 0.0..=2.0).text("Channels"),
            ));
        changed |= fires(
            &ui.add(egui::Slider::new(&mut projection.dune_strength, 0.0..=2.0).text("Dunes")),
        );
        changed |= fires(&ui.add(
            egui::Slider::new(&mut projection.base_crater_count, 0..=100_000).text("Base craters"),
        ));
        changed |= fires(
            &ui.add(
                egui::Slider::new(&mut projection.max_crater_radius_m, 5_000.0..=90_000.0)
                    .text("Max crater m"),
            ),
        );
    });
    changed
}

pub(crate) fn draw_spec_controls(ui: &mut egui::Ui, config: &mut FeatureTerrainConfig) -> bool {
    let mut changed = false;
    ui.collapsing("Spec", |ui| {
        let prev_arch = config.archetype;
        egui::ComboBox::from_label("Archetype")
            .selected_text(format!("{:?}", config.archetype))
            .show_ui(ui, |ui| {
                for arch in [
                    BodyArchetype::AirlessImpactMoon,
                    BodyArchetype::ColdDesertFormerlyWet,
                    BodyArchetype::AgingOceanicHomeworld,
                    BodyArchetype::GenericTerrestrial,
                ] {
                    ui.selectable_value(&mut config.archetype, arch, format!("{arch:?}"));
                }
            });
        if config.archetype != prev_arch {
            changed = true;
        }

        let prev_comp = config.composition;
        egui::ComboBox::from_label("Composition")
            .selected_text(format!("{:?}", config.composition))
            .show_ui(ui, |ui| {
                for comp in [
                    CompositionClass::SilicateDominated,
                    CompositionClass::BasalticSilicate,
                    CompositionClass::IronRichSilicate,
                    CompositionClass::IcySilicate,
                ] {
                    ui.selectable_value(&mut config.composition, comp, format!("{comp:?}"));
                }
            });
        if config.composition != prev_comp {
            changed = true;
        }

        changed |= fires(
            &ui.add(egui::Slider::new(&mut config.body_age_gyr, 0.5..=12.0).text("Age (Gyr)")),
        );

        ui.collapsing("Environment", |ui| {
            changed |= fires(
                &ui.add(
                    egui::Slider::new(&mut config.environment.stellar_flux_earth, 0.0..=3.0)
                        .text("Stellar flux (Earth)"),
                ),
            );
            changed |= draw_atmosphere(ui, &mut config.environment.atmosphere);
            changed |= draw_hydrosphere(ui, &mut config.environment.hydrosphere);

            let prev_ice = config.environment.ice_inventory;
            egui::ComboBox::from_label("Ice inventory")
                .selected_text(format!("{:?}", config.environment.ice_inventory))
                .show_ui(ui, |ui| {
                    for ice in [
                        IceInventory::None,
                        IceInventory::Trace,
                        IceInventory::Moderate,
                        IceInventory::High,
                    ] {
                        ui.selectable_value(
                            &mut config.environment.ice_inventory,
                            ice,
                            format!("{ice:?}"),
                        );
                    }
                });
            if config.environment.ice_inventory != prev_ice {
                changed = true;
            }
        });

        ui.collapsing("Intent", |ui| {
            for intent in [
                TerrainIntent::ReadAsMoon,
                TerrainIntent::DistinctNearSideFace,
                TerrainIntent::DifferentFarSide,
                TerrainIntent::FirstLandingWorld,
                TerrainIntent::ReadAsFirstInterplanetarySurfaceWorld,
                TerrainIntent::ForgivingLandingTerrain,
                TerrainIntent::VisibleAncientWaterStory,
                TerrainIntent::RustDustAndEvaporites,
                TerrainIntent::HomeworldIdentity,
            ] {
                let mut on = config.intent.contains(&intent);
                if ui.checkbox(&mut on, format!("{intent:?}")).changed() {
                    if on {
                        config.intent.push(intent);
                    } else {
                        config.intent.retain(|i| *i != intent);
                    }
                    changed = true;
                }
            }
        });
    });
    changed
}

/// Atmosphere variant selector + payload editor. Switching variants preserves
/// the current pressure_bar across variants that carry one.
pub(crate) fn draw_atmosphere(ui: &mut egui::Ui, atmos: &mut AtmosphereSpec) -> bool {
    let mut changed = false;
    let cur_disc = std::mem::discriminant(atmos);
    let pressure = match *atmos {
        AtmosphereSpec::None => 0.01,
        AtmosphereSpec::ThinCo2 { pressure_bar }
        | AtmosphereSpec::Breathable { pressure_bar }
        | AtmosphereSpec::Other { pressure_bar } => pressure_bar,
    };
    egui::ComboBox::from_label("Atmosphere")
        .selected_text(atmosphere_label(atmos))
        .show_ui(ui, |ui| {
            for candidate in [
                AtmosphereSpec::None,
                AtmosphereSpec::ThinCo2 {
                    pressure_bar: pressure,
                },
                AtmosphereSpec::Breathable {
                    pressure_bar: pressure,
                },
                AtmosphereSpec::Other {
                    pressure_bar: pressure,
                },
            ] {
                let selected = std::mem::discriminant(&candidate) == cur_disc;
                if ui
                    .selectable_label(selected, atmosphere_label(&candidate))
                    .clicked()
                    && !selected
                {
                    *atmos = candidate;
                    changed = true;
                }
            }
        });
    if let AtmosphereSpec::ThinCo2 { pressure_bar }
    | AtmosphereSpec::Breathable { pressure_bar }
    | AtmosphereSpec::Other { pressure_bar } = atmos
    {
        changed |= fires(
            &ui.add(
                egui::Slider::new(pressure_bar, 0.001..=10.0)
                    .logarithmic(true)
                    .text("Pressure (bar)"),
            ),
        );
    }
    changed
}

pub(crate) fn atmosphere_label(a: &AtmosphereSpec) -> &'static str {
    match a {
        AtmosphereSpec::None => "None",
        AtmosphereSpec::ThinCo2 { .. } => "ThinCo2",
        AtmosphereSpec::Breathable { .. } => "Breathable",
        AtmosphereSpec::Other { .. } => "Other",
    }
}

pub(crate) fn draw_hydrosphere(ui: &mut egui::Ui, hydro: &mut HydrosphereSpec) -> bool {
    let mut changed = false;
    let cur_disc = std::mem::discriminant(hydro);
    let fraction = match *hydro {
        HydrosphereSpec::OceanFraction(f) => f,
        _ => 0.7,
    };
    egui::ComboBox::from_label("Hydrosphere")
        .selected_text(hydrosphere_label(hydro))
        .show_ui(ui, |ui| {
            for candidate in [
                HydrosphereSpec::None,
                HydrosphereSpec::Trace,
                HydrosphereSpec::AncientLost,
                HydrosphereSpec::OceanFraction(fraction),
            ] {
                let selected = std::mem::discriminant(&candidate) == cur_disc;
                if ui
                    .selectable_label(selected, hydrosphere_label(&candidate))
                    .clicked()
                    && !selected
                {
                    *hydro = candidate;
                    changed = true;
                }
            }
        });
    if let HydrosphereSpec::OceanFraction(f) = hydro {
        changed |= fires(&ui.add(egui::Slider::new(f, 0.0..=1.0).text("Ocean fraction")));
    }
    changed
}

pub(crate) fn hydrosphere_label(h: &HydrosphereSpec) -> &'static str {
    match h {
        HydrosphereSpec::None => "None",
        HydrosphereSpec::Trace => "Trace",
        HydrosphereSpec::AncientLost => "AncientLost",
        HydrosphereSpec::OceanFraction(_) => "OceanFraction",
    }
}

pub(crate) fn draw_projection_controls(
    ui: &mut egui::Ui,
    projection: &mut FeatureProjectionConfig,
) -> bool {
    match projection {
        FeatureProjectionConfig::Auto => {
            ui.label("Projection: Auto");
            false
        }
        FeatureProjectionConfig::AirlessImpact(config) => {
            draw_airless_projection_controls(ui, config)
        }
        FeatureProjectionConfig::ColdDesert(config) => {
            draw_cold_desert_projection_controls(ui, config)
        }
    }
}

/// Tectonics panel: live stats from the most recent bake and a config
/// sub-section. The layer config is separated from overlay controls so its
/// edits (which *do* trigger rebakes) can't be confused with visualization
/// toggles.
///
/// `archetype_requires_tectonics` locks the layer-on/off checkbox: bodies
/// whose archetype requires a tectonic graph (currently
/// `AgingOceanicHomeworld`) cannot be toggled to None — disabling would put
/// the bake into a guaranteed-fail state.
///
/// Returns true if any edit should trigger a rebake.
pub(crate) fn draw_tectonics_panel(
    ui: &mut egui::Ui,
    tectonics: &mut Option<TectonicConfig>,
    preview: Option<&TectonicSystem>,
    archetype_requires_tectonics: bool,
) -> bool {
    let mut changed = false;
    ui.heading("Tectonics");

    // ── Layer presence ──
    // For required archetypes the toggle is shown disabled with a label so
    // the constraint is visible; for optional archetypes it lets you opt in
    // or out of the layer entirely.
    if archetype_requires_tectonics {
        ui.label("Tectonic layer: required by archetype");
        if tectonics.is_none() {
            // Defensive: an archetype that requires tectonics shouldn't be
            // sitting at None. Seed a default so the bake doesn't fail on
            // the next rebake. Mark as changed so the rebake fires.
            *tectonics = Some(default_tectonic_config());
            changed = true;
        }
    } else {
        let mut enabled = tectonics.is_some();
        if ui
            .checkbox(&mut enabled, "Tectonic layer")
            .on_hover_text("Spherical-Voronoi plate graph; drives the plate-color overlay and (for AgingOceanicHomeworld) terrain shape.")
            .changed()
        {
            if enabled {
                *tectonics = Some(default_tectonic_config());
            } else {
                *tectonics = None;
            }
            changed = true;
        }
    }

    let Some(config) = tectonics.as_mut() else {
        return changed;
    };

    // ── Stats from the most recent bake ──
    // `preview` is None during the brief gap between dispatch and finalize;
    // show "–" placeholders so the layout doesn't jump.
    if let Some(sys) = preview {
        let n_continental = sys
            .plates
            .iter()
            .filter(|p| p.kind == PlateKind::Continental)
            .count();
        let n_oceanic = sys.plates.len() - n_continental;
        let mut convergent = 0usize;
        let mut divergent = 0usize;
        let mut transform = 0usize;
        for b in &sys.boundaries {
            match b.kind {
                BoundaryKind::Convergent => convergent += 1,
                BoundaryKind::Divergent => divergent += 1,
                BoundaryKind::Transform => transform += 1,
            }
        }
        ui.label(format!(
            "Plates: {} (continental {}, oceanic {})",
            sys.plates.len(),
            n_continental,
            n_oceanic,
        ));
        ui.label(format!(
            "Boundaries: {} (convergent {}, divergent {}, transform {})",
            sys.boundaries.len(),
            convergent,
            divergent,
            transform,
        ));
        ui.label(format!("Mesh cells: {}", sys.mesh.cells.len()));
        ui.label(format!("Activity: {}", activity_label(sys.config.activity)));
    } else {
        ui.label("Plates: –");
        ui.label("Boundaries: –");
        ui.label("Mesh cells: –");
        ui.label("Activity: –");
    }

    ui.separator();

    // ── Layer config ──
    // Slider ranges are conservative enough that no value produces a
    // degenerate tectonic graph. plate_count should stay below mesh_cells;
    // we don't enforce that on slider clamp because it's unusual and a
    // deliberate footgun there is fine.
    ui.label("Configuration:");
    ui.horizontal(|ui| {
        changed |= fires(&ui.add(egui::Slider::new(&mut config.seed, 0..=99_999).text("Seed")));
        if ui.button("Reroll").clicked() {
            config.seed = sub_seed(config.seed, "planet_editor:tectonic_seed");
            changed = true;
        }
    });
    changed |= fires(&ui.add(egui::Slider::new(&mut config.plate_count, 1..=64).text("Plates")));
    changed |=
        fires(&ui.add(egui::Slider::new(&mut config.mesh_cells, 256..=8192).text("Mesh cells")));
    changed |= fires(&ui.add(
        egui::Slider::new(&mut config.continental_fraction, 0.0..=1.0).text("Continental fraction"),
    ));

    let prev_activity = config.activity;
    egui::ComboBox::from_label("Activity")
        .selected_text(activity_label(config.activity))
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut config.activity, TectonicActivity::Active, "Active");
            ui.selectable_value(
                &mut config.activity,
                TectonicActivity::StagnantLid,
                "Stagnant lid",
            );
            // Frozen carries an age; pin to a placeholder when toggling
            // from the dropdown. The age field gets its own slider when
            // Frozen is selected.
            ui.selectable_value(
                &mut config.activity,
                TectonicActivity::Frozen { age_my: 1000.0 },
                "Frozen",
            );
        });
    if config.activity != prev_activity {
        changed = true;
    }
    if let TectonicActivity::Frozen { age_my } = &mut config.activity {
        changed |= fires(&ui.add(egui::Slider::new(age_my, 0.0..=4500.0).text("Frozen age (Myr)")));
    }

    changed
}

/// Default tectonic config seeded when the user opts in via the panel.
/// Earth-like ratios, StagnantLid (no live motion) so it's safe on bodies
/// regardless of activity expectations.
pub(crate) fn default_tectonic_config() -> TectonicConfig {
    TectonicConfig {
        plate_count: 12,
        mesh_cells: 2000,
        activity: TectonicActivity::StagnantLid,
        continental_fraction: 0.30,
        seed: 1,
        seed_dirs: None,
        continental_clustering: 0.0,
        equatorial_bias: 0.0,
        primary_size_multiplier: 1.0,
    }
}

pub(crate) fn reroll_authored_seed(
    root_seed: u64,
    id: &FeatureId,
    seed: &mut Option<FeatureSeed>,
    stream: FeatureSeedStream,
) {
    let current = seed.unwrap_or_else(|| FeatureSeed::derive(root_seed, id));
    *seed = Some(current.rerolled(stream, "planet_editor"));
}

/// Draw the manifest as a flat indented selectable list. Returns the feature
/// id newly clicked this frame, if any. Tree depth is small (≤3) so flat
/// rendering with manual indentation is more usable than nested collapsibles.
pub(crate) fn draw_feature_manifest(
    ui: &mut egui::Ui,
    manifest: &FeatureManifest,
    selected: Option<&FeatureId>,
) -> Option<FeatureId> {
    let mut clicked = None;
    ui.collapsing("Feature Manifest", |ui| {
        ui.label(format!("{} features", manifest.features.len()));
        let root_children = manifest
            .get(&manifest.root)
            .map(|root| root.children.clone())
            .unwrap_or_default();
        for child_id in &root_children {
            walk_manifest_flat(ui, manifest, child_id, selected, &mut clicked, 0);
        }
    });
    clicked
}

pub(crate) fn walk_manifest_flat(
    ui: &mut egui::Ui,
    manifest: &FeatureManifest,
    id: &FeatureId,
    selected: Option<&FeatureId>,
    clicked: &mut Option<FeatureId>,
    depth: usize,
) {
    let Some(feature) = manifest.get(id) else {
        return;
    };
    let indent: String = std::iter::repeat_n("  ", depth).collect();
    let scale = if feature.scale_range_m.max_m.is_finite() {
        format!(
            " · {:.1}-{:.1} km",
            feature.scale_range_m.min_m / 1_000.0,
            feature.scale_range_m.max_m / 1_000.0
        )
    } else {
        " · global".to_string()
    };
    let label = format!("{indent}{} · {:?}{scale}", feature.id, feature.kind);
    let is_selected = selected == Some(id);
    if ui.selectable_label(is_selected, label).clicked() {
        *clicked = Some(id.clone());
    }
    let children = feature.children.clone();
    for child_id in &children {
        walk_manifest_flat(ui, manifest, child_id, selected, clicked, depth + 1);
    }
}

/// Inspector panel for the selected manifest feature. Editable for authored
/// features (matched by id against `authored`); read-only for generated ones.
/// Sets `delete` to the feature id the user asked to remove, if any.
pub(crate) fn draw_selected_inspector(
    ui: &mut egui::Ui,
    selected_id: &FeatureId,
    manifest: &FeatureManifest,
    root_seed: u64,
    authored: &mut [AuthoredFeatureConfig],
    delete: &mut Option<FeatureId>,
) -> bool {
    let mut changed = false;
    let Some(feature) = manifest.get(selected_id) else {
        ui.label(format!("(missing feature: {selected_id})"));
        return false;
    };

    ui.heading(feature.id.as_str());
    ui.label(format!("Kind: {:?}", feature.kind));
    ui.label(format!("Era: {:?}", feature.era));
    if feature.scale_range_m.max_m.is_finite() {
        ui.label(format!(
            "Scale: {:.1}-{:.1} km",
            feature.scale_range_m.min_m / 1_000.0,
            feature.scale_range_m.max_m / 1_000.0
        ));
    } else {
        ui.label("Scale: global");
    }

    let authored_index = authored.iter().position(|a| match a {
        AuthoredFeatureConfig::Megabasin(c) => &c.id == selected_id,
    });

    if let Some(idx) = authored_index {
        ui.separator();
        ui.label("(authored)");
        match &mut authored[idx] {
            AuthoredFeatureConfig::Megabasin(config) => {
                changed |= fires(&ui.add(
                    egui::Slider::new(&mut config.radius_km, 50.0..=2000.0).text("Radius (km)"),
                ));
                changed |= fires(
                    &ui.add(egui::Slider::new(&mut config.depth_km, 0.5..=20.0).text("Depth (km)")),
                );

                let mut has_rings = config.ring_count.is_some();
                let mut ring_count = config.ring_count.unwrap_or(2);
                if ui.checkbox(&mut has_rings, "Concentric rings").changed() {
                    config.ring_count = if has_rings { Some(ring_count) } else { None };
                    changed = true;
                }
                if has_rings
                    && fires(&ui.add(egui::Slider::new(&mut ring_count, 1..=4).text("Ring count")))
                {
                    config.ring_count = Some(ring_count);
                    changed = true;
                }

                ui.separator();
                ui.label("Reroll seed:");
                ui.horizontal(|ui| {
                    for (label, stream) in [
                        ("Placement", FeatureSeedStream::Placement),
                        ("Shape", FeatureSeedStream::Shape),
                        ("Detail", FeatureSeedStream::Detail),
                        ("Children", FeatureSeedStream::Children),
                    ] {
                        if ui.small_button(label).clicked() {
                            reroll_authored_seed(root_seed, &config.id, &mut config.seed, stream);
                            changed = true;
                        }
                    }
                });

                let prev_lock = config.lock;
                egui::ComboBox::from_label("Lock")
                    .selected_text(format!("{:?}", config.lock))
                    .show_ui(ui, |ui| {
                        for lock in [
                            FeatureLock::Unlocked,
                            FeatureLock::Placement,
                            FeatureLock::Shape,
                            FeatureLock::Detail,
                            FeatureLock::ShapeAndPlacement,
                            FeatureLock::Full,
                        ] {
                            ui.selectable_value(&mut config.lock, lock, format!("{lock:?}"));
                        }
                    });
                if config.lock != prev_lock {
                    changed = true;
                }

                ui.separator();
                if ui.button("Delete").clicked() {
                    *delete = Some(config.id.clone());
                    changed = true;
                }
            }
        }
    } else {
        ui.label("(generated)");
        for param in &feature.params {
            let line = match &param.value {
                FeatureParamValue::Number(n) => format!("{}: {n:.3}", param.key),
                FeatureParamValue::Text(t) => format!("{}: {t}", param.key),
                FeatureParamValue::Bool(b) => format!("{}: {b}", param.key),
                FeatureParamValue::Direction(_) => format!("{}: <direction>", param.key),
            };
            ui.label(line);
        }
        ui.separator();
        ui.add_enabled(false, egui::Button::new("Promote (TODO)"));
    }

    changed
}

/// Generate a new authored-feature id like `user.megabasin.7` by scanning
/// existing authored features for the highest numeric suffix on `prefix`.
pub(crate) fn next_authored_id(authored: &[AuthoredFeatureConfig], prefix: &str) -> String {
    let mut max_n: u32 = 0;
    for a in authored {
        let id = match a {
            AuthoredFeatureConfig::Megabasin(c) => c.id.as_str(),
        };
        if let Some(rest) = id.strip_prefix(prefix)
            && let Ok(n) = rest.parse::<u32>()
        {
            max_n = max_n.max(n);
        }
    }
    format!("{prefix}{}", max_n + 1)
}

/// Ray-vs-sphere intersection (sphere centered at origin). Returns the
/// surface direction (unit) of the nearer hit, or `None` if the ray misses
/// or the sphere is behind the origin.
pub(crate) fn ray_vs_sphere(origin: Vec3, dir: Vec3, radius: f32) -> Option<Vec3> {
    let b = origin.dot(dir);
    let c = origin.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    if t < 0.0 {
        return None;
    }
    Some((origin + dir * t).normalize())
}

/// Convert a left-click on the 3D view into a new authored feature on the
/// planet surface. Inert when no placement tool is active or when the cursor
/// is over an egui panel.
pub(crate) fn pick_planet_click(
    input: Res<PlanetEditorInputIntent>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<EditorCamera>>,
    mut planet: ResMut<EditedPlanet>,
    mut egui_ctx: bevy_egui::EguiContexts,
) {
    if !input.primary_started {
        return;
    }
    if !planet.tool.placing() {
        return;
    }
    if egui_ctx
        .ctx_mut()
        .is_ok_and(|ctx| ctx.wants_pointer_input())
    {
        return;
    }

    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((camera, cam_transform)) = cameras.single() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(cam_transform, cursor) else {
        return;
    };
    let Some(direction) = ray_vs_sphere(ray.origin, *ray.direction, RENDER_RADIUS) else {
        return;
    };

    let tool = planet.tool;
    let new_id = match tool {
        ToolMode::Inspect => return,
        ToolMode::AddMegabasin => {
            let BodyMode::Terrain {
                ref mut terrain, ..
            } = planet.mode
            else {
                return;
            };
            let TerrainConfig::Feature(config) = terrain else {
                return;
            };
            let id_str = next_authored_id(&config.authored_features, "user.megabasin.");
            let new_id = FeatureId::new(id_str);
            config
                .authored_features
                .push(AuthoredFeatureConfig::Megabasin(MegabasinFeatureConfig {
                    id: new_id.clone(),
                    parent: None,
                    center_dir: direction,
                    radius_km: 250.0,
                    depth_km: 5.0,
                    ring_count: None,
                    seed: None,
                    lock: FeatureLock::Placement,
                }));
            new_id
        }
    };

    planet.selected_feature_id = Some(new_id);
    planet.terrain_dirty = true;
    planet.last_edit = Some(Instant::now());
}

pub(crate) fn equirect_dir(x: u32, y: u32, width: u32, height: u32) -> Vec3 {
    let u = (x as f32 + 0.5) / width as f32;
    let v = (y as f32 + 0.5) / height as f32;
    let lon = (u - 0.5) * std::f32::consts::TAU;
    let lat = (0.5 - v) * std::f32::consts::PI;
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();
    Vec3::new(cos_lat * sin_lon, sin_lat, cos_lat * cos_lon).normalize()
}

pub(crate) fn equirect_lod(surface: &PlanetSurface, width: u32) -> f32 {
    let meters_per_texel = std::f32::consts::TAU * surface.static_surface.radius_m / width as f32;
    meters_per_texel.max(1.0).log2()
}

pub(crate) fn sample_u8_cubemap(
    body: &StaticSurfaceData,
    dir: Vec3,
    which: EquirectFieldKind,
) -> u8 {
    let (face, u, v) = thalos_terrain::cubemap::dir_to_face_uv(dir);
    let res = body.roughness_cubemap.resolution();
    let x = ((u * res as f32) as u32).min(res - 1);
    let y = ((v * res as f32) as u32).min(res - 1);
    match which {
        EquirectFieldKind::StaticRoughness => body.roughness_cubemap.get(face, x, y),
        EquirectFieldKind::MaterialId => body.material_cubemap.get(face, x, y),
        _ => 0,
    }
}

pub(crate) fn sample_static_height_cubemap(body: &StaticSurfaceData, dir: Vec3) -> f32 {
    let (face, u, v) = thalos_terrain::cubemap::dir_to_face_uv(dir);
    let res = body.height_cubemap.resolution();
    let x = ((u * res as f32) as u32).min(res - 1);
    let y = ((v * res as f32) as u32).min(res - 1);
    let texel = body.height_cubemap.get(face, x, y);
    (texel as f32 / 65535.0 * 2.0 - 1.0) * body.height_range
}

pub(crate) fn sample_static_rgba_cubemap(
    body: &StaticSurfaceData,
    dir: Vec3,
    which: EquirectFieldKind,
) -> [u8; 4] {
    let (face, u, v) = thalos_terrain::cubemap::dir_to_face_uv(dir);
    let res = body.albedo_cubemap.resolution();
    let x = ((u * res as f32) as u32).min(res - 1);
    let y = ((v * res as f32) as u32).min(res - 1);
    match which {
        EquirectFieldKind::StaticAlbedo => body.albedo_cubemap.get(face, x, y),
        EquirectFieldKind::BakedNormal => body.normal_cubemap.get(face, x, y),
        _ => [0, 0, 0, 255],
    }
}

pub(crate) fn sample_biome_texel(
    body: &StaticSurfaceData,
    dir: Vec3,
) -> thalos_terrain::surface_field::BiomeMixTexel {
    let (face, u, v) = thalos_terrain::cubemap::dir_to_face_uv(dir);
    let res = body.biome_weights_cubemap.resolution();
    let x = ((u * res as f32) as u32).min(res - 1);
    let y = ((v * res as f32) as u32).min(res - 1);
    body.biome_weights_cubemap.get(face, x, y)
}

pub(crate) fn linear_to_srgb_u8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let srgb = if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (srgb.clamp(0.0, 1.0) * 255.0).round() as u8
}

pub(crate) fn luma_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

pub(crate) fn signed_luma_u8(v: f32, range: f32) -> u8 {
    luma_u8(0.5 + 0.5 * v / range.max(1e-6))
}

pub(crate) fn false_color(id: u32) -> [u8; 4] {
    let mut x = id.wrapping_mul(0x9E37_79B9).wrapping_add(0x85EB_CA6B);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    [
        64 + (x & 0x7F) as u8,
        64 + ((x >> 8) & 0x7F) as u8,
        64 + ((x >> 16) & 0x7F) as u8,
        255,
    ]
}

pub(crate) fn selected_equirect_descriptor(
    kind: EquirectFieldKind,
) -> &'static EquirectFieldDescriptor {
    EQUIRECT_FIELDS
        .iter()
        .find(|field| field.kind == kind)
        .unwrap_or(&EQUIRECT_FIELDS[0])
}

/// Inverse of [`linear_to_srgb_u8`]: decode an sRGB byte back to a linear
/// scalar. Used to feed the overlay shell's vertex colors, which Bevy treats
/// as linear; round-tripping the field's sRGB display color through this keeps
/// the on-planet overlay looking identical to the equirect preview.
pub(crate) fn srgb_u8_to_linear(v: u8) -> f32 {
    let v = v as f32 / 255.0;
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Equirect width that mirrors the baked cubemap resolution, so the preview
/// corresponds 1:1 to the rendered planet: a preview bake (512² cubemap)
/// yields a 1024×512 equirect, a full bake a proportionally larger one. Width
/// is 2× the cube-face resolution (height = face resolution), clamped to a
/// safety range.
pub(crate) fn equirect_width_for(surface: &PlanetSurface) -> u32 {
    let res = surface.static_surface.albedo_cubemap.resolution();
    (res * 2).clamp(EQUIRECT_VIEWER_MIN_WIDTH, EQUIRECT_VIEWER_MAX_WIDTH)
}

/// Single source of truth for how a field is colored. Returns the sRGB display
/// triple at a surface direction. Both the equirect viewer (uses the bytes
/// directly) and the on-planet overlay (decodes them to linear) call this, so
/// the two always show the same thing for a given field.
pub(crate) fn field_display_rgb(
    surface: &PlanetSurface,
    state: &DynamicSurfaceState,
    kind: EquirectFieldKind,
    dir: Vec3,
    lod: f32,
) -> [u8; 3] {
    let body = &surface.static_surface;
    let dynamic_height_range = (body.height_range * 0.1).max(10.0);
    match kind {
        EquirectFieldKind::FullSurfaceAlbedo => {
            let s = sample_surface(surface, state, dir, lod);
            [
                linear_to_srgb_u8(s.albedo.x),
                linear_to_srgb_u8(s.albedo.y),
                linear_to_srgb_u8(s.albedo.z),
            ]
        }
        EquirectFieldKind::FullSurfaceHeight => {
            let s = sample_surface(surface, state, dir, lod);
            let v = signed_luma_u8(s.height, body.height_range.max(1.0));
            [v, v, v]
        }
        EquirectFieldKind::FullSurfaceRoughness => {
            let s = sample_surface(surface, state, dir, lod);
            let v = luma_u8(s.roughness);
            [v, v, v]
        }
        EquirectFieldKind::FullSurfaceNormal => {
            let s = sample_surface(surface, state, dir, lod);
            [
                luma_u8(s.normal.x * 0.5 + 0.5),
                luma_u8(s.normal.y * 0.5 + 0.5),
                luma_u8(s.normal.z * 0.5 + 0.5),
            ]
        }
        EquirectFieldKind::StaticAlbedo | EquirectFieldKind::BakedNormal => {
            let c = sample_static_rgba_cubemap(body, dir, kind);
            [c[0], c[1], c[2]]
        }
        EquirectFieldKind::StaticHeight => {
            let h = sample_static_height_cubemap(body, dir);
            let v = signed_luma_u8(h, body.height_range.max(1.0));
            [v, v, v]
        }
        EquirectFieldKind::StaticRoughness => {
            let v = luma_u8(sample_u8_cubemap(body, dir, kind) as f32 / 255.0);
            [v, v, v]
        }
        EquirectFieldKind::MaterialId => {
            let c = false_color(sample_u8_cubemap(body, dir, kind) as u32);
            [c[0], c[1], c[2]]
        }
        EquirectFieldKind::Plates => surface
            .tectonics
            .as_ref()
            .map(|sys| {
                let sample = sys.sample(dir);
                let c = plate_color(sample.plate_kind, sample.plate_id.0).to_linear();
                [
                    linear_to_srgb_u8(c.red),
                    linear_to_srgb_u8(c.green),
                    linear_to_srgb_u8(c.blue),
                ]
            })
            .unwrap_or([20, 20, 20]),
        EquirectFieldKind::BiomeDominant => {
            let texel = sample_biome_texel(body, dir);
            let id = texel
                .iter_weights()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(id, _)| id)
                .unwrap_or(0);
            let c = biome_color(id).to_linear();
            [
                linear_to_srgb_u8(c.red),
                linear_to_srgb_u8(c.green),
                linear_to_srgb_u8(c.blue),
            ]
        }
        EquirectFieldKind::BiomeWeight => {
            let texel = sample_biome_texel(body, dir);
            let w = texel.iter_weights().map(|(_, w)| w).fold(0.0_f32, f32::max);
            let v = luma_u8(w);
            [v, v, v]
        }
        EquirectFieldKind::DynamicHeightDelta => {
            let full = sample_surface(surface, state, dir, lod);
            let static_sample = sample_static_surface(body, dir, lod);
            let v = signed_luma_u8(full.height - static_sample.height, dynamic_height_range);
            [v, v, v]
        }
        EquirectFieldKind::DynamicAlbedoDelta => {
            let full = sample_surface(surface, state, dir, lod);
            let static_sample = sample_static_surface(body, dir, lod);
            let v = luma_u8((full.albedo - static_sample.albedo).length());
            [v, v, v]
        }
        EquirectFieldKind::DynamicRoughnessDelta => {
            let full = sample_surface(surface, state, dir, lod);
            let static_sample = sample_static_surface(body, dir, lod);
            let v = signed_luma_u8(full.roughness - static_sample.roughness, 1.0);
            [v, v, v]
        }
    }
}

/// Linear-RGB form of [`field_display_rgb`] for the overlay shell's vertex
/// colors (Bevy vertex colors are linear; the framebuffer re-encodes to sRGB).
pub(crate) fn field_overlay_rgb_linear(
    surface: &PlanetSurface,
    state: &DynamicSurfaceState,
    kind: EquirectFieldKind,
    dir: Vec3,
    lod: f32,
) -> [f32; 3] {
    let [r, g, b] = field_display_rgb(surface, state, kind, dir, lod);
    [
        srgb_u8_to_linear(r),
        srgb_u8_to_linear(g),
        srgb_u8_to_linear(b),
    ]
}

/// Plate false-color used by the `Plates` field. Hue is hashed from the plate
/// id; continental plates read warm/green, oceanic plates cool.
pub(crate) fn plate_color(kind: PlateKind, plate_id: u32) -> Color {
    let h = thalos_terrain::seeding::splitmix64(plate_id as u64 ^ 0xB1ADE0FF);
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

/// Curated biome palette used by the `BiomeDominant` field. Falls back to a
/// hashed hue beyond the authored entries.
pub(crate) fn biome_color(biome_id: u8) -> Color {
    const COLORS: [[u8; 3]; 16] = [
        [214, 69, 80],
        [45, 137, 239],
        [62, 183, 98],
        [244, 188, 66],
        [158, 96, 214],
        [33, 184, 169],
        [239, 111, 47],
        [218, 218, 230],
        [132, 184, 64],
        [215, 91, 167],
        [86, 114, 214],
        [180, 136, 72],
        [89, 194, 219],
        [185, 82, 72],
        [124, 206, 144],
        [166, 166, 166],
    ];

    if let Some(rgb) = COLORS.get(biome_id as usize) {
        return Color::srgb(
            rgb[0] as f32 / 255.0,
            rgb[1] as f32 / 255.0,
            rgb[2] as f32 / 255.0,
        );
    }

    let h = thalos_terrain::seeding::splitmix64(biome_id as u64 ^ 0xB10B_1A5E);
    let hue = ((h & 0xFFFF) as f32) / 65535.0 * 360.0;
    Color::hsv(hue, 0.72, 0.88)
}

pub(crate) fn bake_equirect_field(
    surface: &PlanetSurface,
    state: &DynamicSurfaceState,
    kind: EquirectFieldKind,
) -> egui::ColorImage {
    let width = equirect_width_for(surface);
    let height = (width / 2).max(1);
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    let lod = equirect_lod(surface, width);

    for y in 0..height {
        for x in 0..width {
            let dir = equirect_dir(x, y, width, height);
            let [r, g, b] = field_display_rgb(surface, state, kind, dir, lod);
            let i = ((y * width + x) * 4) as usize;
            pixels[i..i + 4].copy_from_slice(&[r, g, b, 255]);
        }
    }

    egui::ColorImage::from_rgba_unmultiplied([width as usize, height as usize], &pixels)
}

/// The field viewer: a dropdown selecting one terrain field, the flattened
/// equirect preview of that field (at the baked cubemap's resolution), and a
/// toggle that projects the same field onto the 3D body as an overlay.
pub(crate) fn draw_field_viewer_panel(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    state: &mut EquirectViewerState,
    active_surface: &ActivePreviewSurface,
) {
    ui.heading("Field viewer");

    let selected = selected_equirect_descriptor(state.selected);
    egui::ComboBox::from_label("Field")
        .selected_text(selected.label)
        .show_ui(ui, |ui| {
            for field in EQUIRECT_FIELDS {
                if ui
                    .selectable_label(state.selected == field.kind, field.label)
                    .on_hover_text(field.help)
                    .clicked()
                {
                    state.selected = field.kind;
                    state.dirty = true;
                }
            }
        });
    ui.label(selected.help);
    ui.checkbox(&mut state.overlay_on_planet, "Overlay on planet")
        .on_hover_text("Project the selected field onto the 3D body. Hold Space to peek through it.");

    let (Some(surface), Some(dynamic_state)) =
        (&active_surface.surface, &active_surface.dynamic_state)
    else {
        ui.label("Waiting for a terrain preview bake…");
        return;
    };

    if state.last_body_name != active_surface.body_name {
        state.last_body_name = active_surface.body_name.clone();
        state.dirty = true;
    }
    if state.dirty || state.texture.is_none() {
        let image = bake_equirect_field(surface, dynamic_state, state.selected);
        if let Some(texture) = &mut state.texture {
            texture.set(image, egui::TextureOptions::LINEAR);
        } else {
            state.texture = Some(ctx.load_texture(
                "planet-editor-equirect-viewer",
                image,
                egui::TextureOptions::LINEAR,
            ));
        }
        state.dirty = false;
    }

    if let Some(texture) = &state.texture {
        let available = ui.available_width().max(64.0);
        let size = texture.size_vec2();
        let scale = (available / size.x).min(1.0);
        ui.add(egui::Image::new((texture.id(), size * scale)));
        ui.label(format!(
            "{}×{} — {}",
            size.x as u32, size.y as u32, active_surface.body_name
        ));
    }
}

pub(crate) fn draw_tile_viewer_panel(
    ui: &mut egui::Ui,
    state: &mut TileViewerState,
    active_surface: &ActivePreviewSurface,
) {
    ui.heading("Tile viewer");
    if ui
        .checkbox(&mut state.enabled, "Enable tile viewer")
        .changed()
    {
        state.dirty = true;
    }
    ui.add_enabled_ui(state.enabled, |ui| {
        ui.horizontal(|ui| {
            ui.label("Camera:");
            if ui
                .selectable_label(
                    state.camera_mode == TileViewerCameraMode::Orbit,
                    "Orbit tile",
                )
                .clicked()
            {
                state.camera_mode = TileViewerCameraMode::Orbit;
            }
            if ui
                .selectable_label(state.camera_mode == TileViewerCameraMode::Free, "Free cam")
                .clicked()
            {
                state.camera_mode = TileViewerCameraMode::Free;
            }
        });
        ui.label(
            "Drag to look. Free cam: W/A/S/D move, Q/E down/up, Shift sprint, wheel changes speed.",
        );

        let mut patch_changed = false;
        patch_changed |=
            fires(&ui.add(egui::Slider::new(&mut state.tile_count, 1..=16).text("N tiles")));
        patch_changed |= fires(
            &ui.add(
                egui::Slider::new(&mut state.tile_size_m, 16.0..=65_536.0)
                    .logarithmic(true)
                    .text("Tile size (m)"),
            ),
        );
        patch_changed |= fires(
            &ui.add(egui::Slider::new(&mut state.verts_per_tile, 4..=96).text("Verts / tile")),
        );
        patch_changed |= fires(
            &ui.add(egui::Slider::new(&mut state.center_lat_deg, -89.9..=89.9).text("Center lat")),
        );
        patch_changed |=
            fires(&ui.add(
                egui::Slider::new(&mut state.center_lon_deg, -180.0..=180.0).text("Center lon"),
            ));
        patch_changed |= fires(
            &ui.add(
                egui::Slider::new(&mut state.vertical_exaggeration, 0.1..=20.0)
                    .logarithmic(true)
                    .text("Vertical exaggeration"),
            ),
        );
        patch_changed |= fires(
            &ui.add(
                egui::Slider::new(&mut state.meters_per_unit, 1.0..=10_000.0)
                    .logarithmic(true)
                    .text("Meters / view unit"),
            ),
        );
        if patch_changed {
            state.dirty = true;
        }
        if ui.button("Rebuild tile patch").clicked() {
            state.dirty = true;
        }
        if active_surface.surface.is_none() {
            ui.label("Waiting for a terrain preview bake…");
        } else if let Some(stats) = state.stats {
            ui.label(format!(
                "Patch: {}×{} tiles, {:.1} km across",
                state.tile_count,
                state.tile_count,
                state.tile_count as f32 * state.tile_size_m / 1000.0
            ));
            ui.label(format!(
                "Height: {:+.1}…{:+.1} m (relief {:.1} m)",
                stats.min_height_m, stats.max_height_m, stats.relief_m
            ));
        }
    });
}

/// Authoring column for the active terrain body: bake controls, sketch tool,
/// seed/spec/projection, feature manifest + inspector, and the tectonic layer.
/// Lives in the left panel alongside the physical-body and definition editors.
/// Writes `requested_bake`, `tool`, and `selected_feature_id` directly; returns
/// whether any edit should mark the terrain dirty.
pub(crate) fn draw_generation_params_panel(
    ui: &mut egui::Ui,
    planet: &mut EditedPlanet,
    status: &TerrainGenStatus,
    active_surface: &ActivePreviewSurface,
) -> bool {
    if !matches!(planet.mode, BodyMode::Terrain { .. }) {
        return false;
    }

    let body_name = planet.selected_body.clone();
    let radius_m = planet.radius_m as f32;
    let gravity_m_s2 = planet.gravity_m_s2;
    let axial_tilt_rad = planet.axial_tilt_rad;
    let mode_label = planet.last_bake_mode.label();
    let busy = status.current_started.is_some();
    // Borrowed independently of `planet`, so it stays valid across the
    // `&mut planet.mode` borrow below.
    let tectonics_preview = active_surface
        .surface
        .as_ref()
        .and_then(|s| s.tectonics.as_ref());

    let mut terrain_changed = false;
    let mut selected_id = planet.selected_feature_id.clone();
    let mut selected_tool = planet.tool;
    let mut requested_bake = None;
    let mut delete_request: Option<FeatureId> = None;

    ui.heading("Generation params");
    match (status.current_started, status.last_duration) {
        (Some(started), _) => {
            let elapsed = started.elapsed().as_secs_f32();
            ui.label(format!("Generating ({mode_label}) for {elapsed:.2}s…"));
        }
        (None, Some(d)) => {
            ui.label(format!("Last bake ({mode_label}): {:.2}s", d.as_secs_f32()));
        }
        (None, None) => {}
    }
    ui.horizontal(|ui| {
        if ui
            .add_enabled(!busy, egui::Button::new("Bake half res"))
            .clicked()
        {
            requested_bake = Some(TerrainBakeMode::Half);
        }
        if ui
            .add_enabled(!busy, egui::Button::new("Bake full res"))
            .clicked()
        {
            requested_bake = Some(TerrainBakeMode::Full);
        }
    });

    ui.horizontal(|ui| {
        ui.label("Tool:");
        for tool in [ToolMode::Inspect, ToolMode::AddMegabasin] {
            let selected = selected_tool == tool;
            if ui.selectable_label(selected, tool.label()).clicked() {
                selected_tool = if selected { ToolMode::Inspect } else { tool };
            }
        }
    });
    ui.separator();

    if let BodyMode::Terrain {
        ref mut terrain,
        ref mut tectonics,
        tidal_axis,
    } = planet.mode
    {
        ui.label(format!("Terrain: {}", terrain.route_label()));
        match terrain {
            TerrainConfig::Feature(config) => {
                ui.horizontal(|ui| {
                    terrain_changed |= fires(
                        &ui.add(egui::Slider::new(&mut config.seed, 0..=9999).text("Seed")),
                    );
                    if ui.button("Reroll World").clicked() {
                        config.seed = sub_seed(config.seed, "planet_editor:world_seed");
                        terrain_changed = true;
                    }
                });
                terrain_changed |= draw_spec_controls(ui, config);
                terrain_changed |= draw_projection_controls(ui, &mut config.projection);

                let compile_context = TerrainCompileContext {
                    body_name: body_name.clone(),
                    radius_m,
                    gravity_m_s2,
                    rotation_hours: None,
                    obliquity_deg: Some(axial_tilt_rad.to_degrees()),
                    tidal_axis,
                    axial_tilt_rad,
                };
                let spec = config.to_planet_spec(&compile_context);
                let plan = plan_initial_compilation(&spec);
                if let Some(c) = draw_feature_manifest(ui, &plan.manifest, selected_id.as_ref()) {
                    selected_id = Some(c);
                }
                if let Some(sel) = selected_id.clone() {
                    ui.separator();
                    ui.heading("Selected");
                    terrain_changed |= draw_selected_inspector(
                        ui,
                        &sel,
                        &plan.manifest,
                        config.seed,
                        &mut config.authored_features,
                        &mut delete_request,
                    );
                    if let Some(del_id) = delete_request.clone() {
                        config.authored_features.retain(|a| match a {
                            AuthoredFeatureConfig::Megabasin(c) => c.id != del_id,
                        });
                        selected_id = None;
                        terrain_changed = true;
                    }
                }
            }
            TerrainConfig::Ocean(ocean) => {
                terrain_changed |=
                    fires(&ui.add(egui::Slider::new(&mut ocean.seed, 0..=9999).text("Seed")));
                terrain_changed |= fires(&ui.add(
                    egui::Slider::new(&mut ocean.sea_level_m, 0.0..=10.0).text("Sea level (m)"),
                ));
                terrain_changed |= fires(&ui.add(
                    egui::Slider::new(&mut ocean.water_roughness, 0.0..=0.3).text("Water roughness"),
                ));
            }
            TerrainConfig::None => {}
        }

        ui.separator();

        let archetype_requires_tectonics = matches!(
            terrain,
            TerrainConfig::Feature(c) if c.archetype == BodyArchetype::AgingOceanicHomeworld
        );
        terrain_changed |= draw_tectonics_panel(
            ui,
            tectonics,
            tectonics_preview,
            archetype_requires_tectonics,
        );
    }

    if let Some(bake) = requested_bake {
        planet.requested_bake = Some(bake);
    }
    planet.tool = selected_tool;
    planet.selected_feature_id = selected_id;
    terrain_changed
}

pub(crate) fn editor_ui(
    mut contexts: bevy_egui::EguiContexts,
    mut planet: ResMut<EditedPlanet>,
    system: Res<SystemData>,
    diagnostics: Res<DiagnosticsStore>,
    status: Res<TerrainGenStatus>,
    mut tile_viewer: ResMut<TileViewerState>,
    mut equirect_viewer: ResMut<EquirectViewerState>,
    active_surface: Res<ActivePreviewSurface>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    let selected_body_id = system.system.name_to_id.get(&planet.selected_body).copied();
    let mut clicked_body = None;
    let mut left_terrain_changed = false;
    let mut left_uniforms_changed = false;

    // Left column: authoring. Body tree, physical body, generation identity +
    // atmosphere, then the generation-params pipeline (bake, seed, spec,
    // manifest, tectonics).
    egui::SidePanel::left("planet_editor_left_panel")
        .resizable(true)
        .default_width(320.0)
        .min_width(260.0)
        .max_width(480.0)
        .show(ctx, |ui| {
            ui.heading("Celestial bodies");
            egui::ScrollArea::both()
                .id_salt("celestial_body_tree_scroll")
                .max_height(220.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    clicked_body = render_body_tree_ui(ui, &system.system, selected_body_id);
                });
            ui.separator();

            egui::ScrollArea::vertical()
                .id_salt("planet_editor_left_scroll")
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    let (terrain_changed, uniforms_changed) =
                        draw_physical_body_params(ui, &mut planet);
                    left_terrain_changed |= terrain_changed;
                    left_uniforms_changed |= uniforms_changed;

                    ui.separator();
                    let (terrain_changed, uniforms_changed) =
                        draw_planet_definition_panel(ui, &mut planet);
                    left_terrain_changed |= terrain_changed;
                    left_uniforms_changed |= uniforms_changed;

                    ui.separator();
                    left_terrain_changed |=
                        draw_generation_params_panel(ui, &mut planet, &status, &active_surface);
                });
        });
    if let Some(body_id) = clicked_body {
        select_body(&mut planet, &system.system, body_id);
    }
    if left_terrain_changed {
        planet.terrain_dirty = true;
        planet.last_edit = Some(Instant::now());
    }
    if left_uniforms_changed {
        planet.uniforms_dirty = true;
    }

    // Right column: visualization. FPS/body readout, the field viewer (equirect
    // + on-planet overlay toggle), the tile viewer, and shading controls.
    egui::SidePanel::right("planet_editor_right_panel")
        .resizable(true)
        .default_width(340.0)
        .min_width(300.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    let fps = diagnostics
                        .get(&FrameTimeDiagnosticsPlugin::FPS)
                        .and_then(|d| d.smoothed())
                        .unwrap_or(0.0);
                    ui.label(format!("FPS: {:.0}", fps));
                    ui.label(format!("Body: {}", planet.selected_body));
                    ui.separator();

                    let mut uniforms_changed = false;

                    if matches!(planet.mode, BodyMode::Terrain { .. }) {
                        draw_field_viewer_panel(ui, ctx, &mut equirect_viewer, &active_surface);
                        ui.separator();

                        ui.heading("View settings");
                        draw_tile_viewer_panel(ui, &mut tile_viewer, &active_surface);
                        ui.separator();
                    }

                    ui.heading("Shading");
                    if planet.atmosphere.is_some() {
                        uniforms_changed |= ui
                            .checkbox(&mut planet.atmosphere_enabled, "Atmosphere")
                            .changed();
                    }
                    uniforms_changed |= ui
                        .checkbox(&mut planet.full_bright, "Full bright")
                        .changed();
                    uniforms_changed |= ui
                        .checkbox(&mut planet.ambient_light, "Ambient light")
                        .changed();
                    let mut sun_azimuth_deg = planet.sun_azimuth.to_degrees();
                    if ui
                        .add(
                            egui::DragValue::new(&mut sun_azimuth_deg)
                                .speed(0.25)
                                .prefix("Sun azimuth: ")
                                .suffix(" deg")
                                .custom_formatter(|n, _| format!("{:.1}", n.rem_euclid(360.0))),
                        )
                        .changed()
                    {
                        planet.sun_azimuth = sun_azimuth_deg.to_radians();
                        uniforms_changed = true;
                    }

                    if uniforms_changed {
                        planet.uniforms_dirty = true;
                    }
                });
        });
}
