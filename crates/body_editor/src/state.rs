#![allow(clippy::too_many_arguments)]

use super::*;

pub(crate) const AMBIENT_INTENSITY: f32 = 0.05;
pub(crate) const DEFAULT_BODY_NAME: &str = "Mira";
pub(crate) const RENDER_RADIUS: f32 = 1.5;

/// Live-edit rebakes wait this long after the last edit before kicking off,
/// so a slider drag doesn't queue dozens of throwaway bakes.
pub(crate) const REBAKE_DEBOUNCE_MS: u128 = 150;
/// Cubemap resolution used for live preview rebakes. Keep this at the same
/// resolution as the normal headless bake so coastline shaping reads in the
/// editor instead of being smoothed away by the preview texture.
pub(crate) const PREVIEW_CUBEMAP_RESOLUTION: u32 = 512;
/// Explicit mid-resolution bake for checking near-final terrain without paying
/// the full 2048² compile cost.
pub(crate) const HALF_CUBEMAP_RESOLUTION: u32 = 1024;

pub(crate) const TILE_VIEWER_DEFAULT_TILE_COUNT: u32 = 4;
pub(crate) const TILE_VIEWER_DEFAULT_TILE_SIZE_M: f32 = 1024.0;
pub(crate) const TILE_VIEWER_DEFAULT_VERTS_PER_TILE: u32 = 32;
pub(crate) const TILE_VIEWER_DEFAULT_METERS_PER_UNIT: f32 = 1000.0;
pub(crate) const TILE_VIEWER_LOD_COUNT: u32 = 16;
pub(crate) const TILE_VIEWER_ATLAS_SIZE: u32 = 384;
pub(crate) const TILE_VIEWER_TEXTURE_SIZE: u32 = 512;
pub(crate) const TILE_VIEWER_TILE_BORDER_SIZE: u32 = 2;
pub(crate) const TILE_VIEWER_MIP_LEVELS: u32 = 4;
/// Safety clamp on the equirect viewer's derived width. The width tracks the
/// baked cubemap resolution (see `equirect_width_for`); these bounds keep a
/// pathological resolution from producing a degenerate or multi-second CPU
/// bake in the editor.
pub(crate) const EQUIRECT_VIEWER_MIN_WIDTH: u32 = 256;
pub(crate) const EQUIRECT_VIEWER_MAX_WIDTH: u32 = 4096;

// Body rendering mode

#[allow(clippy::large_enum_variant)]
pub(crate) enum BodyMode {
    Terrain {
        terrain: TerrainConfig,
        /// Optional tectonic structural prior. Cloned from the body's
        /// `BodyDefinition.tectonics`. Threaded through to the bake task
        /// so the resulting `PlanetSurface` carries a `TectonicSystem`
        /// for downstream visualization.
        tectonics: Option<TectonicConfig>,
        tidal_axis: Option<Vec3>,
    },
    GasGiant {
        layers: Box<GasGiantLayers>,
    },
    Star,
}

/// Ring system parameters held alongside [`BodyMode`] on
/// [`EditedPlanet`]. Sibling, not nested, so any body can have a ring.
pub(crate) struct EditorRings {
    pub(crate) inner_radius_m: f32,
    pub(crate) outer_radius_m: f32,
    pub(crate) layers: Box<RingLayers>,
}

pub(crate) struct EditorAtmosphere {
    pub(crate) block: AtmosphereBlock,
}

/// Active sketching tool. `Inspect` is the default — clicks on the planet
/// pick features instead of placing. The other variants enter placement mode:
/// the next planet click appends an authored feature of the matching kind.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolMode {
    #[default]
    Inspect,
    AddMegabasin,
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TerrainBakeMode {
    #[default]
    Preview,
    Half,
    Full,
}

impl TerrainBakeMode {
    pub(crate) fn resolution_override(self) -> Option<u32> {
        match self {
            Self::Preview => Some(PREVIEW_CUBEMAP_RESOLUTION),
            Self::Half => Some(HALF_CUBEMAP_RESOLUTION),
            Self::Full => None,
        }
    }

    pub(crate) fn label(self) -> String {
        match self {
            Self::Preview => format!("preview {PREVIEW_CUBEMAP_RESOLUTION}²"),
            Self::Half => format!("half {HALF_CUBEMAP_RESOLUTION}²"),
            Self::Full => "full".to_string(),
        }
    }
}

impl ToolMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Inspect => "Inspect",
            Self::AddMegabasin => "+ Megabasin",
        }
    }

    pub(crate) fn placing(self) -> bool {
        !matches!(self, Self::Inspect)
    }
}

// Resources

#[derive(Resource)]
pub(crate) struct SystemData {
    pub(crate) system: SolarSystemDefinition,
}

#[derive(Resource)]
pub(crate) struct EditedPlanet {
    pub(crate) selected_body: String,
    pub(crate) radius_m: f64,
    pub(crate) gravity_m_s2: f32,
    pub(crate) axial_tilt_rad: f32,
    pub(crate) mode: BodyMode,
    pub(crate) rings: Option<EditorRings>,
    pub(crate) atmosphere: Option<EditorAtmosphere>,
    pub(crate) atmosphere_enabled: bool,
    pub(crate) heliocentric_distance_m: f64,
    pub(crate) light_intensity: f32,
    pub(crate) sun_azimuth: f32,
    pub(crate) sun_orbital_elevation: f32,
    pub(crate) full_bright: bool,
    pub(crate) ambient_light: bool,
    pub(crate) terrain_dirty: bool,
    pub(crate) uniforms_dirty: bool,
    /// Body was switched — need to tear down and respawn the preview mesh.
    pub(crate) body_changed: bool,
    /// Wall-clock time of the most recent terrain-affecting edit. Drives the
    /// debounced preview rebake so a slider drag doesn't spawn throwaway tasks.
    pub(crate) last_edit: Option<Instant>,
    /// Set by explicit bake buttons. Bypasses debounce, then resets after
    /// dispatch. Live edits use `Preview`.
    pub(crate) requested_bake: Option<TerrainBakeMode>,
    /// Last or in-flight bake mode, for status display.
    pub(crate) last_bake_mode: TerrainBakeMode,
    /// Currently-selected manifest feature (None = nothing selected). Drives
    /// the per-feature inspector panel.
    pub(crate) selected_feature_id: Option<FeatureId>,
    /// Active sketching tool. While `placing()`, planet clicks add features
    /// and the orbit camera ignores left-button drag.
    pub(crate) tool: ToolMode,
}

#[derive(Resource, Default)]
pub(crate) struct TerrainGenStatus {
    pub(crate) current_started: Option<Instant>,
    pub(crate) last_duration: Option<Duration>,
}

#[derive(Resource)]
pub(crate) struct EditorBigSpaceRoot(pub(crate) Entity);

#[derive(Resource, Default)]
pub(crate) struct ActivePreviewSurface {
    pub(crate) body_name: String,
    pub(crate) surface: Option<Arc<PlanetSurface>>,
    pub(crate) dynamic_state: Option<DynamicSurfaceState>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TileViewerCameraMode {
    Orbit,
    Free,
}

#[derive(Default, Clone, Copy)]
pub(crate) struct TileViewerStats {
    pub(crate) min_height_m: f32,
    pub(crate) max_height_m: f32,
    pub(crate) relief_m: f32,
}

#[derive(Resource)]
pub(crate) struct TileViewerState {
    pub(crate) enabled: bool,
    pub(crate) camera_mode: TileViewerCameraMode,
    pub(crate) tile_count: u32,
    pub(crate) tile_size_m: f32,
    pub(crate) verts_per_tile: u32,
    pub(crate) center_lat_deg: f32,
    pub(crate) center_lon_deg: f32,
    pub(crate) vertical_exaggeration: f32,
    pub(crate) meters_per_unit: f32,
    pub(crate) orbit_azimuth: f32,
    pub(crate) orbit_elevation: f32,
    pub(crate) orbit_distance: f32,
    pub(crate) free_position: Vec3,
    pub(crate) free_yaw: f32,
    pub(crate) free_pitch: f32,
    pub(crate) free_speed_units_s: f32,
    pub(crate) dirty: bool,
    pub(crate) stats: Option<TileViewerStats>,
}

impl Default for TileViewerState {
    fn default() -> Self {
        Self {
            enabled: false,
            camera_mode: TileViewerCameraMode::Orbit,
            tile_count: TILE_VIEWER_DEFAULT_TILE_COUNT,
            tile_size_m: TILE_VIEWER_DEFAULT_TILE_SIZE_M,
            verts_per_tile: TILE_VIEWER_DEFAULT_VERTS_PER_TILE,
            center_lat_deg: 0.0,
            center_lon_deg: 0.0,
            vertical_exaggeration: 1.0,
            meters_per_unit: TILE_VIEWER_DEFAULT_METERS_PER_UNIT,
            orbit_azimuth: 0.0,
            orbit_elevation: 45.0_f32.to_radians(),
            orbit_distance: 8.0,
            free_position: Vec3::new(0.0, 3.0, 8.0),
            free_yaw: std::f32::consts::PI,
            free_pitch: -15.0_f32.to_radians(),
            free_speed_units_s: 8.0,
            dirty: true,
            stats: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EquirectFieldKind {
    FullSurfaceAlbedo,
    FullSurfaceHeight,
    FullSurfaceRoughness,
    FullSurfaceNormal,
    StaticAlbedo,
    StaticHeight,
    StaticRoughness,
    BakedNormal,
    MaterialId,
    Plates,
    BiomeDominant,
    BiomeWeight,
    DynamicHeightDelta,
    DynamicAlbedoDelta,
    DynamicRoughnessDelta,
}

#[derive(Clone, Copy)]
pub(crate) struct EquirectFieldDescriptor {
    pub(crate) kind: EquirectFieldKind,
    pub(crate) label: &'static str,
    pub(crate) help: &'static str,
}

pub(crate) const EQUIRECT_FIELDS: &[EquirectFieldDescriptor] = &[
    EquirectFieldDescriptor {
        kind: EquirectFieldKind::FullSurfaceAlbedo,
        label: "Full surface / albedo",
        help: "Static albedo plus dynamic overlays sampled through sample_surface().",
    },
    EquirectFieldDescriptor {
        kind: EquirectFieldKind::FullSurfaceHeight,
        label: "Full surface / height",
        help: "Static height plus features, high-frequency detail, and dynamic displacement.",
    },
    EquirectFieldDescriptor {
        kind: EquirectFieldKind::FullSurfaceRoughness,
        label: "Full surface / roughness",
        help: "Roughness after material fallback and dynamic overlays.",
    },
    EquirectFieldDescriptor {
        kind: EquirectFieldKind::FullSurfaceNormal,
        label: "Full surface / normal",
        help: "Normal from the canonical full surface sampler.",
    },
    EquirectFieldDescriptor {
        kind: EquirectFieldKind::StaticAlbedo,
        label: "Static cubemap / albedo",
        help: "Baked low-frequency albedo cubemap only.",
    },
    EquirectFieldDescriptor {
        kind: EquirectFieldKind::StaticHeight,
        label: "Static cubemap / height",
        help: "Baked low-frequency R16 height cubemap only.",
    },
    EquirectFieldDescriptor {
        kind: EquirectFieldKind::StaticRoughness,
        label: "Static cubemap / roughness",
        help: "Baked roughness cubemap, with no material fallback.",
    },
    EquirectFieldDescriptor {
        kind: EquirectFieldKind::BakedNormal,
        label: "Static cubemap / normal",
        help: "Baked object-space normal cubemap.",
    },
    EquirectFieldDescriptor {
        kind: EquirectFieldKind::MaterialId,
        label: "Static cubemap / material id",
        help: "False-color material index cubemap.",
    },
    EquirectFieldDescriptor {
        kind: EquirectFieldKind::Plates,
        label: "Tectonic plates",
        help: "False-color plate map (continental vs oceanic). Blank when the body has no tectonic layer.",
    },
    EquirectFieldDescriptor {
        kind: EquirectFieldKind::BiomeDominant,
        label: "Static cubemap / dominant biome",
        help: "False-color dominant biome from the biome-weight cubemap.",
    },
    EquirectFieldDescriptor {
        kind: EquirectFieldKind::BiomeWeight,
        label: "Static cubemap / dominant biome weight",
        help: "Normalized weight of the dominant biome entry.",
    },
    EquirectFieldDescriptor {
        kind: EquirectFieldKind::DynamicHeightDelta,
        label: "Dynamic overlays / height delta",
        help: "Full sampled height minus static sampled height.",
    },
    EquirectFieldDescriptor {
        kind: EquirectFieldKind::DynamicAlbedoDelta,
        label: "Dynamic overlays / albedo delta",
        help: "Magnitude of dynamic albedo change from static to full surface.",
    },
    EquirectFieldDescriptor {
        kind: EquirectFieldKind::DynamicRoughnessDelta,
        label: "Dynamic overlays / roughness delta",
        help: "Signed roughness change from dynamic overlays, visualized around neutral gray.",
    },
];

/// Drives both the equirect image preview (right panel) and the optional
/// on-planet overlay. The selected field is a single concept shared by both:
/// the equirect shows it flattened, the overlay projects the same field onto
/// the 3D body. Resolution is not user-controlled — it follows the baked
/// cubemap so the preview corresponds to the rendered planet.
#[derive(Resource)]
pub(crate) struct EquirectViewerState {
    pub(crate) selected: EquirectFieldKind,
    /// When set, the selected field is projected onto the 3D planet as an
    /// overlay shell. Read by `sync_surface_overlays`.
    pub(crate) overlay_on_planet: bool,
    pub(crate) dirty: bool,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) last_body_name: String,
}

impl Default for EquirectViewerState {
    fn default() -> Self {
        Self {
            selected: EquirectFieldKind::FullSurfaceAlbedo,
            overlay_on_planet: false,
            dirty: true,
            texture: None,
            last_body_name: String::new(),
        }
    }
}

#[derive(Resource)]
pub(crate) struct BillboardMesh(pub(crate) Handle<Mesh>);

#[derive(Component)]
pub(crate) struct TileViewerTerrain;

#[derive(Component)]
pub(crate) struct PendingTerrainGen {
    /// Bake task. Returns `Err` rather than panicking so a transient compile
    /// failure (e.g. an in-progress edit that puts the spec into a state the
    /// compiler rejects) just logs and leaves the existing terrain on
    /// screen, instead of taking the editor down with the task pool.
    pub(crate) task: Task<Result<PlanetSurface, String>>,
    pub(crate) mesh_entity: Entity,
}

#[allow(dead_code)]
#[derive(Component, Clone)]
pub(crate) struct PreviewDynamicSurface {
    pub(crate) layers: DynamicSurfaceLayers,
    pub(crate) state: DynamicSurfaceState,
}

#[derive(Component)]
pub(crate) struct PreviewPlanet;

#[derive(Component)]
pub(crate) struct PreviewRing;

#[derive(Component)]
pub(crate) struct PreviewAtmosphereHalo;

#[derive(Component, Default)]
pub(crate) struct PreviewCloudBandState {
    pub(crate) phases: [f64; CLOUD_BAND_COUNT],
}

#[derive(Resource, Default)]
pub(crate) struct PreviewAtmosphereClock {
    pub(crate) elapsed_s: f64,
}
