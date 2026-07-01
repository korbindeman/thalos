//! Procedural object preview — view the procedural objects (trees, conifers,
//! shrubs; rocks etc. later) on their own, in two modes:
//!
//! - **Headless gallery** (default): renders each object to a PNG under
//!   `tools/preview/out/` and exits. No window, off-screen render target — so it
//!   can run unattended (CI, an agent) and the images be inspected afterwards.
//!   `just preview` / `cargo run -p thalos_body_render --example object_preview`.
//! - **Interactive window** (`--window`): opens a window with an orbit camera so
//!   you can fly around the objects, cycle between them, and grab screenshots.
//!   `just preview-window` / `… --example object_preview -- --window`.
//!
//! Each object is staged as a small **diorama** so it reads like the in-game
//! surface, not a floating cutout: a sky-model-lit [`GroundPatchMaterial`] ground
//! (the same `thalos::lighting` BRDF the in-game terrain uses), a carpet of the
//! real grass blades around plants, and a self-managed **sun-shadow** pass so the
//! tree casts a leaf-shaped shadow on the ground and on itself — exactly the rig
//! the game runs (`thalos_game::rendering::sun_shadow`), trimmed to one cascade.
//! The camera mirrors the game's post stack (AgX tonemap + bloom + SMAA), minus
//! the sensor-sim grain / chromatic aberration that only muddy small asset shots.
//! Add an object by extending [`objects`].

use std::time::Duration;

use bevy::anti_alias::contrast_adaptive_sharpening::ContrastAdaptiveSharpening;
use bevy::anti_alias::smaa::{Smaa, SmaaPreset};
use bevy::app::{AppExit, ScheduleRunnerPlugin};
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::{
    ClearColorConfig, ImageRenderTarget, RenderTarget, ScalingMode,
};
// Bevy 0.19: render passes are systems in the `Core3d` schedule.
use bevy::core_pipeline::core_3d::{main_opaque_pass_3d, main_transparent_pass_3d};
use bevy::core_pipeline::tonemapping::{DebandDither, Tonemapping};
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::post_process::bloom::{Bloom, BloomCompositeMode, BloomPrefilter};
use bevy::prelude::*;
use bevy::camera::Hdr;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::view::window::screenshot::{Screenshot, save_to_disk};
use bevy::render::{
    RenderApp,
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    renderer::{RenderContext, ViewQuery},
    texture::GpuImage,
    view::ViewDepthTexture,
};
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;

use std::sync::Arc;

use thalos_body_render::{
    BakeParams, CanopyStyle, GrassBladeLod, GrassClumpParams, GrassFieldParams, GrassMaterial,
    GrassMaterialPlugin,
    GrassParams, GrassProfile, GroundPatchMaterial, GroundPatchMaterialPlugin, IMPOSTOR_MAX_SPECIES,
    ImpostorAtlasLayout, ImpostorParams, LIGHT_AT_1AU, RockMaterial, RockMaterialPlugin,
    RockMeshData, RockMeshParams, ShadowCascadeBlock, TreeBakeMaterial, TreeImpostorMaterial,
    TreeImpostorMaterialPlugin, TreeMaterial, TreeMaterialPlugin, TreeMeshParams, VegInstance,
    build_foliage_atlas, build_foliage_material_atlas, build_grass_clump_mesh,
    build_grass_field_mesh, build_rock_mesh, build_rock_mesh_data, build_tree_mesh,
    build_tree_mesh_data, combine_impostor_tile_mesh, combine_rock_tile_mesh, fallback_shadow_map,
    hemioct_decode, impostor_bake_rotation, make_impostor_atlas, recenter_tree_mesh,
    tree_bounding_sphere,
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 960;
/// Frames to render before the first capture (pipeline compile + atlas upload).
const WARMUP: u32 = 72;
/// Frames between one object's capture and the next (pose → shadow → capture).
const DWELL: u32 = 8;
/// Frames to keep running after the last capture so the async readback flushes.
const TAIL: u32 = 24;
const OUT_DIR: &str = "tools/preview/out";
const OBJECT_SPACING: f32 = 40.0;
/// Side length (m) of the scattered-pebble field diorama.
const ROCK_FIELD_SIZE_M: f32 = 4.0;

/// Sun direction (toward the star) shared by every object, so they light exactly
/// like the game's vegetation. A side-key at ~36° elevation, offset well to the
/// camera's side so it rakes form across the objects and throws their cast
/// shadows *across* the visible ground rather than hiding them behind the plant.
const SUN_DIR: Vec3 = Vec3::new(0.62, 0.46, -0.05);

/// Resolve the diorama sun direction. Defaults to [`SUN_DIR`]; overridable for
/// time-of-day experiments via `THALOS_PREVIEW_SUN_ELEV_DEG` (sun elevation in
/// degrees, keeping the same horizontal azimuth) — e.g. `=85` puts the sun near
/// the substellar noon overhead to reproduce the "washed-out at noon" surface.
fn preview_sun_dir() -> Vec3 {
    if let Ok(raw) = std::env::var("THALOS_PREVIEW_SUN_ELEV_DEG")
        && let Ok(elev_deg) = raw.trim().parse::<f32>()
    {
        let elev = elev_deg.to_radians();
        // Preserve the committed azimuth (mostly +X, slightly −Z) so shadows
        // still rake across the visible ground.
        let horiz = Vec3::new(SUN_DIR.x, 0.0, SUN_DIR.z).normalize_or_zero();
        return (horiz * elev.cos() + Vec3::Y * elev.sin()).normalize();
    }
    SUN_DIR.normalize()
}
/// Surface sun flux in the same units the terrain `SceneLighting` carries. The
/// game's exposure gain keeps the surface value ~`LIGHT_AT_1AU` regardless of the
/// body's orbital distance (`rendering::lighting`), so that's the value here.
const SUN_FLUX: f32 = LIGHT_AT_1AU;
/// Thalos's authored clear-sky scattering (`assets/bodies/thalos.ron`):
/// Rayleigh τ_v + atmosphere strength, fed to the shared `compute_surface_sky`
/// so the preview sky tint + ambient match the game's surface exactly.
const SKY_TAU: Vec3 = Vec3::new(0.046, 0.108, 0.264);
const SKY_STRENGTH: f32 = 3.0;
/// Background clear colour — a tuned daylight horizon blue behind the diorama.
const SKY: Color = Color::srgb(0.46, 0.62, 0.82);

fn main() {
    std::fs::create_dir_all(OUT_DIR).ok();
    let window_mode = std::env::args()
        .skip(1)
        .any(|a| matches!(a.as_str(), "window" | "--window" | "-w"));

    let mut app = App::new();
    if window_mode {
        app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Thalos — Object Preview".into(),
                resolution: (WIDTH, HEIGHT).into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(
            Startup,
            (setup_impostor_bake, setup_scene, setup_window_camera).chain(),
        )
        .add_systems(
            Update,
            (orbit_camera, update_preview_shadow, screenshot_key)
                .chain(),
        )
        .add_systems(Update, teardown_impostor_bake);
    } else {
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    close_when_requested: false,
                    ..default()
                })
                .disable::<WinitPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 60.0,
        )))
        .add_systems(
            Startup,
            (setup_impostor_bake, setup_scene, setup_headless_camera).chain(),
        )
        .add_systems(
            Update,
            (drive_capture, update_preview_shadow, teardown_impostor_bake).chain(),
        );
    }
    app.add_plugins((
        TreeMaterialPlugin,
        TreeImpostorMaterialPlugin,
        GrassMaterialPlugin,
        GroundPatchMaterialPlugin,
        RockMaterialPlugin,
        PreviewShadowPlugin,
    ))
    .run();
}

/// What an object is built from — a procedural tree/shrub mesh (`TreeMaterial`),
/// the SAME species as an octahedral **impostor** card (`TreeImpostorMaterial`,
/// the far-LOD billboard) for a mesh↔impostor parity check, or a standalone grass
/// tuft / field (`GrassMaterial`).
#[derive(Clone)]
enum AssetKind {
    Tree(TreeMeshParams),
    /// The [`broadleaf`] rendered as its far-band impostor (the preview bakes one
    /// species). Frame it identically to the matching mesh `Tree` object and the
    /// two PNGs should read continuous — the regression check that the bake still
    /// derives from the near material.
    TreeImpostor,
    GrassClump(GrassClumpParams),
    GrassField(GrassFieldParams),
    /// A single procedural pebble / rock (`RockMaterial`), for shape iteration.
    Rock(RockMeshParams),
    /// A scattered patch of mixed pebbles among grass — the in-game look, where
    /// the stones get partly covered by the blades.
    RockField,
}

/// Camera angle a preview is framed from.
#[derive(Clone, Copy)]
enum ViewKind {
    ThreeQuarter,
    Top,
    Side,
}

/// One previewable object: a display name, what to build, how to frame it.
#[derive(Clone)]
struct Preview {
    name: &'static str,
    kind: AssetKind,
    view: ViewKind,
    /// Height (m) of the point the camera looks at, and the camera distance.
    focus_y: f32,
    distance: f32,
}

impl Preview {
    /// Side length (m) of the diorama ground patch under this object — wide
    /// enough to fill the framed view's lower half. Trees grow a grass carpet
    /// out to the same extent (no bare-ground ring); grass objects bring their
    /// own blades, so their patch is just the surrounding clearing.
    fn patch_size_m(&self) -> f32 {
        match self.kind {
            AssetKind::Tree(_) | AssetKind::TreeImpostor => (self.distance * 1.8).clamp(6.0, 24.0),
            // A single pebble wants only a small ground tile under it; the field
            // wants the whole scattered patch.
            AssetKind::Rock(_) => (self.distance * 1.6).clamp(1.0, 6.0),
            AssetKind::RockField => ROCK_FIELD_SIZE_M,
            _ => (self.distance * 2.4).clamp(3.0, 40.0),
        }
    }
}

/// The broadleaf species used for the mesh↔impostor comparison shots. Shared by
/// the `tree_broadleaf` mesh object, its impostor counterpart, and the off-screen
/// impostor bake, so all three describe the SAME tree.
fn broadleaf() -> TreeMeshParams {
    TreeMeshParams {
        trunk_height_m: 4.8,
        trunk_radius_m: 0.30,
        canopy_radius_m: 3.2,
        canopy_height_m: 3.0,
        trunk_color: Vec3::new(0.16, 0.090, 0.045),
        canopy_color: Vec3::new(0.92, 1.0, 0.82),
        style: CanopyStyle::Broadleaf,
        seed: 0xB1_05_50,
        lod: 0,
    }
}

fn objects() -> Vec<Preview> {
    let clump = GrassClumpParams::default();
    vec![
        Preview {
            name: "tree_broadleaf",
            kind: AssetKind::Tree(broadleaf()),
            view: ViewKind::ThreeQuarter,
            focus_y: 4.4,
            distance: 16.0,
        },
        // The mesh LOD chain of the SAME broadleaf, framed identically to
        // `tree_broadleaf` (LOD0) so the LOD steps can be compared for coverage /
        // leaf size — they should read as the same tree, just cheaper, not sparser.
        Preview {
            name: "tree_broadleaf_lod1",
            kind: AssetKind::Tree(TreeMeshParams { lod: 1, ..broadleaf() }),
            view: ViewKind::ThreeQuarter,
            focus_y: 4.4,
            distance: 16.0,
        },
        Preview {
            name: "tree_broadleaf_lod2",
            kind: AssetKind::Tree(TreeMeshParams { lod: 2, ..broadleaf() }),
            view: ViewKind::ThreeQuarter,
            focus_y: 4.4,
            distance: 16.0,
        },
        Preview {
            name: "tree_broadleaf_lod3",
            kind: AssetKind::Tree(TreeMeshParams { lod: 3, ..broadleaf() }),
            view: ViewKind::ThreeQuarter,
            focus_y: 4.4,
            distance: 16.0,
        },
        // The SAME broadleaf as its far-band octahedral impostor, framed
        // identically — the mesh↔impostor parity check. `tree_broadleaf.png` and
        // `tree_broadleaf_impostor.png` should read as the same tree (colour +
        // value); if they diverge, the bake has drifted from the near material.
        Preview {
            name: "tree_broadleaf_impostor",
            kind: AssetKind::TreeImpostor,
            view: ViewKind::ThreeQuarter,
            focus_y: 4.4,
            distance: 16.0,
        },
        // Side view of the same impostor — silhouette + coverage against the mesh
        // side shot (`tree_broadleaf_b`, same species).
        Preview {
            name: "tree_broadleaf_impostor_side",
            kind: AssetKind::TreeImpostor,
            view: ViewKind::Side,
            focus_y: 4.8,
            distance: 17.0,
        },
        // A second broadleaf (different seed, framed level) to judge silhouette +
        // shape variation between trees of the same species.
        Preview {
            name: "tree_broadleaf_b",
            kind: AssetKind::Tree(TreeMeshParams {
                trunk_height_m: 5.2,
                trunk_radius_m: 0.32,
                canopy_radius_m: 3.0,
                canopy_height_m: 3.4,
                trunk_color: Vec3::new(0.16, 0.090, 0.045),
                canopy_color: Vec3::new(0.92, 1.0, 0.82),
                style: CanopyStyle::Broadleaf,
                seed: 0x2E_3A_77,
                lod: 0,
            }),
            view: ViewKind::Side,
            focus_y: 4.8,
            distance: 17.0,
        },
        // A fat, short trunk framed close and side-on so the bark material fills
        // the view (its canopy sits above frame) — the bark study shot.
        Preview {
            name: "bark_log",
            kind: AssetKind::Tree(TreeMeshParams {
                trunk_height_m: 4.0,
                trunk_radius_m: 0.62,
                canopy_radius_m: 1.4,
                canopy_height_m: 1.4,
                trunk_color: Vec3::new(0.16, 0.090, 0.045),
                canopy_color: Vec3::new(0.92, 1.0, 0.82),
                style: CanopyStyle::Broadleaf,
                seed: 0xBA_12_09,
                lod: 0,
            }),
            view: ViewKind::Side,
            focus_y: 1.5,
            distance: 2.4,
        },
        Preview {
            name: "shrub",
            kind: AssetKind::Tree(TreeMeshParams {
                trunk_height_m: 0.35,
                trunk_radius_m: 0.06,
                canopy_radius_m: 0.78,
                canopy_height_m: 0.62,
                trunk_color: Vec3::new(0.13, 0.085, 0.050),
                canopy_color: Vec3::new(0.95, 1.0, 0.86),
                style: CanopyStyle::Round,
                seed: 0x5_417,
                lod: 0,
            }),
            view: ViewKind::ThreeQuarter,
            focus_y: 0.7,
            distance: 3.2,
        },
        Preview {
            name: "grass_clump_top",
            kind: AssetKind::GrassClump(clump),
            view: ViewKind::Top,
            focus_y: clump.profile.height_m * 0.45,
            distance: 1.6,
        },
        Preview {
            name: "grass_clump_side",
            kind: AssetKind::GrassClump(clump),
            view: ViewKind::Side,
            focus_y: clump.profile.height_m * 0.5,
            distance: 1.6,
        },
        Preview {
            name: "grass_clump_3q",
            kind: AssetKind::GrassClump(clump),
            view: ViewKind::ThreeQuarter,
            focus_y: clump.profile.height_m * 0.5,
            distance: 1.5,
        },
        // The two named grass *types* the in-game distribution blends between,
        // side-on so thickness/length/fluffiness read directly: short & thick &
        // fluffy vs. tall & thin & wispy.
        {
            let c = GrassClumpParams {
                profile: GrassProfile::fluffy_short(),
                ..GrassClumpParams::default()
            };
            Preview {
                name: "grass_clump_fluffy_short",
                kind: AssetKind::GrassClump(c),
                view: ViewKind::Side,
                focus_y: GrassProfile::fluffy_short().height_m * 0.5,
                distance: 1.3,
            }
        },
        {
            let c = GrassClumpParams {
                profile: GrassProfile::wispy_tall(),
                ..GrassClumpParams::default()
            };
            Preview {
                name: "grass_clump_wispy_tall",
                kind: AssetKind::GrassClump(c),
                view: ViewKind::Side,
                focus_y: GrassProfile::wispy_tall().height_m * 0.5,
                distance: 2.0,
            }
        },
        {
            // The spaceport-lawn type: short, thick, near-upright. The managed
            // ground cover forced inside a base's `ScatterTreatment::Lawn`
            // footprint (`GRASS_PROFILE_LAWN` in `game::rendering::grass`).
            let c = GrassClumpParams {
                profile: GrassProfile::lawn(),
                ..GrassClumpParams::default()
            };
            Preview {
                name: "grass_clump_lawn",
                kind: AssetKind::GrassClump(c),
                view: ViewKind::Side,
                focus_y: GrassProfile::lawn().height_m * 0.5,
                distance: 0.9,
            }
        },
        {
            // A carpet of the lawn type — the spaceport ground from a grazing
            // eye-line (`_3q`) and from overhead (`_top`). Higher clump density
            // since each lawn tuft is small; reads as kept turf, not wild meadow.
            let field = GrassFieldParams {
                profile: GrassProfile::lawn(),
                clumps_per_m2: 40.0,
                ..GrassFieldParams::default()
            };
            Preview {
                name: "grass_field_lawn_3q",
                kind: AssetKind::GrassField(field),
                view: ViewKind::ThreeQuarter,
                focus_y: field.profile.height_m * 0.5,
                distance: 2.6,
            }
        },
        {
            let field = GrassFieldParams {
                profile: GrassProfile::lawn(),
                clumps_per_m2: 40.0,
                ..GrassFieldParams::default()
            };
            Preview {
                name: "grass_field_lawn_top",
                kind: AssetKind::GrassField(field),
                view: ViewKind::Top,
                focus_y: 0.0,
                distance: 3.2,
            }
        },
        // A field of fountain clumps — the real test of "fluffy" and "looks good
        // from above". `_top` is the aerial regime (aircraft overhead), `_3q` a
        // grazing eye-line, `_dry_3q` the windswept dry-prairie look.
        {
            let field = GrassFieldParams::default();
            Preview {
                name: "grass_field_top",
                kind: AssetKind::GrassField(field),
                view: ViewKind::Top,
                focus_y: 0.0,
                distance: 4.5,
            }
        },
        {
            let field = GrassFieldParams::default();
            Preview {
                name: "grass_field_3q",
                kind: AssetKind::GrassField(field),
                view: ViewKind::ThreeQuarter,
                focus_y: field.profile.height_m * 0.5,
                distance: 4.0,
            }
        },
        {
            let field = GrassFieldParams::default();
            Preview {
                name: "grass_field_side",
                kind: AssetKind::GrassField(field),
                view: ViewKind::Side,
                focus_y: field.profile.height_m * 0.55,
                distance: 3.0,
            }
        },
        {
            // Dry windswept prairie: golden straw colour, the taller wispy type,
            // and a baked wind lean.
            let field = GrassFieldParams {
                color: Vec3::new(0.150, 0.115, 0.034),
                wind_lean: 0.45,
                profile: GrassProfile::wispy_tall(),
                ..GrassFieldParams::default()
            };
            Preview {
                name: "grass_field_dry_3q",
                kind: AssetKind::GrassField(field),
                view: ViewKind::ThreeQuarter,
                focus_y: field.profile.height_m * 0.5,
                distance: 4.0,
            }
        },
        {
            // Clump-CARD field: the far/mid-band billboard tuft representation (one
            // crossed-quad pair per clump with a procedural tuft alpha) — the
            // vertex-cheap LOD. Lower density (cards are wider) + a far-ring profile.
            let field = GrassFieldParams {
                // In-game far-ring density, so the from-distance coverage reads true.
                clumps_per_m2: 2.5,
                profile: GrassProfile::default().scaled(2.8, 1.2, 1.0),
                lod: GrassBladeLod::Card,
                ..GrassFieldParams::default()
            };
            Preview {
                name: "grass_field_card_far",
                kind: AssetKind::GrassField(field),
                view: ViewKind::ThreeQuarter,
                focus_y: 0.5,
                // Far back: the cards are a 160 m+ distance LOD, so judge them at
                // distance (the 4 m close-up is the wrong test). Patch auto-sizes to
                // ~40 m, so the field recedes from ~25 m to ~65 m here.
                distance: 45.0,
            }
        },
        // --- Rocks / pebbles ---
        // A single stylized pebble, framed close from three-quarter + side so the
        // plane-cut facets, sharp edges, and baked cavity-AO read.
        Preview {
            name: "pebble_3q",
            kind: AssetKind::Rock(pebble()),
            view: ViewKind::ThreeQuarter,
            focus_y: 0.06,
            distance: 0.85,
        },
        Preview {
            name: "pebble_side",
            kind: AssetKind::Rock(pebble()),
            view: ViewKind::Side,
            focus_y: 0.07,
            distance: 0.85,
        },
        // A bigger, blockier, more heavily-faceted stone (angular boulder).
        Preview {
            name: "rock_angular",
            kind: AssetKind::Rock(RockMeshParams {
                radius_m: 0.34,
                axes: Vec3::new(1.0, 0.74, 0.9),
                cuts: 18,
                cut_depth: (0.66, 0.95),
                color: Vec3::new(0.46, 0.44, 0.42),
                seed: 0x9A_17_03,
                subdivisions: 3,
                ..RockMeshParams::default()
            }),
            view: ViewKind::ThreeQuarter,
            focus_y: 0.12,
            distance: 1.2,
        },
        // The in-game look: a scattered patch of mixed pebbles in a meadow, so
        // the grass partly covers the smaller stones (`pebbles among grass`).
        Preview {
            name: "rock_field_3q",
            kind: AssetKind::RockField,
            view: ViewKind::ThreeQuarter,
            focus_y: 0.05,
            distance: 2.2,
        },
        Preview {
            name: "rock_field_top",
            kind: AssetKind::RockField,
            view: ViewKind::Top,
            focus_y: 0.0,
            distance: 2.2,
        },
    ]
}

/// A representative small, plane-cut faceted, light pebble for the single-stone
/// shots (matches the dominant in-game species, at hero tessellation).
fn pebble() -> RockMeshParams {
    RockMeshParams {
        radius_m: 0.11,
        axes: Vec3::new(1.0, 0.60, 0.84),
        color: Vec3::new(0.50, 0.47, 0.42),
        seed: 0x7E_BB_1E,
        subdivisions: 3,
        ..RockMeshParams::default()
    }
}

/// The small library of pebble species the scattered field mixes — small,
/// faceted, light natural stone, the same as the game driver scatters (mostly
/// small chips; the field samples them ~uniformly, so most entries are small).
fn rock_field_species() -> Vec<RockMeshParams> {
    let base = RockMeshParams {
        subdivisions: 2,
        ..RockMeshParams::default()
    };
    vec![
        RockMeshParams {
            radius_m: 0.050,
            axes: Vec3::new(1.0, 0.62, 0.86),
            color: Vec3::new(0.50, 0.47, 0.42),
            seed: 0x11,
            subdivisions: 1,
            cuts: 8,
            ..base
        },
        RockMeshParams {
            radius_m: 0.075,
            axes: Vec3::new(1.0, 0.58, 0.82),
            color: Vec3::new(0.52, 0.46, 0.37),
            seed: 0x12,
            ..base
        },
        RockMeshParams {
            radius_m: 0.11,
            axes: Vec3::new(1.0, 0.58, 0.84),
            color: Vec3::new(0.44, 0.45, 0.47),
            seed: 0x22,
            ..base
        },
        RockMeshParams {
            radius_m: 0.16,
            axes: Vec3::new(1.0, 0.60, 0.88),
            color: Vec3::new(0.48, 0.45, 0.41),
            seed: 0x33,
            ..base
        },
        RockMeshParams {
            radius_m: 0.24,
            axes: Vec3::new(1.0, 0.70, 0.90),
            color: Vec3::new(0.46, 0.44, 0.42),
            seed: 0x44,
            cuts: 16,
            ..base
        },
    ]
}

/// Build the scattered-pebble field as one batched mesh (the same combiner the
/// game uses), via a small deterministic hashed scatter over the patch.
fn rock_field_mesh() -> Mesh {
    let species: Vec<Option<Arc<RockMeshData>>> = rock_field_species()
        .iter()
        .map(|p| Some(Arc::new(build_rock_mesh_data(p))))
        .collect();

    let half = ROCK_FIELD_SIZE_M * 0.5 - 0.2;
    let n = 80usize;
    let h = |i: usize, salt: u64| -> f32 {
        let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt.wrapping_mul(0xD1B5_4A32);
        x ^= x >> 31;
        x = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        x ^= x >> 29;
        (x & 0xFFFFFF) as f32 / 0xFFFFFF as f32
    };
    let mut instances = Vec::with_capacity(n);
    for i in 0..n {
        let species_idx = (h(i, 0x5) * species.len() as f32) as u16;
        instances.push(VegInstance {
            species: species_idx.min(species.len() as u16 - 1),
            root_offset_body_m: Vec3::new(
                (h(i, 0x1) * 2.0 - 1.0) * half,
                0.0,
                (h(i, 0x2) * 2.0 - 1.0) * half,
            ),
            up_body: Vec3::Y,
            yaw: h(i, 0x3) * std::f32::consts::TAU,
            scale: 0.7 + 0.8 * h(i, 0x4),
            tilt: h(i, 0x6) * 0.5,
        });
    }
    combine_rock_tile_mesh(&instances, &species).unwrap_or_else(|| build_rock_mesh(&pebble()))
}

/// The placed objects, shared by the camera/capture/orbit systems.
#[derive(Resource)]
struct Scene {
    objects: Vec<Preview>,
}

/// Shared diorama material handles, so the sun-shadow driver can push the live
/// cascade block + depth maps onto the ground + grass + tree materials each frame.
#[derive(Resource)]
struct DioramaMaterials {
    ground: Handle<GroundPatchMaterial>,
    grass: Handle<GrassMaterial>,
    tree: Handle<TreeMaterial>,
    rock: Handle<RockMaterial>,
}

/// World focus point of object `i` (its spot along +X at its framing height).
fn object_focus(objects: &[Preview], i: usize) -> Vec3 {
    Vec3::new(i as f32 * OBJECT_SPACING, objects[i].focus_y, 0.0)
}

/// Camera transform framing object `index` from its `view` angle.
fn frame_transform(obj: &Preview, index: usize) -> Transform {
    let focus = Vec3::new(index as f32 * OBJECT_SPACING, obj.focus_y, 0.0);
    let d = obj.distance;
    match obj.view {
        // Straight down (aerial). `up` is +Z since +Y is degenerate looking down.
        ViewKind::Top => {
            let eye = focus + Vec3::new(0.0, d, 0.0);
            Transform::from_translation(eye).looking_at(focus, Vec3::Z)
        }
        // Level eye-line (grazing horizon).
        ViewKind::Side => {
            let eye = focus + Vec3::new(0.0, 0.0, d);
            Transform::from_translation(eye).looking_at(focus, Vec3::Y)
        }
        ViewKind::ThreeQuarter => {
            let eye = focus + Vec3::new(d * 0.55, obj.focus_y * 0.6 + d * 0.18, d * 0.92);
            Transform::from_translation(eye).looking_at(focus, Vec3::Y)
        }
    }
}

/// Shared sky/sun lighting parameters — identical for every material, so the
/// ground, grass, and trees light from one source exactly like the game.
fn veg_params() -> GrassParams {
    GrassParams {
        sun_dir: preview_sun_dir().extend(SUN_FLUX),
        wind: Vec4::ZERO,
        // Always full size (no clipmap fade): near edge well below any distance,
        // far edge well above.
        time_fade: Vec4::new(0.0, -1.0e9, 1.0e9, 1.0),
        sky_up: Vec3::Y.extend(0.0),
        sky_tau: SKY_TAU.extend(SKY_STRENGTH),
        anchor: Vec4::ZERO,
    }
}

/// Spawn each object's diorama: a sky-model-lit ground patch, a grass carpet
/// around plants, and the object itself (trees tagged onto the shadow-caster
/// layer so they cast into the sun-shadow pass).
fn setup_scene(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut tree_materials: ResMut<Assets<TreeMaterial>>,
    mut grass_materials: ResMut<Assets<GrassMaterial>>,
    mut ground_materials: ResMut<Assets<GroundPatchMaterial>>,
    mut rock_materials: ResMut<Assets<RockMaterial>>,
    impostor_rig: Res<ImpostorRig>,
) {
    let params = veg_params();
    let fallback = images.add(fallback_shadow_map());

    // One shared tree material with the procedural foliage atlas + sky lighting.
    let tree_material = tree_materials.add(TreeMaterial {
        atlas: images.add(build_foliage_atlas()),
        material_atlas: images.add(build_foliage_material_atlas()),
        params,
        // The sun-shadow driver rebinds cascade 0's real depth map each frame;
        // 1 + 2 stay on the fallback (the preview runs a single cascade).
        sun_shadow_map_0: fallback.clone(),
        sun_shadow_map_1: fallback.clone(),
        sun_shadow_map_2: fallback.clone(),
        ..default()
    });
    // One shared grass material (vertex-coloured blades, same sky model). It
    // receives the tree's cast shadow via the same cascade the ground samples;
    // the driver rebinds cascade 0's real depth map each frame.
    let grass_material = grass_materials.add(GrassMaterial {
        params,
        sun_shadow_map_0: fallback.clone(),
        sun_shadow_map_1: fallback.clone(),
        sun_shadow_map_2: fallback.clone(),
        ..default()
    });
    // One shared sky-model-lit ground material (receives the tree shadows).
    let ground_material = ground_materials.add(GroundPatchMaterial {
        params,
        sun_shadow_map_0: fallback.clone(),
        sun_shadow_map_1: fallback.clone(),
        sun_shadow_map_2: fallback.clone(),
        ..default()
    });
    // One shared rock material (vertex-coloured stone, same sky model). Casts
    // and receives the cascade just like the trees.
    let rock_material = rock_materials.add(RockMaterial {
        params,
        sun_shadow_map_0: fallback.clone(),
        sun_shadow_map_1: fallback.clone(),
        sun_shadow_map_2: fallback.clone(),
        ..default()
    });

    let objects = objects();

    for (i, obj) in objects.iter().enumerate() {
        let transform = Transform::from_xyz(i as f32 * OBJECT_SPACING, 0.0, 0.0);
        let patch = obj.patch_size_m();

        // Ground patch (shadow receiver) under every object.
        commands.spawn((
            Mesh3d(meshes.add(Plane3d::default().mesh().size(patch, patch))),
            MeshMaterial3d(ground_material.clone()),
            transform,
        ));

        match &obj.kind {
            AssetKind::Tree(params) => {
                // A grass carpet filling the whole ground patch so the plant sits
                // in a continuous meadow (no bare-ground ring around the blades).
                let carpet = GrassFieldParams {
                    size_m: patch,
                    ..GrassFieldParams::default()
                };
                commands.spawn((
                    Mesh3d(meshes.add(build_grass_field_mesh(&carpet))),
                    MeshMaterial3d(grass_material.clone()),
                    transform,
                ));
                // The plant — on the shadow-caster layer so the same TreeMaterial
                // draw writes leaf-shaped depth into the sun-shadow cascade.
                commands.spawn((
                    Mesh3d(meshes.add(build_tree_mesh(params))),
                    MeshMaterial3d(tree_material.clone()),
                    transform,
                    RenderLayers::from_layers(&[0, CASTER_LAYER]),
                ));
            }
            AssetKind::TreeImpostor => {
                // Same grass meadow as the mesh tree, for an identical context.
                let carpet = GrassFieldParams {
                    size_m: patch,
                    ..GrassFieldParams::default()
                };
                commands.spawn((
                    Mesh3d(meshes.add(build_grass_field_mesh(&carpet))),
                    MeshMaterial3d(grass_material.clone()),
                    transform,
                ));
                // One impostor billboard for this tree at the object origin —
                // built exactly like a single-tree scatter tile; the material's
                // vertex shader billboards + sizes it from the per-species bounds.
                let inst = VegInstance {
                    species: 0,
                    root_offset_body_m: Vec3::ZERO,
                    up_body: Vec3::Y,
                    yaw: 0.0,
                    scale: 1.0,
                    tilt: 0.0,
                };
                if let Some(card) = combine_impostor_tile_mesh(&[inst], &[Some(0)], 1.0) {
                    commands.spawn((
                        Mesh3d(meshes.add(card)),
                        MeshMaterial3d(impostor_rig.material.clone()),
                        transform,
                    ));
                }
            }
            AssetKind::GrassClump(params) => {
                commands.spawn((
                    Mesh3d(meshes.add(build_grass_clump_mesh(params))),
                    MeshMaterial3d(grass_material.clone()),
                    transform,
                ));
            }
            AssetKind::GrassField(params) => {
                commands.spawn((
                    Mesh3d(meshes.add(build_grass_field_mesh(params))),
                    MeshMaterial3d(grass_material.clone()),
                    transform,
                ));
            }
            AssetKind::Rock(params) => {
                // A single stone on the bare ground patch (no grass, so the
                // shape reads), tagged to cast its own shadow.
                commands.spawn((
                    Mesh3d(meshes.add(build_rock_mesh(params))),
                    MeshMaterial3d(rock_material.clone()),
                    transform,
                    RenderLayers::from_layers(&[0, CASTER_LAYER]),
                ));
            }
            AssetKind::RockField => {
                // Grass carpet over the whole patch so the stones get partly
                // covered (the in-game look), plus the scattered-pebble mesh.
                let carpet = GrassFieldParams {
                    size_m: patch,
                    ..GrassFieldParams::default()
                };
                commands.spawn((
                    Mesh3d(meshes.add(build_grass_field_mesh(&carpet))),
                    MeshMaterial3d(grass_material.clone()),
                    transform,
                ));
                commands.spawn((
                    Mesh3d(meshes.add(rock_field_mesh())),
                    MeshMaterial3d(rock_material.clone()),
                    transform,
                    RenderLayers::from_layers(&[0, CASTER_LAYER]),
                ));
            }
        }
    }

    commands.insert_resource(DioramaMaterials {
        ground: ground_material,
        grass: grass_material,
        tree: tree_material,
        rock: rock_material,
    });
    commands.insert_resource(Scene { objects });
}

// ---------------------------------------------------------------------------
// Octahedral impostor bake (mesh↔impostor parity check)
// ---------------------------------------------------------------------------
//
// The far-LOD tree is a single billboard sampling a pre-baked hemisphere
// octahedral atlas of the species (the game's `tree_impostor` path). The atlas
// is rendered off-screen at startup by an orthographic camera over a grid of the
// recentred species mesh rotated to each captured view — a trimmed copy of the
// game's `spawn_impostor_bake_rig` for ONE species. Because the bake shader
// (`tree_bake.wgsl`) and the near shader (`tree.wgsl`) now both derive their leaf
// colour from the shared `thalos::foliage` material model, the impostor card and
// the mesh tree must read as the same tree — this harness makes that
// self-verifiable on every future tree/shrub change.

/// Captured views per octahedral axis + pixels per cell. Lighter than the game
/// (8×8 / 128 px) because this is a colour-parity check, not the shipped atlas:
/// the off-screen bake renders every cell's tree each frame until teardown, and
/// too many full-LOD trees per frame backs the GPU queue up into a device-wait
/// timeout on the headless exit drain. A 4×4 grid of the lighter [`BAKE_LOD`]
/// mesh stays well under that while giving a representative impostor.
const IMPOSTOR_CELLS: u32 = 4;
const IMPOSTOR_CELL_PX: u32 = 96;
/// Mesh LOD baked into the preview's parity atlas — LOD1 (not the game's LOD0) so
/// the per-frame off-screen vertex load stays low enough to avoid the exit-drain
/// timeout. Colour parity is per-fragment (atlas sample × shared
/// `foliage_base_albedo`), so it is LOD-independent; only the silhouette is a
/// touch sparser than the shipped LOD0 atlas.
const BAKE_LOD: u32 = 1;
const IMPOSTOR_ALPHA_CUTOFF: f32 = 0.35;
const IMPOSTOR_CELL_FILL: f32 = 0.84;
/// Dedicated off-screen render layers for the two bake passes (albedo, normal),
/// distinct from the main (0) and shadow-caster ([`CASTER_LAYER`]) layers so the
/// bake grid never bleeds into a captured shot.
const BAKE_ALBEDO_LAYER: usize = 6;
const BAKE_NORMAL_LAYER: usize = 7;
/// Frames the off-screen bake rig renders before teardown — long enough to cover
/// async pipeline compilation + fill the atlas, then despawned so it isn't
/// re-rendering 128 instances × 2 cameras every frame for the rest of the run
/// (which backs the GPU queue up into a device-wait timeout at exit). The atlas
/// Image persists after teardown, so impostors keep sampling it. Kept below the
/// first impostor capture's warmup so the heavy bake phase is short.
const IMPOSTOR_BAKE_FRAMES: u32 = 60;

/// The shared far-band impostor material the `TreeImpostor` objects billboard.
#[derive(Resource)]
struct ImpostorRig {
    material: Handle<TreeImpostorMaterial>,
}

/// Marker on every off-screen bake-rig entity (the 2·N² rotated instances + the
/// two cameras), so the whole rig can be torn down once the atlas is captured.
#[derive(Component)]
struct ImpostorBakeRig;

/// Render the bake rig for [`IMPOSTOR_BAKE_FRAMES`] frames (enough to compile the
/// off-screen pipelines and fill the atlas), then despawn it — the atlas keeps
/// its captured content. Without this the rig re-renders every frame and the
/// final device drain on exit times out.
fn teardown_impostor_bake(
    mut frames: Local<u32>,
    rig: Query<Entity, With<ImpostorBakeRig>>,
    mut commands: Commands,
) {
    *frames += 1;
    if *frames == IMPOSTOR_BAKE_FRAMES {
        for entity in &rig {
            commands.entity(entity).despawn();
        }
    }
}

/// Bake the broadleaf's hemisphere octahedral atlas off-screen and build the
/// shared [`TreeImpostorMaterial`]. Runs before `setup_scene`, which reads the
/// resulting [`ImpostorRig`]; the bake cameras keep the atlas filled while the
/// app runs (the scene is static, so re-rendering is harmless and avoids a
/// teardown system).
fn setup_impostor_bake(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut bake_materials: ResMut<Assets<TreeBakeMaterial>>,
    mut impostor_materials: ResMut<Assets<TreeImpostorMaterial>>,
) {
    let foliage_atlas = images.add(build_foliage_atlas());

    // The species captured into the atlas (the same broadleaf the mesh shows),
    // at the lighter `BAKE_LOD` to keep the per-frame off-screen load down.
    let data = build_tree_mesh_data(&TreeMeshParams {
        lod: BAKE_LOD,
        ..broadleaf()
    });
    let (center, radius) = tree_bounding_sphere(&data);

    let layout = ImpostorAtlasLayout {
        cells: IMPOSTOR_CELLS,
        cell_px: IMPOSTOR_CELL_PX,
        species: 1,
    };
    let albedo_atlas = images.add(make_impostor_atlas(layout));
    let normal_atlas = images.add(make_impostor_atlas(layout));

    let cell_fit = IMPOSTOR_CELL_FILL * 0.5;
    let depth_scale = 0.5 / cell_fit;
    let albedo_mat = bake_materials.add(TreeBakeMaterial {
        params: BakeParams {
            mode: Vec4::new(0.0, depth_scale, 0.0, 0.0),
        },
        atlas: foliage_atlas.clone(),
    });
    let normal_mat = bake_materials.add(TreeBakeMaterial {
        params: BakeParams {
            mode: Vec4::new(1.0, depth_scale, 0.0, 0.0),
        },
        atlas: foliage_atlas,
    });

    // One rotated copy of the recentred mesh per (i, j) view cell, on both bake
    // layers; the two cameras frame the whole N×N grid orthographically.
    let mesh = meshes.add(recenter_tree_mesh(&data, center));
    let scale = Vec3::splat(cell_fit / radius);
    let n = IMPOSTOR_CELLS;
    for j in 0..n {
        for i in 0..n {
            let uv = Vec2::new((i as f32 + 0.5) / n as f32, (j as f32 + 0.5) / n as f32);
            let rot = impostor_bake_rotation(hemioct_decode(uv));
            let cell_xy = Vec3::new(i as f32 + 0.5, j as f32 + 0.5, 0.0);
            let t = Transform {
                translation: cell_xy,
                rotation: rot,
                scale,
            };
            commands.spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(albedo_mat.clone()),
                t,
                Visibility::Visible,
                RenderLayers::layer(BAKE_ALBEDO_LAYER),
                ImpostorBakeRig,
                Name::new("Impostor Bake (albedo)"),
            ));
            commands.spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(normal_mat.clone()),
                t,
                Visibility::Visible,
                RenderLayers::layer(BAKE_NORMAL_LAYER),
                ImpostorBakeRig,
                Name::new("Impostor Bake (normal)"),
            ));
        }
    }

    let grid = n as f32;
    let cam_center = Vec3::new(grid * 0.5, grid * 0.5, 0.0);
    let bake_camera = |order: isize, layer: usize, target: Handle<Image>, name: &'static str| {
        (
            Camera3d::default(),
            Camera {
                order,
                clear_color: ClearColorConfig::Custom(Color::NONE),
                ..default()
            },
            Hdr,
            Tonemapping::None,
            RenderTarget::Image(ImageRenderTarget::from(target)),
            Projection::Orthographic(OrthographicProjection {
                scaling_mode: ScalingMode::Fixed {
                    width: grid,
                    height: grid,
                },
                near: 0.1,
                far: 100.0,
                ..OrthographicProjection::default_3d()
            }),
            Transform::from_translation(cam_center + Vec3::Z * 10.0).looking_at(cam_center, Vec3::Y),
            RenderLayers::layer(layer),
            ImpostorBakeRig,
            Name::new(name),
        )
    };
    commands.spawn(bake_camera(
        -20,
        BAKE_ALBEDO_LAYER,
        albedo_atlas.clone(),
        "Impostor Bake Camera (albedo)",
    ));
    commands.spawn(bake_camera(
        -19,
        BAKE_NORMAL_LAYER,
        normal_atlas.clone(),
        "Impostor Bake Camera (normal)",
    ));

    let mut species_geo = [Vec4::ZERO; IMPOSTOR_MAX_SPECIES];
    species_geo[0] = Vec4::new(radius, center.y, 0.0, 0.0);
    let material = impostor_materials.add(TreeImpostorMaterial {
        // Constant sun / full-size fade (the diorama doesn't move): the SAME
        // shared sky inputs the mesh trees use, so both light identically.
        params: veg_params(),
        impostor: ImpostorParams {
            grid: Vec4::new(n as f32, 1.0, IMPOSTOR_ALPHA_CUTOFF, 0.0),
            atlas: Vec4::new(IMPOSTOR_CELL_FILL, 0.0, 0.0, 0.0),
            species_geo,
        },
        albedo: albedo_atlas,
        normal: normal_atlas,
    });
    commands.insert_resource(ImpostorRig { material });
}

/// The game's space/surface post stack, minus the sensor-sim effects (film
/// grain + chromatic aberration) that only add noise to small asset thumbnails:
/// HDR + AgX filmic tonemap + subtle bloom + SMAA + CAS sharpening. Matches the
/// in-game tonemapping/exposure/AA so the preview reads like a real screenshot.
fn studio_camera_post_stack() -> impl Bundle {
    (
        Msaa::Off,
        Smaa {
            preset: SmaaPreset::High,
        },
        Hdr,
        // AgX: the same filmic tonemap the game's `space_camera_post_stack` uses.
        Tonemapping::AgX,
        DebandDither::Enabled,
        Bloom {
            intensity: 0.30,
            low_frequency_boost: 0.0,
            low_frequency_boost_curvature: 0.0,
            high_pass_frequency: 1.0,
            prefilter: BloomPrefilter {
                threshold: 0.6,
                threshold_softness: 0.3,
            },
            composite_mode: BloomCompositeMode::Additive,
            ..Bloom::NATURAL
        },
        ContrastAdaptiveSharpening {
            enabled: true,
            sharpening_strength: 0.3,
            denoise: false,
        },
    )
}

// ---------------------------------------------------------------------------
// Headless gallery mode
// ---------------------------------------------------------------------------

/// Drives the capture timeline: which object is framed, when each is shot.
#[derive(Resource)]
struct Capture {
    frame: u32,
    target: Handle<Image>,
}

#[derive(Component)]
struct PreviewCamera;

fn setup_headless_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    scene: Res<Scene>,
) {
    // Off-screen render target the camera draws into and the screenshot reads.
    let mut target = Image::new_fill(
        Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    target.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC | TextureUsages::RENDER_ATTACHMENT;
    let target = images.add(target);

    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(SKY),
            order: 0,
            ..default()
        },
        RenderTarget::Image(ImageRenderTarget::from(target.clone())),
        studio_camera_post_stack(),
        frame_transform(&scene.objects[0], 0),
        PreviewCamera,
    ));

    commands.insert_resource(Capture { frame: 0, target });
}

/// Timeline: warm up, then frame + capture each object in turn, then exit.
fn drive_capture(
    mut commands: Commands,
    mut cap: ResMut<Capture>,
    scene: Res<Scene>,
    mut focus: ResMut<ShadowFocus>,
    mut cam: Query<&mut Transform, With<PreviewCamera>>,
    mut exit: MessageWriter<AppExit>,
) {
    cap.frame += 1;
    let n = scene.objects.len() as u32;

    for i in 0..n {
        let pose = WARMUP + i * DWELL;
        let shot = pose + 3;
        if cap.frame == pose
            && let Ok(mut t) = cam.single_mut()
        {
            *t = frame_transform(&scene.objects[i as usize], i as usize);
            // Re-aim the sun-shadow cascade over this object's ground.
            focus.center = Vec3::new(i as f32 * OBJECT_SPACING, 1.0, 0.0);
        }
        if cap.frame == shot {
            let path = format!("{OUT_DIR}/{}.png", scene.objects[i as usize].name);
            commands
                .spawn(Screenshot::image(cap.target.clone()))
                .observe(save_to_disk(path.clone()));
            info!("captured {path}");
        }
    }

    if cap.frame >= WARMUP + (n.saturating_sub(1)) * DWELL + 3 + TAIL {
        exit.write(AppExit::Success);
    }
}

// ---------------------------------------------------------------------------
// Interactive window mode
// ---------------------------------------------------------------------------

/// Orbit-camera state: a focus point with yaw/pitch/distance, plus which object
/// is currently centred.
#[derive(Component)]
struct OrbitCamera {
    focus: Vec3,
    yaw: f32,
    pitch: f32,
    distance: f32,
    current: usize,
}

fn apply_orbit(t: &mut Transform, o: &OrbitCamera) {
    let rot = Quat::from_euler(EulerRot::YXZ, o.yaw, o.pitch, 0.0);
    let eye = o.focus + rot * Vec3::new(0.0, 0.0, o.distance);
    *t = Transform::from_translation(eye).looking_at(o.focus, Vec3::Y);
}

fn setup_window_camera(mut commands: Commands, scene: Res<Scene>) {
    let orbit = OrbitCamera {
        focus: object_focus(&scene.objects, 0),
        yaw: 0.6,
        pitch: -0.28,
        distance: scene.objects[0].distance,
        current: 0,
    };
    let mut transform = Transform::default();
    apply_orbit(&mut transform, &orbit);

    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(SKY),
            order: 0,
            ..default()
        },
        studio_camera_post_stack(),
        transform,
        orbit,
    ));

    info!(
        "Object preview — drag: orbit · scroll: zoom · ←/→: cycle object · S: screenshot → {OUT_DIR}/"
    );
}

fn orbit_camera(
    btn: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    scene: Res<Scene>,
    mut focus: ResMut<ShadowFocus>,
    mut q: Query<(&mut Transform, &mut OrbitCamera)>,
) {
    let Ok((mut t, mut o)) = q.single_mut() else {
        return;
    };

    if btn.pressed(MouseButton::Left) {
        o.yaw -= motion.delta.x * 0.008;
        o.pitch = (o.pitch - motion.delta.y * 0.008).clamp(-1.45, 1.45);
    }
    if scroll.delta.y != 0.0 {
        o.distance = (o.distance * (1.0 - scroll.delta.y * 0.12)).clamp(0.5, 250.0);
    }

    let n = scene.objects.len();
    let mut pick = None;
    if keys.just_pressed(KeyCode::ArrowRight) {
        pick = Some((o.current + 1) % n);
    } else if keys.just_pressed(KeyCode::ArrowLeft) {
        pick = Some((o.current + n - 1) % n);
    }
    if let Some(i) = pick {
        o.current = i;
        o.focus = object_focus(&scene.objects, i);
        o.distance = scene.objects[i].distance;
    }

    // Keep the sun-shadow cascade over whichever object is centred.
    focus.center = Vec3::new(o.current as f32 * OBJECT_SPACING, 1.0, 0.0);
    apply_orbit(&mut t, &o);
}

fn screenshot_key(
    keys: Res<ButtonInput<KeyCode>>,
    scene: Res<Scene>,
    cam: Query<&OrbitCamera>,
    mut commands: Commands,
) {
    if !keys.just_pressed(KeyCode::KeyS) {
        return;
    }
    // Name the shot after whichever object is centred, with a "_view" suffix so it
    // doesn't clobber the headless gallery PNGs.
    let name = cam
        .single()
        .ok()
        .map(|o| scene.objects[o.current].name)
        .unwrap_or("view");
    let path = format!("{OUT_DIR}/{name}_view.png");
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path.clone()));
    info!("saved {path}");
}

// ---------------------------------------------------------------------------
// Sun-shadow rig (single cascade)
// ---------------------------------------------------------------------------
//
// A trimmed copy of `thalos_game::rendering::sun_shadow`: an orthographic camera
// on the caster layer aimed down the sun over the current object renders the
// trees' leaf-shaped depth, a render-graph node copies that depth into a
// sample-able map, and `update_preview_shadow` publishes the cascade transform +
// strength onto the ground + tree materials (which sample it exactly as in game).
// One cascade is enough for a single-object diorama; cascades 1 + 2 are parked
// (zero view-proj → the shader's sentinel skips them).

/// Render layer the caster (tree) meshes + the sun-shadow camera share.
const CASTER_LAYER: usize = 8;
/// Cascade depth-map resolution; ~0.02 m/texel over the 22 m half-extent box.
const SHADOW_MAP_SIZE: u32 = 2048;
const SHADOW_HALF_EXTENT_M: f32 = 22.0;
const SHADOW_NEAR_M: f32 = 0.1;
const SHADOW_FAR_M: f32 = 220.0;
/// How far back along the sun the ortho camera sits above the object.
const SHADOW_BACK_M: f32 = 90.0;
/// Depth-compare bias in metres (orthographic z is linear).
const SHADOW_BIAS_M: f32 = 0.18;
/// Darkening strength (0 = off, 1 = black).
const SHADOW_STRENGTH: f32 = 0.7;

/// Depth map the cascade camera's depth is copied into. Extracted to the render
/// world for [`CopyShadowDepthNode`]; the same handle is bound on the materials.
#[derive(Resource, Clone, ExtractResource)]
struct ShadowImage {
    handle: Handle<Image>,
}

/// Marker on the orthographic sun-shadow camera (extracted so the copy node
/// runs for its view).
#[derive(Component, Clone, Copy, ExtractComponent)]
struct ShadowCascadeCam;

/// Main-world shadow state: the live cascade block + the depth-map handles bound
/// on materials. **Sole writer:** [`update_preview_shadow`].
#[derive(Resource)]
struct ShadowRig {
    image: Handle<Image>,
    fallback: Handle<Image>,
    block: ShadowCascadeBlock,
}

/// Where the cascade is centred (the framed object's ground point). Written by
/// the capture / orbit drivers, read by [`update_preview_shadow`].
#[derive(Resource, Default)]
struct ShadowFocus {
    center: Vec3,
}

/// Copy the shadow cascade's rendered depth into the shadow map. Ported from the
/// former `CopyShadowDepthNode` (`ViewNode`) to a Bevy 0.19 render-pass system;
/// the `ViewQuery` filters to the cascade view and auto-skips the main camera.
fn copy_shadow_depth(
    view: ViewQuery<(&'static ViewDepthTexture, &'static ShadowCascadeCam)>,
    shadow: Option<Res<ShadowImage>>,
    render_assets: Res<RenderAssets<GpuImage>>,
    mut ctx: RenderContext,
) {
    let (depth, _cam) = view.into_inner();

    let Some(shadow) = shadow else {
        return;
    };
    let Some(dest) = render_assets.get(&shadow.handle) else {
        return;
    };
    let src_size = depth.texture.size();
    let dst_size = dest.texture.size();
    if src_size.width != dst_size.width
        || src_size.height != dst_size.height
        || depth.texture.sample_count() != dest.texture.sample_count()
    {
        return;
    }
    ctx.command_encoder().copy_texture_to_texture(
        depth.texture.as_image_copy(),
        dest.texture.as_image_copy(),
        src_size,
    );
}

struct PreviewShadowPlugin;

impl Plugin for PreviewShadowPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<ShadowImage>::default())
            .add_plugins(ExtractComponentPlugin::<ShadowCascadeCam>::default())
            .init_resource::<ShadowFocus>()
            .add_systems(Startup, setup_shadow_rig);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(
                Core3d,
                copy_shadow_depth
                    .in_set(Core3dSystems::MainPass)
                    .after(main_opaque_pass_3d)
                    .before(main_transparent_pass_3d),
            );
        }
    }
}

fn setup_shadow_rig(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    // Sample-able depth map the camera's depth is copied into.
    let mut depth = Image::new_uninit(
        Extent3d {
            width: SHADOW_MAP_SIZE,
            height: SHADOW_MAP_SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Depth32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    depth.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    let depth_handle = images.add(depth);

    // The cascade camera needs a colour attachment (only the depth is read).
    let mut color = Image::new_uninit(
        Extent3d {
            width: SHADOW_MAP_SIZE,
            height: SHADOW_MAP_SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    color.texture_descriptor.usage =
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING;
    let color_handle = images.add(color);

    commands.spawn((
        Camera3d {
            // COPY_SRC so the node can copy this camera's depth into its map.
            depth_texture_usages: (TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC)
                .into(),
            ..default()
        },
        Camera {
            // Render before the main view (order 0) so the copied depth is ready.
            order: -1,
            clear_color: ClearColorConfig::Custom(Color::NONE),
            ..default()
        },
        RenderTarget::Image(ImageRenderTarget::from(color_handle)),
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::Fixed {
                width: SHADOW_HALF_EXTENT_M * 2.0,
                height: SHADOW_HALF_EXTENT_M * 2.0,
            },
            near: SHADOW_NEAR_M,
            far: SHADOW_FAR_M,
            ..OrthographicProjection::default_3d()
        }),
        Msaa::Off,
        RenderLayers::layer(CASTER_LAYER),
        ShadowCascadeCam,
        Name::new("Preview Sun Shadow"),
    ));

    commands.insert_resource(ShadowImage {
        handle: depth_handle.clone(),
    });
    commands.insert_resource(ShadowRig {
        image: depth_handle,
        fallback: images.add(fallback_shadow_map()),
        block: ShadowCascadeBlock::default(),
    });
}

/// Bevy reverse-z orthographic clip matrix (near/far swapped, as
/// `OrthographicProjection::get_clip_from_view` does), built by hand so it stays
/// in lockstep with the camera regardless of system ordering.
fn cascade_clip_from_view(half: f32, far: f32) -> Mat4 {
    Mat4::orthographic_rh(-half, half, -half, half, far, SHADOW_NEAR_M)
}

/// Aim the cascade camera down the sun over the framed object, publish its
/// transform + strength, and bind the live block + depth maps onto the ground +
/// tree materials. **Sole writer** of [`ShadowRig`].
fn update_preview_shadow(
    focus: Res<ShadowFocus>,
    rig: Option<ResMut<ShadowRig>>,
    mats: Option<Res<DioramaMaterials>>,
    mut cam: Query<(&mut Transform, &mut Camera), With<ShadowCascadeCam>>,
    mut ground_materials: ResMut<Assets<GroundPatchMaterial>>,
    mut grass_materials: ResMut<Assets<GrassMaterial>>,
    mut tree_materials: ResMut<Assets<TreeMaterial>>,
    mut rock_materials: ResMut<Assets<RockMaterial>>,
) {
    let (Some(mut rig), Some(mats)) = (rig, mats) else {
        return;
    };

    let sun_dir = preview_sun_dir();
    let center = focus.center;
    let eye = center + sun_dir * SHADOW_BACK_M;
    let up = if sun_dir.dot(Vec3::Y).abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let look = Transform::from_translation(eye).looking_at(center, up);

    if let Ok((mut tf, mut camera)) = cam.single_mut() {
        *tf = look;
        camera.is_active = true;
    }

    let view = look.to_matrix().inverse();
    let mut block = ShadowCascadeBlock::default();
    block.view_proj[0] = cascade_clip_from_view(SHADOW_HALF_EXTENT_M, SHADOW_FAR_M) * view;
    // Orthographic z is linear → clip-space bias = metres / (far − near).
    block.params[0] = Vec4::new(SHADOW_BIAS_M / (SHADOW_FAR_M - SHADOW_NEAR_M), 0.0, 0.0, 0.0);
    // Park the unused cascades: a zero view-proj makes `clip.w <= 0`, so the
    // shader's `cascade_factor` returns its skip sentinel and never samples them.
    block.view_proj[1] = Mat4::ZERO;
    block.view_proj[2] = Mat4::ZERO;
    block.gate = Vec4::new(SHADOW_STRENGTH, 1.0, 0.0, 0.0);
    rig.block = block;

    let (image, fallback) = (rig.image.clone(), rig.fallback.clone());
    if let Some(mut m) = ground_materials.get_mut(&mats.ground) {
        m.shadow = block;
        m.sun_shadow_map_0 = image.clone();
        m.sun_shadow_map_1 = fallback.clone();
        m.sun_shadow_map_2 = fallback.clone();
    }
    if let Some(mut m) = grass_materials.get_mut(&mats.grass) {
        m.shadow = block;
        m.sun_shadow_map_0 = image.clone();
        m.sun_shadow_map_1 = fallback.clone();
        m.sun_shadow_map_2 = fallback.clone();
    }
    if let Some(mut m) = tree_materials.get_mut(&mats.tree) {
        m.shadow = block;
        m.sun_shadow_map_0 = image.clone();
        m.sun_shadow_map_1 = fallback.clone();
        m.sun_shadow_map_2 = fallback.clone();
    }
    if let Some(mut m) = rock_materials.get_mut(&mats.rock) {
        m.shadow = block;
        m.sun_shadow_map_0 = image;
        m.sun_shadow_map_1 = fallback.clone();
        m.sun_shadow_map_2 = fallback;
    }
}
