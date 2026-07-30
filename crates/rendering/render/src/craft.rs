//! Craft (ship hull) surface material.
//!
//! Moved here from `thalos_shipyard::material` so the render layer owns *how a
//! craft surface looks* while `thalos_shipyard` owns *what the craft is*
//! (definition + construction). This is the first cut of the shipyard
//! rendering decoupling (`docs/architecture.md` Phase 4a): the type lives in
//! the render crate, re-exported through `thalos_shipyard` for now so the editor
//! core keeps compiling — a follow-up moves the material *application* out of the
//! editor core and flips the interim `shipyard → body_render` dependency to the
//! clean `body_render → shipyard` direction.
//!
//! Living in the render crate is what unblocks the craft hull *receiving* the
//! shared sun-shadow cascade (graphics-fidelity F6b/F7): `thalos::shadow` and the
//! metallic BRDF branch are right here.

use bevy::asset::RenderAssetUsages;
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, StandardMaterial};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::shader::ShaderRef;

use crate::{CASCADE_COUNT, ShadowCascadeBlock};
use thalos_shipyard::{Adapter, AttachNodes, Decoupler, FuelTank, Fuselage};

/// Full material type for procedurally-detailed ship parts. An
/// [`ExtendedMaterial`] so the base `StandardMaterial` keeps driving PBR
/// lighting, shadows, and tone mapping — we only author the procedural
/// layer (panels, rivets, tint) in the fragment shader.
pub type ShipPartMaterial = ExtendedMaterial<StandardMaterial, ShipPartExtension>;

/// Project construction dimensions into the render material's per-part
/// uniform. Construction owns the inputs; rendering owns this projection and
/// its output type.
pub fn ship_part_params(
    nodes: &AttachNodes,
    tank: Option<&FuelTank>,
    fuselage: Option<&Fuselage>,
    dec: Option<&Decoupler>,
    adapter: Option<&Adapter>,
    seed: u32,
) -> ShipPartParams {
    let top_r = nodes
        .get("top")
        .map(|node| node.diameter * 0.5)
        .unwrap_or(0.5);
    let (radius_top, radius_bottom, length) = if let Some(tank) = tank {
        (top_r, top_r, tank.length)
    } else if let Some(fuselage) = fuselage {
        (top_r, top_r, fuselage.length)
    } else if dec.is_some() {
        (top_r, top_r, 0.2)
    } else if let Some(adapter) = adapter {
        let bottom_r = adapter.target_diameter * 0.5;
        let height = (top_r + bottom_r).max(0.4);
        let delta_r = top_r - bottom_r;
        (
            top_r,
            bottom_r,
            (height * height + delta_r * delta_r).sqrt(),
        )
    } else {
        (top_r, top_r, 1.0)
    };
    ShipPartParams {
        length,
        radius_top,
        radius_bottom,
        seed,
        ..Default::default()
    }
}

/// Per-part uniform block. One instance per part entity so each tank can
/// have its own length/radius/seed and an independently-driven `tint`
/// (used by the editor's selection / hover highlight).
#[derive(Clone, ShaderType, Debug)]
pub struct ShipPartParams {
    /// Surface "axial" length in meters — distance along the mesh from
    /// v=0 to v=1. For a cylinder this is the vertical height; for a
    /// conical frustum it is the *slant* length,
    /// `sqrt(height² + (radius_top − radius_bottom)²)`.
    pub length: f32,
    /// Radius at the mesh's +Y end (UV v = 1), in meters. Equal to
    /// [`Self::radius_bottom`] for cylinders.
    pub radius_top: f32,
    /// Target axial pitch between panel seams, meters. The shader rounds
    /// to an integer number of panels across `length` so seams always
    /// land flush with the end caps.
    pub panel_pitch: f32,
    /// Target circumferential spacing between rivets, meters. Rounded to
    /// an integer count around the circumference so the pattern closes
    /// seamlessly.
    pub rivet_spacing: f32,
    /// Base-color multiplier. Identity white = neutral; the editor
    /// drives selection / hover tints through this without touching the
    /// base `StandardMaterial`.
    pub tint: Vec3,
    /// Rivet bump height (meters). Realistic 1–3mm looks natural on 1–3m
    /// diameter tanks.
    pub rivet_height: f32,
    /// Rivet footprint radius (meters). Controls both the size of the
    /// circular dome and the area where the normal is perturbed.
    pub rivet_radius: f32,
    /// Panel seam groove depth (meters). Shallow — about half a rivet is
    /// typical.
    pub seam_depth: f32,
    /// Panel seam half-width (meters). The groove softens from 0 at its
    /// edge to `seam_depth` at the center.
    pub seam_half_width: f32,
    /// Per-tank hash seed for subtle color / roughness noise. Keeps two
    /// tanks of identical dimensions from looking copy-pasted.
    pub seed: u32,
    /// Axial distance from each seam to the paired ring of rivets that
    /// brackets it (meters). Two rings per seam — one above, one below —
    /// mirroring how skin plates are riveted to bulkhead ring stringers.
    pub rivet_seam_offset: f32,
    /// Additional rivet rings interior to each panel, evenly distributed
    /// between the paired seam rings. 0 = paired only; 1 = one mid-panel
    /// ring; etc.
    pub rivet_mid_rows: u32,
    /// Radius at the mesh's -Y end (UV v = 0), in meters. Differs from
    /// [`Self::radius_top`] for conical frustums.
    pub radius_bottom: f32,
    pub _pad0: f32,
}

impl Default for ShipPartParams {
    fn default() -> Self {
        Self {
            length: 1.0,
            radius_top: 0.5,
            panel_pitch: 1.0,
            rivet_spacing: 0.08,
            tint: Vec3::ONE,
            rivet_height: 0.0025,
            rivet_radius: 0.006,
            seam_depth: 0.0015,
            seam_half_width: 0.006,
            seed: 0,
            rivet_seam_offset: 0.035,
            rivet_mid_rows: 1,
            radius_bottom: 0.5,
            _pad0: 0.0,
        }
    }
}

#[derive(Asset, AsBindGroup, TypePath, Clone, Debug, Default)]
pub struct ShipPartExtension {
    #[uniform(100)]
    pub params: ShipPartParams,
    /// Cascaded sun-shadow transforms + per-cascade compare params, so the hull
    /// RECEIVES the same shadow cascade the terrain/trees cast into (graphics-
    /// fidelity F6b). Driven each frame from the game's `SunShadowState` via
    /// [`CraftShadowMaps`]; `gate.x == 0` (the default, and any off-surface frame)
    /// makes `sun_shadow_factor` skip sampling entirely.
    #[uniform(101)]
    pub shadow: ShadowCascadeBlock,
    /// Per-cascade sun-shadow depth maps (near→far) — the same handles the terrain
    /// binds. Plain `texture_depth_2d` (no array). Always bound to a valid texture
    /// (a 1×1 fallback where no live cascade is pushed — the standalone editor,
    /// freshly-created parts) so the depth `sample_type` slot is never empty.
    #[texture(102, sample_type = "depth")]
    pub sun_shadow_map_0: Handle<Image>,
    #[texture(103, sample_type = "depth")]
    pub sun_shadow_map_1: Handle<Image>,
    #[texture(104, sample_type = "depth")]
    pub sun_shadow_map_2: Handle<Image>,
}

impl MaterialExtension for ShipPartExtension {
    fn fragment_shader() -> ShaderRef {
        "shaders/ship_part.wgsl".into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        "shaders/ship_part.wgsl".into()
    }
}

/// Generic sun-shadow-receiving `StandardMaterial`: stock PBR plus the shared
/// `thalos::shadow` cascade receive, with no procedural detail layer.
///
/// This is what closes the F6 "structures receive" gap: every surface-world
/// mesh that used a bare `StandardMaterial` (base buildings / pads / tanks,
/// the runway paving + posts, the tarmac, plain craft parts — pods, engines,
/// wings, nacelles, gear — and the EVA capsule) uses this instead, so it
/// receives the SAME cascade the terrain / trees / craft cast into. With that
/// in place the stock Bevy CSM on the sun light is disabled entirely — one
/// shadow world.
pub type ShadowedStandardMaterial = ExtendedMaterial<StandardMaterial, ShadowReceiveExtension>;

/// The receive-only extension behind [`ShadowedStandardMaterial`]. Bindings
/// mirror [`ShipPartExtension`]'s shadow half (uniform 100, depth maps
/// 101–103); driven per-frame from [`CraftShadowMaps`] by
/// [`apply_craft_shadow`], exactly like the hull.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct ShadowReceiveExtension {
    #[uniform(100)]
    pub shadow: ShadowCascadeBlock,
    #[texture(101, sample_type = "depth")]
    pub sun_shadow_map_0: Handle<Image>,
    #[texture(102, sample_type = "depth")]
    pub sun_shadow_map_1: Handle<Image>,
    #[texture(103, sample_type = "depth")]
    pub sun_shadow_map_2: Handle<Image>,
}

impl MaterialExtension for ShadowReceiveExtension {
    fn fragment_shader() -> ShaderRef {
        "shaders/shadowed_standard.wgsl".into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        "shaders/shadowed_standard.wgsl".into()
    }
}

/// Wrap a base `StandardMaterial` into the shadow-receiving material with the
/// default (fallback) shadow state — `apply_craft_shadow` patches the live
/// cascade in each frame.
pub fn shadowed(base: StandardMaterial) -> ShadowedStandardMaterial {
    ShadowedStandardMaterial {
        base,
        extension: ShadowReceiveExtension::default(),
    }
}

// ── Craft render plugin + sun-shadow receiving ────────────────────────────────

/// Live sun-shadow cascade for craft materials — the hull/gear *receiving* side.
/// Mirror of the game's `SunShadowState`: the game copies that resource into this
/// one each frame, and [`apply_craft_shadow`] fans it out onto every
/// [`ShipPartMaterial`]. In a binary with no shadow rig (the standalone ship
/// editor) it keeps the 1×1 depth fallback + a zeroed (`gate.x == 0`) block, so
/// hulls render unshadowed without an empty depth binding.
#[derive(Resource, Clone)]
pub struct CraftShadowMaps {
    pub images: [Handle<Image>; CASCADE_COUNT],
    pub block: ShadowCascadeBlock,
}

impl Default for CraftShadowMaps {
    fn default() -> Self {
        Self {
            images: std::array::from_fn(|_| Handle::default()),
            block: ShadowCascadeBlock::default(),
        }
    }
}

/// Registers the craft hull material and wires it to *receive* the shared
/// sun-shadow cascade. Added once per binary (via `ShipyardPlugin`); defensively
/// pulls in [`crate::shading::PlanetLightingPlugin`] so the hull shader's
/// `thalos::shadow` import resolves even in the standalone editor.
pub struct CraftRenderPlugin;

impl Plugin for CraftRenderPlugin {
    fn build(&self, app: &mut App) {
        // The hull shader `#import`s `thalos::shadow`; make sure that library is
        // registered even where the full BodyRenderPlugin isn't added (standalone
        // editor). No-op if already present — mirrors the other sub-plugins.
        if !app.is_plugin_added::<crate::shading::PlanetLightingPlugin>() {
            app.add_plugins(crate::shading::PlanetLightingPlugin);
        }
        app.add_plugins(MaterialPlugin::<ShipPartMaterial>::default())
            .add_plugins(MaterialPlugin::<ShadowedStandardMaterial>::default())
            .init_resource::<CraftShadowMaps>()
            .add_systems(Startup, setup_craft_shadow_fallback)
            .add_systems(Last, apply_craft_shadow);
    }
}

/// Create the 1×1 `Depth32Float` fallback (mirrors the rig's `make_depth_image`)
/// and seed [`CraftShadowMaps`] with it, so every craft material has a valid depth
/// binding before any live cascade is pushed. `gate.x == 0` means it is never
/// actually sampled.
fn setup_craft_shadow_fallback(
    mut images: ResMut<Assets<Image>>,
    mut maps: ResMut<CraftShadowMaps>,
) {
    let mut depth = Image::new_uninit(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Depth32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    depth.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING;
    let handle = images.add(depth);
    maps.images = std::array::from_fn(|_| handle.clone());
}

/// Fan the current [`CraftShadowMaps`] (real cascade in-game, fallback elsewhere)
/// onto every craft material AND every generic [`ShadowedStandardMaterial`]
/// (structures, runway, plain parts) so they all receive the shared sun-shadow
/// cascade. Per-frame, like the terrain material update.
fn apply_craft_shadow(
    maps: Res<CraftShadowMaps>,
    mut materials: ResMut<Assets<ShipPartMaterial>>,
    mut shadowed_materials: ResMut<Assets<ShadowedStandardMaterial>>,
    // Option: the tile-terrain material is registered by `TileTerrainPlugin`
    // (BodyRenderPlugin apps); `CraftRenderPlugin`-only apps (the standalone
    // editor) don't have it.
    tile_materials: Option<ResMut<Assets<crate::tiles::material::TileTerrainMaterial>>>,
) {
    for (_, mat) in materials.iter_mut() {
        let ext = &mut mat.extension;
        ext.shadow = maps.block;
        ext.sun_shadow_map_0 = maps.images[0].clone();
        ext.sun_shadow_map_1 = maps.images[1].clone();
        ext.sun_shadow_map_2 = maps.images[2].clone();
    }
    for (_, mat) in shadowed_materials.iter_mut() {
        let ext = &mut mat.extension;
        ext.shadow = maps.block;
        ext.sun_shadow_map_0 = maps.images[0].clone();
        ext.sun_shadow_map_1 = maps.images[1].clone();
        ext.sun_shadow_map_2 = maps.images[2].clone();
    }
    if let Some(mut tile_materials) = tile_materials {
        for (_, mat) in tile_materials.iter_mut() {
            let ext = &mut mat.extension;
            ext.shadow = maps.block;
            ext.sun_shadow_map_0 = maps.images[0].clone();
            ext.sun_shadow_map_1 = maps.images[1].clone();
            ext.sun_shadow_map_2 = maps.images[2].clone();
        }
    }
}

/// Construct a stainless-steel base `StandardMaterial` tuned to mate
/// with `ShipPartExtension`. The extension's WGSL modulates these values
/// in panel seams and adds normal-map perturbation; the base sets the
/// overall metal / roughness response.
pub fn stainless_steel_base() -> StandardMaterial {
    // Mirror-polish stainless: near-zero roughness so the specular lobe is
    // sharp, metallic 1.0 so the base colour doubles as the F0 reflectance
    // tint. At this roughness the panels are essentially mirrors and will
    // read mostly as the lighting environment — the shader still pushes
    // roughness up inside seams so the welds stay visible.
    StandardMaterial {
        base_color: Color::srgb(0.82, 0.84, 0.87),
        metallic: 1.0,
        perceptual_roughness: 0.08,
        reflectance: 0.5,
        ..default()
    }
}

/// Matte dark finish for landing gear (struts + wheels). Deliberately *not*
/// the mirror-polish stainless of the hull: low metallic and high roughness so
/// the gear reads as painted oleo struts / rubber tyres rather than steel.
pub fn landing_gear_base() -> StandardMaterial {
    StandardMaterial {
        base_color: Color::srgb(0.12, 0.12, 0.13),
        metallic: 0.1,
        perceptual_roughness: 0.85,
        ..default()
    }
}
