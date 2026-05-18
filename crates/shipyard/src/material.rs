use bevy::pbr::{ExtendedMaterial, MaterialExtension, StandardMaterial};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

/// Full material type for procedurally-detailed ship parts. An
/// [`ExtendedMaterial`] so the base `StandardMaterial` keeps driving PBR
/// lighting, shadows, and tone mapping — we only author the procedural
/// layer (panels, rivets, tint) in the fragment shader.
pub type ShipPartMaterial = ExtendedMaterial<StandardMaterial, ShipPartExtension>;

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

#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
pub struct ShipPartExtension {
    #[uniform(100)]
    pub params: ShipPartParams,
}

impl Default for ShipPartExtension {
    fn default() -> Self {
        Self {
            params: ShipPartParams::default(),
        }
    }
}

impl MaterialExtension for ShipPartExtension {
    fn fragment_shader() -> ShaderRef {
        "shaders/ship_part.wgsl".into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        "shaders/ship_part.wgsl".into()
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
