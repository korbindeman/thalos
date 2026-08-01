//! Navball custom material — textured sphere with limb darkening.

use bevy::pbr::Material;
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

/// Per-frame uniform driving the navball's surface shading.
#[derive(Clone, Copy, ShaderType)]
pub struct NavballParams {
    /// Higher gamma → darkening pinches harder toward the limb.
    pub limb_darkening_gamma: f32,
    /// Brightness at the silhouette edge (cos θ = 0). 0 = pitch black limb.
    pub limb_floor: f32,
    /// Strength of the lens-style UV bend at the edges. 0 = none.
    pub edge_distortion: f32,
    pub _pad: f32,
}

impl Default for NavballParams {
    fn default() -> Self {
        Self {
            limb_darkening_gamma: 1.6,
            limb_floor: 0.18,
            edge_distortion: 0.0,
            _pad: 0.0,
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct NavballMaterial {
    #[uniform(0)]
    pub params: NavballParams,
    #[texture(1)]
    #[sampler(2)]
    pub texture: Handle<Image>,
}

impl Material for NavballMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/navball.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "shaders/navball.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
}
