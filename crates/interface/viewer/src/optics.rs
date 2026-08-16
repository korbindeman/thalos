use bevy::prelude::*;
use thalos_render_model::{
    CameraOptics as CameraOpticsSpec, CapturedCameraState, MAX_FOCAL_LENGTH_MM, MIN_FOCAL_LENGTH_MM,
};

use crate::ViewerCamera;

/// Physical framing authority attached to the viewed camera.
#[derive(Component, Clone, Copy, Debug)]
pub struct CameraOptics {
    spec: CameraOpticsSpec,
    zoom_multiplier: f32,
}

impl Default for CameraOptics {
    fn default() -> Self {
        Self {
            spec: CameraOpticsSpec::default(),
            zoom_multiplier: 1.0,
        }
    }
}

impl CameraOptics {
    pub fn from_spec(spec: CameraOpticsSpec) -> Result<Self, String> {
        spec.validate()?;
        Ok(Self {
            spec,
            zoom_multiplier: 1.0,
        })
    }

    pub fn spec(&self) -> CameraOpticsSpec {
        self.spec
    }

    pub fn set_spec(&mut self, spec: CameraOpticsSpec) -> Result<(), String> {
        spec.validate()?;
        self.spec = spec;
        self.zoom_multiplier = 1.0;
        Ok(())
    }

    pub fn base_focal_length_mm(&self) -> f32 {
        self.spec.lens.focal_length_mm
    }

    pub fn set_base_focal_length_mm(&mut self, focal_length_mm: f32) {
        self.spec.lens.focal_length_mm =
            focal_length_mm.clamp(MIN_FOCAL_LENGTH_MM, MAX_FOCAL_LENGTH_MM);
    }

    pub fn zoom_multiplier(&self) -> f32 {
        self.zoom_multiplier
    }

    pub fn set_zoom_multiplier(&mut self, multiplier: f32) {
        self.zoom_multiplier = multiplier.max(1.0);
    }

    pub fn effective_focal_length_mm(&self) -> f32 {
        self.base_focal_length_mm() * self.zoom_multiplier
    }

    pub fn horizontal_fov_rad(&self) -> f32 {
        2.0 * (self.spec.sensor.gate_width_mm / (2.0 * self.effective_focal_length_mm())).atan()
    }

    pub fn vertical_fov_rad(&self) -> f32 {
        let aspect = self.spec.sensor.aspect[0] as f32 / self.spec.sensor.aspect[1] as f32;
        2.0 * ((self.horizontal_fov_rad() * 0.5).tan() / aspect).atan()
    }

    pub fn captured_state(&self, output_extent: [u32; 2]) -> CapturedCameraState {
        CapturedCameraState {
            optics: self.spec,
            effective_focal_length_mm: self.effective_focal_length_mm(),
            derived_vertical_fov_rad: self.vertical_fov_rad(),
            output_extent,
        }
    }

    pub fn apply_to_projection(&self, projection: &mut Projection) {
        let Projection::Perspective(perspective) = projection else {
            return;
        };
        let target = self.vertical_fov_rad();
        if (perspective.fov - target).abs() > 1.0e-6 {
            perspective.fov = target;
        }
    }
}

pub fn sync_camera_optics_projection(
    mut cameras: Query<(&CameraOptics, &mut Projection), With<ViewerCamera>>,
) {
    for (optics, mut projection) in &mut cameras {
        optics.apply_to_projection(&mut projection);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spring_zoom_multiplies_focal_length_not_fov() {
        let spec = CameraOpticsSpec::new(50.0, [16, 9]).unwrap();
        let mut optics = CameraOptics::from_spec(spec).unwrap();
        optics.set_zoom_multiplier(4.0);
        assert_eq!(optics.effective_focal_length_mm(), 200.0);
        let expected = CameraOpticsSpec::new(200.0, [16, 9])
            .unwrap()
            .vertical_fov_rad();
        assert!((optics.vertical_fov_rad() - expected).abs() < 1.0e-6);
    }
}
