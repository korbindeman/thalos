//! Portable physical-camera optics shared by interactive viewers and capture.

use serde::{Deserialize, Serialize};

pub const FULL_FRAME_GATE_WIDTH_MM: f32 = 36.0;
pub const MIN_FOCAL_LENGTH_MM: f32 = 12.0;
pub const MAX_FOCAL_LENGTH_MM: f32 = 400.0;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CameraLensModel {
    #[default]
    FullFrameHorizontal,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct CameraLens {
    pub model: CameraLensModel,
    pub focal_length_mm: f32,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SensorCrop {
    #[default]
    Full,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct CameraSensor {
    pub gate_width_mm: f32,
    /// Reduced sensor-window aspect ratio, never an output pixel extent.
    pub aspect: [u32; 2],
    pub crop: SensorCrop,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct CameraOptics {
    pub lens: CameraLens,
    pub sensor: CameraSensor,
}

impl Default for CameraOptics {
    fn default() -> Self {
        Self::from_vertical_fov(45.0_f32.to_radians(), [16, 9])
            .expect("default camera optics are valid")
    }
}

impl CameraOptics {
    pub fn new(focal_length_mm: f32, aspect: [u32; 2]) -> Result<Self, String> {
        let aspect = reduced_aspect(aspect)?;
        let optics = Self {
            lens: CameraLens {
                model: CameraLensModel::FullFrameHorizontal,
                focal_length_mm,
            },
            sensor: CameraSensor {
                gate_width_mm: FULL_FRAME_GATE_WIDTH_MM,
                aspect,
                crop: SensorCrop::Full,
            },
        };
        optics.validate()?;
        Ok(optics)
    }

    pub fn from_vertical_fov(vertical_fov_rad: f32, aspect: [u32; 2]) -> Result<Self, String> {
        if !vertical_fov_rad.is_finite()
            || !(1.0_f32.to_radians()..179.0_f32.to_radians()).contains(&vertical_fov_rad)
        {
            return Err(format!("invalid vertical FOV {vertical_fov_rad}"));
        }
        let [width, height] = reduced_aspect(aspect)?;
        let aspect = width as f32 / height as f32;
        let horizontal_fov_rad = 2.0 * ((vertical_fov_rad * 0.5).tan() * aspect).atan();
        let mut focal_length_mm =
            FULL_FRAME_GATE_WIDTH_MM / (2.0 * (horizontal_fov_rad * 0.5).tan());
        if (focal_length_mm - MIN_FOCAL_LENGTH_MM).abs() <= 1.0e-4 {
            focal_length_mm = MIN_FOCAL_LENGTH_MM;
        } else if (focal_length_mm - MAX_FOCAL_LENGTH_MM).abs() <= 1.0e-3 {
            focal_length_mm = MAX_FOCAL_LENGTH_MM;
        }
        Self::new(focal_length_mm, [width, height])
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.lens.model != CameraLensModel::FullFrameHorizontal {
            return Err("unsupported camera lens model".into());
        }
        if !self.lens.focal_length_mm.is_finite()
            || !(MIN_FOCAL_LENGTH_MM..=MAX_FOCAL_LENGTH_MM).contains(&self.lens.focal_length_mm)
        {
            return Err(format!(
                "focal length {} mm is outside {MIN_FOCAL_LENGTH_MM}..={MAX_FOCAL_LENGTH_MM}",
                self.lens.focal_length_mm
            ));
        }
        if !self.sensor.gate_width_mm.is_finite()
            || (self.sensor.gate_width_mm - FULL_FRAME_GATE_WIDTH_MM).abs() > 1.0e-4
        {
            return Err(format!(
                "full-frame-horizontal optics require a {FULL_FRAME_GATE_WIDTH_MM} mm gate"
            ));
        }
        if reduced_aspect(self.sensor.aspect)? != self.sensor.aspect {
            return Err("camera sensor aspect must be reduced".into());
        }
        Ok(())
    }

    pub fn horizontal_fov_rad(&self) -> f32 {
        2.0 * (self.sensor.gate_width_mm / (2.0 * self.lens.focal_length_mm)).atan()
    }

    pub fn vertical_fov_rad(&self) -> f32 {
        let aspect = self.sensor.aspect[0] as f32 / self.sensor.aspect[1] as f32;
        2.0 * ((self.horizontal_fov_rad() * 0.5).tan() / aspect).atan()
    }

    pub fn with_focal_length_mm(mut self, focal_length_mm: f32) -> Result<Self, String> {
        self.lens.focal_length_mm = focal_length_mm;
        self.validate()?;
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct CapturedCameraState {
    pub optics: CameraOptics,
    pub effective_focal_length_mm: f32,
    pub derived_vertical_fov_rad: f32,
    pub output_extent: [u32; 2],
}

pub fn reduced_aspect([width, height]: [u32; 2]) -> Result<[u32; 2], String> {
    if width == 0 || height == 0 {
        return Err("camera sensor aspect dimensions must be non-zero".into());
    }
    let divisor = gcd(width, height);
    Ok([width / divisor, height / divisor])
}

const fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optics_round_trip_across_common_aspects() {
        for aspect in [[16, 9], [4, 3], [21, 9], [9, 16]] {
            let optics = CameraOptics::new(50.0, aspect).unwrap();
            let restored =
                CameraOptics::from_vertical_fov(optics.vertical_fov_rad(), aspect).unwrap();
            assert!((restored.lens.focal_length_mm - 50.0).abs() < 1.0e-4);
        }
    }
}
