use glam::Vec3;

/// Identifies one face of a cubemap, following the standard OpenGL convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CubemapFace {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

impl CubemapFace {
    pub const ALL: [CubemapFace; 6] = [
        CubemapFace::PosX,
        CubemapFace::NegX,
        CubemapFace::PosY,
        CubemapFace::NegY,
        CubemapFace::PosZ,
        CubemapFace::NegZ,
    ];
}

/// A typed cubemap with 6 square faces of `resolution × resolution` texels.
///
/// Face data is stored row-major, top-to-bottom.  The generic parameter `T`
/// allows different pixel formats: `f32` for height accumulators, `[f32; 4]`
/// for albedo accumulators, `u16` for finalized R16 height, `[u8; 4]` for
/// finalized RGBA8 albedo.
#[derive(Clone, Debug)]
pub struct Cubemap<T: Copy + Default> {
    resolution: u32,
    faces: [Vec<T>; 6],
}

impl<T: Copy + Default> Cubemap<T> {
    /// Create a cubemap with all texels set to `T::default()`.
    pub fn new(resolution: u32) -> Self {
        assert!(resolution > 0, "cubemap resolution must be > 0");
        let n = (resolution * resolution) as usize;
        Self {
            resolution,
            faces: std::array::from_fn(|_| vec![T::default(); n]),
        }
    }

    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    pub fn get(&self, face: CubemapFace, x: u32, y: u32) -> T {
        self.faces[face as usize][(y * self.resolution + x) as usize]
    }

    pub fn set(&mut self, face: CubemapFace, x: u32, y: u32, val: T) {
        self.faces[face as usize][(y * self.resolution + x) as usize] = val;
    }

    /// Raw face data, row-major, top-to-bottom.
    pub fn face_data(&self, face: CubemapFace) -> &[T] {
        &self.faces[face as usize]
    }

    pub fn face_data_mut(&mut self, face: CubemapFace) -> &mut [T] {
        &mut self.faces[face as usize]
    }
}

/// Contiguous byte export for GPU upload.  Faces are emitted in order
/// PosX, NegX, PosY, NegY, PosZ, NegZ (matching `CubemapFace` discriminants).
impl<T: Copy + Default + bytemuck_compat::Pod> Cubemap<T> {
    pub fn as_bytes(&self) -> Vec<u8> {
        let texel_size = std::mem::size_of::<T>();
        let face_bytes = (self.resolution * self.resolution) as usize * texel_size;
        let mut out = Vec::with_capacity(face_bytes * 6);
        for face in &self.faces {
            let ptr = face.as_ptr() as *const u8;
            out.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, face_bytes) });
        }
        out
    }
}

/// Bilinear sampling for f32 cubemaps (height).
impl Cubemap<f32> {
    pub fn sample_bilinear(&self, dir: Vec3) -> f32 {
        let (face, u, v) = dir_to_face_uv(dir);
        let res = self.resolution as f32;
        // Map [0,1] UV to texel centers: pixel i covers [(i)/(res), (i+1)/(res)]
        let px = (u * res - 0.5).clamp(0.0, res - 1.001);
        let py = (v * res - 0.5).clamp(0.0, res - 1.001);
        let x0 = px.floor() as u32;
        let y0 = py.floor() as u32;
        let x1 = (x0 + 1).min(self.resolution - 1);
        let y1 = (y0 + 1).min(self.resolution - 1);
        let fx = px - px.floor();
        let fy = py - py.floor();

        let c00 = self.get(face, x0, y0);
        let c10 = self.get(face, x1, y0);
        let c01 = self.get(face, x0, y1);
        let c11 = self.get(face, x1, y1);

        let top = c00 + (c10 - c00) * fx;
        let bot = c01 + (c11 - c01) * fx;
        top + (bot - top) * fy
    }
}

/// Bilinear sampling for [f32; 4] cubemaps (albedo accumulator).
impl Cubemap<[f32; 4]> {
    pub fn sample_bilinear(&self, dir: Vec3) -> [f32; 4] {
        let (face, u, v) = dir_to_face_uv(dir);
        let res = self.resolution as f32;
        let px = (u * res - 0.5).clamp(0.0, res - 1.001);
        let py = (v * res - 0.5).clamp(0.0, res - 1.001);
        let x0 = px.floor() as u32;
        let y0 = py.floor() as u32;
        let x1 = (x0 + 1).min(self.resolution - 1);
        let y1 = (y0 + 1).min(self.resolution - 1);
        let fx = px - px.floor();
        let fy = py - py.floor();

        let c00 = self.get(face, x0, y0);
        let c10 = self.get(face, x1, y0);
        let c01 = self.get(face, x0, y1);
        let c11 = self.get(face, x1, y1);

        let mut result = [0.0f32; 4];
        for i in 0..4 {
            let top = c00[i] + (c10[i] - c00[i]) * fx;
            let bot = c01[i] + (c11[i] - c01[i]) * fx;
            result[i] = top + (bot - top) * fy;
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Direction ↔ face+UV conversions (standard OpenGL cubemap convention)
// ---------------------------------------------------------------------------

/// Map a direction vector to a cubemap face and UV in [0, 1]².
///
/// Uses the OpenGL cubemap convention:
/// - The face is selected by the axis with the largest absolute component.
/// - UV (0,0) is top-left of the face.
pub fn dir_to_face_uv(dir: Vec3) -> (CubemapFace, f32, f32) {
    let abs = dir.abs();
    let (face, sc, tc, ma) = if abs.x >= abs.y && abs.x >= abs.z {
        if dir.x > 0.0 {
            (CubemapFace::PosX, -dir.z, -dir.y, abs.x)
        } else {
            (CubemapFace::NegX, dir.z, -dir.y, abs.x)
        }
    } else if abs.y >= abs.x && abs.y >= abs.z {
        if dir.y > 0.0 {
            (CubemapFace::PosY, dir.x, dir.z, abs.y)
        } else {
            (CubemapFace::NegY, dir.x, -dir.z, abs.y)
        }
    } else if dir.z > 0.0 {
        (CubemapFace::PosZ, dir.x, -dir.y, abs.z)
    } else {
        (CubemapFace::NegZ, -dir.x, -dir.y, abs.z)
    };
    // Map [-1, 1] to [0, 1]
    let u = (sc / ma + 1.0) * 0.5;
    let v = (tc / ma + 1.0) * 0.5;
    (face, u, v)
}

/// Inverse: map a cubemap face + UV to a unit direction vector.
pub fn face_uv_to_dir(face: CubemapFace, u: f32, v: f32) -> Vec3 {
    // Map [0, 1] to [-1, 1]
    let sc = u * 2.0 - 1.0;
    let tc = v * 2.0 - 1.0;
    let dir = match face {
        CubemapFace::PosX => Vec3::new(1.0, -tc, -sc),
        CubemapFace::NegX => Vec3::new(-1.0, -tc, sc),
        CubemapFace::PosY => Vec3::new(sc, 1.0, tc),
        CubemapFace::NegY => Vec3::new(sc, -1.0, -tc),
        CubemapFace::PosZ => Vec3::new(sc, -tc, 1.0),
        CubemapFace::NegZ => Vec3::new(-sc, -tc, -1.0),
    };
    dir.normalize()
}

/// Compute a reasonable cubemap resolution for a body of the given radius.
/// Targets ~3 km/texel at the equator, clamped to [4, 2048] and rounded to
/// the next power of two.
pub fn default_resolution(radius_m: f32) -> u32 {
    // Equator circumference / target_texel_size = total texels around equator
    // Each face spans 90° of the equator, so face_res ≈ total / 4.
    // With target 3 km/texel: face_res ≈ (2π × radius_m) / (4 × 3000)
    let raw = (std::f32::consts::TAU * radius_m / (4.0 * 3000.0)).ceil() as u32;
    raw.next_power_of_two().clamp(4, 2048)
}

// ---------------------------------------------------------------------------
// CubemapAccumulator — build-time additive accumulation
// ---------------------------------------------------------------------------

/// Build-time accumulator for cubemap contributions.
///
/// Stages write to this during pipeline execution.  After all stages run,
/// `finalize_height()` and `finalize_albedo()` produce the immutable cubemaps
/// for `BodyData`.
pub struct CubemapAccumulator {
    pub height: Cubemap<f32>,
    pub albedo: Cubemap<[f32; 4]>,
}

impl CubemapAccumulator {
    pub fn new(resolution: u32) -> Self {
        Self {
            height: Cubemap::new(resolution),
            albedo: Cubemap::new(resolution),
        }
    }

    pub fn resolution(&self) -> u32 {
        self.height.resolution()
    }

    pub fn add_height(&mut self, face: CubemapFace, x: u32, y: u32, delta: f32) {
        let cur = self.height.get(face, x, y);
        self.height.set(face, x, y, cur + delta);
    }

    pub fn add_albedo(&mut self, face: CubemapFace, x: u32, y: u32, rgba_delta: [f32; 4]) {
        let cur = self.albedo.get(face, x, y);
        self.albedo.set(face, x, y, [
            cur[0] + rgba_delta[0],
            cur[1] + rgba_delta[1],
            cur[2] + rgba_delta[2],
            cur[3] + rgba_delta[3],
        ]);
    }

    /// Splat a height delta at the texel nearest to `dir`.
    pub fn add_height_at_dir(&mut self, dir: Vec3, delta: f32) {
        let (face, u, v) = dir_to_face_uv(dir);
        let res = self.height.resolution();
        let x = ((u * res as f32) as u32).min(res - 1);
        let y = ((v * res as f32) as u32).min(res - 1);
        self.add_height(face, x, y, delta);
    }

    /// Splat an albedo delta at the texel nearest to `dir`.
    pub fn add_albedo_at_dir(&mut self, dir: Vec3, rgba_delta: [f32; 4]) {
        let (face, u, v) = dir_to_face_uv(dir);
        let res = self.albedo.resolution();
        let x = ((u * res as f32) as u32).min(res - 1);
        let y = ((v * res as f32) as u32).min(res - 1);
        self.add_albedo(face, x, y, rgba_delta);
    }

    /// Quantize accumulated height (f32 meters) into R16Unorm.
    ///
    /// Encoding: value 32768 = 0 meters displacement.  Each unit = range/65535
    /// meters.  `range` is the maximum absolute displacement found, padded by 1%.
    /// Returns the cubemap and the range (meters) needed by the shader to decode.
    pub fn finalize_height(&self) -> (Cubemap<u16>, f32) {
        let res = self.height.resolution();
        let mut max_abs: f32 = 0.0;
        for face in CubemapFace::ALL {
            for val in self.height.face_data(face) {
                max_abs = max_abs.max(val.abs());
            }
        }
        // Avoid division by zero for flat surfaces.
        let range = if max_abs < 1e-6 { 1.0 } else { max_abs * 1.01 };

        let mut out = Cubemap::new(res);
        for face in CubemapFace::ALL {
            let src = self.height.face_data(face);
            let dst = out.face_data_mut(face);
            for (i, &val) in src.iter().enumerate() {
                // Map [-range, range] → [0, 65535]
                let normalized = (val / range + 1.0) * 0.5;
                dst[i] = (normalized.clamp(0.0, 1.0) * 65535.0) as u16;
            }
        }
        (out, range)
    }

    /// Quantize accumulated albedo (linear f32 RGBA) into sRGB RGBA8.
    ///
    /// If no albedo has been written, produces a uniform mid-grey (128, 128, 128, 255).
    pub fn finalize_albedo(&self) -> Cubemap<[u8; 4]> {
        let res = self.albedo.resolution();

        // Check if any albedo was actually written.
        let has_data = CubemapFace::ALL.iter().any(|&face| {
            self.albedo.face_data(face).iter().any(|v| v[3] > 0.0)
        });

        let mut out = Cubemap::new(res);
        if !has_data {
            // Default: mid-grey, fully opaque.
            for face in CubemapFace::ALL {
                for val in out.face_data_mut(face) {
                    *val = [128, 128, 128, 255];
                }
            }
            return out;
        }

        for face in CubemapFace::ALL {
            let src = self.albedo.face_data(face);
            let dst = out.face_data_mut(face);
            for (i, val) in src.iter().enumerate() {
                // Alpha channel carries coverage/weight; divide through if > 0.
                let w = if val[3] > 0.0 { val[3] } else { 1.0 };
                dst[i] = [
                    linear_to_srgb_u8(val[0] / w),
                    linear_to_srgb_u8(val[1] / w),
                    linear_to_srgb_u8(val[2] / w),
                    255,
                ];
            }
        }
        out
    }
}

fn linear_to_srgb_u8(linear: f32) -> u8 {
    let srgb = if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (srgb.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

// ---------------------------------------------------------------------------
// Pod-like trait — we don't pull in `bytemuck` so we gate `as_bytes` on this.
// ---------------------------------------------------------------------------

/// Marker trait for types safe to transmute to bytes.  Only implemented for
/// the concrete texel types we use.
///
/// # Safety
/// The type must have no padding, no internal pointers, and be `Copy`.
pub mod bytemuck_compat {
    /// # Safety
    ///
    /// The type must have no padding, no internal pointers, and be `Copy`.
    /// It must be safe to interpret a contiguous slice of `T` as raw bytes.
    pub unsafe trait Pod: Copy + 'static {}
    unsafe impl Pod for u16 {}
    unsafe impl Pod for [u8; 4] {}
    unsafe impl Pod for f32 {}
    unsafe impl Pod for [f32; 4] {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_aligned_face_selection() {
        let cases = [
            (Vec3::X, CubemapFace::PosX),
            (Vec3::NEG_X, CubemapFace::NegX),
            (Vec3::Y, CubemapFace::PosY),
            (Vec3::NEG_Y, CubemapFace::NegY),
            (Vec3::Z, CubemapFace::PosZ),
            (Vec3::NEG_Z, CubemapFace::NegZ),
        ];
        for (dir, expected_face) in cases {
            let (face, u, v) = dir_to_face_uv(dir);
            assert_eq!(face, expected_face, "wrong face for {dir}");
            assert!((u - 0.5).abs() < 0.01, "u not centered for axis dir {dir}: {u}");
            assert!((v - 0.5).abs() < 0.01, "v not centered for axis dir {dir}: {v}");
        }
    }

    #[test]
    fn dir_to_face_uv_round_trip() {
        let dirs = [
            Vec3::new(1.0, 0.3, -0.7).normalize(),
            Vec3::new(-0.5, 1.0, 0.2).normalize(),
            Vec3::new(0.1, -0.2, 1.0).normalize(),
            Vec3::new(-1.0, -0.5, -0.3).normalize(),
        ];
        for dir in dirs {
            let (face, u, v) = dir_to_face_uv(dir);
            let recovered = face_uv_to_dir(face, u, v);
            let dot = dir.dot(recovered);
            assert!(
                dot > 0.999,
                "round-trip failed for {dir}: got {recovered}, dot={dot}"
            );
        }
    }

    #[test]
    fn default_resolution_values() {
        // Mira: 869 km radius → should be ~512
        let mira = default_resolution(869_000.0);
        assert_eq!(mira, 512);

        // Homeworld: 3186 km → should be ~2048
        let home = default_resolution(3_186_000.0);
        assert!(home >= 1024 && home <= 2048, "homeworld resolution: {home}");

        // Tiny body → clamped to minimum
        let tiny = default_resolution(100.0);
        assert_eq!(tiny, 4);
    }

    #[test]
    fn accumulator_finalize_flat() {
        let acc = CubemapAccumulator::new(4);
        let (height, range) = acc.finalize_height();
        // All zero height → all texels should be 32768 (midpoint)
        for face in CubemapFace::ALL {
            for &val in height.face_data(face) {
                assert_eq!(val, 32767, "expected midpoint for zero height");
            }
        }
        assert!(range > 0.0);
    }

    #[test]
    fn accumulator_finalize_albedo_default_grey() {
        let acc = CubemapAccumulator::new(4);
        let albedo = acc.finalize_albedo();
        for face in CubemapFace::ALL {
            for val in albedo.face_data(face) {
                assert_eq!(*val, [128, 128, 128, 255]);
            }
        }
    }

    #[test]
    fn bilinear_sample_center() {
        let mut cm = Cubemap::<f32>::new(2);
        // Set all texels on PosX face to 1.0
        for y in 0..2 {
            for x in 0..2 {
                cm.set(CubemapFace::PosX, x, y, 1.0);
            }
        }
        // Sample at +X axis should hit PosX face center
        let val = cm.sample_bilinear(Vec3::X);
        assert!((val - 1.0).abs() < 0.01, "expected ~1.0, got {val}");
    }

    #[test]
    fn linear_to_srgb_boundaries() {
        assert_eq!(linear_to_srgb_u8(0.0), 0);
        assert_eq!(linear_to_srgb_u8(1.0), 255);
        // Mid-grey in linear should map to ~188 in sRGB
        let mid = linear_to_srgb_u8(0.5);
        assert!(mid > 180 && mid < 200, "mid-grey sRGB: {mid}");
    }
}
