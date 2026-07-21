use crate::math::{TerrainModel, C_SQR};
use bevy::{
    math::{DVec2, DVec3, IVec2, UVec2},
    render::render_resource::ShaderType,
};
use std::fmt;

const NEIGHBOURING_SIDES: [[u32; 5]; 6] = [
    [0, 4, 2, 1, 5],
    [1, 0, 2, 3, 5],
    [2, 0, 4, 3, 1],
    [3, 2, 4, 5, 1],
    [4, 2, 0, 5, 3],
    [5, 4, 0, 1, 3],
];

#[derive(Clone, Copy)]
enum SideInfo {
    Fixed0,
    Fixed1,
    PositiveS,
    PositiveT,
}

impl SideInfo {
    const EVEN_LIST: [[SideInfo; 2]; 6] = [
        [SideInfo::PositiveS, SideInfo::PositiveT],
        [SideInfo::Fixed0, SideInfo::PositiveT],
        [SideInfo::Fixed0, SideInfo::PositiveS],
        [SideInfo::PositiveT, SideInfo::PositiveS],
        [SideInfo::PositiveT, SideInfo::Fixed0],
        [SideInfo::PositiveS, SideInfo::Fixed0],
    ];
    const ODD_LIST: [[SideInfo; 2]; 6] = [
        [SideInfo::PositiveS, SideInfo::PositiveT],
        [SideInfo::PositiveS, SideInfo::Fixed1],
        [SideInfo::PositiveT, SideInfo::Fixed1],
        [SideInfo::PositiveT, SideInfo::PositiveS],
        [SideInfo::Fixed1, SideInfo::PositiveS],
        [SideInfo::Fixed1, SideInfo::PositiveT],
    ];

    fn project_to_side(side: u32, other_side: u32) -> [SideInfo; 2] {
        let index = ((6 + other_side - side) % 6) as usize;

        if side.is_multiple_of(2) {
            SideInfo::EVEN_LIST[index]
        } else {
            SideInfo::ODD_LIST[index]
        }
    }
}

/// Describes a location on the unit cube sphere.
/// The side index refers to one of the six cube faces and the uv coordinate describes the location within this side.
#[derive(Copy, Clone, Debug, Default)]
pub struct Coordinate {
    pub side: u32,
    pub uv: DVec2,
}

impl Coordinate {
    pub fn new(side: u32, uv: DVec2) -> Self {
        Self { side, uv }
    }

    /// Calculates the coordinate for for the local position on the unit cube sphere.
    pub fn from_world_position(world_position: DVec3, model: &TerrainModel) -> Self {
        let local_position = model.position_world_to_local(world_position);

        let (side, uv) = if model.is_spherical() {
            let normal = local_position;
            let abs_normal = normal.abs();

            let (side, uv) = if abs_normal.x > abs_normal.y && abs_normal.x > abs_normal.z {
                if normal.x < 0.0 {
                    (0, DVec2::new(-normal.z / normal.x, normal.y / normal.x))
                } else {
                    (3, DVec2::new(-normal.y / normal.x, normal.z / normal.x))
                }
            } else if abs_normal.z > abs_normal.y {
                if normal.z > 0.0 {
                    (1, DVec2::new(normal.x / normal.z, -normal.y / normal.z))
                } else {
                    (4, DVec2::new(normal.y / normal.z, -normal.x / normal.z))
                }
            } else {
                if normal.y > 0.0 {
                    (2, DVec2::new(normal.x / normal.y, normal.z / normal.y))
                } else {
                    (5, DVec2::new(-normal.z / normal.y, -normal.x / normal.y))
                }
            };

            let w = uv * ((1.0 + C_SQR) / (1.0 + C_SQR * uv * uv)).powf(0.5);
            let uv = 0.5 * w + 0.5;

            (side, uv)
        } else {
            let uv = DVec2::new(local_position.x + 0.5, local_position.z + 0.5)
                .clamp(DVec2::ZERO, DVec2::ONE);

            (0, uv)
        };

        Self { side, uv }
    }

    /// Maps the cube-sphere coordinate to a world-space position on the terrain
    /// model's surface (or offset along the surface normal by `height`).
    pub fn world_position(self, model: &TerrainModel, height: f32) -> DVec3 {
        let local_position = if model.is_spherical() {
            let w = (self.uv - 0.5) / 0.5;
            let uv = w / (1.0 + C_SQR - C_SQR * w * w).powf(0.5);

            match self.side {
                0 => DVec3::new(-1.0, -uv.y, uv.x),
                1 => DVec3::new(uv.x, -uv.y, 1.0),
                2 => DVec3::new(uv.x, 1.0, uv.y),
                3 => DVec3::new(1.0, -uv.x, uv.y),
                4 => DVec3::new(uv.y, -uv.x, -1.0),
                5 => DVec3::new(uv.y, -1.0, uv.x),
                _ => unreachable!(),
            }
            .normalize()
        } else {
            DVec3::new(self.uv.x - 0.5, 0.0, self.uv.y - 0.5)
        };

        model.position_local_to_world(local_position, height as f64)
    }

    /// Projects the coordinate onto one of the six cube faces.
    /// Thereby it chooses the closest location on this face to the original coordinate.
    pub(crate) fn project_to_side(self, side: u32, model: &TerrainModel) -> Self {
        if model.is_spherical() {
            let info = SideInfo::project_to_side(self.side, side);

            let uv = info
                .map(|info| match info {
                    SideInfo::Fixed0 => 0.0,
                    SideInfo::Fixed1 => 1.0,
                    SideInfo::PositiveS => self.uv.x,
                    SideInfo::PositiveT => self.uv.y,
                })
                .into();

            Self { side, uv }
        } else {
            self
        }
    }
}

/// The global coordinate and identifier of a tile.
#[derive(Copy, Clone, Default, Debug, Hash, Eq, PartialEq, ShaderType)]
pub struct TileCoordinate {
    /// The side of the cube sphere the tile is located on.
    pub side: u32,
    /// The lod of the tile, where 0 is the highest level of detail with the smallest size
    /// and highest resolution
    pub lod: u32,
    /// The x position of the tile in tile sizes.
    pub x: u32,
    /// The y position of the tile in tile sizes.
    pub y: u32,
}

impl TileCoordinate {
    pub const INVALID: TileCoordinate = TileCoordinate {
        side: u32::MAX,
        lod: u32::MAX,
        x: u32::MAX,
        y: u32::MAX,
    };

    pub fn new(side: u32, lod: u32, x: u32, y: u32) -> Self {
        Self { side, lod, x, y }
    }

    pub fn count(lod: u32) -> u32 {
        1 << lod
    }

    /// Returns the cube-sphere [`Coordinate`] of the texel center for the given
    /// `pixel` index inside a tile texture of size `texture_size` with a
    /// `border_size`-pixel border.
    ///
    /// This is the canonical pixel→position mapping used by the renderer when
    /// sampling tile data. [`TileProvider`](crate::terrain_data::TileProvider)
    /// implementations should evaluate their data source at the world position
    /// returned by passing this coordinate to [`Coordinate::world_position`],
    /// so that values agree exactly across tile borders. Pixels in the border
    /// region (outside `[border_size, texture_size - border_size)`) will lie
    /// outside the tile's logical extent and overlap a neighbouring tile.
    pub fn pixel_coordinate(self, pixel: UVec2, texture_size: u32, border_size: u32) -> Coordinate {
        let inner = (texture_size - 2 * border_size) as f64;
        let in_tile_uv =
            (DVec2::new(pixel.x as f64, pixel.y as f64) + 0.5 - border_size as f64) / inner;
        let face_uv =
            (DVec2::new(self.x as f64, self.y as f64) + in_tile_uv) / Self::count(self.lod) as f64;

        Coordinate {
            side: self.side,
            uv: face_uv,
        }
    }

    /// Returns the cube-sphere coordinate for a runtime-generated tile texel,
    /// with border pixels stitched the same way as the former offline
    /// `stitch.wgsl` pass.
    ///
    /// For spherical terrain, border texels outside the tile's logical center
    /// are evaluated in the neighbouring tile/face orientation instead of by
    /// extrapolating the current cube face past `uv < 0` or `uv > 1`. That
    /// keeps bilinear vertex samples on shared tile and face edges sampling
    /// matching heights. For planar terrain or missing cube-corner neighbours,
    /// border pixels repeat the nearest center texel.
    pub fn stitched_pixel_coordinate(
        self,
        pixel: UVec2,
        texture_size: u32,
        border_size: u32,
        spherical: bool,
    ) -> Coordinate {
        if border_size == 0
            || texture_size <= 2 * border_size
            || !is_border_pixel(pixel, texture_size, border_size)
        {
            return self.pixel_coordinate(pixel, texture_size, border_size);
        }

        if !spherical {
            let repeat = repeat_border_pixel(pixel, texture_size, border_size);
            return self.pixel_coordinate(repeat, texture_size, border_size);
        }

        let Some(neighbour_index) = border_neighbour_index(pixel, texture_size, border_size) else {
            let repeat = repeat_border_pixel(pixel, texture_size, border_size);
            return self.pixel_coordinate(repeat, texture_size, border_size);
        };
        let Some(neighbour) = self.neighbours(true).nth(neighbour_index) else {
            let repeat = repeat_border_pixel(pixel, texture_size, border_size);
            return self.pixel_coordinate(repeat, texture_size, border_size);
        };
        if neighbour == Self::INVALID {
            let repeat = repeat_border_pixel(pixel, texture_size, border_size);
            return self.pixel_coordinate(repeat, texture_size, border_size);
        }

        // Shift the pixel into the neighbour's tile coordinate frame before
        // projecting axes. This mirrors `stitch.wgsl::neighbour_data`, which
        // reads `coords + offsets[neighbour_index]` from the neighbour atlas
        // slot. Without this translation the border pixel ends up referring to
        // a position one tile-width past the boundary, so adjacent tiles'
        // border heights disagree by hundreds of metres and leave visible
        // cracks at every same-LOD tile seam.
        let adjusted_pixel = (pixel.as_ivec2()
            + border_neighbour_pixel_offset(neighbour_index, texture_size, border_size))
        .as_uvec2();
        let source_pixel =
            project_border_pixel_to_side(adjusted_pixel, self.side, neighbour.side, texture_size);
        neighbour.pixel_coordinate(source_pixel, texture_size, border_size)
    }

    pub fn parent(self) -> Self {
        Self {
            side: self.side,
            lod: self.lod.wrapping_sub(1),
            x: self.x >> 1,
            y: self.y >> 1,
        }
    }

    pub fn children(self) -> impl Iterator<Item = Self> {
        (0..4).map(move |index| {
            TileCoordinate::new(
                self.side,
                self.lod + 1,
                (self.x << 1) + index % 2,
                (self.y << 1) + index / 2,
            )
        })
    }

    pub fn neighbours(self, spherical: bool) -> impl Iterator<Item = Self> {
        const OFFSETS: [IVec2; 8] = [
            IVec2::new(0, -1),
            IVec2::new(1, 0),
            IVec2::new(0, 1),
            IVec2::new(-1, 0),
            IVec2::new(-1, -1),
            IVec2::new(1, -1),
            IVec2::new(1, 1),
            IVec2::new(-1, 1),
        ];

        OFFSETS.iter().map(move |&offset| {
            let neighbour_position = IVec2::new(self.x as i32, self.y as i32) + offset;

            self.neighbour_coordinate(neighbour_position, spherical)
        })
    }

    fn neighbour_coordinate(self, neighbour_position: IVec2, spherical: bool) -> Self {
        let tile_count = Self::count(self.lod) as i32;

        if spherical {
            let edge_index = match neighbour_position {
                IVec2 { x, y } if (x < 0 || x >= tile_count) && (y < 0 || y >= tile_count) => {
                    return Self::INVALID;
                }
                IVec2 { x, .. } if x < 0 => 1,
                IVec2 { y, .. } if y < 0 => 2,
                IVec2 { x, .. } if x >= tile_count => 3,
                IVec2 { y, .. } if y >= tile_count => 4,
                _ => 0,
            };

            let neighbour_position = neighbour_position
                .clamp(IVec2::ZERO, IVec2::splat(tile_count - 1))
                .as_uvec2();

            let neighbour_side = NEIGHBOURING_SIDES[self.side as usize][edge_index];

            let info = SideInfo::project_to_side(self.side, neighbour_side);

            let [x, y] = info.map(|info| match info {
                SideInfo::Fixed0 => 0,
                SideInfo::Fixed1 => tile_count as u32 - 1,
                SideInfo::PositiveS => neighbour_position.x,
                SideInfo::PositiveT => neighbour_position.y,
            });

            Self::new(neighbour_side, self.lod, x, y)
        } else {
            if neighbour_position.x < 0
                || neighbour_position.y < 0
                || neighbour_position.x >= tile_count
                || neighbour_position.y >= tile_count
            {
                Self::INVALID
            } else {
                Self::new(
                    self.side,
                    self.lod,
                    neighbour_position.x as u32,
                    neighbour_position.y as u32,
                )
            }
        }
    }
}

fn is_border_pixel(pixel: UVec2, texture_size: u32, border_size: u32) -> bool {
    let center_size = texture_size - 2 * border_size;
    !inside_pixel(pixel, UVec2::splat(border_size), UVec2::splat(center_size))
}

fn repeat_border_pixel(pixel: UVec2, texture_size: u32, border_size: u32) -> UVec2 {
    let max_center = border_size + (texture_size - 2 * border_size).saturating_sub(1);
    pixel.clamp(UVec2::splat(border_size), UVec2::splat(max_center))
}

/// Per-neighbour pixel translation matching `stitch.wgsl::neighbour_data`'s
/// `offsets` array. Maps a border pixel in the current tile into the
/// corresponding interior-pixel position of the neighbour tile, before any
/// cross-face axis projection runs. Order matches [`TileCoordinate::neighbours`]
/// and [`border_neighbour_index`]: 0..4 cardinals (top, right, bottom, left),
/// 4..8 corners (top-left, top-right, bottom-right, bottom-left).
fn border_neighbour_pixel_offset(
    neighbour_index: usize,
    texture_size: u32,
    border_size: u32,
) -> IVec2 {
    let center_size = (texture_size - 2 * border_size) as i32;
    match neighbour_index {
        0 => IVec2::new(0, center_size),
        1 => IVec2::new(-center_size, 0),
        2 => IVec2::new(0, -center_size),
        3 => IVec2::new(center_size, 0),
        4 => IVec2::new(center_size, center_size),
        5 => IVec2::new(-center_size, center_size),
        6 => IVec2::new(-center_size, -center_size),
        7 => IVec2::new(center_size, -center_size),
        _ => IVec2::ZERO,
    }
}

fn border_neighbour_index(pixel: UVec2, texture_size: u32, border_size: u32) -> Option<usize> {
    let center_size = texture_size - 2 * border_size;
    let offset_size = border_size + center_size;
    let bounds = [
        (
            UVec2::new(border_size, 0),
            UVec2::new(center_size, border_size),
        ),
        (
            UVec2::new(offset_size, border_size),
            UVec2::new(border_size, center_size),
        ),
        (
            UVec2::new(border_size, offset_size),
            UVec2::new(center_size, border_size),
        ),
        (
            UVec2::new(0, border_size),
            UVec2::new(border_size, center_size),
        ),
        (UVec2::new(0, 0), UVec2::new(border_size, border_size)),
        (
            UVec2::new(offset_size, 0),
            UVec2::new(border_size, border_size),
        ),
        (
            UVec2::new(offset_size, offset_size),
            UVec2::new(border_size, border_size),
        ),
        (
            UVec2::new(0, offset_size),
            UVec2::new(border_size, border_size),
        ),
    ];

    bounds
        .iter()
        .position(|(origin, size)| inside_pixel(pixel, *origin, *size))
}

fn inside_pixel(pixel: UVec2, origin: UVec2, size: UVec2) -> bool {
    pixel.x >= origin.x
        && pixel.x < origin.x + size.x
        && pixel.y >= origin.y
        && pixel.y < origin.y + size.y
}

#[derive(Clone, Copy)]
enum PixelAxis {
    PositiveS,
    PositiveT,
    NegativeS,
    NegativeT,
}

fn project_border_pixel_to_side(
    pixel: UVec2,
    original_side: u32,
    projected_side: u32,
    texture_size: u32,
) -> UVec2 {
    use PixelAxis::{NegativeS, NegativeT, PositiveS, PositiveT};

    const EVEN_LIST: [[PixelAxis; 2]; 6] = [
        [PositiveS, PositiveT],
        [PositiveS, PositiveT],
        [NegativeT, PositiveS],
        [NegativeT, NegativeS],
        [PositiveT, NegativeS],
        [PositiveS, PositiveT],
    ];
    const ODD_LIST: [[PixelAxis; 2]; 6] = [
        [PositiveS, PositiveT],
        [PositiveS, PositiveT],
        [PositiveT, NegativeS],
        [PositiveT, PositiveS],
        [NegativeT, PositiveS],
        [PositiveS, PositiveT],
    ];

    let index = ((6 + projected_side - original_side) % 6) as usize;
    let axes = if original_side.is_multiple_of(2) {
        EVEN_LIST[index]
    } else {
        ODD_LIST[index]
    };

    UVec2::new(
        project_pixel_axis(pixel, axes[0], texture_size),
        project_pixel_axis(pixel, axes[1], texture_size),
    )
}

fn project_pixel_axis(pixel: UVec2, axis: PixelAxis, texture_size: u32) -> u32 {
    match axis {
        PixelAxis::PositiveS => pixel.x,
        PixelAxis::PositiveT => pixel.y,
        PixelAxis::NegativeS => texture_size - 1 - pixel.x,
        PixelAxis::NegativeT => texture_size - 1 - pixel.y,
    }
}

impl fmt::Display for TileCoordinate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}_{}_{}_{}", self.side, self.lod, self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::TerrainModel;

    const TEX_SIZE: u32 = 8;
    const BORDER: u32 = 2;

    fn sphere() -> TerrainModel {
        TerrainModel::sphere(DVec3::ZERO, 1_000_000.0, -1000.0, 1000.0)
    }

    /// Expected world position at a border pixel, derived directly from
    /// `stitch.wgsl::neighbour_data` semantics: read from the neighbour atlas
    /// slot at `(pixel + offset)` after axis projection.
    fn expected_world_position(tile: TileCoordinate, pixel: UVec2, model: &TerrainModel) -> DVec3 {
        let neighbour_index = border_neighbour_index(pixel, TEX_SIZE, BORDER)
            .expect("test pixel must lie in a border region");
        let neighbour = tile.neighbours(true).nth(neighbour_index).unwrap();
        assert_ne!(neighbour, TileCoordinate::INVALID);
        let adjusted = (pixel.as_ivec2()
            + border_neighbour_pixel_offset(neighbour_index, TEX_SIZE, BORDER))
        .as_uvec2();
        let source_pixel =
            project_border_pixel_to_side(adjusted, tile.side, neighbour.side, TEX_SIZE);
        neighbour
            .pixel_coordinate(source_pixel, TEX_SIZE, BORDER)
            .world_position(model, 0.0)
    }

    fn assert_stitch_matches_neighbour(tile: TileCoordinate, pixel: UVec2, label: &str) {
        let model = sphere();
        let stitched = tile
            .stitched_pixel_coordinate(pixel, TEX_SIZE, BORDER, true)
            .world_position(&model, 0.0);
        let expected = expected_world_position(tile, pixel, &model);
        let delta = (stitched - expected).length();
        assert!(
            delta < 1.0e-3,
            "{label}: stitched={stitched:?} expected={expected:?} delta={delta}",
        );
    }

    #[test]
    fn same_face_right_border_matches_neighbour_interior() {
        // Tile in the middle of a face → right border lands on a same-face
        // neighbour, no cross-face axis flip. Pre-fix this case sampled one
        // full tile-width past the boundary.
        let tile = TileCoordinate::new(0, 3, 3, 4);
        let pixel = UVec2::new(TEX_SIZE - 1, 4);
        assert_stitch_matches_neighbour(tile, pixel, "same-face right border");
    }

    #[test]
    fn same_face_top_border_matches_neighbour_interior() {
        let tile = TileCoordinate::new(2, 3, 4, 4);
        let pixel = UVec2::new(4, 0);
        assert_stitch_matches_neighbour(tile, pixel, "same-face top border");
    }

    #[test]
    fn same_face_left_border_matches_neighbour_interior() {
        let tile = TileCoordinate::new(1, 3, 4, 4);
        let pixel = UVec2::new(0, 4);
        assert_stitch_matches_neighbour(tile, pixel, "same-face left border");
    }

    #[test]
    fn same_face_bottom_border_matches_neighbour_interior() {
        let tile = TileCoordinate::new(3, 3, 4, 4);
        let pixel = UVec2::new(4, TEX_SIZE - 1);
        assert_stitch_matches_neighbour(tile, pixel, "same-face bottom border");
    }

    #[test]
    fn same_face_corner_matches_neighbour_interior() {
        // Top-right corner of an interior tile picks the diagonal neighbour
        // (offset (+1, -1)). Both border edges go through projection without
        // axis flip, so the pixel-offset translation is the only thing that
        // makes the corner align.
        let tile = TileCoordinate::new(0, 3, 3, 3);
        let pixel = UVec2::new(TEX_SIZE - 1, 0);
        assert_stitch_matches_neighbour(tile, pixel, "same-face top-right corner");
    }

    #[test]
    fn cross_face_right_border_matches_neighbour_interior() {
        // Tile on the right edge of side 0 → right border crosses to a
        // different cube face, exercising the axis-projection path alongside
        // the pixel-offset translation.
        let count = TileCoordinate::count(3);
        let tile = TileCoordinate::new(0, 3, count - 1, 4);
        let pixel = UVec2::new(TEX_SIZE - 1, 4);
        assert_stitch_matches_neighbour(tile, pixel, "cross-face right border");
    }

    #[test]
    fn non_border_pixel_uses_pixel_coordinate_directly() {
        // Sanity check the non-border early-return still matches the plain
        // `pixel_coordinate` mapping after the fix.
        let model = sphere();
        let tile = TileCoordinate::new(0, 3, 3, 4);
        let pixel = UVec2::new(4, 4);
        let stitched = tile
            .stitched_pixel_coordinate(pixel, TEX_SIZE, BORDER, true)
            .world_position(&model, 0.0);
        let direct = tile
            .pixel_coordinate(pixel, TEX_SIZE, BORDER)
            .world_position(&model, 0.0);
        assert!((stitched - direct).length() < 1.0e-6);
    }
}
