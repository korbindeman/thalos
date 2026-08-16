use anyhow::{Context, Result};
use bevy::{
    math::{DVec2, DVec3},
    prelude::Resource,
};
use thalos_geodetic::{
    CuracaoEgm2008, EcefPosition, Egm2008Position, GeographicPosition, LocalTangentFrame,
    UtmPosition,
};

use crate::{cli::SpatialMode, terrain::TerrainDataset};

pub const KORSOU_UTM_ZONE: u8 = 19;

/// Maps the dataset's stable local coordinates into renderer coordinates.
///
/// Simulation, streaming, and viewpoints stay in local UTM metres. Only this
/// adapter knows whether render geometry remains planar or is bent through
/// WGS 84 ECEF into a local east/north/up tangent frame.
#[derive(Resource)]
pub struct TerrainSpatialFrame {
    local_origin_utm_m: [f64; 2],
    tangent: Option<LocalTangentFrame>,
}

impl TerrainSpatialFrame {
    pub fn new(dataset: &TerrainDataset, mode: SpatialMode) -> Result<Self> {
        let local_origin_utm_m = dataset.metadata.local_origin_utm_m;
        let tangent = match mode {
            SpatialMode::Planar => None,
            SpatialMode::Ellipsoid => {
                let horizontal = UtmPosition::new_north(
                    KORSOU_UTM_ZONE,
                    local_origin_utm_m[0],
                    local_origin_utm_m[1],
                )
                .and_then(UtmPosition::to_wgs84)
                .context("convert Curaçao local origin from EPSG:32619 to WGS 84")?;
                let origin =
                    Egm2008Position::new(horizontal.latitude_deg, horizontal.longitude_deg, 0.0)
                        .and_then(|position| position.to_ellipsoid(&CuracaoEgm2008))
                        .context("convert Curaçao EGM2008 sea level to WGS 84 ellipsoid height")?;
                Some(LocalTangentFrame::new(origin))
            }
        };
        Ok(Self {
            local_origin_utm_m,
            tangent,
        })
    }

    /// Returns Bevy +X east, +Y up, -Z north in tangent-frame metres.
    pub fn project(&self, local_position_m: DVec3) -> DVec3 {
        let Some(tangent) = self.tangent else {
            return local_position_m;
        };
        let horizontal = self.local_to_wgs84(local_position_m);
        let ellipsoid = Egm2008Position::new(
            horizontal.latitude_deg,
            horizontal.longitude_deg,
            local_position_m.y,
        )
        .and_then(|position| position.to_ellipsoid(&CuracaoEgm2008))
        .expect("rendered Curaçao terrain must stay inside the checked EGM2008 grid");
        let enu = tangent.to_enu(EcefPosition::from(ellipsoid));
        DVec3::new(enu.east_m, enu.up_m, -enu.north_m)
    }

    /// Planar tiles retain their traditional local entity origin. Ellipsoid
    /// tiles share the tangent-frame origin so a coarse quadtree tile may
    /// overhang the DEM crop without asking the regional geoid to extrapolate
    /// merely to place its entity.
    pub fn tile_origin(&self, local_x_m: f64, local_z_m: f64) -> DVec3 {
        if self.tangent.is_some() {
            DVec3::ZERO
        } else {
            DVec3::new(local_x_m, 0.0, local_z_m)
        }
    }

    pub fn project_direction(&self, at: DVec3, direction: DVec3) -> DVec3 {
        let start = self.project(at);
        (self.project(at + direction) - start).normalize()
    }

    pub fn local_to_utm(&self, local_position_m: DVec3) -> UtmPosition {
        UtmPosition::new_north(
            KORSOU_UTM_ZONE,
            local_position_m.x + self.local_origin_utm_m[0],
            self.local_origin_utm_m[1] - local_position_m.z,
        )
        .expect("validated Curaçao local coordinates must convert to UTM zone 19N")
    }

    pub fn local_to_wgs84(&self, local_position_m: DVec3) -> GeographicPosition {
        self.local_to_utm(local_position_m)
            .to_wgs84()
            .expect("validated Curaçao local coordinates must convert to WGS 84")
    }

    pub fn utm_to_local_xz(&self, position: UtmPosition) -> DVec2 {
        assert_eq!(
            position.zone, KORSOU_UTM_ZONE,
            "Kòrsou local positions require UTM zone 19N"
        );
        DVec2::new(
            position.easting_m - self.local_origin_utm_m[0],
            self.local_origin_utm_m[1] - position.northing_m,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn dataset() -> TerrainDataset {
        let asset_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/terrain/curacao");
        TerrainDataset::load(&asset_dir).unwrap()
    }

    #[test]
    fn planar_adapter_is_identity() {
        let frame = TerrainSpatialFrame::new(&dataset(), SpatialMode::Planar).unwrap();
        let point = DVec3::new(1_200.0, 300.0, -900.0);
        assert_eq!(frame.project(point), point);
    }

    #[test]
    fn local_and_utm_horizontal_coordinates_round_trip() {
        let frame = TerrainSpatialFrame::new(&dataset(), SpatialMode::Planar).unwrap();
        let local = DVec3::new(8_900.0, 42.0, 14_700.0);
        let round_trip = frame.utm_to_local_xz(frame.local_to_utm(local));
        assert_eq!(round_trip, DVec2::new(local.x, local.z));
    }

    #[test]
    fn ellipsoid_adapter_keeps_origin_at_egm2008_sea_level() {
        let frame = TerrainSpatialFrame::new(&dataset(), SpatialMode::Ellipsoid).unwrap();
        let projected = frame.project(DVec3::ZERO);
        assert!(
            projected.length() < 1.0e-6,
            "origin drifted by {projected:?}"
        );
    }

    #[test]
    fn ellipsoid_adapter_exposes_earth_curvature() {
        let frame = TerrainSpatialFrame::new(&dataset(), SpatialMode::Ellipsoid).unwrap();
        let east = frame.project(DVec3::new(20_000.0, 0.0, 0.0));
        assert!(
            east.y < -25.0,
            "20 km chord should fall below tangent plane: {east:?}"
        );
    }
}
