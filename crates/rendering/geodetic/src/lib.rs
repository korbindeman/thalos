//! Typed geodetic conversions for real-world render adapters.
//!
//! All public coordinates state both their horizontal representation and their
//! height datum. Render code must cross the orthometric/ellipsoidal boundary
//! explicitly instead of treating a DEM height as a WGS 84 ellipsoid height.

use std::{error::Error, fmt};

pub const WGS84_ELLIPSOID_EPSG: u16 = 7030;
pub const WGS84_GEOGRAPHIC_3D_EPSG: u16 = 4979;
pub const EGM2008_HEIGHT_EPSG: u16 = 3855;
pub const WGS84_UTM_ZONE_19N_EPSG: u16 = 32619;

const WGS84_SEMI_MAJOR_AXIS_M: f64 = 6_378_137.0;
const WGS84_INVERSE_FLATTENING: f64 = 298.257_223_563;
const UTM_SCALE: f64 = 0.9996;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeographicPosition {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
}

impl GeographicPosition {
    pub fn new(latitude_deg: f64, longitude_deg: f64) -> Result<Self, GeodeticError> {
        if !latitude_deg.is_finite() || !(-90.0..=90.0).contains(&latitude_deg) {
            return Err(GeodeticError::InvalidLatitude(latitude_deg));
        }
        if !longitude_deg.is_finite() || !(-180.0..=180.0).contains(&longitude_deg) {
            return Err(GeodeticError::InvalidLongitude(longitude_deg));
        }
        Ok(Self {
            latitude_deg,
            longitude_deg,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EllipsoidPosition {
    pub horizontal: GeographicPosition,
    pub height_m: f64,
}

impl EllipsoidPosition {
    pub fn new(
        latitude_deg: f64,
        longitude_deg: f64,
        height_m: f64,
    ) -> Result<Self, GeodeticError> {
        if !height_m.is_finite() {
            return Err(GeodeticError::InvalidHeight(height_m));
        }
        Ok(Self {
            horizontal: GeographicPosition::new(latitude_deg, longitude_deg)?,
            height_m,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Egm2008Position {
    pub horizontal: GeographicPosition,
    pub orthometric_height_m: f64,
}

impl Egm2008Position {
    pub fn new(
        latitude_deg: f64,
        longitude_deg: f64,
        orthometric_height_m: f64,
    ) -> Result<Self, GeodeticError> {
        if !orthometric_height_m.is_finite() {
            return Err(GeodeticError::InvalidHeight(orthometric_height_m));
        }
        Ok(Self {
            horizontal: GeographicPosition::new(latitude_deg, longitude_deg)?,
            orthometric_height_m,
        })
    }

    /// Converts EGM2008 orthometric height `H` to WGS 84 ellipsoid height
    /// `h` using the explicit relation `h = H + N`.
    pub fn to_ellipsoid(self, geoid: &impl GeoidModel) -> Result<EllipsoidPosition, GeodeticError> {
        let undulation_m = geoid.undulation_m(self.horizontal)?;
        EllipsoidPosition::new(
            self.horizontal.latitude_deg,
            self.horizontal.longitude_deg,
            self.orthometric_height_m + undulation_m,
        )
    }
}

pub trait GeoidModel {
    fn undulation_m(&self, position: GeographicPosition) -> Result<f64, GeodeticError>;
}

/// A checked-in 5×5 EGM2008 sample over the Curaçao GLO-30 crop.
///
/// Values are geoid height above WGS 84, sampled with GeographicLib 2.5.2's
/// `egm2008-1` grid on 2026-08-09. Bilinear interpolation keeps this tracer
/// deterministic and avoids shipping the global 1-minute model.
#[derive(Clone, Copy, Debug, Default)]
pub struct CuracaoEgm2008;

impl CuracaoEgm2008 {
    pub const SOUTH_DEG: f64 = 12.0;
    pub const NORTH_DEG: f64 = 12.43;
    pub const WEST_DEG: f64 = -69.2;
    pub const EAST_DEG: f64 = -68.7;
    pub const WIDTH: usize = 5;
    pub const HEIGHT: usize = 5;
    /// Largest absolute error observed at the 16 cell midpoints against the
    /// same GeographicLib EGM2008 grid used to author the samples.
    pub const MEASURED_MIDPOINT_MAX_ERROR_M: f64 = 0.30;

    const VALUES_M: [[f64; Self::WIDTH]; Self::HEIGHT] = [
        [-24.4916, -24.2413, -23.5410, -23.1846, -23.2289],
        [-25.2083, -24.5193, -23.7719, -23.6387, -24.4299],
        [-25.8210, -24.8348, -24.7684, -25.3911, -26.2201],
        [-26.4483, -25.7212, -26.5127, -27.6706, -28.3029],
        [-27.3661, -27.3703, -28.3885, -29.5984, -30.3248],
    ];
}

impl GeoidModel for CuracaoEgm2008 {
    fn undulation_m(&self, position: GeographicPosition) -> Result<f64, GeodeticError> {
        if position.latitude_deg < Self::SOUTH_DEG
            || position.latitude_deg > Self::NORTH_DEG
            || position.longitude_deg < Self::WEST_DEG
            || position.longitude_deg > Self::EAST_DEG
        {
            return Err(GeodeticError::OutsideGeoidCoverage(position));
        }

        let grid_x = (position.longitude_deg - Self::WEST_DEG) / (Self::EAST_DEG - Self::WEST_DEG)
            * (Self::WIDTH - 1) as f64;
        let grid_y = (position.latitude_deg - Self::SOUTH_DEG)
            / (Self::NORTH_DEG - Self::SOUTH_DEG)
            * (Self::HEIGHT - 1) as f64;
        let x0 = (grid_x.floor() as usize).min(Self::WIDTH - 2);
        let y0 = (grid_y.floor() as usize).min(Self::HEIGHT - 2);
        let tx = (grid_x - x0 as f64).clamp(0.0, 1.0);
        let ty = (grid_y - y0 as f64).clamp(0.0, 1.0);
        let north = lerp(Self::VALUES_M[y0][x0], Self::VALUES_M[y0][x0 + 1], tx);
        let south = lerp(
            Self::VALUES_M[y0 + 1][x0],
            Self::VALUES_M[y0 + 1][x0 + 1],
            tx,
        );
        Ok(lerp(north, south, ty))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EcefPosition {
    pub x_m: f64,
    pub y_m: f64,
    pub z_m: f64,
}

impl EcefPosition {
    pub fn to_wgs84(self) -> Result<EllipsoidPosition, GeodeticError> {
        if !self.x_m.is_finite() || !self.y_m.is_finite() || !self.z_m.is_finite() {
            return Err(GeodeticError::InvalidEcef);
        }
        let a = WGS84_SEMI_MAJOR_AXIS_M;
        let flattening = 1.0 / WGS84_INVERSE_FLATTENING;
        let b = a * (1.0 - flattening);
        let eccentricity_squared = flattening * (2.0 - flattening);
        let second_eccentricity_squared = (a * a - b * b) / (b * b);
        let p = self.x_m.hypot(self.y_m);
        if p == 0.0 && self.z_m == 0.0 {
            return Err(GeodeticError::InvalidEcef);
        }
        let longitude = self.y_m.atan2(self.x_m);
        let theta = (self.z_m * a).atan2(p * b);
        let latitude = (self.z_m + second_eccentricity_squared * b * theta.sin().powi(3))
            .atan2(p - eccentricity_squared * a * theta.cos().powi(3));
        let prime_vertical = a / (1.0 - eccentricity_squared * latitude.sin().powi(2)).sqrt();
        let height_m = if latitude.cos().abs() > 1.0e-12 {
            p / latitude.cos() - prime_vertical
        } else {
            self.z_m / latitude.sin() - prime_vertical * (1.0 - eccentricity_squared)
        };
        EllipsoidPosition::new(latitude.to_degrees(), longitude.to_degrees(), height_m)
    }
}

impl From<EllipsoidPosition> for EcefPosition {
    fn from(position: EllipsoidPosition) -> Self {
        let flattening = 1.0 / WGS84_INVERSE_FLATTENING;
        let eccentricity_squared = flattening * (2.0 - flattening);
        let latitude = position.horizontal.latitude_deg.to_radians();
        let longitude = position.horizontal.longitude_deg.to_radians();
        let prime_vertical =
            WGS84_SEMI_MAJOR_AXIS_M / (1.0 - eccentricity_squared * latitude.sin().powi(2)).sqrt();
        let radial = (prime_vertical + position.height_m) * latitude.cos();
        Self {
            x_m: radial * longitude.cos(),
            y_m: radial * longitude.sin(),
            z_m: (prime_vertical * (1.0 - eccentricity_squared) + position.height_m)
                * latitude.sin(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnuPosition {
    pub east_m: f64,
    pub north_m: f64,
    pub up_m: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct LocalTangentFrame {
    origin: EcefPosition,
    east: [f64; 3],
    north: [f64; 3],
    up: [f64; 3],
}

impl LocalTangentFrame {
    pub fn new(origin: EllipsoidPosition) -> Self {
        let latitude = origin.horizontal.latitude_deg.to_radians();
        let longitude = origin.horizontal.longitude_deg.to_radians();
        Self {
            origin: origin.into(),
            east: [-longitude.sin(), longitude.cos(), 0.0],
            north: [
                -latitude.sin() * longitude.cos(),
                -latitude.sin() * longitude.sin(),
                latitude.cos(),
            ],
            up: [
                latitude.cos() * longitude.cos(),
                latitude.cos() * longitude.sin(),
                latitude.sin(),
            ],
        }
    }

    pub fn to_enu(self, position: EcefPosition) -> EnuPosition {
        let delta = [
            position.x_m - self.origin.x_m,
            position.y_m - self.origin.y_m,
            position.z_m - self.origin.z_m,
        ];
        EnuPosition {
            east_m: dot(delta, self.east),
            north_m: dot(delta, self.north),
            up_m: dot(delta, self.up),
        }
    }

    pub fn to_ecef(self, position: EnuPosition) -> EcefPosition {
        EcefPosition {
            x_m: self.origin.x_m
                + self.east[0] * position.east_m
                + self.north[0] * position.north_m
                + self.up[0] * position.up_m,
            y_m: self.origin.y_m
                + self.east[1] * position.east_m
                + self.north[1] * position.north_m
                + self.up[1] * position.up_m,
            z_m: self.origin.z_m
                + self.east[2] * position.east_m
                + self.north[2] * position.north_m
                + self.up[2] * position.up_m,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UtmPosition {
    pub zone: u8,
    pub easting_m: f64,
    pub northing_m: f64,
}

impl UtmPosition {
    pub fn new_north(zone: u8, easting_m: f64, northing_m: f64) -> Result<Self, GeodeticError> {
        validate_zone(zone)?;
        if !easting_m.is_finite() || !northing_m.is_finite() {
            return Err(GeodeticError::InvalidUtm);
        }
        Ok(Self {
            zone,
            easting_m,
            northing_m,
        })
    }

    pub fn to_wgs84(self) -> Result<GeographicPosition, GeodeticError> {
        let (latitude_deg, longitude_deg) = utm_inverse(self.easting_m, self.northing_m, self.zone);
        GeographicPosition::new(latitude_deg, longitude_deg)
    }
}

pub fn wgs84_to_utm_north(
    position: GeographicPosition,
    zone: u8,
) -> Result<UtmPosition, GeodeticError> {
    validate_zone(zone)?;
    if position.latitude_deg < 0.0 || position.latitude_deg > 84.0 {
        return Err(GeodeticError::OutsideNorthernUtm(position.latitude_deg));
    }
    let (easting_m, northing_m) = utm_forward(position.latitude_deg, position.longitude_deg, zone);
    UtmPosition::new_north(zone, easting_m, northing_m)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GeodeticError {
    InvalidLatitude(f64),
    InvalidLongitude(f64),
    InvalidHeight(f64),
    InvalidUtmZone(u8),
    InvalidUtm,
    InvalidEcef,
    OutsideNorthernUtm(f64),
    OutsideGeoidCoverage(GeographicPosition),
}

impl fmt::Display for GeodeticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLatitude(value) => write!(formatter, "invalid latitude {value}"),
            Self::InvalidLongitude(value) => write!(formatter, "invalid longitude {value}"),
            Self::InvalidHeight(value) => write!(formatter, "invalid height {value}"),
            Self::InvalidUtmZone(value) => write!(formatter, "invalid UTM zone {value}"),
            Self::InvalidUtm => formatter.write_str("UTM coordinates must be finite"),
            Self::InvalidEcef => formatter.write_str("ECEF coordinates are invalid"),
            Self::OutsideNorthernUtm(value) => {
                write!(formatter, "latitude {value} is outside northern UTM")
            }
            Self::OutsideGeoidCoverage(position) => write!(
                formatter,
                "coordinate {}, {} is outside the regional EGM2008 grid",
                position.latitude_deg, position.longitude_deg
            ),
        }
    }
}

impl Error for GeodeticError {}

fn validate_zone(zone: u8) -> Result<(), GeodeticError> {
    if (1..=60).contains(&zone) {
        Ok(())
    } else {
        Err(GeodeticError::InvalidUtmZone(zone))
    }
}

// WGS 84 Transverse Mercator equations, specialized to northern UTM zones.
fn utm_forward(lat_deg: f64, lon_deg: f64, zone: u8) -> (f64, f64) {
    let flattening = 1.0 / WGS84_INVERSE_FLATTENING;
    let eccentricity_squared = flattening * (2.0 - flattening);
    let latitude = lat_deg.to_radians();
    let longitude = lon_deg.to_radians();
    let longitude_origin = ((zone as f64 - 1.0) * 6.0 - 180.0 + 3.0).to_radians();
    let second_eccentricity_squared = eccentricity_squared / (1.0 - eccentricity_squared);
    let n = WGS84_SEMI_MAJOR_AXIS_M / (1.0 - eccentricity_squared * latitude.sin().powi(2)).sqrt();
    let t = latitude.tan().powi(2);
    let c = second_eccentricity_squared * latitude.cos().powi(2);
    let a = latitude.cos() * (longitude - longitude_origin);
    let m = WGS84_SEMI_MAJOR_AXIS_M
        * ((1.0
            - eccentricity_squared / 4.0
            - 3.0 * eccentricity_squared.powi(2) / 64.0
            - 5.0 * eccentricity_squared.powi(3) / 256.0)
            * latitude
            - (3.0 * eccentricity_squared / 8.0
                + 3.0 * eccentricity_squared.powi(2) / 32.0
                + 45.0 * eccentricity_squared.powi(3) / 1024.0)
                * (2.0 * latitude).sin()
            + (15.0 * eccentricity_squared.powi(2) / 256.0
                + 45.0 * eccentricity_squared.powi(3) / 1024.0)
                * (4.0 * latitude).sin()
            - 35.0 * eccentricity_squared.powi(3) / 3072.0 * (6.0 * latitude).sin());
    let easting_m = UTM_SCALE
        * n
        * (a + (1.0 - t + c) * a.powi(3) / 6.0
            + (5.0 - 18.0 * t + t.powi(2) + 72.0 * c - 58.0 * second_eccentricity_squared)
                * a.powi(5)
                / 120.0)
        + 500_000.0;
    let northing_m = UTM_SCALE
        * (m + n
            * latitude.tan()
            * (a.powi(2) / 2.0
                + (5.0 - t + 9.0 * c + 4.0 * c.powi(2)) * a.powi(4) / 24.0
                + (61.0 - 58.0 * t + t.powi(2) + 600.0 * c - 330.0 * second_eccentricity_squared)
                    * a.powi(6)
                    / 720.0));
    (easting_m, northing_m)
}

fn utm_inverse(easting_m: f64, northing_m: f64, zone: u8) -> (f64, f64) {
    let flattening = 1.0 / WGS84_INVERSE_FLATTENING;
    let eccentricity_squared = flattening * (2.0 - flattening);
    let second_eccentricity_squared = eccentricity_squared / (1.0 - eccentricity_squared);
    let e1 =
        (1.0 - (1.0 - eccentricity_squared).sqrt()) / (1.0 + (1.0 - eccentricity_squared).sqrt());
    let x = easting_m - 500_000.0;
    let longitude_origin = ((zone as f64 - 1.0) * 6.0 - 180.0 + 3.0).to_radians();
    let m = northing_m / UTM_SCALE;
    let mu = m
        / (WGS84_SEMI_MAJOR_AXIS_M
            * (1.0
                - eccentricity_squared / 4.0
                - 3.0 * eccentricity_squared.powi(2) / 64.0
                - 5.0 * eccentricity_squared.powi(3) / 256.0));
    let phi1 = mu
        + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
        + (21.0 * e1.powi(2) / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
        + 151.0 * e1.powi(3) / 96.0 * (6.0 * mu).sin()
        + 1097.0 * e1.powi(4) / 512.0 * (8.0 * mu).sin();
    let n1 = WGS84_SEMI_MAJOR_AXIS_M / (1.0 - eccentricity_squared * phi1.sin().powi(2)).sqrt();
    let t1 = phi1.tan().powi(2);
    let c1 = second_eccentricity_squared * phi1.cos().powi(2);
    let r1 = WGS84_SEMI_MAJOR_AXIS_M * (1.0 - eccentricity_squared)
        / (1.0 - eccentricity_squared * phi1.sin().powi(2)).powf(1.5);
    let d = x / (n1 * UTM_SCALE);
    let latitude = phi1
        - (n1 * phi1.tan() / r1)
            * (d.powi(2) / 2.0
                - (5.0 + 3.0 * t1 + 10.0 * c1
                    - 4.0 * c1.powi(2)
                    - 9.0 * second_eccentricity_squared)
                    * d.powi(4)
                    / 24.0
                + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1.powi(2)
                    - 252.0 * second_eccentricity_squared
                    - 3.0 * c1.powi(2))
                    * d.powi(6)
                    / 720.0);
    let longitude = longitude_origin
        + (d - (1.0 + 2.0 * t1 + c1) * d.powi(3) / 6.0
            + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1.powi(2)
                + 8.0 * second_eccentricity_squared
                + 24.0 * t1.powi(2))
                * d.powi(5)
                / 120.0)
            / phi1.cos();
    (latitude.to_degrees(), longitude.to_degrees())
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn lerp(start: f64, end: f64, amount: f64) -> f64 {
    start + (end - start) * amount
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual} (tolerance {tolerance})"
        );
    }

    #[test]
    fn curacao_utm_round_trips() {
        for (latitude, longitude) in [(12.1696, -68.99), (12.38, -69.15), (12.04, -68.75)] {
            let original = GeographicPosition::new(latitude, longitude).unwrap();
            let utm = wgs84_to_utm_north(original, 19).unwrap();
            let round_trip = utm.to_wgs84().unwrap();
            assert_close(round_trip.latitude_deg, latitude, 1.0e-7);
            assert_close(round_trip.longitude_deg, longitude, 1.0e-7);
        }
    }

    #[test]
    fn wgs84_ecef_round_trips() {
        for original in [
            EllipsoidPosition::new(12.1696, -68.99, -24.7).unwrap(),
            EllipsoidPosition::new(0.0, 0.0, 0.0).unwrap(),
            EllipsoidPosition::new(89.0, 45.0, 1_000.0).unwrap(),
        ] {
            let round_trip = EcefPosition::from(original).to_wgs84().unwrap();
            assert_close(
                round_trip.horizontal.latitude_deg,
                original.horizontal.latitude_deg,
                1.0e-9,
            );
            assert_close(
                round_trip.horizontal.longitude_deg,
                original.horizontal.longitude_deg,
                1.0e-9,
            );
            assert_close(round_trip.height_m, original.height_m, 1.0e-5);
        }
    }

    #[test]
    fn egm2008_conversion_has_explicit_sign() {
        let orthometric = Egm2008Position::new(12.215, -68.95, 100.0).unwrap();
        let ellipsoid = orthometric.to_ellipsoid(&CuracaoEgm2008).unwrap();
        assert_close(ellipsoid.height_m, 75.2316, 1.0e-10);
    }

    #[test]
    fn local_tangent_frame_round_trips() {
        let origin = Egm2008Position::new(12.215, -68.95, 0.0)
            .unwrap()
            .to_ellipsoid(&CuracaoEgm2008)
            .unwrap();
        let frame = LocalTangentFrame::new(origin);
        let point = Egm2008Position::new(12.25, -68.9, 320.0)
            .unwrap()
            .to_ellipsoid(&CuracaoEgm2008)
            .unwrap();
        let ecef = EcefPosition::from(point);
        let round_trip = frame.to_ecef(frame.to_enu(ecef));
        assert_close(round_trip.x_m, ecef.x_m, 1.0e-9);
        assert_close(round_trip.y_m, ecef.y_m, 1.0e-9);
        assert_close(round_trip.z_m, ecef.z_m, 1.0e-9);
    }

    #[test]
    fn regional_geoid_rejects_silent_extrapolation() {
        let outside = GeographicPosition::new(13.0, -68.95).unwrap();
        assert!(matches!(
            CuracaoEgm2008.undulation_m(outside),
            Err(GeodeticError::OutsideGeoidCoverage(_))
        ));
    }

    #[test]
    fn regional_geoid_meets_measured_midpoint_budget() {
        let latitudes = [12.05375, 12.16125, 12.26875, 12.37625];
        let longitudes = [-69.1375, -69.0125, -68.8875, -68.7625];
        let references_m = [
            [-24.6953, -23.9877, -23.3428, -23.4056],
            [-25.1114, -24.2565, -24.1997, -24.7851],
            [-25.5560, -25.2272, -26.1124, -26.9349],
            [-26.5451, -26.7051, -28.1445, -29.0565],
        ];

        for (y, latitude) in latitudes.into_iter().enumerate() {
            for (x, longitude) in longitudes.into_iter().enumerate() {
                let actual = CuracaoEgm2008
                    .undulation_m(GeographicPosition::new(latitude, longitude).unwrap())
                    .unwrap();
                assert_close(
                    actual,
                    references_m[y][x],
                    CuracaoEgm2008::MEASURED_MIDPOINT_MAX_ERROR_M,
                );
            }
        }
    }

    #[test]
    fn local_f32_render_coordinates_stay_within_centimetre_budget() {
        let origin = Egm2008Position::new(12.215, -68.95, 0.0)
            .unwrap()
            .to_ellipsoid(&CuracaoEgm2008)
            .unwrap();
        let frame = LocalTangentFrame::new(origin);
        for point in [
            Egm2008Position::new(12.0, -69.2, 0.0).unwrap(),
            Egm2008Position::new(12.43, -68.7, 1_000.0).unwrap(),
            Egm2008Position::new(12.215, -68.95, 300.0).unwrap(),
        ] {
            let enu = frame.to_enu(EcefPosition::from(
                point.to_ellipsoid(&CuracaoEgm2008).unwrap(),
            ));
            for value in [enu.east_m, enu.north_m, enu.up_m] {
                let error_m = (f64::from(value as f32) - value).abs();
                assert!(error_m <= 0.01, "f32 conversion lost {error_m} m");
            }
        }
    }
}
