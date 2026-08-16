use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialAdapter {
    LocalPlanar,
    Planetary,
    GeodeticEllipsoid,
}

impl SpatialAdapter {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LocalPlanar => "local_planar",
            Self::Planetary => "planetary",
            Self::GeodeticEllipsoid => "geodetic_ellipsoid",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerrainAdapter {
    PlanarRtin,
    CubeSphereTiles,
    LegacyUdlod,
    GeodeticHeightfield,
}

impl TerrainAdapter {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PlanarRtin => "planar_rtin",
            Self::CubeSphereTiles => "cube_sphere_tiles",
            Self::LegacyUdlod => "legacy_udlod",
            Self::GeodeticHeightfield => "geodetic_heightfield",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AtmosphereAdapter {
    None,
    BevyEarth,
    PlanetaryRaymarch,
}

impl AtmosphereAdapter {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::BevyEarth => "bevy_earth",
            Self::PlanetaryRaymarch => "planetary_raymarch",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OceanAdapter {
    None,
    PlanarClipmap,
    AnalyticPlanet,
}

impl OceanAdapter {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PlanarClipmap => "planar_clipmap",
            Self::AnalyticPlanet => "analytic_planet",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CloudAdapter {
    None,
    PlanetaryVolume,
}

impl CloudAdapter {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PlanetaryVolume => "planetary_volume",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LightingAdapter {
    BevyStandard,
    Planetary,
}

impl LightingAdapter {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BevyStandard => "bevy_standard",
            Self::Planetary => "planetary",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FarBodyAdapter {
    None,
    Impostor,
}

impl FarBodyAdapter {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Impostor => "impostor",
        }
    }
}

/// Restart-time renderer composition selected by an application.
///
/// This is deliberately a concrete capability record rather than a renderer
/// trait. Applications validate it once and install the matching plugins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderPlan {
    pub spatial: SpatialAdapter,
    pub terrain: TerrainAdapter,
    pub atmosphere: AtmosphereAdapter,
    pub ocean: OceanAdapter,
    pub clouds: CloudAdapter,
    pub lighting: LightingAdapter,
    pub far_body: FarBodyAdapter,
}

impl RenderPlan {
    pub const fn thalos_tiles() -> Self {
        Self {
            spatial: SpatialAdapter::Planetary,
            terrain: TerrainAdapter::CubeSphereTiles,
            atmosphere: AtmosphereAdapter::PlanetaryRaymarch,
            ocean: OceanAdapter::AnalyticPlanet,
            clouds: CloudAdapter::PlanetaryVolume,
            lighting: LightingAdapter::Planetary,
            far_body: FarBodyAdapter::Impostor,
        }
    }

    pub const fn thalos_legacy_udlod() -> Self {
        Self {
            terrain: TerrainAdapter::LegacyUdlod,
            ..Self::thalos_tiles()
        }
    }

    pub const fn korsou_planar() -> Self {
        Self {
            spatial: SpatialAdapter::LocalPlanar,
            terrain: TerrainAdapter::PlanarRtin,
            atmosphere: AtmosphereAdapter::BevyEarth,
            ocean: OceanAdapter::PlanarClipmap,
            clouds: CloudAdapter::None,
            lighting: LightingAdapter::BevyStandard,
            far_body: FarBodyAdapter::None,
        }
    }

    pub const fn korsou_geodetic() -> Self {
        Self {
            spatial: SpatialAdapter::GeodeticEllipsoid,
            terrain: TerrainAdapter::GeodeticHeightfield,
            ocean: OceanAdapter::None,
            ..Self::korsou_planar()
        }
    }

    pub fn validate(
        self,
        capabilities: RenderCapabilities,
    ) -> Result<ValidatedRenderPlan, RenderPlanError> {
        if !capabilities.supports_spatial(self.spatial) {
            return Err(RenderPlanError::CapabilityUnavailable(
                self.spatial.as_str(),
            ));
        }
        if self.far_body == FarBodyAdapter::Impostor && !capabilities.far_body {
            return Err(RenderPlanError::CapabilityUnavailable(
                self.far_body.as_str(),
            ));
        }
        if self.terrain == TerrainAdapter::LegacyUdlod && !capabilities.legacy_udlod {
            return Err(RenderPlanError::CapabilityUnavailable(
                self.terrain.as_str(),
            ));
        }

        let compatible = match self.spatial {
            SpatialAdapter::LocalPlanar => {
                self.terrain == TerrainAdapter::PlanarRtin
                    && matches!(
                        self.atmosphere,
                        AtmosphereAdapter::None | AtmosphereAdapter::BevyEarth
                    )
                    && matches!(self.ocean, OceanAdapter::None | OceanAdapter::PlanarClipmap)
                    && self.clouds == CloudAdapter::None
                    && self.lighting == LightingAdapter::BevyStandard
                    && self.far_body == FarBodyAdapter::None
            }
            SpatialAdapter::Planetary => {
                matches!(
                    self.terrain,
                    TerrainAdapter::CubeSphereTiles | TerrainAdapter::LegacyUdlod
                ) && matches!(
                    self.atmosphere,
                    AtmosphereAdapter::None | AtmosphereAdapter::PlanetaryRaymarch
                ) && matches!(
                    self.ocean,
                    OceanAdapter::None | OceanAdapter::AnalyticPlanet
                ) && matches!(
                    self.clouds,
                    CloudAdapter::None | CloudAdapter::PlanetaryVolume
                ) && self.lighting == LightingAdapter::Planetary
            }
            SpatialAdapter::GeodeticEllipsoid => {
                self.terrain == TerrainAdapter::GeodeticHeightfield
                    && matches!(
                        self.atmosphere,
                        AtmosphereAdapter::None | AtmosphereAdapter::BevyEarth
                    )
                    && matches!(self.ocean, OceanAdapter::None | OceanAdapter::PlanarClipmap)
                    && self.clouds == CloudAdapter::None
                    && self.lighting == LightingAdapter::BevyStandard
            }
        };
        if !compatible {
            return Err(RenderPlanError::IncompatibleComposition {
                spatial: self.spatial,
                terrain: self.terrain,
                atmosphere: self.atmosphere,
                ocean: self.ocean,
                clouds: self.clouds,
                lighting: self.lighting,
                far_body: self.far_body,
            });
        }

        Ok(ValidatedRenderPlan(self))
    }
}

/// Adapter families compiled into one application binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderCapabilities {
    pub local_planar: bool,
    pub planetary: bool,
    pub geodetic_ellipsoid: bool,
    pub far_body: bool,
    pub legacy_udlod: bool,
}

impl RenderCapabilities {
    pub const THALOS_WITH_LEGACY: Self = Self {
        local_planar: false,
        planetary: true,
        geodetic_ellipsoid: false,
        far_body: true,
        legacy_udlod: true,
    };

    pub const THALOS: Self = Self {
        legacy_udlod: false,
        ..Self::THALOS_WITH_LEGACY
    };

    pub const KORSOU: Self = Self {
        local_planar: true,
        planetary: false,
        geodetic_ellipsoid: true,
        far_body: false,
        legacy_udlod: false,
    };

    const fn supports_spatial(self, spatial: SpatialAdapter) -> bool {
        match spatial {
            SpatialAdapter::LocalPlanar => self.local_planar,
            SpatialAdapter::Planetary => self.planetary,
            SpatialAdapter::GeodeticEllipsoid => self.geodetic_ellipsoid,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidatedRenderPlan(RenderPlan);

impl ValidatedRenderPlan {
    pub const fn plan(self) -> RenderPlan {
        self.0
    }

    pub fn uses_tile_terrain(self) -> bool {
        self.0.terrain == TerrainAdapter::CubeSphereTiles
    }

    pub fn uses_legacy_udlod(self) -> bool {
        self.0.terrain == TerrainAdapter::LegacyUdlod
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderPlanError {
    CapabilityUnavailable(&'static str),
    IncompatibleComposition {
        spatial: SpatialAdapter,
        terrain: TerrainAdapter,
        atmosphere: AtmosphereAdapter,
        ocean: OceanAdapter,
        clouds: CloudAdapter,
        lighting: LightingAdapter,
        far_body: FarBodyAdapter,
    },
}

impl fmt::Display for RenderPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapabilityUnavailable(capability) => {
                write!(
                    formatter,
                    "render capability `{capability}` is not compiled into this application"
                )
            }
            Self::IncompatibleComposition {
                spatial,
                terrain,
                atmosphere,
                ocean,
                clouds,
                lighting,
                far_body,
            } => write!(
                formatter,
                "incompatible render plan: spatial={} terrain={} atmosphere={} ocean={} clouds={} lighting={} far_body={}",
                spatial.as_str(),
                terrain.as_str(),
                atmosphere.as_str(),
                ocean.as_str(),
                clouds.as_str(),
                lighting.as_str(),
                far_body.as_str(),
            ),
        }
    }
}

impl std::error::Error for RenderPlanError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_plans_validate_against_their_applications() {
        assert!(
            RenderPlan::thalos_tiles()
                .validate(RenderCapabilities::THALOS)
                .is_ok()
        );
        assert!(
            RenderPlan::thalos_legacy_udlod()
                .validate(RenderCapabilities::THALOS_WITH_LEGACY)
                .is_ok()
        );
        assert!(
            RenderPlan::korsou_planar()
                .validate(RenderCapabilities::KORSOU)
                .is_ok()
        );
        assert!(
            RenderPlan::korsou_geodetic()
                .validate(RenderCapabilities::KORSOU)
                .is_ok()
        );
    }

    #[test]
    fn unavailable_legacy_renderer_is_rejected() {
        assert_eq!(
            RenderPlan::thalos_legacy_udlod().validate(RenderCapabilities::THALOS),
            Err(RenderPlanError::CapabilityUnavailable("legacy_udlod"))
        );
    }

    #[test]
    fn spatially_mixed_plan_is_rejected() {
        let mixed = RenderPlan {
            terrain: TerrainAdapter::PlanarRtin,
            ..RenderPlan::thalos_tiles()
        };
        assert!(matches!(
            mixed.validate(RenderCapabilities::THALOS_WITH_LEGACY),
            Err(RenderPlanError::IncompatibleComposition { .. })
        ));
    }
}
