//! Compatibility aliases for the original Vaelen cold-desert surface field.

pub use crate::cold_desert_field::{
    COLD_DESERT_BIOME_COUNT as VAELEN_BIOME_COUNT,
    COLD_DESERT_MAT_DARK_BASALT as VAELEN_MAT_DARK_BASALT,
    COLD_DESERT_MAT_DUNE_SAND as VAELEN_MAT_DUNE_SAND,
    COLD_DESERT_MAT_EVAPORITE as VAELEN_MAT_EVAPORITE,
    COLD_DESERT_MAT_PALE_SEDIMENT as VAELEN_MAT_PALE_SEDIMENT,
    COLD_DESERT_MAT_RUST_DUST as VAELEN_MAT_RUST_DUST,
    COLD_DESERT_SHIELD_VOLCANO_HEIGHT_M as VAELEN_SHIELD_VOLCANO_HEIGHT_M,
    COLD_DESERT_SHIELD_VOLCANO_RADIUS_M as VAELEN_SHIELD_VOLCANO_RADIUS_M,
    ColdDesertBiome as VaelenBiome, ColdDesertBiomeWeights as VaelenBiomeWeights,
    ColdDesertDuneBasinMask as VaelenDuneBasinMask, ColdDesertField as VaelenColdDesertField,
    ColdDesertSutureDebug as VaelenSutureDebug,
    cold_desert_relief_palettes as vaelen_relief_palettes,
    sample_cold_desert_biomes as sample_vaelen_biomes,
};
