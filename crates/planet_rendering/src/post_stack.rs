//! Shared camera post-processing stack for space views (game + planet editor).
//!
//! Returns a bundle of components to attach alongside `Camera3d` on any camera
//! that renders planets from vacuum. Keeps both binaries visually consistent.

use bevy::anti_alias::{
    contrast_adaptive_sharpening::ContrastAdaptiveSharpening,
    smaa::{Smaa, SmaaPreset},
};
use bevy::core_pipeline::tonemapping::{DebandDither, Tonemapping};
use bevy::post_process::auto_exposure::AutoExposure;
use bevy::post_process::bloom::{Bloom, BloomCompositeMode, BloomPrefilter};
use bevy::post_process::effect_stack::ChromaticAberration;
use bevy::prelude::*;
use bevy::render::view::{ColorGrading, ColorGradingGlobal, ColorGradingSection, Hdr, Msaa};

use crate::film_grain::FilmGrain;

/// Components to attach to a `Camera3d` entity for the space post stack:
/// HDR + TonyMcMapface tonemap, subtle bloom, auto exposure metered against
/// the lit planet face (voids ignored), conservative color grading, SMAA,
/// CAS sharpening, mild chromatic aberration, and exposure-driven film grain.
pub fn space_camera_post_stack() -> impl Bundle {
    (
        // The game renders many shader impostors, thin line overlays, and UI
        // composites. Prefer a stable post AA pass over MSAA or TAA until the
        // depth/motion-vector story is explicit across those passes.
        Msaa::Off,
        Smaa {
            preset: SmaaPreset::High,
        },
        Hdr,
        Tonemapping::TonyMcMapface,
        DebandDither::Enabled,
        space_camera_color_grading(),
        Bloom {
            intensity: 0.35,
            low_frequency_boost: 0.0,
            low_frequency_boost_curvature: 0.0,
            high_pass_frequency: 1.0,
            prefilter: BloomPrefilter {
                threshold: 0.6,
                threshold_softness: 0.3,
            },
            composite_mode: BloomCompositeMode::Additive,
            ..Bloom::NATURAL
        },
        // Vacuum scene: most of frame is pure black, so ignore the lower
        // half of the histogram — meter against the lit planet face rather
        // than the void.
        // Upper bound is deliberately tight (was 12). Beyond ~6 EV, the
        // metering would drag dim outer-system bodies back to mid-gray on
        // top of `CameraExposure`'s manual gain — two compensations
        // compounding. Capping the headroom here lets deep space stay
        // visibly underexposed and sensor-limited instead.
        AutoExposure {
            range: -4.0..=6.0,
            filter: 0.30..=0.95,
            speed_brighten: 2.0,
            speed_darken: 1.0,
            ..default()
        },
        ContrastAdaptiveSharpening {
            enabled: true,
            sharpening_strength: 0.3,
            denoise: false,
        },
        ChromaticAberration {
            intensity: 0.003,
            max_samples: 8,
            color_lut: None,
        },
        FilmGrain::default(),
    )
}

fn space_camera_color_grading() -> ColorGrading {
    ColorGrading {
        global: ColorGradingGlobal {
            // Stay close to Bevy's neutral defaults. Larger section changes
            // visibly band planet terminators after tonemapping.
            post_saturation: 0.995,
            ..default()
        },
        // Keep the shadow enrichment that helps Thalos read, but make it
        // small enough that low-gradient terminators do not posterize.
        shadows: ColorGradingSection {
            contrast: 1.012,
            ..default()
        },
        midtones: ColorGradingSection::default(),
        highlights: ColorGradingSection::default(),
    }
}
