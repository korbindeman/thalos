//! Shared camera post-processing stack for space views (game + planet editor).
//!
//! Returns a bundle of components to attach alongside `Camera3d` on any camera
//! that renders planets from vacuum. Keeps both binaries visually consistent.

use bevy::anti_alias::contrast_adaptive_sharpening::ContrastAdaptiveSharpening;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::post_process::auto_exposure::AutoExposure;
use bevy::post_process::bloom::{Bloom, BloomCompositeMode, BloomPrefilter};
use bevy::post_process::effect_stack::ChromaticAberration;
use bevy::prelude::*;
use bevy::render::view::Hdr;

use crate::film_grain::FilmGrain;

/// Components to attach to a `Camera3d` entity for the space post stack:
/// HDR + TonyMcMapface tonemap, subtle bloom, auto exposure metered against
/// the lit planet face (voids ignored), CAS sharpening, mild chromatic
/// aberration.
pub fn space_camera_post_stack() -> impl Bundle {
    (
        Hdr,
        Tonemapping::TonyMcMapface,
        Bloom {
            intensity: 0.25,
            low_frequency_boost: 0.0,
            low_frequency_boost_curvature: 0.0,
            high_pass_frequency: 1.0,
            prefilter: BloomPrefilter {
                threshold: 1.0,
                threshold_softness: 0.2,
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
