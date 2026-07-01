//! Shared camera post-processing stack for space views (game + planet editor).
//!
//! Returns a bundle of components to attach alongside `Camera3d` on any camera
//! that renders planets from vacuum. Keeps both binaries visually consistent.

use bevy::anti_alias::{
    contrast_adaptive_sharpening::ContrastAdaptiveSharpening,
    smaa::{Smaa, SmaaPreset},
};
use bevy::core_pipeline::tonemapping::{DebandDither, Tonemapping};
use bevy::post_process::bloom::{Bloom, BloomCompositeMode, BloomPrefilter};
use bevy::post_process::effect_stack::ChromaticAberration;
use bevy::prelude::*;
use bevy::camera::Hdr;
use bevy::render::view::{ColorGrading, ColorGradingGlobal, ColorGradingSection, Msaa};

use crate::impostor::film_grain::FilmGrain;

/// Components to attach to a `Camera3d` entity for the space post stack:
/// HDR + AgX tonemap, subtle bloom, conservative color grading, SMAA, CAS
/// sharpening, mild chromatic aberration, and exposure-driven film grain.
///
/// **No auto-exposure** (graphics-fidelity F2 — `docs/graphics_fidelity.md` §3).
/// The Bevy `AutoExposure` histogram was retired because it was a second
/// exposure authority compounding with `CameraExposure`: it wrote a global
/// `color_grading.exposure` at tonemap on top of the distance gain
/// `CameraExposure` already applies at every surface's flux/illuminance input.
/// Brightness is now governed *solely* by that input gain (the artist distance
/// curve), plus the fixed global-exposure baseline in
/// [`space_camera_color_grading`]. This is what lets the hull, terrain, and sky
/// track one exposure instead of the histogram floating the whole buffer.
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
        // AgX: filmic, neutral highlight rolloff — pulls saturated daylight
        // terrain/sky back toward a photoreal look (closer to the MSFS
        // reference than TonyMcMapface's punchier transform). Global to all
        // space/surface views; revert to `TonyMcMapface` here to A/B.
        Tonemapping::AgX,
        // Intentional post-process dithering. Do not disable this when
        // investigating generic atmosphere/sky "noise" reports; it is part of
        // the desired camera look and should only change when explicitly named.
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
        // Intentional exposure-driven sensor grain. Do not disable this when
        // investigating generic atmosphere/sky "noise" reports; it is part of
        // the desired camera look and should only change when explicitly named.
        FilmGrain::default(),
    )
}

/// Fixed global exposure baseline, in EV stops applied at tonemap
/// (`color_grading.exposure`). This is the single knob that replaced the
/// retired `AutoExposure` histogram (graphics-fidelity F2): a constant, not a
/// per-frame metered value, so the scene no longer auto-adapts. `0.0` = identity
/// (the raw scene radiance already carries `CameraExposure`'s distance gain at
/// the input). Nudge **negative** if the substellar-noon surface reads too hot,
/// **positive** to lift a dim scene. Tune from a `just game runway` / `orbit`
/// screenshot.
const GLOBAL_EXPOSURE_STOPS: f32 = 0.0;

fn space_camera_color_grading() -> ColorGrading {
    ColorGrading {
        global: ColorGradingGlobal {
            // Stay close to Bevy's neutral defaults. Larger section changes
            // visibly band planet terminators after tonemapping.
            post_saturation: 0.995,
            // Fixed exposure baseline (F2) — replaces the AutoExposure float.
            exposure: GLOBAL_EXPOSURE_STOPS,
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
