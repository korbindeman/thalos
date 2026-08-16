//! Lightweight capability facade shared by Thalos applications.
//!
//! The crate has no default features. Applications opt into coarse bundles;
//! disabled bundles are absent from the dependency graph rather than dormant
//! at runtime. The existing complete game composition is retained behind
//! [`game`](crate#game), while `interactive` selects only lightweight shared
//! application capabilities such as preferences, the viewer, and photo mode.

/// Product-level capabilities compiled into this application.
///
/// These names describe the stable application contract. The transitional
/// `game` Cargo feature currently supplies interaction, simulation, gameplay,
/// and planetary composition together until their implementations finish
/// moving behind independent crate boundaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeCapability {
    Base,
    Interactive,
    Simulation,
    Gameplay,
    Planetary,
    Capture,
}

impl RuntimeCapability {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Interactive => "interactive",
            Self::Simulation => "simulation",
            Self::Gameplay => "gameplay",
            Self::Planetary => "planetary",
            Self::Capture => "capture",
        }
    }
}

/// Capabilities present in this compiled facade.
pub const COMPILED_CAPABILITIES: &[RuntimeCapability] = &[
    RuntimeCapability::Base,
    #[cfg(feature = "interactive")]
    RuntimeCapability::Interactive,
    #[cfg(feature = "game")]
    RuntimeCapability::Simulation,
    #[cfg(feature = "game")]
    RuntimeCapability::Gameplay,
    #[cfg(feature = "game")]
    RuntimeCapability::Planetary,
    #[cfg(feature = "capture")]
    RuntimeCapability::Capture,
];

/// Shared window, graphics, persistence, and settings-menu capability.
#[cfg(feature = "interactive")]
pub use thalos_preferences as preferences;

/// Shared F3 state, frame history, panel/graph, and extension seam.
#[cfg(feature = "interactive")]
pub use thalos_diagnostics_ui as diagnostics_ui;

/// Shared visual language and native UI building blocks.
#[cfg(feature = "interactive")]
pub use thalos_ui as ui;

/// Shared F1 clean-view state and visibility arbitration.
#[cfg(feature = "interactive")]
pub use thalos_photo_mode as photo_mode;

/// Shared physical optics, freecam motion, and viewer control surface.
#[cfg(feature = "interactive")]
pub use thalos_viewer as viewer;

/// The complete canonical game API, available only to applications that select
/// the explicit `game` bundle.
#[cfg(feature = "game")]
pub use thalos_game_runtime::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_is_always_present() {
        assert_eq!(COMPILED_CAPABILITIES[0], RuntimeCapability::Base);
    }

    #[test]
    fn capability_names_are_unique() {
        for (index, capability) in COMPILED_CAPABILITIES.iter().enumerate() {
            assert!(
                COMPILED_CAPABILITIES[..index]
                    .iter()
                    .all(|other| other.name() != capability.name()),
                "duplicate capability {}",
                capability.name()
            );
        }
    }
}
