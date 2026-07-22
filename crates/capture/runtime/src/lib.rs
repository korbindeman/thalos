//! Headless application shell for the canonical Thalos runtime.
//!
//! The capture systems still execute inside `thalos_runtime`; this package is
//! the stable host-facing boundary while those tightly coupled systems are
//! progressively projected onto public runtime services.

use bevy::prelude::App;

/// Process exit code for a capture that completed but logged at least one
/// `ERROR`. Distinct from 1 so a caller can tell "the render was invalid" from
/// "the process failed to start".
pub const EXIT_CAPTURE_INVALID: i32 = 3;

#[derive(Clone, Copy, Debug, Default)]
pub struct CaptureAppBuilder;

impl CaptureAppBuilder {
    pub const fn new() -> Self {
        Self
    }

    pub fn build(self) -> App {
        assert!(
            std::env::var_os("THALOS_SCREENSHOT").is_some(),
            "thalos_capture_host requires a capture request (set THALOS_SCREENSHOT)"
        );
        thalos_runtime::AppBuilder::new().build()
    }

    /// Run the capture and exit.
    ///
    /// Exits [`EXIT_CAPTURE_INVALID`] if any `ERROR` was logged during the run.
    /// Bevy keeps rendering after a shader or pipeline validation failure, so
    /// without this the process writes a PNG that is missing a render layer and
    /// still reports success — the BL-20 gap that makes "the output file
    /// exists" worthless as evidence. A capture that logged an error is not
    /// partially valid; callers should discard it rather than compare it.
    pub fn run(self) {
        self.build().run();

        let errors = thalos_runtime::capture_health::error_count();
        if errors == 0 {
            return;
        }
        eprintln!("\ncapture INVALID: {errors} error(s) logged during this run.");
        eprintln!("A capture that logged an error is not partially valid — discard it.");
        for message in thalos_runtime::capture_health::error_messages() {
            eprintln!("  - {message}");
        }
        std::process::exit(EXIT_CAPTURE_INVALID);
    }
}
