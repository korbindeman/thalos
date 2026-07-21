//! Headless application shell for the canonical Thalos runtime.
//!
//! The capture systems still execute inside `thalos_runtime`; this package is
//! the stable host-facing boundary while those tightly coupled systems are
//! progressively projected onto public runtime services.

use bevy::prelude::App;

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

    pub fn run(self) {
        self.build().run();
    }
}
