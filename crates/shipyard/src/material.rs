//! Craft hull material — re-exported from the render crate.
//!
//! The material *definition* moved to [`thalos_body_render::craft`] (Phase 4a of
//! the shipyard rendering decoupling — see `docs/architecture.md`): the render
//! layer owns *how a craft surface looks*, while this crate owns *what the craft
//! is*. This shim re-exports the type so the editor core (`crate::editor`) keeps
//! compiling unchanged while the material-*application* split is deferred to the
//! follow-up.
//!
//! **Interim debt:** re-exporting from the render crate is the only reason
//! `thalos_shipyard` depends on `thalos_body_render` (a backwards edge — the
//! construction crate shouldn't pull in the render stack). The follow-up moves
//! material application out of the editor core, then flips this to the clean
//! `body_render → shipyard` direction and drops this dependency.

pub use thalos_body_render::craft::{
    ShipPartExtension, ShipPartMaterial, ShipPartParams, landing_gear_base, stainless_steel_base,
};
