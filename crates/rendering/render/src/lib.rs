//! Concrete celestial-body render adapters over shared appearance mechanisms.
//! `planetary` composes standard-path cube-sphere tiles with analytic
//! atmosphere/ocean and clouds; `far_body` composes distant impostors and rings.
//! The old UDLOD ground renderer is available only behind `legacy-udlod`.
pub mod clouds;
pub mod composite_order;
pub mod craft;
pub mod far_body;
pub mod ground;
pub mod impostor;
pub mod planetary;
pub mod rt;
pub mod tiles;

pub use clouds::*;
pub use craft::*;
pub use far_body::FarBodyRenderPlugin;
pub use ground::*;
pub use impostor::*;
pub use planetary::PlanetaryRenderPlugin;
/// The shared body-surface shading types + WGSL libraries, extracted to the
/// `thalos_body_shading` leaf crate (ADR-20260724T022732Z) so shading edits
/// recompile a small crate. Re-exported as the `shading` module and flat at the
/// crate root so existing `thalos_body_render::{shading::*, SceneLighting, …}`
/// and internal `crate::shading::…` paths keep resolving unchanged.
pub use thalos_body_shading::{self as shading, *};

/// The vendored UDLOD terrain renderer, re-exported so the rest of the
/// workspace depends on it *through* `body_render` — its single consumer —
/// rather than importing `thalos_udlod` directly. Reach tile/atlas/precision
/// handles via `thalos_body_render::udlod::{prelude, math, big_space}`.
#[cfg(feature = "legacy-udlod")]
pub use thalos_udlod as udlod;
