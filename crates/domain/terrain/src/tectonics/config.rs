//! Authored configuration for a body's tectonic layer.

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Per-body tectonic configuration. Authored in the body's RON detail file
/// alongside `terrain`. Bodies that omit this field have no tectonic layer.
///
/// All randomized output (mesh point placement, plate seed selection, plate
/// flood-fill order, Euler-pole direction and angular speed, oceanic/continental
/// flagging) is derived deterministically from `seed` combined with the body's
/// `root_seed` via [`crate::seeding::sub_seed`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TectonicConfig {
    /// Number of plates to seed. Earth-like geology lands around 8–16 major
    /// plates; Thalos's StagnantLid scenario is happy in that range too.
    pub plate_count: u32,

    /// Number of cells in the spherical Voronoi mesh. ~2k gives ~80 km cell
    /// pitch on a Mars-sized body, which is plenty for plate-boundary
    /// structure. Per-fragment "am I within 5 km of a transform fault"
    /// queries would want a denser mesh, addressed only when needed.
    pub mesh_cells: u32,

    /// How current plate motion factors into the sample API. See
    /// [`TectonicActivity`].
    pub activity: TectonicActivity,

    /// Fraction of plates that are continental (vs oceanic). Continental
    /// plates produce land surface; oceanic plates produce sea floor. Earth
    /// is ~30% continental by area; per-plate this depends on plate size,
    /// but using the count fraction is a fine approximation.
    pub continental_fraction: f32,

    /// Per-system seed; combined with the body's `root_seed` to derive every
    /// other random quantity.
    pub seed: u64,

    /// Optional authored override for plate seed positions. When present,
    /// the first `min(seed_dirs.len(), plate_count)` plate centers are
    /// snapped to the mesh cell nearest each direction; remaining plates
    /// fall back to random selection. Reserved for future hand-placement
    /// in the editor; default deserialization is `None`.
    #[serde(default)]
    pub seed_dirs: Option<Vec<Vec3>>,

    /// Continental seed clustering strength.
    ///
    /// `0.0` (default): pure Mitchell repulsion — continental seeds spread
    /// evenly across the sphere (current behavior; produces ~tetrahedral
    /// arrangement at four continentals). `1.0`: secondary continental seeds
    /// drawn from a tight spherical cap (~30°) around the primary, with one
    /// outlier placed by Mitchell. Intermediate values lerp the cap radius.
    /// Use this to produce a "main supercontinent + outliers" pattern.
    #[serde(default)]
    pub continental_clustering: f32,

    /// Equatorial-bias strength on the primary continental seed.
    ///
    /// `0.0` (default): no bias (current behavior). Larger values add a
    /// `|dir.y| * equatorial_bias` penalty to Mitchell's score so picks
    /// near the equator beat picks near the poles. `0.5–1.0` is a moderate
    /// pull; values above ~1.5 effectively pin the primary to the equator.
    #[serde(default)]
    pub equatorial_bias: f32,

    /// Growth-rate multiplier for the primary continental plate during
    /// round-robin BFS.
    ///
    /// `1.0` (default): all plates grow equally (current behavior).
    /// `2.0`: primary gets one extra expansion per round. Clamped to
    /// `[1.0, 4.0]` internally so the primary cannot eat the sphere.
    #[serde(default = "default_primary_size_multiplier")]
    pub primary_size_multiplier: f32,
}

fn default_primary_size_multiplier() -> f32 {
    1.0
}

/// Activity mode for a tectonic system.
///
/// **Important:** Boundary classification (`Convergent`/`Divergent`/`Transform`)
/// and boundary distance fields are *always* derived from the encoded Euler
/// poles, regardless of activity. That is what makes "StagnantLid" mean
/// "frozen scars from historical motion" — the structural signature of past
/// plates is the same data structure as a live one. Only
/// [`crate::tectonics::TectonicSample::plate_velocity_m_per_yr`] is gated by
/// activity: for `StagnantLid` and `Frozen`, sampled velocity reads zero so
/// downstream consumers (e.g. the editor's motion-arrow overlay) see no live
/// motion, while mountain-belt placement at convergent boundaries still works.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub enum TectonicActivity {
    /// Plates moving now. `plate_velocity_m_per_yr` reflects live Euler-pole
    /// rotation; boundary kinds and magnitudes describe present motion.
    Active,
    /// Stagnant lid — historical motion encoded in plate Euler poles, but
    /// no current surface motion. Boundary kinds/distances retain their
    /// "frozen scar" meaning; sampled velocity reads zero.
    StagnantLid,
    /// Frozen at a specific epoch. Equivalent to `StagnantLid` for sampling;
    /// the age field is reserved for future crustal-age computations.
    Frozen { age_my: f32 },
}

impl TectonicActivity {
    /// Whether sampled plate velocity should reflect Euler-pole rotation
    /// (i.e. plates are moving now). False for `StagnantLid` and `Frozen`.
    pub fn live_velocity(self) -> bool {
        matches!(self, Self::Active)
    }
}
