//! Draw order for the body-centred fullscreen composites.
//!
//! The celestial backdrop, the analytic atmosphere, the analytic ocean and the
//! cloud composite are not world geometry: they are whole-frame passes that
//! happen to be expressed as meshes so they ride Bevy's material pipeline. They
//! land in `Transparent3d` alongside ordinary see-through geometry (engine
//! plumes, trails, impostors), and that phase is sorted **back to front by the
//! view-space depth of each mesh's centre**.
//!
//! That sort is the wrong tool for a fullscreen pass, and the failure is not
//! subtle. The three body composites are parented to the body, so their sort
//! point is the *planet centre*. It is tempting to read that as "always
//! enormously far away, therefore always drawn first" — and that is exactly
//! what the previous ordering assumed. But the sort key is not the distance to
//! the planet centre, it is how far **along the view axis** the planet centre
//! lies. Stand on the surface and look at the horizon and the centre is
//! straight *down*: perpendicular to the view, so the key collapses to roughly
//! zero and the atmosphere sorts as the nearest object in the scene. It is then
//! painted last, over every transparent that was drawn before it — and on sky
//! pixels the atmosphere is opaque, so those transparents are erased outright.
//! Pitch the camera down and the key swings to −R, the order flips back, and
//! they reappear. See `docs/incidents/20260725T*-plume-erased-by-the-sky.md`.
//!
//! So the order is pinned instead of derived. Each pass gets a `depth_bias`
//! (added straight to the sort key by Bevy) large enough in magnitude that the
//! geometric term can never reorder the stack, while the offsets *between*
//! slots stay exactly representable in `f32` at that magnitude:
//!
//! ```text
//!   celestial backdrop  ── stars + galaxies, dimmed by the air in front
//!   atmosphere          ── BodySky in-scatter + transmittance
//!   ocean               ── analytic sea surface
//!   clouds              ── cloud composite
//!   ─────────────────── everything below sorts by real distance ───────────
//!   world transparents  ── engine plumes, trails, …
//! ```
//!
//! **Rule: a fullscreen composite must claim a slot here — never bias 0.** A
//! pass left on the geometric sort is a pass whose position in the stack
//! depends on where the camera is pointed.

/// Spacing between slots. A power of two well above the `f32` ulp at
/// [`BASE`]'s magnitude (128), so every slot is exact and distinct after the
/// geometric term (at most a body radius, ~7e6 m) is added.
const SLOT: f32 = 65_536.0;

/// Far end of the pinned band. Two orders of magnitude beyond any body-centre
/// depth the geometric term can produce, so no camera orientation can lift a
/// composite out of the band or push a world transparent into it.
const BASE: f32 = -2.0e9;

/// Stars and galaxies: behind the air, so the atmosphere's `(1 − alpha)`
/// transmittance dims them. This is what makes the per-pixel star crush in
/// `body_sky.wgsl` work at dusk (the global daylight suppression in
/// `sky_render::CelestialBackdropVisibility` is the coarse companion term).
pub const CELESTIAL_BACKDROP: f32 = BASE;

/// The custom atmosphere (`BodySkyMaterial`).
pub const ATMOSPHERE: f32 = BASE + SLOT;

/// The analytic ocean (`BodyOceanMaterial`) — over the atmosphere's ground-level
/// haze, under the clouds.
pub const OCEAN: f32 = BASE + 2.0 * SLOT;

/// The cloud composite (`CloudCompositeMaterial`), the last of the fullscreen
/// passes: clouds occlude everything the other three drew.
pub const CLOUDS: f32 = BASE + 3.0 * SLOT;

#[cfg(test)]
mod tests {
    use super::*;

    /// The slots must stay distinct *after* the geometric term is added, or the
    /// pinned order silently degenerates back into the coin-flip it replaced.
    #[test]
    fn slots_survive_the_geometric_term() {
        // Worst case: a body-centre depth of a large planet radius, either sign.
        for geometric in [-7.0e6_f32, 0.0, 7.0e6] {
            let keys = [
                CELESTIAL_BACKDROP + geometric,
                ATMOSPHERE + geometric,
                OCEAN + geometric,
                CLOUDS + geometric,
            ];
            for pair in keys.windows(2) {
                assert!(
                    pair[0] < pair[1],
                    "slot order collapsed at geometric term {geometric}: {pair:?}"
                );
            }
        }
    }

    /// Every pinned slot must sort ahead of any world transparent, whose key is
    /// its own view depth with no bias. Nothing in a rendered scene is a
    /// hundred million metres deep.
    #[test]
    fn pinned_band_is_below_world_transparents() {
        let nearest_slot = CLOUDS + 7.0e6;
        assert!(nearest_slot < -1.0e8);
    }
}
