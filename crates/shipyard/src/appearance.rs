//! Maps a part's construction dimensions to its render-material uniform.
//!
//! The inputs are construction types (`AttachNodes`, `FuelTank`, …), so this
//! bridge lives in the construction crate; the output [`ShipPartParams`] is owned
//! by the render crate (re-exported via [`crate::material`]). One definition,
//! shared by the in-game editor's live rebuild and the flight view's ship build
//! (`thalos_game::ship_view`) so the procedural panel/rivet layout matches across
//! both — it was previously copy-pasted in each.

use crate::{Adapter, AttachNodes, Decoupler, FuelTank, Fuselage, ShipPartParams};

/// Pick `ShipPartMaterial` uniforms for a part from its attach-node + dimension
/// components. Length / radius drive the procedural panel + rivet layout; each
/// part picks its own dimensions so the pattern reads consistently across
/// tank–decoupler boundaries without sharing an asset handle.
pub fn ship_part_params(
    nodes: &AttachNodes,
    tank: Option<&FuelTank>,
    fuselage: Option<&Fuselage>,
    dec: Option<&Decoupler>,
    adapter: Option<&Adapter>,
    seed: u32,
) -> ShipPartParams {
    let top_r = nodes.get("top").map(|n| n.diameter * 0.5).unwrap_or(0.5);
    // Tanks and decouplers are cylinders; adapters are conical frustums from
    // `top_r` at the mesh's +Y end to `target_diameter / 2` at the -Y end.
    let (radius_top, radius_bottom, length) = if let Some(t) = tank {
        (top_r, top_r, t.length)
    } else if let Some(f) = fuselage {
        // Near-cylindrical barrel: the panel shader treats it like a tank.
        (top_r, top_r, f.length)
    } else if dec.is_some() {
        (top_r, top_r, 0.2)
    } else if let Some(a) = adapter {
        let bot_r = a.target_diameter * 0.5;
        let h = (top_r + bot_r).max(0.4); // same formula as the adapter mesh
        let dr = top_r - bot_r;
        let slant = (h * h + dr * dr).sqrt();
        (top_r, bot_r, slant)
    } else {
        (top_r, top_r, 1.0)
    };
    ShipPartParams {
        length,
        radius_top,
        radius_bottom,
        seed,
        ..Default::default()
    }
}
