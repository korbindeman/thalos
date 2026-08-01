//! Wing-body junction fairing profile.
//!
//! A fairing is **derived geometry, not a part**: one right-hand main-wing
//! panel supplies the longitudinal station and root chord, and the host
//! fuselage folds that profile into its own loft. This is intentionally not a
//! second mesh laid over the belly. Sharing the fuselage's rings, normals, and
//! material is what makes the wing carry-through read as part of the airframe
//! rather than a blister intersecting it.

use crate::part::{Fuselage, Wing};

/// Fairing input consumed by the fuselage loft. Copyable so visual rebuilders
/// can derive it directly from the host's surface-mounted wing query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FuselageFairing {
    pub station: f32,
    pub root_chord: f32,
}

impl FuselageFairing {
    pub fn from_wing(wing: &Wing, station: f32) -> Self {
        Self {
            station,
            root_chord: wing.root_chord,
        }
    }
}

/// Cross-section exponent at the wing root. A circle is 2; this modestly
/// squares the lower half into a flatter, fuller wing carry-through while
/// keeping every point inside the fuselage's existing maximum half-width.
const FAIRING_EXPONENT: f32 = 3.2;
/// Fairing reach ahead of and behind the wing mount, in root chords.
const FWD_CHORDS: f32 = 0.65;
const AFT_CHORDS: f32 = 1.05;

#[derive(Clone, Copy)]
struct LoftKey {
    t: f32,
    strength: f32,
}

/// Whether this wing mount earns a junction fairing: the **right-hand panel**
/// of a pair (one body-symmetric fairing), mounted on the side-to-lower hull,
/// and large enough to be a main wing rather than a tailplane.
pub fn wants_wing_fairing(wing: &Wing, angle: f32, fus: &Fuselage) -> bool {
    angle.sin() > 0.5 && angle.cos() < 0.35 && wing.root_chord >= 0.12 * fus.length
}

/// Fairing strength at a longitudinal loft station. The mount is an authored
/// key rather than the midpoint: the lower body grows gently into the wing
/// root, remains full just aft of it, then returns to the ordinary fuselage.
fn loft_profile(t: f32, mount_t: f32) -> f32 {
    let keys = [
        LoftKey {
            t: 0.0,
            strength: 0.0,
        },
        LoftKey {
            t: mount_t * 0.45,
            strength: 0.10,
        },
        LoftKey {
            t: mount_t * 0.82,
            strength: 0.62,
        },
        LoftKey {
            t: mount_t,
            strength: 1.0,
        },
        LoftKey {
            t: (mount_t + 0.14).min(0.76),
            strength: 0.88,
        },
        LoftKey {
            t: 0.84,
            strength: 0.28,
        },
        LoftKey {
            t: 1.0,
            strength: 0.0,
        },
    ];

    for pair in keys.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if t <= b.t {
            let span = (b.t - a.t).max(1.0e-5);
            let u = ((t - a.t) / span).clamp(0.0, 1.0);
            // Zero derivative at every authored station: changing section does
            // not introduce a visible knuckle into the continuous skin.
            let u = u * u * (3.0 - 2.0 * u);
            return a.strength + (b.strength - a.strength) * u;
        }
    }
    0.0
}

/// Deform one point on the fuselage's lower cross-section. `station01` runs
/// nose→tail, `lower_mu` runs waterline→keel, and `(base_x, base_z)` is the
/// already-superelliptic fuselage coordinate before centerline upsweep.
pub(crate) fn deform_lower_section(
    fairing: FuselageFairing,
    fuselage_length: f32,
    local_radius: f32,
    station01: f32,
    lower_mu: f32,
    base_x: f32,
    base_z: f32,
) -> (f32, f32) {
    let len = fuselage_length.max(0.01);
    let y_fwd = (FWD_CHORDS * fairing.root_chord).min(fairing.station * len);
    let y_aft = -(AFT_CHORDS * fairing.root_chord).min((1.0 - fairing.station).max(0.0) * len);
    let y_local = (fairing.station - station01) * len;
    if y_local > y_fwd || y_local < y_aft {
        return (base_x, base_z);
    }

    let t = (y_fwd - y_local) / (y_fwd - y_aft).max(1.0e-5);
    let mount_t = y_fwd / (y_fwd - y_aft).max(1.0e-5);
    let profile = loft_profile(t, mount_t);

    // Blend the lower circle toward a restrained superellipse. The target
    // reaches neither wider than `radius` nor deeper than the original keel;
    // it only fills and flattens the lower shoulders where the wings enter the
    // body. Sharing the original full fuselage rings keeps the transition
    // smooth in both directions.
    let mu = lower_mu.clamp(0.0, 1.0);
    let radius = local_radius.max(1.0e-3);
    let target_x = base_x.signum() * radius * (1.0 - mu * mu).max(0.0).powf(1.0 / FAIRING_EXPONENT);
    let target_z = -radius * mu.powf(2.0 / FAIRING_EXPONENT);
    let section_blend = 4.0 * mu * (1.0 - mu);
    let blend = profile * section_blend;
    let x = base_x + (target_x - base_x) * blend;
    let z = base_z + (target_z - base_z) * blend;
    (x, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fuselage() -> Fuselage {
        Fuselage {
            length: 35.0,
            max_width: 3.3,
            max_height: 3.3,
            roundness: 1.0,
            nose_fraction: 0.13,
            nose_bluntness: 0.55,
            tail_fraction: 0.34,
            nose_droop: 0.0,
            tail_upsweep: 1.05,
            tail_tip_diameter: 0.0,
            tail_bluntness: 0.6,
            dry_mass: 0.0,
        }
    }

    fn main_wing() -> Wing {
        Wing {
            span: 15.0,
            root_chord: 5.2,
            tip_chord: 1.5,
            sweep: 0.52,
            dihedral: 0.365,
            thickness: 0.11,
            incidence: 0.0,
            dry_mass: 0.0,
            control_surfaces: Vec::new(),
        }
    }

    #[test]
    fn main_wing_pair_gets_a_fairing_tailplane_and_fin_do_not() {
        let fus = fuselage();
        assert!(wants_wing_fairing(&main_wing(), 1.85, &fus));
        assert!(!wants_wing_fairing(&main_wing(), -1.85, &fus));
        let stab = Wing {
            root_chord: 2.6,
            span: 4.6,
            ..main_wing()
        };
        assert!(!wants_wing_fairing(&stab, 1.5708, &fus));
        assert!(!wants_wing_fairing(&main_wing(), 0.0, &fus));
        assert!(!wants_wing_fairing(&main_wing(), 0.9, &fus));
    }

    #[test]
    fn deformation_is_exactly_zero_at_waterline_and_longitudinal_ends() {
        let fairing = FuselageFairing::from_wing(&main_wing(), 0.44);
        let at_waterline = deform_lower_section(fairing, 35.0, 1.65, 0.44, 0.0, 1.65, 0.0);
        assert_eq!(at_waterline, (1.65, 0.0));

        let ahead = deform_lower_section(fairing, 35.0, 1.65, 0.01, 1.0, 0.0, -1.65);
        let behind = deform_lower_section(fairing, 35.0, 1.65, 0.99, 1.0, 0.0, -1.65);
        assert_eq!(ahead, (0.0, -1.65));
        assert_eq!(behind, (0.0, -1.65));
    }
}
