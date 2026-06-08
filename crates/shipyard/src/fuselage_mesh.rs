//! Stationed-loft fuselage geometry, shared by the editor and the in-game
//! ship view so a saved airframe looks identical in both.
//!
//! A [`crate::Fuselage`] is a **stationed loft** (`docs/construction.md`
//! §4.2): a straight body axis with a sequence of **superellipse**
//! cross-section stations, the skin lofted between them. High-level airliner
//! params (length, nose/tail fractions, droop, upsweep, tail-tip diameter)
//! *generate* the station set — the player authors a handful of numbers, not
//! a raw station list. This replaces the old "straight cylinder + straight
//! cone" pair with one continuous, **upswept** body: the realism lever is the
//! per-station vertical offset (`v_offset`) raising the tail centerline (and
//! optionally drooping the nose) while the tailcone necks on a curved ogive
//! rather than a straight pencil-point.
//!
//! ## Frame
//!
//! Authored **centred on the origin** along Y, spanning `[+length/2,
//! −length/2]`: the barrel top is at `+length/2`, the tail tip at
//! `−length/2`. This matches the cylinder/frustum/cockpit meshes — the
//! caller offsets the body child by `−height/2` so the top lands at the part
//! origin (the `top` attach node) and the tail at `−length` (the `bottom`
//! node), exactly like the tank it replaces. Cross-sections lie in X (right)
//! / Z (dorsal, "up"); `v_offset` shifts a station along +Z.
//!
//! ## Cross-section (superellipse)
//!
//! A superellipse with half-width `a`, half-height `b`, traced parametrically:
//!
//! ```text
//! x = a · sign(cos θ) · |cos θ|^(2/n)
//! z = b · sign(sin θ) · |sin θ|^(2/n)
//! ```
//!
//! `n = 2` is a true ellipse (round fuselage); larger `n` squares the
//! corners toward a rounded-rectangle belly. `roundness ∈ [0, 1]` maps to
//! `n`: `1` → ellipse, `0` → boxy. The A220 is essentially circular
//! (`roundness ≈ 1`, `a ≈ b`).
//!
//! ## Host skin query
//!
//! [`skin_radius`] / [`v_offset_at`] are the **surface-mount seam**: wings,
//! gear, and nacelles mount at a `(station, angle)` and need the skin radius
//! and centerline offset *at that station*, not a single constant radius like
//! a cylinder. For a circular section [`skin_radius`] reduces to the barrel
//! radius, so a wing on a plain tank is unaffected.

use crate::part::Fuselage;
use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

/// Radial segments around each cross-section.
const RADIAL_SEGMENTS: usize = 48;
/// Axial stations along the body. Generous so the necked, upswept tailcone
/// reads as a smooth curve and not a faceted cone.
const AXIAL_STATIONS: usize = 64;

/// One generated cross-section: position along the body and its superellipse
/// half-extents, with the centerline vertical offset already applied to the
/// caller via [`v_offset_at`].
#[derive(Clone, Copy, Debug)]
struct Station {
    /// Axial position, metres (+length/2 at the top, −length/2 at the tail).
    y: f32,
    /// Half-width (X) and half-height (Z), metres.
    a: f32,
    b: f32,
    /// Centerline vertical offset (Z), metres — droop (−) / upsweep (+).
    v: f32,
}

/// Superellipse exponent from the authored `roundness ∈ [0, 1]`: `1` → a
/// true ellipse (`n = 2`), `0` → a boxy rounded-rectangle (`n = 6`).
fn superellipse_exponent(roundness: f32) -> f32 {
    2.0 + (1.0 - roundness.clamp(0.0, 1.0)) * 4.0
}

/// Smooth Hermite ease on `[0, 1]`, used for both the curved tailcone neck
/// and the centerline droop/upsweep so neither is a straight-line ramp.
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Uniform scale a child fuselage inherits when its `top` node diameter is
/// driven by the parent (mirrors a tank inheriting its parent's diameter).
/// All cross-section extents and the tail-tip scale by this so the authored
/// proportions are preserved at the inherited size.
fn diameter_scale(fus: &Fuselage, effective_diameter: f32) -> f32 {
    if fus.max_width > 0.0 {
        effective_diameter / fus.max_width
    } else {
        1.0
    }
}

/// Generate the cross-section stations for a fuselage rendered at
/// `effective_diameter` (the barrel diameter, possibly inherited from the
/// parent). `f ∈ [0, 1]` runs top → tail.
fn stations(fus: &Fuselage, effective_diameter: f32) -> Vec<Station> {
    let s = diameter_scale(fus, effective_diameter);
    let half_w = 0.5 * effective_diameter;
    let aspect = if fus.max_width > 0.0 {
        fus.max_height / fus.max_width
    } else {
        1.0
    };
    let half_h = half_w * aspect;
    let tail_tip_r = 0.5 * fus.tail_tip_diameter * s;

    let nose_end = fus.nose_fraction.clamp(0.0, 0.49);
    let tail_start = (1.0 - fus.tail_fraction.clamp(0.0, 0.95)).max(nose_end);

    (0..AXIAL_STATIONS)
        .map(|i| {
            let f = i as f32 / (AXIAL_STATIONS - 1) as f32;
            let y = fus.length * 0.5 - f * fus.length;

            // Radial profile: nose growth → constant barrel → curved tail neck.
            let (radius_frac, v) = if f < nose_end && nose_end > 0.0 {
                // Nose: grow from the tip (t=0) to the full barrel (t=1). The
                // shape blends a straight cone (linear) with a rounded radome
                // (convex ellipsoid `√(t·(2−t))`) by `nose_bluntness`. Droop
                // lowers the tip and eases out toward the barrel.
                let t = f / nose_end;
                let conic = t;
                let radome = (t * (2.0 - t)).max(0.0).sqrt();
                let grow = conic + (radome - conic) * fus.nose_bluntness.clamp(0.0, 1.0);
                (grow.max(0.02), -fus.nose_droop * s * (1.0 - smoothstep(t)))
            } else if f > tail_start && tail_start < 1.0 {
                // Tail: neck from full barrel to the tail tip on a smooth
                // curve, centerline sweeping up.
                let t = (f - tail_start) / (1.0 - tail_start);
                let e = smoothstep(t);
                let barrel = 1.0;
                let tip_frac = tail_tip_r / half_w.max(1e-3);
                (barrel * (1.0 - e) + tip_frac * e, fus.tail_upsweep * s * e)
            } else {
                (1.0, 0.0)
            };

            Station {
                y,
                a: half_w * radius_frac,
                b: half_h * radius_frac,
                v,
            }
        })
        .collect()
}

/// Superellipse point at parameter `θ` for half-extents `(a, b)` and exponent
/// `n`, in the cross-section plane (X, Z) before the vertical offset.
fn section_point(a: f32, b: f32, n: f32, theta: f32) -> Vec2 {
    let (s, c) = theta.sin_cos();
    let powx = c.abs().powf(2.0 / n);
    let powz = s.abs().powf(2.0 / n);
    Vec2::new(a * c.signum() * powx, b * s.signum() * powz)
}

/// Build the host-local fuselage skin mesh at the given barrel diameter.
/// Smooth-shaded along the whole loft; the tail-tip cap carries its own
/// vertices so its rim stays crisp. The top (nose-end) is left open — the
/// cockpit end-cap covers it, exactly as the old cylinder + nose pair did.
pub fn build_fuselage_mesh(fus: &Fuselage, effective_diameter: f32) -> Mesh {
    let stations = stations(fus, effective_diameter);
    let n_exp = superellipse_exponent(fus.roundness);
    let rings = stations.len();

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Loft rings (shared vertices → smooth skin).
    for st in &stations {
        for j in 0..RADIAL_SEGMENTS {
            let theta = std::f32::consts::TAU * (j as f32 / RADIAL_SEGMENTS as f32);
            let p = section_point(st.a, st.b, n_exp, theta);
            positions.push([p.x, st.y, p.y + st.v]);
        }
    }
    for i in 0..(rings - 1) {
        for j in 0..RADIAL_SEGMENTS {
            let j2 = (j + 1) % RADIAL_SEGMENTS;
            let a = (i * RADIAL_SEGMENTS + j) as u32;
            let b = (i * RADIAL_SEGMENTS + j2) as u32;
            let c = ((i + 1) * RADIAL_SEGMENTS + j2) as u32;
            let d = ((i + 1) * RADIAL_SEGMENTS + j) as u32;
            // Outward winding (lower ring first → faces away from the axis).
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }

    // Tail-tip cap (own vertices for a crisp rim).
    if let Some(tail) = stations.last() {
        let center = positions.len() as u32;
        positions.push([0.0, tail.y, tail.v]);
        let rim_base = positions.len() as u32;
        for j in 0..RADIAL_SEGMENTS {
            let theta = std::f32::consts::TAU * (j as f32 / RADIAL_SEGMENTS as f32);
            let p = section_point(tail.a, tail.b, n_exp, theta);
            positions.push([p.x, tail.y, p.y + tail.v]);
        }
        for j in 0..RADIAL_SEGMENTS {
            let j2 = ((j + 1) % RADIAL_SEGMENTS) as u32;
            // Faces −Y (aft, outward at the tail). The rings wind CCW about +Y,
            // so following that order (`j → j2`) gives a −Y-facing normal — the
            // outward direction at the aft end. (Reversed winding faces +Y, into
            // the body, rendering the cap dark/inside-out.)
            indices.extend_from_slice(&[center, rim_base + j as u32, rim_base + j2]);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    let uv = vec![[0.0_f32, 0.0]; positions.len()];
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
    mesh.insert_indices(Indices::U32(indices));
    mesh.compute_smooth_normals();
    mesh
}

/// Skin radius along the mount radial at `(station01, angle)` for a fuselage
/// rendered at `effective_diameter`. `angle` follows the surface-mount
/// convention (`0` → +Z dorsal, `π/2` → +X right, `π` → −Z belly). For a
/// circular section this is just the local barrel radius, so a wing/gear/
/// nacelle on a plain tank (which always reports its constant radius) is
/// unaffected. Elliptical sections use the ellipse radius in the ray
/// direction; the superellipse exponent is ignored here (it only squares the
/// *mesh* corners, a sub-decimetre effect on the near-circular mount radial).
pub fn skin_radius(fus: &Fuselage, effective_diameter: f32, station01: f32, angle: f32) -> f32 {
    let (a, b) = section_extents(fus, effective_diameter, station01);
    // r_hat = (sin θ in X, cos θ in Z); ellipse radius in that direction.
    let (s, c) = angle.sin_cos();
    let inv = ((s / a.max(1e-3)).powi(2) + (c / b.max(1e-3)).powi(2)).sqrt();
    if inv > 0.0 { 1.0 / inv } else { a }
}

/// Centerline vertical offset (Z) at `station01` for a fuselage rendered at
/// `effective_diameter` — the upsweep/droop a surface mount must follow so it
/// sits on the (raised) skin rather than the straight axis.
pub fn v_offset_at(fus: &Fuselage, effective_diameter: f32, station01: f32) -> f32 {
    interp_station(fus, effective_diameter, station01).v
}

/// The surface-mount seam, unified across host kinds: `(radius, v_offset)` a
/// wing / gear / nacelle should mount at on a host whose `top` node diameter
/// is `host_top_diameter`. A loft host answers from its skin at the mount
/// `(station, angle)`; any other host (a plain tank/cylinder) reports its
/// constant radius and zero offset, so non-loft mounts are unchanged.
pub fn host_mount_geometry(
    fus: Option<&Fuselage>,
    host_top_diameter: f32,
    station01: f32,
    angle: f32,
) -> (f32, f32) {
    match fus {
        Some(f) => (
            skin_radius(f, host_top_diameter, station01, angle),
            v_offset_at(f, host_top_diameter, station01),
        ),
        None => (host_top_diameter * 0.5, 0.0),
    }
}

/// Interpolated `(half_width, half_height)` at a continuous station.
fn section_extents(fus: &Fuselage, effective_diameter: f32, station01: f32) -> (f32, f32) {
    let st = interp_station(fus, effective_diameter, station01);
    (st.a, st.b)
}

fn interp_station(fus: &Fuselage, effective_diameter: f32, station01: f32) -> Station {
    let stations = stations(fus, effective_diameter);
    let f = station01.clamp(0.0, 1.0);
    let x = f * (stations.len() - 1) as f32;
    let i = x.floor() as usize;
    if i + 1 >= stations.len() {
        return stations[stations.len() - 1];
    }
    let t = x - i as f32;
    let lo = stations[i];
    let hi = stations[i + 1];
    Station {
        y: lo.y + (hi.y - lo.y) * t,
        a: lo.a + (hi.a - lo.a) * t,
        b: lo.b + (hi.b - lo.b) * t,
        v: lo.v + (hi.v - lo.v) * t,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a220_fuselage() -> Fuselage {
        Fuselage {
            length: 26.0,
            max_width: 3.5,
            max_height: 3.5,
            roundness: 1.0,
            nose_fraction: 0.0,
            nose_bluntness: 0.85,
            tail_fraction: 0.4,
            nose_droop: 0.0,
            tail_upsweep: 1.2,
            tail_tip_diameter: 0.6,
            dry_mass: 0.0,
        }
    }

    fn extents(mesh: &Mesh) -> (Vec3, Vec3) {
        let pos = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for p in pos {
            let p = Vec3::from_array(*p);
            min = min.min(p);
            max = max.max(p);
        }
        (min, max)
    }

    #[test]
    fn spans_length_and_barrel_width() {
        let f = a220_fuselage();
        let m = build_fuselage_mesh(&f, f.max_width);
        let (min, max) = extents(&m);
        assert!((min.y + f.length * 0.5).abs() < 1e-3, "tail at -length/2");
        assert!((max.y - f.length * 0.5).abs() < 1e-3, "top at +length/2");
        // Widest point reaches the barrel half-width.
        assert!((max.x - f.max_width * 0.5).abs() < 0.05, "barrel half-width in X");
        assert!(m.attribute(Mesh::ATTRIBUTE_NORMAL).is_some());
    }

    #[test]
    fn tail_necks_and_sweeps_up() {
        let f = a220_fuselage();
        // Barrel is full radius; the tail necks well below it.
        let r_barrel = skin_radius(&f, f.max_width, 0.3, std::f32::consts::FRAC_PI_2);
        let r_tail = skin_radius(&f, f.max_width, 0.98, std::f32::consts::FRAC_PI_2);
        assert!((r_barrel - f.max_width * 0.5).abs() < 1e-3, "barrel = half width");
        assert!(r_tail < 0.5 * r_barrel, "tail necks down ({r_tail} vs {r_barrel})");
        // Centerline sweeps up toward the tail; the barrel stays level.
        assert!(v_offset_at(&f, f.max_width, 0.3).abs() < 1e-3, "barrel level");
        assert!(v_offset_at(&f, f.max_width, 1.0) > 0.5 * f.tail_upsweep, "tail upsweep");
    }

    #[test]
    fn parametric_nose_tapers_and_is_blunter_when_rounder() {
        // A fuselage with a real nose: the front necks from the barrel down to
        // a tip, and at a fixed nose station a rounder (blunter) nose is wider
        // than a conic one.
        let base = Fuselage {
            nose_fraction: 0.25,
            tail_fraction: 0.3,
            ..a220_fuselage()
        };
        // Barrel full radius, nose well below it near the tip.
        let r_barrel = skin_radius(&base, base.max_width, 0.5, 0.0);
        let r_nose = skin_radius(&base, base.max_width, 0.05, 0.0);
        assert!(r_nose < 0.6 * r_barrel, "nose necks toward the tip");

        let conic = Fuselage { nose_bluntness: 0.0, ..base.clone() };
        let radome = Fuselage { nose_bluntness: 1.0, ..base.clone() };
        // Halfway along the nose the rounded radome bulges past the cone.
        let t = base.nose_fraction * 0.5;
        assert!(
            skin_radius(&radome, base.max_width, t, 0.0)
                > skin_radius(&conic, base.max_width, t, 0.0) + 0.05,
            "rounded nose is fuller than the cone at the same station"
        );
    }

    #[test]
    fn circular_skin_radius_is_angle_independent() {
        let f = a220_fuselage();
        let r_up = skin_radius(&f, f.max_width, 0.3, 0.0);
        let r_side = skin_radius(&f, f.max_width, 0.3, std::f32::consts::FRAC_PI_2);
        assert!((r_up - r_side).abs() < 1e-3, "round section: radius constant in angle");
    }

    #[test]
    fn tail_cap_faces_aft_not_into_the_body() {
        // The tail-tip cap must face −Y (aft / outward). Its centre vertex is
        // used only by the cap fan, so its smoothed normal is the cap normal.
        // A +Y normal would render the cap dark / inside-out.
        let f = a220_fuselage();
        let m = build_fuselage_mesh(&f, f.max_width);
        let pos = m.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().as_float3().unwrap();
        let nor = m.attribute(Mesh::ATTRIBUTE_NORMAL).unwrap().as_float3().unwrap();
        // The cap centre is the unique vertex at (x≈0, y=tail, z≈v_offset) — the
        // rim / last loft ring share its y but sit a half-height off in z.
        let tail_y = -f.length * 0.5;
        let vt = v_offset_at(&f, f.max_width, 1.0);
        let mut found = None;
        for (p, n) in pos.iter().zip(nor) {
            if (p[1] - tail_y).abs() < 1e-3 && p[0].abs() < 1e-3 && (p[2] - vt).abs() < 1e-3 {
                found = Some(*n);
            }
        }
        let n = found.expect("tail cap centre vertex");
        assert!(n[1] < -0.8, "tail cap faces aft (−Y), got normal {n:?}");
    }

    #[test]
    fn inherited_diameter_scales_proportionally() {
        let f = a220_fuselage();
        let m = build_fuselage_mesh(&f, 7.0); // double the barrel diameter
        let (_, max) = extents(&m);
        assert!((max.x - 3.5).abs() < 0.1, "barrel half-width tracks inherited diameter");
        // Tail tip scales with it too.
        let r_tail = skin_radius(&f, 7.0, 1.0, 0.0);
        assert!((r_tail - 0.6).abs() < 0.05, "tail tip radius scales (0.6 = 0.3·2)");
    }
}
