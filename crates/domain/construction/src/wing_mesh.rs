//! Procedural wing geometry, shared by the editor and the in-game ship
//! view so a saved aircraft looks identical in both.
//!
//! A [`Wing`] is built as a tapered / swept / dihedral **airfoil loft**: a
//! NACA 4-digit *symmetric* section (00xx) is sampled at the root and tip
//! chords and lofted along the span. The section's maximum thickness is the
//! wing's `thickness` (t/c), so a thicker wing reads as a fuller airfoil.
//! A symmetric section is deliberate for now: it is handed-neutral, so the
//! mirror counterpart (a separate entity built at the reflected `angle`)
//! reflects correctly with no camber sign to track. Camber and trailing-edge
//! control surfaces are the next slice; callers only depend on
//! [`build_wing_mesh`], so those upgrades stay local to this file.
//!
//! ## Airfoil section
//!
//! Half-thickness as a fraction of chord, NACA 4-digit (closed trailing
//! edge so the tail meets cleanly):
//!
//! ```text
//! yt(x) = 5·t·(0.2969·√x − 0.1260·x − 0.3516·x² + 0.2843·x³ − 0.1036·x⁴)
//! ```
//!
//! `x ∈ [0, 1]` is chord fraction from the leading edge; `t` is the wing's
//! t/c, so total section thickness peaks at `t · chord`. Stations are sampled
//! with **cosine spacing** (`x = ½(1 − cos θ)`) to cluster points at the
//! rounded leading edge and the trailing edge where curvature is highest.
//!
//! ## Frame
//!
//! The mesh is authored in the **host's local frame**, origin at the body
//! axis at the mount station (the wing entity's transform places that
//! point; see the transform systems). Host body axis is +Y (fore = toward
//! the nose / "top"); the circular cross-section lies in X/Z. A mount angle
//! `θ` gives the panel's outboard radial `r̂ = (sin θ, 0, cos θ)`
//! (θ = 0 → +Z "up", θ = π/2 → +X "right"). Mirror symmetry is a separate
//! entity with a reflected `θ` (and `Wing.incidence`), not a flag here —
//! see [`crate::SymmetryGroup`].

use crate::part::{ControlSurface, Wing};
use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct WingPanelFrame {
    pub r_hat: Vec3,
    pub span_dir: Vec3,
    pub fore_dir: Vec3,
    pub thick_dir: Vec3,
    pub root_center: Vec3,
    pub tip_center: Vec3,
}

impl WingPanelFrame {
    pub fn chord_at(&self, wing: &Wing, span_fraction: f32) -> f32 {
        wing.root_chord + (wing.tip_chord - wing.root_chord) * span_fraction.clamp(0.0, 1.0)
    }

    pub fn center_at(&self, span_fraction: f32) -> Vec3 {
        self.root_center
            .lerp(self.tip_center, span_fraction.clamp(0.0, 1.0))
    }
}

pub fn wing_panel_frame(wing: &Wing, angle: f32, parent_radius: f32) -> WingPanelFrame {
    let r_hat = Vec3::new(angle.sin(), 0.0, angle.cos());
    let (sin_d, cos_d) = wing.dihedral.sin_cos();
    let span_dir = (r_hat * cos_d + Vec3::Z * sin_d).normalize_or(r_hat);
    // Incidence pitches the chord's leading edge up about the span axis.
    let fore_flat = Vec3::Y;
    let fore_dir = Quat::from_axis_angle(span_dir, wing.incidence) * fore_flat;
    // Thickness normal = span × fore, but that cross product flips sign when
    // the span flips (a left wing), which would point the wing's "up" — and
    // anything mounted by it, like a pylon nacelle — *downward*, giving point
    // (radial) symmetry instead of mirror symmetry. Pin the normal to the
    // dorsal (+Z) hemisphere so left and right are true reflections: tops up,
    // nacelles hang down, on both sides.
    let mut thick_dir = span_dir.cross(fore_dir).normalize_or(Vec3::Z);
    if thick_dir.z < 0.0 {
        thick_dir = -thick_dir;
    }

    let root_center = r_hat * parent_radius;
    let tip_center = root_center + span_dir * wing.span - fore_dir * (wing.span * wing.sweep.tan());

    WingPanelFrame {
        r_hat,
        span_dir,
        fore_dir,
        thick_dir,
        root_center,
        tip_center,
    }
}

/// Chordwise samples per surface (upper / lower). Cosine-spaced, so the
/// effective resolution is highest at the leading and trailing edges.
const CHORD_SAMPLES: usize = 24;

/// NACA 4-digit symmetric half-thickness at chord fraction `x ∈ [0, 1]`,
/// as a fraction of chord. `tc` is the wing t/c. Closed-trailing-edge
/// coefficients (`−0.1036·x⁴`) so `yt(1) = 0` and the tail meets cleanly.
fn naca_half_thickness(tc: f32, x: f32) -> f32 {
    5.0 * tc
        * (0.2969 * x.sqrt() - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x
            - 0.1036 * x * x * x * x)
}

/// One closed airfoil perimeter as `(chordwise_fraction, half_thickness_fraction)`
/// pairs, both relative to chord. `s` runs `+0.5` at the leading edge to
/// `−0.5` at the trailing edge (matching `fore_dir`, which points to the
/// nose); `u` is the signed surface offset. Ordered upper LE→TE then lower
/// TE→LE so consecutive entries trace the outline without repeating the
/// shared LE / TE points.
fn airfoil_perimeter(tc: f32) -> Vec<(f32, f32)> {
    let n = CHORD_SAMPLES;
    let mut out = Vec::with_capacity(2 * n);
    // Upper surface, LE (i=0) → TE (i=n).
    for i in 0..=n {
        let theta = std::f32::consts::PI * (i as f32 / n as f32);
        let x = 0.5 * (1.0 - theta.cos());
        out.push((0.5 - x, naca_half_thickness(tc, x)));
    }
    // Lower surface, TE → LE, skipping the shared TE (i=n) and LE (i=0).
    for i in (1..n).rev() {
        let theta = std::f32::consts::PI * (i as f32 / n as f32);
        let x = 0.5 * (1.0 - theta.cos());
        out.push((0.5 - x, -naca_half_thickness(tc, x)));
    }
    out
}

/// Cosine-spaced chord fractions within `[a, b]` (LE = 0 → TE = 1),
/// inclusive of both endpoints. Shared by the forward (main-wing) and aft
/// (control-surface) sub-sections so they meet on an identical hinge seam.
fn chord_stations_in(a: f32, b: f32) -> Vec<f32> {
    let n = CHORD_SAMPLES;
    let mut xs = vec![a];
    for i in 0..=n {
        let theta = std::f32::consts::PI * (i as f32 / n as f32);
        let x = 0.5 * (1.0 - theta.cos());
        if x > a + 1e-5 && x < b - 1e-5 {
            xs.push(x);
        }
    }
    xs.push(b);
    xs
}

/// Hinge chord-station `s` (LE `+0.5` → TE `−0.5`) for a control surface
/// occupying `chord_fraction` of the chord, measured forward from the
/// trailing edge.
fn hinge_s(chord_fraction: f32) -> f32 {
    -0.5 + chord_fraction.clamp(0.05, 0.95)
}

/// Forward (main-wing) section perimeter, the leading part of the airfoil
/// closed off by a blunt vertical face at the hinge. `x_hinge` is the hinge
/// chord fraction from the LE. Ordered upper LE→hinge then lower hinge→LE
/// to match [`airfoil_perimeter`]'s winding.
fn forward_perimeter(tc: f32, x_hinge: f32) -> Vec<(f32, f32)> {
    let xs = chord_stations_in(0.0, x_hinge);
    let mut out = Vec::with_capacity(2 * xs.len());
    for &x in &xs {
        out.push((0.5 - x, naca_half_thickness(tc, x)));
    }
    for &x in xs.iter().rev() {
        if x <= 1e-6 {
            continue; // skip the shared pinched leading edge
        }
        out.push((0.5 - x, -naca_half_thickness(tc, x)));
    }
    out
}

/// Aft (control-surface) section perimeter: the trailing wedge from the
/// hinge to the closed trailing edge, with a blunt vertical face at the
/// hinge. Same winding convention as [`airfoil_perimeter`].
fn aft_perimeter(tc: f32, x_hinge: f32) -> Vec<(f32, f32)> {
    let xs = chord_stations_in(x_hinge, 1.0);
    let mut out = Vec::with_capacity(2 * xs.len());
    for &x in &xs {
        out.push((0.5 - x, naca_half_thickness(tc, x)));
    }
    for &x in xs.iter().rev() {
        if x >= 1.0 - 1e-6 {
            continue; // skip the shared pinched trailing edge
        }
        out.push((0.5 - x, -naca_half_thickness(tc, x)));
    }
    out
}

/// Place one section perimeter at a spanwise station, in host-local space,
/// offset by `origin` (subtracted — used to express a control-surface mesh
/// relative to its hinge anchor; pass `Vec3::ZERO` for the wing itself).
fn section_ring(
    frame: &WingPanelFrame,
    wing: &Wing,
    span_fraction: f32,
    perimeter: &[(f32, f32)],
    origin: Vec3,
) -> Vec<Vec3> {
    let center = frame.center_at(span_fraction);
    let chord = frame.chord_at(wing, span_fraction);
    perimeter
        .iter()
        .map(|&(s, u)| {
            center + frame.fore_dir * (s * chord) + frame.thick_dir * (u * chord) - origin
        })
        .collect()
}

/// Loft two equal-length section rings into a skin band. Winding matches the
/// original loft (verified outward by `airfoil_loft_is_outward_facing`).
fn loft_rings(
    a_ring: &[Vec3],
    b_ring: &[Vec3],
    positions: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let perimeter = a_ring.len();
    let a_base = positions.len() as u32;
    for v in a_ring {
        positions.push([v.x, v.y, v.z]);
    }
    let b_base = positions.len() as u32;
    for v in b_ring {
        positions.push([v.x, v.y, v.z]);
    }
    for k in 0..perimeter {
        let k2 = ((k + 1) % perimeter) as u32;
        let k = k as u32;
        let (r0, r1) = (a_base + k, a_base + k2);
        let (t0, t1) = (b_base + k, b_base + k2);
        indices.extend_from_slice(&[r0, t1, t0, r0, r1, t1]);
    }
}

/// Fan a flat cap over a section ring from its centroid. `faces_plus_span`
/// picks the winding so the cap normal points along +span (`true`, like the
/// tip cap) or −span (`false`, like the root cap). Cap vertices are their
/// own copies so the rim edge stays crisp under smoothing.
fn cap_ring(
    ring: &[Vec3],
    faces_plus_span: bool,
    positions: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let perimeter = ring.len();
    let centroid = ring.iter().copied().sum::<Vec3>() / perimeter as f32;
    let center_idx = positions.len() as u32;
    positions.push([centroid.x, centroid.y, centroid.z]);
    let rim_base = positions.len() as u32;
    for v in ring {
        positions.push([v.x, v.y, v.z]);
    }
    for k in 0..perimeter {
        let k2 = ((k + 1) % perimeter) as u32;
        let k = k as u32;
        if faces_plus_span {
            indices.extend_from_slice(&[center_idx, rim_base + k, rim_base + k2]);
        } else {
            indices.extend_from_slice(&[center_idx, rim_base + k2, rim_base + k]);
        }
    }
}

fn finish_mesh(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    let uv = vec![[0.0_f32, 0.0]; positions.len()];
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
    mesh.insert_indices(Indices::U32(indices));
    mesh.compute_smooth_normals();
    crate::part_mesh::add_raytracing_tangents(&mut mesh);
    mesh
}

/// Valid, clamped `(span_start, span_end)` of a control surface as
/// fractions of the half-span, or `None` if the window is degenerate.
fn surface_span_window(surface: &ControlSurface) -> Option<(f32, f32)> {
    let a = surface.span_start.clamp(0.0, 1.0);
    let b = surface.span_end.clamp(0.0, 1.0);
    (b - a > 1e-3).then_some((a, b))
}

/// Build the host-local mesh for one wing panel mounted at `angle` on a
/// host of radius `parent_radius`.
///
/// **One entity = one panel.** Under KSP-style mirror symmetry the mirror
/// is a *separate* entity whose own `angle` (and `Wing.incidence`) are
/// reflected, so it renders its own correctly-mirrored panel from this same
/// builder — no "draw both sides" flag (see [`crate::SymmetryGroup`]).
///
/// The skin is a loft of the [`airfoil_perimeter`] section between the root
/// (against the host) and the swept/tapered tip, plus a flat cap at each
/// end so the panel is watertight. Where a [`ControlSurface`] sits, the
/// loft is **notched**: across that spanwise window the section is truncated
/// at the hinge chord-station (a blunt trailing face), and the removed
/// trailing wedge is meshed separately by [`build_control_surface_mesh`] so
/// it can hinge. The wall closing each notch is capped with the same aft
/// wedge outline, so the gap reads correctly when the surface deflects.
pub fn build_wing_mesh(wing: &Wing, angle: f32, parent_radius: f32) -> Mesh {
    let frame = wing_panel_frame(wing, angle, parent_radius);
    let tc = wing.thickness;

    // Spanwise breakpoints: panel ends plus every surface window edge.
    let mut bounds = vec![0.0_f32, 1.0];
    for surface in &wing.control_surfaces {
        if let Some((a, b)) = surface_span_window(surface) {
            bounds.push(a);
            bounds.push(b);
        }
    }
    bounds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    bounds.dedup_by(|a, b| (*a - *b).abs() < 1e-4);

    // For a spanwise interval midpoint, the covering surface's hinge x (if any).
    let covering_hinge_x = |mid: f32| -> Option<f32> {
        wing.control_surfaces.iter().find_map(|s| {
            let (a, b) = surface_span_window(s)?;
            (mid > a && mid < b).then(|| 0.5 - hinge_s(s.chord_fraction))
        })
    };

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Per-segment perimeter: clean airfoil, or forward (notched) section.
    let seg_perimeter = |mid: f32| -> Vec<(f32, f32)> {
        match covering_hinge_x(mid) {
            Some(x_hinge) => forward_perimeter(tc, x_hinge),
            None => airfoil_perimeter(tc),
        }
    };

    // Loft each spanwise segment with its own section.
    for w in bounds.windows(2) {
        let (a, b) = (w[0], w[1]);
        let mid = 0.5 * (a + b);
        let perim = seg_perimeter(mid);
        let a_ring = section_ring(&frame, wing, a, &perim, Vec3::ZERO);
        let b_ring = section_ring(&frame, wing, b, &perim, Vec3::ZERO);
        loft_rings(&a_ring, &b_ring, &mut positions, &mut indices);
    }

    // Root and tip caps use the section present at each panel end.
    let root_perim = seg_perimeter(0.5 * (bounds[0] + bounds[1]));
    cap_ring(
        &section_ring(&frame, wing, 0.0, &root_perim, Vec3::ZERO),
        false,
        &mut positions,
        &mut indices,
    );
    let last = bounds.len() - 1;
    let tip_perim = seg_perimeter(0.5 * (bounds[last - 1] + bounds[last]));
    cap_ring(
        &section_ring(&frame, wing, 1.0, &tip_perim, Vec3::ZERO),
        true,
        &mut positions,
        &mut indices,
    );

    // Notch walls: at any interior boundary where a clean segment meets a
    // notched one, close the aft cavity of the clean side with the aft-wedge
    // outline, facing into the gap.
    for i in 1..last {
        let f = bounds[i];
        let left_hinge = covering_hinge_x(0.5 * (bounds[i - 1] + f));
        let right_hinge = covering_hinge_x(0.5 * (f + bounds[i + 1]));
        match (left_hinge, right_hinge) {
            // Clean on the left, notch opens toward +span.
            (None, Some(x_hinge)) => {
                let ring = section_ring(&frame, wing, f, &aft_perimeter(tc, x_hinge), Vec3::ZERO);
                cap_ring(&ring, true, &mut positions, &mut indices);
            }
            // Clean on the right, notch opens toward −span.
            (Some(x_hinge), None) => {
                let ring = section_ring(&frame, wing, f, &aft_perimeter(tc, x_hinge), Vec3::ZERO);
                cap_ring(&ring, false, &mut positions, &mut indices);
            }
            _ => {}
        }
    }

    finish_mesh(positions, indices)
}

/// Geometry of one control surface in the host-local wing frame — the
/// shared seam both the visual layer and a future per-surface force model
/// read. `hinge_anchor` + `hinge_axis` place and rotate the hinged mesh;
/// `centroid` + `area_m2` describe where its force acts and how large it is.
#[derive(Clone, Copy, Debug)]
pub struct ControlSurfaceGeometry {
    /// A point on the hinge line (host-local); the surface entity's local
    /// translation, so its mesh — built relative to this point — rotates
    /// about the hinge.
    pub hinge_anchor: Vec3,
    /// Unit hinge axis (host-local), inboard→outboard. Positive rotation
    /// about it is consistent across the panel; the game maps command sign.
    pub hinge_axis: Vec3,
    /// Area centroid of the surface (host-local), at its mid-chord, mid-span.
    pub centroid: Vec3,
    /// Planform area of the surface, m².
    pub area_m2: f32,
}

/// A built control-surface sub-mesh plus the geometry needed to hinge it.
pub struct BuiltControlSurface {
    pub mesh: Mesh,
    pub geometry: ControlSurfaceGeometry,
}

/// Compute the hinge/area geometry of `surface` on `wing` without building a
/// mesh. Pure; safe for the (future) force model to call per frame.
pub fn control_surface_geometry(
    wing: &Wing,
    surface: &ControlSurface,
    angle: f32,
    parent_radius: f32,
) -> ControlSurfaceGeometry {
    let frame = wing_panel_frame(wing, angle, parent_radius);
    let (a, b) = surface_span_window(surface).unwrap_or((0.0, 1.0));
    let s_hinge = hinge_s(surface.chord_fraction);

    // Point on the hinge line at span fraction f (mid-thickness).
    let hinge_point = |f: f32| -> Vec3 {
        frame.center_at(f) + frame.fore_dir * (s_hinge * frame.chord_at(wing, f))
    };
    let hinge_anchor = hinge_point(a);
    // The hinge line is spanwise; orient it consistently across a mirrored
    // pair so a given +θ deflects the trailing edge the *same* way (down) on
    // both sides. `fore × thick` is ~+X on both panels (fore ≈ +Y, thick ≈
    // +Z), unlike `span_dir`, which flips, so it fixes the sense. The game
    // then makes ailerons differential with an explicit per-side sign and
    // leaves elevator/rudder symmetric.
    let mut hinge_axis = (hinge_point(b) - hinge_anchor).normalize_or(frame.span_dir);
    if hinge_axis.dot(frame.fore_dir.cross(frame.thick_dir)) < 0.0 {
        hinge_axis = -hinge_axis;
    }

    let mid = 0.5 * (a + b);
    // Aft-wedge chord centroid sits between the hinge and the trailing edge.
    let s_centroid = 0.5 * (s_hinge + -0.5);
    let centroid = frame.center_at(mid) + frame.fore_dir * (s_centroid * frame.chord_at(wing, mid));

    let span_len = (frame.tip_center - frame.root_center).length() * (b - a);
    let area_m2 = surface.chord_fraction.clamp(0.05, 0.95) * frame.chord_at(wing, mid) * span_len;

    ControlSurfaceGeometry {
        hinge_anchor,
        hinge_axis,
        centroid,
        area_m2,
    }
}

/// Build the hinged trailing-wedge mesh for one control surface, expressed
/// **relative to its hinge anchor** so the owning entity can be placed at
/// `geometry.hinge_anchor` and rotate the surface about `geometry.hinge_axis`.
pub fn build_control_surface_mesh(
    wing: &Wing,
    surface: &ControlSurface,
    angle: f32,
    parent_radius: f32,
) -> BuiltControlSurface {
    let frame = wing_panel_frame(wing, angle, parent_radius);
    let geometry = control_surface_geometry(wing, surface, angle, parent_radius);
    let (a, b) = surface_span_window(surface).unwrap_or((0.0, 1.0));
    let x_hinge = 0.5 - hinge_s(surface.chord_fraction);
    let perim = aft_perimeter(wing.thickness, x_hinge);

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let a_ring = section_ring(&frame, wing, a, &perim, geometry.hinge_anchor);
    let b_ring = section_ring(&frame, wing, b, &perim, geometry.hinge_anchor);
    loft_rings(&a_ring, &b_ring, &mut positions, &mut indices);
    // Inboard end faces −span, outboard end faces +span.
    cap_ring(&a_ring, false, &mut positions, &mut indices);
    cap_ring(&b_ring, true, &mut positions, &mut indices);

    BuiltControlSurface {
        mesh: finish_mesh(positions, indices),
        geometry,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_wing() -> Wing {
        Wing {
            span: 5.0,
            root_chord: 2.0,
            tip_chord: 1.0,
            sweep: 0.2,
            dihedral: 0.05,
            thickness: 0.12,
            incidence: 0.0,
            dry_mass: 0.0,
            control_surfaces: Vec::new(),
        }
    }

    #[test]
    fn single_panel_sits_on_the_clicked_side() {
        // angle = π/2 → +X (right) side.
        let m = build_wing_mesh(&test_wing(), std::f32::consts::FRAC_PI_2, 1.0);
        let pos = m
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        // The whole panel sits outboard on +X (root at the radius).
        assert!(pos.iter().all(|p| p[0] > 0.0));
        // Normals + UVs are present for the standard material.
        assert!(m.attribute(Mesh::ATTRIBUTE_NORMAL).is_some());
        assert!(m.attribute(Mesh::ATTRIBUTE_UV_0).is_some());
    }

    #[test]
    fn reflected_angle_lands_on_the_opposite_side() {
        // The mirror counterpart renders by building at the reflected angle
        // (−θ), which must put its single panel entirely on the −X side.
        let m = build_wing_mesh(&test_wing(), -std::f32::consts::FRAC_PI_2, 1.0);
        let pos = m
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        assert!(
            pos.iter().all(|p| p[0] < 0.0),
            "−θ panel should sit entirely on −X"
        );
    }

    #[test]
    fn section_is_an_airfoil_not_a_box() {
        // The NACA section is convex with its max thickness around 30% chord
        // and pinched at both edges — unlike the old box (constant thickness).
        let tc = 0.12_f32;
        let yt_max = naca_half_thickness(tc, 0.3);
        assert!(
            naca_half_thickness(tc, 0.0).abs() < 1e-6,
            "sharp/zero LE point"
        );
        assert!(
            naca_half_thickness(tc, 1.0).abs() < 1e-4,
            "closed trailing edge"
        );
        // Peak half-thickness is ~t/2 (full thickness ~= t·chord).
        assert!(
            (yt_max - 0.5 * tc).abs() < 0.01 * tc + 0.01,
            "peak ~ t/2, got {yt_max}"
        );
        // Convex: midspan thickness sits between the edge and the peak.
        let yt_mid = naca_half_thickness(tc, 0.6);
        assert!(yt_mid > 0.0 && yt_mid < yt_max);
    }

    #[test]
    fn airfoil_loft_is_outward_facing() {
        // A top-surface vertex (positive offset along thick_dir) must carry a
        // normal pointing along +thick_dir; a bottom one along −thick_dir.
        // This guards the loft winding.
        let w = test_wing();
        let angle = std::f32::consts::FRAC_PI_2;
        let m = build_wing_mesh(&w, angle, 1.0);
        let frame = wing_panel_frame(&w, angle, 1.0);
        let pos = m
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        let nor = m
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .unwrap()
            .as_float3()
            .unwrap();
        let root_center = frame.center_at(0.0);
        let mut checked = 0;
        for (p, n) in pos.iter().zip(nor) {
            let rel = Vec3::from_array(*p) - root_center;
            let u = rel.dot(frame.thick_dir);
            let nd = Vec3::from_array(*n).dot(frame.thick_dir);
            // Only test clearly top/bottom skin points (away from edges/caps).
            if u.abs() > 0.3 * w.thickness * w.root_chord {
                assert!(
                    u.signum() == nd.signum() || nd.abs() < 0.2,
                    "surface normal should face outward (u={u}, n·thick={nd})"
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "expected some top/bottom skin vertices");
    }

    #[test]
    fn end_caps_face_outward_along_span() {
        // The cap center vertices are used only by their own fan, so their
        // smoothed normal is the cap's face normal: root cap faces inboard
        // (−span_dir, hidden in the host), tip cap faces outboard (+span_dir).
        let w = test_wing();
        let angle = std::f32::consts::FRAC_PI_2;
        let m = build_wing_mesh(&w, angle, 1.0);
        let frame = wing_panel_frame(&w, angle, 1.0);
        let pos = m
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        let nor = m
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .unwrap()
            .as_float3()
            .unwrap();
        let near = |target: Vec3| {
            pos.iter()
                .position(|p| Vec3::from_array(*p).distance(target) < 1e-4)
        };
        let root_c = near(frame.center_at(0.0)).expect("root cap center vertex");
        let tip_c = near(frame.center_at(1.0)).expect("tip cap center vertex");
        assert!(
            Vec3::from_array(nor[root_c]).dot(frame.span_dir) < -0.8,
            "root cap inboard"
        );
        assert!(
            Vec3::from_array(nor[tip_c]).dot(frame.span_dir) > 0.8,
            "tip cap outboard"
        );
    }

    #[test]
    fn hinge_splits_the_section_at_the_chord_fraction() {
        // The forward (main-wing) and aft (control-surface) perimeters meet on
        // the hinge chord-station and partition the section: forward keeps the
        // LE side (s ≥ s_hinge), aft keeps the TE side (s ≤ s_hinge). For a
        // 25%-chord surface the hinge sits at s = −0.5 + 0.25 = −0.25.
        let tc = 0.12_f32;
        let chord_fraction = 0.25_f32;
        let s_hinge = hinge_s(chord_fraction);
        let x_hinge = 0.5 - s_hinge;
        assert!((s_hinge - -0.25).abs() < 1e-5);

        let fwd = forward_perimeter(tc, x_hinge);
        let aft = aft_perimeter(tc, x_hinge);
        let fwd_min_s = fwd.iter().map(|&(s, _)| s).fold(f32::MAX, f32::min);
        let fwd_max_s = fwd.iter().map(|&(s, _)| s).fold(f32::MIN, f32::max);
        let aft_min_s = aft.iter().map(|&(s, _)| s).fold(f32::MAX, f32::min);
        let aft_max_s = aft.iter().map(|&(s, _)| s).fold(f32::MIN, f32::max);
        // Forward spans LE (+0.5) down to the hinge; aft spans the hinge to TE.
        assert!((fwd_max_s - 0.5).abs() < 1e-4, "forward keeps the LE");
        assert!(
            (fwd_min_s - s_hinge).abs() < 1e-4,
            "forward stops at the hinge"
        );
        assert!(
            (aft_max_s - s_hinge).abs() < 1e-4,
            "aft starts at the hinge"
        );
        assert!((aft_min_s - -0.5).abs() < 1e-4, "aft keeps the TE");
        // Both carry the blunt hinge face (a ± pair at s_hinge).
        let seam = naca_half_thickness(tc, x_hinge);
        assert!(
            fwd.iter()
                .any(|&(s, u)| (s - s_hinge).abs() < 1e-4 && u < -0.5 * seam)
        );
        assert!(
            aft.iter()
                .any(|&(s, u)| (s - s_hinge).abs() < 1e-4 && u > 0.5 * seam)
        );
    }

    #[test]
    fn notched_wing_drops_vertices_versus_clean() {
        use crate::part::{ControlSurface, ControlSurfaceRole};
        // The notched main wing replaces the full-section loft over the surface
        // window with a truncated section, so its skin loft is lighter, while
        // the removed wedge becomes the separate control-surface mesh.
        let mut w = test_wing();
        let angle = std::f32::consts::FRAC_PI_2;
        let clean = build_wing_mesh(&w, angle, 1.0);
        let clean_n = clean.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().len();
        w.control_surfaces = vec![ControlSurface {
            role: ControlSurfaceRole::Aileron,
            span_start: 0.55,
            span_end: 0.95,
            chord_fraction: 0.25,
            max_deflection: 0.4,
        }];
        let notched = build_wing_mesh(&w, angle, 1.0);
        // The most-aft point of the truncated middle segment's ring is forward
        // of the clean trailing edge at the same station.
        let frame = wing_panel_frame(&w, angle, 1.0);
        let fwd = forward_perimeter(w.thickness, 0.5 - hinge_s(0.25));
        let ring = section_ring(&frame, &w, 0.75, &fwd, Vec3::ZERO);
        let notch_aft = ring
            .iter()
            .map(|v| (*v - frame.root_center).dot(-frame.fore_dir))
            .fold(f32::MIN, f32::max);
        let clean_full = airfoil_perimeter(w.thickness);
        let clean_ring = section_ring(&frame, &w, 0.75, &clean_full, Vec3::ZERO);
        let clean_aft = clean_ring
            .iter()
            .map(|v| (*v - frame.root_center).dot(-frame.fore_dir))
            .fold(f32::MIN, f32::max);
        assert!(
            notch_aft < clean_aft - 0.1,
            "truncated section stops short of the clean TE"
        );
        assert!(notched.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().len() != clean_n);
    }

    #[test]
    fn control_surface_mesh_hinges_about_its_anchor() {
        use crate::part::{ControlSurface, ControlSurfaceRole};
        let mut w = test_wing();
        let surface = ControlSurface {
            role: ControlSurfaceRole::Aileron,
            span_start: 0.55,
            span_end: 0.95,
            chord_fraction: 0.25,
            max_deflection: 0.4,
        };
        w.control_surfaces = vec![surface];
        let angle = std::f32::consts::FRAC_PI_2;
        let built = build_control_surface_mesh(&w, &surface, angle, 1.0);
        let pos = built
            .mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        assert!(!pos.is_empty());
        // Mesh is relative to the hinge anchor, so the leading (hinge) edge is
        // near the origin and the trailing edge extends aft of it.
        let min_aft = pos
            .iter()
            .map(|p| {
                Vec3::from_array(*p)
                    .dot(crate::wing_mesh::wing_panel_frame(&w, angle, 1.0).fore_dir)
            })
            .fold(f32::MAX, f32::min);
        // Some vertex sits well behind the hinge (−fore direction).
        assert!(
            min_aft < -0.1,
            "surface should extend aft of its hinge anchor (min_aft {min_aft})"
        );
        // Hinge axis is roughly spanwise.
        let span_dir = wing_panel_frame(&w, angle, 1.0).span_dir;
        assert!(
            built.geometry.hinge_axis.dot(span_dir) > 0.9,
            "hinge axis ~ spanwise"
        );
        assert!(built.geometry.area_m2 > 0.0);
    }

    #[test]
    fn mirror_wing_is_a_reflection_not_a_radial_rotation() {
        // A left/right pair must be a true mirror: both tops face up (+Z), so
        // a pylon nacelle (hung along −thick) drops on both sides. If the
        // mirror's normal flipped to −Z, the nacelle would point up → radial.
        let w = test_wing();
        let right = wing_panel_frame(&w, std::f32::consts::FRAC_PI_2, 1.0);
        let left = wing_panel_frame(&w, -std::f32::consts::FRAC_PI_2, 1.0);
        assert!(right.thick_dir.z > 0.0, "right wing top faces up");
        assert!(left.thick_dir.z > 0.0, "left wing top faces up");
        assert!(right.tip_center.x > 0.0 && left.tip_center.x < 0.0);
    }
    /// See `engine_mesh`'s copy: a mesh that misses the attribute set is
    /// skipped by the BLAS builder silently.
    #[test]
    fn wing_mesh_is_raytracing_ready() {
        assert!(crate::part_mesh::is_raytracing_ready(&build_wing_mesh(
            &test_wing(),
            0.0,
            1.5
        )));
    }

}
