//! Runway **geometry**: the body-fixed frame, the paving/skirt/marking meshes,
//! the ICAO designator rasterizer, the post and surface materials, and the
//! authored site math.
//!
//! Pure appearance — a frame in, meshes and materials out. Everything that
//! *drives* a runway (deferred placement, the Avian collider, the per-frame
//! f64 anchoring, spaceport orchestration) stays in `thalos_runtime`: those
//! read canonical craft state and mutate the world, which is exactly the
//! boundary this crate exists to hold.

use bevy::image::Image;
use bevy::math::DVec3;
use bevy::mesh::Mesh;
use bevy::prelude::*;
use thalos_body_render::{ShadowedStandardMaterial, TerrainPatchBasis, shadowed};
use thalos_world::BodyId;

pub const RUNWAY_LENGTH_M: f64 = 5000.0;

pub const RUNWAY_HALF_LENGTH_M: f64 = RUNWAY_LENGTH_M * 0.5;

pub const RUNWAY_WIDTH_M: f64 = 90.0;

pub const RUNWAY_HALF_WIDTH_M: f64 = RUNWAY_WIDTH_M * 0.5;

// ---------------------------------------------------------------------------
// Secondary runway (a shorter crosswind strip angled off the primary)
// ---------------------------------------------------------------------------
//
// The base presents two runways in a **V**: the primary, plus a shorter
// crosswind strip that diverges from near the primary's `−along` threshold at
// `SECONDARY_HEADING_OFFSET_DEG` toward the empty (`+across`) side, opposite
// the launch complex — the classic main-plus-diagonal layout (Dulles' 12/30
// beside its parallels). The strips never intersect (the secondary starts
// offset to the side and only fans further away), and the shared V corner is
// where the taxiway system joins the two (see
// [`crate::base_editor::spawn_default_base`] — no run-around past the runway
// end, no runway crossing). The heading divergence gives each strip its own
// true-heading designator numbers, so no L/R suffix pair is needed. It is a
// plain parametric `StructureKind::Runway` registry entry that renders +
// collides through the exact same generalized path as the primary.
pub const SECONDARY_LENGTH_M: f64 = 3600.0;

pub const SECONDARY_WIDTH_M: f64 = 80.0;

/// Heading divergence of the secondary strip, rotated toward `+across` (the
/// side it sits on) so the pair fans apart with along, never converging.
pub const SECONDARY_HEADING_OFFSET_DEG: f64 = 30.0;

/// The secondary's near threshold (the V corner), in the primary's runway
/// frame: just short of the primary's `−along` threshold, offset to the empty
/// (`+across`) side far enough that the strips and their taxiways stay clear
/// of the primary strip.
pub const SEC_NEAR_ALONG_M: f64 = -2400.0;

pub const SEC_NEAR_ACROSS_M: f64 = 420.0;

/// Asphalt strip sits this far above the flat terrain pad so the paving reads
/// as a surface on the ground (and never z-fights the flattened tiles). The
/// strip's edge then drops back to the ground as a [`RUNWAY_SKIRT_DEPTH_M`]
/// skirt, so this lift shows as a curb rather than a see-through floating lip.
pub const RUNWAY_ASPHALT_LIFT_M: f64 = 0.12;

/// The paved strip's perimeter drops this far (m) as a vertical skirt that
/// buries into the levelled terrain. Without it the asphalt lift above shows as
/// a floating edge with grass visible underneath; the skirt fills that gap with
/// a curb whose lower edge sits below the (now flat, parallel) terrain plane, so
/// only the short above-ground band reads. Generous enough to clear the basin's
/// slight cut/fill and any tile-streaming height jitter.
pub const RUNWAY_SKIRT_DEPTH_M: f64 = 0.6;

/// Markings sit just above the asphalt to avoid z-fighting.
pub const RUNWAY_MARKING_LIFT_M: f64 = 0.17;

/// Top tessellation: segments along the length / across the width. The slab is
/// flat, so this is only for lighting/curvature — it can be coarse.
pub const RUNWAY_TOP_SEGMENTS_LEN: usize = 120;

pub const RUNWAY_TOP_SEGMENTS_W: usize = 4;

/// Subdivision length for marking strips (kept fine so dashes read cleanly).
pub const RUNWAY_MARKING_SEG_LEN_M: f64 = 25.0;

// --- Runway designator numbers (painted from the real ICAO font) ---
/// Height of the number block along the runway (m). Aviation numbers are large
/// — read from short final, filling a good half of the strip width. The
/// across-width follows the glyph aspect.
pub const NUM_DIGIT_H_M: f64 = 45.0;

/// Distance from the threshold to the near (baseline) edge of the number (m).
pub const NUM_THRESHOLD_MARGIN_M: f64 = 90.0;

/// Pixel height the ICAO glyphs are rasterized at (resolution of the decal
/// texture; the quad is metric, so this is just texture crispness).
pub const NUM_RASTER_PX_H: u32 = 512;

pub const POST_SPACING_M: f64 = 300.0;

pub const POST_EDGE_OFFSET_M: f64 = 4.0;

pub const POST_HEIGHT_M: f32 = 4.0;

pub const POST_THRESHOLD_HEIGHT_M: f32 = 6.0;

pub const POST_SIZE_M: f32 = 0.5;

/// Runway-centre latitude (deg, body-fixed, +north).
pub const RUNWAY_SITE_LAT_DEG: f64 = 7.6;

/// Runway-centre longitude (deg, body-fixed, +east; `atan2(z, x)`, matching the
/// site log line).
pub const RUNWAY_SITE_LON_DEG: f64 = 178.0;

// Under `THALOS_TERRAIN=diffusion` (NTR-X2a) the same site holds: the
// conditioned diffusion export keeps the canonical continents, and the
// corrected 90 m window puts lat 7.6/lon 178 on a ~700 m plateau with ~19 m
// relief over 5.8 km — flat enough that the basin flatten barely works.
// Re-scan (`thalos_export.py` + a block scan) if the export seed changes.
/// Takeoff-heading azimuth (deg) in the local tangent frame, measured from
/// `TerrainPatchBasis::tangent_x` toward `tangent_z`. Any fixed value gives a
/// constant strip; the pad is flat regardless of which way it points.
pub const RUNWAY_SITE_HEADING_DEG: f64 = 30.0;

/// Embedded ICAO runway-designator font (the aviation typeface for runway
/// numbers). Rasterized to an alpha decal painted on the strip.
pub const ICAO_FONT: &[u8] = include_bytes!("../../../../assets/fonts/ICAORWYID.ttf");

/// The body-fixed runway frame on the **shared basin tangent plane**. All
/// runways on the basin reference the *same* plane (normal `center_dir`, at
/// elevation `E`), so they lie flush with the single flattened terrain — a strip
/// is positioned within that plane by `center_offset`, not given its own tangent
/// plane at its own centre (which, offset across the sphere, would tilt away from
/// the flattened ground and sink the strip into it).
pub struct RunwayFrame {
    pub body_id: BodyId,
    /// Basin tangent-plane normal (the shared "up" for every runway on the
    /// basin). NOT the strip's own centre direction when it is offset.
    pub center_dir: DVec3,
    pub heading: DVec3,
    pub across: DVec3,
    pub body_radius_m: f64,
    pub elevation_m: f64,
    /// In-plane offset (world metres) of this strip's centre from the plane
    /// origin (`center_dir·(R+E)`) — zero for the primary, `across·SEP` for the
    /// offset secondary. Keeps every strip on the one basin plane.
    pub center_offset: DVec3,
    /// Which side of a parallel-runway pair this strip is on, for the L/R
    /// designator suffix: `-1` = the `−across` side, `+1` = the `+across` side,
    /// `0` = a lone runway (no suffix). The suffix flips L↔R between the two
    /// thresholds (as in real "07L / 25R").
    pub pair_side: i8,
    /// Half-length along the heading (m) — per-runway, so the primary and the
    /// secondary strip share one geometry path.
    pub half_length_m: f64,
    /// Half-width across the heading (m).
    pub half_width_m: f64,
}

impl RunwayFrame {
    pub fn center_surface(&self) -> DVec3 {
        self.center_dir * (self.body_radius_m + self.elevation_m) + self.center_offset
    }

    /// Body-fixed offset (from the pad centre) of a runway coordinate, plus the
    /// surface normal there. The runway is a **true flat plane** — the tangent
    /// plane at the site centre — not a sphere-draped strip: the normal is the
    /// constant `center_dir` and `height_offset` lifts straight up off the
    /// plane. This is what makes the paving, markings, the cuboid collider, and
    /// the parked-craft rest pose share one surface. A sphere-draped strip would
    /// diverge from the flat collider by the curvature drop (~0.3 m at the
    /// parked station, ~0.4 m at the runway ends) — exactly the gap that buried
    /// the gear at rest and launched the craft up when physics took over.
    pub fn level(&self, along_m: f64, across_m: f64, height_offset: f64) -> (DVec3, DVec3) {
        let up = self.center_dir;
        let offset = self.heading * along_m + self.across * across_m + up * height_offset;
        (offset, up)
    }
}

pub fn latlon_dir(lat_deg: f64, lon_deg: f64) -> DVec3 {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin()).normalize()
}

/// The fixed runway site: a constant body-fixed centre direction plus the
/// takeoff-heading tangent, from the compile-time `RUNWAY_SITE_*` constants and
/// overridable at runtime with `THALOS_RUNWAY_SITE="lat_deg,lon_deg[,heading_deg]"`.
/// No terrain sampling and no epoch dependence — the same spot every spawn.
pub fn fixed_runway_site() -> (DVec3, DVec3) {
    let (lat_deg, lon_deg, heading_deg) = std::env::var("THALOS_RUNWAY_SITE")
        .ok()
        .and_then(|raw| match parse_site_override(&raw) {
            Some(site) => Some(site),
            None => {
                warn!("THALOS_RUNWAY_SITE=\"{raw}\" is not \"lat,lon[,heading]\" — using defaults");
                None
            }
        })
        .unwrap_or((
            RUNWAY_SITE_LAT_DEG,
            RUNWAY_SITE_LON_DEG,
            RUNWAY_SITE_HEADING_DEG,
        ));

    let center_dir = latlon_dir(lat_deg, lon_deg);
    let basis = TerrainPatchBasis::from_normal(center_dir);
    let az = heading_deg.to_radians();
    let heading = (basis.tangent_x * az.cos() + basis.tangent_z * az.sin())
        .try_normalize()
        .unwrap_or(basis.tangent_x);
    (center_dir, heading)
}

/// Parse `"lat,lon"` or `"lat,lon,heading"` (degrees). Returns `None` (defaults
/// used, with a warning at the call site) on any malformed field.
pub fn parse_site_override(raw: &str) -> Option<(f64, f64, f64)> {
    let mut parts = raw.split(',').map(|p| p.trim());
    let lat = parts.next()?.parse::<f64>().ok()?;
    let lon = parts.next()?.parse::<f64>().ok()?;
    let heading = match parts.next() {
        Some(h) => h.parse::<f64>().ok()?,
        None => RUNWAY_SITE_HEADING_DEG,
    };
    // Reject trailing garbage like "0,0,0,extra".
    if parts.next().is_some() {
        return None;
    }
    Some((lat, lon, heading))
}

pub fn build_mesh(
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
) -> Mesh {
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::{Indices, PrimitiveTopology};
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

pub fn build_top_mesh(frame: &RunwayFrame) -> Mesh {
    let nl = RUNWAY_TOP_SEGMENTS_LEN;
    let nw = RUNWAY_TOP_SEGMENTS_W;
    let length = 2.0 * frame.half_length_m;
    let width = 2.0 * frame.half_width_m;
    let mut positions = Vec::with_capacity((nl + 1) * (nw + 1));
    let mut normals = Vec::with_capacity((nl + 1) * (nw + 1));
    let mut uvs = Vec::with_capacity((nl + 1) * (nw + 1));
    let mut indices = Vec::with_capacity(nl * nw * 6);
    for i in 0..=nl {
        let along = -frame.half_length_m + length * (i as f64 / nl as f64);
        let v = i as f32 / nl as f32;
        for j in 0..=nw {
            let across_m = -frame.half_width_m + width * (j as f64 / nw as f64);
            let u = j as f32 / nw as f32;
            let (off, up) = frame.level(along, across_m, RUNWAY_ASPHALT_LIFT_M);
            positions.push([off.x as f32, off.y as f32, off.z as f32]);
            normals.push([up.x as f32, up.y as f32, up.z as f32]);
            uvs.push([u, v]);
        }
    }
    let row = (nw + 1) as u32;
    for i in 0..nl as u32 {
        for j in 0..nw as u32 {
            let a = i * row + j;
            let b = a + 1;
            let c = a + row;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    build_mesh(positions, normals, uvs, indices)
}

/// A thin vertical skirt around the paved strip's perimeter, dropping from the
/// asphalt edge (`RUNWAY_ASPHALT_LIFT_M`) down `RUNWAY_SKIRT_DEPTH_M` into the
/// terrain. The strip top is lifted a few cm off the levelled ground so it reads
/// as paving and never z-fights the tiles; without this skirt that lift shows as
/// a floating lip with grass visible under the edge. The skirt fills it with a
/// curb — its lower edge buries below the (now flat, parallel) terrain plane, so
/// only the short above-ground band is visible. Four flat quads suffice: the
/// strip is a true flat plane, so its perimeter is an exact rectangle.
pub fn build_skirt_mesh(frame: &RunwayFrame) -> Mesh {
    let half_l = frame.half_length_m;
    let half_w = frame.half_width_m;
    let top = RUNWAY_ASPHALT_LIFT_M;
    let bot = RUNWAY_ASPHALT_LIFT_M - RUNWAY_SKIRT_DEPTH_M;
    // Perimeter corners (along, across), counter-clockwise around the strip.
    let corners = [
        (-half_l, -half_w),
        (half_l, -half_w),
        (half_l, half_w),
        (-half_l, half_w),
    ];
    let mut positions = Vec::with_capacity(16);
    let mut normals = Vec::with_capacity(16);
    let mut uvs = Vec::with_capacity(16);
    let mut indices = Vec::with_capacity(24);
    for i in 0..4 {
        let (a0, c0) = corners[i];
        let (a1, c1) = corners[(i + 1) % 4];
        let (top0, up) = frame.level(a0, c0, top);
        let (top1, _) = frame.level(a1, c1, top);
        let (bot0, _) = frame.level(a0, c0, bot);
        let (bot1, _) = frame.level(a1, c1, bot);
        // Outward face normal (edge × up). The asphalt material is double-sided,
        // so the sign only affects shading, not visibility.
        let outward = (top1 - top0).cross(up).normalize_or_zero();
        let n = [outward.x as f32, outward.y as f32, outward.z as f32];
        let base = positions.len() as u32;
        for (pt, uv) in [
            (top0, [0.0, 0.0]),
            (top1, [1.0, 0.0]),
            (bot0, [0.0, 1.0]),
            (bot1, [1.0, 1.0]),
        ] {
            positions.push([pt.x as f32, pt.y as f32, pt.z as f32]);
            normals.push(n);
            uvs.push(uv);
        }
        indices.extend_from_slice(&[base, base + 2, base + 1, base + 1, base + 2, base + 3]);
    }
    build_mesh(positions, normals, uvs, indices)
}

pub fn build_markings_mesh(frame: &RunwayFrame) -> Mesh {
    let mut p = Vec::new();
    let mut n = Vec::new();
    let mut u = Vec::new();
    let mut idx = Vec::new();

    let half_w = frame.half_width_m;
    let half_l = frame.half_length_m;

    // Side edge lines (1 m wide, set in 1.5 m from the edge).
    let edge_c = half_w - 1.5;
    for sign in [-1.0, 1.0] {
        let c = sign * edge_c;
        push_marking_strip(
            &mut p,
            &mut n,
            &mut u,
            &mut idx,
            frame,
            -half_l + 60.0,
            half_l - 60.0,
            c - 0.5,
            c + 0.5,
        );
    }
    // Dashed centreline (1 m wide; 30 m dash / 20 m gap).
    let mut a = -half_l + 120.0;
    while a + 30.0 < half_l - 120.0 {
        push_marking_strip(
            &mut p,
            &mut n,
            &mut u,
            &mut idx,
            frame,
            a,
            a + 30.0,
            -0.5,
            0.5,
        );
        a += 50.0;
    }
    // Threshold bars (solid, ~10 m along, near each end).
    let bar_in = half_w - 3.0;
    push_marking_strip(
        &mut p,
        &mut n,
        &mut u,
        &mut idx,
        frame,
        -half_l + 30.0,
        -half_l + 40.0,
        -bar_in,
        bar_in,
    );
    push_marking_strip(
        &mut p,
        &mut n,
        &mut u,
        &mut idx,
        frame,
        half_l - 40.0,
        half_l - 30.0,
        -bar_in,
        bar_in,
    );
    // Touchdown aiming blocks (a pair flanking the centreline near each end).
    for end in [-1.0, 1.0] {
        let a0 = end * (half_l - 360.0);
        let a1 = end * (half_l - 280.0);
        let (lo, hi) = if a0 < a1 { (a0, a1) } else { (a1, a0) };
        for off in [-9.0, 5.0] {
            push_marking_strip(
                &mut p,
                &mut n,
                &mut u,
                &mut idx,
                frame,
                lo,
                hi,
                off,
                off + 4.0,
            );
        }
    }

    // Note: the runway designator *numbers* are not part of this white paint
    // mesh — they are painted from the real ICAO font as textured decal quads,
    // see `spawn_runway_numbers`.

    build_mesh(p, n, u, idx)
}

#[allow(clippy::too_many_arguments)]
pub fn push_marking_strip(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u32>,
    frame: &RunwayFrame,
    along0: f64,
    along1: f64,
    across0: f64,
    across1: f64,
) {
    let len = (along1 - along0).abs();
    let segs = ((len / RUNWAY_MARKING_SEG_LEN_M).ceil() as usize).max(1);
    let base = positions.len() as u32;
    for i in 0..=segs {
        let t = i as f64 / segs as f64;
        let along = along0 + (along1 - along0) * t;
        for (j, &ac) in [across0, across1].iter().enumerate() {
            let (off, up) = frame.level(along, ac, RUNWAY_MARKING_LIFT_M);
            positions.push([off.x as f32, off.y as f32, off.z as f32]);
            normals.push([up.x as f32, up.y as f32, up.z as f32]);
            uvs.push([j as f32, t as f32]);
        }
    }
    let row = 2u32;
    for i in 0..segs as u32 {
        let a = base + i * row;
        let b = a + 1;
        let c = a + row;
        let d = c + 1;
        indices.extend_from_slice(&[a, c, b, b, c, d]);
    }
}
/// A flat quad on the runway plane for a designator decal, centred across the
/// strip at `along_center`, with the glyph texture UV-mapped so the top of the
/// digits points down-runway (`rot180` flips it 180° for the far threshold).
pub fn build_number_quad(
    frame: &RunwayFrame,
    along_center: f64,
    half_along: f64,
    half_across: f64,
    rot180: bool,
) -> Mesh {
    let s = if rot180 { -1.0 } else { 1.0 };
    // (u, v, along-sign, across-sign): v=0 (image top) → +along (down-runway),
    // and u=0 (image left) → +across. A pilot on approach at the near threshold
    // has forward = +heading and up, so their right = heading × up = −across;
    // mapping the image's left edge to +across therefore puts the text's left on
    // the pilot's left, so the digits read upright instead of mirrored. The far
    // end (s = −1) flips both signs to match its opposite approach direction.
    let verts = [
        (0.0f32, 0.0f32, 1.0f64, 1.0f64),
        (1.0, 0.0, 1.0, -1.0),
        (0.0, 1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0, -1.0),
    ];
    let mut positions = Vec::with_capacity(4);
    let mut normals = Vec::with_capacity(4);
    let mut uvs = Vec::with_capacity(4);
    for (u, v, a_sign, c_sign) in verts {
        let along = along_center + s * a_sign * half_along;
        let across = s * c_sign * half_across;
        let (off, up) = frame.level(along, across, RUNWAY_MARKING_LIFT_M);
        positions.push([off.x as f32, off.y as f32, off.z as f32]);
        normals.push([up.x as f32, up.y as f32, up.z as f32]);
        uvs.push([u, v]);
    }
    // Wind the front face UP. The material is double-sided, so a back-facing
    // fragment has its shading normal flipped to −up (facing away from the sun),
    // shading the digits ambient-only — a dark smudge instead of white paint.
    // Flipping the across-sign above (to un-mirror the text) reversed the
    // triangle handedness, so the winding is reversed here (vs. the pre-fix
    // `0,1,2,1,3,2`) to keep the top face the front, sun-lit one.
    build_mesh(positions, normals, uvs, vec![0, 2, 1, 1, 2, 3])
}

/// True compass heading (deg, 0 = north, 90 = east) of the runway's takeoff
/// direction (`frame.heading`), computed in the local ENU frame at the runway
/// centre. Thalos spins about +Y, so +Y is the north-pole axis.
pub fn runway_heading_deg(frame: &RunwayFrame) -> f64 {
    let up = frame.center_dir;
    let pole = DVec3::Y;
    let north = (pole - up * pole.dot(up))
        .try_normalize()
        .unwrap_or_else(|| {
            // At a pole the tangent north is undefined; pick any tangent.
            let seed = if up.x.abs() < 0.9 { DVec3::X } else { DVec3::Z };
            (seed - up * seed.dot(up)).normalize()
        });
    let east = up.cross(north).normalize();
    let h = frame.heading;
    h.dot(east)
        .atan2(h.dot(north))
        .to_degrees()
        .rem_euclid(360.0)
}

/// Runway designator digit (01–36) from a compass heading, rounded to the
/// nearest 10°. Matches the HUD's `hud::mfd::runway_number` convention.
pub fn runway_designator(heading_deg: f64) -> u8 {
    let mut n = (heading_deg.rem_euclid(360.0) / 10.0).round() as i32;
    if n <= 0 {
        n += 36;
    } else if n > 36 {
        n -= 36;
    }
    n as u8
}

/// Rasterize a runway designator string (e.g. "07", "25R") from the ICAO font
/// into an RGBA8 image: white (RGB = 0xE6) with the glyph coverage in the alpha
/// channel. Returns `(width, height, pixels)`, tightly fit to the glyphs. `None`
/// if the font fails to load.
pub fn rasterize_designator(text: &str) -> Option<(u32, u32, Vec<u8>)> {
    use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
    let font = FontRef::try_from_slice(ICAO_FONT).ok()?;
    let scale = PxScale::from(NUM_RASTER_PX_H as f32);
    let scaled = font.as_scaled(scale);

    // Outline every glyph at its pen position and collect the union of the ink
    // bounds. The bitmap is then cropped tight to the actual painted pixels, so
    // the quad centres on the glyphs themselves — not on the font's metric box
    // (whose side bearings, trailing advance, and baseline/ascent padding would
    // otherwise push the number off the runway centreline and along-runway).
    let baseline = scaled.ascent();
    let mut pen_x = 0.0f32;
    let mut outlines = Vec::new();
    let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
    let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
    for ch in text.chars() {
        let id = font.glyph_id(ch);
        let glyph = id.with_scale_and_position(scale, ab_glyph::point(pen_x, baseline));
        if let Some(outlined) = font.outline_glyph(glyph) {
            let b = outlined.px_bounds();
            min_x = min_x.min(b.min.x);
            min_y = min_y.min(b.min.y);
            max_x = max_x.max(b.max.x);
            max_y = max_y.max(b.max.y);
            outlines.push(outlined);
        }
        pen_x += scaled.h_advance(id);
    }
    if outlines.is_empty() {
        return None;
    }
    let off_x = min_x.floor() as i32;
    let off_y = min_y.floor() as i32;
    let w = (max_x.ceil() as i32 - off_x).max(1) as u32;
    let h = (max_y.ceil() as i32 - off_y).max(1) as u32;

    // White pixels, alpha 0 (transparent) — the glyph coverage fills alpha.
    let mut pixels = vec![0u8; (w * h * 4) as usize];
    for px in pixels.chunks_exact_mut(4) {
        px[0] = 230;
        px[1] = 230;
        px[2] = 230;
    }

    for outlined in &outlines {
        let bounds = outlined.px_bounds();
        outlined.draw(|gx, gy, coverage| {
            let x = bounds.min.x as i32 - off_x + gx as i32;
            let y = bounds.min.y as i32 - off_y + gy as i32;
            if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
                let i = ((y as u32 * w + x as u32) * 4) as usize;
                let a = (coverage * 255.0).clamp(0.0, 255.0) as u8;
                pixels[i + 3] = pixels[i + 3].max(a);
            }
        });
    }
    Some((w, h, pixels))
}

/// Wrap RGBA8 pixels in a Bevy `Image` (mirrors `navball::markers::image_from_rgba8`).
pub fn image_from_alpha_rgba8(width: u32, height: u32, pixels: Vec<u8>) -> Image {
    use bevy::asset::RenderAssetUsages;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
    Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixels,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    )
}

pub fn flat_runway_material(
    materials: &mut Assets<ShadowedStandardMaterial>,
    color: Color,
    rough: f32,
) -> Handle<ShadowedStandardMaterial> {
    // Shadow-receiving (F6): the runway paving darkens under the craft, the
    // posts, and the base structures like the terrain around it does.
    materials.add(shadowed(StandardMaterial {
        base_color: color,
        perceptual_roughness: rough,
        metallic: 0.0,
        double_sided: true,
        cull_mode: None,
        ..default()
    }))
}

pub fn post_material(color: Color) -> ShadowedStandardMaterial {
    shadowed(StandardMaterial {
        base_color: color,
        emissive: color.to_linear() * 0.25,
        perceptual_roughness: 0.6,
        metallic: 0.0,
        ..default()
    })
}
