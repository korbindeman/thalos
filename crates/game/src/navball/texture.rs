//! Procedural navball texture generation.
//!
//! Produces an equirectangular RGBA8 image:
//! - Upper hemisphere (lat > 0): sky blue.
//! - Lower hemisphere (lat < 0): ground brown.
//! - Equator: a bright horizon band.
//! - Latitude/longitude grid at 10° (minor) and 30° (major).
//! - Cardinal letters N/E/S/W on the sky side just above the horizon.
//! - Pitch labels "30"/"60" at each cardinal longitude, both hemispheres.
//! - Zenith / nadir markers at the poles.
//!
//! No Bevy dependency — returns `Vec<u8>`, RGBA8, row-major.
//! Layout: pixel (x, y) → longitude in [-180°, 180°), latitude in [+90°, -90°].
//! u = 0.5 corresponds to lon = 0° (the "N" face).

use ab_glyph::{Font, FontRef, PxScale, ScaleFont};

/// Default texture dimensions. Width should always be 2× height.
pub const DEFAULT_WIDTH: u32 = 2048;
pub const DEFAULT_HEIGHT: u32 = 1024;

/// Sky-hemisphere base colour.
const SKY: [u8; 3] = [70, 130, 200];
/// Ground-hemisphere base colour.
const GROUND: [u8; 3] = [165, 110, 70];
/// Horizon band colour (bright accent at lat = 0).
const HORIZON: [u8; 3] = [240, 240, 240];
/// Major grid line colour (every 30°). Toned down so the grid doesn't read as a wireframe.
const LINE_MAJOR: [u8; 3] = [205, 205, 205];
/// Minor grid line colour (every 10°).
const LINE_MINOR: [u8; 3] = [180, 180, 180];
/// Cardinal-letter glyph colour.
const GLYPH_CARDINAL: [u8; 3] = [25, 25, 25];
/// Pitch-number glyph colour. Slightly lighter so the cardinals dominate.
const GLYPH_PITCH: [u8; 3] = [40, 40, 40];
/// Marker fill for zenith / nadir.
const POLE_MARK: [u8; 3] = [25, 25, 25];

/// Embedded font for glyphs. Fira Code Regular, OFL.
const FONT_BYTES: &[u8] = include_bytes!("../../../../assets/fonts/FiraCode-Regular.ttf");

/// Generate an equirectangular navball texture.
///
/// `width` should equal `2 * height` (true equirect aspect). Returns RGBA8
/// pixel data of length `width * height * 4`.
pub fn generate_navball_rgba8(width: u32, height: u32) -> Vec<u8> {
    assert!(width > 0 && height > 0, "navball texture must be non-empty");

    let mut pixels = vec![0u8; (width as usize) * (height as usize) * 4];

    // ---- Hemispheres + grid + horizon ----------------------------------
    for y in 0..height {
        let v = (y as f32 + 0.5) / height as f32;
        let lat_deg = (0.5 - v) * 180.0;

        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let lon_deg = (u - 0.5) * 360.0;

            let mut color = if lat_deg >= 0.0 { SKY } else { GROUND };

            // Longitude lines clip near the poles to avoid solid-white caps.
            let lon_lat_clip = 80.0;

            // Minor grid (10°).
            let near_minor_lat = nearest_grid_dist(lat_deg, 10.0) < 0.30;
            let near_minor_lon =
                nearest_grid_dist(lon_deg, 10.0) < 0.30 && lat_deg.abs() < lon_lat_clip;
            if near_minor_lat || near_minor_lon {
                color = blend(color, LINE_MINOR, 0.45);
            }

            // Major grid (30°) — overlay on the minor grid.
            let near_major_lat = nearest_grid_dist(lat_deg, 30.0) < 0.45;
            let near_major_lon =
                nearest_grid_dist(lon_deg, 30.0) < 0.45 && lat_deg.abs() < lon_lat_clip + 5.0;
            if near_major_lat || near_major_lon {
                color = blend(color, LINE_MAJOR, 0.85);
            }

            // Horizon band.
            if lat_deg.abs() < 1.0 {
                color = HORIZON;
            }

            let i = ((y * width + x) * 4) as usize;
            pixels[i] = color[0];
            pixels[i + 1] = color[1];
            pixels[i + 2] = color[2];
            pixels[i + 3] = 255;
        }
    }

    // ---- Pole markers (zenith / nadir) ---------------------------------
    draw_pole_marker(&mut pixels, width, height, true); // zenith
    draw_pole_marker(&mut pixels, width, height, false); // nadir

    // ---- Glyphs --------------------------------------------------------
    let font = FontRef::try_from_slice(FONT_BYTES).expect("valid embedded TTF");

    // Sizing scales with texture height so the look is resolution-independent.
    let cardinal_px = height as f32 * 0.08;
    let pitch_px = height as f32 * 0.055;

    // Cardinal letters, ~12° above the horizon on the sky side.
    let cardinal_lat = 12.0;
    for (ch, lon) in [('N', 0.0), ('E', 90.0), ('S', 180.0), ('W', -90.0)] {
        draw_text_centered(
            &mut pixels,
            width,
            height,
            &font,
            cardinal_px,
            &ch.to_string(),
            lon,
            cardinal_lat,
            GLYPH_CARDINAL,
            /* knockout */ true,
        );
    }

    // Pitch labels at (±30°, ±60°) × cardinal longitudes.
    let pitch_lons = [0.0_f32, 90.0, 180.0, -90.0];
    for &lon in &pitch_lons {
        for &lat in &[30.0_f32, 60.0, -30.0, -60.0] {
            let label = format!("{}", lat.abs() as i32);
            draw_text_centered(
                &mut pixels,
                width,
                height,
                &font,
                pitch_px,
                &label,
                lon,
                lat,
                GLYPH_PITCH,
                /* knockout */ true,
            );
        }
    }

    pixels
}

/// Distance in degrees from `deg` to the nearest multiple of `grid`.
fn nearest_grid_dist(deg: f32, grid: f32) -> f32 {
    let shifted = (deg + grid * 0.5).rem_euclid(grid);
    (shifted - grid * 0.5).abs()
}

fn blend(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        lerp_u8(a[0], b[0], t),
        lerp_u8(a[1], b[1], t),
        lerp_u8(a[2], b[2], t),
    ]
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 * (1.0 - t) + b as f32 * t)
        .round()
        .clamp(0.0, 255.0) as u8
}

// ---------------------------------------------------------------------------
// Glyph drawing via ab_glyph.
// ---------------------------------------------------------------------------

/// Render `text` centred at the given (lon, lat) on the equirect texture.
///
/// When `knockout` is true, the glyph footprint is repainted with the local
/// hemisphere colour before the glyphs are blended on top, so grid / horizon
/// lines don't intrude into the letterforms.
fn draw_text_centered(
    pixels: &mut [u8],
    tex_w: u32,
    tex_h: u32,
    font: &FontRef<'_>,
    px: f32,
    text: &str,
    lon_deg: f32,
    lat_deg: f32,
    color: [u8; 3],
    knockout: bool,
) {
    let scale = PxScale::from(px);
    let scaled = font.as_scaled(scale);

    // Measure the run.
    let mut width_px = 0.0_f32;
    let mut last_glyph_id = None;
    for ch in text.chars() {
        let id = scaled.glyph_id(ch);
        if let Some(prev) = last_glyph_id {
            width_px += scaled.kern(prev, id);
        }
        width_px += scaled.h_advance(id);
        last_glyph_id = Some(id);
    }
    let ascent = scaled.ascent();
    let descent = scaled.descent();
    let text_height = ascent - descent;

    // Pixel-space centre of the text on the texture.
    let u_center = (lon_deg / 360.0) + 0.5;
    let v_center = 0.5 - (lat_deg / 180.0);
    let cx = u_center * tex_w as f32;
    let cy = v_center * tex_h as f32;

    // Knockout padding so the cleared region is a hair larger than the glyph
    // run — keeps grid lines from touching letter edges.
    let pad = (px * 0.18).max(2.0);

    if knockout {
        let left = (cx - width_px * 0.5 - pad).floor() as i32;
        let right = (cx + width_px * 0.5 + pad).ceil() as i32;
        let top = (cy - text_height * 0.5 - pad).floor() as i32;
        let bottom = (cy + text_height * 0.5 + pad).ceil() as i32;

        for py in top..bottom {
            if py < 0 || py >= tex_h as i32 {
                continue;
            }
            let v = (py as f32 + 0.5) / tex_h as f32;
            let lat = (0.5 - v) * 180.0;
            let base = if lat >= 0.0 { SKY } else { GROUND };
            for px_x in left..right {
                let wx = px_x.rem_euclid(tex_w as i32) as u32;
                let wy = py as u32;
                let i = ((wy * tex_w + wx) * 4) as usize;
                pixels[i] = base[0];
                pixels[i + 1] = base[1];
                pixels[i + 2] = base[2];
                pixels[i + 3] = 255;
            }
        }
    }

    // Lay glyphs left-to-right starting at the centre offset.
    let baseline_x = cx - width_px * 0.5;
    let baseline_y = cy + text_height * 0.5 - descent.abs();

    let mut pen_x = baseline_x;
    let mut prev_id = None;
    for ch in text.chars() {
        let id = scaled.glyph_id(ch);
        if let Some(prev) = prev_id {
            pen_x += scaled.kern(prev, id);
        }
        let glyph = id.with_scale_and_position(scale, ab_glyph::point(pen_x, baseline_y));
        if let Some(outlined) = font.outline_glyph(glyph) {
            let bb = outlined.px_bounds();
            outlined.draw(|gx, gy, coverage| {
                if coverage <= 0.0 {
                    return;
                }
                let px_x = bb.min.x as i32 + gx as i32;
                let py = bb.min.y as i32 + gy as i32;
                if py < 0 || py >= tex_h as i32 {
                    return;
                }
                let wx = px_x.rem_euclid(tex_w as i32) as u32;
                let wy = py as u32;
                let i = ((wy * tex_w + wx) * 4) as usize;
                let dst = [pixels[i], pixels[i + 1], pixels[i + 2]];
                let blended = blend(dst, color, coverage);
                pixels[i] = blended[0];
                pixels[i + 1] = blended[1];
                pixels[i + 2] = blended[2];
                pixels[i + 3] = 255;
            });
        }
        pen_x += scaled.h_advance(id);
        prev_id = Some(id);
    }
}

// ---------------------------------------------------------------------------
// Pole markers.
// ---------------------------------------------------------------------------

/// Solid filled triangle near the pole. Pointing up at zenith, down at nadir.
fn draw_pole_marker(pixels: &mut [u8], tex_w: u32, tex_h: u32, zenith: bool) {
    // Marker covers a band of ~3° around the pole, drawn as a solid coloured
    // strip — equirect compresses it to a point on the sphere anyway.
    let band_height = (tex_h as f32 * (3.0 / 180.0)) as i32;
    let pole_y = if zenith { 0 } else { tex_h as i32 - 1 };
    let dir: i32 = if zenith { 1 } else { -1 };

    for k in 0..band_height {
        let py = pole_y + dir * k;
        if py < 0 || py >= tex_h as i32 {
            continue;
        }
        for x in 0..tex_w {
            let i = ((py as u32 * tex_w + x) * 4) as usize;
            pixels[i] = POLE_MARK[0];
            pixels[i + 1] = POLE_MARK[1];
            pixels[i + 2] = POLE_MARK[2];
            pixels[i + 3] = 255;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dump the navball texture to a PNG for visual inspection.
    ///
    /// Run with `cargo test -p thalos_game dump_png -- --nocapture`.
    #[test]
    fn dump_png() {
        let w = DEFAULT_WIDTH;
        let h = DEFAULT_HEIGHT;
        let pixels = generate_navball_rgba8(w, h);
        assert_eq!(pixels.len(), (w * h * 4) as usize);

        let img = image::RgbaImage::from_raw(w, h, pixels).expect("buffer size matches");
        let path = std::env::temp_dir().join("thalos_navball.png");
        img.save(&path).expect("write navball PNG");
        println!("wrote navball texture to {}", path.display());
    }
}
