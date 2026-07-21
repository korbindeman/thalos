//! Visualize the shared [`landcover`](thalos_body_render::sample_landcover)
//! field — the large-scale moisture → grass-colour / coverage model the in-game
//! grass (and the terrain albedo) read from. Pure CPU; writes PNGs under
//! `artifacts/visual/latest/object_preview/` so the dynamic-coloration field can be inspected without
//! a running game.
//!
//! `cargo run -p thalos_body_render --example landcover_preview`
//!
//! Two maps over a ~2 km patch of the surface at lowland altitude:
//! - `landcover_color.png` — the vegetation colour (sRGB) the grass adopts.
//! - `landcover_coverage.png` — the grass coverage mask (white = full, black =
//!   thinned to bare ground on the driest patches).

use bevy::math::DVec3;
use thalos_body_render::sample_landcover;
use thalos_terrain::ProceduralSurface;

const SIZE: u32 = 768;
const SPAN_M: f64 = 2400.0; // metres across the image
const RADIUS_M: f64 = 3_186_000.0; // Thalos reference radius
const ALTITUDE_M: f32 = 1500.0; // lowland lush band (shows grass/dry/forest)
const OUT_DIR: &str = "artifacts/visual/latest/object_preview";

fn main() {
    std::fs::create_dir_all(OUT_DIR).ok();

    // The planet-scale macro moisture comes from the real generator (Thalos =
    // seed 2), composed with the wrapped fine tier exactly as the game does.
    let surface = ProceduralSurface::new(RADIUS_M as f32, 2);

    // A tangent patch on the +X face of the body: vary the two tangent axes (Y,
    // Z) over ±SPAN/2, hold the surface at the lowland altitude.
    let mut color = image::RgbaImage::new(SIZE, SIZE);
    let mut coverage = image::RgbaImage::new(SIZE, SIZE);
    for py in 0..SIZE {
        for px in 0..SIZE {
            let u = (px as f64 / SIZE as f64 - 0.5) * SPAN_M;
            let v = (py as f64 / SIZE as f64 - 0.5) * SPAN_M;
            let pos = DVec3::new(RADIUS_M + ALTITUDE_M as f64, u, v);
            let macro_moisture = {
                use thalos_terrain::query::SurfaceQuery;
                surface.landcover_moisture(pos.normalize())
            };
            let s = sample_landcover(
                pos,
                ALTITUDE_M,
                macro_moisture,
                pos.normalize().y.abs() as f32,
            );

            let c = s.veg_color;
            color.put_pixel(px, py, image::Rgba(rgb8(c.x, c.y, c.z)));
            let g = lin_to_srgb(s.coverage);
            coverage.put_pixel(px, py, image::Rgba([g, g, g, 255]));
        }
    }

    for (name, img) in [
        ("landcover_color", &color),
        ("landcover_coverage", &coverage),
    ] {
        let path = format!("{OUT_DIR}/{name}.png");
        img.save(&path).expect("write PNG");
        println!("wrote {path} ({SIZE}×{SIZE}, {SPAN_M} m span)");
    }
}

fn rgb8(r: f32, g: f32, b: f32) -> [u8; 4] {
    [lin_to_srgb(r), lin_to_srgb(g), lin_to_srgb(b), 255]
}

/// Linear → sRGB → 8-bit (the previews are viewed as ordinary images).
fn lin_to_srgb(x: f32) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let s = if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round().clamp(0.0, 255.0) as u8
}
