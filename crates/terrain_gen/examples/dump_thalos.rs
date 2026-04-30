//! Diagnostic: bake Thalos at low res using the EXACT generator block from
//! `assets/solar_system.ron` and dump every interesting cubemap as a 4×3
//! cubemap-cross PNM file. Output goes to `/tmp/thalos_*.{pgm,ppm}`.
//!
//! The generator block is loaded straight from the asset RON via the
//! shared `GeneratorParams` deserializer, so the colours that come out of
//! the `final_albedo` dump are exactly what the planet impostor renders
//! in the editor under `Full bright`. Keeps the diagnostic in lock-step
//! with the in-game pipeline without copy-pasting biome lists.

use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};

use thalos_terrain_gen::{BodyBuilder, CubemapFace, GeneratorParams};

const RES: u32 = 2048; // match the in-game bake exactly

fn main() {
    // ── Locate Thalos's generator block in solar_system.ron ────────────
    //
    // Hand-rolled extraction because pulling in the physics crate (which
    // owns `BodyDefinition`'s deserializer) would create a workspace
    // cycle (physics → terrain_gen). Walking the RON as a generic value
    // is enough for one body and avoids defining stub structs for every
    // BodyDefinition field.
    let path = std::env::var("CARGO_MANIFEST_DIR").unwrap() + "/../../assets/solar_system.ron";
    let raw = fs::read_to_string(&path).expect("read solar_system.ron");
    let gen_str = extract_generator_block(&raw, "Thalos").expect("find Thalos generator block");
    // Re-prepend the `#![enable(implicit_some)]` directive that lives at
    // the top of solar_system.ron — it's a parser feature flag, not part
    // of the document's syntax tree, so extracting a sub-block strips it
    // and `Some(...)`-implicit fields like `fresh_albedo` then fail
    // to parse.
    let with_directive = format!("#![enable(implicit_some)]\n{}", gen_str);
    let mut params: GeneratorParams =
        ron::from_str(&with_directive).expect("parse Thalos generator block");

    // ── Build the body and run the pipeline ────────────────────────────
    //
    // Override resolution to RES so the diagnostic bakes in seconds.
    params.cubemap_resolution = RES;

    let radius_m = 3_186_000.0_f32;
    let axial_tilt_rad = 23.0_f32.to_radians();
    let mut builder = BodyBuilder::new(
        radius_m,
        params.seed,
        params.composition,
        params.cubemap_resolution,
        params.body_age_gyr,
        None,
        axial_tilt_rad,
    );

    let pipeline = thalos_terrain_gen::Pipeline::new(
        params
            .pipeline
            .into_iter()
            .map(|s| s.into_stage())
            .collect(),
    );
    let t = std::time::Instant::now();
    pipeline.run(&mut builder);
    eprintln!(
        "baked Thalos at {RES}² in {:.2}s",
        t.elapsed().as_secs_f32()
    );

    // Snapshot intermediates BEFORE build() consumes them.
    let plate_map = builder.plates.clone().expect("plates");
    let orogen = builder.orogen_intensity.clone();
    let boundary_dist = builder.boundary_distance_km.clone();
    let height_field = builder.height_contributions.height.clone();

    let (mut min_h, mut max_h) = (f32::INFINITY, f32::NEG_INFINITY);
    for face in CubemapFace::ALL {
        for &v in height_field.face_data(face) {
            min_h = min_h.min(v);
            max_h = max_h.max(v);
        }
    }
    let abs_max = min_h.abs().max(max_h.abs()).max(1.0);
    eprintln!("height range: {:.0} m to {:.0} m (sea = 0)", min_h, max_h);

    write_gray_cross("/tmp/thalos_height.pgm", RES, |face, x, y| {
        let v = height_field.get(face, x, y);
        let norm = (v / abs_max * 0.5 + 0.5).clamp(0.0, 1.0);
        (norm * 255.0) as u8
    });

    write_gray_cross("/tmp/thalos_landmask.pgm", RES, |face, x, y| {
        if height_field.get(face, x, y) > 0.0 {
            255
        } else {
            0
        }
    });

    write_color_cross("/tmp/thalos_plates.ppm", RES, |face, x, y| {
        let id = plate_map.plate_id_cubemap.get(face, x, y) as u32;
        let h = id.wrapping_mul(0x9E3779B1);
        [
            ((h >> 0) & 0xFF) as u8,
            ((h >> 8) & 0xFF) as u8,
            ((h >> 16) & 0xFF) as u8,
        ]
    });

    write_color_cross("/tmp/thalos_plate_kind.ppm", RES, |face, x, y| {
        let id = plate_map.plate_id_cubemap.get(face, x, y) as usize;
        let p = &plate_map.plates[id];
        match p.kind {
            thalos_terrain_gen::PlateKind::Continental => [120, 180, 80],
            thalos_terrain_gen::PlateKind::Oceanic => [40, 60, 140],
        }
    });

    write_gray_cross("/tmp/thalos_orogen.pgm", RES, |face, x, y| {
        (orogen.get(face, x, y).clamp(0.0, 1.0) * 255.0) as u8
    });

    write_gray_cross("/tmp/thalos_boundary_dist.pgm", RES, |face, x, y| {
        let d_km = boundary_dist.get(face, x, y);
        if !d_km.is_finite() {
            return 255;
        }
        let v = (d_km.max(1.0).log2() / 12.0).clamp(0.0, 1.0);
        (v * 255.0) as u8
    });

    let body = builder.build();

    write_color_cross("/tmp/thalos_materials.ppm", RES, |face, x, y| {
        let id = body.material_cubemap.get(face, x, y) as u32;
        let h = id.wrapping_mul(0x9E3779B1).wrapping_add(0x12345);
        [
            ((h >> 0) & 0xFF) as u8,
            ((h >> 8) & 0xFF) as u8,
            ((h >> 16) & 0xFF) as u8,
        ]
    });

    // Final shader-view colour. Reproduces the planet impostor's tint
    // formula CPU-side under `Full bright` (sun_flux = 0, ambient = 1):
    //     final = mat_albedo × baked_tint × 2 × regional
    // We fold `regional = 1.0` here because the shader's
    // `regional_albedo_mod` adds only ±18 % low-frequency noise that
    // doesn't affect biome boundary visibility — the diagnostic point is
    // colour identity, not ±2 stops of variation.
    let mats = &body.materials;
    write_color_cross("/tmp/thalos_final_albedo.ppm", RES, |face, x, y| {
        let mat_id = (body.material_cubemap.get(face, x, y) as usize).min(mats.len() - 1);
        let mat = &mats[mat_id];
        let tint = body.albedo_cubemap.get(face, x, y);
        let dec = |v: u8| {
            let s = v as f32 / 255.0;
            if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        };
        let baked = [dec(tint[0]), dec(tint[1]), dec(tint[2])];
        let r = (mat.albedo[0] * baked[0] * 2.0).clamp(0.0, 1.0);
        let g = (mat.albedo[1] * baked[1] * 2.0).clamp(0.0, 1.0);
        let b = (mat.albedo[2] * baked[2] * 2.0).clamp(0.0, 1.0);
        let enc = |v: f32| {
            let s = if v <= 0.003_130_8 {
                v * 12.92
            } else {
                1.055 * v.powf(1.0 / 2.4) - 0.055
            };
            (s * 255.0).clamp(0.0, 255.0) as u8
        };
        [enc(r), enc(g), enc(b)]
    });

    eprintln!("dumped:");
    eprintln!("  /tmp/thalos_height.pgm        (signed height, 127 = sea level)");
    eprintln!("  /tmp/thalos_landmask.pgm      (white = land, black = ocean)");
    eprintln!("  /tmp/thalos_plates.ppm        (per-plate hash color)");
    eprintln!("  /tmp/thalos_plate_kind.ppm    (green=continental, blue=oceanic)");
    eprintln!("  /tmp/thalos_orogen.pgm        (orogen intensity)");
    eprintln!("  /tmp/thalos_boundary_dist.pgm (log distance to nearest boundary)");
    eprintln!("  /tmp/thalos_materials.ppm     (per-texel material id)");
    eprintln!("  /tmp/thalos_final_albedo.ppm  (shader-view final colour)");
}

/// Locate `name`'s body in solar_system.ron and return the value of its
/// `generator: Some(...)` field as a parseable RON snippet (the inner `(...)`).
///
/// Brittle (depends on the file being formatted with one body-record per
/// `name: "..."` line and `generator: Some((...))` on its own line) but
/// accurate enough for one body and avoids defining stub structs for the
/// whole BodyDefinition deserializer chain. Fails loudly if the format
/// drifts.
fn extract_generator_block(raw: &str, name: &str) -> Option<String> {
    let needle = format!("name: \"{name}\"");
    let body_start = raw.find(&needle)?;
    let after = &raw[body_start..];
    let gen_marker = "generator: Some(";
    let gen_offset = after.find(gen_marker)?;
    let inner_start = body_start + gen_offset + gen_marker.len();
    let bytes = raw.as_bytes();
    let mut depth: i32 = 1;
    let mut i = inner_start;
    while i < bytes.len() && depth > 0 {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => depth -= 1,
            _ => {}
        }
        i += 1;
    }
    if depth != 0 {
        return None;
    }
    // The Some(...) wrapper's outer `)` is at i-1; the GeneratorParams
    // struct literal sits between inner_start and i-1.
    Some(raw[inner_start..i - 1].to_string())
}

const FACE_LAYOUT: [(CubemapFace, usize, usize); 6] = [
    (CubemapFace::PosY, 1, 0),
    (CubemapFace::NegX, 0, 1),
    (CubemapFace::PosZ, 1, 1),
    (CubemapFace::PosX, 2, 1),
    (CubemapFace::NegZ, 3, 1),
    (CubemapFace::NegY, 1, 2),
];

fn write_gray_cross<F: Fn(CubemapFace, u32, u32) -> u8>(path: &str, res: u32, sample: F) {
    let cross_w = (4 * res) as usize;
    let cross_h = (3 * res) as usize;
    let mut pixels = vec![0u8; cross_w * cross_h];
    for (face, fx, fy) in FACE_LAYOUT {
        for y in 0..res {
            for x in 0..res {
                let px = fx * res as usize + x as usize;
                let py = fy * res as usize + y as usize;
                pixels[py * cross_w + px] = sample(face, x, y);
            }
        }
    }
    let mut w = BufWriter::new(File::create(path).expect("create file"));
    write!(w, "P5\n{} {}\n255\n", cross_w, cross_h).unwrap();
    w.write_all(&pixels).unwrap();
}

fn write_color_cross<F: Fn(CubemapFace, u32, u32) -> [u8; 3]>(path: &str, res: u32, sample: F) {
    let cross_w = (4 * res) as usize;
    let cross_h = (3 * res) as usize;
    let mut pixels = vec![0u8; cross_w * cross_h * 3];
    for (face, fx, fy) in FACE_LAYOUT {
        for y in 0..res {
            for x in 0..res {
                let px = fx * res as usize + x as usize;
                let py = fy * res as usize + y as usize;
                let rgb = sample(face, x, y);
                let i = (py * cross_w + px) * 3;
                pixels[i] = rgb[0];
                pixels[i + 1] = rgb[1];
                pixels[i + 2] = rgb[2];
            }
        }
    }
    let mut w = BufWriter::new(File::create(path).expect("create file"));
    write!(w, "P6\n{} {}\n255\n", cross_w, cross_h).unwrap();
    w.write_all(&pixels).unwrap();
}
