use std::{fs, path::Path};

use serde::Serialize;
use thalos_texgen::terrain::terrain_material_set;
use thalos_texgen::{TextureData, foliage_atlas, foliage_material_atlas, grass_card_atlas};

#[derive(Serialize)]
struct Manifest {
    schema_version: u32,
    generator_version: u32,
    assets: Vec<AssetRecord>,
}

#[derive(Serialize)]
struct AssetRecord {
    name: String,
    file: String,
    width: u32,
    height: u32,
    fnv1a64: String,
}

fn main() {
    bake(
        "assets/generated/vegetation",
        vec![
            ("foliage_atlas", foliage_atlas()),
            ("foliage_material_atlas", foliage_material_atlas()),
            ("grass_card_atlas", grass_card_atlas()),
        ],
    );

    // Terrain material set: both halves come out of one synthesis pass, since
    // the normal/roughness/AO derive from the same height field the albedo's
    // cavities do (see `thalos_texgen::terrain`).
    let set = terrain_material_set();
    bake(
        "assets/generated/terrain",
        vec![
            ("terrain_albedo_array", set.albedo),
            ("terrain_material_array", set.material),
        ],
    );
}

/// Write every texture into `dir` and stamp a manifest beside them. The
/// manifest's `generator_version` is what the runtime checks a baked asset
/// against, so it is written last — a half-written directory has no manifest
/// and reads as stale rather than as current.
fn bake(dir: &str, textures: Vec<(&str, TextureData)>) {
    let output = Path::new(dir);
    fs::create_dir_all(output).expect("create generated asset directory");
    let assets = textures
        .into_iter()
        .map(|(name, texture)| write_png(output, name, texture))
        .collect();
    let manifest = Manifest {
        schema_version: 1,
        generator_version: thalos_texgen::GENERATOR_VERSION,
        assets,
    };
    let bytes = serde_json::to_vec_pretty(&manifest).expect("serialize manifest");
    fs::write(output.join("manifest.json"), bytes).expect("write manifest");
}

fn write_png(output: &Path, name: &str, texture: TextureData) -> AssetRecord {
    let filename = format!("{name}.png");
    let path = output.join(&filename);
    let hash = fnv1a64(&texture.rgba);
    let image = image::RgbaImage::from_raw(texture.width, texture.height, texture.rgba)
        .expect("texture dimensions match its buffer");
    image.save(&path).expect("write generated atlas");
    println!("wrote {}", path.display());
    AssetRecord {
        name: name.to_owned(),
        file: filename,
        width: texture.width,
        height: texture.height,
        fnv1a64: format!("{hash:016x}"),
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}
