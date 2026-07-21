use std::{fs, path::Path};

use serde::Serialize;
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
    let output = Path::new("assets/generated/vegetation");
    fs::create_dir_all(output).expect("create generated vegetation directory");

    let mut assets = Vec::new();
    for (name, texture) in [
        ("foliage_atlas", foliage_atlas()),
        ("foliage_material_atlas", foliage_material_atlas()),
        ("grass_card_atlas", grass_card_atlas()),
    ] {
        assets.push(write_png(output, name, texture));
    }

    let manifest = Manifest {
        schema_version: 1,
        generator_version: thalos_texgen::GENERATOR_VERSION,
        assets,
    };
    let bytes = serde_json::to_vec_pretty(&manifest).expect("serialize manifest");
    fs::write(output.join("manifest.json"), bytes).expect("write vegetation manifest");
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
