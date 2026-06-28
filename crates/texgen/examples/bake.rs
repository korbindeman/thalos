//! Bake procedural textures to PNG — inspect them, or prebake for the game.
//!
//! `cargo run -p thalos_texgen --example bake` → `tools/texgen/out/*.png`.

use thalos_texgen::{TextureData, foliage_atlas, grass_blades};

fn main() {
    let out_dir = "tools/texgen/out";
    std::fs::create_dir_all(out_dir).expect("create out dir");

    for (name, tex) in [
        ("foliage_atlas", foliage_atlas()),
        ("grass_blades", grass_blades()),
    ] {
        write_png(out_dir, name, tex);
    }
}

fn write_png(out_dir: &str, name: &str, tex: TextureData) {
    let path = format!("{out_dir}/{name}.png");
    let img = image::RgbaImage::from_raw(tex.width, tex.height, tex.rgba)
        .expect("texture buffer matches dimensions");
    img.save(&path).expect("write PNG");
    println!("wrote {path} ({}×{})", tex.width, tex.height);
}
