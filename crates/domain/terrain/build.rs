use std::env;
use std::fs;
use std::path::{Path, PathBuf};

mod build_support;

use build_support::{FNV_OFFSET, hash_byte, hash_bytes, hash_rust_source};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_support.rs");
    println!("cargo:rerun-if-changed=src");

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let src_dir = manifest_dir.join("src");
    let mut files = Vec::new();
    collect_rust_files(&src_dir, &mut files);
    files.sort();

    let mut hash = FNV_OFFSET;
    for file in files {
        println!("cargo:rerun-if-changed={}", file.display());
        let relative = file
            .strip_prefix(&manifest_dir)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        hash_bytes(&mut hash, relative.as_bytes());
        hash_byte(&mut hash, 0);

        let bytes =
            fs::read(&file).unwrap_or_else(|e| panic!("failed to read {}: {e}", file.display()));
        // Git may materialize text as CRLF on Windows. Source identity is the
        // repository text, not the checkout's line-ending policy; hashing raw
        // bytes made one Mira package look stale only on Windows.
        hash_rust_source(&mut hash, &bytes);
        hash_byte(&mut hash, 0xff);
    }

    println!("cargo:rustc-env=THALOS_TERRAIN_SOURCE_HASH={hash:016x}");
}

fn collect_rust_files(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in
        fs::read_dir(dir).unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()))
    {
        let entry = entry.unwrap_or_else(|e| panic!("failed to read directory entry: {e}"));
        let path = entry.path();
        if path.is_dir() {
            collect_rust_files(&path, files);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            files.push(path);
        }
    }
}
