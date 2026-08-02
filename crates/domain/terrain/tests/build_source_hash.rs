#[path = "../build_support.rs"]
mod build_support;

use build_support::{FNV_OFFSET, hash_bytes, hash_rust_source};

fn source_hash(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    hash_rust_source(&mut hash, bytes);
    hash
}

fn raw_hash(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    hash_bytes(&mut hash, bytes);
    hash
}

#[test]
fn rust_source_hash_is_independent_of_checkout_line_endings() {
    let lf = b"fn first() {}\nfn second() {}\n";
    let crlf = b"fn first() {}\r\nfn second() {}\r\n";

    assert_ne!(raw_hash(lf), raw_hash(crlf));
    assert_eq!(source_hash(lf), source_hash(crlf));
}

#[test]
fn rust_source_hash_still_changes_with_source_content() {
    assert_ne!(
        source_hash(b"const VALUE: u8 = 1;\n"),
        source_hash(b"const VALUE: u8 = 2;\n")
    );
}
