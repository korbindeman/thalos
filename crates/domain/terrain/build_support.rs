//! Pure helpers shared by the terrain build script and its regression test.

pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

pub fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        hash_byte(hash, byte);
    }
}

/// Hash Rust source with repository-canonical LF line endings.
pub fn hash_rust_source(hash: &mut u64, bytes: &[u8]) {
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\r' && bytes.get(index + 1) == Some(&b'\n') {
            hash_byte(hash, b'\n');
            index += 2;
        } else {
            hash_byte(hash, bytes[index]);
            index += 1;
        }
    }
}

pub fn hash_byte(hash: &mut u64, byte: u8) {
    *hash ^= u64::from(byte);
    *hash = hash.wrapping_mul(FNV_PRIME);
}
