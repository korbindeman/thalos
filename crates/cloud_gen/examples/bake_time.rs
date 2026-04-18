//! Timed end-to-end bake at Wedekind defaults. Run with:
//!   cargo run --release -p thalos_cloud_gen --example bake_time
//!
//! (Not a proper criterion benchmark — just a smoke-test timer to
//! validate the CPU bake fits in the per-planet load budget.)

use std::time::Instant;
use thalos_cloud_gen::{CloudBakeConfig, bake_cloud_cover};

fn main() {
    for &size in &[128u32, 256, 512] {
        let cfg = CloudBakeConfig::wedekind_defaults(0xDEAD_BEEF, size);
        let start = Instant::now();
        let cube = bake_cloud_cover(&cfg);
        let dt = start.elapsed();
        // Touch the output so the optimiser doesn't dead-code-eliminate
        // the whole bake.
        let mut mn = f32::MAX;
        let mut mx = f32::MIN;
        for face in thalos_terrain_gen::cubemap::CubemapFace::ALL {
            for &v in cube.face_data(face) {
                mn = mn.min(v);
                mx = mx.max(v);
            }
        }
        println!(
            "size={:>3}²  faces=6  iters=50  elapsed={:?}  density=[{:.3}, {:.3}]",
            size, dt, mn, mx,
        );
    }
}
