//! Throwaway timing harness for the Thalos `OceanicTerrestrial` bake.
//!
//! Mirrors `compile_oceanic_terrestrial` (the editor's per-edit preview path)
//! so we can see where the time actually goes: field eval vs albedo paint vs
//! finalize. Run with `RAYON_NUM_THREADS=1` to see the serial baseline.
//!
//!   cargo run --release -p thalos_terrain --example oceanic_bake_timing
//!   RAYON_NUM_THREADS=1 cargo run --release -p thalos_terrain --example oceanic_bake_timing
//!
//! Not a test (no per-body generation tests, per CLAUDE.md); delete freely.

use std::time::Instant;

use thalos_terrain::{
    BodyBuilder, Composition, FieldSurface, OceanicContinentalField, OceanicContinentalParams,
    RuntimeTerrainDetail, SurfaceColorSpec, bake_surface_field_into_builder, paint_surface_albedo,
};

fn main() {
    // Thalos physical params (assets/bodies/thalos.ron + oceanic_terrestrial prior).
    let radius_m = 3_186_000.0_f32;
    let seed = 1003_u64;
    let age_gyr = 5.5_f32;
    let relief_scale_m = 4_800.0_f32;
    let ocean_fraction = 0.62_f32;

    println!("threads = {}", rayon::current_num_threads());

    // FieldSurface cross-edit reuse: first bake builds the continent intent
    // cache; subsequent bakes after a *non-shape* edit reuse it and skip the
    // kernel (~80% of the field cost). Mimics the editor holding one
    // FieldSurface across slider drags.
    {
        let params = OceanicContinentalParams::from_seed_parts(
            seed,
            seed ^ 0x9999,
            relief_scale_m,
            ocean_fraction,
        );
        let mut field = FieldSurface::oceanic(
            radius_m,
            params,
            Composition::new(0.68, 0.30, 0.0, 0.02, 0.0),
            age_gyr,
            0.0,
            seed,
        );
        let res = 512;
        let t0 = Instant::now();
        let _ = field.bake(Some(res));
        let first = t0.elapsed();
        let t1 = Instant::now();
        let _ = field.bake(Some(res)); // non-shape "edit": cache reused
        let reuse = t1.elapsed();
        // shape edit: invalidates the cache → rebuild
        let mut shifted = params;
        shifted.seed_macro ^= 0xABCD;
        field.set_params(shifted);
        let t2 = Instant::now();
        let _ = field.bake(Some(res));
        let shape_edit = t2.elapsed();
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        println!(
            "FieldSurface {res}²  first {:>8.1}ms  reuse {:>8.1}ms  shape-edit {:>8.1}ms",
            ms(first),
            ms(reuse),
            ms(shape_edit),
        );
    }

    for &res in &[256u32, 512, 1024] {
        let t0 = Instant::now();
        let mut builder = BodyBuilder::new(
            radius_m,
            seed,
            Composition::new(0.68, 0.30, 0.0, 0.02, 0.0),
            Some(res),
            age_gyr,
            None,
            0.0,
        );
        let params = OceanicContinentalParams::from_seed_parts(
            seed,
            seed ^ 0x9999,
            relief_scale_m,
            ocean_fraction,
        );
        let field = OceanicContinentalField::new(params, radius_m);
        let t_setup = t0.elapsed();

        let t1 = Instant::now();
        bake_surface_field_into_builder(&mut builder, &field);
        builder.runtime_detail = RuntimeTerrainDetail::OceanicContinental(params);
        builder.sea_level_m = Some(0.0);
        let t_field = t1.elapsed();

        let t2 = Instant::now();
        paint_surface_albedo(
            &mut builder,
            &SurfaceColorSpec::aging_oceanic_homeworld(seed, 0.0),
        );
        let t_paint = t2.elapsed();

        let t3 = Instant::now();
        let _ = builder.build();
        let t_build = t3.elapsed();

        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        println!(
            "res {res:>4}²  total {:>8.1}ms | setup {:>6.1}  field {:>8.1}  paint {:>8.1}  build {:>6.1}",
            ms(t0.elapsed()),
            ms(t_setup),
            ms(t_field),
            ms(t_paint),
            ms(t_build),
        );
    }
}
