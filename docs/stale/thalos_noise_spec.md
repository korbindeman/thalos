# `thalos_noise` — SIMD SuperSimplex Crate Spec

## Goal

Replace `fastnoise2` (C++ FFI) with a pure-Rust, SIMD-accelerated crate that
implements only the noise operations Thalos' terrain pipeline actually uses.
Not a general FastNoise2 clone. Scope is deliberately tiny.

Success = `thalos_terrain_gen` compiles with `fastnoise2` removed, bake output
is deterministic and visually equivalent, and bake wall time does not regress
beyond the perf budget below.

## Non-goals

- Matching FastNoise2 bit-for-bit. Output will differ; we rebake goldens once.
- Other noise types (perlin, value, cellular, white). Only SuperSimplex 3D.
- A node-graph builder mirroring FastNoise2's full DSL. Only the handful of
  combinators listed under **API**.
- 2D or 4D noise. 3D only.
- `no_std`, WASM, or non-x86_64 SIMD. Target is native x86_64 with AVX2.
  ARM/Apple Silicon fallback is scalar and acceptable (dev machine is Apple
  Silicon — scalar must still beat the perf budget there, see below).
- `std::simd` (nightly). Use the `wide` crate (stable) for portable SIMD.

## Required API surface

The terrain crate uses exactly this subset of `fastnoise2`:

```rust
supersimplex()
    .fbm(gain, weighted_strength, octaves, lacunarity)   // weighted_strength always 0.0
    .domain_scale(frequency)
    .domain_warp_gradient(amplitude, frequency)          // optional, only in mare_flood
    // scalar mul: `node * 0.7`
    // node add:   `node_a + node_b`
    .build()

SafeNode::gen_position_array_3d(
    out: &mut [f32],
    xs: &[f32], ys: &[f32], zs: &[f32],
    ox: f32, oy: f32, oz: f32,    // always 0.0 at call sites
    seed: i32,
)
```

Parameters actually seen in the codebase (grep before changing these —
`crates/terrain_gen/src/stages/mare_flood.rs` and `regolith.rs`):

- `gain ∈ {0.5, 0.55}`
- `octaves ∈ {3, 4, 5}`
- `lacunarity = 2.0` always
- `weighted_strength = 0.0` always (so fbm stays pure textbook: `sum +=
  amp * noise(p); p *= lacunarity; amp *= gain;`)
- `domain_warp_gradient` amplitude `∈ {0.02, 0.03}`, used only in `mare_flood`
- Scalar multiply and node addition are used in one place:
  `(a.fbm(...).domain_scale(lo) * 0.7) + (b.fbm(...).domain_scale(hi) * 0.3)`
- Offsets (`ox/oy/oz`) are always `0.0` at call sites. Support them (cheap) but
  don't optimize around them.

The crate goes in `crates/noise/` as a new workspace member `thalos_noise`.

### Proposed Rust API

Mirror FastNoise2's shape closely enough that the call sites in `mare_flood.rs`
and `regolith.rs` change in trivial ways (rename import, adjust one or two
method names if needed). Concretely:

```rust
pub struct Node { /* boxed op tree */ }

pub fn supersimplex() -> Node;

impl Node {
    pub fn fbm(self, gain: f32, weighted_strength: f32, octaves: u32, lacunarity: f32) -> Node;
    pub fn domain_scale(self, frequency: f32) -> Node;
    pub fn domain_warp_gradient(self, amplitude: f32, frequency: f32) -> Node;
    pub fn build(self) -> Node; // no-op, kept for call-site parity; or remove if refactor is cheap
}

impl core::ops::Add for Node { type Output = Node; /* ... */ }
impl core::ops::Mul<f32> for Node { type Output = Node; /* ... */ }

impl Node {
    pub fn gen_position_array_3d(
        &self,
        out: &mut [f32],
        xs: &[f32], ys: &[f32], zs: &[f32],
        ox: f32, oy: f32, oz: f32,
        seed: i32,
    );
}
```

`weighted_strength` is accepted but **must panic if nonzero** until someone
needs it. Don't silently ignore it.

## Correctness

1. **Reference scalar implementation first.** Write `supersimplex_3d_scalar`
   from a known public-domain reference (KdotJPG's OpenSimplex2 / SuperSimplex
   3D reference). Seed via a permutation table derived from the i32 seed with a
   stable hash (splitmix64 → fill permutation). Document the exact hash.
2. **Output range.** Single-octave SuperSimplex 3D output must land in roughly
   `[-1, 1]`. Empirically measure over 1M random inputs, record min/max, assert
   `|x| < 1.01`. FBM output is `sum(amp * noise) / max_possible_sum`, so also
   `[-1, 1]`. Document the normalization constant.
3. **Golden test.** Check in a tiny golden: for seed `12345`, a fixed 32×32×32
   grid of positions, dump single-octave values to a `.bin` file in
   `crates/noise/tests/golden/`. Test recomputes and compares with tolerance
   `1e-6`. This catches accidental regressions in the scalar reference.
4. **SIMD-vs-scalar parity.** Property test: for 10k random positions and
   seeds, `scalar_sample(p, s) == simd_sample_one(p, s)` within `1e-5`. Must
   hold for every op (base, fbm, domain_scale, domain_warp_gradient, add, mul).
5. **Determinism.** Same inputs → same outputs across runs and across thread
   counts. No global state, no RNG that depends on call order. Verify by
   running the terrain bake twice and diffing `BodyData` height cubemaps
   bit-for-bit.
6. **Visual parity with old fastnoise2 is NOT required.** Rebake goldens. But
   the shapes must still look like "mare blobs and regolith weight fields" —
   not stripes, not grid artifacts, not DC offset. Include a manual check step
   in the acceptance protocol.

## SIMD implementation notes

- Use `wide::f32x8` and `wide::i32x8`. 8-wide is the sweet spot on AVX2.
- Process `gen_position_array_3d` in lanes of 8. Scalar tail for remainder
  (length is `res*res`, typically 1024² or 2048² — tail is ≤7).
- SuperSimplex 3D has 4 simplex corners per sample, each needing a gradient
  lookup from the permutation table. The gather is the bottleneck. Two options:
  - **Manual scalar gather:** extract 8 indices, do 8 scalar table lookups,
    repack into `f32x8`. Simple, fast enough — perm table is 512 bytes, fits
    L1, gather latency dominated by L1 hit.
  - **`simd_gather`:** only available on nightly `std::simd`. Skip for now;
    the manual gather is within a few percent.
- FBM loop is trivially vectorizable: octave loop is outer, lane-wise sum
  accumulator is inner.
- `domain_warp_gradient` samples the base node 3 extra times per output to
  form the offset vector `(dx, dy, dz) = amp * (n(p+o1), n(p+o2), n(p+o3))`
  then returns `base.sample(p + (dx,dy,dz))`. Match FastNoise2's convention
  (offsets along axes or hardcoded seed offsets — confirm from FastNoise2
  source in `FastNoise2/include/FastNoise/Generators/DomainWarp.inl` before
  implementing). Exact match with FastNoise2 not required, but the warp must
  actually warp (domain shift visible in outputs).
- `domain_scale` is input multiply only — `p *= frequency` before sampling.
  Make sure this composes correctly with warp (warp applies *after* scale in
  the current call sites: `.fbm().domain_scale().domain_warp_gradient()` —
  check the node tree order matches FastNoise2's. If it doesn't, the call
  sites need updating, not the library).

## Performance budget

Benchmark on an Apple M-series dev machine (scalar fallback) AND, if
available, an AVX2 x86_64 machine. Report both.

Targets for `mare_flood.rs`-style workload (FBM of SuperSimplex, 5 octaves,
with and without domain warp) at `2048×2048` positions per call:

| Platform           | vs fastnoise2 | Hard fail threshold |
|--------------------|---------------|---------------------|
| x86_64 AVX2        | ≤ 1.5× slower | 2.5× slower         |
| Apple Silicon      | ≤ 3× slower   | 5× slower           |

End-to-end bake budget: **`just build && cargo run -p thalos_planet_editor`
completes a full Mira bake in under 40 s on the dev machine** (baseline
~25 s with fastnoise2). If bake exceeds 40 s, stop and report.

Write benches under `crates/noise/benches/` using `criterion`. At minimum:
- `single_octave_bulk_1024x1024`
- `fbm5_bulk_1024x1024`
- `fbm5_warped_bulk_1024x1024`

Bench must also run the same workload via `fastnoise2` (kept as a
`dev-dependency`) and print the ratio. Commit baseline numbers to
`docs/thalos_noise_bench.md`.

## Deliverables

1. New crate `crates/noise/` added to `[workspace.members]` in root
   `Cargo.toml`. Name: `thalos_noise`.
2. Source layout:
   ```
   crates/noise/
     Cargo.toml
     src/
       lib.rs              # public API, Node type
       scalar.rs           # reference scalar supersimplex_3d
       simd.rs             # f32x8 bulk path
       fbm.rs              # fbm wrapper (scalar + SIMD)
       warp.rs             # domain_warp_gradient
       perm.rs             # seed → permutation table
     tests/
       golden.rs
       parity.rs           # scalar vs SIMD
       range.rs            # empirical output range check
       golden/
         supersimplex_seed12345_32cubed.bin
     benches/
       bulk.rs             # criterion benches incl. fastnoise2 comparison
   ```
3. `fastnoise2` removed from `crates/terrain_gen/Cargo.toml`. Replaced by
   `thalos_noise`. Call sites in `mare_flood.rs` and `regolith.rs` updated.
4. `docs/thalos_noise_bench.md` with bench table and short perf commentary.
5. Updated `CLAUDE.md` entry under the terrain_gen crate section mentioning
   `thalos_noise` replaces fastnoise2.

## Acceptance protocol

Run in order. Stop and report if any step fails.

1. `cargo test -p thalos_noise` — all tests pass including golden + parity.
2. `cargo bench -p thalos_noise` — ratios printed, all within budget table
   above. If over budget, iterate on SIMD, do not proceed.
3. `cargo build --workspace` — whole repo compiles with fastnoise2 removed.
4. `cargo clippy --workspace -- -D warnings` — clean.
5. `just test` — existing test suites still pass.
6. `cargo run -p thalos_planet_editor` — Mira bake completes, wall time
   measured and under 40 s. Save a screenshot of the baked body for manual
   visual check. Both `mare_flood` and `regolith` visibly affect the body.
7. Run the bake twice, assert height cubemap bytes are identical across runs.
8. `cargo run --release -p thalos_game` — game launches, no panics, no
   regressions in rendering (quick manual smoke).
9. Call advisor with the full test + bench output before declaring done.

## Hard rules for the implementing agent

- **Do not** try to match FastNoise2's exact output. That's a rabbit hole.
- **Do not** add noise types beyond SuperSimplex 3D. No feature creep.
- **Do not** delete the bench comparison against fastnoise2 even after
  removing it from terrain_gen. Keep it as a dev-dependency of the bench
  crate only, gated so normal builds stay clean.
- **Do not** commit if benches exceed the hard-fail threshold. Report and
  stop.
- **Do not** leave `TODO`/`FIXME` in committed code. If something is
  incomplete the whole effort is not done.
- **Do** write the scalar reference first, get it correct, then SIMD.
- **Do** measure output range empirically and document the normalization
  constant — don't copy a magic number from another impl without verifying.
- **Do** use `wide` crate (stable) not `std::simd` (nightly).
- **If blocked** for >1 hour on SIMD correctness, fall back to `wide` scalar
  emulation and measure. If that alone hits the budget on Apple Silicon,
  that's acceptable for the first landing — x86_64 SIMD can be a follow-up.
- **If final numbers blow the budget** after honest effort, revert the
  terrain_gen switch, keep `thalos_noise` as an unused crate behind a feature
  flag, and write up findings in `docs/thalos_noise_bench.md`. Do not merge a
  perf regression.
