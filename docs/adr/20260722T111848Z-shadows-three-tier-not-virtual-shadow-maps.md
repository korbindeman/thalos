# ADR-20260722T111848Z-shadows-three-tier-not-virtual-shadow-maps: Shadows are three range regimes with three mechanisms; virtual shadow maps are rejected

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

The one-shadow-world rig (F6/W5/W6, landed 2026-07-02) has been through nine
rounds of tuning (`shadow_unification_prompt.md`). Reading the rig as it stands:

- `sun_shadow.rs` — three cascades, 4096², half-extents **400 / 1500 / 4000 m**,
  fars 1500 / 5000 / 12000 m, with altitude footprint scaling (power-of-two
  quantized, hysteresis, capped at 32×), body-fixed texel snap, 0.1° sun
  quantization, and a `craft_local` mode above 50 km.
- `thalos::shadow` — a bias/offset apparatus of six hand-tuned constants
  (`BIAS_TEXELS`, `BIAS_MIN_M`/`BIAS_MAX_M`, `NO_NORMAL_BIAS_SCALE`,
  `NORMAL_OFFSET_TEXELS`/`NORMAL_OFFSET_MAX_M`, `MAX_SLOPE_SCALE`), every one of
  them **hard-capped**, with a module comment explaining that far-cascade texels
  are metres wide and an uncapped texel-proportional bias would exceed a whole
  tree and erase its shadow.

That comment is the tell. The tuning treadmill is not a tuning problem — it is
**one mechanism being asked to serve three range regimes it cannot simultaneously
serve**:

| Regime | Range | Cascade capability |
|---|---|---|
| **Contact** | 0–50 m | Cascade 0 is ~0.2 m/texel — coarser than a landing-gear strut. Cannot ground objects. |
| **Mid-field** | 50 m – 5 km | What cascades are for. Adequate today. |
| **Far-field** | 5 km – horizon | A 100 km ortho box at 4096² is 24 m/texel. Structurally impossible. |

Virtual shadow maps (UE5) were raised as the planetary-scale answer and are the
alternative this ADR exists to reject on the record, since the obvious reading of
"nine rounds of cascade tuning" is "replace the cascades".

## Decision

**Stop extending the cascades' range. Serve each regime with its own mechanism,
and keep cascades scoped to the middle one.**

- **Contact (0–50 m) → screen-space contact shadows** (W18a). A short march in
  `SceneDepthImage`, ~8–16 steps, sub-metre reach. `SceneDepthImage` and the
  custom-`Core3d`-node pattern both already exist (built for F5's SSAO), so this
  reuses infrastructure rather than adding a rig.
- **Mid-field (50 m – 5 km) → the existing cascades**, with two quality items:
  PCSS contact-hardening on cascades 0/1 (W18b) and a cross-cascade blend (only
  the outermost cascade fades today, so 0→1 and 1→2 hand off hard — a seam that
  gets *more* visible once penumbra width varies).
- **Far-field (5 km – horizon) → the horizon-angle term** (W12/W12r), a
  per-fragment horizon-elevation lookup against the terrain height field. The
  terrain side already ships; the object side moves from today's per-object CPU
  f64 march to a per-fragment term off the same height atlas.
- **`cascade_factor` moves to a hardware comparison sampler.** It currently issues
  **16 `textureLoad`s** and hand-weights a separable tent, explicitly to avoid
  binding a `sampler_comparison`. `textureSampleCompare` gives hardware 2×2 PCF
  per tap, so 3×3 taps reach equivalent filter quality at 9 samples with free
  bilinear. This is a prerequisite for PCSS, not an independent optimisation — it
  frees the budget PCSS spends.
- **Cascade extents stay where they are.** Requests to "reach further" are
  answered by the horizon term, not by growing the boxes.

## Alternatives

- **Virtual shadow maps.** Rejected. VSM's win is allocating shadow resolution by
  screen-space need, which serves the mid-field — the one regime already adequate.
  It does not address contact or far-field at all. Three specific reasons it pays
  even worse here than the general case: (1) Thalos's dominant receiver, the
  terrain, **never renders into the cascade maps** (noted in `thalos::shadow`), so
  the caster set is small — trees, rocks, craft, buildings — which is exactly the
  regime where cascades are efficient and a page table is overhead; (2) VSM wants
  Nanite-style clustered geometry to feed pages cheaply, which does not exist and
  is not planned (see ADR-20260722T105146Z-stay-on-bevy-reject-engine-migration);
  (3) page-table allocation, caching, and invalidation is a large from-scratch
  build in Bevy. Reopening requires evidence that mid-field resolution — not
  contact or far-field — is the binding fidelity constraint.
- **Grow the cascades / add a fourth cascade for the far field.** Rejected: at
  100 km a 4096² cascade is 24 m/texel, below the size of the casters that matter,
  and each added cascade widens the bias-cap conflict that already forces every
  constant in `thalos::shadow` to be clamped.
- **Raytraced shadows.** Rejected for this sprint: requires a BVH over
  runtime-streamed procedural terrain, which is the same height-authority problem
  that blocks GPU tile production (`terrain_lod_optimization.md`).
- **Keep tuning the existing rig.** Rejected: nine rounds is sufficient evidence
  that the remaining error is structural. The residual artifacts (floating craft,
  no mountain shadows in valleys) are precisely the two regimes cascades do not
  serve.

## Consequences

- **Three mechanisms to maintain instead of one**, with two handoffs (contact →
  cascade at ~0.5 m, cascade → horizon at ~4 km). Accepted deliberately: each
  mechanism is simple and correct in its band, versus one mechanism wrong in two
  of three. Both handoffs need the `shadow` comparison axis (BL-37).
- **Pressure comes off the bias constants.** Once the cascades are not asked to
  cover 100 km, the hard caps in `thalos::shadow` stop being load-bearing and can
  relax toward physically-derived values.
- **W18a is the highest value-per-effort item on the shadow list** — contact
  shadows are what make craft, gear, and trunk bases stop reading as pasted on,
  and they reuse existing infrastructure entirely.
- **Ordering constraint:** the comparison-sampler refactor precedes PCSS, and
  W11's froxel volume (ADR-20260722T111847Z) precedes volumetric shafts. Shafts
  are the payoff for doing aerial perspective before further shadow filtering.
- **We accept worse mid-field shadow resolution than UE5** for the foreseeable
  future. For a game judged on scale and planets rather than on interiors, that is
  the right place to be behind.
