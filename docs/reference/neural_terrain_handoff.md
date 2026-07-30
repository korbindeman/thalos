# Neural terrain — continuation handoff

**Written 2026-07-29** by two agents working in parallel (`AGENT-A`: conditioning
generator + province bench; `AGENT-B`: coastline, pipeline, assets, measurement),
each cross-checking the other's measurements. **This document is the complete
durable output of that work** — the negotiation log it was distilled from has
been deleted deliberately, because it carried four superseded answers to the same
question and a grep would surface the wrong one without its retraction. Where a
number here supersedes an earlier one, it says so inline.

**Goal:** a full Thalos bake that reads as a **gorgeous, diverse planet**, with
terrain we can iterate on for a while.

**Read first:** `CLAUDE.md` · `docs/backlog.md` rows `NTR-X2*`, `X3`, `X4`, `X5`,
`X6` · `docs/roadmap/neural_terrain_renderer.md` ·
[ADR-20260729T054550Z](../adr/20260729T054550Z-coastline-is-authored-on-every-terrain-backing.md) ·
`BENCH_NOTES.md` in the sibling `~/Documents/terrain-diffusion` checkout.

---

## Rule zero — capture before you build

`just screenshot` works. Two agents spent a full session measuring rasters with a
working lane available, and the one frame finally taken did not look like the
numbers implied. **Do not repeat that.**

**RESOLVED 2026-07-29 (afternoon session).** The `massif-aerial` "memory
runaway" was **not a terrain leak**: two renderers (the interactive game plus a
capture host, some still running pre-lease binaries) shared the 12 GB card,
WDDM evicted video memory into system RAM, and the ballooned working set
tripped the 8 GiB RSS watchdog. Diagnosis + fix are recorded in
INC-20260729T081809Z (renderer overlap; the `renderer_lease` is the boundary).
Verified solo: massif-aerial captures at **1.9 GiB peak RSS** on warm
canonical, warm diffusion, *and* cold-cache diffusion. Two hardenings landed
with the verification: `frame_gauge` now carries `rss_mib` +
`mesh_cpu_mib`/`image_cpu_mib` (the CPU side was unmeasured — the OOM was
undiagnosable from GPU gauges alone), and `stop_server` escalates a missed
graceful shutdown to a confirmed forced kill instead of abandoning a live
zombie host (the pre-lease double-render trigger; post-lease it would strand
the lane in "renderer busy"). The old `memory_growth` reading (tile+slab boot
ramp that settles) was a non-finding; the check now prefers `rss_mib`.

Second, cheaper lever on the same goal: `capture_boot_rate` was 72 % — three
quarters of shots paying a host boot, `restart_stale_source` dominating, with
`capture_latency` p95 167 s / p50 36 s. Capture cost is dominated by **rebuilds,
not render**. **Trimmed 2026-07-29:** `examples/`, `tests/`, and `benches/`
trees no longer count in the host's build fingerprint (they never link into the
host binary, and exporter/bench iteration was forcing spurious ~2 min restarts
on every agent). The remaining restarts are legitimate — real source edits and
startup-override switches.

## The axis is decided: the remaining gap is CONTENT

**Answered 2026-07-29 from fresh captures** (massif-valley / massif-ridge /
massif-aerial on the diffusion backing, receipts exact — do not re-litigate
from prose):

- **Presentation is landed at valley/ridge framings.** `NTR-X5`'s exposure fix
  is real in the frame, and `NTR-X6`'s in-band half (tiles cast into the
  cascade, landed 2026-07-26) visibly works: ridges shadow their leeward
  gullies across the massif-valley flank where `shadow_f` used to be ≈ 1.
- **The one open presentation item** is `NTR-X6`'s far-field half: beyond the
  ~6.5–13 km caster band (i.e. exactly what `massif-aerial` at 22 km frames)
  there is still no cast-shadow mechanism, so aerial framings stay flatter
  than mid framings. That row is `verify` and its horizon-term half is open —
  it is the *only* presentation workstream; do not invent another.
- **Everything else that keeps the frames short of "gorgeous" is content and
  surfacing**: soft 90 m forms (bilinear over the height band) and monotone
  clay-like material masses — no strata, no scree, no canopy grain. That is
  `NTR-X4`'s brief (`docs/reference/showcase_patch_prompt.md`) plus `NTR-X3`
  coverage, i.e. the content plan below.

## BLOCKER — do not run the X3 bake on the 5-channel chart

**Added 2026-07-29 evening, after the user flew it.** The bake takes the coarse
chart as its input, and the chart that was installed when the plan below was
written has been **reverted**.

The user's verdict: *"you made the terrain flatter, it looks worse, the features
are far less pronounced on a surface scale, few mountains."* Then: *"yes revert
it."* The 3-channel chart is installed (`sha 0e02294c87d90427`), verified back to
land mean 911.2 m / 69 km relief p50 **451.0 m** / max 8297.6 m.

**Why, and it is not a uniform flattening.** Binned by the old chart's own 69 km
relief, area-weighted, new/old ratio:

| old 69 km relief | % of land | kept |
|---|---:|---:|
| 0–150 m | 8.8 % | 0.75 |
| 150–300 m | 22.0 % | 0.64 |
| **300–600 m** | **30.4 %** | **0.60** |
| 600–1000 m | 17.7 % | 0.65 |
| 1000–1600 m | 11.5 % | 0.80 |
| 1600–2600 m | 7.7 % | 0.89 |
| 2600 m+ | 1.9 % | 0.90 |

**Genuine mountains kept ~90 % of their relief; the rolling terrain covering 48 %
of the land kept 60–65 %.** The peaks survived and everything between them
flattened, which is exactly why it reads as "few mountains" — **variety
collapsed, not scale.** This is the shape the province bench predicted (gentle
provinces lose ~1.9–2.4×, high-orogeny ones barely), measured on the real planet.

**Ruled out, so nobody re-derives it:** the two height-coupled sub-band gains
barely moved — `chart_rough_scale` (fine + erosion amplitude outside the detail
window) 0.626 → 0.608, `relief_w` (sub-chart octaves) 1.268 → 1.111. There is no
conditioning-coupling amplifier; the surface-scale loss is the macro band itself.

**Consequences:**

- **`NTR-X2d`'s conditioning is un-shipped, not wrong.** All five channels, the
  four post-`finalize` transforms, the province classifier and the area-weighted
  reporting stand in the generator and are correct. Only their *output* is out of
  the asset.
- **Both charts are preserved at `artifacts/terrain_charts/`** with provenance.
  The 3-channel one is **not reproducible** — it needs the pre-`NTR-X2d` generator
  and its conditioning rasters were overwritten. That directory is the only copy;
  it is not scratch.
- **The re-ship gate is a hypothesis, and it is not level.** The `temp_sd` sweep
  recovered only 29 %, so the cause is not how strong the channels are. Best
  remaining candidate: **the authored channels are spatially smooth** — a latitude
  ramp plus one moisture field — where the Perlin fallback carried multi-scale
  structure. If the producer responds to *variety* in the conditioning rather than
  its level, correctness and relief are both available. The bench can test it with
  no bake: same province, same level, authored-smooth vs authored-with-structure.
  **That test is the real gate on step 1 below.**
- No `GENERATOR_VERSION` bump was needed for the revert: `content_fingerprint` is
  content-hashed since `NTR-X2i`, so swapping the chart re-namespaces the tile
  cache automatically. The length-only hash it replaced would have served
  3-channel tiles out of the 5-channel cache — same dimensions, different content.

## The content plan

1. **`NTR-X3` — planet-wide 720 m band.** ~99 % of Thalos is analytic filler
   today, and the repo's own user-verified lesson (`NTR-M4`) is that *perceived
   quality tracked model coverage*. This is the band we ship. **Strip gate
   MEASURED 2026-07-29** (`ntr_latent_strip.py`, BENCH_NOTES.md): 28,600 latent
   px/s on the 4070 Ti → **the full planet band is ≈ 3.75 h, one evening**.
   Corr +0.965 vs the installed chart; 720 m slope median 1.07° matches
   NTR-X2k's independently-derived 1.039°. Remaining X3 work: run the bake
   (after X2f, to avoid the wasted pre-province re-bake), int16-residual
   encode, and the DiffusionSurface mid band + multi-window loader.
2. **`NTR-X2f` — landform provinces authored INTO `ProceduralSurface`**, developed
   **in parallel** on the province-atlas bench (no bake, no `GENERATOR_VERSION`,
   ~4.5 min/matrix). Thalos is 63 % plain / 2.2 % plateau by area, and plateau is
   the province that yields canyon geometry. When it lands, X3 re-bakes once —
   a bounded cost, deliberately accepted.
3. **Curated 30 m showcase windows.** 30 m is materially better for arid
   structural terrain and is a **window** tool, never a planet tool.

## What we ship

Measured, not estimated — int16 @1 m stored as a residual against the parent band
(the cascade already composes that way, so the encoding is free), lzma:

| | rate | planet-wide |
|---|---|---|
| 23 km chart | — | **1.5 MB** (have it) |
| 720 m band | 0.99 B/px | **~0.38 GB** |
| one 553 km 90 m window | 0.58 B/px | **22 MB** |
| 90 m everywhere, pixels | 0.58 B/px | ~14 GB |
| 90 m everywhere, **latents** | — | ~3 GB + on-device decode |
| 30 m everywhere | — | ~220 GB — never |

**Ship: chart + 720 m band + ~20 curated 90 m windows ≈ 0.8 GB.**
"Full bake" therefore means *full planet at the mid band plus curated native
windows*, not full planet at native resolution.

### This is a compromise — know where it costs

Between 720 m and 90 m sits a **factor of 8**: landforms from roughly **200 m to
2 km**. That is valley networks, ridge branching, drainage texture, and canyon
tributaries — and it is *learned* structure. The bench's `plateau_wet` case (a
branching incised network with steep walls) lives exactly there.

The analytic bands below do not replace it. The erosion band **is** slope-steered
(`bevy_erosion_filter`, oriented by finite differences of the bands above), so it
is not pure noise — but it can only follow the fall lines **its parent implies**.
With a 720 m parent it reproduces 720 m-scale drainage; it cannot invent the
few-hundred-metre tributary network the model learned.

And a user has already judged this exact configuration: `NTR-M4`'s
*"the user's 'unrealistic' reports were all from outside the exported window"* —
outside the window is precisely where analytic bands were carrying that band.

**Measured on the real shipped window** (`thalos_site_detail_6144_90m.f32`,
553 km at 90 m; 720 m parent = 8× decimate + bicubic, i.e. what the player
actually gets in an uncurated region):

| | full 90 m terrain | 720 m band alone |
|---|---:|---:|
| height variance in the 90–720 m band | — | **0.35 %** |
| slope median | 1.638° | **1.039°** |
| slope p90 | 13.32° | **7.42°** |
| slope p99 | 27.64° | **16.55°** |

Read those together, because they disagree on purpose. **Shape survives**: 99.65 %
of height variance is coarser than 720 m, so where the mountains are and how tall
they stand is carried by the mid band — orbit and approach framings are
essentially unaffected. **Steepness does not**: the steep tail is roughly halved.
Slope is what the eye reads — shading, silhouette, whether a ridge is rugged or a
smooth dome. Uncurated terrain will have the right mountains in the right places,
visibly softened.

Caveat, and it matters: this is one window, sited on the showcase massif —
terrain curated for being dramatic. A plain carries less in that band, so **37 %
of median slope is nearer an upper bound than an average.** Re-measure on a
typical region before treating it as the planet-wide figure.

*Compression itself is not where fidelity is lost.* lzma is lossless and the
int16 @1 m quantisation costs 0.29 m rms / 0.5 m max, leaving slope median
1.638° → 1.623° and p99 27.643° → 27.640° — unchanged to three decimals. The
residual spans ±470 m against int16's ±32767, so **1.4 % of the range is in use**:
if quantisation ever shows as crinkle on flat ground, drop to 0.1 m for ~10× less
error at ~1.4× the bytes. Held in reserve, not needed today.

So the cost is real and it is concentrated in **uncurated regions at the
200 m–2 km scale**, which is ~99 % of the surface. Two consequences:

- **Window placement is a first-class design task**, not a byte-budget
  afterthought. Twenty windows is 0.43 GB and covers ~0.4 % of the surface, so
  *where* they go decides whether the compromise is invisible or obvious. Track
  where players actually fly: spaceport and runway approaches, the showcase
  massif, transited coastlines, and anywhere a scenario directs flight.
- **Open question for the user, not for an agent:** is the download budget **1 GB
  or 5 GB**? At ~3 GB, latents + on-device decode give 90 m-equivalent detail
  everywhere and curated windows become a stopgap rather than the design. "Under
  a gigabyte" is a target we set ourselves; nobody gave it to us. This is open
  fork **Q10** and the answer changes the shape of the plan.

## Invariants — expensive to rediscover

- **The coastline is authored on every backing.** ADR-20260729T054550Z. Layer A
  (`ProceduralSurface::macro_signed_height_m`) is the signed sea field and is
  **not a band**. A learned band may never own the zero crossing: the relief
  cascade is LOD-aware, and a waterline that moves with camera distance is
  INC-0003.
- **Band cascade contract.** Each band contributes `sample − parent`, where parent
  is the **accumulated sum of all coarser bands**. Adding a band is one line in
  `DiffusionSurface::height`; `elevation_payloads` makes the cache namespace and
  the LOD height budget update by construction. Pinned by
  `detail_residual_counts_parent_once` — that test fails on the double-count bug.
- **`cond_snr` is a NOISE level — lower is stronger.** Shipped values are already
  near-optimal. Raising it "to trust our data more" weakens it.
- **Mesas are unreachable by conditioning** at any strength or seasonality.
  Canyons **are** reachable, via *humid* plateau conditioning — and landcover is
  decoupled (`macro_albedo_for`), so wet-conditioned geometry can be painted arid.
  **Caveat: that verdict came from hillshades of raw heightfields, not rendered
  frames.** With landcover, lighting and shadows the perceived answer may differ.
- **The `temp_sd` lever is measured and dead.** +21 % relief at maximum (~29 % of
  the gap), costs plateau character at the strong setting, and is invisible in a
  matched hillshade. `precip_cv` is not a lever (+4 %). Do not re-run the sweep.
- **Province cuts are self-calibrating quantiles** (orogeny p70, relief p35/p75,
  elevation p75 of *this body's own land*). **You cannot evaluate X2f by "did
  plateau % go up"** — adding plateaus raises the flatness cut and can leave the
  reported share flat or falling. Fix absolute metre thresholds *before* the
  change or X2f will measure itself into a null result.
- **`finalize_synthetic_map` is bypassed on the import path.** The chart owes the
  producer all four transforms itself (lapse rate, the 20 °C contrast stretch, the
  temp-sd baseline, the precip-CV damping). Breaks silently; looks like bad
  terrain.
- **A channel pinned at a producer-side clamp floor is dead conditioning** that
  still looks present in the raster.
- **The `macro_conditioning::prior` envelope is measured**, not assumed (bench
  `calibrate`, 512² cells). A different model means recalibrate, not reuse.
- **Bench operating rule:** province blocks must exceed the coarse stage's 64-cell
  tile, or no tile the model evaluates ever lies inside one province.
- **Area-weight every equirect statistic** (`cos(lat)`). This produced five wrong
  numbers across two agents in one session, one of them in shipped crate code.

## Method rules, earned the hard way

- **When two independent paths report the same quantity, diff them.** A
  disagreement is a defect **in the pair**, not in whichever file it happens to
  live in — *"it's the other agent's problem"* is not an explanation. A
  0.301-vs-35.2 % land-fraction disagreement sat in plain view for hours, was
  correctly diagnosed by one agent, and was filed as someone else's file.
- **Read the whole channel before attacking a number in it.** A confident,
  entirely wrong critique of the compression budget was drafted and nearly posted
  because the encoding was documented in a notice that had been skipped.
- **A plausible image is not evidence.** The `coastline` preset rendered a
  beautiful open-ocean sunset while its site search was returning a fallback at
  the north pole. Only the `coast_site` diagnostic caught it. Check the receipt
  and the diagnostic, not the picture (BL-20).

## Working as two agents (recommended)

- **AGENT-A** owns the generator, the province bench, `macro_conditioning`.
- **AGENT-B** owns pipeline invocation, installed assets, `DiffusionSurface`,
  measurement.
- Create an append-only channel file with explicit turns (`>>> TURN: <id>`,
  sequence numbers, one message per turn). It worked: **six numeric corrections,
  zero design corrections**, every one found by re-deriving rather than
  re-reading. **Delete it when you are done** and distil what survived into a
  document like this one — a turn-by-turn log is a liability to whoever comes
  next, because superseded numbers outnumber surviving ones and a grep cannot
  tell them apart.
- **Pre-claim a block of backlog row IDs before using any.** Sequential
  `NTR-X2c`-style IDs have no collision resistance and both agents allocated the
  same two. *Open question for the user: should backlog rows use chronological
  IDs like ADRs and incidents (ADR-20260722T170714Z)?* Do not change the
  convention silently.

## Done means

A user flies orbit → surface and calls it gorgeous, on a planet whose diversity is
visible from orbit and holds up at ground level, shipping inside the agreed
download budget.
