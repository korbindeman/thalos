# Expert review — 2026-07-30 — graphics/lighting

- **Run:** 20260730T011353Z · commit `2fb6db6`
- **Slices:** `shading` · `ground-scatter` · `render-integration` · `clouds` (all lens `graphics`)
- **Scope:** user-named — tree lighting inconsistencies, ship-in-space darkness, shadow inconsistency for clouds and terrain scatter. Code-only; no captures were taken.
- **Evidence:** full (toolchain available; all verdicts rest on traced code paths and arithmetic, no repro tests — the slices are Bevy/WGSL, where the skill prefers cited paths over generation-adjacent tests)
- **Findings:** 15 confirmed · 1 plausible · 8 dropped

Three experts independently derived the same two mechanisms (the discarded
double-sided normal flip; the night cascade gating moonlight) — those are merged
below under one id each with the duplicate ids noted. Every finding carries the
refuter's corrections **in the finding**; several fixes were rescoped or
reversed by the refutation pass, so read the fix lines, not just the claims.

**How the findings map to the reported symptoms:**

- *"Tree lighting at times"* → *(1)* back-face leaf lighting flips with view/wind
  (normal-flip discard), *(8)* impostor trees don't receive the shadows they
  cast, *(10)* the whole forest can be lit/extinguished by the **craft's**
  horizon rather than the view's, *(5)* tree cloud-shadows jitter frame-to-frame.
- *"Ship in space a bit dark"* → *(9)* the reflection probe's planet disc and
  starfield never got the photometric conversion the sun disc got (the probe is
  the dominant broad-field light on a metallic hull). The flat space ambient
  being ~half its constant's face value was **dismissed as by-design**
  (eyeball-tuned with the tint in place; see Dropped) — brightness retunes route
  through GF-CAL.
- *"Shadows inconsistent for clouds"* → *(3)* the cloud-shadow march can't cover
  the shell (vanishes at low sun exactly when real shadows are longest), *(6)*
  its clear-air cull samples weather tens of km down-beam, *(2)* the phase
  function is sign-flipped so glare/silver-lining renders 180° from the sun,
  *(7)* receivers above cloud base are shadowed by cloud below them.
- *"…and terrain scatter"* → *(1)*, *(4)* runway/structures/hull don't dim under
  clouds while terrain+trees now do, *(5)*, *(8)*, and *(16)* spine scatter
  (grass/rocks) receives no moonlight at all.

---

## Confirmed

### 1. shading-gfx-1 — Mesh trees discard Bevy's double-sided normal flip; back-facing leaves shade front-lit with a dead transmission lobe
**`logic`** · [`assets/shaders/tree_standard.wgsl:252`](../../assets/shaders/tree_standard.wgsl#L252) (and `:287`) · slice `shading` · lens `graphics` · *(independently filed as scatter-gfx-1)*

**Mechanism.** `tree_base_material()` sets `double_sided: true, cull_mode: None`
with a comment saying "the standard path flips the normal for us on back-facing
fragments" (`crates/rendering/render/src/ground/tree_material.rs:113-116`). Bevy
0.19 does perform that flip (`prepare_world_normal` — verified against the
registry source for the locked `bevy_pbr`; no tangents and no normal map, so
the flip branch is live). But the fragment then rebuilds
`n_geo = normalize(in.world_normal.xyz)` from the **raw varying** (line 252),
perturbs it, and overwrites `pbr_input.N = n` (line 287) — discarding the flip.
The comment at `:288-289` claiming `world_normal` still "drives the double-sided
flip" is false: once `N` is overwritten, `world_normal` feeds only shadow-fetch
biasing.

**Failure.** A leaf card lit from its front, viewed from behind (half of any
wind-agitated canopy): renders fully front-lit diffuse instead of dim diffuse +
warm `diffuse_transmission`. Brightness pops as wind or camera motion flips
which face is seen — a direct mechanism for "tree lighting looks wrong at
times."

**Refuter (confirmed, narrowed).** Every link verified against
`bevy_pbr-0.19.0` registry source and the actual tree mesh attribute sets (no
`ATTRIBUTE_TANGENT` in either the standalone builder or the runtime combiner).
Three corrections: the backlit-canopy look is **diluted roughly by half**, not
eliminated (front-facing away-tilted cards still transmit correctly); the
"garbage Fresnel" sub-claim is overstated (Bevy clamps `NdotV`); the bark
sub-claim is mechanically true but visually negligible (closed cylinder,
interior back faces occluded, zero transmission).

**Fix.** Derive the perturbation base from `pbr_input.world_normal` (post-flip)
instead of `in.world_normal`; correct the two misleading comments
(`tree_material.rs:113-116`, `tree_standard.wgsl:288-289`). Verification
belongs on the existing `BL-20260729T031500Z-veg-standard-path` row's
not-yet-run "backlit transmit" capture check.

### 2. clouds-gfx-2 — The cloud phase-function argument is sign-flipped: forward glare and silver lining render 180° from the sun
**`logic`** · [`crates/rendering/render/src/clouds/shaders/clouds_compute.wgsl:945`](../../crates/rendering/render/src/clouds/shaders/clouds_compute.wgsl#L945) · slice `clouds` · lens `graphics`

**Mechanism.** `sun_dir` points toward the star (four independent code
witnesses). The correct scattering cosine for the camera is
`dot(ray_dir, sun_dir)`; the code computes `dot(ray_dir, -sun_dir)` — its
negation. The local Henyey-Greenstein (`:650-653`) is the standard form peaking
at +1 for g>0, and no compensating negation exists anywhere on the path (checked
at every commit touching the shader — the flip predates all capture rounds).
The powder term (`:779-786`, whose comment is arithmetically false as written)
and `ms_aniso` (`:1134`) inherit the same flip.

**Failure.** Looking sunward, the mixed lobe evaluates at ≈0.077; anti-solar at
≈1.813 — a **~23×** asymmetry pointed the wrong way (the expert's "35× after
the clamp" was overstated; the clamp never engages at the shipped g-mix).
Sunward cloud edges lack glare/silver lining; anti-solar cells carry an
unphysical bright rim; the far tier (symmetric n·l) disagrees with the near tier
by sun azimuth at the handoff.

**Refuter (confirmed).** Full independent sign derivation; explains why no
capture caught it (the flip predates every capture, and all recorded round-8
checks are direction-agnostic). Corroborating historical tell: a comment records
that a former 0.85 "anti-sun darkening painted lobes near-black and read as
dirt" and was hand-weakened — under the flip that darkening was actually hitting
the *sunward* side; someone tuned around this bug's signature without finding it.

**Fix.** `let ray_dot_sun = dot(ray_dir, config.sun_dir.xyz);` plus the two
false comments — **but** the powder constants (0.35/0.35), `MS_ANISO`, and
possibly the 2.2 forward clamp were tuned against the flipped geometry: a
sign-only fix brightens sunward thin edges ~23× and needs a
`cloud-sunset`/`cloud-cruise` recapture/retune pass in the same change.

### 3. clouds-gfx-1 — The cloud-shadow sun march is hard-capped at 8 km of beam; shadows lighten and vanish at low sun, exactly when real ones are longest
**`logic`** (filed `fundamental`, lowered) · [`crates/rendering/render/src/clouds/shaders/clouds_compute.wgsl:1578`](../../crates/rendering/render/src/clouds/shaders/clouds_compute.wgsl#L1578) · slice `clouds` · lens `graphics`

**Mechanism.** `step_m = min(span/CLOUD_SHADOW_STEPS, CLOUD_SHADOW_MAX_STEP_M)`
with 20 steps × 400 m caps the marched path at 8 km; the loop's only other exit
is `t > exit`, and the march always starts at the **shell bottom** (900 m
altitude), not the local layer base. Thalos's shell is 900–11,400 m, and the
authored layer bands are not low (storm to ~11.2 km, cirrus veil at
8.7–10.9 km) — the header comment's "~1–3 km slab" premise that sized the
budget is false as authored.

**Failure.** Marched ceiling ≈ 900 + 8000·sin(elev) m. Above ~25–33° sun the
ordinary deck is fully covered (the expert's "under-integrates at every
elevation" was wrong — only cirrus and storm anvils are lost up high, and
opaque storm columns saturate τ anyway). The visually real regime is **below
~25°**: at 15° only the bottom ~2 km integrates, and at ~12° a high-base
cumulus deck (base 2685 m) gets **zero samples in cloud** — an opaque overcast
lays no shadow at all, because clear air below the deck exhausts the entire
budget. This breaks the module's own invariant (`:1505-1507`).

**Refuter (confirmed, corrected as above).** Not recorded anywhere as a cost
decision; CLOUD-5's known-limits list does not include it.

**Fix.** Distribute the step budget over the density-bearing span (start at
`max(entry, local-base crossing)`, drop or raise the 400 m cap) and feed the
resulting step into the density `filter_m` so the field band-limits to the
stride — the same alias-safety contract the view march uses.

### 4. shading-gfx-4 — Cloud shadows now split the *non-terrain* world: terrain + trees dim under a deck, runway/structures/hull do not
**`design`** · [`assets/shaders/shadowed_standard.wgsl:43`](../../assets/shaders/shadowed_standard.wgsl#L43) · slice `shading` · lens `graphics` · *sharpens CLOUD-5 (`docs/backlog.md`)*

**Mechanism.** Tree and impostor materials now bind and sample
`cloud_sun_transmittance` (`tree_standard.wgsl:308-314`,
`tree_impostor_standard.wgsl:312-317`; bindings verified populated at runtime
via `update_tree_material`). `shadowed_standard.wgsl` (runway/structures) and
`ship_part.wgsl` bind only the sun cascade — no cloud block at all.

**Failure.** Broken cumulus over the spaceport: taxiway and canopy carry
drifting cloud shadows; the hangar, runway strips, and parked craft stay
full-sun inside the same shadow footprint — a hard material seam in one frame.

**Refuter (confirmed as a sharpener).** The CLOUD-5 row and `clouds.md` §3.5 are
**stale in two directions**: they say trees do *not* receive (now false), and
their remaining-work enumerations never name structures/runway. The craft half
is tracked verbatim ("scatter + craft receivers"). The right action is amending
CLOUD-5's remaining-work list — trees done; structures/hull added explicitly,
arguably ahead of rocks since they share a frame with the dimming canopy — not
a new row.

**Fix.** Extend `ShadowReceiveExtension`/`ShipPartExtension` with the
`CloudShadowBlock` + texture pair and fold `cloud_sun_transmittance` into the
existing direct-only shadow multiply (~10 shader lines per material; the exact
Rust fan-in template exists in `tree_material.rs` + `update_tree_material`).

### 5. clouds-gfx-4 — Trees sample the cloud-shadow map through a one-frame-stale frame block: multi-texel shadow jitter every unpaused frame
**`logic`** · [`crates/runtime/game/src/rendering/vegetation.rs:1506`](../../crates/runtime/game/src/rendering/vegetation.rs#L1506) · slice `clouds` · lens `graphics` · *sharpens CLOUD-5*

**Mechanism.** `drive_clouds` writes `CloudShadowMap` (frame + `world_to_body` +
`body_center_ws`) in PostUpdate; `update_tree_material` copies the block in
Update — **deterministically one frame stale, every frame** — while the map
texture is re-marched in place with the current frame's placement. The godray
block writer (`update_body_terrain_atmosphere`, PostUpdate) is mutually
unordered against `drive_clouds` despite a resource conflict. The tile fan-out
runs in `Last` with a comment naming this exact hazard.

**Failure.** The refuter *raised* the filed magnitude: the stale fields are
render-space anchors, and in big_space the body's render translation moves at
orbital speed (~29.8 km/s ⇒ ~500 m/frame at 60 fps, alternating with 1 km cell
snaps). Trees mislocate their cloud-shadow lookup by **2 texels (top rung) to
16 texels (bottom rung) every unpaused frame** while the ground beside them is
exact — frame-rate shadow jitter on canopies. Paused captures hide it entirely,
which is why the just-landed tree receiver could pass screenshot verification.

**Refuter (confirmed).** All four schedule registrations verified; an in-tree
incident (`20260729T051825Z-shadow-state-was-not-a-frame-transaction`) already
establishes same-frame fan-out as the design for exactly this class of bug.

**Fix.** Move only the cloud-block half of `update_tree_material` to `Last`
beside `apply_cloud_shadow`; order `update_body_terrain_atmosphere`
`.after(drive_clouds)` (the composite sync is already ordered; the sky-block
writer was missed).

### 6. clouds-gfx-5 — The shadow march's clear-air cull samples weather up to ~87 km down-beam; solid texels write fully-lit holes
**`logic`** · [`crates/rendering/render/src/clouds/shaders/clouds_compute.wgsl:1564`](../../crates/rendering/render/src/clouds/shaders/clouds_compute.wgsl#L1564) · slice `clouds` · lens `graphics`

**Mechanism.** The shadow entry anchors its coverage early-out (single level-0
weather fetch; `coverage ≤ 1e-3 → transmittance 1`) at the **unclamped**
midpoint of the full shell chord. The view march clamps its equivalent anchor
to ≤25 km and probes coarse mips before culling — a guard added for exactly
this failure ("clouds vanish whenever the anchor lands in the clear lane ahead
of a system") that was never propagated; the shadow entry's comment still
claims it samples "exactly as the view march takes it," which is no longer true.

**Failure.** The midpoint's horizontal displacement exceeds one weather texel
for **every sun elevation below ~47°** (~1.9 texels at 30°, ~18 at the 3.4°
floor). With the producer's authored true-zero clear lanes, a texel under solid
cloud whose midpoint lands in a lane writes transmittance 1 — holes in the
cloud shadow that migrate with sun azimuth/elevation.

**Refuter (confirmed, narrowed).** The march itself fetches per-step local
weather, so the integral is essentially correct when it runs; the damage is
confined to the early-out cull (plus a ±0.025 threshold nudge). No effect under
unbroken overcast; pervasive at low sun over lane-cut morphology, which the
producer authors deliberately.

**Fix.** Clamp the context point to ≤25 km past entry and cull on a coarse-mip
region probe — mirror the view march's own pattern in the same file.

### 7. clouds-gfx-3 — The cloud-shadow cascade has no altitude term: receivers above cloud base are shadowed by cloud below them
**`logic`** (latent) · [`crates/rendering/shading/src/shaders/cloud_shadow.wgsl:97`](../../crates/rendering/shading/src/shaders/cloud_shadow.wgsl#L97) · slice `clouds` · lens `graphics`

**Mechanism.** Each texel integrates the entire shell (from 900 m altitude) and
the receiver lookup intersects the sun ray with the sea-level tangent plane; its
own comment asserts "the segment between plane and receiver is clear air either
way" — true only below cloud base. Thalos terrain reaches ~4.9 km (massif) vs a
900 m base. The godray term applies the same transmittance at every atmosphere
sample with no altitude gate, so wrongly-darkened columns also extend *above*
the deck (the below-deck half is correct; the expert's "instead of" was
overstated).

**Failure.** A summit above a broken low deck takes cloud shadow from cloud
strictly below it; a band of air in the first few km above the local cloud top
loses single-scatter airlight. Conditional on cloud mass sunward within the
cascade footprint, not universal.

**Refuter (confirmed, urgency lowered).** Real but **latent**: every land
preset sits in an authored clear lane and the one cloudy-land probe selects
land at ~644 m — below base — so nothing in the current capture inventory can
show it. Belongs on CLOUD-5's known-limits/remaining-work list, not as an
urgent fix.

**Fix.** The map's alpha channel is genuinely spare: store an
extinction-weighted mean deck height per texel and fade the term by receiver
altitude in `cloud_shadow_lookup` — a defensible first cut (a single height
moment cannot represent multi-layer columns; exact behaviour needs a τ profile,
out of scope).

### 8. scatter-gfx-4 — Impostor trees cast sun shadows into cascades they refuse to sample, on a stale "beyond the cascade region" rationale
**`logic`** · [`assets/shaders/tree_impostor_standard.wgsl:309`](../../assets/shaders/tree_impostor_standard.wgsl#L309) · slice `ground-scatter` · lens `graphics` · *sharpens BL-20260726T225010Z*

**Mechanism.** The shader comment says "the impostor band is beyond the cascade
region." `CASCADE_MIN_HALF_M = [0, 3000, 6500]` says the opposite — its own doc
comment calls shadows running out inside that band "a coverage bug." Impostor
tiles in rings 0–1 are placed on `SHADOW_CASTER_LAYER` and render the canopy
silhouette **into** those cascades; the fragment never samples them. LOD is
slant-keyed (`view_d = √(d² + agl²)`), so from ≥ ~1.2 km AGL the *entire*
forest below is impostors.

**Failure.** (a) Terrain shadow crossing a valley darkens ground while impostor
trees inside it stay lit — bright trees on dark ground; (b) at the 1.2 km
mesh↔impostor swap a shaded tree pops to lit; (c) aerial framings — the
keystone sprint's target surface — show a uniformly lit canopy above its own
cast shadows.

**Refuter (confirmed as a sharpener, narrowed).** The stale rationale is false
specifically in the 1.2–6.5 km caster band (still true past 6.5 km, where
rings 2–3 don't cast and W12 owns the far field). The comment postdates the
tracked row by three days, so the row cannot cover it; `vegetation.md` §12
("Impostors do not cast") actively contradicts current code. The row's own fix
note ("probably the W12 horizon term rather than extending cascades") is partly
wrong: no extension is needed — the cascades already cover the band; sampling
them is 1–3 depth taps per card pixel, and the two fixes split cleanly at
`TREE_SHADOW_CASTER_MAX_M`.

**Fix.** Fold into BL-20260726T225010Z: sample cascades 1/2 in the impostor
fragment (the standard-path material family already binds the block pattern),
delete the stale comment, and fix `vegetation.md` §12.

### 9. integ-gfx-2 — The reflection probe's planet disc and starfield never got the photometric conversion its sun disc got
**`logic`** · [`crates/runtime/game/src/reflection_probe.rs:840`](../../crates/runtime/game/src/reflection_probe.rs#L840) · slice `render-integration` · lens `graphics` · *the mechanism behind frozen row F8b's "magic constants"*

**Mechanism.** The probe declares one photometric unit system
(`reflection_probe.rs:93-98`: "keeps the reflected sky and the sun in one unit
system"). `sun_disc_radiance = flux·SUN_DISC_GAIN` was converted to that system
(`:750`, `:783`); `planet_color = (0.25, 0.35, 0.55)` and
`starfield_tint = 0.015` (`:840`, `:842`) were never converted — an incomplete
migration, not a decision (no record says the constants deliberately stay
pre-photometric).

**Failure.** At the homeworld a Lambertian disc should paint ≈ albedo·flux/π ≈
0.95; the code paints ≤ ~0.55 — **~2–4× under-bright**, and on a
metallic-1.0/roughness-0.08 hull the specular probe *is* the dominant
broad-field light (ambient's diffuse term is annihilated by the metal branch).
The error's sign flips with focus distance: at outer-system focus the fixed
disc becomes roughly an order of magnitude too **bright** against a
correctly-exposed sun disc. A concrete "ship in space is a bit dark"
contributor that survives independent of any ambient retune.

**Refuter (confirmed, corrected).** Line anchors drifted (correct ones above);
and the proposed fix as filed misses the branch that actually renders at the
homeworld — `orbital_sample` (`:942-944`) uses the baked `ImpostorAlbedo`
directly, with the same missing flux/π. The scale factor belongs in
`orbital_sample`, applied to **both** branches.

**Fix.** Apply the flux-derived factor (planet: albedo·flux/π; starfield
similarly) inside `orbital_sample` for both the painted and baked-albedo
branches. Weight as a calibration inconsistency in a stand-in with a recorded
retirement path (F8b, frozen) — but the internal inconsistency is fixable now
without F7.

### 10. integ-gfx-3 — The global sun (and moon) light is gated by daylight and horizon visibility evaluated at the craft, not the view
**`logic`** · [`crates/runtime/game/src/rendering/lighting.rs:517`](../../crates/runtime/game/src/rendering/lighting.rs#L517) · slice `render-integration` · lens `graphics`

**Mechanism.** `craft_pos` feeds both the terminator gate and the W12 horizon
march (active below 15 km AGL); the product scales the single
`DirectionalLight` every standard-path surface shares. `horizon_vis` has **no
floor** — it reaches exactly 0. The shadow rig (INC-20260724T232104Z) and
celestial visibility (BL-12) were both already moved to the ViewAnchor for this
class of bug; the light was not.

**Failure.** The sharp case: craft parked in a valley at low sun →
`horizon_vis → 0` → every sunlit ridge, tree, and structure in a freecam/
god-view frame goes unlit under a daylight sky, and rising the camera out of
the valley does not restore it. Since trees moved to the standard path
(2026-07-29) this gate extinguishes whole forests, not just the parked ship.
The terminator half is marginal (the band is hundreds of km wide; ordinary
captures won't show it), and terrain *ambient* already divides the craft's
daylight out per fragment — the uncorrected term is the direct illuminance.

**Refuter (confirmed, fix rescoped).** The proposed ViewAnchor re-anchor is
**wrong for the horizon term**: its recorded purpose is dropping the parked
craft into mountain shadow, and view-anchoring would relight it whenever the
camera swings outside the occlusion. With one global scalar light no anchor is
correct — the terminator term may follow BL-12 to the ViewAnchor; the horizon
term's real resolution is per-receiver sun visibility (tracked: NTR-RT2/W12r).
Interim mitigation, if wanted: bound the horizon term's effect when view and
craft separate.

### 11. shading-gfx-2 — The sun cascade never stands down at night and gates the *moon's* direct light along a below-horizon sun's geometry
**`logic`** · [`crates/runtime/game/src/rendering/sun_shadow.rs:777`](../../crates/runtime/game/src/rendering/sun_shadow.rs#L777) · slice `shading` · lens `graphics` · *(independently filed as scatter-gfx-3 and integ-gfx-4 — three experts, one mechanism)*

**Mechanism.** `gate.x = SHADOW_STRENGTH` (0.88) is published whenever a
terrain body is near; the only deactivation is losing the body. No daylight or
elevation gate exists anywhere in the rig (verified: no `SunDaylight` consumer;
`SHADOW_MIN_SUN_SIN` clamps only the slack term). The cascade cameras are
independent `Camera3d`s that keep rendering casters (trees, structures, craft,
terrain caster child) from a below-horizon sun direction all night. Every
direct/indirect split then applies the sun-projected factor to the **summed**
direct term — which at night is the `MoonLight` directional (up to 60 lux,
deliberately ~12–15× the night floor).

**Failure.** Moonlit night: canopy/rock/hull/ground brightness carved by
up-to-88 % patches whose geometry follows an invisible underground sun,
drifting as it moves. Plus three 4096² cascade renders per frame all night for
a light at zero illuminance. (One filed detail corrected: with backface
culling, the night maps carry caster-shaped anti-shadows, not guaranteed
horizon-length streaks.)

**Refuter (confirmed, fix corrected).** The filed fix — multiply `gate.x` by
`SunDaylight` — is **wrong**: that ramp is fractional up to ~7° elevation and
would fade legitimate golden-hour shadows exactly when they are longest. The
correct gate is 1 for any sun above the horizon, ramping to 0 only below (e.g.
over `sin_elev ∈ [-0.06, 0]`), deactivating the cascade cameras with it.
Zeroing `gate.z` (contact) alongside is safe — it is documented as a
direct-sun-only gate. Twilight moon-share mis-gating remains second-order.

### 12. integ-gfx-5 — Two unordered `Last` systems both rewrite every tile material's shadow payload
**`design`** · [`crates/runtime/game/src/rendering/sun_shadow.rs:319`](../../crates/runtime/game/src/rendering/sun_shadow.rs#L319) vs [`crates/rendering/render/src/craft.rs:326`](../../crates/rendering/render/src/craft.rs#L326) · slice `render-integration` · lens `graphics`

**Mechanism.** `sync_shadow_receivers` and `apply_craft_shadow` both write
`TileTerrainMaterial.extension.shadow` + the three cascade handles, registered
as bare `.add_systems(Last, …)` with no ordering; they serialize in
nondeterministic order. Values provably coincide today (both trace to the same
`SunShadowState` block by `Last`) — and the coincidence is load-bearing: any
future craft-specialization of `CraftShadowMaps` makes the tile payload
frame-order-dependent. The docs disagree about which system is the canonical
fan-in (`tiles/material.rs:13` names one, `sun_shadow.rs:266-273` the other).

**Refuter (confirmed; one sub-claim struck).** "Doubles per-frame
dirty-marking" is wrong — per-frame dirtying is the deliberate baseline (a
third system's comment says so); the hazard is purely the unordered dual
ownership. Nothing can be wrong on screen today.

**Fix.** Delete the tile arm from one writer — `sync_shadow_receivers` is the
natural sole writer since only it carries the contact map — and update the four
doc comments that name the other as canonical.

### 13. shading-gfx-6 — Three of five direct/indirect split sites omit the emissive zeroing their own template documents as required
**`nit`** · [`assets/shaders/tile_terrain.wgsl:1181`](../../assets/shaders/tile_terrain.wgsl#L1181), `tree_standard.wgsl:326`, `tree_impostor_standard.wgsl:318` · slice `shading` · lens `graphics`

`shadowed_standard.wgsl:65-74` / `ship_part.wgsl:347-353` state the contract
("occlusions zeroed **and emissive removed** … exactly exposure·direct") and
implement it; the three cited sites zero only occlusions — one even cites
`shadowed_standard.wgsl` as its template while diverging from it. Verified
harmless today (no code path sets emissive on those materials); latent
shadow-kills-emissive trap for future lava/city-lights/glow features. Bonus
from the refuter: backlog row `BL-20260726T222119Z-shadow-round` claims the
tile split landed *with* emissive removal — the record mis-describes the code.
**Fix:** add the zeroing (or hoist the whole split into one shared
`thalos::lighting` helper).

### 14. scatter-gfx-6 — Per-tree appearance seed is tile-centre-relative, so a tree's two coexisting LOD cards disagree in hue during the ring cross-fade
**`nit`** · [`crates/rendering/render/src/ground/scatter.rs:794`](../../crates/rendering/render/src/ground/scatter.rs#L794) · slice `ground-scatter` · lens `graphics`

`root_offset_body_m` is relative to the tile centre; ring 0 (200 m tiles) and
ring 1 (500 m tiles) give the same world tree different offsets, and both
shaders hash it into `foliage_hue_tint` (±5 % r/b). Ring 1 deliberately shares
ring 0's grid so the same tree **is** drawn by both rings at partial scale
across the 2400±300 m band — the exact configuration the shared-grid design
exists to keep stable. Refuter confirmed reachability and the ±5 % figure;
corrections: the invariance statement is `vegetation.md` §1 ¶3 (not §14, which
covers position only), the per-instance landcover tint is already
ring-invariant (defect confined to the jitter channel), and visibility at
2.4 km is undemonstrated — nit stands. **Fix:** seed from a ring-invariant
quantity (e.g. bake a hash of the global Poisson cell into a spare channel).

### 15. clouds-gfx-7 — `shadow_frame.rs` self-contradicts on texel snapping, and `sun_elevation_cos` stores a sine
**`nit`** · [`crates/rendering/render/src/clouds/shadow_frame.rs:58`](../../crates/rendering/render/src/clouds/shadow_frame.rs#L58) · slice `clouds` · lens `graphics`

The `center` field doc says "snapped to the texel lattice"; `resolve()` and the
code deliberately do not snap. `sun_elevation_cos = up·sun` is the sine of
elevation (usage is self-consistent; the 0.06 threshold really is the ~3.4°
stand-down). Refuter verified both. **Fix:** delete the stale phrase; rename or
re-comment the field.

---

## Plausible

### 16. shading-gfx-3 — Spine scatter (grass, rocks, ground patches) receives no moonlight while every standard-path surface does
**`design`** · [`crates/rendering/shading/src/shaders/lighting.wgsl:768`](../../crates/rendering/shading/src/shaders/lighting.wgsl#L768) · slice `shading` · lens `graphics`

**Mechanism (as corrected by the refuter — the filed version was wrong).**
`shade_foliage` (grass's only lighting routine) has no moon term and goes to
the ~0.01 night floor. The filed contrast — "rocks beside it get
`moonlight_radiance`" — is **false**: `rock.wgsl` passes a zeroed
`SceneLighting`, so the moonlight helper early-returns; on the default tile
path the spine `moonlight_radiance` term is *effectively dead code* (its only
live caller is the EOL legacy terrain material). The real seam is **all spine
scatter (grass, rocks, patches — moonless) vs all standard-path surfaces (tile
ground, trees, hull, structures — `MoonLight` at up to 60 lux, authored to be
clearly visible)**.

**Failure.** Full-moon night near the surface: ground and trees show
directional moonlight; grass and rocks render as unlit dark cut-outs.

**Why plausible, not confirmed.** No night preset exists to demonstrate
perceptibility; mechanism verified by read-through only. Partially adjacent to
tracked records (gfx §4.1 "two moon models still to merge", GF-CAL) but neither
names the scatter-gets-zero mechanism.

**Fix.** Not the filed one-liner (`shade_foliage` takes no `SceneLighting`, and
grass binds none of the moonlight fields) — requires uniform plumbing, or
lands free when the tracked veg-standard-path port migrates grass/rocks onto
Bevy's light path. Fixing grass alone would just move the seam to rocks.

---

## Dropped

| id | claim | verdict | reason |
|---|---|---|---|
| scatter-gfx-2 | 5.49 m no-normal shadow bias erases tree/shrub shadows on scatter receivers in cascade ≥ 1 below ~43° sun | `wrong` | Shadow geometry inverted: the depth compare runs along the sun ray, so separation is h/**sin**(elev), not h·sin — low sun makes shadows *more* bias-robust; an 8–13 m tree clears even the 6.25 m ceiling at every elevation. Surviving residue (nit-grade, for whoever next touches `shadow.wgsl`): cascade 1's texel doubled after the `NO_NORMAL_BIAS_SCALE` tuning comment was written (the "pre-W6 hand value" now computes to 5.49 m, not 2.5 m), and the 2.5 m cap sits before the ×2.5 — a ~2× drift in contact-gap width and shrub-shadow thresholds, no wholesale erasure. |
| integ-gfx-1 | Space ambient reaches surfaces at ~half authored brightness (un-normalized `AMBIENT_DAY_TINT`, luma 0.478) | `by-design` | Arithmetic correct, premise false: tint and brightness were born together and every eyeball-tune (including INC-20260724T204059Z's deliberate look-preserving 700→1940 rescale) happened with the tint in the multiply. "Normalizing" now would double the space ambient and regress the preserved look. Brightness retunes route through GF-CAL; residue is a one-line doc tightening. |
| scatter-gfx-5 | Rocks are the worst-hit spine/standard lighting split; port rocks first; live double-fog via `object_aerial_recession` | `wrong` | Fully tracked: BL-20260729T031500Z-veg-standard-path names grass/GPU-grass/rocks/patches as its follow-up and records the double-fog trap verbatim. NTR-X5's "~1 stop" was tile-vs-udlod and was fixed the same day (INC-20260724T204059Z, parity by construction). Rocks-first arithmetic fails: rocks despawn by 260 m AGL and `object_aerial_recession` is an identity below 1000 m — a no-op at every distance a rock draws. |
| shading-gfx-5 | `tile_terrain.wgsl` re-implements the library's specular-AA with a divergent clamp | `wrong` | Different algorithms for different variance sources: the library caps a noisy screen-space `dpdx` estimator; the tile path feeds analytically accumulated retired-detail variance (NTR-X7 P1) where the 0.18 cap would truncate real variance and reintroduce the documented far-field satin sheen. The clamp divergence is load-bearing. |
| shading-gfx-7 | Impostor normal accumulation can normalize a near-zero vector → NaN fireflies | `wrong` | The bake does allow opposing normals (that kill failed), but `normalize` of a near-zero f32 vector is well-defined; exact zero requires a compound measure-zero coincidence of 8-bit decodes, quantized alphas, and ULP-exact bilinear weights. As filed (NaN fireflies), unreachable. |
| clouds-gfx-6 | CloudShadowBlock fan-outs dirty every resident tile/tree material every frame | `wrong` | The tile path iterates ~one shared material (the per-level Vec collapsed to a single handle); tree/impostor materials are a fixed handful and are per-frame-dirty anyway (wind-time uniform). The unconditional `Last` fan-out is the recorded fix of INC-20260729T051825Z (shadow state as a frame transaction) — change-detection gating is the hazard, not the improvement. Per-frame re-march cost is already tracked in CLOUD-5. |
| integ-gfx-6 | Mesh tree tiles missing `NotShadowCaster` falsifies the stated invariant; latent double-cast | `wrong` | The comment scopes to impostor tiles ("every tile" = every impostor ring tile), matching `vegetation.md`:766. Double-cast protection lives at the light (`shadow_maps_enabled: false` with an explicit "don't" comment), and under the hypothetical flip the entire standard-path world — which has zero `NotShadowCaster` anywhere in `crates/rendering/render` — would double-cast; mesh tree tiles are not a distinctively unguarded caster. |
| integ-gfx-7 | Craft-local shadow centre should use the `active_position_m()` accessor like lighting does | `wrong` | Misread accessor: `active_position_m()` is the EVA on-foot controller position, `None` in flight — substituting it changes nothing in flight and regresses the documented craft-as-caster purpose on EVA. The single-craft read is recorded ("Single-vehicle by construction") and its migration is tracked (CL-E1/CL-E2/fleet kernel); one of ~25 interchangeable `ship_state().position` sites. |
| *(sub-claim of shading-gfx-3)* | Rocks beside grass show directional moonlight via `shade_surface` | `wrong` | `rock.wgsl:128` passes a zeroed `SceneLighting`; `moonlight_radiance` early-returns. On the default tile path the spine moonlight term's only live caller is the EOL legacy terrain material — effectively dead code. Parent survives rewritten (see finding 16). |
| *(sub-claim of integ-gfx-5)* | The dual writer doubles per-frame dirty-marking of tile materials | `wrong` | Per-frame dirtying is the deliberate baseline — `update_tile_material_params` dirties the same assets every frame by design and says so; render-world re-prepare is once per changed asset per extract regardless of writer count. Parent survives on the ownership hazard alone. |

---

## Questions for a capture session

The harness cannot screenshot. These would each settle or size a finding above:

1. **Phase sign (finding 2):** any sunset framing, compare cloud-edge brightness
   at matched optical thickness around the solar vs anti-solar point — with the
   current code the bright fringed edges sit **anti-solar**. One capture pair
   before the sign fix lands makes the before/after self-documenting.
2. **Shadow-march cap (finding 3):** `THALOS_CLOUD_SHADOW=show` over the same
   authored deck at ~45° vs ~15° sun — the raw transmittance field should
   visibly empty out as the sun lowers.
3. **Back-face leaves (finding 1):** one broadleaf framed sun-side and
   anti-sun-side — currently the canopy reads near-identically from both sides;
   after the fix the anti-sun side should gain the warm transmission glow.
4. **Night phantom shadows (finding 11):** a moonlit surface preset (sun ~10°
   below horizon) — dark patches on canopy/ground uncorrelated with the moon
   direction. No night preset currently exists; one is worth adding for
   finding 16 as well (grass/rocks as unlit cut-outs vs moonlit ground).
5. **Impostor receive gap (finding 8):** low-sun shot across a valley in
   terrain shadow at 2–5 km — impostor trees inside the shadowed region lit?
6. **Cloud-shadow jitter (finding 5):** needs an *unpaused* sequence (paused
   captures hide it by construction) — a short in-game clip over broken cloud,
   watching tree shadows against the ground's.
7. **Ship-in-orbit brightness (finding 9 + dismissed integ-gfx-1):** an orbit
   preset now, then after the `orbital_sample` flux fix — does the recovered
   planet-disc reflection account for "a bit dark", with any residue routed to
   GF-CAL as a deliberate retune?
