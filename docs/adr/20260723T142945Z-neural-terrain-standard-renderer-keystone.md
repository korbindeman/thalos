# ADR-20260723T142945Z-neural-terrain-standard-renderer-keystone: The keystone is paired neural terrain + a standard-path renderer; Q9 resolves to Bevy's default render path

- **Status:** Accepted
- **Date:** 2026-07-23
- **Related:** ADR-20260722T105146Z-stay-on-bevy-reject-engine-migration ·
  ADR-20260722T105147Z-tile-native-surface-seam (part 2 superseded; parts 1 and 3
  stand) · ADR-20260720T211046Z-offline-terrain-packages (stands) ·
  ADR-20260721T185221Z-custom-rocky-atmosphere (stands) ·
  ADR-20260723T143155Z-planetary-model-pytorch-finetune (companion) ·
  ADR-20260722T084155Z-mira-hero-visual-slice-parallel (visual-slice scope closes
  with this pivot)

## Context

Thalos's two stated goals are **scale** and **visual fidelity**.
ADR-20260722T105146Z committed to reaching them inside Bevy by "adopting more of
Bevy's renderer rather than building parallel replacements." Three pressures have
since converged:

1. **The renderer debt is structural, not incremental.** The two-lighting-universes
   problem (`graphics_fidelity.md` §3) was being closed by pulling crafts onto the
   custom `thalos::lighting` spine (F1–F9). Meanwhile the terrain renderer
   (`thalos_udlod`) sits entirely off Bevy's default render path. BL-34/BL-35 — the
   Mira grazing-angle silhouette bugs — burned a full diagnostic session inside
   udlod's tile/LOD machinery before concluding the machinery itself is the layer at
   fault. They were parked 2026-07-22 citing "the renderer replacement" before any
   such decision was recorded; **this ADR is that record**, made properly.
2. **The earth-like neural terrain case is proven externally.**
   [terrain-diffusion](https://xandergos.github.io/terrain-diffusion/) (reviewed
   again 2026-07-23) is MIT-licensed, ships pretrained weights (30 m and 90 m
   models on Hugging Face), is trained on real ETOPO elevation **plus WorldClim
   climate**, and demonstrates hierarchical, lazy, seed-deterministic O(1)
   random-access generation — the exact properties `mira_airless_mvp.md` §6's S0–S4
   ladder was designing toward, already working for the earth-like family.
3. **ADR-20260722T105147Z part 2** deferred the terrain→`ExtendedMaterial` rework
   behind a trigger (package schema frozen + one body producing real tiles + the
   `verify` queue burned down, then a Mira-only in-repo prototype). The user has
   directed a keystone re-orientation: *make Thalos look good*, with a new renderer
   and neural terrain generation built in harmony, paired for each other.

The standing user-authored probe design ("Standalone Neural Terrain Probe",
2026-07-21, adapted into `docs/roadmap/neural_terrain_renderer.md`) supplies the
vehicle: a new, isolated Bevy application advancing generation, meshing, LOD,
lighting, and streaming as one vertical slice.

## Decision

Five parts.

**1. The keystone sprint is paired neural terrain + a standard-path renderer.**
"Make Thalos look good" becomes the primary sprint, ahead of architecture cleanup
(now background) and the surviving graphics-fidelity work. Terrain generation and
rendering advance as one vertical slice — the tile contract is co-designed with
the producer that fills it, not retrofitted. Strategy lives in
`docs/roadmap/neural_terrain_renderer.md` (cross-ref prefix `ntr §N`).

**2. Q9 resolves: every opaque surface renders through Bevy's standard path.**
Terrain becomes ordinary `Mesh` assets under `StandardMaterial` (or
`ExtendedMaterial` extensions that preserve Bevy's PBR and shadow passes — the
pattern `ShadowedStandardMaterial` already established for crafts). Bevy owns
lighting, shadows, and visibility for opaque surfaces. **Solari is an evaluated
option, not a dependency**: baseline is standard PBR + CSM (+ 0.19 contact
shadows / SSR where they fit); Solari enters only via measured probe milestones.
This supersedes ADR-20260722T105147Z **part 2** (the deferral trigger and the
Mira-first in-repo prototype). Parts 1 and 3 of that ADR **stand and gain force**:
the tile is the surface authority (the probe's `TerrainTileProvider` is that
contract's first clean implementation), and every band below the package remains a
conditional refinement of its parent, never additive content.

**3. The vehicle is a standalone probe, which is what legitimizes overriding the
trigger.** The deferral's verify-queue precondition existed because reworking the
most load-bearing render subsystem atop ten unverified landings makes regressions
un-diagnosable. A clean-room repo dissolves that objection rather than overriding
it: there is no unverified stack beneath the probe. The probe runs at Earth scale
(6,371 km — harder than Thalos's 3,186 km) with `big_space` from the start, and
its M5 milestone produces the measured extraction plan back into Thalos.

**4. Earth-like/Thalos first, reversing airless-first.** The pretrained earth-like
model and its real-DEM data story exist today; the airless family is mid-campaign
(MIRA-1 at the L2 gate). MIRA-1 **finishes its remaining L2 gate evidence, then
pauses** — abandoning it one gate short wastes the campaign and the L2 pass is the
proof the Rust training pipeline works — but nothing downstream of L2 (MIRA-2's
whole-sphere bake, MIRA-3/4) starts. Airless becomes the second model family, per
the companion training-stack ADR's unified-architecture end state.

**5. Volumetrics and sky remain custom composites — the explicit carve-out.**
"Standard path" governs opaque *surfaces*. The `BodySky` atmosphere
(ADR-20260721T185221Z stands), volumetric clouds, the analytic ocean, and the
celestial sky are screen-space/volume composites Bevy has no equivalent for. They
couple to standard-path surfaces the way the W11 froxel design already specifies
(`color·T + L` applied per opaque fragment) — that in-flight design transfers.

**Graphics-sprint triage (consequence of 2 and 5):** work coupled to udlod or to
porting surfaces onto the custom spine is **frozen** (F4r spine port, F5r spine
receivers, F7, F8a, F8b, F9, W12r, TM1, BL-36's udlod-side half; GF-CAL shrinks to
calibrating survivors). The surviving composites (clouds, atmosphere, ocean,
plumes, celestial) continue, including in-flight wip. The `clean` track continues
as background — it is largely renderer-agnostic.

## Alternatives

- **Wait for the ADR-20260722T105147Z trigger.** Rejected. The trigger protected
  against (a) building a consumer on a moving producer contract and (b) landing on
  an unverified stack. Pairing terrain and renderer in one slice removes (a) — the
  contract and its producer are designed together; the standalone probe removes
  (b) by isolation. Meanwhile every udlod defect diagnosed (BL-34/35) is sunk cost
  on a layer now scheduled for deletion.
- **Mira-first prototype (the superseded plan).** Rejected as the *first* slice:
  earth-like has a pretrained model, released weights, and a real-DEM training
  corpus; airless has a mid-campaign model one gate short of L2. Mira's smaller
  integration surface was its argument, but the probe's isolation delivers the
  same de-risking without giving up the proven family. Mira follows as the second
  family.
- **Keep the F-series direction (crafts onto the custom spine).** Inverted, not
  merely rejected: the one-world goal is unchanged, but it is reached by moving
  terrain (and everything opaque) onto Bevy's universe rather than moving crafts
  onto ours. The craft side already lives there (`ShadowedStandardMaterial`); the
  custom spine's surface-shading half becomes end-of-life once the probe's
  standard-path fidelity is measured as sufficient.
- **Solari-first renderer.** Rejected: experimental, RT-hardware-gated, and
  streamed terrain tiles imply continuous BLAS rebuild costs nobody has measured.
  The probe measures the standard path first (its own stated non-goal: no custom
  replacement for Bevy PBR before the standard path is measured).
- **Continue polishing udlod (fix BL-34/35, land BL-34-seam incrementally).**
  Rejected as primary investment: the same grazing-silhouette defect class has now
  cost multiple sessions, udlod's architecture (shared-grid + height-atlas vertex
  displacement, off-path pipeline, custom queue ordering) is the root of both the
  debugging opacity and the two-universes debt, and its strongest features (tile
  cache, provider contract, coarse-first admission) transfer as *concepts* to the
  new tile contract.

## Consequences

- `thalos_udlod`, `BodyTerrainMaterial` and the terrain WGSL stack, and — pending
  probe measurements — the custom shadow rig and SSAO node become end-of-life.
  They keep running in Thalos until extraction lands; only defect-driven fixes, no
  new investment.
- BL-34/BL-35's 2026-07-22 parking is retroactively legitimized by this record.
  Their falsification evidence carries over to the new renderer as documented.
- The backlog gains an `ntr` track (primary); frozen gfx rows are annotated, not
  deleted; Q9 leaves *Decisions pending*.
- The MIRA visual slice (MIRA-V*) closes out: its shading knowledge (Hapke `w`
  fix, crater depth law, landmark registry) survives — the depth-law and provinces
  are producer-side and renderer-independent — but no further udlod-side airless
  calibration starts.
- terrain-diffusion's WorldClim climate channels open a path for learned climate
  conditioning to feed landcover/biomes, making the hand-built macro
  moisture/climate pipeline (TM-P1/P2/P3) a consumer of generated conditioning
  rather than a parallel authored system. Design-level for now; recorded in
  `ntr` open decisions.
- Risk accepted: the probe can diverge from Thalos runtime realities (structures,
  flatten pads, colliders, ocean coupling). Mitigation: the tile contract is
  shared, the probe's M5 extraction plan is a gate not a formality, and Thalos
  integration sequencing is owned by `ntr §6`.
- Risk accepted: `StandardMaterial` cannot express Hapke for airless bodies; the
  `ExtendedMaterial` pattern is assumed sufficient. If a probe milestone
  falsifies that, this ADR's part 2 gets a follow-up, not a silent fork.
- Reopening requires evidence from the probe's own budget gates (M5), not
  renewed preference.
