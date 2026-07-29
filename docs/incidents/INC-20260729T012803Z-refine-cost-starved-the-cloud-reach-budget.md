# INC-20260729T012803Z — Refinement cost starved the cloud reach budget; silhouettes broke into lace

**Symptom.** From altitude (`tex_look` viewpoint, ~8 km, grazing view over a
cumulus field), distant cloud tops seen **against terrain** carried a thin,
jagged dark-teal fringe tracing every silhouette — a "lace" of 1–3 px filaments
from the mid-distance band outward. Silhouettes against the sky were clean.
Separately, the same framing rendered every distant cloud over land
increasingly translucent ("washed out").

**Mechanism (two independent defects).**

1. *The lace: the near march truncated short of the frontier both tiers agree
   on.* `cloud_march_stop_m` — the closed-form reach frontier that the far tier
   uses to place its complementary fade-in — integrates the **broad** step
   ladder only. But the march loop charged every step, broad or fine, to one
   `ray_step_limit` counter, and refinement costs ~5 fine steps per broad hit
   plus a rewind. A ray grazing a silhouette skims low-density fringe for
   kilometres: it refines near-constantly, never saturates transmittance, and
   so exhausted the budget far **before** the closed-form frontier — no
   reach-fade (that is keyed to the closed form), no far-tier cover (ownership
   says the near tier reached further), just an abrupt truncation. Interior
   rays hit dense cloud and break cheaply on the transmittance floor, so only
   silhouettes failed: a lace of holes showing the terrain behind.
   Fix: the reach budget counts broad probes only, with a separate hard
   iteration cap for TDR safety, and the loop breaks at the frontier (density
   is fully dissolved there anyway).

2. *The washout: `near_visibility` divided by the whole band chord.* The
   composite's scene-occlusion partition estimated "fraction of cloud in front
   of the scene" as `(scene_t − cloud_near) / (band_far − cloud_near)` with
   `band_far` = the ray's **entire cloud-shell chord** — hundreds of km on a
   grazing ray, for a cloud a few km deep. Every distant cloud seen against
   terrain was multiplied toward transparency; against sky the function
   early-outs at 1.0, which is why the defect tracked the terrain background
   exactly. Fix: bound the assumed extent by the coarsest cell period (5.4 km).

**The tell for a recurrence.** Cloud artifacts that appear only where the
background is terrain (never sky) implicate the composite's scene partition;
artifacts that hug silhouettes only beyond a stable distance band and shrink
when `THALOS_SCREENSHOT_CLOUD_QUALITY=reference` raises the step budget
implicate the reach frontier. Note the inversion of
INC-20260726T012035Z's rule of thumb: there, "more steps don't help ⇒ stride
not budget"; here more steps *did* help, because since ADR-20260726T000929Z
step count buys **reach**, not resolution.

**Falsified on the way (do not re-derive):**
- Temporal/sparse reconstruction — `RECONSTRUCTION=raw` reproduces the lace.
- The far tier — `CLOUD_TIER=near-only` reproduces it; `far-only` is
  lace-free.
- The floor/round reference-texel mismatch between `near_visibility` and
  `sample_near_cloud` — aligning them changed nothing visible.
- Cloud shadows on terrain — lace pixels measure as *undarkened* background
  terrain (holes), not darkened terrain.

**Context.** The budgets were sized when `CLOUD_MARCH_MIN_STEP_M` was 600 m
("176 steps reach the full 300 km cap"). INC-20260726T012035Z halved the floor
to 300 m for correctness, which silently halved the distance every budget
buys; that put the frontier (~90 km for this framing's shell entry) in the
middle of the visible cloud field, where defect 1 could be seen at all.
