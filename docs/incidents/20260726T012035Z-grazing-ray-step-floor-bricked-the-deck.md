# INC-20260726T012035Z — A cost-only step floor bricked every deck viewed edge-on

**Symptom.** From at or below deck altitude, looking roughly level, the cloud
deck rendered as a field of **horizontal bricks** — dozens of hard-edged bars
across the whole mid-distance band, with the sky showing through between them.
The user read this as two defects ("straight artifacts cutting through" and
"we can easily see through them"); they are one, because the gaps between the
bars *are* the see-through. Repro:
`THALOS_SCREENSHOT_CAMERA_ALTITUDE=700 THALOS_SCREENSHOT_LOOK_ELEVATION=3`
on `cloud-cruise`, evidence in `artifacts/visual/runs/cloud-ghost-level/`.

**Mechanism.** `CLOUD_MARCH_MIN_STEP_M` (600 m) is a pure *cost* floor under the
footprint step law, and it overrode the physically-correct step at exactly the
ranges that matter. At 30–60 km a pixel covers 35–70 m, so the honest
footprint-derived broad step is ~100 m — the floor was forcing 600 m. A grazing
ray climbs almost nothing per step, so a 600 m stride through a ~1.15 km deck
under-resolves the density along the ray, and the stride's phase relative to the
deck varies smoothly with screen-y → banding. Dropping the floor to 300 m clears
it; 150 m is also clean.

**The tell.** The bars are *horizontal* and appear only at grazing incidence, and
they survive every change to sampling *count*. If more samples don't help but a
smaller **step** does, the defect is stride length, not budget.

**Falsified first — do not re-derive these:**

- **The far tier / ownership.** `THALOS_SCREENSHOT_CLOUD_TIER=near-only` is
  pixel-identical to the composite at this framing.
- **Step count, sparse scheduling, temporal reconstruction.**
  `THALOS_SCREENSHOT_CLOUD_QUALITY=reference` reproduces it identically. Note the
  trap: since ADR-20260726T000929Z the step *size* comes from the footprint law,
  so `reference` raises the step COUNT and changes nothing about the stride —
  this discriminator is much weaker than it looks and nearly sent the
  investigation the wrong way.
- **Strata cube texel reconstruction.** Forcing the marcher's strata fetch to
  mip 2 (16× blurrier) changed nothing.
- **Weather cube texel reconstruction.** Same test on `sample_weather`: nothing.
- **The strata cube's vertical reconstruction** was a *separate, real* defect
  found on the way (`cloud_surface_shape` was piecewise linear, so its slope
  broke at the four knots and a deck viewed edge-on rendered four shelves per
  cloud). Fixed with a C1 Catmull-Rom reconstruction in all three lockstep
  sites; it visibly cleaned the near field but left the distant band bricked, so
  it is not this incident's cause.

**Why now.** Nothing masked either defect before the cell field landed
(ADR-20260725T222409Z): the periodic Cartesian volume used to vary down the
column and along the ray. The cell field is a *column* field by construction, so
the strata reconstruction and the march stride became the only vertical
structure — and both were visible immediately.

**Cost.** Cruise GPU on the development 4070 Ti at 1280×720: 600 m → 1.98 ms
mean / 3.33 p95; **300 m → 2.36 / 4.38**; 150 m → 2.74 / 5.12. 300 m is the
accepted point — the artifact is a correctness defect and CLOUD-0's 3.5 ms p95
target is explicitly provisional. This is the knob to move if the budget is
re-tightened, and it must move with a matching capture at this framing.

**Standing rule.** A step floor that exists for cost must never be coarser than
the sampler's own footprint-derived step at ranges the player looks at. If it is,
it is not a floor — it is an undocumented LOD, and grazing geometry will find it.
