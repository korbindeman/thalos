# INC-20260724T001023Z — two "body-fixed" frames: tile shell rendered 132° from the camera on a tidally-locked moon

**Status:** fixed
**Area:** frame authority — `ViewAnchor` × real-space body grid × capture
framings × NTR-X1 tile renderer
**Predecessor:** INC-20260723T232652Z (the impostor-cover + landmark-metadata
findings of the same investigation; its "open successor question" is resolved
here)

## Symptom

After the impostor handoff fix, `THALOS_TILE_RENDERER=1` Mira still rendered
as a near-featureless sphere: no craters at any framing, no albedo provinces,
identical frames across renderer-state changes. Meanwhile every data probe
came back healthy — the landmark crater sampled −5,220 m at its centre on
both the f32 and f64 query paths at every LOD, and the built tile meshes
carried the full provider height range (verified to ±117 km under a ×20
debug exaggeration that still rendered glass-smooth).

## The decisive probe

A one-shot render-state log after first coverage: resident tile entities sat
**~1.6 Mm from the camera** — on the correct 869 km sphere, but rotated
~132° (law of cosines on 869 km / 879 km / 1.6 Mm) from where the camera
hovered. A rotated *complete* shell occupies the same spherical locus, so the
planet still looked whole: the camera's near field was covered by coarse
far-side tiles (smooth, near-uniform albedo — the "flat painted" read), while
the finely-refined crater tiles rendered on the far side. Every earlier
observation fit: the red-tint discriminator turned the sphere red (it *was*
our tile shell), and exaggeration changed nothing visible (the near-field
tiles were coarse ones).

## Root cause

The codebase had **two different "body-fixed" frames** and no single
authority:

- The **render/surface frame**: `transforms::surface_body_to_world_orientation_f64`
  — what the real-space body grid's rotation, `PreciseRotation`, the udlod
  terrain, and (by construction) the height sources and terrain packages use.
  For a tidally-locked moon this is the tidal-lock composition.
- The **raw ephemeris frame**: `BodyState::orientation` — what `ViewAnchor`
  (`update_view_anchor`), `AnchorBody::cam_world`, parts of `sun_shadow`, and
  ~10 body↔world conversion sites in the screenshot framings used.

For ordinary planets the two coincide, so every consumer worked on Thalos and
the split stayed invisible. Mira is a tidally-locked moon: the frames differ
by the full lock rotation. The tile renderer was the first system to consume
`ViewAnchor.cam_body` as a *geometric surface position* on a moon — its
selection eye and its tiles' parenting sat in different frames, displacing
the refined shell by the lock angle. The capture framings had the twin bug
independently (camera posed over `landmark.dir` through the ephemeris frame),
which is why fixing `ViewAnchor` alone moved the defect into the framing.

## Fix (one frame authority)

- `transforms::authored_lock_parent(&BodyDefinition)` — the ONE tidal-lock
  rule (moon + parent), now used by `spawn`'s `TidallyLocked` insertion and
  by every non-ECS consumer.
- `transforms::surface_orientation_authored(bodies, body_id, states)` — the
  surface orientation resolvable from authored data + evaluated states alone.
- `ViewAnchor` resolves `cam_body`/`cam_dir` in the surface frame and carries
  `lock_parent`; `AnchorBody::cam_world(states)` re-projects through the same
  authority (signature changed from `(&BodyState)`).
- `sun_shadow` re-projects the anchor with the authority and takes the
  anchor's surface-frame nadir for its terrain-height probe.
- Screenshot framings (ocean/hub site, airless daylight survey, EVA context,
  dry-belt, cloud site, F8 saved-perspective save + replay) converted to the
  surface frame.

## Prevention / standing rules

- **"Body-fixed" must mean the surface frame, everywhere.** Any
  `state.orientation * dir` / `.inverse() *` conversion of a *surface*
  position or direction is wrong on a tidally-locked body. New conversions go
  through `surface_orientation_authored` (or the grid's `PreciseRotation`),
  never `BodyState::orientation` directly. Remaining known ephemeris-frame
  uses are deliberate (orbital mechanics, sun-lock quantization where any
  consistent frame cancels, the disc framing's fallback seed).
- **A "consistently wrong pair" hides until a third consumer arrives.** The
  framing and the old `ViewAnchor` agreed with each other (both ephemeris),
  so the mira presets *looked* self-consistent while both were 132° off the
  rendered ground. When two systems must agree on a frame, they must derive
  it from one function, not implement it twice.
- The probe ladder that cracked it, for reuse: data-side transects (exonerate
  content) → mesh-side range telemetry (exonerate geometry) → debug height
  exaggeration (make displacement undeniable) → render-state entity log
  (positions/visibility — the step that actually localized it). The first
  three all passing while the frame stays wrong points at placement, not
  content.

## Recurrence tells

- A moon whose surface renders smooth/uniform while data probes show relief:
  suspect frame mismatch before content.
- A view-anchored system (scatter, shadows, tile streaming) refining around a
  point ~a fixed great-circle angle away from the camera on a moon only.
- `mira-*` presets framing empty ground while landmark transects verify.
