# Camera optics and photographic capture

**Status:** CAM-1 landed compile- and capture-clean 2026-07-29; interactive
freecam/viewpoint verification remains in `docs/backlog.md`.

Thalos should be as satisfying to photograph as it is to fly. The camera
therefore needs a coherent photographic model that works in the interactive
game, saved viewpoints, headless stills, and later deterministic video. Freecam
is the first editing surface because it already owns deliberate framing, but it
is not the owner of the optics.

## 1. Responsibilities

Three concerns stay separate:

- A **camera rig** owns pose and movement. Flight orbit, freecam, god view, a
  saved viewpoint, and a future camera track are rigs.
- **Camera optics** own how the real `ShipCamera` sees: initially focal length,
  sensor gate/filmback, and sensor crop; later focus distance and aperture. A
  rig may seed optics when it takes ownership, but it does not keep a parallel
  lens model.
- **Photo mode** owns presentation and capture UX: clean overlays today;
  composition guides, focus picking, output framing, and capture controls
  later. It edits the same optics rather than growing a photo-only camera.

The real `ShipCamera` remains the one rendered view. Terrain streaming,
view-dependent detail, atmosphere, shadows, post effects, saved viewpoints, and
headless capture all continue to observe that camera through their existing
authorities.

## 2. Lens and framing contract

The user-facing unit is **35 mm/full-frame-equivalent focal length**:

- reference filmback width: **36 mm**;
- gate fit: **horizontal**;
- valid initial editing range: **12–400 mm**;
- focal length is canonical; Bevy's vertical FOV is a derived projection value.

For the active sensor-window aspect ratio `a = width / height` and focal length
`f` in millimetres:

```text
horizontal_fov = 2 atan(36 / (2 f))
vertical_fov   = 2 atan((36 / a) / (2 f))
```

Horizontal fit makes a lens preserve horizontal composition when the sensor
aspect changes. A wider sensor window reveals more vertically; it does not
quietly turn a 50 mm lens into another lens. Reducing the sensor gate or
selecting a crop window narrows the ray cone and is therefore a camera/framing
change.

Output resolution is separate. A 1920×1080 and 3840×2160 capture sample the same
16:9 sensor window at different fidelity and must have identical projection.
Interactive **render scale** is the same idea: the 3D main target is
`window_physical × scale`, then upscaled to the swapchain. The HUD camera
clears to transparent and alpha-blends over that blit so undrawn pixels do
not cover the scene. Projection and window-logical picking stay on the
native window. Capture stays at 1.0.
An output aspect that differs from the active sensor window must not silently
change the camera: the caller supplies an explicit sensor crop/fit policy, or
capture rejects the mismatch.

At 16:9, the existing Bevy default of 45° vertical FOV is approximately
**24.4 mm** under this convention. Existing viewpoints migrate to their
equivalent focal length using their recorded viewport, so the first optics
change is framing-neutral.

Focal length changes framing, not perspective. The familiar compressed
telephoto composition comes from moving the camera back while increasing focal
length. A future dolly-zoom tool may couple those operations explicitly; the
lens control itself never moves the camera.

## 3. Ownership and transitions

A shared optics component on the `ShipCamera` is the one live authority. It
holds the base focal length and, as later slices land, focus/aperture state.
Projection synchronization is the sole writer that converts the optics plus
the current viewport into `PerspectiveProjection::fov`.

Rig transitions obey these rules:

- F4 entry converts the currently presented projection to the equivalent focal
  length before freecam takes control, so activation does not jump.
- Freecam edits the shared base lens. Its panel is simply the first control
  surface.
- Leaving freecam hands optics policy back to the receiving rig. A normal
  flight camera may restore its authored/default framing; lens state does not
  leak into a rig that did not opt into it.
- Applying a saved viewpoint enters freecam at that exact pose and saved lens.
- The existing `Z` hold remains a spring-loaded telephoto aid, expressed as an
  effective focal-length multiplier rather than by dividing FOV. The base lens
  remains stable when it is released.

Every projection change must participate in the existing camera-cut/history
contract. Smooth lens animation may be treated as camera motion by consumers
that support it; a consumer that cannot reproject an FOV change must reject its
history rather than smear the frame.

## 4. Persistence and capture

The next viewpoint-catalog schema stores typed lens and sensor descriptions,
initially:

```text
lens:
  model: full-frame-horizontal
  focal_length_mm: 35
sensor:
  gate_width_mm: 36
  aspect: [16, 9]
  crop: full
```

Readers migrate the current `vertical_fov_rad + viewport` representation with
the inverse of §2, retaining the legacy viewport's aspect while discarding its
absolute pixel dimensions. Writers emit only the new canonical representation
once the catalog migration is complete. F8, F9, scripted capture framings, and
headless replay all route through the same conversion/application core.

`CaptureRequest.camera` has a typed lens override rather than another environment
variable. A capture receipt records the base and effective focal length, lens
model, sensor gate/crop/aspect, derived vertical FOV, output extent, and
fidelity. Consequently a human and an agent can reproduce the same composition
without reverse-engineering a Bevy projection.

## 5. Freecam first slice

The initial UI belongs in `freecam::panel`:

- a prominent `24 mm`-style value;
- a logarithmic slider over 12–400 mm;
- familiar reference marks at 14, 24, 35, 50, 85, 135, and 200 mm;
- horizontal and vertical angle-of-view readout;
- `Z` shown as a temporary effective lens when held.

The mouse wheel remains freecam translation speed. A lens wheel modifier can be
added only if it is discoverable in the panel and does not collide with UI
input ownership.

In CAM-1 the workflow is deliberate and simple: adjust the lens in freecam,
then press `P` to hide every overlay for the clean frame. Photo mode gains a
richer, hideable setup overlay only in CAM-4; that panel edits the same optics
and never becomes a second state resource.

## 6. Aperture, focus, and exposure

These are staged deliberately because they touch different renderer contracts.

### Focus and aperture

Focus distance is in world metres and may be set manually or by picking a
visible surface/object through the real camera. Aperture is expressed as an
f-number on the same full-frame model. Bevy 0.19's depth-of-field pass is the
first implementation candidate, but adoption requires evidence that:

- its depth sees every relevant opaque surface;
- its render-graph slot composes correctly with Thalos atmosphere, ocean,
  clouds, celestial sky, and other fullscreen passes;
- sky/infinite depth and maximum circle-of-confusion are bounded cleanly;
- a disabled effect is genuinely free enough for normal play.

The first aperture UI is **aperture priority**: changing aperture changes depth
of field while brightness is held. This lets optical composition arrive before
shutter motion blur or ISO noise exists, without pretending that an isolated
f-number is a complete exposure model.

### Exposure

The existing `CameraExposure` remains the one brightness authority. A
photographic control first adds explicit exposure compensation to that
authority:

```text
total_gain = scene_distance_gain * 2^exposure_compensation_ev
```

Bevy histogram auto-exposure does not return. Aperture, shutter, and ISO may
drive the same authority later if all three have meaningful consequences:
aperture for depth of field, shutter for motion/long exposure, and ISO for
sensor noise. Until then, presenting a decorative exposure triangle would be
misleading.

The existing solar-distance gain and its grain response are scene adaptation,
not automatically equivalent to ISO. User exposure compensation likewise does
not add grain unless a later sensor model assigns the gain to ISO explicitly.

## 7. Delivery slices

Backlog status and priority live only in `docs/backlog.md`.

- **CAM-1 — Shared optics authority and freecam control.**
  Full-frame-horizontal focal length, sensor gate/crop, projection
  synchronization, freecam panel, spring zoom, viewpoint migration, typed
  capture persistence, and receipts.
- **CAM-2 — Focus and aperture.** Focus picking/manual distance, aperture
  priority, depth-of-field integration, composite ordering, and exact
  viewpoint/capture persistence.
- **CAM-3 — Photometric controls.** Manual exposure compensation folded into
  `CameraExposure`; only then evaluate physical shutter/ISO and motion/long
  exposure.
- **CAM-4 — Photo workspace.** Composition guides, focus visualization,
  aspect/output framing, capture actions, and a hideable setup overlay. Video
  tracks and frame sequencing remain CAP-4 in `docs/development/capture.md`.

## 8. Verification

The agent-owned CAM-1 gates are satisfied:

- conversion tests cover focal length ↔ vertical FOV across common sensor
  aspects;
- current catalog viewpoints migrate with unchanged derived framing;
- changing output resolution at the same sensor aspect preserves the
  projection;
- changing sensor aspect preserves horizontal angle of view;
- a mismatched output aspect cannot silently reframe the sensor window;
- F9 save → F8 apply → headless replay reports and renders the same lens;
- a headless matrix at 24/35/50/85/135 mm contains valid receipts and visibly
  different framing from one unchanged pose.

The matrix is under `artifacts/visual/runs/cam-1-lens/`. All five receipts
record the requested focal length, a 36 mm horizontal gate, the same 16:9
sensor window and 1920×1080 output, exact source relation, and
`workspace_matches: true`.

CAM-1 remains `verify` until the user confirms freecam entry/exit has no
unexpected jump; slider, angle readout, and spring zoom agree; and an
F9-save → F8-apply round trip restores the same framing.

Later slices add matched focus/aperture captures, composite regression captures,
and exposure comparisons with every other camera and scene input held fixed.
