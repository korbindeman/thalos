# ADR-20260729T020148Z-one-camera-optics-authority: one full-frame-horizontal camera optics authority

- **Status:** Accepted
- **Date:** 2026-07-29

## Context

Freecam already stores an unzoomed vertical FOV, saved viewpoints persist a
vertical FOV plus viewport, headless capture reapplies that projection, and
photo mode names focal length/aperture/exposure as future work. Adding a focal
length slider directly to freecam would create the first of several competing
lens models. The representation also has to survive aspect-ratio changes and a
catalog schema migration without shifting authored compositions.

## Decision

The real `ShipCamera` gets one optics authority, separate from whichever rig
owns its pose. Freecam is the first editing surface, while viewpoints, photo
mode, headless capture, and later camera tracks read and write the same state.

The canonical framing unit is a 35 mm/full-frame-equivalent focal length against
a 36 mm-wide filmback with horizontal gate fit. Vertical FOV is derived from
focal length and current viewport aspect. Existing vertical-FOV viewpoints are
migrated through their recorded viewport so their framing remains unchanged.

Focus distance and aperture extend this optics state later. Exposure controls
extend the existing single `CameraExposure` authority; they do not reintroduce
Bevy histogram auto-exposure or a photo-mode-only exposure path.

## Alternatives

- **Keep vertical FOV canonical and show a calculated “equivalent” focal
  length** — rejected because the displayed lens would change when aspect ratio
  changes, making saved/captured lens metadata misleading.
- **Use vertical full-frame fit** — rejected because common wide outputs would
  reveal more horizontally and alter the dominant composition. Horizontal fit
  gives a stable, explicit framing axis.
- **Use diagonal equivalence** — rejected because it preserves neither
  horizontal nor vertical composition across aspect changes and is harder to
  reason about in capture tooling.
- **Put focal length in `FreeCam`** — rejected because applied viewpoints and
  headless captures already bypass or seed freecam state, and photo mode/video
  would need parallel copies.
- **Adopt Bevy physical exposure wholesale** — rejected because Thalos already
  has one intentional solar-distance exposure authority. A second multiplier
  would recreate the exposure conflict removed by graphics-fidelity F2.

## Consequences

The next viewpoint schema must migrate persisted FOVs, and capture receipts need
enough lens/aspect metadata to prove exact framing. Every camera rig must state
whether it inherits, seeds, or restores optics when it takes ownership.

Horizontal framing remains stable across aspect ratios, while the vertical crop
changes. Users who need an exact final crop must therefore save/request the
viewport aspect as well as the lens. The current 45° vertical framing migrates
to roughly 24.4 mm at 16:9 rather than changing visually.

The decision creates a clean route to physical depth of field, but it does not
commit Thalos to decorative shutter/ISO controls before motion blur and sensor
noise make those parameters meaningful.
