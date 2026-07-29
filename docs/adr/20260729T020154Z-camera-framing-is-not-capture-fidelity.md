# ADR-20260729T020154Z-camera-framing-is-not-capture-fidelity: Separate camera framing from capture fidelity

- **Status:** Accepted
- **Date:** 2026-07-29
- **Narrows:** ADR-20260729T020148Z, whose “current viewport aspect” is replaced
  as projection input by the saved sensor-window aspect. Output extent remains
  capture metadata, not camera state.

## Context

Saved viewpoints currently carry the window pixel dimensions present when the
player pressed F9. Headless replay then used those dimensions as the render
extent. That accidentally made a 4K desktop part of the identity of a camera
bookmark, drove ordinary agent captures at 4K, and encouraged the inverse
workaround—reducing the viewpoint resolution to protect the machine—even when
the resulting evidence was too blurry.

The upcoming camera system also needs real photographic framing. Pose and one
field-of-view number cannot distinguish changing lens focal length, changing
the sensor gate, or selecting a crop window. In particular, a smaller sensor at
fixed focal length narrows the ray cone and frames a smaller area; it is a
camera change, not a lower-resolution image.

## Decision

A viewpoint owns replayable camera and world state: pose, lens/projection,
sensor gate/filmback and crop, scene/body, and canonical time. It never owns the
capture output pixel count.

Capture independently owns a named fidelity profile and any explicit output
extent override. Fidelity controls the pixel budget and renderer-internal
quality/convergence settings. Changing fidelity preserves normalized sensor
rays and composition; changing sensor gate, crop, focal length, or pose changes
the viewpoint.

The standard agent workflow may promote one unchanged viewpoint through
`standard` → `high` → `reference`. Receipts and comparison manifests record the
requested profile and every effective setting. The legacy viewpoint `viewport`
field is migration input: retain the aspect needed to reconstruct the sensor
gate, discard its absolute pixel dimensions.

Effective render extent may remain part of the persistent host's boot context
because it sizes GPU resources. That implementation constraint does not make it
viewpoint identity.

## Alternatives

- **Keep output width/height in viewpoints** — rejected because identical
  framing at another fidelity becomes a different viewpoint and desktop
  resolution silently dictates automation cost.
- **Make one globally safe resolution** — rejected because it was visibly too
  blurry for some judgments and still cannot express intentional promotion.
- **Crop an already-rendered wide PNG** — rejected as the camera model: it
  wastes work and gives viewport-sensitive LOD, shadows, screen-space effects,
  and temporal reconstruction a different view from the final image.
- **Expose only raw quality knobs** — rejected because agents need a simple,
  reproducible choice and manifests need a stable semantic label.

## Consequences

The camera-system schema remains free to choose physical units and exact field
layout, but it must preserve this authority boundary. Capture gains typed
fidelity tiers plus promotion guidance. Existing catalogs require a versioned
migration, and current 1080p safety fitting is temporary scaffolding rather than
the final fidelity model.
