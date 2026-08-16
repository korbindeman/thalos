# ADR-20260809T222136Z-viewpoint-catalog-is-frame-tagged-viewer-data: Viewpoints live below capture and name their spatial frame

- **Status:** Accepted
- **Date:** 2026-08-09
- **Supersedes:** ADR-20260724T221858Z-unified-viewpoint-registry

## Context

The unified game registry correctly gave saved and scripted views one public
catalog, but its model lived in `thalos_capture_protocol` and assumed every
saved pose was authored-body-fixed. Kòrsou proved two constraints that model
could not express: an interactive application needs viewpoints without gaining
the capture capability, and a projected-local pose must not be mistaken for a
planetary body-fixed pose.

Keeping a second Kòrsou schema, store, and manager would make save/replay
behavior diverge again. Depending on the capture protocol from the light
interactive bundle would also violate the capability boundary.

## Decision

The Bevy-free viewpoint model lives in `thalos_render_model`; capture protocol
reexports it for source compatibility. Schema `thalos.viewpoints.v3` tags every
saved pose with one explicit frame:

- `authored-body-fixed` carries body, canonical spawn, hub flag, and simulation
  epoch;
- `projected-local` carries the application-owned spatial-reference identity.

`thalos_viewer` owns the one validated store, atomic write path, CRUD manager,
F8/F9 interaction, exact pose/optics snapshot, and apply request. Applications
own only adapters that capture and apply their frame. The game retains body
state/floating-origin projection and scripted diagnostic executors. Kòrsou
retains its DEM bounds and planar/ellipsoid projection and rejects foreign
frames.

Each application may have a different catalog file because the authored worlds
are different, but both files use the same schema and implementation. Legacy
game v1/v2 and Kòrsou v1 yaw/pitch data migrate on read; all writes are v3.

## Consequences

The light graph contains no capture package. Saved roll and physical optics now
round-trip in Kòrsou, and a frame mismatch fails explicitly instead of posing a
plausible camera in the wrong space. The game no longer carries its separate
egui manager or quick-save implementation. Adding another interactive spatial
adapter requires a snapshot/apply adapter, not another catalog or UI.
