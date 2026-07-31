# INC-20260731T004704Z — craft-local shadow gate read the craft, not the view

**Symptom.** All building/structure/tree shadows gone at a surface view (hub
god view, ship view parked at the pad after a flight), while the player craft
kept its own shadow. Reported as "the building shadows are gone".

**Mechanism.** `update_sun_shadow_camera`'s craft-local gate (self-shadow-only
mode above `SHADOW_MAX_ALTITUDE_M`) computed the view altitude from the raw
`ShipCamera` entity's `(CellCoord, Transform)`. That entity is written by
several drivers at different points in the frame: the flight follow-cam posts
it at the **craft**, and the god-view / capture drivers re-pose it at the
**view** later. The gate samples in `PostUpdate`, between those writers, so in
any mode where the view is not the craft it read the craft. The space-center
hub parks its placeholder craft in a 200 km orbit — the gate saw 200 km at a
600 m god view, latched craft-local mode, parked cascades 1–3, and every
non-craft shadow vanished. The same read is why the mode stuck after returning
from orbit to the pad. This is precisely the craft-anchoring failure
`view_anchor.rs` was built to end (its module doc names the hub as the
cautionary tale); the cascade *centre* already used the anchor, the *gate* did
not.

**Fix.** The gate (and the nearest-body/altitude selection feeding it) now
derives the view position from `ViewAnchor::cam_world` when the anchor is
resolved, falling back to the raw camera query only on pre-anchor boot frames.

**Tell.** `thalos::diagnostic::shadow` `stability_gauge` with
`active_cascades = 2` (craft-local) while the sky lane's `environment_paint`
reports a surface `altitude_m` for the same span — or directly:
`gate_alt_m` (added in the same change) disagreeing with the view altitude by
orders of magnitude, e.g. `gate_alt_m = 200004.5` at a 636 m god view.
