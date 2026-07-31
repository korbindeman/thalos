# INC-20260731T011523Z — cascade cull frustum built from the default ±1 m area

**Symptom.** Small casters — buildings, storage tanks, runway posts — cast no
shadow while terrain (and, in some sessions, tree tiles) still did. Rendering
otherwise perfect; no error anywhere; exit 0 (BL-20 class).

**Mechanism.** `update_sun_shadow_camera` replaces each cascade camera's
`Projection` every frame with
`OrthographicProjection { scaling_mode: Fixed{..}, ..default_3d() }`.
`default_3d()` leaves `area` at a **±1 m placeholder**, and
`get_clip_from_view()` reads `area`, not the scaling mode. Only Bevy's
`camera_system` recomputes `area` — and it is **unordered against the rig's
write**. On frames where `camera_system` ran first, `update_frusta`
(post-`Propagate`, reading the live projection) built a **two-metre culling
frustum** for every cascade view: km-scale terrain-tile caster twins still
intersected it, but every small caster was culled out of its own shadow map.
Rendering stayed correct because camera extraction uses
`camera.computed.clip_from_view`, the matrix `camera_system` cached from the
*previous* replace — so the maps were framed right and simply missing the
props. Which way the race fell was executor-order luck; the 2026-07-31
scheduling change (`.after(CellCoord::recenter_large_transforms)`) settled it
on the losing side consistently, which is when the symptom was reported.

**Fix.** Seat `area` at the write site: build the `OrthographicProjection`,
call `CameraProjection::update(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)` (exact for
`ScalingMode::Fixed`, arguments ignored), then store it. The live projection
never carries the placeholder, so `update_frusta` is correct regardless of
ordering.

**Tell.** Shadows present for terrain relief but absent for props at any sun
elevation; `stability_gauge.cascadeN_visible` (per-cascade post-cull mesh
counts, added during this diagnosis) staying near the resident caster-tile
count while structures stand in frame; and the definitive splitter — a prop
given `NoFrustumCulling` casts again while its neighbours don't.

**Rule.** When hand-writing a whole `Projection` component, never leave
`area` at the default: any consumer that reads the live projection before
`camera_system` runs sees a 2 m box. Call `update()` yourself.
