# INC-20260725T195500Z-tiles-rode-an-f32-body-rotation: the ground jittered a decimetre against everything built on it

- **Date:** 2026-07-25 · **Surface:** `just game`, following an ascending craft over the spaceport

## Symptom

The space center's flat structures — runway asphalt, taxiways, roads, aprons — flickered in
and out of the terrain. The user's three discriminating observations were the whole diagnosis:

- it happens **while following an ascending craft**,
- with **freecam and the sim paused it does not happen**,
- with the **sim paused it is stable — but "either submerged or on top, it doesn't settle"**.

That last one rules out z-fighting outright. Z-fighting is two surfaces at *equal* depth
resolving per-pixel; it produces a stipple and it does not freeze into "entirely buried" or
"entirely fine" when you pause. A whole-base, uniform, binary flip that freezes when the sim
freezes is a **placement** error whose value is re-rolled every frame, and the only per-frame
input that changes while the sim runs and stops when it pauses is the body's spin.

## Root cause

Tiles were spawned as `ChildOf` the body's **rotating** big_space grid. A tile's body-fixed
origin has magnitude ≈ the planet radius, and big_space rotates that multi-Mm offset into
world space with the grid entity's `Transform.rotation` — an **f32** quaternion
(`real_space.rs`, `transform.rotation = rotation.as_quat()`).

f32 quaternion ULP at Thalos's 3,186 km, measured over the spin cycle:

| | metres |
|---|---|
| mean | 0.055 |
| p95 | 0.165 |
| worst | 0.256 |
| change frame to frame | 0.055 |
| **runway asphalt lift** | **0.120** |

The error *straddles* the paving lift. Each frame the ground landed somewhere in that band, so
it crossed above and below the paving — submerged, then fine, then submerged. Pause the sim
and the orientation stops changing, so it freezes wherever it happened to be: stable, and
wrong half the time. Higher warp spins the body further per frame and makes it worse.

**This trap was already documented in this repo, twice, and every other consumer already
dodged it.** `update_runway_transform` exists specifically for it: anchoring the runway as a
fixed-cell child of the rotating body grid "makes big_space rotate its multi-Mm cell offset by
an f32 quaternion, which jitters frame-to-frame at high warp… ≈ decimetre ULP at planet
radius". `transforms.rs` calls the same quantity "a flickering decimetre". udlod's terrain
dodges it via `PreciseRotation`. The tile renderer was the one ground consumer that took the
natural-looking parenting — and its module doc asserted the *opposite*, that co-rotating
children keep planet-scale precision. The f64 *composition* big_space does is irrelevant when
the rotation being composed is only f32-precise.

## Why the earlier measurements all came back clean

Two rounds of offline tests (`meshed_ground_lands_on_the_pad_plane`) proved the tile mesh sits
on the pad's tangent plane to under 1 cm at every level, and reading the flatten, the runway
frame and the connection frame showed all three sharing one anchor. All of that was true. The
error is not in the tile, the pad, or the paving — it is injected *after* all of them, by the
body→world rotation at render time. **A geometry check in the body frame cannot see a defect
in the frame transform**, which is why the fault survived being "ruled out" twice.

## Fix

Tiles are now root-grid children placed in f64 every frame from the body's pose, exactly as
`update_runway_transform` places the runway:

- `TileEyeTarget` carries the body's f64 position and its `surface_orientation_authored`
  rotation — the same shared surface frame the body grid, height sources and capture framings
  resolve (so a tidally-locked body can't land in a different frame, INC-20260723T232652Z).
- `TileBodyOrigin` keeps each tile's body-fixed origin; `stream_tile_terrain` re-places every
  resident tile from one pose value per frame, so tiles that landed this frame and tiles
  already up can never be a frame apart.
- The f32 `Transform.rotation` now acts only on in-tile vertex offsets: ≤ 0.04 m of ULP for a
  `MIN_LEVEL` tile and micrometres near the surface.
- The driver chain is pinned `.after(sync_solar_system_state).before(TileStreamSet)`. A frame
  of slip here would slide the ground metres against the base — 232 m/s of surface speed at
  Thalos's equator.

`f32_body_rotation_cannot_place_ground_under_a_runway` keeps the arithmetic executable: it
asserts the f32 path exceeds the paving lift and the f64 path does not.

## The tell

- **A flat drape that is binary (buried or fine), uniform over the whole footprint, and
  *freezes* when the sim pauses is not z-fighting.** Z-fighting stipples and keeps stippling.
  Binary + uniform + frozen-on-pause = a per-frame placement error; look for what moves only
  while the sim runs.
- **f32 rotation × planet radius = decimetres.** Any near-surface geometry parented to a
  rotating body grid inherits that. Place it in f64 against the root grid, or read
  `PreciseRotation`.
- A body-frame geometry test cannot exonerate a frame-transform bug. When measurements keep
  coming back clean, check whether they are all measuring in the same frame.
