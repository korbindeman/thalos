# INC-20260810T195226Z: Kòrsou viewer status closed an Update schedule cycle

- **Status:** Fixed
- **Date:** 2026-08-10
- **Severity:** startup panic
- **Surface:** interactive Kòrsou startup

## Symptom

Kòrsou panicked while Bevy initialized `Update`, reporting twelve equivalent
before/after cycles through the `Input`, `Movement`, and `Locate` sets. Release
builds omitted system names, so the panic did not identify the application
systems involved.

## Mechanism

The shared viewpoint schedule orders `Input` before `Apply`. Kòrsou then added
these valid local dependencies:

```text
Viewpoint Input -> camera Movement -> place Locate
```

`project_viewer_status` needed the post-movement place result for the Location
row, but the same system also projected the settings-modal gate consumed by
viewpoint input. It was consequently declared both after `PlaceSet::Locate`
and before `ViewpointSet::Input`, closing the cycle:

```text
Viewpoint Input -> camera Movement -> place Locate -> viewer status -> Viewpoint Input
```

This was not duplicate plugin registration or a Bevy scheduler defect. A
minimal schedule using the real shared viewpoint plugin and Kòrsou's set edges
reproduced the failure deterministically.

## Fix

The two lifecycle responsibilities are now separate systems:

- `project_viewer_interaction` runs after settings-menu input and before
  viewpoint input, publishing only the modal interaction gate.
- `project_viewer_display` runs after camera movement and place location,
  publishing active/visible state, location, and altitude before viewpoint UI.

The regression test `camera_and_viewpoint_schedule_is_acyclic` initializes the
composed `Update` schedule directly. It failed before the split and passes
afterward without launching a renderer.

## Prevention and recurrence tell

A resource can contain both early input gates and late display data, but one
system must not project fields whose correct ordering belongs on opposite sides
of the frame. Any startup panic whose cycle reads `Input -> Movement -> Locate
-> Input` is a recurrence; run the schedule-initialization test before looking
at rendering or window setup.
