# ADR-20260801T070234Z-permanent-widget-launcher-and-floating-workspace: HUD widgets launch from a permanent panel into independent floating windows

- **Status:** Accepted
- **Date:** 2026-08-01

## Context

ND and TRAJ began as the contents of one top-right MFD slot. The slot made
contextual selection possible, but it also made the instrument catalogue and the
instrument surface the same object: exactly one widget could exist, opening a
different one replaced the current one, and the slot's position was fixed.

That is the wrong interaction for instruments a pilot may want together. ND and
TRAJ answer different questions, and future docking, transfer, systems, and
targeting displays make a tab row increasingly hard to discover and increasingly
expensive in screen space. Hiding the catalogue behind a separate layout-edit
mode would preserve the same accessibility problem.

## Decision

The normal flight HUD has one compact, **permanent widget launcher panel**. It is
always reachable in the flight HUD (except when the HUD as a whole is suppressed,
such as photo mode or a non-flight context), is not itself draggable or closable,
and presents the widgets available to the active craft.

Selecting a launcher entry opens or focuses an independent **floating widget
instance**. Instances live directly on the HUD and can be dragged by their
header, closed, and later resized subject to the widget's declared constraints.
Essential placement does not require entering an edit mode. Reopening a closed
widget restores its last placement. The model permits several widget kinds, and
several instances of one kind, to coexist.

The launcher is only a catalogue and lifecycle surface. It does not host widget
content and does not own simulation or navigation state. Each instance has a
stable ID and its own view state (for example ND range), placement, size, and
event routing. Widgets remain projections of the canonical game-state
authorities and emit requests back to those authorities.

All add, focus, move, resize, close, and reset operations route through one HUD
workspace command path with one layout owner. The resulting layout is persisted
through the unified application settings seam. Moving a window may update a
draft every frame, but persistence commits only when the interaction ends so a
drag cannot cause a stream of file writes.

## Alternatives

- **Keep the current single MFD slot and add more tabs** — rejected. It keeps the
  catalogue compact by making the instruments mutually exclusive, which is the
  limitation this decision removes.
- **Expose the widget catalogue only in an Edit HUD mode** — rejected. Layout
  editing can still exist for advanced operations, but opening, dragging, and
  closing ordinary instruments must remain immediately available during flight.
- **Automatically spawn and rearrange widgets from flight context** — rejected.
  Context may choose the default catalogue and first-run layout, but it must not
  undo an explicit pilot arrangement. An AUTO widget may change its own content;
  it does not rearrange the workspace.
- **Use a rigid docking grid** — rejected as the only layout model. A free
  screen-space workspace with edge/widget snapping fits the navball, centred PFD,
  ultrawide displays, and unusual arrangements without forcing empty grid cells.
- **Move the whole workspace into `thalos_ui` immediately** — rejected. The
  workspace is flight-HUD policy and its widgets consume gameplay state. Generic
  draggable-panel chrome moves down only when another feature has a real reuse.

## Consequences

- The current singleton `MfdSelection`, `ActiveWidget`, `NavZoom`, and range
  state must become per-instance state or derived workspace state. No widget
  update or input system may assume one root of its kind.
- The current MFD slot survives only as migration input for the default first-run
  layout; the permanent launcher replaces its tab row as the catalogue.
- The launcher filters or marks widgets by craft capability, but an unavailable
  widget is not silently deleted from a saved layout.
- The fixed launcher and every floating instance participate in the existing HUD
  visibility and pointer-gate contracts. Dragging a window must not steer the
  craft or move the scene camera.
- Camera-aligned PFD symbology, world markers, and global status controls remain
  fixed overlays. Self-contained boxed instruments are candidates for the
  workspace.
- Headless UI evidence must cover simultaneous ND + TRAJ instances, active and
  inactive launcher states, off-screen sanitisation, and at least 16:9,
  ultrawide, and small-window layouts.
