# INC-20260803T060930Z — relief clamped authored land to ambiguous zero height

## Symptom

Replacing the independent mountain mask with tectonic orogeny left
`ProceduralSurface::macro_signed_height_m` unchanged, yet the exported
conditioning chart lost about 0.44% of the planet's area from land to water.
The land fraction moved even though the only coastline authority had not been
edited.

## Mechanism

`combine_macro_and_relief` prevented relief on macro land from going below sea
level by clamping it to exactly `0.0`. The exporter and other terrain consumers
use the intentional convention `height > 0.0` for land. A macro-land cell with
negative relief was therefore classified as water after the clamp. Changing
orogeny changed which cells reached that exact-zero state, making relief a
hidden second coastline authority despite never crossing below the waterline.

## Fix

Authored macro land now retains a 0.01 m positive clearance when relief would
otherwise reach the waterline. This is below rendering and collision relevance,
but preserves the semantic sign. A 32,768-direction Fibonacci-sphere regression
test asserts that composed height and `macro_signed_height_m` classify every
sample identically.

## Tell

The signed sea field or its coastline mask is byte-stable, but the final-height
land fraction changes after a relief-only edit. Count transitions by sign, not
only the aggregate percentage: land cells turning into exact `0.0` identify this
failure, while a genuine coastline change alters the signed sea field itself.
