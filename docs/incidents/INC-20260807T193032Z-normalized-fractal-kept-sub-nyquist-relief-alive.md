# INC-20260807T193032Z: Normalized fractal kept sub-Nyquist relief alive

## Symptom

The planet-scale relief map showed similarly sized bumps and streaks across
nearly every continent. Tectonic highs looked like isolated blobs even though
the causal plate atlas contained connected contacts and broad provinces.

## Mechanism

`octaves_for_lod` appeared to make the procedural relief cascade anti-aliased,
but it never removed the coarsest octave. `fbm` and `ridged` normalize by the
sum of admitted octave weights, so even a fractional first octave returns
full-strength noise. At a 14 km footprint the 6 km hills, 1.4 km swell, and
20 km mountain base therefore remained visible far below Nyquist and aliased
into macro-scale blotches.

Increasing montane uplift initially made the map worse because inherited
terranes shared the same amplitude path as active collision cores. That was the
tell that scale visibility and geologic character were two separate defects.

## Fix and recurrence tell

Each relief band now carries an explicit two-to-four-samples footprint gate.
Inherited terranes have a lower ceiling, while convergence keeps a separate low
continuous spine beneath regionally preserved massifs.

`relief_band_gain_rejects_sub_nyquist_base_octaves` is the scale regression.
`convergent_contacts_keep_a_low_continuous_relief_spine` is the topology
regression. The permanent atlas prints tectonic-belt versus quiet-interior
macro slope; a collapsing contrast or rising quiet-interior slope identifies a
recurrence before visual tuning begins.
