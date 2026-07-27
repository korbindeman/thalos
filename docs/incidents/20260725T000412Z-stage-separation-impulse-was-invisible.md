# INC-20260725T000412Z-stage-separation-impulse-was-invisible: the standard decoupler opened an imperceptibly slow gap

- **Date:** 2026-07-25 · **Surface:** physical stage separation, especially wet launch vehicles

## Symptom

The standard Saturn decoupler tuning could not open a visible gap between wet
stages on a useful timescale.

The plausible causes were a wrong attach-graph cut, a hidden or unsynchronized
detached root, and immediate canonical collision. The Saturn cut does contain
the decoupler, lower tank, and lower engine; the detached root is visible,
render-layer propagation explicitly handles reparented descendants, and every
`CraftRoot` is synchronized from its canonical `CraftId`.

## Root cause

The standard decoupler supplied only 100 N·s per metre of diameter. Saturn's
4 m ring therefore delivered 400 N·s. Against conservative wet stage masses of
roughly 42 t and 87 t, equal/opposite impulse produces only about 0.014 m/s of
relative separation: opening a one-metre gap takes around 70 seconds. Two
correctly rendered stages would therefore remain effectively superimposed.

This was a real tuning defect, but it was **not** the cause of the separately
reported missing-booster defect: a later runtime trace confirmed canonical
vessel creation at the tuned impulse while the booster geometry was still
absent. That rendering/hierarchy defect is tracked independently.

## Fix

The standard decoupler now supplies 2,000 N·s per metre, or 8 kN·s at Saturn's
diameter. That produces about 0.28 m/s at conservative wet masses and a faster
departure after booster burnout. The impulse remains physical and
equal/opposite. A Saturn-scale regression test enforces at least 0.25 m/s, and
the separation event now logs detached mass, impulse, and resulting relative
speed.

## Recurrence signal

Look for `stage separation created persistent vessel` in the runtime log. If it
reports a vessel and a relative separation speed measured in only centimetres
per second, investigate part-catalog impulse tuning before the graph cut or
renderer.
