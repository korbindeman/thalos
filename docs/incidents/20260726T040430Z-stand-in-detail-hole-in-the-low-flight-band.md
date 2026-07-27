# INC-20260726T040430Z-stand-in-detail-hole: open ground was flat paint through the whole low-flight band

- **Date:** 2026-07-26 · **Surface:** any inside-atmosphere view between ~550 m and ~10 km AGL (`just screenshot scene`, `scene2`)

## Symptom

From 800 m AGL over the spaceport the ground rendered as a single flat green with no
texture of any kind — bottom of frame to the tree line. Not "low contrast": near-field
ground luminance measured **σ = 1.09 / 255**, i.e. a constant colour. The far field was
fine (σ = 9.4 at the horizon band) and the accepted showcase framings were fine, which is
why nothing caught it.

## Root cause

Two detail sources cover open ground, and both are off in that altitude band.

- GPU grass hides above `HIDE_AGL_M = 550` (`rendering/gpu_grass.rs`). Confirmed not to be
  the cause on its own: flipping `grass` on in `user/settings.ron` and re-shooting moved
  the frame by 0.57/255 mean and near-field σ 1.09 → 1.20.
- The tile shader's stand-in detail — meadow mottle, clutter normal, canopy stipple,
  rock/scree grain — was retired below a **4 m/px** ground footprint by a single global
  near cut (`STANDIN_OFF_M = 1.0`, `STANDIN_ON_M = 4.0`). At 45° vfov / 2160 px that
  footprint is not reached until ~11 km of slant range, so across essentially the whole
  visible ground plane every stand-in was at zero.

The near cut's *intent* was right — a stand-in must dissolve as the real feature it stands
in for comes into reach — but it was expressed in metres per pixel, globally. "Has the real
feature come into reach" is a property of the term's own wavelength: an 8 m clutter bump and
a 64 m rock grain stop being texture at completely different camera distances, and one
constant in m/px cannot express both. The value that satisfied the showcase framings closed
every stand-in at once, well above where grass takes over.

The shader comment recording the calibration says so explicitly — the band was tuned to
leave the three showcase framings (≥2 m/px) untouched "while clearing the low-altitude near
field entirely". The near field was ceded to grass. Nobody checked the band in between,
which happens to be the normal low-flight altitude.

## Fix

The near cut is now stated in **on-screen pixels per cycle**, not metres per pixel:
full strength at ≤ `STANDIN_ON_PX` (64 px), retired by `STANDIN_OFF_PX` (192 px), with the
term's wavelength recovered as `2 · on_m` from the far-fade knee it already carries. One
rule, correct per-term, no caller restates anything. At the framings the showcase
calibration was tuned from every stand-in projects to ≤15 px/cycle, so the near term is
exactly 1 there and that tuning is preserved bit-for-bit.

That exposed the ladder's real limit — nothing below the 16 m meadow mottle existed — so a
contact-scale rung was added: a 4 m zero-mean tonal grain (`MEADOW_GRAIN_M`), albedo only,
whose positive tail also breaks thin cover to soil and whose negative tail joins the damp
term. Albedo only is deliberate: this file's two prior failures (the 32 m "roll" that
corrugated the plain, the canopy dimple that read as crumpled foil) were both stand-in
*normals* landing inside a shipped footprint. The normal's share of the near field is the
existing `MEADOW_CLUTTER_M`, which the per-wavelength cut hands back.

Measured on `scene`: near-field luminance σ 1.20 → 3.25, hue (R/G) σ 0.0139 → 0.0496, mean
luminance unchanged (63.55 → 63.90). Matched before/after on the showcase presets:
mean |Δ| ≤ 1.3/255 and σ unchanged to two decimals on massif-aerial / -ridge / -valley,
spaceport-aerial and dry-belt.

## An instructive wrong turn

The soil breakthrough was first gated on `macro_variation`'s dry tail, reasoning that bare
ground belongs on dry rises. That *reduced* hue variation back to baseline: `macro_variation`
is a 250 m mottle, so across the near field it is very nearly a constant and the gate simply
switched the effect off. Dry-crest-versus-damp-hollow is a **local** relation; at contact
scale the grain is the only field that carries it. Gate a contact-scale term on a
contact-scale field.

## Recurrence signal

Near-field ground luminance σ below ~2/255 in any inside-atmosphere capture. The falsifier
is an altitude bracket: if σ collapses between 550 m and ~10 km AGL but is healthy above and
below, the ladder has a hole again. Any new stand-in must carry a far-fade knee (`on_m`)
that honestly reflects its wavelength — `footprint_band` derives the near cut from it.
