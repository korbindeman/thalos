# INC-0014: Density and LUT-unit adapters erased the Bevy sunset

- **Status:** Fixed
- **Date:** 2026-07-21 (observed) / 2026-07-21 (fixed)
- **Severity:** visual
- **Surface:** `just compare cloud-sunset atmosphere`, Bevy rocky-body atmosphere and cloud lighting

## Summary

At a deterministic 1° sun elevation, the canonical Bevy atmosphere produced a
navy sky and white clouds while the legacy renderer produced a warm sunset.
Two unit-breaking adapters were responsible: the atmosphere projection reduced
the authored optical column to 10%, and cloud ambient replaced its calibrated
scene-light energy with raw photometric sky-view-LUT radiance. Restoring the
physical column and using the LUT for chromaticity only made the sky, clouds,
and ocean response coherent.

## Symptoms

- The Bevy `cloud-sunset` variant retained a dark blue/navy sky at a 1° sun.
- Clouds stayed nearly white instead of carrying the attenuated low-sun colour.
- The ocean sun road was present, so this was not a missing sun or water pass.
- Both variants completed without a shader or render-pipeline error.

## Evidence

The controlled `cloud-sunset / atmosphere` comparison held framing, sun
elevation, weather, render quality, and revision fixed. With the `0.1` density
adapter and raw LUT ambient, custom versus Bevy differed by MAE 45.88/255 over
93.1% of pixels. Restoring density to `1.0` corrected the atmospheric optical
depth and reduced MAE to 24.75/255, but clouds remained pale. Peak-normalizing
the sky-view LUT and retaining the calibrated cloud-ambient energy reduced the
final MAE to 15.55/255 and restored warm bronze low-sun clouds.

The final canonical Bevy `cloud-sunset`, `ocean`, `ocean-slopes`, and
`runway-atmosphere` captures all completed cleanly. The surface probe retained
a blue sky and readable long-path aerial recession rather than becoming a
ground-level white fog.

## Hypotheses considered

- **BigSpace placed the stock atmosphere at the wrong center.** Ruled out by
  the concentric horizon/sky geometry, stable framing, and INC-0007's existing
  camera-local proxy correction. Changing density altered the spectral result
  without changing placement.
- **The cloud shader did not bind Bevy's atmosphere LUTs.** Ruled out by the
  active LUT branch and by the cloud response changing with the LUT unit
  normalization.
- **The stock model simply cannot produce a sunset.** Ruled out when restoring
  the authored optical column immediately brought back long-path scattering.
- **Atmosphere density and LUT energy crossed incompatible unit domains.**
  Confirmed by the two controlled corrections and their monotonic comparison
  improvement.

## Root cause

`BEVY_ATMOSPHERE_DENSITY_SCALE = 0.1` treated density as an exposure control.
That reduced the Rayleigh/Mie optical column by 90%, changing transmittance and
spectral colour instead of merely changing display brightness. Separately, the
cloud march used raw photometric sky-view-LUT radiance as ambient energy while
its direct sun and authored ambient remained on Thalos's calibrated scene-flux
scale. The large, nearly neutral LUT value overwhelmed the correctly reddened
direct term.

## Fix

The Bevy adapter now projects the authored atmosphere at density multiplier
`1.0`. Brightness reconciliation remains F7 work in shared light/exposure
units. Cloud ambient samples Bevy's sky-view LUT for chromaticity, normalizes
that colour by its peak channel, and applies the existing calibrated
top/bottom ambient energy. Bevy transmittance remains authoritative for direct
and view-path attenuation.

## Prevention & recurrence signals

- Never tune rendered brightness by changing a physically authored atmosphere
  density or scale height. Use shared light, camera exposure, or output
  transforms.
- Never mix a photometric LUT and an independently calibrated scene-light term
  as if their magnitudes shared units. Normalize colour-only bridges explicitly
  until F7 establishes one scale.
- A rocky-atmosphere change must include the deterministic 1°
  `cloud-sunset / atmosphere` comparison plus `runway-atmosphere`. A return of
  navy low-sun sky, white clouds, or surface white-out is a recurrence signal.
- See [atmosphere.md](../atmosphere.md) and [clouds.md](../clouds.md).
