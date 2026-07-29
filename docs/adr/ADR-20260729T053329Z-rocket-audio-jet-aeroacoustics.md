# ADR-20260729T053329Z — Runtime rocket audio is a jet-aeroacoustics model, not a chamber simulation

**Status:** accepted (2026-07-29)

## Context

Thalos has no audio at all: no `bevy_audio`, `kira`, `rodio`, or `cpal` anywhere
in the workspace. Engine sound is a green field, and the user's framing was
[engine-sim](https://www.engine-sim.parts/)-style runtime generation from sim
state, "perhaps more focused".

Three approaches were live:

1. **Port engine-sim's architecture** — simulate chamber/duct gas dynamics in the
   time domain and take the audio from the pressure at the outlet.
2. **Sample playback** with throttle-driven crossfades and pitch/filter shaping.
3. **Model the jet plume's aeroacoustics** and synthesise from the radiated field.

## Decision

**Model jet aeroacoustics (NASA SP-8072 / Eldred in shape).** The nozzle model
exists only to produce the four quantities that set the sound — exit velocity,
exit diameter, mass flow, and exit/ambient pressure mismatch.

Developed in a standalone clean-room probe,
**`korbindeman/thalos-audio-probe`** (private; sibling checkout at
`~/Documents/thalos-audio-probe`), gates A0–A5, extracted back here at A5 as a
pure-Rust no-Bevy `thalos_acoustics` crate. Same vehicle pattern as
`thalos-terrain-probe` under ADR-20260723T142945Z. **The probe's README is the
strategy document**; this repo carries the decision and the backlog rows.

## Why not engine-sim's architecture

engine-sim works because IC engine sound is **deterministic and periodic**:
firing events at a known frequency, character from exhaust-runner resonances, so
the audio genuinely *is* the simulated pressure at the pipe tip.

A rocket is the opposite. Its sound is **broadband stochastic noise generated
outside the engine**, in the turbulent shear layer where the supersonic plume
mixes with ambient air. The combustion chamber is a near-steady pressure vessel.
Simulating it in the time domain and listening to it builds an expensive silence
generator — the audible sound is made in a region the chamber model does not
contain.

This is the trap this record exists to prevent: the analogy to engine-sim is
strong, the repo is a compelling reference, and the failure is not obvious until
the model is built.

## Why not samples

Samples fall over exactly where Thalos lives: continuous throttle, shipyard-
composed vessels with arbitrary engine counts and nozzle geometry, ambient
pressure from sea level to vacuum, and listener range from metres to tens of
kilometres. A physical model makes vacuum silence, altitude fade, directivity,
and Doppler *emergent* rather than authored. If the game only ever had one rocket
at one distance, samples would be the right call.

## Consequences

- **Acoustic power is η·½ṁUₑ² with η ≈ 0.5 %, not Lighthill's U⁸ law.** U⁸ is the
  *subsonic* jet scaling; supersonic jets plateau at roughly constant efficiency.
  Using U⁸ for a rocket is a standard mistake and will be re-proposed.
- **Retarded time is mandatory, and it constrains the runtime design.** Tapping a
  delay line at `r(t)/c` stalls the read head the instant a vehicle's radial
  recession rate reaches Mach 1 — hard spectral notch, then garbage. The delay
  must solve `τ + r(τ)/c = t`, so `thalos_acoustics` needs a bounded *emission
  history*, not a current-position lookup. Thalos has supersonic vehicles.
- **Perceptual voicing is a separate, labelled stage.** The physics says a
  Saturn-class booster radiates most of its power below 30 Hz, which is correct
  and unreproducible. The transposition that fixes it is not physics and must not
  be smuggled into the source model's constants (probe: `voicing.rs`, disabled by
  `--no-voicing`).
- **Only three calibration constants are tunable** (`acoustic_efficiency`,
  `slice_strouhal`, `spread_rate`). Everything else is structural. Fitting the
  model to recordings by moving other values destroys its ability to extrapolate
  to engines nobody has recorded, which is the entire reason for choosing it.
- Verification is a headless render loop (WAV + spectrogram + level table),
  agent-runnable, mirroring `just screenshot`. Judging *sound* stays user-run for
  the same reason `just game` does.

## Alternatives if this is reopened

Sample playback remains viable if the engine roster ever collapses to a handful
of fixed vehicles heard at fixed distances. Chamber simulation does not become
viable at any scale — it is not a cost trade, it models the wrong region.
