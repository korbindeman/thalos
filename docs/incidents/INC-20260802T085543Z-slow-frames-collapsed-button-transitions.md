# INC-20260802T085543Z-slow-frames-collapsed-button-transitions: quick input vanished during slow frames

- **Status:** Fixed
- **Date:** 2026-08-02
- **Severity:** behavioral
- **Surface:** keyboard, mouse-button, and gamepad-button actions

## Summary

A press and release that both reached Bevy during one long render frame could
vanish from Thalos's semantic input layer. Bevy retained both raw messages and
both transient flags, but `bevy_enhanced_input` sampled only the final
`pressed` value. That value was false, so the action appeared idle and emitted
neither edge.

## Evidence

The deterministic repro writes `MouseButtonInput::Pressed` and
`MouseButtonInput::Released` before one `App::update`. Before the fix,
`GameInputIntent::primary_started` remained false. Multiple `MouseMotion`
messages in the same setup summed correctly, ruling out event expiry and
isolating the defect to discrete state reduction.

## Hypotheses considered

- **Input events expired before gameplay read them.** Ruled out: the raw
  messages and Bevy's `just_pressed` / `just_released` flags were present.
- **Mouse motion overwrote earlier samples.** Ruled out: Bevy's accumulated
  motion resource returned the sum of every injected delta.
- **Enhanced input collapsed transitions into final held state.** Confirmed:
  its keyboard and mouse bindings read `ButtonInput::pressed` only.
- **A consumer drained shared input before another consumer.** Ruled out: the
  failed action was already idle at the shared intent boundary.

## Fix

`FrameIndependentInputPlugin` records raw keyboard, mouse-button, and processed
gamepad-button state changes after Bevy input processing. It exposes one state
change per button per game frame and queues any remaining changes for later
frames. This preserves every edge without duplicating configurable bindings.
Mouse motion and scrolling stay on Bevy's additive accumulators.

## Prevention and recurrence signal

The input-crate tests inject clicks, repeated clicks, key taps, gamepad taps,
and multiple mouse-motion events between updates. A quick tap that fails to
produce a start followed by a release, or two clicks that produce fewer than
four ordered edges, means frame-rate-dependent input loss has returned.
