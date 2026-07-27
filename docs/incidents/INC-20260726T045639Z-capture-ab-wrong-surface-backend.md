# INC-20260726T045639Z — capture A/B rendered a different surface backend than the one edited

## Symptom

An erosion band added to `DiffusionSurface` produced a **provably large geometry
change** (numeric probe: ~190 m mean height difference on the showcase massif)
while before/after `just screenshot` captures of the same framings were
**pixel-identical**. Two full capture rounds were burned confirming a
"no-op" that wasn't.

## Mechanism

Thalos renders through `DiffusionSurface` only when the process boots with
`THALOS_TERRAIN=diffusion` (`terrain_registry::thalos_diffusion_enabled`, a
boot-time `OnceLock`). The env var is **not** in the capture lane's
`CAPTURE_OVERRIDE_KEYS`, and the persistent capture host had been started
without it — so every capture rendered the canonical `ProceduralSurface`
backend, which the edit never touched. The A/B compared baseline against
baseline. A second trap layered on top: the host only restarts on stale
*sources*, so exporting the env on later client invocations changed nothing
until an explicit `just capture-stop`.

## Fix

Stop the host and relaunch with the env set:
`just capture-stop`, then `THALOS_TERRAIN=diffusion just screenshot …` (the
client inherits the shell env into the host it spawns). Structural option
filed on the backlog row: teach the capture lane a restart-requiring env key
set that includes `THALOS_TERRAIN`, so a mismatch restarts the host instead of
silently shooting the wrong backend.

## Tell

A terrain edit that a CPU probe (`SurfaceQuery` sampled directly) proves large
but captures show as zero difference. Confirm which backend the host booted:
the boot log prints `Thalos: terrain-diffusion surface active (fingerprint …)`
when the gate is on — its absence means every capture is canonical-surface
evidence, whatever the edit touched.
