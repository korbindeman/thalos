# INC-20260729T092010Z — A single Vulkan renderer lost the NVIDIA adapter

- **Date:** 2026-07-29
- **Status:** reproduced across clean driver update; card thermal/hardware
  remediation required
- **Surface:** interactive game, Windows, Vulkan, RTX 4070 Ti

## Symptom

At 11:20 local time the display/GPU entered the same state that previously
required a reboot. `nvidia-smi` reported:

```text
Unable to determine the device handle for GPU0: 0000:01:00.0: GPU is lost.
Reboot the system to recover this GPU
```

`Win+Ctrl+Shift+B` did not restore the device.

The user was running `just game runway` and the failure happened essentially at
the start of the scene: the game hung for a while, the display went black,
the image returned for about one second, then went black again and the adapter
remained lost. That visible sequence is consistent with Windows attempting a
driver/GPU recovery and the recovered device failing immediately, even though
the usual Event 4101/DxgKrnl record was absent.

## Evidence that changed the diagnosis

Runtime session `3736-1785316764594` was the only renderer. It began at
11:19:24, after `just game`'s capture-stop invocation, and its last record was
11:20:10. Every tile gauge reported `instances=1`; no other Thalos, capture,
Cargo, or Rust process survived the loss.

The final frame gauge was not an application-RAM runaway:

- process RSS 2,483 MiB;
- mesh slabs 1,511 MiB;
- live tile estimate 1,320 MiB / 3,890 tiles;
- 134 images, 5,660 meshes, 12,015 entities;
- 14.5 ms mean GPU time and 51.8 fps.

The session emitted no warning, error, WGPU validation failure, or
`DeviceLost` record. Windows reported the PCI device as `Started`, but NVML
could no longer obtain its handle. The System, Application, DxgKrnl Admin and
DxgKrnl Operational logs contained no WHEA, display TDR, live-kernel report, or
application fault at the loss. Pressing the Windows driver-reset chord produced
no recovery event.

This falsifies renderer overlap as the complete explanation and does not look
like the earlier explicit WGPU `Out of memory` path. It does **not** yet
distinguish:

1. whole-card VRAM pressure hidden from the per-process counters;
2. temperature, board-power, or clock instability;
3. an NVIDIA Vulkan-driver failure;
4. a GPU shader hang or other Thalos workload that bypasses WGPU validation;
5. GPU/PCIe/PSU hardware instability.

The failure's timing narrows the workload shape: it is reachable during the
runway scene's cold terrain/vegetation/resource stream, before a long-lived
working set could accumulate. A gradual application-memory leak is therefore
not a viable explanation for this recurrence.

## Instrumentation and containment

With `THALOS_GPU_HEALTH=1`, `thalos_diagnostics` samples NVML once per second
into `thalos::diagnostic::gpu_health`:

- whole-card used/total VRAM and fraction;
- temperature;
- draw/limit power and fraction;
- GPU/memory utilization;
- graphics clock, performance state, and clock-throttle reasons.

The first failed whole-card query is an `ERROR sample_error`, then the sampler
stops rather than repeatedly querying a wedged driver. `just diag` owns three
checks: `gpu_adapter_lost`, `gpu_memory_pressure` (≥90% whole-card VRAM), and
`gpu_thermal_pressure` (≥88 °C). This is the missing discriminator the failed
session did not carry.

The sampler was always-on while collecting the controlled recurrences below.
After it identified the thermal/hardware mechanism, it became opt-in so ordinary
play and capture do not pay for one-second NVML queries or their record volume.
The temporary `.env.just` window-mode, resolution, VSync, and 1 GiB tile-budget
envelope was removed once a GPU power reduction provided the useful
containment.

## Next falsifiable test

After reboot:

1. Confirm `nvidia-smi` sees the card.
2. Install the current NVIDIA WHQL driver before testing if the installed
   version is behind.
3. Run the same game mode once through `just game`; do not run captures.
4. If stable for ten minutes, close normally and run `just diag 1`.
5. If the adapter is lost, reboot and read the final `gpu_health/sample` before
   `sample_error`.

A loss with low VRAM and power rules out capacity pressure. Core temperature
alone cannot rule out a hidden memory/hotspot limit, so throttle reasons must be
read beside it. If neither is implicated, move to a one-axis Vulkan↔DX12
comparison. A loss in an external test at stock settings is
hardware/power-delivery evidence and is the point to test another PSU/GPU or
pursue an RMA—not before.

## Controlled recurrence: 11:51

The exact workload was `THALOS_TERRAIN=diffusion just game runway`, session
`22448-1785318492639`, with one renderer and the containment envelope active.
The adapter was lost after 193 seconds. The experiment rules out the two leading
application-memory candidates:

- whole-card VRAM peaked at 5,798 / 12,282 MiB (47.2%);
- tile residency was limited to 1 GiB and had fallen to 176–449 MiB near loss;
- board power peaked at 210.3 / 285 W (73.8%);
- visible core temperature peaked at 81 °C and was 78 °C in the final sample.

GPU utilization remained 99%. More importantly, NVML asserted software thermal
slowdown (`clock_throttle_reasons & 0x20`) in 79 / 194 samples, beginning 21
seconds after process start and remaining asserted in the final four samples.
NVIDIA defines this bit as the GPU **or memory** exceeding maximum operating
temperature. This card/driver reports no memory temperature, so the 78–81 °C
core reading cannot rule out a memory or hotspot thermal problem. No software
power-cap or hardware-power-brake pressure was sustained. Six samples also
asserted hardware thermal slowdown (`0x40`) together with hardware slowdown
(`0x08`); the hottest sample's full mask was `0x68`. Hardware thermal slowdown
is NVIDIA's protection path reducing core clocks by a factor of two or more,
not ordinary boost management.

At 11:51:25 Windows' `WATCHDOG` component successfully created live-kernel dump
`WATCHDOG-20260729-1151.dmp`, precisely when telemetry stopped. `nvidia-smi`
then reported no devices and the game process was gone. The normal System log
again had no display/TDR/WHEA record. The dump lives under the
administrator-only `C:\Windows\LiveKernelReports` tree and must be copied out
after reboot for analysis.

This recurrence does **not** support buying a larger-VRAM GPU: more than half of
VRAM and one quarter of the board-power limit remained unused. It promotes
three candidates: a 610.62 driver defect (its reported temperature-limit fields
were also nonsensical), a GPU cooler/contact/VRAM-hotspot fault, or marginal
GPU/PCIe power/hardware stability under sustained 99% load. The next order is:
current WHQL driver, inspect the watchdog dump, then a reduced-power external
stress A/B. A stock-setting failure outside Thalos is RMA/PSU evidence; only
then does replacement hardware become the fix.

## Watchdog dump result

The copied dump is 1,344,838 bytes with SHA-256
`84CC98F9D4AACEE4F0C7C954ED8ADB66778BF538700FC93E87996D6D222DA84F`.
WinDbg 10.0.29617.1000 with Microsoft symbols reports:

```text
VIDEO_ENGINE_TIMEOUT_DETECTED (141)
Failure.Bucket: LKD_0x141_IMAGE_nvlddmkm.sys
Failure.Exception.IP.Module: nvlddmkm
PROCESS_NAME: System
```

The kernel stack is
`dxgmms2!VidSchiCheckHwProgress → VidSchiResetEngines →
VidSchiResetHwEngine → dxgkrnl!TdrCollectDbgInfoStage1`: Windows detected that
one GPU engine made no timely progress, tried the per-engine TDR reset path, and
captured the live dump. The embedded TDR payload names `thalos_game.exe` as the
client whose GPU work timed out; `System` is the watchdog thread collecting the
dump, not evidence that the game was uninvolved.

Parameter 2 and the failure bucket point into `nvlddmkm.sys` from driver 610.62
(file timestamp 2026-06-11). A triage live dump cannot distinguish malformed
client work, an NVIDIA kernel-driver defect, and physical GPU non-response by
itself. In this incident there was no WGPU validation fault or resource
pressure, while NVML independently recorded software and hardware thermal
protection, so driver-versus-card stability remains the relevant split.

The workstation was clean-updated to Game Ready 610.88 (Windows driver
32.0.16.1088, dated 2026-07-22). After reboot the card was visible and idle at
37–38 °C, 5 W, 22 MiB VRAM, with no throttle bits. The next test keeps every
containment input fixed and changes only the driver; any `0x40` hardware-thermal
sample is an immediate stop condition rather than waiting for another TDR.

## Clean-driver recurrence: 12:38

The one-axis 610.88 rerun, session `5880-1785321331449`, lost the adapter after
152 seconds and produced `WATCHDOG-20260729-1238.dmp`. It reproduced the same
mechanism with:

- 6,066 / 12,282 MiB peak whole-card VRAM (49.4%);
- 200.3 / 285 W peak board power (70.3%);
- 83 °C peak visible core temperature;
- 99% GPU utilization;
- 61 software-thermal samples;
- 9 hardware-thermal (`0x40`) and hardware-slowdown (`0x08`) samples;
- no hardware power-brake (`0x80`) sample;
- only 338 MiB tile residency in the last gauge and no runtime/WGPU error.

The final seven NVML samples carried mask `0x4c` (software power scaling,
hardware slowdown, hardware thermal slowdown) at a repeated 80 °C / 194.5 W,
then telemetry and the process stopped. Those identical final values are likely
the last cached card state while the engine was already wedged; their important
property is that hardware thermal protection remained asserted through the
timeout.

The second dump is 1,343,736 bytes with SHA-256
`BD3334AECC2D0A49D5B547132F81394AF7AE30FDE1936D0ADE17A9EA437C4001`.
WinDbg positively loads the July 22 610.88 `nvlddmkm.sys` and reports the same:

```text
VIDEO_ENGINE_TIMEOUT_DETECTED (141)
Failure.Bucket: LKD_0x141_IMAGE_nvlddmkm.sys
Failure.Hash: {341dd0b3-9ebd-47a8-9de8-23f4b00fabbc}
```

The 610.62 dump has the identical failure bucket, failure hash, and
`VidSchiCheckHwProgress → VidSchiResetEngines → VidSchiResetHwEngine →
TdrCollectDbgInfoStage1` stack. Only the expected private NVIDIA instruction
offset changed (`0x1957e60` → `0x1959fa0`) between builds. This is not merely
two visually similar crashes: Windows bucketed them as the same failure
identity across two driver versions.

This falsifies 610.62 as the sole cause and makes a larger-capacity GPU
irrelevant: neither memory nor board power approached its limit. A valid 99%
GPU workload may throttle, but a healthy adapter must remain recoverable.
Repeated card-internal hardware-thermal protection followed by an unrecoverable
0x141 on clean 610.88 is sufficient to stop full-load testing and pursue
warranty/RMA or professional cooler/thermal-pad inspection. Do not open or
re-pad a card still under warranty. A 60% power limit plus 60 FPS cap is only a
temporary containment option, not the permanent repair.
