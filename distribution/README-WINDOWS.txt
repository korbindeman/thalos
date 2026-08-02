THALOS PRE-ALPHA - WINDOWS X64

This is a portable, unsigned pre-alpha build. Extract the complete ZIP before
running it, then launch thalos_game.exe. Do not move the executable away from
the assets and ships folders beside it.

The canonical pre-alpha build includes the neural-terrain capability and uses
it by default. BUILD_INFO.txt records the exact features, source revision, and
target used for this archive. The current learned content is a planet-wide
coarse chart plus a native 90 m, 553 km detail region around the main Thalos
spaceport; the future planet-wide mid-resolution band is not part of this build.

Requirements
  - 64-bit Windows 10 or 11
  - A current GPU driver with Vulkan support
  - Enough free storage to extract the complete archive

Useful commands (PowerShell)
  .\thalos_game.exe --verify-install
      Checks the runtime content without opening a window or GPU device.

  $env:THALOS_TERRAIN="procedural"; .\thalos_game.exe
      Runs the procedural terrain fallback for a controlled comparison.

Windows SmartScreen may warn because this pre-alpha executable is not yet code
signed. Verify the ZIP against its accompanying .sha256 file before running it.

Basic flight controls
  W/S pitch, A/D yaw, Q/E roll
  Shift/Ctrl throttle, Z full throttle, X cut throttle
  T toggle SAS, Space stage
  M map, V camera, Esc pause

Settings and diagnostics use the normal Windows application-data locations.
The licenses and attribution terms distributed with this build apply.
