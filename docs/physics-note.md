Real-world astrodynamics tools (GMAT, STK, JPL's SPICE-adjacent stuff) use a layered approach:

- Kepler — exact closed-form for pure two-body. No drift, any timestep, including warp.
- Encke's method — Kepler reference orbit + numerical integration of the deviation from Kepler caused by perturbations. The deviation is tiny, so even forward Euler stays accurate at huge timesteps. This is exactly what you want for solar radiation pressure, slow drag, third-body perturbations, J2 oblateness, etc. Works fine under warp.
- Cowell (full numerical integration of the actual equations of motion) — fallback when perturbations are large enough that the Kepler reference is no longer a useful base.
- Avian / rigid-body integrator — when forces are large and sub-orbital-period: thrust, hard contact, dense atmosphere. Real-time only.
---
- Coasting in vacuum → Kepler (real-time and warp)
- Coasting with perturbations (solar pressure, gentle drag, third-body) → Encke (real-time and warp; orbital decay falls out naturally)
- Thrust active / hard contact / dense atmosphere → Avian (real-time only; warp is gated off here, same as KSP)
- Avian's rigid body stays alive for contact graph, collider geometry, future docking/debris regardless of who owns translation. Position is overwritten from the active propagator when Avian doesn't own translation. This is your "Option D" intuition, generalized.
