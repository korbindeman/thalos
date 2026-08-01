# Autoland used full gear stroke and the hull backstop

## Symptom

LAND completed and stopped, but touchdown was extremely bumpy. A stopped
screenshot could not distinguish a hard flare from poor landing-gear tuning.

## Mechanism

Runtime session `28324-1785560800045` separated both causes:

- At 5.92 m over the runway the Meridian still descended at 2.82 m/s. The old
  flare law blended pitch only from height and had already cut throttle to idle
  at 18 m, so it commanded about 5.1 degrees without closing the loop on the
  sink that the aircraft actually achieved.
- Contact then unloaded the wheels, re-contact drove suspension compression to
  100%, and `backstop_intervention` reported 0.61 m hull penetration. This was
  real floor contact, not camera motion.
- `GearTuning` used one fixed spring stiffness and derived every damper from
  `mass / wheel_count`. The shipped Meridian's support geometry is 12.1% nose /
  87.9% mains, so each main was under-damped while the nose damper was about
  3.5 times its load-correct value.

## Fix and recurrence tell

The flare now schedules a 0.75 m/s touchdown sink, feeds achieved sink error
back into pitch, retains approach power until 9 m, and reaches idle at 3 m. A
post-touchdown wheel unload holds the shallow touchdown attitude rather than a
7-degree climb attitude.

Gear spring/damper coefficients now derive from CoM/axle reaction, surface
gravity, authored stroke, and a common loaded sag fraction; main compression
damping accounts for the mains initially absorbing the whole craft. A late
quadratic end stop protects the final stroke. A deterministic Meridian drop
test at 3.05 m/s pins transport-aircraft impact capacity without bottoming.

In a recurrence, `appr_frame` distinguishes actual from target sink
(`sink_rate_m_s`, `target_sink_rate_m_s`), while `gear_contact` reaching
`max_compression_frac = 1` followed by `backstop_intervention` proves the gear
ran out of stroke.
