# INC-20260729T073116Z — wingtip-stand freeze and buried-nosewheel slide

**Symptom.** (a) A plane that missed its wheels on landing ended up balanced
on a wingtip — upright, one wing pointing skyward — and stayed there
indefinitely. (b) After a hard touchdown the nose wheel visually buried
itself in the terrain, the brakes stopped working, and the craft slid for a
long time. (c) Every landing felt like a slam regardless of how gently it
was flared.

**Mechanism.** Three independent gaps in the wheeled-craft ground model,
which by design has *no* solver contact between hull and ground
(`wheeled_craft_collision_layers`) — the raycast gear is the only contact,
with `terrain_floor_backstop` as the safety net:

1. *Wingtip stand:* the backstop was a pure **translation** clamp — it lifted
   the CoM along the radial and zeroed radial velocity, applying zero torque.
   Gravity enters the integrator as a linear acceleration at the CoM, so no
   toppling moment existed anywhere: whatever attitude the craft had when its
   deepest hull point pinned was frozen. Fixed by resolving the residual
   approach as a contact impulse (normal + Coulomb friction) **at the support
   point** (`deepest_hull_support`), which restores the angular response and
   topples the craft flat.
2. *Buried nose wheel:* the suspension ray started at the strut top on the
   hull skin. The backstop tolerates 0.5 m of hull penetration, so a hard
   landing could put the strut top *below* the terrain heightfield — and a ray
   cast from underneath a heightfield surface hits nothing. The wheel silently
   unloaded (no normal force ⇒ no brake force, since braking is clamped to
   μ·N), while the frictionless backstop carried the weight ⇒ the slide.
   Fixed by starting the ray `ray_start_lift_m` above the strut top and, if it
   still misses, falling back to the analytic `HeightSource` the backstop
   itself reads.
3. *Slam landings:* the damper coefficient was near-critical (ζ = 1.2) and
   engaged instantly — at contact onset compression is ~0 but the compression
   *rate* is the full sink speed, so the damper delivered a multi-g step force
   the moment the wheels touched. Fixed with the real-oleo asymmetric
   schedule: soft compression damping (ζ = 0.4) ramped in over the first 20 %
   of travel, firm rebound damping (ζ = 1.2) to eat the stored energy.

**Tell.** A craft resting at an attitude no real object would hold (wingtip /
nose-stand) with zero angular velocity = a ground response path that moves
the CoM without torque. Wheels visually under the surface + no braking
authority = suspension rays failing to find ground from inside it (check the
`landing_gear` diagnostics and whether `weight_on_wheels` reads false while
the hull is at terrain height).

**Related latent fix in the same change:** `detect_terrain_impact` measured
impact speed as `|v|` (including horizontal ground speed) — masked only by
the TEMP `SHIP_IMPACT_TOLERANCE_M_S = 1e9`. Any finite tolerance would have
destroyed the craft on every normal ~60 m/s runway touchdown. Now measures
the into-surface radial sink rate, and the tolerance is restored to 12 m/s.
