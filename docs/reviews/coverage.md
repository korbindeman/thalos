# Review coverage ledger

Which `(slice, lens)` pairs the retired expert-review harness looked at.
Historical. Slice definitions lived with the skill, which is not in tree.

**A pair not listed here has never been reviewed** and outranks everything listed.
Among listed pairs, priority is staleness × churn: commits touching the slice's
paths since `Commit`. A slice with no churn since its last review is near-worthless
to re-read.

```bash
git log --oneline <commit>..HEAD -- <slice paths>
```

| Slice | Lens | Date | Commit | Kept | Dropped | Report |
|---|---|---|---|---|---|---|
| `propagator` | `astrodynamicist` | 2026-07-29 | `2fb6db6` | 5 | 2 | [20260729T063800Z-propagator-trial](20260729T063800Z-propagator-trial.md) |
| `shading` | `graphics` | 2026-07-30 | `2fb6db6` | 5 | 2 | [20260730T011353Z-graphics-lighting](20260730T011353Z-graphics-lighting.md) |
| `ground-scatter` | `graphics` | 2026-07-30 | `2fb6db6` | 2 | 2 | [20260730T011353Z-graphics-lighting](20260730T011353Z-graphics-lighting.md) |
| `render-integration` | `graphics` | 2026-07-30 | `2fb6db6` | 3 | 3 | [20260730T011353Z-graphics-lighting](20260730T011353Z-graphics-lighting.md) |
| `clouds` | `graphics` | 2026-07-30 | `2fb6db6` | 6 | 1 | [20260730T011353Z-graphics-lighting](20260730T011353Z-graphics-lighting.md) |
