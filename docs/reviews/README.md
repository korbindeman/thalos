# Reviews

Historical output of the adversarial expert-review harness. The skill
(`.claude/skills/expert-review/`) is **retired** as of 2026-08-19
(ADR-20260819T065557Z) — it was advertised in every agent session and the
directory was already gone. Existing reports stay; do not run the cadence
until the skill is restored.

Reports are **claims that survived scrutiny**, not tracked work. The backlog
is the status authority. Before filing a defect an earlier run may have
settled, check [`dismissed.md`](dismissed.md).

## What lives here

| File | Role |
|---|---|
| `<YYYYMMDDTHHMMSSZ>-<slug>.md` | One run's report. Immutable once written. |
| [`repro/`](repro/README.md) | Paste-back test source and probe harnesses backing the findings. **Never compiled** — outside every crate on purpose. |
| `coverage.md` | Which `(slice, lens)` pairs were reviewed. Historical. |
| `dismissed.md` | Findings ruled `by-design` or `wrong`. Read before filing. |

Slice definitions lived with the retired skill; they are not in tree.

## What this is not

**Not a status authority.** [`backlog.jsonl`](../backlog.jsonl) is (`just queue`), and the harness
never writes to it. A report is a pile of *claims that survived scrutiny*, not
work that exists. You promote what you agree with — by hand or via the `steer`
skill — and everything else stays here as a record of what was considered.

**Not a fix.** Reports carry no patches. A confirmed finding may carry a repro
test as a code block, reverted from the tree so `just test` stays green; paste it
back when you fix the thing.

**Not visual.** The harness is headless and cannot screenshot. Visual questions
it cannot settle are collected in each report's final section for your next
capture session.

## Reading a report

Findings are ordered `confirmed` before `plausible`, then `fundamental` → `nit`.

- **`confirmed`** — the refuter reproduced or read through the evidence. Treat as
  real.
- **`plausible`** — the reasoning survived attack but nothing demonstrates it.
  Worth reading; not worth acting on without checking.

The dropped count in the header is the honest signal. A run that drops nothing
has a broken refuter; a run that drops almost everything means the lens is
poorly matched to the slice — either is worth fixing in the skill.

## Report template

```markdown
# Expert review — <YYYY-MM-DD>

- **Run:** <YYYYMMDDTHHMMSSZ> · commit `<sha>`
- **Slices:** `<slice>` (`<lens>`) × 4
- **Evidence:** full | static-only  <!-- static-only caps everything at plausible -->
- **Findings:** N confirmed · M plausible · K dropped

## Confirmed

### <id> — <one-line claim>
**`<severity>`** · [`path/file.rs:123`](../../path/file.rs#L123) · slice `<slice>` · lens `<lens>`

**Mechanism.** Why it is wrong — the causal chain.

**Failure.** Concrete inputs or state → wrong output, crash, or artifact.

**Evidence.** Test source + real output, or the traced code path.

**Fix.** The shape of the change, one or two sentences.

**Refuter.** What was attacked and why it survived.

## Plausible

<same shape>

## Dropped

| id | claim | verdict | reason |
|---|---|---|---|
| … | … | by-design | ADR-… |

## Questions for a capture session

Visual claims the harness cannot settle headlessly.
```
