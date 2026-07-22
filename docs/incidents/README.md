# Incident post-mortems

Git-committed write-ups of **fixed** non-obvious bugs — visual, behavioral, crash, or perf —
so future agents can answer "why did that break?" without re-deriving the diagnosis. ADRs
capture design *decisions*; these capture *incident* forensics: symptoms, evidence, the
hypothesis differential, root cause, fix, and how to spot a recurrence.

Not every bug earns one. Write a post-mortem when the diagnosis was non-obvious (the CLAUDE.md
bug-fixing loop ran a real differential), when the root cause teaches a standing invariant, or
when the same class could plausibly recur. A typo-grade fix doesn't need one.

## Workflow

1. **Diagnose first** (CLAUDE.md "Bug fixing"): hypothesis set → targeted falsifiable tests →
   agreed root cause. Search here for a matching prior (`rg 'RenderOrigin|jitter|<symptom>'
   docs/incidents/`) before re-deriving.
2. **Fix the mechanism**, not the symptom.
3. **Write the post-mortem in the same change**: copy `template.md` to
   `YYYYMMDDTHHMMSSZ-kebab-title.md`, using the current UTC time (`date -u '+%Y%m%dT%H%M%SZ'`).
   Fill every section — especially **Evidence**, **Hypotheses considered**, and **Prevention**.
   If the lesson is a standing rule, add/extend the matching CLAUDE.md gotcha or spec-doc
   invariant and link it from **Prevention**.
4. **Reference later**: cite `INC-YYYYMMDDTHHMMSSZ-slug` in discussion and backlog rows.

Historical incidents migrated from auto-memory keep their original dates.

## Identity and ordering

Incident identifiers have the form `INC-YYYYMMDDTHHMMSSZ-short-title`. The filename is that
identifier without the `INC-` prefix, plus `.md`, so lexical directory order is chronological
recording order. The UTC timestamp is the record's creation time; the semantic slug keeps
independently authored records on separate paths, so parallel branches don't compete for one
number. See `ADR-20260722T170714Z-one-chronological-identity-rule` (which extends
`ADR-20260721T034338Z-distributed-chronological-identifiers` to every mintable record).

**`INC-0001`–`INC-0021` are frozen legacy identifiers.** They keep their sequential numbers,
filenames, and headings permanently, and remain valid citations — they were deliberately not
migrated, to avoid rewriting references across in-flight branches. Do not mint new sequential
numbers, and do not renumber an old one.

There is deliberately no hand-maintained index: it was a merge hotspot on this path and had
already drifted out of sync with the directory. The timestamp-sorted files are the index. List
titles in order with `rg --sort path '^# INC-' docs/incidents`, or search content with
`rg '<symptom>' docs/incidents`.
