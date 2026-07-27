# Incident post-mortems

Write-ups of **fixed** non-obvious bugs — visual, behavioral, crash, or perf — so a future
agent can answer "why did that break?" without re-deriving the diagnosis. ADRs capture
*decisions*; these capture *forensics*.

## The bar

Write one when the diagnosis was **non-obvious**: the first plausible hypothesis was wrong,
the mechanism isn't guessable from the symptom, or the same class could plausibly recur. A
typo-grade fix, or a bug whose cause was evident from the stack trace, doesn't need one.

**Keep it short.** Symptom, mechanism, fix, recurrence signal — `template.md` is four
sections and you may drop any that has nothing real to say. A terse post-mortem that gets
written beats a thorough one that doesn't.

## Workflow

1. **Search first** — `rg '<symptom>' docs/incidents` — before re-deriving a diagnosis.
   `rg --sort path '^# INC-' docs/incidents` lists titles chronologically. No hand-maintained
   index; it was a merge hotspot and drifted.
2. **Diagnose, then fix the mechanism** (CLAUDE.md "Bug fixing"): hypothesis set → targeted
   falsifiable tests → root cause.
3. **Write it in the same change as the fix.** Copy `template.md` to
   `YYYYMMDDTHHMMSSZ-kebab-title.md` using `date -u '+%Y%m%dT%H%M%SZ'`.
4. **Cite** `INC-YYYYMMDDTHHMMSSZ-slug` in discussion and backlog rows.

## Identity

`INC-YYYYMMDDTHHMMSSZ-short-title`; the filename drops the `INC-` prefix, so lexical order is
chronological. Never mint a sequential number (ADR-20260722T170714Z-one-chronological-identity-rule).
**`INC-0001`–`INC-0021` are frozen legacy identifiers** — permanently valid citations, never
renumbered, never extended.
