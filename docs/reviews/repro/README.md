# Repro artifacts

Failing tests and probe harnesses produced by the retired expert-review
harness, one directory per run: `<run-timestamp>-<slug>/`.

**These files are never compiled.** They live outside every crate on purpose —
a `.rs` file here is *source to paste back*, not a test the workspace builds.
`just test` stays green; nothing here is in a `tests/` directory or a `mod`
tree, and nothing here is in any `Cargo.toml`.

## Why they exist

A review finding backed by a failing test is worth ten backed by prose, but the
test has to be reverted to keep the workspace green — so without this directory
the evidence dies with the agent that produced it. That happened once already
(see `20260729T063800Z-propagator/astro-4-golden-section-replica.md`), which is
why the skill now requires evidence to be written here **in the same step that
produces it**, never held in an agent's reply.

## Using one

Each `.rs` file opens with a header giving: the finding id and claim, the exact
module to paste into, the helpers it depends on, the observed output verbatim,
and the assertion that fails. Read the header's corrections section first —
**every preserved test here has at least one correction from the refutation
pass**, and in one case (`astro-2`) the originally-proposed fix would have made
the code worse.

Paste, run, fix, then delete the paste — do not commit the test into the crate
unless it is a genuinely valuable regression test on its own terms, in which
case it stops being a repro artifact and this copy should be deleted.

Anchors (`file.rs:123`) are captured at the run's commit and drift. Re-check them
before trusting any of this.
