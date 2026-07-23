# ADR-20260723T143155Z-planetary-model-pytorch-finetune: Earth-like planetary bands fine-tune terrain-diffusion in PyTorch; one unified architecture is the end state

- **Status:** Accepted
- **Date:** 2026-07-23
- **Related:** ADR-20260721T033713Z-rust-native-learned-terrain (amended, not
  superseded) · ADR-20260720T211046Z-offline-terrain-packages ·
  ADR-20260723T142945Z-neural-terrain-standard-renderer-keystone

## Context

ADR-20260721T033713Z committed MIRA learned models to a single Rust/Burn
authorship so that training and any future runtime inference share one model
definition ("two model definitions and tensor conventions will drift"). That
decision was made for the airless family, trained from scratch.

The keystone pivot (companion ADR) makes the **earth-like** family first, and the
fastest credible path to "Thalos looks good" is fine-tuning
[terrain-diffusion](https://github.com/xandergos/terrain-diffusion)'s released
pretrained models (MIT; 30 m and 90 m weights on Hugging Face; ETOPO + WorldClim
training data; PyTorch/diffusers + the infinite-tensor windowed-fusion runtime).
Retraining an equivalent model in Burn from scratch would forfeit the pretrained
weights that make Thalos-first cheap.

Meanwhile ADR-20260720T211046Z (offline terrain packages) always allowed
Python/PyTorch **behind the package boundary**: player devices consume baked
packages and never run the planetary model. The drift objection therefore only
bites for models that must execute in Rust — the client-side close-band
reconstruction (`Rclient`, MIRA-3) and any future bundled authoring tool.

The user's standing direction: fine-tune now, but **eventually one unified model
architecture for all bodies**.

## Decision

- **Earth-like planetary-band models are authored by fine-tuning
  terrain-diffusion in PyTorch**, run offline as a package producer. The package
  is the durable boundary; the runtime never depends on the Python stack.
- **Burn remains the Rust-side stack** for everything that runs on player
  machines or inside the Thalos process: the `Rclient` close-band path, and the
  airless family (the MIRA-1 v5b line), which finishes its L2 gate and then
  pauses per the companion ADR.
- **The end state is one unified architecture family across all body classes**:
  shared cascade structure (parent-conditioned residual bands), shared
  conditioning vocabulary (unit direction, scale level, body seed, physical
  sample spacing, body-class/climate channels), shared cube-sphere addressing and
  seam strategy — with per-body-class weights. The interim two-stack divergence
  (PyTorch earth-like / Burn airless) is accepted as bounded and temporary: when
  the unified architecture is designed, it is implemented **once**, in whichever
  stack then serves both the offline producer and any Rust-resident consumer,
  and the other line is retired. Divergence in *architecture* (not just stack) is
  the thing to actively resist — new capabilities added to one family should be
  expressed in the shared conditioning vocabulary so the unification stays
  mechanical.

ADR-20260721T033713Z is **amended**: its Rust-once rule now scopes to
Rust-resident models rather than all learned terrain. Its Burn-over-Candle choice
and its package-first gameplay rule are unchanged.

## Alternatives

- **Retrain the earth-like family from scratch in Burn on ETOPO/WorldClim.**
  Rejected: forfeits released pretrained weights, and reaching parity means
  reproducing a multi-week training campaign before the first Thalos-relevant
  result — the opposite of the keystone's pacing. Remains the fallback if the
  fine-tune path hits a wall (license, quality ceiling, conditioning
  inflexibility).
- **Port the fine-tuned weights to Burn for inference.** Rejected for the
  planetary bands: they never run in-game (packages are the boundary), so the
  port buys nothing and adds a subtle op-for-op mismatch risk between two
  implementations of the same sampler.
- **Adopt PyTorch for everything, including `Rclient`.** Rejected: client-side
  reconstruction must run deterministically inside the shipped Rust game process
  on player hardware; embedding a Python/libtorch runtime there contradicts
  ADR-20260721T033713Z's still-standing consequences and the package-first rule.
- **Declare the unified architecture now and build both families onto it.**
  Rejected: premature — the earth-like fine-tune has not yet revealed which of
  terrain-diffusion's architectural choices (Laplacian elevation encoding, its
  cascade coupling, climate co-generation) Thalos keeps. Unification is an
  end-state decision made from evidence of both families working.

## Consequences

- A pinned Python environment (PyTorch/diffusers/infinite-tensor fork) becomes
  part of the offline producer toolchain; it needs the same determinism
  discipline as the Rust side (frozen model artifacts, content hashes, pinned
  seeds) since package reproducibility depends on it.
- Fine-tune provenance must be recorded: upstream MIT attribution travels with
  the fork; base-model version + fine-tune dataset + config are part of the
  package's producer identity (the schema's model-identity fields already exist,
  MIRA-0).
- The Burn investment (v5b architecture, corpus tooling, resume machinery) is
  retained, not stranded: it is the proof-of-capability for Rust-resident models
  and the airless family's home until unification.
- Cross-family drift is a standing review item: `ntr` carries an open decision
  row for the unified-architecture design, to be scheduled once the earth-like
  fine-tune has produced accepted Thalos terrain.
