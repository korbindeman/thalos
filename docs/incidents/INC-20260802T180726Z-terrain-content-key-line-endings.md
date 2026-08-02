# INC-20260802T180726Z — Checkout line endings invalidated terrain packages

## Symptom

The first `v0.1.0` Windows distribution build compiled successfully, then its
extracted-package verifier rejected `Mira.bin` as stale. The package carried
content key `d82afad744437ccb`, which matched Linux/macOS and the checked-in
bake, while the Windows executable expected `01500082c9584eba` from the same
commit and authored assets.

## Root cause

`thalos_terrain`'s build script folded the raw bytes of every Rust source file
into `THALOS_TERRAIN_SOURCE_HASH`. Git may materialize text as CRLF on Windows;
this repository did not force LF through `.gitattributes`. Identical Rust source
therefore produced a different source hash and terrain content key depending on
the checkout platform.

The package verifier worked correctly: it prevented a release whose Mira body
would have degraded at runtime.

## Fix

- Hash Rust sources after canonicalizing CRLF to repository-form LF.
- Lock the behavior with a test proving LF and CRLF inputs hash identically
  while a real source edit still changes the hash.
- Split the expensive Windows compilation and cheap package verification into
  separate Actions jobs. The compiled executable is retained as an intermediate
  artifact, so a packaging-only retry no longer recompiles the game.

## Recurrence tell

A terrain package key differs by build platform while the source revision,
authored inputs, and package bytes are identical.
