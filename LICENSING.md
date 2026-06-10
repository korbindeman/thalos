# Licensing

Thalos is fully source-available. The licensing is deliberately split so that
the **game itself stays mine to sell**, while the **content ecosystem around it
is open** — including commercially.

The one-line policy: **you can't sell the game; you can sell content for it.**

## What's covered by what

| Layer | License | What it means |
|---|---|---|
| **Source code** | [PolyForm Noncommercial 1.0.0](LICENSE) | Use, modify, fork, and redistribute for any **noncommercial** purpose. Selling the game (or other commercial use of the code) is reserved to the copyright holder. |
| **Assets** (art, audio, authored content incl. RON under `assets/` and `ships/`) | [CC BY 4.0](LICENSE-ASSETS) | Share and adapt for **any** purpose, **including commercially**, with attribution. Paid planet packs / part packs built on Thalos content are explicitly fine. |
| **`crates/udlod`** (vendored) | MIT OR Apache-2.0 | Upstream license, kept intact. |
| **`crates/volumetric_clouds`** (vendored) | MIT | Upstream license, kept intact. |
| **The name "Thalos" and the logo** | Not licensed | Reserved. See "Trademark & brand" below. |

## Why this split

The community is welcome — encouraged — to fork Thalos and make it better. The
goal is an open ecosystem where the best community art and content can flourish,
and a paid modding economy can exist on top of it (potentially via an in-game
marketplace).

- **Code is noncommercial** so the game itself isn't resold out from under the
  project. Anyone can compile and play for free; revenue relies on goodwill,
  official builds, and the marketplace.
- **Assets are CC BY (commercial-allowed)** because the modding economy needs
  it: a CC BY-**NC** asset license would forbid the paid packs we *want*.
  Plain CC BY (not BY-SA / ShareAlike) lets a modder ship a proprietary, paid
  pack built on the base assets — ShareAlike would force every derivative back
  open and undermine paid packs.
- Because assets are CC BY (inbound = outbound), community asset improvements
  can flow back into the official build with attribution.

## Contributing

By default, **inbound = outbound**: contributions are accepted under the same
license as the layer they touch (PolyForm Noncommercial for code, CC BY 4.0 for
assets).

There is currently **no Contributor License Agreement (CLA)**. Practical
consequence: community **code** contributions are noncommercial-licensed, so
they cannot be included in a commercial/paid build of the game until a CLA is in
place. Community **asset** contributions, being CC BY, *can* be used in the paid
build (with attribution). A CLA may be added later to close the code gap.

## Vendored crates

Some crates under `crates/` are forks of third-party projects and keep their
**upstream licenses** — these override the repo-wide code license for those
directories:

- **`crates/udlod`** — MIT OR Apache-2.0, forked from
  [`kurtkuehnert/bevy_terrain`](https://github.com/kurtkuehnert/bevy_terrain).
- **`crates/volumetric_clouds`** — MIT, forked from `evroon/bevy-volumetric-clouds`.

Their `LICENSE`/attribution files travel with the source and must stay intact.

## Trademark & brand

The **"Thalos" name and logo are not licensed** by any of the above. Forks are
welcome, but may not present themselves as "Thalos" or as official builds. The
brand, the official releases, and the marketplace are the project's commercial
moat — not the bits.

---

*This document explains the intent in plain language. Where it differs from the
actual license texts ([LICENSE](LICENSE), [LICENSE-ASSETS](LICENSE-ASSETS)), the
license texts govern.*
