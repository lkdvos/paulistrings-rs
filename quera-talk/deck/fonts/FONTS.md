# Fonts

Project-local subset, so the deck compiles with `--font-path fonts` and does not depend on
system font installation. Each subdirectory carries its own `LICENSE.txt`.

| Family | Role | Files | License | Source |
| --- | --- | --- | --- | --- |
| Caprasimo | display (cover, dividers) | `caprasimo/Caprasimo-Regular.ttf` | OFL 1.1, copyright 2023 The Caprasimo Project Authors | google/fonts `ofl/caprasimo` |
| Figtree | body/reading text | `figtree/Figtree-{Regular,SemiBold,Bold}.ttf` | OFL 1.1, copyright 2022 The Figtree Project Authors | google/fonts `ofl/figtree` |
| STIX Two Math | mathematics | `stix-two-math/STIX2Math.otf` | OFL 1.1, STIX Font License 15 Apr 2019 | Rocky Linux `stix-fonts-2.0.2-11.el9` package |
| Source Code Pro | code | `source-code-pro/SourceCodePro-{Regular,Bold}.otf` | OFL 1.1 (Adobe) | Rocky Linux `adobe-source-code-pro-fonts-2.030.1.050-12.el9.1` package |

## Provenance and verification

Caprasimo and Figtree were already present on this build host at
`~/.local/share/fonts/organic/` (installed per the reference archive's own README, from Google
Fonts' `css2` API, TrueType). Their `LICENSE.txt` files here reproduce the standard OFL 1.1 text
with the copyright line taken from each project's `METADATA.pb` in the `google/fonts` repository;
that repository is the copy of record for exact version/build hashes, and this project pins no
specific commit — treat these as the "current Google Fonts" build until re-verified before a
real interview delivery.

STIX Two Math and Source Code Pro came from this host's Rocky Linux package manager. Their
`LICENSE.txt` files are extracted verbatim from the installed package's bundled license
(`/usr/share/licenses/stix-fonts/STIX_2.0.2_license.pdf`,
`/usr/share/licenses/adobe-source-code-pro-fonts/LICENSE.txt`) via `pdftotext`, so those two are
exact, versioned, and reproducible from the package versions named above.

## Verifying the build actually used these faces

```bash
typst fonts --font-path fonts | grep -Ei 'caprasimo|figtree|stix two math|source code pro'
```

`scripts/build.sh` runs this check and fails loudly if any of the four families is missing, so a
build never silently falls back to `organic.typ`'s documented fallback chain (URW Bookman /
Carlito / DejaVu Sans Mono) without saying so.

## If a required font cannot be obtained on another machine

`organic.typ` already documents and ships the fallback chain: Caprasimo falls through to URW
Bookman (bump `heading-weight` to 600 so Bookman lands on Demi, not Light — see the comment at
the top of `theme/organic.typ`), Figtree to Carlito. `extensions.typ`'s `code-block` falls
through to DejaVu Sans Mono. There is no bundled fallback for STIX Two Math; without it, Typst's
default math font renders instead and every equation slide should be re-checked before delivery.
