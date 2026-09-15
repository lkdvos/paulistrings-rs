// Section 0 — Introduction (outline ids 1-3, slides 1-3, ~2:30 prepared).
// intro-title is prototype P1; the other two are scaffolded from settled
// outline content, ready for Lukas's own timeline/background details.

#import "../theme/organic.typ": *
#import "../theme/extensions.typ": *
#import "../components.typ": *

#let render(mode, date: "[interview date TBD]") = {
  // P1 — Cover. Full submitted title with a deliberate break before the
  // idiom; the gloss is the only subtitle, no invented numerical claim.
  slide("intro-title", mode, () => cover-slide(
    [Pauli propagation at scale: \ "Wie niet sterk is, moet slim zijn"],
    subtitle: ["If you aren't strong, you have to be clever."],
    meta: (
      "Lukas Devos",
      "QuEra Computing — technical interview",
      "Sept 15, 2026",
    ),
  ))

  slide("intro-who-i-am", mode, () => content-slide("Background")[
    #bullets((
      [Background: tensor networks and symmetry-aware numerical algorithms, primarily in Julia],
      [Software research fellow at CCQ, The Flatiron Institute],
      [Selected software contributions — #text(fill: accent-700, "[slot: 3-4 short contribution labels, not a logo wall]")],
    ))
  ])

  slide("intro-path-to-problem", mode, () => content-slide("How I arrived at this problem")[
    #bullets((
      [JuliaCon 2024: encountered PauliStrings.jl while reviewing quantum minisymposium submissions],
      [Contributed several CPU-side improvements while learning about caches and pipelines],
      [Never found a threading strategy I was satisfied with],
      [This project: a Rust implementation built to answer that open question],
    ))
  ])
}
