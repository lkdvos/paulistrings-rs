// Section 5 — Technical conclusion (outline ids 29-30, slides 29-30).

#import "../theme/organic.typ": *
#import "../theme/extensions.typ": *
#import "../components.typ": *

#let render(mode) = {
  slide("approach-summary", mode, () => content-slide("What the approach establishes", align-center: true)[
    #math-setup[
      $ h(p plus.o g) = h(p) plus.o h(g) $
    ]
    #v(baseline)
    #text(size: t-body, fill: ink-70)[
      Respecting the XOR structure gives predictable movement between buckets — supported outcome: #text(fill: accent-700, "[slot: strongest supported performance result]")
    ]
    #v(baseline - 20pt)
    #text(size: t-small, fill: ink-55, style: "italic", "\u{201C}Wie niet sterk is, moet slim zijn.\u{201D} The hardware still matters — the algorithm makes better use of it.")
  ])

  slide("scope-next-steps", mode, () => content-slide("Scope and next steps")[
    #columns-body((
      ([Demonstrated scope], [#text(fill: accent-700, "[slot: current demonstrated scope, stated precisely — not \"kernel- and truncation-independent\" without qualification]")]),
      ([Next steps], [#text(fill: accent-700, "[slot: API needs]"), documentation, validation breadth, integration; GPU and distributed extensions kept brief and conditional]),
    ))
  ])
}
