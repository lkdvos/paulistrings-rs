// Section 2 — Optimizations and the memory bottleneck (outline ids 10-14).
// baseline-performance is prototype P5 (the recurring figure's first
// appearance and the fixed placement/typography every later stage reuses).

#import "../theme/organic.typ": *
#import "../theme/extensions.typ": *
#import "../components.typ": *

#let render(mode) = {
  // P5 — recurring figure, first appearance: real placement/typography, all
  // data slots explicitly pending. See figure-contract.json for dimensions.
  slide("baseline-performance", mode, () => content-slide("Baseline performance", align-center: true)[
    #recurring-figure(variants-through: fig-stage.baseline-performance)
    #v(baseline - 20pt)
    #text(size: t-kicker, fill: ink-55)[
      Throughput: higher is better. Runtime: lower is better. Hardware and configuration: #text(fill: accent-700, "[slot]").
    ]
  ])

  slide("kernel-optimization", mode, () => content-slide("Making the kernel faster", align-center: true)[
    #recurring-figure(variants-through: fig-stage.kernel-optimization)
    #v(baseline - 20pt)
    #text(size: t-kicker, fill: ink-55)[
      #text(fill: accent-700, "[slot: actual kernel optimization, measured single-thread cycles/update or equivalent cycles/update at a stated reference frequency]")
    ]
  ])

  slide("threading-obstacle", mode, () => content-slide("The threading obstacle", align-center: true)[
    #recurring-figure(variants-through: fig-stage.threading-obstacle)
    #v(baseline - 20pt)
    #text(size: t-kicker, fill: ink-55)[
      Attempted decomposition: #text(fill: accent-700, "[slot: e.g. worker-local dictionaries + consolidation]") — measured limiting cost: #text(fill: accent-700, "[slot]")
    ]
  ])

  slide("profiling-results", mode, () => content-slide("What profiling revealed")[
    #set par(spacing: 0pt)
    #block(width: 100%, height: 460pt, radius: radius-md, fill: surface, inset: 24pt)[
      #align(center + horizon, text(fill: ink-55, style: "italic", "[slot: cropped profile, 2-3 annotations]"))
    ]
    #v(baseline - 20pt)
    #text(size: t-small, fill: ink-70, "Locates the time; a separate memory-traffic estimate (next slide) distinguishes bandwidth, latency, or computation as the limiting resource.")
  ])

  slide("memory-budget", mode, () => content-slide("The memory budget")[
    #set par(spacing: 0pt)
    #math-setup[
      $ R_"updates" lt.tilde B_"sustained" \/ T_"bytes/update" $
    ]
    #v(baseline - 20pt)
    #recurring-figure(variants-through: fig-stage.memory-budget)
    #v(baseline - 20pt)
    #text(size: t-kicker, fill: ink-55)[
      Bytes/update: #text(fill: accent-700, "[slot]") · sustained bandwidth: #text(fill: accent-700, "[slot]") · estimate vs. measured: #text(fill: accent-700, "[slot]")
    ]
  ])
}
