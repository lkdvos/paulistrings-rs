// Section 4 — Results and tuning (outline ids 24-28, slides 24-28). Pure
// scaffold: no prototype owns this section, but it reuses baseline-
// performance's recurring-figure exactly, per fig-stage in components.typ.

#import "../theme/organic.typ": *
#import "../theme/extensions.typ": *
#import "../components.typ": *

#let render(mode) = {
  slide("correctness-accuracy", mode, () => content-slide("Correctness and comparable accuracy")[
    #bullets((
      [Consistency check: #text(fill: accent-700, "[slot: matched existing-library result or published observable/convergence plot]")],
      [Benchmark source and end-of-gate truncation policy stated alongside the comparison],
      [Timings that follow use #text(fill: accent-700, "[slot: matching settings, or explicitly described differences]")],
    ))
  ])

  slide("performance-bucketed", mode, () => content-slide("Single-thread performance with bucketing", align-center: true)[
    #recurring-figure(variants-through: fig-stage.performance-bucketed)
    #v(baseline - 20pt)
    #text(size: t-kicker, fill: ink-55)[
      Bucketed 1-thread throughput: #text(fill: accent-700, "[slot: measured result]") · tolerance-sweep runtime: #text(fill: accent-700, "[slot: measured result]")
    ]
  ])

  slide("bucketing-vs-threading", mode, () => content-slide("Separating bucketing from threading", align-center: true)[
    #recurring-figure(variants-through: fig-stage.bucketing-vs-threading)
    #v(baseline - 20pt)
    #text(size: t-kicker, fill: ink-55)[
      Baseline vs. bucketed/1-thread vs. bucketed/multithread at matched workloads, thread count labeled — #text(fill: accent-700, "[slot: measured results]")
    ]
  ])

  slide("thread-scaling-tuning", mode, () => content-slide("Thread scaling and bucket tuning")[
    #block(width: 100%, height: 500pt, radius: radius-md, fill: surface, inset: 24pt)[
      #align(center + horizon, text(fill: ink-55, style: "italic", "[slot: fixed-workload thread-scaling curve with ideal reference; small bucket-count sweep]"))
    ]
    #v(baseline)
    #text(size: t-small, fill: ink-70, "Speedup is defined relative to the same bucketed algorithm on one thread, not the original baseline.")
  ])

  slide("distributed-and-hashing", mode, () => content-slide("Multiple nodes and communication-aware hashing", align-center: true)[
    #recurring-figure(variants-through: fig-stage.distributed-and-hashing)
    #v(baseline - 20pt)
    #text(size: t-kicker, fill: ink-55)[
      Nodes: #text(fill: accent-700, "[slot]") · peak strings: #text(fill: accent-700, "[slot]") · per-node/aggregate memory: #text(fill: accent-700, "[slot]")
    ]
    #v(baseline - 30pt)
    #text(size: t-kicker, fill: ink-55)[
      Hash comparison: #raw("PartitionRows::cut") (z-only block-parity) vs. random rows — communication bytes/gate, communication time, full gate time at fixed workload. #text(fill: accent-700, "[slot: measured comparison + occupancy]")
    ]
  ])
}
