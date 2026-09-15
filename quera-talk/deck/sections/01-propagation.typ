// Section 1 — Being strong: Pauli propagation (outline ids 4-9, slides 4-9).
// pauli-encoding is prototype P2; the rest are scaffolded from settled
// outline content, with bracketed slots for circuit/benchmark specifics.

#import "../theme/organic.typ": *
#import "../theme/extensions.typ": *
#import "../components.typ": *

#let render(mode) = {
  slide("pauli-observable-evolution", mode, () => content-slide("Evolving an observable")[
    #set par(spacing: 0pt)
    #grid(columns: (1fr, auto), column-gutter: 64pt, align: horizon,
      [
        #math-setup[
          $ angle.l O angle.r = "Tr"(rho U^dagger O U), quad O = sum_p c_p P_p $
        ]
        #v(baseline)
        #bullets((
          [The central object is a sparse operator, expanded in Pauli strings],
          [The number of terms can grow rapidly — practical calculations need truncation],
        ))
      ],
      circuit-diagram(width: 620pt, height: 280pt),
    )
  ])

  slide("kicked-ising-benchmark", mode, () => content-slide("The kicked Ising benchmark")[
    #set par(spacing: 0pt)
    #grid(columns: (1fr, auto), column-gutter: 64pt, align: horizon,
      [
        #bullets((
          [Heavy-hex connectivity, #text(fill: accent-700, "[slot: gate ordering]"), #text(fill: accent-700, "[slot: angles]")],
          [Initial state: #text(fill: accent-700, "[slot: initial state]"); observable: #text(fill: accent-700, "[slot: observable]")],
          [Depth range: #text(fill: accent-700, "[slot: depth range]")],
          [The same circuit is used throughout the talk as a demanding, reproducible workload],
        ))
      ],
      hex-lattice-diagram(heavy-hex-flower, R: 42pt),
    )
  ])

  // P2 — Pauli encoding. Starts single-qubit, then generalizes to a full
  // string — verifies display, body, math, monospace and numeric-table
  // typography together (handoff requirement for this slide).
  progressive-slide("pauli-encoding", mode, 4, stage => content-slide("Representing a Pauli string")[
    #set par(spacing: 0pt)
    #grid(columns: (1fr, auto), column-gutter: 96pt, align: horizon,
      [
        #text(size: t-body, fill: ink)[One qubit: a Pauli operator is a pair of bits $(x,z)$.]
        #v(baseline - 10pt)
        #reveal(stage >= 2, text(size: t-body, fill: ink)[
          #h(0pt) $n$ qubits: stack the bits, $p=(x,z)$ with $x,z in {0,1}^n$ — #raw("x: [u64; W]") and #raw("z: [u64; W]") in code.
        ])
      ],
      data-table(
        header: ("", "I", "X", "Z", "Y"),
        rows: (([x], [0], [1], [0], [1]), ([z], [0], [0], [1], [1])),
        numeric-from: 1,
        cols: (auto, auto, auto, auto, auto),
      ),
    )
    #v(baseline)
    #math-setup[
      #reveal(stage >= 3, text(size: t-body, [A product's label is an XOR: $(p q)_"label" = p plus.o q$, with a phase $i^k$ tracked separately.]))
      #v(baseline)
      #reveal(stage >= 4, text(size: t-body, [Two strings commute exactly when, over $bb(F)_2$: $x_p dot z_q plus.o z_p dot x_q = 0$.]))
    ]
  ])

  slide("clifford-and-rotation-gates", mode, () => content-slide("Clifford gates and Pauli rotations")[
    #set par(spacing: 0pt)
    #align(center, split-diagram())
    #v(baseline)
    #text(size: t-small, fill: ink-70, "Every input follows a highly structured routing rule — the extra branch differs from the original by XOR with the generator.")
  ])

  slide("growing-pauli-sum", mode, () => content-slide("Managing a growing Pauli sum")[
    #set par(spacing: 0pt)
    #columns-body((
      ([Sorted storage], [Equal labels are adjacent after a sort; accumulation is a linear merge.]),
      ([Hash map], [Equal labels collide by hash; accumulation is a lookup and update per contribution.]),
    ))
    #v(baseline)
    #sum-structure-diagram()
    #v(baseline - 20pt)
    #text(size: t-small, fill: ink-70, "Generating a branch and assembling the resulting sparse sum are different costs — the data structure determines the memory traffic of the second one.")
  ])

  slide("end-of-gate-truncation", mode, () => content-slide("End-of-gate truncation")[
    #set par(spacing: 0pt)
    #bullets((
      [Combine a gate's contributions, then truncate — truncation is a semantic boundary, not a separate pass],
      [#raw("CoefficientThreshold") filters fully summed coefficients inside the merge],
      [#raw("ApproxTopN"): 2048 exponent bins of squared magnitudes, retains at most the requested count (can undershoot — whole bins are kept)],
      [Partitioned execution all-reduces the histogram and length before applying the shared edge],
    ))
    #v(baseline)
    #text(size: t-kicker, fill: ink-55)[
      Main tolerance sweep: #text(fill: accent-700, "[slot: published, matched cutoff configuration]")
    ]
  ])
}
