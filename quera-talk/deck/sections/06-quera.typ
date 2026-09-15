// Section 6 — QuEra vision (outline ids 31-37, slides 31-37, ~12:00).
// querakit-architecture is prototype P6; the rest are scaffolded from
// settled outline content.

#import "../theme/organic.typ": *
#import "../theme/extensions.typ": *
#import "../components.typ": *

#let render(mode) = {
  slide("why-quera", mode, () => content-slide("Why QuEra", align-center: true)[
    #stat-row((
      ("Dynamics", "Scientific simulation connected to device-relevant questions", accent),
      ("Noise", "Modeling what actually limits a device", accent2-750),
      ("Reusable software", "Interfaces and maintenance matter as much as performance", accent),
    ))
    #v(baseline)
    #text(size: t-small, fill: ink-70, "I want to work closer to the scientific questions that determine which tools are most useful, in an environment where interface design and helping colleagues use the software have sustained attention.")
  ])

  slide("tensor-network-background", mode, () => content-slide("What I bring from tensor networks")[
    #bullets((
      [Careful attention to representation, approximation, and the numerical operations that dominate a calculation],
      [Example: in #text(fill: accent-700, "[slot: project]"), responsible for #text(fill: accent-700, "[slot: contribution]"), which enabled #text(fill: accent-700, "[slot: supported outcome]")],
    ))
    #v(baseline)
    #text(size: t-small, fill: ink-70, "Those habits apply when choosing and implementing other simulation methods as well.")
  ])

  // P6 — QuantumKitHub architecture and collaboration. A concrete example
  // linking a user/developer need to Lukas's contribution and its effect;
  // no invented adoption counts, collaborator statements, or outcomes.
  progressive-slide("querakit-architecture", mode, 2, stage => content-slide("QuantumKitHub: architecture and shared software")[
    #set par(spacing: 0pt)
    #let card(title, body, tcol: ink) = block(fill: neutral-200, radius: radius-lg, inset: 32pt, width: 100%, height: 280pt)[
      #text(font: font-heading, weight: heading-weight, size: t-small, fill: tcol, title)
      #v(baseline - 30pt)
      #text(size: t-small, fill: ink-70, body)
    ]
    #grid(columns: (1fr, 1fr, 1fr), column-gutter: 42pt,
      card("The need", [#text(fill: accent-700, "[slot: user or developer need]")]),
      card("My responsibility", [#text(fill: accent-700, "[slot: design and maintenance contribution — interface design, cross-package integration, differentiation support, or mentoring]")], tcol: accent-700),
      card("The effect", [#text(fill: accent-700, "[slot: effect on other developers or users]")]),
    )
    #v(baseline)
    #reveal(stage >= 2)[
      #text(size: t-kicker, fill: ink-55, "I care about this as much as performance: the interface and continuing maintenance determine whether other people can build on the method.")
    ]
  ])

  slide("scientific-workflow-example", mode, () => content-slide("A scientific workflow I would like to support")[
    #bullets((
      [Concrete question: #text(fill: accent-700, "[slot: e.g. how a control choice affects an observable in a noisy model]")],
      [Model assumptions, then a reference calculation in a tractable regime],
      [A scalable solver chosen for the regime — Pauli propagation need not be the selected method],
      [Comparison of scientifically relevant outputs colleagues can interpret and repeat],
    ))
    #v(baseline)
    #text(size: t-kicker, fill: ink-55, "Presented as a possible workflow, not an assertion about QuEra's current roadmap.")
  ])

  slide("problem-approach", mode, () => content-slide("How I approach a new problem")[
    #bullets((
      [What is the scientific output?],
      [What can we validate?],
      [What limits the current approach?],
      [What change can we integrate and measure?],
    ))
    #v(baseline)
    #text(size: t-small, fill: ink-70, "In a team setting, I also want users involved in choosing the target and assessing whether the result solves their problem.")
  ])

  slide("joining-the-team", mode, () => content-slide("How I would start with the team")[
    #bullets((
      [Learn the current workflow and where the recurring difficulties are],
      [Agree on one bounded contribution with a clear scientific output],
      [Establish reference behavior, then implement and integrate],
      [Assess the next need],
    ))
  ])

  slide("responsibility-growth", mode, () => content-slide("The responsibility I want to grow into")[
    #bullets((
      [Own substantial components of scientific software],
      [Contribute to architecture — how pieces fit together],
      [Support colleagues, building on existing maintenance and mentoring experience],
    ))
    #v(baseline)
    #text(size: t-body, fill: accent-700, style: "italic", "Where do you see the biggest gap today between the simulations the team needs and what the current tools make practical?")
  ])
}
