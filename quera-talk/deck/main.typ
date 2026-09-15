// Pauli propagation at scale — QuEra technical interview talk.
// Entry point: selects prototype (6 concepts) or full (37 conceptual slides)
// via --input mode=prototype|full (default full). Section order and content
// live in sections/*.typ; theme in theme/organic.typ + theme/extensions.typ;
// shared talk-specific helpers in components.typ. See README.md to build.

#import "theme/organic.typ": *
#import "theme/extensions.typ": *
#import "components.typ": *

#let mode = sys.inputs.at("mode", default: "full")
#let interview-date = sys.inputs.at("date", default: "[interview date TBD]")

#set text(font: font-body, size: t-body)
#folio-note.update("Draft — not for distribution")

// Each section file exports one #let render(mode, ..) function so main.typ
// stays the single place that decides build mode and slide order; sections
// never read sys.inputs themselves.
#import "sections/00-introduction.typ": render as intro
#import "sections/01-propagation.typ": render as propagation
#import "sections/02-memory.typ": render as memory
#import "sections/03-partitioning.typ": render as partitioning
#import "sections/04-results.typ": render as results
#import "sections/05-conclusion.typ": render as conclusion
#import "sections/06-quera.typ": render as quera

#intro(mode, date: interview-date)
#propagation(mode)
#memory(mode)
#partitioning(mode)
#results(mode)
#conclusion(mode)
#quera(mode)
