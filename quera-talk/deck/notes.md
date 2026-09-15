# Speaker notes

Keyed by the stable conceptual id in `slides.json` — page numbers move as builds change, ids don't.
Transitions, timing targets, and evidence pointers live here, not on audience-facing slides.
Speaker cues are a first draft to adapt, not a script (`quera-talk-outline.md` is the source draft
this file transcribes and keeps in sync with the actual `.typ` content).

Rehearsal budget (initial pacing target, retime after a spoken rehearsal with real figures):

| Section | Ids | Prepared time | Cumulative |
| --- | --- | ---: | ---: |
| Introduction | intro-title … intro-path-to-problem | 2:30 | 2:30 |
| Being strong | pauli-observable-evolution … end-of-gate-truncation | 6:30 | 9:00 |
| Memory bottleneck | baseline-performance … memory-budget | 5:30 | 14:30 |
| Being smart | memory-opportunities … sort-merge-work-unit | 11:00 | 25:30 |
| Results and tuning | correctness-accuracy … distributed-and-hashing | 6:00 | 31:30 |
| Technical conclusion | approach-summary, scope-next-steps | 1:30 | 33:00 |
| QuEra vision | why-quera … responsibility-growth | 12:00 | separate segment |

Total technical slot: 40 min including interruptions (~33 min prepared + ~7 min buffer). Vision:
10–15 min (12 min prepared). Then Q&A. If time runs long: cut kernel history detail first, then
slide 27's tuning depth — never cut the rotation update, bucket-pair argument, local algorithm, or
correctness evidence, and never cut slide 28's distributed capacity / hash-engineering result.

## 00 — Introduction

**intro-title** (0:45) — "This is a Dutch expression I grew up with. Pauli propagation is an
interesting place to test it: we can exploit an enormous number of very cheap operations, but
eventually we have to think about how we organize those operations. By the end, I hope to make
the case for being both." → transition: the experience that led to this problem.

**intro-who-i-am** (0:45) — "Most of my background is in tensor networks and symmetry-aware
numerical algorithms, primarily in Julia. The part I particularly enjoy is turning that
mathematical structure into software that other researchers can use. This project gave me a
chance to apply the same habits to a different representation and a new language." **Fill in:**
current role/affiliation, 3–4 contribution labels. → transition: the original PauliStrings.jl
encounter.

**intro-path-to-problem** (1:00) — "While reviewing submissions for the JuliaCon quantum
minisymposium, I encountered PauliStrings.jl... I contributed several improvements, but never
found a threading strategy I was satisfied with. Preparing for this opportunity gave me a concrete
reason to return to that question." **Caution:** discuss the public project and your own
experience without revealing submission-review details. → transition: "Let me first show you what
one of these updates actually has to do."

## 01 — Being strong: Pauli propagation

**pauli-observable-evolution** (1:00) — expand the observable in Pauli strings, update through the
circuit; terms can grow rapidly, so truncation matters. → transition: a concrete running example.

**kicked-ising-benchmark** (1:00) — **Fill in exactly:** gate ordering, angles, initial state,
observable, depth range, from the matched published source (PauliPropagation paper or original
IBM experiment — pick one, don't mix conventions). Cite the source here. Distinguish qubit count
from retained term count. → transition: zoom into one term's representation.

**pauli-encoding** (1:15, 3 builds) — "The label of a product is an XOR. Commutation is another
small collection of bit operations. We still have to track phases correctly, but we can separate
that bookkeeping from the label routing that matters later." Build order: encoding table → XOR
product rule → commutation test. State the bit convention used in the actual code; XOR alone does
not determine the full operator product (phase is separate). → transition: apply to gates.

**clifford-and-rotation-gates** (1:15) — Cliffords: one output label up to sign. Rotations: a
second branch at `p ⊕ g` when anticommuting. **If showing the full rotation formula,** use the
exact conjugation/angle convention from the implementation — otherwise label branches by
coefficient only. → transition: why simple updates become a hard collection problem.

**growing-pauli-sum** (1:00) — generating a branch vs. assembling the resulting sum are different
costs; the data structure (sorted vs. hash map) determines the memory traffic of the second. →
transition: how growth is controlled.

**end-of-gate-truncation** (1:00) — combine, then truncate, at the gate boundary.
`CoefficientThreshold` (fully summed coefficients, inside the merge) and `ApproxTopN` (2048
exponent bins, can undershoot) are both implemented; a weight cutoff is background at most. **Fill
in:** the exact matched cutoff configuration for the main tolerance sweep. → transition: "With the
workload and approximation fixed, we can ask where the time goes."

## 02 — Optimizations and the memory bottleneck

**baseline-performance** (1:00) — introduces the recurring two-panel figure. **Fill in:** hardware
and configuration, workload growth description, error-bar/repetition method, package names
(neutral in-slide, versions in notes). → transition: first optimization hypothesis.

**kernel-optimization** (0:45) — **preserve real history**: name the actual optimization
attempted; if something (e.g. SIMD) was only evaluated conceptually, call it a hypothesis, not an
implemented experiment. Annotate single-thread cycles/update or clearly labeled equivalent
cycles/update at a fixed reference frequency. → transition: why more workers didn't straightforwardly help.

**threading-obstacle** (1:00) — one attempted decomposition (e.g. worker-local dictionaries +
consolidation); name the measured limiting cost. Keep other attempts (parallel sort/merge, etc.)
in backup with their actual measurements. → transition: inspect the cost directly.

**profiling-results** (1:00) — a cropped, readable profile with 2–3 annotations; a flame graph
alone is not proof of bandwidth saturation. Add a relevant counter or scaling observation if
available. → transition: quantify expected traffic.

**memory-budget** (1:45) — bytes/update, estimated reads+writes, `R ≲ B_sustained / T_bytes/update`.
Compare the estimate against measured throughput on the actual workload; translate to
equivalent-cycles/update for the single-thread case. Distinguish payload bytes from actual traffic
(temporaries, writes); a single thread may not saturate available bandwidth. Mark only *measured*
memory failures on the runtime panel. → transition: "To make a larger difference, I needed to
change how the data moves through the computation."

## 03 — Being smart: structured partitioning

*(Build slides 16–23 first — outline's own preparation order. They carry the core explanation.)*

**memory-opportunities** (0:45) — four distinct opportunities (smaller representation, fewer
transfers, more reuse, more bandwidth) plus a separate capacity-limit note. → transition: the
small ordering problem that suggested the structure.

**bit-flip** (0:45, 2 builds) — the checked example: sorted `[0,1,2,4,5,7]` XOR 2 →
`[2,3,0,6,7,5]`. "The operation is trivial, but the output is no longer sorted. Do I need to sort
everything again?" → transition: reveal the order that survives.

**two-sorted-streams** (1:00, 3 builds) — bit-0 subsequence `[0,1,4,5]→[2,3,6,7]`, bit-1
`[2,7]→[0,5]`, merged `[0,2,3,5,6,7]`. O(N) linear merge, **not** the O(N log(N/b)) cost of
independently sorting balanced buckets — a different operation, keep the distinction explicit if
asked. → transition: from preserved ordering to predictable movement.

**whole-bucket-movement** (1:00) — XOR either preserves or swaps a whole bucket; this specific
ordering property was particular to the example, the generalization keeps routing, with local
ordering handled separately. → transition: "Can we construct many buckets with this same routing
property?"

**predictable-destination-rule** (1:00) — `h(p⊕g) = h(p)⊕h(g)`; ordinary hashing doesn't generally
give this. **Backup:** the precise sense in which this requirement implies an affine/linear map. →
transition: a hash family that supplies it directly.

**gf2-linear-hash** (1:30) — `h(p)=Ap`, rank-k ⇒ 2^k buckets = cosets of ker A. Equal coset sizes
in the full space do not guarantee equal occupancy in the actual propagated operator. → transition:
apply to a rotation's two branches.

**rotation-bucket-pairs** (1:30, 3 builds) — `d=h(g)=101`, four pairs (000/101, 001/100, 010/111,
011/110); stay-or-partner-at-`b⊕d`; applying twice returns to `b`, so the bucket graph splits into
disjoint pairs. `d=0` is the singleton case, added as a small final build. → transition: generalize
past rotations.

**coset-closure** (2:00, 2 builds) — `V=span(h(D))`, work units = `b+V`; the full timed-gate
pseudocode. Rotation pairs are the special case `D={g}`. A task owns every partition its coset can
exchange with (read+write in place); a Clifford can still split one partition across destinations
while remaining inside a closed coset. **Scope reminder:** histogram-based total-count truncation
may need information beyond one coset — that cost is inside the timed gate, not free. → transition:
sort-merge inside the owned set.

**sort-merge-work-unit** (1.30 nominal outline / adjust) — `fill_coset` swaps source columns into
scratch; identity-delta stream skips sorting, rest stream is sorted before `merge2_into`; dense
identity plans can borrow key columns; adaptive comparison-vs-radix sort chosen per layer. Use
measured traffic/cache evidence for the gain, not code structure alone. → transition: "We can now
measure the complete gate application."

## 04 — Results and tuning

**correctness-accuracy** (0:45) — **Fill in:** the actual comparison run (matched
existing-library result, or published observable/convergence plot at matched parameters), the
benchmark source, and truncation policy in force. Not a proposed checklist — actual completed
checks only.

**performance-bucketed** (1:30) — bucketed 1-thread joins the recurring figure. **Fill in:**
measured throughput change, tolerance-sweep runtime change; versions/hardware/thread
counts/timing boundaries/truncation semantics go in notes/backup, not the slide. Credit relevant
packages neutrally.

**bucketing-vs-threading** (1:00) — baseline vs. bucketed-1t vs. bucketed-mt at matched
workloads; isolates bucketing gain from parallel speedup. **Disclose** if the baseline uses
different kernels/other changes — an ablation only isolates what it actually controls.

**thread-scaling-tuning** (1:30) — fixed-workload scaling curve + ideal reference; speedup is
relative to the bucketed algorithm on one thread, **not** the original baseline. Small bucket-count
sweep if legible — split to 27b rather than rushing. Don't infer cache fit from a knee alone;
use measured evidence.

**distributed-and-hashing** (1:15 + rehearsal adjustment) — distributed run extends the tolerance
range past single-node memory limits; label nodes/threads/per-node & aggregate memory/peak string
count. **Essential hash-engineering build** (backup or 28b if it can't fit): `PartitionRows::cut`
(z-only block parity — X rotations and within-block ZZ avoid cross-rank movement; cut-edge ZZ can
communicate) vs. random rows, at fixed workload: comm bytes/gate, comm time, full gate time, plus
occupancy (reduced communication doesn't by itself mean balanced work). **Scope:** independent
arithmetic isn't communication-free; use `result="local"` + reduced scalar observable for the
capacity demo, not full gather on rank 0.

## 05 — Technical conclusion

**approach-summary** (0:45) — the routing equation, the strongest *supported* result, its scope.
Return to the idiom.

**scope-next-steps** (0:45) — demonstrated scope stated precisely (not an unqualified
"kernel-/truncation-independent" claim); next: API needs, docs, validation breadth, integration;
GPU/distributed extensions brief and conditional.

## 06 — QuEra vision

**why-quera** (1:30) — dynamics/noise/reusable-software themes from the role description; don't
imply knowledge of internal priorities not actually communicated in interviews.

**tensor-network-background** (1:30) — one contribution explainable in under a minute — pick the
clearest, not a second technical lecture. **Fill in:** project, contribution, supported outcome.

**querakit-architecture** (1:45, 2 builds) — need → responsibility → effect, one concrete example.
Candidate topics: interface design, cross-package integration, differentiation support, mentoring.
Pick the one with the clearest evidence of *your own* responsibility — no invented adoption counts
or collaborator statements.

**scientific-workflow-example** (2:15) — hypothetical noisy-control workflow: agree on quantity/
accuracy needed → tractable reference → scale-appropriate method (Pauli propagation need not be
selected). Present as a possible workflow, not a roadmap assertion.

**problem-approach** (1:45) — four guiding questions (scientific output / what's validated / what
limits it / what to integrate). **Optional Rust/LLM sentence:** one factual sentence on tasks
assisted + your actual review process, if you want to address it proactively — keep a detailed
account ready for follow-up questions.

**joining-the-team** (1:45) — learn workflow → agree one bounded contribution → reference baseline
→ integrate/document → assess next need. Avoid speculative dated promises.

**responsibility-growth** (1:30) — own substantial components, contribute to architecture, support
colleagues. **Close with the discussion prompt:** "Where do you see the biggest gap today between
the simulations the team needs and what the current tools make practical?"

## Backup slides (answer on demand, not a second lecture)

Phase/rotation conventions · exact benchmark definition · correctness/convergence detail ·
benchmark environment (versions, compilers, hardware, threads, timing boundaries) · affine
necessity proof (`h(p)⊕h(0)` linear from the translation requirement) · multiple generators
(hash-space orbits in cosets of `D=span{Ag_1,...,Ag_m}`) · Clifford handling (`AS=BA` for
whole-partition routing vs. closed work-unit cosets) · local ordering guarantees · hash occupancy
under structured/imbalanced operators · which truncation policies stay local vs. need
coordination · memory traffic/cache evidence (bandwidth vs. latency) · NUMA/distributed
ownership (where data lives, when it moves) · Rust/LLM assistance detail · where this method is a
poor choice.

## First cuts if rehearsal runs long

1. Additional unsuccessful optimization approaches (backup only).
2. Detailed hash construction/tuning depth on **thread-scaling-tuning**.
3. Distributed mechanics beyond the required capacity/speedup/hash-communication results.
4. Full necessity proof and Clifford condition (keep accurate scope language in the main talk
   regardless).

Never cut: the rotation update, the bucket-pair argument, the correctness evidence, or the QuEra
vision segment.
