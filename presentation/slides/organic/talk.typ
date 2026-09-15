// The full talk in the Organic design system.
//
// Compile from the repo root:
//   typst compile --root . presentation/slides/organic/talk.typ
// With the real faces:
//   ... --font-path ~/.local/share/fonts/organic
//
// Figures come from `presentation/figures/organic/` and are placed at the
// exact size they were authored at (see `figures/organic/sizes.json`), so a
// 24pt tick label is 24pt on the stage. Never scale one to fit -- re-author
// it at the width it gets.

#import "organic.typ": *
#import "extensions.typ": *

#let F = "/presentation/figures/organic/"

#set text(font: font-body, size: t-body, fill: ink, lang: "en")
#set par(leading: lead-body, justify: false)
#show: math-setup
#folio-note.update("Pauli propagation · CCQ 2026")

// ═══════════════════════════════════════════════════════════════════ opening
#cover-slide(
  "Wie niet sterk is, moet slim zijn",
  subtitle: "Pauli propagation at 10⁷ terms: how a GF(2)-linear hash turned a memory-bound simulator strong and smart.",
  meta: ("Lukas Devos", "CCQ, Flatiron Institute", "2026"),
)

#contents-slide(rows: (
  "Sterk — brute force against an exponential wall",
  "The wall is memory",
  "Sterk én slim — a hash that respects XOR",
  "The gift that keeps on giving",
))

// ═════════════════════════════════════════════════════════════════════ ACT 1
#divider-slide("01", "Sterk")

#content-slide("Pay for the operator, not the state")[
  #bullets((
    [State vectors carry $2^n$ amplitudes — exact to about thirty qubits, then nothing.],
    [Tensor networks pay for entanglement, not for qubits.],
    [Pauli propagation evolves the observable and truncates the sum.],
  ))
]

#content-slide("The whole method on one line", align-center: true)[
  #v(baseline)
  #text(size: 46pt)[$ U^dagger O U = sum_P c_P P, quad P in {I, X, Y, Z}^(times.circle n) $]
  #v(2 * baseline)
  #block(width: 1300pt, text(fill: ink-70)[
    Evolve $O$ backwards through the gates, drop the coefficients that do not matter, contract with $rho_0$ at the end. Exactness traded for #emph[sparsity].
  ])
]

#quote-slide(
  [It reproduces the 127-qubit IBM utility observables, and it is bit-exact against stim at the Clifford points.],
  attribution: [Kim et al. 2023; `benchmarks/python/bench_a_clifford.py`],
)

#content-slide("A Pauli string is two bit vectors")[
  #v(baseline)
  #grid(columns: (1fr, 1fr), column-gutter: 96pt, align: horizon,
    data-table(
      header: ("", "I", "X", "Z", "Y"),
      rows: (([x], [0], [1], [0], [1]), ([z], [0], [0], [1], [1])),
      numeric-from: 1,
      cols: (auto, auto, auto, auto, auto),
    ),
    block[
      #set text(size: t-small)
      #set par(leading: 16pt)
      Two words per string, `x: [u64; W]` and `z: [u64; W]`. A product is an XOR of both and a phase $i^k$; two strings commute exactly when $x_1 dot z_2 + z_1 dot x_2 = 0$.

      #v(baseline)
      No pointers, no allocation, `Copy`.
    ])
]

#hero-slide(
  "48 B",
  [One term at 128 qubits: two sixteen-byte keys and a complex coefficient. Remember that number — every argument in this talk is a multiple of it.],
)

#content-slide("A Clifford gate is a permutation")[
  #v(baseline)
  #grid(columns: (1fr, 1fr), column-gutter: 96pt, align: horizon,
    data-table(
      header: ("Gate", "In", "Out"),
      rows: (([H], [$X$, $Z$], [$Z$, $X$]),
             ([S], [$X$, $Y$], [$Y$, $-X$]),
             ([CNOT], [$X_c$], [$X_c X_t$]),
             ([CNOT], [$Z_t$], [$Z_c Z_t$])),
      numeric-from: 9,
      cols: (auto, auto, auto),
    ),
    block[
      #set text(size: t-small)
      #set par(leading: 16pt)
      One string in, one string out. In bits the key changes only on the gate's support, by an XOR with a vector from a small delta set $D$.

      #v(baseline)
      Term count is invariant. Truncation never fires. Clifford circuits are free.
    ])
]

#content-slide("Beyond Clifford, one string becomes two", align-center: true)[
  #v(baseline)
  #text(size: 42pt)[$ R^dagger Q R = cases(Q & quad [P\, Q] = 0, cos theta thin Q + sin theta thin (i P Q) & quad {P\, Q} = 0) $]
  #v(2 * baseline)
  #block(width: 1300pt, text(fill: ink-70)[
    The anticommuting part doubles every layer. Truncation is what keeps the sum finite: drop small coefficients, drop high weight, keep the top $N$.
  ])
]

#figure-slide(
  "Three billion term updates",
  image(F + "fig0_term_growth_10steps.svg", width: 1180pt),
  caption: [127-qubit heavy-hex kicked Ising, observable $Z_62$, Heisenberg. Terms surviving after each channel, one curve per truncation threshold.],
)

#content-slide("The working point", align-center: true)[
  #v(baseline)
  #stat-row((
    ([2710], [rotations — ten Trotter steps on 127 qubits], accent),
    ([1.07 M], [terms at the peak, at $epsilon = 2^(-12)$], accent2-750),
    ([3·10⁹], [term updates in one run], accent),
  ))
  #v(2 * baseline)
  #block(width: 1300pt, text(size: t-small, fill: ink-70)[
    A CPU does $10^11$ simple operations per second. This should take three seconds.
  ])
]

#code-slide(
  "The naive algorithm",
  ```
  for gate in circuit.reversed():
      out = {}
      for (P, c) in sum:
          for (P', c') in gate.apply(P, c):
              out[P'] += c'
      sum = {P: c for P, c in out if |c| >= eps}
  ```,
  note: [Or with arrays: emit every output, sort, merge equal keys, filter. Either way, one global order per layer.],
)

#figure-slide(
  "Forty-nine seconds",
  image(F + "organic_ladder.svg", width: 1400pt),
  caption: [Median wall time for the 2710-channel circuit. This chart returns after every idea.],
)

#content-slide("Being strong: make the kernel fast")[
  #bullets((
    [A fast non-cryptographic hash — FxHash, not SipHash.],
    [Popcount and XOR are already single instructions.],
    [Let LLVM vectorize: `-C target-cpu=native`, fat LTO, one codegen unit.],
  ))
]

#figure-slide(
  "A real gain, a hundred times too small",
  image(F + "fig2_targetcpu.svg", width: 900pt),
  caption: [Paired runs, alternated abba, one thread: 3 % on the hash map, 8 % on the sorted engine — consistent in every pair, and nowhere near enough.],
)

#figure-slide(
  "The ladder, after being strong",
  image(F + "organic_ladder2.svg", width: 1400pt),
  caption: [Tuning the kernel moves the bar by a few percent. The arithmetic was never the cost.],
)

// ═════════════════════════════════════════════════════════════════════ ACT 2
#divider-slide("02", "The wall is memory")

#content-slide("Count the bytes, not the flops")[
  #bullets((
    [Read the sum, write the outputs, sort or probe them, write the merged result.],
    [At a million terms and fanout two that is 300–400 MB #emph[per layer].],
    [The arithmetic per term is a few XORs and one multiply — under a nanosecond.],
  ))
]

#content-slide("Two ways to lose")[
  #v(baseline)
  #columns-body((
    ("A hash map is random access",
     [Every probe into a fifty-megabyte table is a cache miss: about a hundred nanoseconds, one at a time, and the core waits for each one.]),
    ("A sort is streaming",
     [Every pass runs at memory bandwidth: about ten gigabytes per second per core, so thirty to forty milliseconds a layer just to move the bytes.]),
  ))
]

#data-table-slide(
  "Measured bandwidth, reference host",
  header: ("", "Placement", "Read", "Write", "Copy"),
  rows: (
    (swatch(neutral-500), [1 core], [11.3], [10.1], [9.5]),
    (swatch(neutral-600), [1 core, other socket's memory], [7.8], [7.2], [5.5]),
    (swatch(accent2-600), [8 cores, one socket], [39.0], [18.6], [35.6]),
    (swatch(accent), [16 cores, both sockets], [45.0], [25.3], [40.0]),
    (swatch(accent-800), [32 threads (SMT)], [48.8], [23.1], [33.8]),
  ),
  unit: [GB/s, STREAM-style, `crates/membench`. The spec sheet says 141 GB/s per socket: two of six memory channels are populated.],
)

#hero-slide(
  "45 GB/s",
  [Everything the machine has, shared by every core. At 48 bytes a term that is a hard ceiling of a billion term-moves per second, before any algorithm is written.],
)

#content-slide("Hyperthreads add no bandwidth")[
  #bullets((
    [The second socket adds 15–25 %, not a factor of two — a remote read runs at 7.8 GB/s.],
    [Write bandwidth is half of read.],
    [Thirty-two threads move no more bytes than sixteen cores do.],
  ))
]

#quadrant-slide(
  "Three levers, and a hard limit",
  (
    ("Fewer bytes", [Compact keys, no pointers, structure of arrays, coefficients that are not boxed.]),
    ("More work per byte", [Do everything a term needs while it is in cache. Locality, not clock speed.]),
    ("More bandwidth", [More cores, more sockets, more nodes — and the caches and capacity they bring.]),
    ("The hard limit", [$10^7$ terms is half a gigabyte; a global sort needs several times that. At $10^8$ a node runs out of memory before it runs out of time.]),
  ),
)

#content-slide("What I tried first")[
  #v(baseline)
  #columns-body((
    ("Per-thread dictionaries",
     [Split the terms into chunks, one hash map per thread, apply the gate locally — then merge every map into one at the end of the layer.]),
    ("Parallel mergesort",
     [One flat array of rows, every thread appends its outputs, then a parallel sort of the whole array and a merge pass for equal keys.]),
  ))
  #v(baseline)
  #text(size: t-small, fill: ink-70)[Both are what a competent engineer writes in an afternoon. Both keep one global order.]
]

#figure-slide(
  "Thirty-two cores, a factor of two",
  image(F + "fig3_old_attempts.svg", width: 1100pt),
  caption: [Reconstructed in Rust on the same circuit, gated against the reference engine to $10^(-9)$. Bands are min–max over repeated runs.],
)

#content-slide("Amdahl does the rest")[
  #bullets((
    [Mergesort: 1.9× at sixteen threads, then flat.],
    [Per-thread maps: never faster than one thread — the final merge re-inserts every term.],
    [Memory per term goes #emph[up]: 280 and 830 bytes against 240 for the plain map.],
  ))
]

#quote-slide(
  [We were strong. We were not smart.],
  attribution: [every attempt kept one global order, and paid for it with one global merge],
)

// ═════════════════════════════════════════════════════════════════════ ACT 3
#divider-slide("03", "Sterk én slim")

#content-slide("A hash that respects XOR", align-center: true)[
  #v(baseline)
  #text(size: 46pt)[$ h(v) = H v space (mod 2), quad h(v xor d) = h(v) xor h(d) $]
  #v(2 * baseline)
  #block(width: 1300pt, text(fill: ink-70)[
    Partition the sum into $B = 2^b$ buckets by a random binary matrix $H$. A gate changes a key by an XOR with one of at most sixteen deltas, so a bucket's output bucket is known #emph[before any term is touched].
  ])
]

#content-slide("What linearity buys")[
  #bullets((
    [A bucket's outputs go to at most $2^(dim D)$ statically known buckets.],
    [Two equal output keys have equal hashes: duplicates never straddle buckets.],
    [So deduplication is bucket-local, and there is no global sort anywhere.],
  ))
]

#figure-slide(
  "Cosets are closed tasks",
  image(F + "bucket-cosets.svg", width: 1000pt),
  caption: [Sixteen buckets, $h(D)$ spanning two vectors: four closed cosets. Arrows trace one coset's entire data movement — every edge stays inside it.],
)

#content-slide("No locks, no atomics, no barriers")[
  #bullets((
    [A coset's outputs read only the same coset: an independent, write-disjoint task.],
    [Each task sorts and merges a cache-sized run.],
    [Rayon work-steals cosets. That is the whole parallel runtime.],
  ))
]

#content-slide("Load balance comes from randomness", align-center: true)[
  #v(baseline)
  #text(size: 40pt)[$ max_beta |"bucket"_beta| <= m/B + O(sqrt((m log B)/B)) $]
  #v(2 * baseline)
  #block(width: 1300pt, text(fill: ink-70)[
    A dense random $H$ is a universal hash family, so the bound holds whatever the input looks like. Truncated sums are extremely structured — a coordinate-based partition would put everything in one bucket. The random $H$ does not care.
  ])
]

#figure-slide(
  "Where it ends up",
  image(F + "fig3_all.svg", width: 1100pt),
  caption: [Speedup against each variant's own single-thread run; the two reconstructions greyed. Sixteen physical cores, thirty-two hyperthreads.],
)

#figure-slide(
  "The ladder, finished",
  image(F + "organic_ladder7.svg", width: 1400pt),
  caption: [Single-threaded the bucketed engine is already three times faster: there is no global structure to maintain.],
)

#hero-slide(
  "10.5×",
  [Sixteen cores on this circuit — eleven to thirteen on pure rotation layers — against the factor of two that both earlier attempts stalled at. Hyperthreads still add nothing.],
)

#figure-slide(
  "The bucket size decides whether cores help at all",
  image(F + "fig4b_bucket_speedup.svg", width: 1100pt),
  caption: [Same code, same circuit, sixteen threads. Below: worker busy time over loop wall × threads, from the engine's phase counters.],
)

#content-slide("Cores bring cache, not just clocks")[
  #bullets((
    [Past about $10^4$ terms per bucket the gather run stops fitting L2.],
    [And there are fewer cosets than threads, so workers starve.],
    [Sixteen cores are sixteen megabytes of private L2 — buckets are what let the working set live there.],
  ))
]

#figure-slide(
  "One thread does not care; sixteen care a lot",
  image(F + "fig5_bucket_sweep.svg", width: 1306pt),   // sizes.json: tight-cropped
  caption: [Per-term cost at one and sixteen threads, with single-thread L2 and LLC demand-miss rates from `perf stat`, against bucket size.],
)

#content-slide("What must fit is the gather run")[
  #bullets((
    [Fanout × bucket: 1024 × 48 B × 2 ≈ 100 KB against a mebibyte of L2.],
    [Dense two-qubit unitaries have fanout sixteen: 750 KB runs, and the write ceiling at sixteen threads.],
    [Same algorithm, same bytes — only the order in which they are touched changed.],
  ))
]

#figure-slide(
  "Where the time goes now",
  image(F + "phase-shares.svg", width: 900pt),
  caption: [Share of worker busy time per phase, three layer classes, single-threaded. Gather dominates the sparse layers; the dense PTM is sort-bound.],
)

#figure-slide(
  "Against the measured ceilings",
  image(F + "roofline-threads.svg", width: 900pt),
  caption: [Sparse layers are latency-bound at half the write ceiling even at thirty-two threads; only dense PTMs reach it. The single-thread byte model over-counts DRAM traffic by 2.5–13×: the design's whole point.],
)

#figure-slide(
  "Footprint",
  image(F + "fig6_memory.svg", width: 900pt),
  caption: [Peak RSS above the process floor, per peak term. Structure of arrays, no pointers, capacity retained across layers: the steady state allocates nothing.],
)

#figure-slide(
  "Against the libraries people use",
  image(F + "baseline-ops.svg", width: 1363pt),
  caption: [Construction and Clifford conjugation. Single-threaded against PauliPropagation.jl: two to three times faster above a few thousand terms, five times less memory per term.],
)

// ══════════════════════════════════════════════════════════════════════ CODA
#divider-slide("04", "The gift that keeps on giving")

#content-slide("One structure, many wins")[
  #bullets((
    [#strong[Locality] — cache-sized runs, and cores that bring cache.],
    [#strong[Footprint] — bucket-local scratch, no global sort buffers.],
    [#strong[GPU-ready] — `#[repr(C)]` columns and a fixed fanout.],
    [#strong[A communication plan] — the same partition, one level up.],
  ))
]

#content-slide("The partition is a communication plan", align-center: true)[
  #block(width: 1400pt, text(fill: ink-70)[
    Partition on $h$ across ranks and a layer becomes a sparse, statically known exchange, computed before any term is touched. No key is ever split across ranks.
  ])
  #v(baseline)
  #stat-row((
    ([4], [of 271 layers per step cross a boundary — 139 with random rows], accent2-750),
    ([21 %], [faster per step at two partitions than one process], accent),
    ([1], [partition per MPI rank, running the same layer loop], accent2-750),
  ))
]

#content-slide("Still open")[
  #bullets((
    [A bits all-reduce on every layer, and ingestion that replicates the input per rank.],
    [The dense-PTM write ceiling: smaller gather runs, or fused passes.],
    [Rank-deficient hash seeds, and a channel-aware bucket floor.],
  ))
]

#quote-slide(
  [Wie sterk is, moet ook slim zijn.],
  attribution: [strong is 45 GB/s and every byte of L2; smart is knowing where everything goes before you move it],
)

#cover-slide(
  "Dank u wel",
  subtitle: "github.com/lkdvos/paulistrings-rs — Rust core, Python bindings, and every figure in this talk reproducible from the repository.",
  meta: ("Lukas Devos", "ldevos@flatironinstitute.org"),
)

// ═════════════════════════════════════════════════════════════════ APPENDIX
#divider-slide("05", "Appendix")

#content-slide("Things that did not work")[
  #bullets((
    [#strong[Bucket by support bits and concatenate] — provably wrong: buckets interleave. A four-term counterexample led to the linear hash.],
    [#strong[Static coset → worker placement] — 1.25–1.9× slower than work stealing in seven of eight cells.],
    [#strong[Three gather/merge variants] — recompute-in-merge, segment-copy, interleaved keys: all rejected by paired A/B.],
  ))
]

#content-slide("Things that did not work")[
  #bullets((
    [#strong[Reserving a safe upper bound for merge output] — 2.5× worse peak RSS. The merge's job is to shrink.],
    [#strong[LTO code layout is real] — one `#[inline]` hint moves the merge kernel by 6–34 %, in both directions.],
    [Every one of these is written up in `research/notes/`, with the numbers.],
  ))
]

#figure-slide(
  "Coarse against default buckets",
  image(F + "fig4_superlinear.svg", width: 900pt),
  caption: [$1.07 dot 10^6$ peak terms; coarse is 16 384 terms per bucket, sixty-four buckets. Normalised to the coarse single-thread run.],
)

#figure-slide(
  "And at four times the terms",
  image(F + "fig4_superlinear_large.svg", width: 900pt),
  caption: [$3.9 dot 10^6$ peak terms at $epsilon = 2^(-13)$. With enough cosets the coarse arm keeps up: no superlinear regime on this host and circuit.],
)

#content-slide("Reproducing every number")[
  #set text(size: t-small)
  #bullets((
    [#strong[Host] — ccqlin038, two Xeon Gold 6244, sixteen cores, two NUMA nodes, 1 MiB L2 per core.],
    [#strong[Circuit] — `heavy_hex_kicked_ising(127, 10, 5π/16, −π/2)`, observable $Z_62$, Heisenberg, `CoefficientThreshold(2^-12)`.],
    [#strong[Baselines] — `presentation/bench`, each gated against `propagate` to $10^(-9)$.],
    [#strong[Protocol] — `RUST_LOG` unset, one warm-up, medians of repeated runs, paired A/B for anything under 10 %.],
  ))
]
