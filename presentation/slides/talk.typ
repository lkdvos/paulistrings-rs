// Compile from the repo root:  typst compile --root . presentation/slides/talk.typ presentation/slides/talk.pdf
#import "lib.typ": *
#show: setup

#let F = "/presentation/figures/"
#let D = "/docs/figures/"

// ───────────────────────────────────────────────────────────────────────────── title
#title-slide[Pauli propagation at 10#super[7] terms][How a GF(2)-linear hash turned a memory-bound simulator strong _and_ smart][Lukas Devos][CCQ, Flatiron Institute · 2026]

// ═════════════════════════════════════════════════════════════════════════════ ACT 1
#section(act: "Act 1", sub: [Brute force against the exponential wall])[Sterk]

#slide[Why Pauli propagation][
  #two-col(ratio: (1.15fr, 1fr))[
    *The question.* $chevron.l O chevron.r$ after a deep circuit on 100+ qubits.

    - State vectors: $2^n$ amplitudes — exact to ~30 qubits, then nothing.
    - Tensor networks: pay for entanglement, not for qubits.
    - *Pauli propagation:* pay for the _operator_, not the state.

    $ U^dagger O U = sum_P c_P P, quad P in {I, X, Y, Z}^(times.o n) $

    Evolve $O$ backwards through the gates, keep the sum sparse by *truncation*, contract with $rho_0$ at the end.
  ][
    #punch(color: c-blue)[
      Exactness traded for _sparsity_: the sum stays small when most of the $4^n$ coefficients are negligible — deep, noisy, or weakly non-Clifford circuits.
    ]
    #v(0.4em)
    #small[Reproduces the 127-qubit IBM "utility" observables (Kim et al. 2023); bit-exact against `stim` at the Clifford points.]
  ]
]

#slide[Pauli strings as bits: the symplectic encoding][
  #two-col(ratio: (1fr, 1.1fr))[
    Each qubit's Pauli is a bit pair $(x, z)$:
    #v(0.3em)
    #align(center, table(columns: 4, inset: 6pt, stroke: 0.5pt + c-grid,
      [*I*], [*X*], [*Z*], [*Y*],
      [$(0,0)$], [$(1,0)$], [$(0,1)$], [$(1,1)$]))
    #v(0.3em)
    A string on $n$ qubits is two words: `x: [u64; W]`, `z: [u64; W]`, $W = ceil(n\/64)$.

    - 128 qubits → *32 bytes* per string, no pointers, `Copy`
    - product: $(x_1 xor x_2, z_1 xor z_2)$ and a phase $i^k$
    - commute iff $x_1 dot z_2 + z_1 dot x_2 = 0 mod 2$ — two `AND`s and a popcount
  ][
    #align(center, {
      set text(font: "Source Code Pro", size: 15pt)
      table(columns: 6, inset: 7pt, stroke: 0.5pt + c-grid, align: center,
        [qubit], [0], [1], [2], [3], [],
        [Pauli], text(fill: c-blue)[X], [I], text(fill: c-aqua)[Z], text(fill: c-orange)[Y], [],
        [x], text(fill: c-blue)[1], [0], [0], text(fill: c-orange)[1], [= `0b1001`],
        [z], [0], [0], text(fill: c-aqua)[1], text(fill: c-orange)[1], [= `0b1100`])
    })
    #v(0.6em)
    #punch[A sum of Pauli strings is an array of $(x, z, c)$ rows: 48 bytes per term at 128 qubits. Remember that number.]
  ]
]

#slide[Clifford gates permute Pauli strings][
  #two-col[
    A Clifford gate maps one Pauli string to *one* Pauli string (up to sign):
    #v(0.3em)
    #align(center, table(columns: 3, inset: 6pt, stroke: 0.5pt + c-grid,
      [*gate*], [*in*], [*out*],
      [H], [$X, Z$], [$Z, X$],
      [S], [$X, Y$], [$Y, -X$],
      [CNOT], [$X_c$], [$X_c X_t$],
      [CNOT], [$Z_t$], [$Z_c Z_t$]))
    #v(0.3em)
    In bits: the key changes only on the gate's *support*, by an XOR with a vector from a small *delta set* $D$.
  ][
    - Term count is *invariant*: $m$ in, $m$ out.
    - One table lookup per term: a permutation of the sum.
    - Truncation never triggers.

    #v(0.5em)
    #punch(color: c-aqua)[Clifford circuits are free. A stabilizer simulator does this for a single string in $O(n^2)$; we do it for $10^7$ strings at once.]
  ]
]

#slide[Beyond Clifford: one string becomes two][
  #two-col(ratio: (1.15fr, 1fr))[
    A Pauli rotation $R = exp(-i theta P \/ 2)$ acting on a term $Q$:

    $ R^dagger Q R = cases(Q & "if" [P, Q] = 0, cos theta thin Q + sin theta thin (i P Q) & "if" {P, Q} = 0) $

    - the anticommuting part of the sum *doubles* every layer: up to $2^d$ terms after $d$ kicks
    - $i P Q$ is again a Pauli string — an XOR of keys

    *Truncation* keeps it tractable: drop $|c| < epsilon$, drop weight $> k$, keep the top $N$. Noise shrinks high-weight terms and makes the problem *easier*.
  ][
    #align(center, {
      set text(size: 16pt)
      let box(body, fill) = rect(inset: 8pt, radius: 4pt, fill: fill.lighten(85%), stroke: 1pt + fill, body)
      stack(dir: ttb, spacing: 10pt,
        box([$c dot Z_1 Z_2$], c-blue),
        text(fill: c-muted, size: 13pt)[$R_X (theta)$ on qubit 1: $X_1$ anticommutes with $Z_1$],
        grid(columns: 2, column-gutter: 14pt,
          box([$c cos theta dot Z_1 Z_2$], c-blue),
          box([$c sin theta dot Y_1 Z_2$], c-orange)),
        text(fill: c-muted, size: 13pt)[next $R_X$ layer: both split again → 4, 8, 16, …])
    })
  ]
]

#slide[The exponential wall, on the talk's circuit][
  #two-col(ratio: (1.35fr, 1fr))[
    #fig(F + "fig0_term_growth.svg", caption: [127-qubit heavy-hex kicked Ising, $theta_(Z Z) = -pi\/2$, $theta_h = 5pi\/16$, observable $Z_62$, Heisenberg. Terms after each of the 1355 channels; grey bands are alternate Trotter steps.])
  ][
    - 127 qubits, 144 couplings, 271 rotations per Trotter step
    - the operator's light cone grows step by step: five steps are needed before the sum is large at all
    - working point for every benchmark that follows: *10 steps, 2710 rotations, $epsilon = 2^(-12)$* — the sum sits near *$10^6$ terms* for half the circuit
    - $10^6$ terms × ~1400 heavy layers × fanout 2 ≈ *$3 dot 10^9$ term updates*
    #v(0.3em)
    #punch[Strong means $10^9$–$10^(10)$ updates per second. Fine — a CPU does $10^(11)$ simple ops per second.]
  ]
]

#slide[The naive algorithm: a hash map per layer][
  #two-col(ratio: (1fr, 1.1fr))[
    ```
    for gate in circuit.reversed():
        out = {}
        for (P, c) in sum:
            for (P', c') in gate.apply(P, c):
                out[P'] += c'
        sum = {P: c for P, c in out
               if |c| >= eps}
    ```
    Or with arrays: emit all outputs, *sort*, *merge* equal keys, filter.

    #small[This is the engine's shipped small-sum path: hashbrown + FxHash, `Channel::apply` per term.]
  ][
    #placeholder([F1 · stage 1: naive hash map, 1 thread], height: 7cm)
    #small[Time to propagate the 1355-channel circuit at ~$10^6$ terms. This chart returns after every idea.]
  ]
]

#slide[Being strong: make the kernel fast][
  #two-col(ratio: (1fr, 1.1fr))[
    What one would do first:
    - a fast non-cryptographic hash (FxHash instead of SipHash)
    - popcount and XOR are already single instructions
    - let LLVM vectorize: `-C target-cpu=native` (AVX-512 on this box)
    - fat LTO, one codegen unit — already on

    #punch[Paired A/B, default vs `target-cpu=native`, alternated back-to-back: a real, consistent gain — of a few percent. The wall is a hundred times further away.]
  ][
    #placeholder([F1 · stage 2: + kernel flags (barely moves)], height: 5cm)
    #v(0.2em)
    #placeholder([targetcpu A/B paired deltas], height: 3.2cm)
  ]
]

// ═════════════════════════════════════════════════════════════════════════════ ACT 2
#section(act: "Act 2", sub: [Why faster arithmetic does not help])[The wall is memory]

#slide[Count the bytes, not the flops][
  #two-col(ratio: (1.1fr, 1fr))[
    One term: $2 dot 16$ B key + 16 B coefficient = *48 B*.

    One layer at $m = 10^6$ terms, fanout 2:
    - read the sum: $m dot 48$ B
    - write the outputs: $2m dot 48$ B
    - sort or probe them: $2m dot 48$ B, read and written again
    - write the merged result: $m dot 48$ B

    ≈ *300–400 MB per layer*, 0.5 TB per run. The arithmetic per term: a few XORs, one multiply — *< 1 ns*.
  ][
    #punch(color: c-red)[
      Two ways to lose:
      - *Hash map* = random access. Each probe into a 50 MB table is a cache miss: ~100 ns, one at a time.
      - *Sort* = streaming. Each pass runs at memory bandwidth: ~10 GB/s per core → 30–40 ms per layer just to move the bytes.
    ]
    #v(0.3em)
    Either way the core waits on memory.
  ]
]

#slide[What the machine actually delivers][
  #two-col(ratio: (1.2fr, 1fr))[
    Measured on the reference host (2× Xeon Gold 6244, 16 cores, STREAM-style, `crates/membench`):
    #v(0.3em)
    #set text(size: 17pt)
    #table(columns: 4, inset: 6pt, stroke: 0.5pt + c-grid, align: (left, right, right, right),
      [*placement*], [*read*], [*write*], [*copy*],
      [1 core], [11.3], [10.1], [9.5],
      [1 core, other socket's memory], [7.8], [7.2], [5.5],
      [8 cores, one socket], [39.0], [18.6], [35.6],
      [16 cores, both sockets], [45.0], [25.3], [40.0],
      [32 threads (SMT)], [48.8], [23.1], [33.8])
    #v(0.2em)
    #small[GB/s. Spec sheet says 141 GB/s per socket — only 2 of 6 memory channels are populated.]
  ][
    - hyperthreads add *no* bandwidth
    - the second socket adds *15–25 %*, not 2× — remote reads run at 7.8 GB/s
    - write bandwidth is *half* of read

    #v(0.5em)
    #punch[Strong is 45 GB/s, shared by everyone. At 48 B/term that is a hard ceiling of ~$10^9$ term-moves per second — before any algorithm.]
  ]
]

#slide[Three levers, and a hard limit][
  #align(center, grid(columns: 3, column-gutter: 1cm, row-gutter: 0.4cm,
    ..([Fewer bytes], [More work per byte], [More bandwidth]).map(t => text(size: 24pt, weight: "bold", fill: c-dark, t)),
    [Compact keys, no pointers, structure-of-arrays, coefficients that are not boxed.],
    [Do everything you need with a term while it is in cache. Locality, not clock speed.],
    [More cores, more sockets, more nodes — *and* their caches and their memory capacity.],
  ))
  #v(1em)
  #punch(color: c-red)[The other wall: $10^7$ terms is 0.5 GB; the sort scratch of a global algorithm is several times that. At $10^8$ terms a single node runs out of memory before it runs out of time.]
  #v(0.6em)
  So: multithreading — not for the cores, for the *bandwidth* and the *memory* they bring.
]

#slide[What I tried first (a reconstruction)][
  #two-col[
    *A. Per-thread dictionaries*
    - split the terms in chunks, one hash map per thread
    - apply the gate locally
    - *merge all maps into one* at the end of the layer
    - truncate, repeat

    #v(0.4em)
    *B. Parallel mergesort*
    - one flat array of $(x, z, c)$ rows
    - every thread appends its outputs
    - *parallel sort* of the whole array, then a merge pass for equal keys
  ][
    Both are what a competent engineer writes in an afternoon. Both share a problem:

    #punch(color: c-red)[The *merge* is global. It touches every output row once more, through memory, and is serial or nearly so. Amdahl does the rest.]
    #v(0.4em)
    #small[The Julia-era attempts are not in this repository; these two are reimplemented in Rust on the same circuit and gated against the reference engine to $10^(-9)$. The numbers on the next slide are fresh, the memory of "never past 2× on 32 cores" is old.]
  ]
]

#slide[Strong, not smart: 32 cores, a factor two][
  #two-col(ratio: (1.3fr, 1fr))[
    #placeholder([F2 · speedup vs threads: per-thread maps, parallel mergesort], height: 8.5cm)
  ][
    - both saturate early
    - the merge phase grows with the thread count's output, not shrinks
    - memory traffic per layer goes *up* (extra copies), bandwidth does not

    #v(0.5em)
    #punch[Amdahl: with a serial fraction $s$, $"speedup" <= 1\/s$. A merge that is 40 % of the layer caps you at 2.5×, whatever the core count.]
  ]
]

#slide[Wie niet sterk is, moet slim zijn][
  #v(1.2em)
  #align(center, text(size: 30pt, weight: "bold", fill: c-dark)[We were strong. We were not smart.])
  #v(0.8em)
  #align(center, block(width: 85%, text(size: 21pt)[
    Every attempt kept *one global order* — one map, one sorted array — and paid for it with one global merge. The parallel part was easy; the sequential part was the algorithm.
  ]))
  #v(1.2em)
  #align(center, muted[The project rested for a while.])
]

// ═════════════════════════════════════════════════════════════════════════════ ACT 3
#section(act: "Act 3", sub: [GF(2)-linear hash bucketing])[Sterk én slim]

#slide[The idea: a hash that respects XOR][
  #two-col(ratio: (1.1fr, 1fr))[
    Pick a random binary matrix $H in "GF"(2)^(b times 2n)$ and partition the sum into $B = 2^b$ buckets by
    $ h(v) = H v quad (mod 2). $
    A gate changes a key by an XOR: $v -> v xor d$, $d in D$, $|D| <= 16$ for any 2-qubit gate.

    *Linearity:*
    $ h(v xor d) = h(v) xor h(d). $

    So every term in bucket $beta$ lands in bucket $beta xor h(d)$ — *known before any term is touched*.
  ][
    #punch(color: c-aqua)[
      - a bucket's outputs go to $<= 2^(dim D)$ statically known buckets
      - a bucket's inputs come from the same small set
      - two equal output keys have equal hashes: *duplicates never straddle buckets*
    ]
    #v(0.4em)
    #small[Compare: a general hash scatters a gate's outputs over all $B$ buckets, and the coordinate projection "bucket by qubit 0's bits" collapses under weight truncation — everything is in bucket 0.]
  ]
]

#slide[Consequences: closed tasks, local merges][
  #two-col(ratio: (1.05fr, 1fr))[
    #fig(D + "design/bucket-cosets.svg", height: 8cm, caption: [16 buckets, $h(D)$ spanning two vectors: four closed cosets. Arrows: one coset's entire data movement.])
  ][
    The image of $D$ spans a subspace $S = "span"(h(D))$; its *cosets* partition the buckets.

    - a coset's outputs read only the *same* coset → a *closed task*
    - write-disjoint by construction: *no locks, no atomics, no barriers*
    - dedup is *bucket-local*: sort and merge a cache-sized run
    - *no global sort* anywhere in the loop

    #punch[Rayon work-steals cosets. That is the whole parallel runtime.]
  ]
]

#slide[Load balance comes from randomness][
  #two-col[
    A dense random $H$ is a *universal hash family*:
    $ max_beta |"bucket"_beta| <= m/B + O(sqrt((m log B)/B)) $
    with high probability, *independently of the input's structure*.

    Weight-truncated sums are extremely structured (mostly low weight, identities everywhere) — a coordinate-based partition would put everything in a handful of buckets. The random $H$ does not care.
  ][
    $ "rank"(H|_D) = dim D "with prob." >= 1 - 2^(dim D - b) $
    - full rank: outputs spread over $2^(dim D)$ buckets, gather runs arrive presorted
    - deficient rank (~10 % of 2-qubit placements at $B = 128$): still correct, sort works harder

    #v(0.4em)
    Bucket count follows the sum: target *1024 terms per bucket*, floor 128 buckets, grow-only. Refining splits every bucket in two, in order, in one parity pass — no re-sort.
  ]
]

#slide[Results on the kicked-Ising circuit][
  #two-col(ratio: (1.25fr, 1fr))[
    #placeholder([F3 · speedup vs threads: bucketed engine (with F2 curves greyed)], height: 8.5cm)
  ][
    #placeholder([F1 · stage 3: + bucketed 1T, + bucketed 32T], height: 5.5cm)
    #v(0.3em)
    - single-threaded it is already faster: no global structure to maintain
    - 32 threads: 11–13× on rotation layers (fact sheet), against ≤2× before
  ]
]

#slide[More than linear][
  #two-col(ratio: (1.25fr, 1fr))[
    #placeholder([F4 · speedup vs threads for coarse vs fine buckets, normalized to coarse 1T], height: 8cm)
  ][
    With *enough* buckets, speedup beats the thread count. Not an artefact — this:
    #v(0.2em)
    #set text(size: 17pt)
    #table(columns: 3, inset: 5pt, stroke: 0.5pt + c-grid, align: (left, right, right),
      [*level*], [*per core*], [*×16 cores*],
      [L1d], [32 KB], [0.5 MB],
      [L2], [1 MiB], [16 MiB],
      [L3 (shared)], [—], [2 × 24.75 MiB],
      [DRAM], [—], [45 GB/s total])
    #v(0.2em)
    #punch[Cores bring *cache*, not just clocks: 16 cores are 16 MiB of private L2. A working set that lived in DRAM now lives in L2.]
  ]
]

#slide[The mechanism: bucket size against L2][
  #two-col(ratio: (1.3fr, 1fr))[
    #placeholder([F5 · ns per term-layer and L2/LLC miss rate vs terms per bucket, 1 thread], height: 8.5cm)
  ][
    - 1024 terms × 48 B ≈ 48 KB per bucket; a rotation's gather run is twice that: comfortably in L2
    - the *gather run*, not the bucket, must fit: fanout × bucket
    - dense 2-qubit unitaries have fanout 16 → ~750 KB runs → the *write ceiling* at 16 threads (6.6×, then regression at 32)

    #v(0.3em)
    #punch[Same algorithm, same bytes — only the *order* in which they are touched changed. Locality is the lever.]
  ]
]

#slide[Where the time goes now][
  #two-col[
    #fig(D + "design/phase-shares.svg", caption: [Share of worker busy time per phase, three layer classes, 1 → 32 threads (fact sheet, 2026-09-01).])
  ][
    #fig(D + "design/roofline-threads.svg", caption: [Achieved DRAM traffic against the measured ceilings.])
  ]
  #v(0.2em)
  #small[Sparse layers are *latency*-bound at half the write ceiling even at 32 threads (70–90 % of modelled traffic is served from cache); only dense 2-qubit PTMs hit the write ceiling. The single-thread byte model over-counts DRAM traffic 2.5–13×: the design's whole point.]
]

#slide[Footprint, and where it stands][
  #two-col[
    #placeholder([F7 · bytes per peak term: naive map, per-thread maps, mergesort, bucketed], height: 6.5cm)
    #small[Structure-of-arrays, no pointers, capacity retained across layers: the steady state of a propagation allocates nothing.]
  ][
    #fig(D + "comparisons/baseline-ops.svg", height: 6.5cm, caption: [Construction and Clifford conjugation vs `qiskit.SparsePauliOp` and `openfermion.QubitOperator`.])
    #small[Single-threaded vs `PauliPropagation.jl`: 2–3× faster above a few thousand terms, 5× less memory per term. Below that, its hash map wins — and the engine switches to one.]
  ]
]

// ═════════════════════════════════════════════════════════════════════════════ CODA
#section(act: "Coda", sub: [One structure, many wins])[The gift that keeps on giving]

#slide[One structure, many wins][
  #two-col[
    The partition on $h$ was built for parallelism. It also gave:
    - *locality* — cache-sized runs, superlinear scaling
    - *footprint* — bucket-local scratch, no global sort buffers
    - *GPU-ready* buffers — `#[repr(C)]` SoA columns, fixed fanout

    #punch(color: c-aqua)[And it is a *communication plan*. Partition on $h$ across ranks: a layer is a sparse, statically known exchange, computed from $H$ and $D$ before any term is touched. No key is ever split across ranks.]
  ][
    #align(center, {
      set text(size: 14pt)
      let rank(name, bs, fill) = rect(inset: 8pt, radius: 4pt, fill: fill.lighten(85%), stroke: 1pt + fill,
        stack(dir: ttb, spacing: 4pt, text(weight: "bold", name), text(font: "Source Code Pro", size: 12pt, bs)))
      grid(columns: 2, column-gutter: 1cm, row-gutter: 0.8cm,
        rank("rank 0", "buckets 00xx", c-blue), rank("rank 1", "buckets 01xx", c-orange),
        rank("rank 2", "buckets 10xx", c-aqua), rank("rank 3", "buckets 11xx", c-violet))
      v(0.6em)
      text(fill: c-muted, size: 13pt)[a gate with $h(D) subset "span"(0100, 1000)$ exchanges only between ranks in the same coset — most gates touch no rank boundary at all]
    })
    #v(0.3em)
    #small[NUMA is the same problem one level down: two sockets, two "ranks", remote reads at 7.8 GB/s.]
  ]
]

#slide[Open threads, and the proverb the right way round][
  #two-col[
    - *NUMA-aware placement.* A static coset→worker map lost 1.25–1.9× to work stealing (stragglers). The smart version has to steal _within_ a socket first.
    - *Dense-PTM write ceiling.* Fanout-16 gather runs at 16 threads saturate the write path; smaller runs or fused passes.
    - *Distributed prototype.* The exchange plan exists on paper; the buckets are already the message boundaries.
    - *Rank-deficient hash seeds*, a channel-aware bucket floor, more channel types.
  ][
    #v(0.5em)
    #punch(color: c-blue)[
      *Sterk:* 32 cores, 45 GB/s, every byte of L2.

      *Slim:* an algebraic structure — a linear hash — that tells you where everything goes before you move it.
    ]
    #v(0.8em)
    #align(center, text(size: 22pt, style: "italic", fill: c-dark)[Wie sterk is, moet ook slim zijn.])
    #v(0.5em)
    #small[github.com/lkdvos/paulistrings-rs — Rust core, Python bindings, all figures reproducible from the repository.]
  ]
]

// ═════════════════════════════════════════════════════════════════════════════ APPENDIX
#section(sub: [Negative results and reproduction])[Appendix]

#slide[Things that did not work (and are written down)][
  #set text(size: 17pt)
  - *Bucket by support bits and concatenate* (v0.1 design): provably wrong — buckets interleave, they do not concatenate in sorted order. Four-term counterexample. → led to the linear hash.
  - *Static coset → worker placement* for NUMA locality: 1.25–1.9× slower than work stealing in 7 of 8 cells. Stragglers cost more than locality recovers.
  - *Recompute-in-merge borrowing, segment-copy merging, interleaved transient key layout*: three gather/merge variants, all rejected by paired A/B; one narrow survivor (−3 to −11 %).
  - *Reserving a "safe upper bound" for merge output*: 2.5× worse peak RSS on high-collision channels — the merge's job is to shrink.
  - *LTO code-layout effects are real*: an `#[inline]` hint moves the merge kernel by 6–34 %. Every attribute in `engine/merge.rs` is A/B-verified in both directions.
  #v(0.3em)
  #small[`research/notes/` in the repository: one note per negative result, with the numbers.]
]

#slide[Reproducing every number in this talk][
  #set text(size: 17pt)
  - *Host:* ccqlin038 — 2× Xeon Gold 6244 @ 3.6 GHz, 16 cores / 32 threads, 2 NUMA nodes, 1 MiB L2 per core, governor `powersave`. Bandwidth ceilings from `scripts/bandwidth.sh`.
  - *Circuit:* `examples/common/circuits.py::heavy_hex_kicked_ising(127, 10, 5π/16, −π/2)`; Rust port in `presentation/bench/src/workload.rs`. Observable $Z_62$, Heisenberg, `CoefficientThreshold(2^-12)`.
  - *Baselines:* `presentation/bench` — `naive` (the shipped direct hash-map path), `threadmaps`, `mergesort`, `bucketed`; each gated against `propagate` to $10^(-9)$.
  - *Data:* `presentation/data/*.jsonl` with provenance headers (commit, rustc, host, date, load). One script per figure in `presentation/plots/`.
  - *Protocol:* `RUST_LOG` unset, dedicated Rayon pool per cell, one warm-up, medians of repeated runs; paired A/B with direction consistency for anything under 10 %.
  - *Engine knob* added for this talk: `PropagateOptions { target_bucket_len, min_buckets }`.
]
