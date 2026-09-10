# Story: *Wie niet sterk is, moet slim zijn*

A 40-minute talk on `paulistrings-rs`. Audience: quantum-simulation researchers and scientific-computing staff.
Language: English; the Dutch proverb ("those who are not strong must be smart") is the recurring motif.

The arc: Pauli propagation begins as the proverb inverted — brute compute thrown at an exponential wall. The
first attempts at being strong (fast kernels, many cores) fail because the problem is memory, not arithmetic.
The GF(2)-linear hash bucketing is the "smart" step, and it turns out to be what makes "strong" work at all:
cores now add cache, not just clocks. The method ends up both strong and smart, and the same structure keeps
paying: locality, footprint, GPU buffers, distributed exchange.

One benchmark carries the talk: the 127-qubit heavy-hex kicked Ising circuit (IBM utility experiment lattice,
θzz = −π/2, θh = 5π/16, 5 Trotter steps, 1355 channels), observable Z₆₂ in the Heisenberg picture, coefficient
truncation tuned to ~10⁶ resident terms. Its "time to propagate" bar chart (F1) reappears at every stage.

## Act 1 — Sterk: brute force against the exponential wall (~9 min)

| # | scene | claim | evidence |
|---|---|---|---|
| 1 | Why Pauli propagation | 100+ qubit circuits are out of reach for state vectors (~30 qubits exact) and hard for tensor networks at high entanglement; expectation values only need the operator, and operators stay sparse in the Pauli basis under truncation. Reproduces Kim et al. 2023 observables. | `docs/book`, `benchmarks/python/bench_a_clifford.py` (bit-exact weight-10/17 observables) |
| 2 | Symplectic encoding | Pauli string = (x, z) bit vectors; I=(0,0) X=(1,0) Z=(0,1) Y=(1,1); `[u64; W]` ×2 = 16·W bytes; product = XOR + phase i^k; commutation = symplectic parity. | `ARCHITECTURE.md §Data-Model`, `pauli_string.rs` |
| 3 | Clifford = permutation | A Clifford gate maps one string to one string: key XOR a delta from a small set D; no growth. | `ARCHITECTURE.md §Bucketing` (delta sets) |
| 4 | Beyond Clifford | exp(−iθP/2): commuting strings pass, anticommuting split into cos θ·Q + sin θ·(iPQ). Doubling per layer → exponential; truncation by coefficient / weight / top-N; noise damps high weight. | F0 term growth |
| 5 | The naive algorithm | one hash map (or sort+dedup) per layer; O(m·fanout) updates; 1355 layers × 10⁶ terms ≈ 3·10⁹ map updates. F1 bar 1: naive, 1 thread. | `engine/direct.rs`, `data/engine_ladder.jsonl` |
| 6 | Being strong: the kernel | FxHash, popcount, autovectorization with `-C target-cpu=native`: F1 bar 2 barely moves. The arithmetic is not the cost. | `data/targetcpu_ab.md` |

## Act 2 — The wall is memory (~9 min)

| # | scene | claim | evidence |
|---|---|---|---|
| 7 | Count the bytes | 48 B/term at 128 qubits; a layer streams the sum several times → ~5×48 B×m per layer; single core: 11 GB/s stream, ~100 ns random access. Hash map = random access (latency); sort = streaming (bandwidth). Spec-sheet bandwidth vs measured: 2 of 6 channels populated. | `research/notes/2026-08-30-bandwidth-ceiling-ccqlin038.md`, `benchmarks/PROFILING.md §Roofline` |
| 8 | Three levers | fewer bytes per term; more work per byte moved (locality); more bandwidth and capacity (cores, sockets, nodes). And a hard limit: total memory. | — |
| 9 | What I tried first | per-thread maps merged per layer; parallel mergesort of one flat array. Both are dominated by the merge; ≤2× on 32 cores (memory of the Julia-era attempts; plotted numbers are Rust reconstructions). Amdahl: the merge is the serial fraction. | F2, `data/thread_scaling.jsonl`, `bench/src/{threadmaps,mergesort}.rs` |
| 10 | Proverb, inverted | We were strong (32 cores) and not smart: the algorithm asked for one global order. | — |

## Act 3 — Sterk én slim: GF(2)-linear hash bucketing (~16 min)

| # | scene | claim | evidence |
|---|---|---|---|
| 11 | The idea | h(v) = H·v over GF(2). A gate changes a key by v ⊕ d, so h(v ⊕ d) = h(v) ⊕ h(d): the output bucket is known before touching a term. | `ARCHITECTURE.md §Bucketing`, D3 (`docs/figures/design/bucket-cosets.svg`) |
| 12 | Consequences | output buckets write-disjoint; duplicates cannot straddle buckets → dedup bucket-local; cosets of span(h(D)) are closed tasks; no locks, no atomics, no global sort; each task sorts/merges a cache-sized run. | `ARCHITECTURE.md §Engine`, `engine/coset.rs` |
| 13 | Load balance from randomness | dense random H is a universal hash: m/B ± O(√(m log B / B)) regardless of low-weight structure; rank(H|D) = dim D w.p. ≥ 1 − 2^(dim D − b). | `ARCHITECTURE.md §Hash` |
| 14 | Results | F3 speedup vs threads; F1 gains bucketed 1T and 32T bars. | `data/thread_scaling.jsonl`, `data/engine_ladder.jsonl` |
| 15 | Superlinear | F4: speedup vs threads for coarse vs fine buckets, normalized to the coarse 1T run. Cores bring L2 (1 MiB each): aggregate cache grows with the core count. | `data/thread_scaling{,_large}.jsonl` |
| 16 | Mechanism | F5: ns/term and L2/LLC miss rates vs bucket size at 1 thread; default 1024 terms ≈ 48 KB; the gather run, not the bucket, must fit; dense 2-qubit PTMs (fanout 16) blow through L2 and hit the write ceiling at 16 threads. | `data/bucket_sweep{,_perf}.jsonl`, `research/notes/2026-09-01-roofline-ccqlin038.md` |
| 17 | Roofline today | sparse layers latency-bound at half the write ceiling (11–13× at 32t); dense PTMs write-bound at 16t (6.6×). Phase shares gather/sort/merge. Memory: F7 bytes per term naive vs bucketed; 95 vs 479 B/term vs PauliPropagation.jl. | `docs/figures/design/*.svg`, `data/memory.jsonl` |
| 18 | Where it stands | 2–3× vs PauliPropagation.jl single-threaded, 10–35× vs qiskit/openfermion construction; honest caveats: per-layer fixed cost at small m (direct path), 16 threads optimum for dense PTMs, hash-seed rank deficiency. | `docs/figures/comparisons/baseline-ops.svg`, `research/notes/2026-09-01-jl-optimization-history.md` |

## Coda — the gift that keeps on giving (~4 min)

| # | scene | claim | evidence |
|---|---|---|---|
| 19 | One structure, many wins | parallelism; locality; footprint (SoA, no pointers, 48 B/term); GPU-ready buffers; distributed memory: partition on h across ranks, a layer is a statically known sparse exchange computed from H and D; no key is ever split across ranks. NUMA is the same problem one level down. Implemented since this campaign, so state it as built, not planned. | `ARCHITECTURE.md §GPU-Readiness`, `§Partitioning` |
| 20 | Open threads | The exchange plan is built and measured: cut partition rows leave 4 of 271 heavy-hex layers per step remote (139 random), in-process P=2 is 21 % faster per step than one process, and the same layer loop runs one partition per MPI rank. Still open there: the per-layer bits all-reduce, and ingestion replicating the input per rank. Also open: dense-PTM write ceiling; rank-deficient hash seeds, channel-aware bucket floor. Static coset→worker placement stays a negative result — the answer was partition rows, not smarter stealing. Close: strong and smart. | `research/notes/2026-09-09-partition-row-tuning-results.md`, `2026-09-08-numa-partitioning-results.md`, `2026-08-30-static-coset-placement.md` |

## Figures

F0 term growth · F1 engine ladder (progressive) · F2 old attempts scaling · F3 bucketed scaling · F4 superlinear
coarse vs fine · F5 bucket-size sweep with cache counters · F6 phase shares / roofline (reuse) · F7 memory ·
F8 cross-library (reuse). Diagrams: D1 encoding, D2 rotation branching, D3 bucket cosets (reuse), D4 memory
hierarchy, D5 distributed exchange.
