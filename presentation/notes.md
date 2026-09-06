# Speaker notes — *Wie niet sterk is, moet slim zijn*

Target 40 minutes. Minute marks are cumulative targets at the *start* of each slide. Every number quoted
carries its source: a file under `presentation/data/`, a repository note, or a figure script.

| slide | title | start | notes |
|---|---|---|---|
| 1 | Title | 0:00 | Read the proverb in Dutch, translate. Tell the room the talk is one argument: the method began as the proverb inverted and ended up honouring it. |
| 2 | Act 1 · Sterk | 1:00 | — |
| 3 | Why Pauli propagation | 1:15 | State vectors: 30 qubits ≈ 16 GiB of amplitudes. The 127-qubit IBM experiment is the motivating case; we reproduce its Clifford-point observables bit-exactly against `stim` (`benchmarks/python/bench_a_clifford.py`). Heisenberg picture: evolve the observable, contract at the end. |
| 4 | Symplectic encoding | 3:00 | 48 bytes per term at 128 qubits (32 B key + 16 B complex coefficient) is the number the whole second act hangs on. Product is XOR plus a phase; commutation is two ANDs and a popcount. |
| 5 | Clifford = permutation | 5:00 | Term count is invariant; a Clifford layer is a table lookup per term. The XOR-delta framing is deliberate: it is what the hash will exploit later. |
| 6 | Beyond Clifford | 6:15 | The split: cos θ Q + sin θ (iPQ). Every kick layer doubles the anticommuting part. Truncation makes it approximate but controlled; noise makes it easier (`examples/b2_noisy_verification`). |
| 7 | The exponential wall | 7:45 | F0 (`data/term_growth.jsonl`, `plots/fig0_term_growth.py`). Peak terms ≈ 1.4·ε^-1.52. ε = 2^-13 → 1.16 M peak terms (the "~1e6" working point). Note the light-cone structure: nearly all growth happens in the last Trotter step. 10^6 × 1355 × 2 ≈ 3·10^9 updates per run. |
| 8 | The naive algorithm | 9:00 | F1 stage 1 (`data/engine_ladder.jsonl`). The naive arm *is* production code: the engine's direct hash-map path (`engine/direct.rs`) run without its size threshold. |
| 9 | Kernel flags | 10:15 | F1 stage 2 and `data/targetcpu_ab.md`. Paired A/B default vs `-C target-cpu=native`; acceptance is sign consistency across pairs (`benchmarks/PROFILING.md`). Expected: inconsistent = no effect. AVX-512 licence downclock can make it a small loss. |
| 10 | Act 2 · The wall is memory | 11:30 | — |
| 11 | Count the bytes | 11:45 | Byte model from `benchmarks/PROFILING.md §Roofline`: ~5 passes × 48 B × m per layer. Hash map = random access ≈ 100 ns per miss; sort = streaming ≈ 10 GB/s per core. |
| 12 | What the machine delivers | 13:30 | Table from `research/notes/2026-08-30-bandwidth-ceiling-ccqlin038.md`. Spec sheet 141 GB/s per socket vs measured 39: two of six channels populated. Hyperthreads add nothing; the second socket adds 15–25 % because remote reads run at 7.8 GB/s. |
| 13 | Three levers | 15:15 | Fewer bytes, more work per byte, more bandwidth and capacity. The capacity wall: 10^7 terms is 0.5 GB of terms and several times that in global sort scratch. |
| 14 | What I tried first | 16:30 | Say explicitly: these are reconstructions (`bench/src/threadmaps.rs`, `bench/src/mergesort.rs`), gated to 1e-9 against the engine (`bench/tests/agreement.rs`). The historical memory is the ≤2× on 32 cores. |
| 15 | A factor two | 18:00 | F2 (`data/thread_scaling.jsonl`). Point at where each curve saturates and relate to the merge share. Amdahl: 40 % serial caps at 2.5×. |
| 16 | Proverb inverted | 19:30 | Short. "One global order, one global merge." Then the pause: the project rested. |
| 17 | Act 3 · Sterk én slim | 20:15 | — |
| 18 | The idea | 20:30 | h(v) = Hv over GF(2). Linearity: h(v⊕d) = h(v)⊕h(d). Contrast with a general hash (scatters everywhere) and with coordinate projection (collapses under weight truncation). `ARCHITECTURE.md §Bucketing`. |
| 19 | Consequences | 22:30 | Diagram `docs/figures/design/bucket-cosets.svg`. Cosets of span(h(D)) are closed tasks; write-disjoint; dedup bucket-local; no global sort. Rayon steals cosets. |
| 20 | Load balance | 24:30 | Universal hash bound; rank(H|D) = dim D with probability ≥ 1 − 2^(dim D − b); ~10 % of two-qubit placements are rank-deficient at B = 128 (`research/notes/2026-09-01-bucket-cliff.md`). Policy: 1024 terms per bucket, floor 128, grow-only. |
| 21 | Results | 26:00 | F3 and F1 stage 3 (`data/thread_scaling.jsonl`, `data/engine_ladder.jsonl`). Fact sheet: 11.3–13.1× at 32 threads on rotation layers (`research/notes/2026-09-01-roofline-ccqlin038.md`). |
| 22 | More than linear | 28:00 | F4 (`data/thread_scaling.jsonl`, `data/thread_scaling_large.jsonl`). Normalisation: coarse-bucket 1-thread run. Caveat to say aloud: coarse buckets also mean fewer cosets, so the parallel-efficiency panel separates the two effects. Cache table from `lscpu`. |
| 23 | Mechanism | 30:30 | F5 (`data/bucket_sweep.jsonl`, `data/bucket_sweep_perf.jsonl`). 1024 × 48 B ≈ 48 KB; the gather run (fanout × bucket) is what must fit. Dense two-qubit PTMs: fanout 16, ~750 KB runs, write ceiling at 16 threads (6.6×, regression at 32). |
| 24 | Where the time goes | 32:30 | `docs/figures/design/phase-shares.svg`, `roofline-threads.svg`. Sparse layers latency-bound at half the write ceiling; the single-thread byte model over-counts DRAM traffic 2.5–13× because the working set is cache-served — the design's point. |
| 25 | Footprint and standing | 34:00 | F7 (`data/memory.jsonl`), `docs/figures/comparisons/baseline-ops.svg`. vs PauliPropagation.jl: 2–3× single-threaded, 95 vs 479 B per peak term (`research/notes/2026-09-01-jl-optimization-history.md`). Below ~2000 terms its hash map wins, so the engine switches to one. |
| 26 | Coda | 35:30 | — |
| 27 | One structure, many wins | 35:45 | Parallelism, locality, footprint, GPU buffers, and the distributed exchange plan (`ARCHITECTURE.md §GPU-Readiness`). NUMA is the same problem one level down. |
| 28 | Open threads, proverb | 37:45 | Static placement lost 1.25–1.9× (`research/notes/2026-08-30-static-coset-placement.md`). Close on "wie sterk is, moet ook slim zijn". |
| A1 | Negative results | backup | One note per item in `research/notes/`. |
| A2 | Reproduction | backup | Host, circuit, baselines, data, protocol. |
