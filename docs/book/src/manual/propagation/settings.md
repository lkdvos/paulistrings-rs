# Stats, memory and logging

The pages before this one were about what `propagate` computes.
This one is about watching it run: the per-layer counters, how to turn a term count into a memory bill before committing to a run, the log stream, and the handful of arguments and environment variables that change how fast the answer arrives without changing the answer.

## Stats {#stats}

`propagate_with_stats` returns the same evolved sum `propagate` would, alongside a `PropagationStats` with one entry per layer in application order.
Enabling it does not change the propagated sum.

```python
from paulistrings import Circuit, PauliSum, truncation

observable = PauliSum.from_strings({"XIII": 0.25, "IXII": 0.25, "IIXI": 0.25, "IIIX": 0.25}, num_qubits=4)
circuit = Circuit(4)
circuit.rz(0.3, 0)
circuit.cnot(0, 1)
circuit.depolarize(0.01, [0, 1])
policy = truncation.coeff(1e-10)

evolved, stats = observable.propagate_with_stats(circuit, policy, direction="heisenberg")

print(stats.layers, stats.peak_terms, stats.final_terms)
print(stats.terms_in, stats.terms_out)   # one entry per layer, post-truncation
print(stats.gate_name)
print(stats.circuit_index)
```

```text
4 5 5
[4, 4, 4, 4] [4, 4, 4, 5]
['Depolarizing', 'Depolarizing', 'Clifford2Q', 'PauliRotation']
[3, 2, 1, 0]
```

A layer is one applied channel ([The loop](index.md#the-loop)), so the broadcast `depolarize(p, [0, 1])` shows up as two entries and `len(circuit) == stats.layers`.
The lists run in **application** order, which under `direction="heisenberg"` is the reverse of the circuit as written: the first entry above is the depolarizing channel pushed last.
`circuit_index` recovers the position as written regardless of direction, and `application_index` is the loop counter.

`terms_out[k]` is the count *after* layer `k`'s truncation, `terms_in[k + 1] == terms_out[k]`, and `peak_terms` is the maximum of those resident counts.
`nanos` is wall-clock time per layer; for a partitioned or distributed run it is the maximum over partitions, and `stats.partition` holds the per-partition detail, otherwise `None` ([NUMA partitions](partitions.md), [MPI ranks](mpi.md)).
The full field list is in the [Library](../../library/propagate.md#propagationstats).

## Estimate the memory a run needs {#estimate-the-memory-a-run-needs}

A term costs `16 × width + 16` bytes: two `uint64` key words per `width` word (the x and z columns) plus a `complex128` coefficient.
`PauliSum.width` is the compile-time word tier the qubit count picked — 1 word up to 64 qubits, 2 up to 128, then 4, 8, 16 — so a 127-qubit sum is 48 B/term and a 1024-qubit sum 272 B/term.

`stats.peak_terms` is the peak *resident* term count between layers, so `peak_terms × (16 × width + 16)` is the sum's own high-water mark:

```python
print(stats.peak_terms, evolved.width, stats.peak_terms * (16 * evolved.width + 16))
```

```text
5 1 160
```

Two things that estimate does not include.
Each layer also allocates a transient in-layer expansion — a term fans out over the channel's delta set before the merge collapses it — bounded by the fanout of the widest channel in the circuit (2 for a Pauli rotation, up to 16 for a dense two-qubit unitary) times one [coset](engine.md#write-disjoint-cosets)'s working set per worker, not times the whole sum.
And buckets keep their capacity across layers, so a sum that shrinks after its peak does not give the memory back.

Run a short prefix of the circuit first and extrapolate `peak_terms` from the per-layer growth in `terms_out`: the growth is what sets the bill, and the [truncation policy](truncation.md#choosing-a-policy) is the knob on it.
A `topn` or `approx_topn` budget is the direct way to cap `peak_terms` when the bill has to be known in advance.
The layout this arithmetic comes from is on [Inside the engine](engine.md#layout); [Showcase B7](../../examples/showcases/b7-stabilizer-prep.md#performance) reports this engine's own measured peak RSS (10.2 GiB single-threaded) against a run this arithmetic would estimate for.

## Logging {#logging}

For running logs instead of a post-hoc summary, the library logs through the `log` facade under the target `paulistrings.propagate` — INFO on entry and exit of each `propagate` call, DEBUG once per layer:

```python
import logging
import paulistrings

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("paulistrings.propagate").setLevel(logging.DEBUG)
paulistrings.reset_log_cache()   # pyo3-log caches each logger's effective level
```

Call [`reset_log_cache()`](../../library/module-helpers.md) again after changing the level mid-process; the cache is otherwise stale for the rest of the process.
Leave logging off when timing — an enabled DEBUG filter adds a clock read per layer.

## Threads {#threads}

In a single unpartitioned process, set the thread count via `RAYON_NUM_THREADS` **before the interpreter starts** — Rayon builds its global pool at the first `propagate` call and never resizes it, so setting the variable from inside an already-running script does not reliably reach it:

```bash
RAYON_NUM_THREADS=32 python my_script.py
```

`RAYON_NUM_THREADS` does not reach a partitioned run; there the thread count comes from the placement ([NUMA partitions](partitions.md)).

More threads are not always faster, and the layer class decides.
On the reference two-socket host, dense-PTM-heavy circuits (general two-qubit unitaries) saturate the machine's write bandwidth at about 16 threads and extra threads only add contention, while sparse-rotation and Clifford circuits stay latency-bound and take the full thread count profitably — 11–13× at 32 threads.
Hyperthreads add no bandwidth and help only the latency-bound phases.
The measured roofline behind those numbers is on [Engine performance](../../examples/benchmarks/engine-performance.md#multi-thread-roofline).

Every benchmark on this site is run single-threaded, `RAYON_NUM_THREADS=1` exported before the interpreter starts, so that term counts and wall times are comparable across pages ([Comparability rules](../../examples/benchmarks/index.md#comparability-rules)).

## Engine selection {#engine-selection}

`engine` picks the storage path and changes nothing about the operator computed; all three settings agree to floating-point tolerance.

- `"sorted"` (the default when `None`) is the bucketed engine every other page describes.
- `"auto"` routes layers whose input has fewer than `small_sum_threshold` terms — [`paulistrings.DEFAULT_SMALL_SUM_THRESHOLD`](../../library/module-helpers.md), `2048` — through a term-by-term direct-apply path and switches to the bucketed engine above it, unless the policy needs a whole-layer pass such as `topn`.
- `"direct"` uses the direct path below the same threshold unconditionally.

```python
small = observable.propagate(circuit, policy, direction="heisenberg", engine="auto")
assert len(small) == len(evolved)
```

The bucketed pipeline has a per-layer fixed cost that is nearly independent of the term count, which is what makes it lose to a hash-map engine on sums of a few hundred terms.
`engine="auto"` removes that cost where it matters: on the configurations below the cross-engine crossover it was measured worth 1.08–2.69× on the same binary, and above its threshold it is inert, measured as its own control — a 84 836-term SU(4) run gave 1.409× with the path on and 1.416× with it off — [Against other tools](../../examples/comparisons.md#below-the-crossover).
Reach for it when a run spends most of its layers on a small sum, such as a local observable through a shallow circuit; on anything that grows past a few thousand terms early it changes nothing.
`engine` is ignored under `partitions=`.

`target_bucket_len` and `min_buckets` size the bucketed engine's storage per layer; `PauliSum.num_buckets` reads back what was realized, which only ever grows.
They are tuning knobs for the engine's own benchmarks rather than for users, and the defaults are what every number on this site was measured with.

## Determinism

The correctness bar is agreement to floating-point tolerance, not bit-identical output.
Equal-key summation order is unspecified: two runs of the same propagation can differ in the last bits of a coefficient, a partitioned run can differ from an unpartitioned one, and `engine="auto"` can differ from `"sorted"`, all within tolerance and all equally correct.
Compare evolved sums with `numpy.allclose` on `coefficients_array()`, as [Splitting a circuit is free](incremental.md#splitting-a-circuit-is-free) does, never with equality.

Bit-identical output is only expected between runs at a fixed bucket count and the same hash seed — cosets are write-disjoint and each is applied sequentially, so this holds *across thread counts too*, and a `partitions=1` run reproduces the unpartitioned path byte for byte; across bucket counts, seeds or partition counts the bar drops back to tolerance.
Term counts, on the other hand, are exact integers and are the load-independent quantity every benchmark on this site quotes and gates on.

## Untruncated runs

`policy=None` applies no truncation at all; only coefficients that are exactly zero after a merge are dropped.
The result is the exact evolved operator, and it is the reference the [cutoff sweep](validation.md#score-against-an-exact-reference-where-you-can) is scored against wherever the problem is small enough to afford it.

```python
exact = observable.propagate(circuit, None, direction="heisenberg")
print(len(exact), "terms untruncated")
```

```text
5 terms untruncated
```

Afford it deliberately: term growth without a filter is exponential in depth for a generic circuit, and at a Clifford angle the numerically dead residual branches described under [Choosing a policy](truncation.md#choosing-a-policy) fan out without bound.
Estimate `peak_terms` from a prefix and the bytes-per-term arithmetic above before running a deep circuit untruncated, and expect the untruncated run to be the most expensive point of any sweep by a wide margin.
