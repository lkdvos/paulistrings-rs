# Propagation stats and logs

```python
from paulistrings import Circuit, PauliSum, truncation

observable = PauliSum.from_strings({"XIII": 0.25, "IXII": 0.25, "IIXI": 0.25, "IIIX": 0.25}, num_qubits=4)
circuit = Circuit(4)
circuit.rz(0.3, 0)
policy = truncation.coeff(1e-10)

evolved, stats = observable.propagate_with_stats(circuit, policy, direction="heisenberg")

print(stats.layers, stats.peak_terms, stats.final_terms)
print(stats.terms_in, stats.terms_out)   # one entry per layer, post-truncation
```

`propagate_with_stats` returns the same evolved sum `propagate` would, alongside a `PropagationStats` with per-layer term counts in and out, `peak_terms`, and per-layer gate-trace fields (`gate_name`, `nanos`, ...).
Enabling it does not change the propagated sum.

## Estimate the memory a run needs

A term costs `16 × width + 16` bytes: two `uint64` key words per `width` word (the x and z columns) plus a `complex128` coefficient.
`PauliSum.width` is the compile-time word tier the qubit count picked — 1 word up to 64 qubits, 2 up to 128, then 4, 8, 16 — so a 127-qubit sum is 48 B/term and a 1024-qubit sum 272 B/term.

`stats.peak_terms` is the peak *resident* term count between layers, so `peak_terms × (16 × width + 16)` is the sum's own high-water mark:

```python
print(stats.peak_terms, evolved.width, stats.peak_terms * (16 * evolved.width + 16))
```

Two things that estimate does not include.
Each layer also allocates a transient in-layer expansion — a term fans out over the channel's delta set before the merge collapses it — bounded by the fanout of the widest channel in the circuit (2 for a Pauli rotation, up to 16 for a dense two-qubit unitary) times one coset's working set per worker, not times the whole sum.
And buckets keep their capacity across layers, so a sum that shrinks after its peak does not give the memory back.

Run a short prefix of the circuit first and extrapolate `peak_terms` from the per-layer growth in `terms_out`: the growth is what sets the bill, and the truncation policy is the knob on it.
[Performance](../explanation/performance.md#layout-structure-of-arrays-compile-time-width) has the layout this arithmetic comes from.

For running logs instead of a post-hoc summary, the library logs through the `log` facade under the target `paulistrings.propagate` — INFO on entry/exit of `propagate`, DEBUG once per layer:

```python
import logging
import paulistrings

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("paulistrings.propagate").setLevel(logging.DEBUG)
paulistrings.reset_log_cache()   # pyo3-log caches each logger's effective level
```

Call `reset_log_cache()` again after changing the level mid-process; the cache is otherwise stale for the rest of the process.
Leave logging off when timing — an enabled DEBUG filter adds a clock read per layer.

See [PropagationStats reference](../reference/propagate.md) for the full field list.
