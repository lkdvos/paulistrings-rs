# Read propagation stats and logs

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
