# Observable vs time

A Trotterized time series costs one propagation, not one per time point: propagate a single step, read the expectation value, propagate the next step from where the last one stopped.

```python
from paulistrings import Circuit, PauliSum, truncation

n, steps = 12, 6

step = Circuit(n)
for q in range(n - 1):
    step.pauli_rotation("ZZ", [q, q + 1], 1.0)
for q in range(n):
    step.rx(1.1, q)
for q in range(n):
    step.rz(0.6, q)

observable = PauliSum.from_strings({"Z" + "I" * (n - 1): 1.0}, num_qubits=n)
policy = truncation.coeff(1e-4)

evolved = observable
series = [(0, len(evolved), evolved.expectation("z+").real)]
for k in range(1, steps + 1):
    evolved = evolved.propagate(step, policy, direction="heisenberg")
    series.append((k, len(evolved), evolved.expectation("z+").real))

for k, terms, value in series:
    print(f"step {k}  terms {terms:5d}  <Z0> {value:+.6f}")
```

```text
step 0  terms     1  <Z0> +1.000000
step 1  terms     3  <Z0> +0.453596
step 2  terms    14  <Z0> +0.022743
step 3  terms    69  <Z0> +0.241941
step 4  terms   279  <Z0> +0.531583
step 5  terms  1116  <Z0> +0.498573
step 6  terms  4392  <Z0> +0.324827
```

The last point is identical to propagating the whole six-step circuit in one call — same 4 392 terms, same value — because truncation runs after every channel either way, so splitting a circuit changes nothing ([Truncation](../explanation/truncation.md#splitting-a-circuit-is-free)).
The series is therefore free: it is the same single pass, read out at every step boundary.

Each `expectation` call is one masked pass over the surviving terms and does not consume the sum, so recording several observables per time point costs only those passes.

## The Heisenberg-ordering trap

Under `direction="heisenberg"` each `propagate` call conjugates the *outside* of what is already there, so an incremental call **prepends** its circuit rather than appending it.
Propagating `a` and then `b` gives what the single circuit `b + a` gives, not `a + b`.

This is invisible above because every Trotter step is the same channel list: `U` conjugated `k` times is `(U^k)† O U^k` whichever way the steps are ordered.
It is not invisible for a **time-dependent** schedule — a ramp, a varying angle, a noise rate that changes with depth.
There the loop above computes the steps in the wrong time order, and each time point has to be propagated from the seed with the steps walked in reverse: `O(t_k)` needs step `k` applied first and step `1` last.

Under `direction="forward"` the ordering is the ordinary one — an incremental call appends — so a forward time series over a time-dependent schedule does loop in time order.
See [Direction semantics](../reference/direction.md) for what each picture computes.

## Keeping the series honest

A time series is a grid of truncated results, so every point carries the same obligation a single number does: sweep the cutoff and check the retained norm, and expect the later points to converge last.
See [Validate a result](validate-a-result.md).

Term growth across the series is the cost signal — `propagate_with_stats` reports it per layer rather than per step, see [Propagation stats and logs](read-propagation-stats-and-logs.md).
