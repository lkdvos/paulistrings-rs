# Incremental propagation

`propagate` returns a new sum, and that sum is a perfectly good input to another `propagate` call.
Because truncation runs after every layer rather than once per call, chaining calls costs nothing in accuracy, and that identity is what turns a time series from one run per time point into a single pass read out along the way.
It also has one trap, which follows directly from [push order](direction.md#push-order).

## Splitting a circuit is free {#splitting-a-circuit-is-free}

Propagating through `a` and then `b` applies the same sequence of (apply, merge, truncate) steps as propagating through `a + b` in one call, so the two agree exactly, in application order:

```python
import numpy as np
from paulistrings import Circuit, PauliSum, truncation

observable = PauliSum.from_strings({"ZIII": 1.0}, num_qubits=4)
policy = truncation.coeff(1e-10)

front = Circuit(4)
front.h(0)
front.cnot(0, 1)

back = Circuit(4)
back.rz(0.3, 1)

split = observable.propagate(front, policy, direction="forward").propagate(
    back, policy, direction="forward"
)
whole = observable.propagate(front + back, policy, direction="forward")

assert len(split) == len(whole)
assert np.allclose(split.coefficients_array(), whole.coefficients_array())
```

Application order is circuit order under `direction="forward"`, as above, but the *reverse* of circuit order under `direction="heisenberg"` ([Push order](direction.md#push-order)).
A Heisenberg split must therefore propagate `back` before `front`, not `front` before `back`, to reproduce `observable.propagate(front + back, ..., direction="heisenberg")`.

The identity is what [Showcase B5](../../examples/showcases/b5-operator-backpropagation.md) rests on: the tail of a circuit is back-propagated classically, and the shorter front circuit plus the evolved observable are handed to a device, at no cost in accuracy over propagating the whole thing.
It is also what makes the next section one pass rather than many.

## A time series in one pass

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

The last point is identical to propagating the whole six-step circuit in one call — same 4 392 terms, same value, and the same row as the `1e-4` line of the [cutoff sweep](validation.md#sweep-the-cutoff) on the previous page — because truncation runs after every layer either way.
The series is therefore free: it is the same single pass, read out at every step boundary.

Each `expectation` call is one masked pass over the surviving terms and does not consume the sum, so recording several observables per time point costs only those passes.
The [2D Ising quench](../../examples/index.md#the-2d-ising-quench) on the Examples landing page is the same pattern at 36 qubits and 40 steps, run against the crate's Rust walkthrough rather than shown again in Python here.

## The Heisenberg-ordering trap {#the-heisenberg-ordering-trap}

Under `direction="heisenberg"` each `propagate` call conjugates the *outside* of what is already there, so an incremental call **prepends** its circuit rather than appending it.
Propagating `a` and then `b` gives what the single circuit `b + a` gives, not `a + b`.

This is invisible above because every Trotter step is the same channel list: `U` conjugated `k` times is `(U^k)† O U^k` whichever way the steps are ordered.
It is not invisible for a **time-dependent** schedule — a ramp, a varying angle, a noise rate that changes with depth.
There the loop above computes the steps in the wrong time order, and each time point has to be propagated from the seed with the steps walked in reverse: `O(t_k)` needs step `k` applied first and step `1` last.
A time-dependent Heisenberg series is therefore *not* one pass; it is one pass per time point, each from the seed, and the cost difference is the price of asking "what does the circuit up to `t_k` measure?" for several `k`.

Under `direction="forward"` the ordering is the ordinary one — an incremental call appends — so a forward time series over a time-dependent schedule does loop in time order and stays one pass.
The two-row table on [Direction](direction.md#push-order) is the rule this follows from; nothing here is specific to time series.

## Keeping the series honest

A time series is a grid of truncated results, so every point carries the same obligation a single number does: sweep the cutoff and check the retained norm, and expect the later points to converge last, since they sit at the largest term count.
The sweep on [Validating a result](validation.md) *is* the series above at step 6, swept over five cutoffs, and shows that `1e-4` is one row short of bottoming out.

Term growth across the series is the cost signal — `propagate_with_stats` reports it per layer rather than per step, and running the loop on a short prefix is the cheapest way to see whether the growth is saturating ([Stats](settings.md#stats)).
[Showcase B1](../../examples/showcases/b1-operator-scrambling.md) reads support growth and OTOCs off exactly this kind of series, and [Benchmark D](../../examples/benchmarks/d-xxz-chain.md) measures how the term count at each step scales with the chain length.
