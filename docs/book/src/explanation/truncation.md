# Truncation

Truncation is what makes the method work, and it is applied **after every channel** — not after every `propagate` call, and not after every "layer".
Three consequences follow, and all three are load-bearing elsewhere in this book.

## Splitting a circuit is free

Propagating through `a` and then `b` applies the same sequence of (apply, truncate) steps as propagating through `a` followed by `b` in one call, so the two agree exactly, in application order:

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

Application order is circuit order under `direction="forward"`, as above, but the *reverse* of circuit order under `direction="heisenberg"` — see [Direction semantics](../reference/direction.md).
A Heisenberg split must therefore propagate `back` before `front`, not `front` before `back`.
A time series still costs one pass rather than one pass per time point, and a hybrid split still costs nothing in accuracy, once split at the right end — the identity [Showcase B5](case-studies/b5-operator-backpropagation.md) rests on.

## A gate is a truncation point

Fusing two gates into one channel changes the answer.
That is why every circuit in this repository's example suite is built one gate per push, and why the [cross-engine comparison](comparisons.md) can be compared *per layer* at all.

## A noise channel is a truncation point too

In the Pauli basis a depolarizing channel is a coefficient rescale, so adding noise makes a fixed threshold bite harder at every depth.
That is the whole mechanism behind [Showcase B2](case-studies/b2-noisy-verification.md), where noise makes the simulation cheaper rather than more expensive.

## No variational bound

A truncated Pauli sum has no variational bound.
Discarded terms carry signs, so a partial sum can sit on either side of the truth and the error need not fall monotonically as the cutoff tightens.
This is measured, not hypothetical — [Benchmark B](case-studies/b-theta-sweep.md) and [Benchmark D](case-studies/d-xxz-chain.md) both record non-monotone rows.
Read a convergence sweep as a trend across the grid, never as a point-to-point improvement.

Which policy to reach for is a separate question — see [Truncation policy](../how-to/choose-a-truncation-policy.md).
