# Validating a result

One expectation value at one cutoff is not a result.
Truncation has [no variational bound](truncation.md#no-variational-bound), so nothing in the number itself says how far it is from the truth.
Three checks turn it into one: sweep the cutoff, read the retained norm, and score against an exact reference wherever that is affordable.

## Sweep the cutoff

Run the same propagation at a few decreasing `truncation.coeff` thresholds and watch the answer, the term count and the retained norm together.

```python
import numpy as np
from paulistrings import Circuit, PauliSum, truncation

n, steps = 12, 6

step = Circuit(n)
for q in range(n - 1):
    step.pauli_rotation("ZZ", [q, q + 1], 1.0)
for q in range(n):
    step.rx(1.1, q)
for q in range(n):
    step.rz(0.6, q)

circuit = Circuit(n)
for _ in range(steps):
    circuit = circuit + step

observable = PauliSum.from_strings({"Z" + "I" * (n - 1): 1.0}, num_qubits=n)
norm_in = float(np.sum(np.abs(observable.coefficients_array()) ** 2))

for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    evolved = observable.propagate(circuit, truncation.coeff(eps), direction="heisenberg")
    retained = float(np.sum(np.abs(evolved.coefficients_array()) ** 2)) / norm_in
    print(f"{eps:7.0e}  terms={len(evolved):6d}  <Z0>={evolved.expectation('z+').real:+.9f}  retained={retained:.6f}")
```

```text
  1e-01  terms=     0  <Z0>=+0.000000000  retained=0.000000
  1e-02  terms=   708  <Z0>=+0.301649751  retained=0.856704
  1e-03  terms=  3466  <Z0>=+0.323737940  retained=0.998545
  1e-04  terms=  4392  <Z0>=+0.324826803  retained=0.999998
  1e-05  terms=  4478  <Z0>=+0.324877392  retained=1.000000
```

Three things to read off, in order.
The `1e-1` row truncated the sum to zero terms and reports `0.0`; a zero-term sum is not a converged answer however stable it looks, and the run has to be thrown away rather than quoted.
The value then moves by 2.2·10⁻² from `1e-2` to `1e-3`, by 1.1·10⁻³ to `1e-4`, and by 5.1·10⁻⁵ to `1e-5` — a falling sequence of differences, which is what a converging sweep looks like.
The retained norm rounds to `1.000000` at `1e-5`, but it is not exactly 1 — that row is one term short of the 4 479 an untruncated run returns (see below) — so read it as "the sweep has bottomed out", not as "this row is exact".

The differences are the uncertainty estimate, not the cutoff.
Quote the tightest row's value with the last difference as its error bar, and never quote a value from a cutoff you did not sweep past.

The sweep must be run on the circuit exactly as it will be used — same gate granularity, same noise, same direction — because [a gate is a truncation point](truncation.md#a-gate-is-a-truncation-point) and changing any of them changes the schedule the sweep is measuring.

## Read the retained norm

The retained Hilbert–Schmidt norm is `N = Σ_P |c_P|²`, which `coefficients_array()` gives directly:

```python
def retained_norm(pauli_sum):
    return float(np.sum(np.abs(pauli_sum.coefficients_array()) ** 2))
```

`N` is conserved under exact unitary evolution, so the fraction of the operator a truncated run deleted is `1 − N_out / N_in`, measured against the same quantity on the input sum.
For a single Pauli seed `N_in` is 1 and the deleted fraction is just `1 − N_out`, which is the form the [showcases](../../examples/showcases/index.md) and [benchmarks](../../examples/benchmarks/index.md) quote.

Two caveats.
A noise channel rescales coefficients by design, so under a noisy circuit `N` falls whether or not anything was truncated and the diagnostic no longer isolates truncation — see [A noise channel is a truncation point too](truncation.md#a-noise-channel-is-a-truncation-point-too).
And a large retained norm does not by itself mean a correct expectation value: the deleted terms carry signs and may be exactly the ones the state projects onto, which is why the norm is read *alongside* the sweep rather than instead of it.

## Apply the plateau criterion {#apply-the-plateau-criterion}

Where no exact reference exists, the sweep is the only reference, and the criterion for accepting one is stricter than "two successive values agree".

A self-converged value may be quoted only when both of the following hold:

- the two most recent successive differences are small against the accuracy you need, **and**
- either the term count has stopped growing — the sum has saturated, and the plateau is the exact answer — or both of those differences are strictly nonzero.

A flat value on a still-growing sum is **rejected**: at a small rotation angle, loosening the cutoff by a decade can admit thousands of terms none of which contributes to the state being measured, so the value does not move while the sum keeps growing, and an exactly-zero difference there means "no relevant term has arrived yet".
A sum truncated to zero terms is rejected outright, however flat it looks.

That distinction is not a nicety.
The obvious test — tighten the cutoff until two successive values agree to `tol` — was run against an *exact* reference at a small kick angle in [Benchmark B](../../examples/benchmarks/b-theta-sweep.md#the-self-convergence-criterion-and-the-measured-reason-it-is-not-the-obvious-one) and declared convergence with an estimated uncertainty of exactly zero while the value was still `5.6·10⁻⁷` from the truth, because at that angle the only terms contributing to `⟨0|O|0⟩` are those rotated all the way to pure `Z`, and a decade of extra terms contained none.
Replacing it with the criterion above is worth a measured **190×** in the accuracy of the reported uncertainty, and the benchmark suite imports that criterion as one function object rather than re-implementing it per page.

Truncation has **no variational bound**, so read a sweep as a trend across the whole grid, never as a point-to-point improvement: a partial sum can sit on either side of the truth and the error need not fall monotonically — see [No variational bound](truncation.md#no-variational-bound).
The criterion does not survive every regime either: [Benchmark C](../../examples/benchmarks/c-deep-trotter.md#non-bound-uncertainty-estimate) measures a depth at which a budget-stopped sweep reports a falsely confident uncertainty, and the honest verdict there is "no usable estimate", not a weaker one.

## Score against an exact reference where you can

The sweep estimates its own uncertainty; an exact reference measures the error.
Reach for one whenever the problem admits it:

- **Drop the policy entirely.** A small system or a shallow circuit may fit untruncated, in which case `policy=None` is the exact answer and the sweep can be scored against it ([Untruncated runs](settings.md#untruncated-runs)).
  The sweep above bottoms out at 4 478 terms, one short of the 4 479 that `observable.propagate(circuit, None, direction="heisenberg")` returns.
- **Restrict to the causal cone.** A local observable only spreads as far as its light cone, so the exact problem is often far smaller than the full register.
- **Use a dense simulator.** A state-vector reference (qiskit Aer, or a hand-rolled Kronecker construction) is exact to ~26–30 qubits and does not care about depth — see [Against other tools](../../examples/comparisons.md#vs-state-vector-simulation).
- **Use `stim` at a Clifford point.** Where the circuit is Clifford, a tableau simulator gives the exact ±1 integer at any qubit count — [Benchmark A](../../examples/benchmarks/a-clifford.md) exists to be scored that way.

An exact reference at a reachable corner of the parameter space also calibrates the sweep at the corners you cannot reach.

A time series is a grid of truncated results and carries the same obligation at every point — [Incremental propagation](incremental.md#keeping-the-series-honest).
The term counts and memory behind a sweep are on [Stats, memory and logging](settings.md), and the sweep as a cross-engine convergence panel is what every page under [Benchmarks](../../examples/benchmarks/index.md#the-plateau-criterion) ends with.
