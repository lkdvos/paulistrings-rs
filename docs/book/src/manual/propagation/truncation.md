# Truncation

Truncation is what makes Pauli propagation work, and it is the one argument to `propagate` that trades accuracy for time.
It runs **after every layer** — after every applied channel, not once per `propagate` call ([The loop](index.md#the-loop)) — and what it deletes is gone for every later layer.
This page is about what that does to the result and which policy to reach for; [Validating a result](validation.md) is how to find out whether the result can be quoted.

## Why truncate

Each layer fans terms out and merges them, and without a filter the number of distinct Pauli strings grows with depth until it saturates at `4ⁿ` or exhausts memory, whichever comes first.
For a generic circuit at a generic angle that growth is exponential in depth, and the sum is only tractable because most of the coefficients it produces are tiny.
A truncation policy names which terms count as tiny and deletes them at every layer.

The cost of a run is then set by the policy rather than by the circuit alone: the same circuit under `coeff(1e-2)` and under `coeff(1e-5)` can differ by orders of magnitude in term count, wall time and memory.
The [cost model](index.md#cost-model) is terms, and the policy is the knob on it.

## Choosing a policy {#choosing-a-policy}

```python
from paulistrings import truncation

policy = truncation.coeff(1e-10)
```

**`coeff(eps)` is the default choice**, and the only knob whose sweep gives a convergence statement that is comparable across engines.
It keeps a term when `|c| > eps` — strictly greater, so a coefficient exactly equal to the threshold is dropped.
Reach for it unless you have a specific reason not to.

**`weight(k)` is a blunt instrument.**
It keeps terms of Pauli weight at most `k`, which is useful when the causal cone is the real constraint and measurably useless when it is not: on a 2D lattice at high entangling strength it can delete most of the operator for no time saved, and at Clifford angles the operator passes through high weight mid-circuit even when it lands on a low-weight string, so a tight cap can truncate the whole sum to zero terms.

**`topn(k)` bounds memory, but changes what "converged" means.**
It keeps at most `k` terms, largest `|c|` first, so the error is set by the discarded tail rather than by a threshold; a `topn` run cannot carry a `coeff`-sweep convergence panel, and it has no equivalent in `PauliPropagation.jl` for cross-engine comparison.
`topn` never splits a tie group of exactly-equal coefficient magnitudes — the whole group is kept if it fits within `k`, dropped whole otherwise — which matters for symmetric problems, where a tied group is a symmetry orbit and can be large.
**`approx_topn(n)`** is the cheap sibling: it bins terms by octave of `|c|²` and keeps whole octaves from the top down while they fit in `n`, so at most `n` terms survive, the shortfall is bounded by the population of the coarsest excluded octave, and a tie group is again kept whole.
Exact `topn` has no partitioned form, so `approx_topn` is the policy for [NUMA partitions](partitions.md) and [MPI ranks](mpi.md).

Policies compose with `&` (keep when both agree) and `|` (keep when either does):

```python
policy = truncation.weight(6) & truncation.coeff(1e-10)
budgeted = truncation.coeff(1e-10) & truncation.topn(200_000)
```

Keep a `coeff` threshold above roughly `1e-12` on deep circuits.
`cos(π/2)` evaluates to `6.123233995736766e-17`, not zero, so at a Clifford angle every rotation leaves a numerically dead residual branch alongside the real one, and with no threshold — or one below the residual — those branches fan out without bound and the run pays for terms that carry no physics.
`policy=None` disables truncation entirely, dropping only exact zeros; it is the exact reference where the problem is small enough to afford it ([Untruncated runs](settings.md#untruncated-runs)).

Whichever policy you pick, the cutoff is a guess until it has been swept.
The full policy table is in the [Library](../../library/truncation.md).

## A gate is a truncation point {#a-gate-is-a-truncation-point}

Because truncation runs after every layer, the granularity of the circuit is part of the computation.
Fusing two gates into one channel — replacing `rx` followed by `rz` by a single `unitary_1q`, say — removes a truncation point, so the fused circuit and the two-gate circuit produce different sums under the same policy.
Neither is wrong; they are different approximations, and only the untruncated results agree.

```python
import numpy as np
from paulistrings import Circuit, PauliSum, truncation

def rz(theta):
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])

def rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])

observable = PauliSum.from_strings({"X": 0.6, "Z": 0.6}, num_qubits=1)

two_gates = Circuit(1)
two_gates.rx(1.0, 0)
two_gates.rz(0.15, 0)

fused = Circuit(1)
fused.unitary_1q(0, rz(0.15) @ rx(1.0))

for policy in (None, truncation.coeff(0.1)):
    a = observable.propagate(two_gates, policy, direction="heisenberg")
    b = observable.propagate(fused, policy, direction="heisenberg")
    print(f"{str(policy):22s} two gates: {len(a)} terms  <Z>={a.expectation('z+').real:+.6f}   fused: {len(b)} terms  <Z>={b.expectation('z+').real:+.6f}")
```

```text
None                   two gates: 3 terms  <Z>=+0.399630   fused: 3 terms  <Z>=+0.399630
Truncation(Coeff(0.1)) two gates: 3 terms  <Z>=+0.324181   fused: 3 terms  <Z>=+0.399630
```

The small `rz` angle produces a `Y` branch of magnitude `0.6·sin(0.15) ≈ 0.09`, which the two-gate circuit truncates at its intermediate point and the fused circuit never sees in isolation.
The term counts agree and the answers do not — a 20% difference in `⟨Z⟩` at a cutoff that looks harmless — and no amount of tightening on one circuit says anything about the other.

That is why every circuit in this repository's example suite is built one gate per push ([One gate, one truncation point](../circuits.md#one-gate-one-channel)), and why the [cross-engine comparison](../../examples/comparisons.md#methodology) can be compared *per layer* at all: both engines truncate once per gate object, so they see the same schedule.
It is also why the next page's cutoff sweep must be run on the circuit as it will be used — a sweep on a fused circuit says nothing about the unfused one.

## A noise channel is a truncation point too

A noise channel is a layer like any other, and in the Pauli basis the common ones are coefficient rescales: a single-qubit depolarizing channel multiplies a term's coefficient by `1 − 4p/3` for each non-identity Pauli on its support, dephasing rescales `X` and `Y`, amplitude damping rescales and adds an identity contribution.
So adding noise makes a fixed `coeff` threshold bite harder at every depth — a weight-`w` string crossing `d` noise layers is down by `(1 − 4p/3)^{wd}` before any gate has touched it.

The consequence is the reverse of what a density-matrix simulator sees.
There, noise costs extra: a `4ⁿ` object instead of `2ⁿ`, and each channel more expensive than a gate.
Here, noise makes the run **cheaper**: the filter is exponential in weight, so the threshold acts as an effective weight cap that tightens with depth, and the high-weight tail that dominates a scrambling circuit's term count is exactly what gets deleted.
[Showcase B2](../../examples/showcases/b2-noisy-verification.md#tracked-set-shrinkage-from-noise) measures this on a 127-qubit circuit: at `p = 3e-2` the tracked set peaks at 651× fewer terms and finishes 1078× faster than the noiseless run, at the same cutoff.

Two cautions follow.
The retained-norm diagnostic on the next page no longer isolates truncation under noise, because the norm falls whether or not anything was deleted.
And the granularity rule above applies to noise as well: a broadcast `depolarize(p, [0, 1])` is two layers with a truncation between them, not one ([Noise channels](../circuits.md#noise-channels)).

## No variational bound {#no-variational-bound}

A truncated Pauli sum has no variational bound.
Discarded terms carry signs, so a partial sum can sit on either side of the truth, and the error need not fall monotonically as the cutoff tightens.

This is measured, not hypothetical.
[Benchmark B](../../examples/benchmarks/b-theta-sweep.md#non-monotone-truncation-error) records a weight-10 observable at `θ_h = π/4` whose error against an exact reference is `1.10e-3` at `coeff(1e-2)`, `2.31e-3` at `1e-3` and `2.00e-3` at `1e-4` — the loosest cutoff is the closest.
[Benchmark D](../../examples/benchmarks/d-xxz-chain.md#convergence-panels) records the same thing on an XXZ chain: `3.3e-6` at `1e-4` then `9.0e-6` at `1e-5` in the free regime, `2.9e-7` at `1e-7` then `4.3e-7` at `1e-8` in the interacting one, while the trend across the whole grid still converges.

Read a convergence sweep as a trend across the grid, never as a point-to-point improvement, and never read agreement between two adjacent cutoffs as an error bound.
That is why a single number at a single cutoff is not a result, and why the next page exists.
The same absence of a bound also means the sweep's self-estimated uncertainty can be wrong in either direction; [Benchmark C](../../examples/benchmarks/c-deep-trotter.md#non-bound-uncertainty-estimate) measures a case where a budget-stopped sweep reports an uncertainty 15.7× smaller than the true error.

## See it in use

- [Benchmark B — Kick-angle sweep](../../examples/benchmarks/b-theta-sweep.md#non-monotone-truncation-error) and [Benchmark D — XXZ chain](../../examples/benchmarks/d-xxz-chain.md#convergence-panels) both record non-monotone truncation error against an exact reference.
- [Benchmark C — Deep Trotter circuits](../../examples/benchmarks/c-deep-trotter.md#non-bound-uncertainty-estimate) shows the self-estimated uncertainty flipping from conservative to falsely confident at depth.
- [Showcase B2 — Noisy circuit verification](../../examples/showcases/b2-noisy-verification.md) is noise making truncation cheaper, with a dense noisy reference to score it against.
- [Benchmark E — Random SU(4) brickwork](../../examples/benchmarks/e-su4-brickwork.md) is the generic worst case for a `coeff` policy: rise, plateau, then collapse to zero terms.
- [First propagation](../../examples/first-propagation.md#check-the-cutoff) runs a first cutoff sweep on four qubits.
