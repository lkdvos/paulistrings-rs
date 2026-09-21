# Choose a truncation policy

```python
from paulistrings import truncation

policy = truncation.coeff(1e-10)
```

**`coeff` is the default choice**, and the only knob whose sweep gives a convergence statement that's comparable across engines.
Reach for it unless you have a specific reason not to.

**`weight` is a blunt instrument.** Useful when the causal cone is the real constraint; measurably useless when it isn't — on a 2D lattice at high entangling strength it can delete most of the operator for no time saved, and at Clifford angles the operator passes through high weight mid-circuit even when it lands on a low-weight string, so a tight cap can truncate the whole sum to zero terms.

**`topn` bounds memory, but changes what "converged" means.** With a fixed budget the error is set by the discarded tail rather than a threshold, so a `topn` run can't carry a `coeff`-sweep convergence panel, and it has no equivalent in `PauliPropagation.jl` for cross-engine comparison.
`TopN` never splits a tie group of exactly-equal coefficient magnitudes — the whole group is kept if it fits within `k`, dropped otherwise.

Compose with `&`/`|`:

```python
policy = truncation.weight(6) & truncation.coeff(1e-10)
```

Keep `min_abs_coeff` above ~1e-12 on deep circuits — `cos(π/2)` is `6.123233995736766e-17`, not zero, so at a Clifford angle every rotation leaves a numerically dead residual branch that fans out without bound if untruncated.

A truncated Pauli sum has no variational bound: dropped terms carry signs, so error need not fall monotonically as the cutoff tightens.
Read a convergence sweep as a trend across a grid, not a point-to-point improvement.

See [Truncation reference](../reference/truncation.md) for the full policy table and [How it works](../explanation/index.md) for the mechanism.
