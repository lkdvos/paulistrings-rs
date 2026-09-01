# B5 — Hybrid depth reduction

<p class="lead">Split a circuit near the end, back-propagate the observable
through the tail classically, and hand a QPU the shorter front circuit plus an
evolved observable. The composed expectation is <em>exactly</em> the
full-circuit one — so the trade is QPU depth against classical term count, and
this page measures both sides of it.</p>

![Evolved-observable term count and residual front-circuit size against tail depth](../assets/b5/depth_vs_terms.svg)

*The trade, measured: as the tail depth `k` grows the residual front circuit
shrinks linearly in Trotter steps (186 → 0 gates) while the weight-capped
evolved observable grows roughly exponentially (1 → 12 413 terms).*

## The idea

Split a circuit at layer `k` from the end:

```text
|0...0> --[ front circuit, depth L-k ]--[ tail circuit, depth k ]--> measure O
```

Instead of running the whole circuit on a QPU, back-propagate `O` through the
*tail* classically — `direction="heisenberg"` on the last `k` layers, with the
same engine this repository ships for everything else — to get
`O' = U_tail† O U_tail`. Composing the halves,

```text
<psi_front| O' |psi_front>   where   |psi_front> = U_front |0...0>
```

is *exactly* `⟨ψ|O|ψ⟩` for the full circuit, because Heisenberg conjugation
composes: `U_full† O U_full = U_front† (U_tail† O U_tail) U_front` for
`U_full = U_tail · U_front`. The QPU only has to run the shorter front circuit;
the observable it measures is `O'`, computed once, off the QPU.

The artifact this produces is a **schema-v1 task file** — the residual front
circuit plus the evolved observable, ready for a QPU-side runner to consume.
`task_exact.json` and `task_truncated.json` in the showcase directory are exactly
that, and they are the same schema the
[cross-engine comparison](../comparisons.md) drives both engines from.

## Validation: the round trip is exact

8-qubit, 4-layer seeded `hardware_efficient_ansatz`, observable `Z_4`, tail
depth `k = 1` (front: 3 layers / 69 gates; tail: 1 layer / 23 gates).

| check | value | gap |
|---|---|---|
| full-circuit Heisenberg expectation | −0.175584682492551 | — |
| composed value, from the task file read back off disk | −0.175584682492551 | **0.0** |
| qiskit-Aer statevector cross-check | −0.175584682492551 | 1.665e-16 |

The round-trip gap is exactly `0.0`, well inside the `1e-12` bound. And it is
genuinely a round trip through disk: the evolved observable is saved to `.npz`
via `paulistrings.io` and read back *before* being embedded in the task file, so
the file on disk — not the in-memory object — is what the check exercises.

## Truncation: the gap is the truncation error, and nothing else

Sweeping `min_abs_coeff` and comparing the *composed* (split) expectation against
the exact reference above:

| `min_abs_coeff` | terms | value | \|gap\| |
|---:|---:|---:|---:|
| 1e-1 | 0 | 0.0000000000 | 1.756e-01 |
| 3e-2 | 82 | −0.2048309093 | 2.925e-02 |
| 1e-2 | 1598 | −0.2388320796 | 6.325e-02 |
| 3e-3 | 7389 | −0.1926315277 | 1.705e-02 |
| 1e-3 | 18011 | −0.1870452698 | 1.146e-02 |
| 1e-4 | 45654 | −0.1758257628 | 2.411e-04 |
| 1e-6 | 64465 | −0.1755848308 | 1.483e-07 |
| 1e-8 | 64786 | −0.1755846840 | 1.531e-09 |

![Convergence panel against the exact reference](../assets/b5/convergence_panel.svg)

> **A subtlety worth stating precisely.** Truncation is applied after every
> *channel*, not after every Python call — so `tail_evolved.propagate(front,
> policy, ...)` (two calls) and `observable.propagate(full_circuit, policy, ...)`
> (one call) apply the exact same sequence of (apply-adjoint, truncate) steps and
> therefore agree exactly, **for any split point.** A test pins it.

So the table above is *independent of where `k` is chosen*: splitting a circuit
for hybrid execution costs nothing in accuracy beyond whatever truncation error
the whole circuit would already pay in one shot. Note also that the gap is **not
monotone** in the cutoff (2.9e-2 at 3e-2, then 6.3e-2 at 1e-2) — dropped terms
carry signs, and a truncated Pauli sum has no variational bound.

What splitting *does* cost is the next section.

## The actual trade-off: depth against term count

16-qubit heavy-hex kicked-Ising sublattice, 6 Trotter steps, `θ_h = 0.6` (a
generic, non-Clifford kick — no free stabilizer shortcut), observable `Z_8`,
truncated at `weight <= 6` throughout. Unbounded backpropagation of a local
operator through a brickwork circuit is bounded by the causal light cone in
principle, but a fixed weight cap is what makes the *classical* half tractable in
practice — which is the whole reason Pauli propagation with truncation, rather
than exact backpropagation, is the tool for this job.

Sweeping the tail depth `k` from 0 (nothing back-propagated) to 6 (front empty):

| `k` (tail steps) | front layers | front gates | evolved-observable terms |
|---:|---:|---:|---:|
| 0 | 6 | 186 | 1 |
| 1 | 5 | 155 | 2 |
| 2 | 4 | 124 | 14 |
| 3 | 3 | 93 | 132 |
| 4 | 2 | 62 | 1118 |
| 5 | 1 | 31 | 4608 |
| 6 | 0 | 0 | 12413 |

The residual circuit shrinks linearly in Trotter steps while the weight-capped
evolved-observable term count grows roughly exponentially with tail depth before
the cap starts to bite — the classic "operator spreading vs. causal light cone"
shape, and exactly the cost that has to be paid classically for every layer
moved off the QPU.

## Reproducing

```bash
source .venv/bin/activate
python examples/b5_operator_backpropagation/run_b5.py
```

Regenerates every artifact in the directory — both JSON task files, the `.npz`,
the CSV and both SVG figures — in about 3 seconds on a laptop-class machine.
Nothing here is a performance claim. The CI-visible correctness gate is
numpy-only and runs well under a second:

```bash
pytest python/paulistrings/tests/test_showcase_b5.py
```

**Source for every number on this page:**
[`examples/b5_operator_backpropagation/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/b5_operator_backpropagation/README.md),
with the raw sweep in `depth_vs_terms.csv` next to it.
