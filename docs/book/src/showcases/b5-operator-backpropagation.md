# B5 — Hybrid depth reduction

<p class="lead">Split a circuit near the end, back-propagate the observable through the tail classically, and hand a QPU the shorter front circuit plus an evolved observable. The composed expectation is <em>exactly</em> the full-circuit one — so the trade is QPU depth against classical term count, and this page measures both sides of it.</p>

![Evolved-observable term count and residual front-circuit size against tail depth](../assets/b5/depth_vs_terms.svg)

*As the tail depth `k` grows the residual front circuit shrinks linearly in Trotter steps (186 → 0 gates) while the weight-capped evolved observable grows roughly exponentially (1 → 12 413 terms).*

## The idea

Split a circuit at layer `k` from the end:

```text
|0...0> --[ front circuit, depth L-k ]--[ tail circuit, depth k ]--> measure O
```

Back-propagate `O` through the *tail* classically — `direction="heisenberg"` on the last `k` layers, with the same engine this repository ships for everything else — to get `O' = U_tail† O U_tail`. Composing the halves, `⟨psi_front| O' |psi_front⟩` (with `|psi_front⟩ = U_front |0...0⟩`) equals `⟨ψ|O|ψ⟩` for the full circuit exactly, since Heisenberg conjugation composes. The QPU only runs the shorter front circuit; `O'` is computed once, off the QPU.

The artifact is a **schema-v1 task file** — the residual front circuit plus the evolved observable, ready for a QPU-side runner. `task_exact.json` and `task_truncated.json` in the showcase directory are exactly that, and they are the same schema the [cross-engine comparison](../comparisons.md) drives both engines from.

## Running it

```bash
source .venv/bin/activate
python examples/b5_operator_backpropagation/run_b5.py
```

Key calls:

```python
tail_evolved = observable.propagate(tail_circuit, policy, direction="heisenberg")
psio.save(npz_path, tail_evolved)                 # round-trip through disk
loaded = interop.load_task(task_path)
composed = loaded.observable.propagate(loaded.circuit, loaded.truncation,
                                        direction=loaded.direction)
```

This regenerates every artifact in the directory — both task JSONs, the `.npz`, the CSV, both SVG figures — and the whole sweep runs in about **3 seconds** on a laptop-class machine; this page is about depth reduction, not speed. The CI-visible correctness gate is numpy-only and runs well under a second: `pytest python/paulistrings/tests/test_showcase_b5.py`.

## Validation

8-qubit, 4-layer seeded `hardware_efficient_ansatz`, observable `Z_4`, tail depth `k = 1` (front: 3 layers / 69 gates; tail: 1 layer / 23 gates):

| check | value | gap |
|---|---|---|
| full-circuit Heisenberg expectation | −0.175584682492551 | — |
| composed value, task file read back off disk | −0.175584682492551 | **0.0** |
| qiskit-Aer statevector cross-check | −0.175584682492551 | 1.665e-16 |

The round-trip gap is exactly `0.0`, well inside the `1e-12` bound, and it is a genuine round trip through disk: the evolved observable is saved to `.npz` via `paulistrings.io` and read back before being embedded in the task file.

Sweeping `min_abs_coeff` against the exact reference above:

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

Truncation is applied after every *channel*, not after every Python call, so splitting a circuit and truncating separately on each half agrees exactly, for any split point, with truncating the full circuit in one shot — a test pins it. The table above is therefore independent of where `k` is chosen. The gap is **not monotone** in the cutoff (2.9e-2 at 3e-2, then 6.3e-2 at 1e-2) — dropped terms carry signs, and a truncated Pauli sum has no variational bound.

## The actual trade-off: depth against term count

16-qubit heavy-hex kicked-Ising sublattice, 6 Trotter steps, `θ_h = 0.6` (a generic, non-Clifford kick), observable `Z_8`, truncated at `weight <= 6` throughout — the weight cap is what makes the classical half tractable in practice. Sweeping the tail depth `k` from 0 (nothing back-propagated) to 6 (front empty):

| `k` (tail steps) | front layers | front gates | evolved-observable terms |
|---:|---:|---:|---:|
| 0 | 6 | 186 | 1 |
| 1 | 5 | 155 | 2 |
| 2 | 4 | 124 | 14 |
| 3 | 3 | 93 | 132 |
| 4 | 2 | 62 | 1118 |
| 5 | 1 | 31 | 4608 |
| 6 | 0 | 0 | 12413 |

The residual circuit shrinks linearly in Trotter steps while the weight-capped evolved-observable term count grows roughly exponentially before the cap bites — the classic operator-spreading-vs-light-cone shape, and exactly the cost paid classically for every layer moved off the QPU.

**Numbers:** every value on this page comes from [`examples/b5_operator_backpropagation/run_b5.py`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/b5_operator_backpropagation/run_b5.py), committed alongside its outputs — `task_exact.json`, `task_truncated.json`, `evolved_observable_exact.npz`, `depth_vs_terms.csv`, `depth_vs_terms.svg`, and `convergence_panel.svg` — in the showcase directory.
