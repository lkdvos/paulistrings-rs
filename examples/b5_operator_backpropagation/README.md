# B5 — hybrid depth reduction via operator backpropagation

Handoff item B5; adapted spec in
`research/plans/2026-08-31-examples-benchmarks-suite.md` §6 Part B. Script:
[`run_b5.py`](run_b5.py). CI-safe correctness gate:
[`python/paulistrings/tests/test_showcase_b5.py`](../../python/paulistrings/tests/test_showcase_b5.py).

## The idea

Split a circuit at layer `k` from the end:

```
|0...0> --[ front circuit, depth L-k ]--[ tail circuit, depth k ]--> measure O
```

Instead of running the whole circuit on a QPU, back-propagate the observable
`O` through the *tail* classically — `direction="heisenberg"` on the last `k`
layers, exactly, with the same Pauli-propagation engine this repo ships for
everything else — to get an evolved observable `O' = U_tail^dagger O U_tail`.
Composing the two halves,

```
<psi_front| O' |psi_front>   where   |psi_front> = U_front |0...0>
```

is *exactly* `<psi| O |psi>` for the full circuit, because Heisenberg
conjugation composes: `U_full^dagger O U_full = U_front^dagger (U_tail^dagger
O U_tail) U_front` for `U_full = U_tail . U_front`. The QPU only has to run
the shorter `front` circuit; the observable it measures is `O'`, computed
once, off the QPU.

The artifact this produces is a **schema-v1 task file**
(`paulistrings.interop.load_task`): the residual front circuit plus the
evolved observable, ready for a QPU-side runner to consume. That is exactly
what [`task_exact.json`](task_exact.json) and
[`task_truncated.json`](task_truncated.json) in this directory are.

## Part 1 — validation (small `n`, exact reference)

Setup: an 8-qubit, 4-layer `hardware_efficient_ansatz` (seeded, `seed=0`),
observable `Z_4`, split with tail depth `k=1` (front: 3 layers / 69 gates;
tail: 1 layer / 23 gates).

1. **Round trip.** The full circuit's Heisenberg-propagated expectation and
   the composed value obtained by loading `task_exact.json` back through
   `interop.load_task` and propagating the *front* circuit against the
   *loaded* observable agree to **`0.0`** (`policy=None`, well inside the
   `1e-12` bound):

   ```
   full-circuit expectation      = -0.175584682492551
   composed (task-file) value    = -0.175584682492551
   round-trip gap                = 0.000e+00
   ```

2. **Independent cross-check.** A qiskit-Aer statevector simulation of the
   same circuit (unitary-only, `n=8 <= 16`) agrees with the Pauli-propagation
   answer to `1.665e-16`:

   ```
   statevector cross-check value = -0.175584682492551
   statevector gap                = 1.665e-16
   ```

3. **The evolved observable itself is round-tripped through
   `paulistrings.io`** — saved to
   [`evolved_observable_exact.npz`](evolved_observable_exact.npz) and read
   back (`io.save` / `io.load`) before being embedded in the task file, so
   the file on disk, not the in-memory object, is what the round-trip check
   above actually exercises.

4. **Truncated variant: the gap is the truncation error.** Sweeping
   `min_abs_coeff` from loose to tight (via `harness.convergence_sweep`) and
   comparing the *composed* (split) expectation against the same exact
   reference from step 1:

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

   plotted in [`convergence_panel.svg`](convergence_panel.svg) against the
   exact reference (dashed line). The gap shrinks monotonically to the
   1e-9-scale floor set by the loosest end of the grid.

   **A subtlety worth stating precisely.** Truncation is applied after every
   *channel*, not after every Python call — so `tail_evolved.propagate(front,
   policy, ...)` (two calls) and `observable.propagate(full_circuit, policy,
   ...)` (one call) apply the exact same sequence of (apply-adjoint,
   truncate) steps and therefore agree exactly, **for any split point**
   (pinned by
   `test_backpropagated_task_reproduces_full_circuit_under_a_shared_policy`
   in the test file). The numbers in the table above are consequently
   independent of where `k` is chosen: splitting a circuit for hybrid
   execution costs nothing in accuracy beyond whatever truncation error the
   *whole* circuit would already pay in one shot. The tradeoff this showcase
   is actually about — what splitting *does* cost — is depth on the QPU side
   traded for term count on the classical side, which is Part 2.

## Part 2 — the actual tradeoff: depth vs. term count

Setup: a 16-qubit heavy-hex kicked-Ising sublattice
(`circuits.heavy_hex_kicked_ising`), 6 Trotter steps, `theta_h=0.6` (a
generic, non-Clifford kick — no free stabilizer shortcut), observable `Z_8`,
truncated at `weight <= 6` throughout (unbounded backpropagation of a local
operator through a brickwork circuit is bounded by the causal light cone in
principle, but a fixed weight cap is what makes the *classical* half of this
tradeoff tractable in practice — this is the whole reason Pauli propagation
with truncation, rather than exact backpropagation, is the tool for the job).

Sweeping the tail depth `k` from 0 (no backpropagation, front = the whole
circuit) to 6 (front is empty, everything backpropagated):

| `k` (tail steps) | front layers | front gates | evolved-observable terms |
|---:|---:|---:|---:|
| 0 | 6 | 186 | 1 |
| 1 | 5 | 155 | 2 |
| 2 | 4 | 124 | 14 |
| 3 | 3 | 93 | 132 |
| 4 | 2 | 62 | 1118 |
| 5 | 1 | 31 | 4608 |
| 6 | 0 | 0 | 12413 |

(raw data: [`depth_vs_terms.csv`](depth_vs_terms.csv); figure:
[`depth_vs_terms.svg`](depth_vs_terms.svg)). The residual circuit shrinks
linearly in Trotter steps while the (weight-capped) evolved-observable term
count grows roughly exponentially with tail depth before the weight cap
starts to bite — the classic "operator spreading vs. causal light cone"
shape, and exactly the cost that has to be paid classically for every layer
moved off the QPU.

## Reproducing

```
source .venv/bin/activate
python examples/b5_operator_backpropagation/run_b5.py
```

Regenerates every artifact in this directory (both JSON task files, the npz,
the CSV, and both SVG figures) in about 3 seconds on a laptop-class machine —
nothing here is a performance claim. The CI-visible correctness gate is
`pytest python/paulistrings/tests/test_showcase_b5.py` (numpy-only, no
`importorskip` needed, well under a second).
