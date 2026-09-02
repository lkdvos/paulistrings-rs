# B6 — Resource probes

<p class="lead">Two diagnostics read off the evolved sum's numpy export, each answering "how hard is this operator?" under a different cost model: Pauli-spectrum entropy (a magic-adjacent diagnostic for truncation-based Pauli propagation) and operator entanglement, the cost model for matrix-product-operator methods. Both are zero on a single Pauli string.</p>

![Both diagnostics against the kick angle, with the two Clifford angles marked](../assets/b6/theta_sweep.svg)

*Exact (untruncated) sweep of the kick angle on a 16-qubit kicked-Ising chain. Both diagnostics vanish at both Clifford endpoints and are strictly positive at all 15 interior angles. Dotted verticals mark the Clifford angles.*

## The two quantities

| | quantity | cost model |
|---|---|---|
| Pauli spectrum | `S_2 = −ln Σ_P p_P²`, `L = 1 − Σ_P p_P²` over `p_P = \|c_P\|²/Σ\|c\|²` | truncation-based Pauli propagation (this engine) |
| Operator entanglement | `S_op = −Σ_k λ_k ln λ_k`, Schmidt spectrum across `[0,n/2) \| [n/2,n)` | matrix-product-operator methods |

All entropies are in nats, computed in pure Python over `PauliSum.x_array()` / `z_array()` / `coefficients_array()` — no core additions.

## Pauli-spectrum entropy

This is the operator quantity, not the state stabilizer Rényi entropy (SRE) of Leone, Oliviero and Hamma, built on a *pure state* and normalized to vanish on stabilizer states — the quantity for which Leone and Bittel proved α ≥ 2 a magic monotone, **for pure states only**. Neither result transfers here, and nothing in this showcase is presented as a magic monotone.

For `O = Σ_P c_P P` in this repository's Hermitian convention, `p_P = |c_P|²/Σ|c|²` is a probability distribution over the Pauli basis, `S_α` its Rényi-α entropy, `L` the linear (purity) form of α=2, and `S_2 = −ln(1 − L)` identically. Shao, Cheng and Liu name the operator-side quantity the Operator Stabilizer Rényi entropy (OSE) and prove it governs Pauli-propagation truncation error and the Top-K budget for a target accuracy; their `c_i` is this repository's `c_P`, and when `Σc_i² = 1` (true for a single Pauli string, preserved by exact unitary Heisenberg evolution), their definition is algebraically identical to the renormalized `S_α` here. Truncation breaks `Σ|c|² = 1` by `(α/(1−α)) ln(Σ|c|²)`; this showcase always renormalizes and reports the surviving weight `Σ|c|²` in every table, so the difference is never hidden.

Zero on a single Pauli string, invariant under Clifford conjugation, raised by any non-Clifford rotation, additive over tensor factors, and basis-dependent by construction — a cost model for *this* representation, not a basis-independent resource measure.

## Operator entanglement

Every Pauli string factorizes across a cut, `P = P_A ⊗ P_B`, so the coefficient vector reshapes into `M[a,b] = c_P`, and `M`'s SVD *is* the operator Schmidt decomposition: Zanardi's operator entanglement, whose entropy is the operator space entanglement entropy of Prosen and Pižorn, introduced there as the statement that simulating observables, unlike typical states, is efficient for initially local operators. Unlike the Pauli-spectrum entropy it is *not* Clifford-invariant (a CNOT across the cut raises it), but a Clifford circuit still maps a single Pauli string to a single Pauli string, whose operator entanglement is zero for any cut.

`M` is dense-allocated and SVD'd: `O(n_left·n_right·min(n_left,n_right))` time and `16·n_left·n_right` bytes, with `n_left·n_right` able to approach `T²` in the worst case. The routine refuses past `max_entries` (4·10⁶ entries = 64 MiB), and every table below prints the Schmidt shape; the largest across the depth sweep was `1027 × 3328` (3.4 M entries).

## Dense cross-check

At each size the dense `2ⁿ × 2ⁿ` matrix is rebuilt with `numpy.kron`, and both diagnostics are recomputed by routes sharing no code with the array-based probes — brute force over all `4ⁿ` traces for the Pauli spectrum, SVD of the reshaped dense matrix for the Schmidt spectrum. Bound: `1e-10`.

| case | quantity | gap |
|---|---|---:|
| `n=6`, 4 steps, 572 terms, cut 3 | `pauli_renyi2` | 8.882e-16 |
| | `pauli_shannon` | 8.882e-16 |
| | `pauli_linear` | 0.000e+00 |
| | `op_entanglement` | 0.000e+00 |
| | `hs_weight` | 4.441e-16 |
| `n=8`, 4 steps, 1430 terms, cut 2 | `op_entanglement` | 5.829e-16 |
| `n=10`, 3 steps, 132 terms, cut 5 | `op_entanglement` | 0.000e+00 |

**Every gap is at or below 8.9·10⁻¹⁶**, twelve orders inside the bound; `hs_weight` is a fourth identity, `Σ_P|c_P|² = tr(O†O)/2ⁿ`, confirming that untruncated unitary evolution preserves spectral weight exactly. Depth and cut vary on purpose — at fixed depth the light cone makes the operator identical at n = 6, 8, 10, so a fixed `(depth, cut)` would give three oracles on one number. The exhaustive `4ⁿ` spectrum runs at n = 6 only (0.6 s; 99 s at n = 8, refused above it). At the two Clifford points (`θ_h = 0`, `θ_h = π/2`) a single-Pauli seed must stay a single Pauli string, and it does to floating-point dust: both diagnostics land at or below 5.4·10⁻³⁰ against bounds of 0 and 1e-25.

## Kick-angle sweep

1D kicked-Ising chain, n = 16, 5 Trotter steps, seed `Z_8`, bipartition `[0,8) | [8,16)` — crossing exactly one lattice bond (a chain, not a heavy-hex sublattice, keeps the cut count from confounding the reading). `policy=None` throughout: exact, nothing truncated, no convergence panel owed.

| `θ_h` | terms | `S_2` | `L` | `S_op` | `S_op^(2)` |
|---:|---:|---:|---:|---:|---:|
| 0.00000 | 1 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| 0.09817 | 16796 | 0.24168 | 0.21470 | 0.16777 | 0.07731 |
| 0.29452 | 16796 | 1.51185 | 0.77950 | 0.82211 | 0.64980 |
| 0.49087 | 16796 | 2.85736 | 0.94258 | 1.28895 | 1.16493 |
| 0.68722 | 16796 | 3.93575 | 0.98047 | 1.21522 | 0.94597 |
| 0.88357 | 16796 | 3.25175 | 0.96129 | 1.05827 | 0.62639 |
| 1.17810 | 16796 | **4.67954** | 0.99072 | **1.39224** | 1.34330 |
| 1.37445 | 16796 | 1.88321 | 0.84790 | 0.92336 | 0.68184 |
| 1.57080 | 4181 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

*(Nine of the seventeen measured angles; full table in `theta_sweep.csv`.)*

- Both diagnostics vanish at both Clifford endpoints and are strictly positive at all 15 interior angles; term count does not follow the same pattern — 16 796 terms at `θ_h → 0⁺`, 4181 at π/2, 1 exactly at 0. Stored-term count tracks how coefficients round; entropy tracks where the weight actually is.
- The interior is non-monotone: a first local maximum at `θ_h ≈ 0.687`, a dip near 0.884, and the global maximum at `θ_h = 3π/8` (`S_2 = 4.68` nats ≈ 108 effective Pauli strings out of 16 796 stored). `S_op` never exceeds 1.40 nats while `S_2` reaches 4.68 — different cost models, same operator.
- `S_op^(2) ≤ S_op` and `S_2 ≤ S_1` at every point, as Rényi entropies must be.

## Depth sweep: exact against truncated

![Both diagnostics against depth, exact and truncated](../assets/b6/depth_sweep.svg)

Same chain at n = 20, generic `θ_h = 0.6`, seed `Z_10`, one cut bond. Exact through depth 6 (208 012 terms, a 462×1715 Schmidt matrix); depth 7 exact is 2.67 M terms and past the guard, so it is truncated-only.

| depth | exact T | exact `S_2` | exact `S_op` | trunc T | trunc `S_2` | trunc `S_op` | kept `Σ\|c\|²` | Schmidt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.56978 | 0.00000 | 2 | 0.56978 | 0.00000 | 1.0000000000 | 1×2 |
| 3 | 132 | 1.54703 | 0.95466 | 70 | 1.54703 | 0.95466 | 1.0000000000 | 9×27 |
| 5 | 16796 | 3.64539 | 1.30311 | 5534 | 3.64539 | 1.30311 | 1.0000000000 | 100×340 |
| 6 | 208012 | 3.91353 | 1.30714 | 31344 | 3.91353 | 1.30714 | 0.9999999964 | 344×1100 |
| 7 | — | — | — | 112727 | 4.60660 | 1.21844 | 0.9999999689 | 827×2569 |

- `S_2` grows steadily with depth (0.57 → 4.61 nats, i.e. 1.8 → 100 effective Pauli strings), while stored terms grow far faster (2 → 2.67 M exact): propagation cost tracks term count, achievable accuracy at a budget tracks `S_2`.
- `S_op` *saturates* around 1.3 nats from depth 5 on, even dipping at depth 7 — operator entanglement across a fixed 1D cut is generated only by the gates crossing that cut, so it need not grow the way the Pauli spectrum does. The same operator gets steadily harder for one method and not for the other.
- Truncation at `min_abs_coeff = 1e-6` reproduces every exact value to five decimals at depth 6, keeping 6.6× fewer terms with 0.999999996 of the spectral weight surviving.

## Truncation convergence

![Convergence panel: rows are depth, columns are diagnostic](../assets/b6/convergence_panel.svg)

Depth 6 converges monotonically against the exact value over five orders of `min_abs_coeff`, ending at **4.6·10⁻¹¹** (`S_2`) and **3.7·10⁻¹¹** (`S_op`); at `1e-7` the truncated sum already has 4.5× fewer terms than exact while agreeing to eleven digits. Depth 7 has no exact reference, so it is shown self-converging instead: successive drift falls by roughly 1.5 orders per decade of cutoff, converging the quoted depth-7 values to about 10⁻⁷. Both statements are asserted by the script, not eyeballed. Full tables are in the source README.

## Performance

Reproducing the whole showcase (both CSVs, the cross-check JSON, and all three SVGs) takes well under a minute. That is not a performance claim about the engine; it does depend on one control, though: the script pins `OMP/OPENBLAS/MKL_NUM_THREADS=1` before importing numpy, because with LAPACK left to spawn its own pool on a busy shared host the same 462×1715 SVD was observed at **56 s** instead of 0.14 s.

```bash
source .venv/bin/activate
python examples/b6_resource_probes/run_b6.py
```

The CI gate is 18 tests, numpy-only, under a second:

```bash
pytest python/paulistrings/tests/test_showcase_b6.py
```

## References

1. L. Leone, S. F. E. Oliviero, A. Hamma, "Stabilizer Rényi Entropy", Phys. Rev. Lett. 128, 050402 (2022); arXiv:2106.12587. *State* SRE — the quantity this showcase's Pauli-spectrum entropy resembles but is not.
2. L. Leone, L. Bittel, "Stabilizer entropies are monotones for magic-state resource theory", Phys. Rev. A 110, L040403 (2024); arXiv:2404.11652. Monotonicity for α ≥ 2, **restricted to pure states**.
3. Y. Shao, S. Cheng, Z. Liu, "Characterizing Pauli Propagation via Operator Complexity", arXiv:2510.22311 (2025). Operator Stabilizer Rényi entropy (OSE), Definition 1; the truncation-error and Top-K budget bounds.
4. P. Zanardi, "Entanglement of quantum evolutions", Phys. Rev. A 63, 040304(R) (2001). Operator Schmidt decomposition / operator entanglement.
5. T. Prosen, I. Pižorn, "Operator space entanglement entropy in transverse Ising chain", Phys. Rev. A 76, 032316 (2007). OSEE as the cost model for simulating observables.

**Numbers:** raw sweeps in `examples/b6_resource_probes/theta_sweep.csv` / `depth_sweep.csv`, cross-check in `exact_cross_check.json`, full convergence tables and reproduction detail in [`examples/b6_resource_probes/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/b6_resource_probes/README.md).
