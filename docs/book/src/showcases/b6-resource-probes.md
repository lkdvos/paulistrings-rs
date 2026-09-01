# B6 — Resource probes of the evolved operator

<p class="lead">Two diagnostics read straight off the evolved sum's numpy
export, both answering "how hard is this operator?" under different cost models
— one for truncation-based Pauli propagation, one for matrix-product-operator
methods — and both zero on a single Pauli string. The point of plotting them
together is that <em>the same operator gets steadily harder for one method and
not for the other</em>.</p>

![Both diagnostics against the kick angle, with the two Clifford angles marked](../assets/b6/theta_sweep.svg)

*Exact (untruncated) sweep of the kick angle on a 16-qubit kicked-Ising chain.
Both diagnostics vanish at both Clifford endpoints and are strictly positive at
all 15 interior angles. Dotted verticals mark the Clifford angles.*

## The two quantities

| | quantity | cost model it speaks to |
|---|---|---|
| Pauli spectrum | `S_2 = −ln Σ_P p_P²`, `L = 1 − Σ_P p_P²` over `p_P = \|c_P\|²/Σ\|c\|²` | truncation-based **Pauli propagation** (this engine) |
| Operator entanglement | `S_op = −Σ_k λ_k ln λ_k` from the operator Schmidt spectrum across `[0,n/2) \| [n/2,n)` | **matrix-product-operator** methods |

All entropies are in nats. Both are computed in pure Python over
`PauliSum.x_array()` / `z_array()` / `coefficients_array()` — no core additions.

### What the Pauli spectrum is, and what it is not

For `O = Σ_P c_P P` in this repository's Hermitian convention,
`p_P = |c_P|²/Σ|c|²` is a probability distribution over the Pauli basis, `S_α`
its Rényi-α entropy, and `L` the linear (purity) form of the α=2 case. The two
carry the same information: `S_2 = −ln(1 − L)` identically.

**This is the operator quantity, not the state stabilizer Rényi entropy.** The
state SRE of Leone, Oliviero and Hamma has the same shape but is built on a
*pure state*, with the `−log d` normalization that makes it vanish on stabilizer
states; and it is *that* state quantity for which α ≥ 2 was proved a magic
monotone, **for pure states**, by Leone and Bittel. Neither result transfers to
what this showcase computes, and **nothing here is presented as a magic
monotone.**

The operator-side quantity does have a name in exactly this engine's literature:
Shao, Cheng and Liu call it the **Operator Stabilizer Rényi entropy** (OSE) and
prove it is the quantity governing Pauli-propagation truncation error and the
Top-K budget needed for a target accuracy. Their `c_i` *is* this repository's
`c_P`, and when `Σ c_i² = 1` — which a single Pauli string satisfies and exact
unitary Heisenberg evolution preserves — their definition is algebraically
identical to the renormalized `S_α` here. Truncation breaks `Σ|c|² = 1`, and the
two then differ by `(α/(1−α)) ln(Σ|c|²)`; this showcase always renormalizes, so a
truncated curve stays comparable with the exact curve it converges to, and it
reports the surviving weight `Σ|c|²` in every table, so the difference is never
hidden.

Properties, derived in the module rather than cited: zero on a single Pauli
string; invariant under Clifford conjugation (a Clifford permutes Pauli strings
up to sign, leaving the multiset `{p_P}` unchanged); raised by any non-Clifford
rotation, which splits anticommuting terms into `cos`/`sin` branches; additive
over tensor factors; and **basis-dependent by construction** — which is the
point, since it is a cost model for *this* representation, not a
basis-independent resource measure.

### What operator entanglement is

Every Pauli string factorizes across a cut, `P = P_A ⊗ P_B`, so the coefficient
vector reshapes into `M[a,b] = c_P` indexed by the *distinct* left and right
factors, and `M`'s SVD **is** the operator Schmidt decomposition — the rescaled
Pauli strings being an orthonormal operator basis under the Hilbert–Schmidt inner
product. `S_op` is the Shannon entropy of the Schmidt weights. This is Zanardi's
operator entanglement; its entropy is the operator space entanglement entropy of
Prosen and Pižorn, introduced there precisely as the statement that *simulating
observables* — unlike simulating typical states — is efficient for initially
local operators.

Unlike the Pauli-spectrum entropy this is **not** Clifford-invariant: a CNOT
across the cut is Clifford and raises it. What is true, and what the showcase
uses, is the weaker statement that a Clifford circuit maps a single Pauli string
to a single Pauli string, whose operator entanglement is zero for any cut.

**Scaling and guards.** `M` is dense-allocated and SVD'd, so cost is
`O(n_left·n_right·min(n_left,n_right))` time and `16·n_left·n_right` bytes, and
`n_left·n_right` can approach `T²` in the worst case. The routine therefore
**refuses** past `max_entries` (default 4·10⁶ entries = 64 MiB) and reports what
it would have needed; a companion gives the shape without building the matrix.
Every table below prints the shape, so the guard is visible rather than
theoretical. Over the whole depth sweep the largest was `1027 × 3328`
(3.4 M entries).

## Exact dense cross-check

At each size the dense `2ⁿ × 2ⁿ` matrix is rebuilt with `numpy.kron` from the
term labels, and both diagnostics are recomputed by routes that share **no code**
with the array-based probes: the Pauli spectrum by brute force over all `4ⁿ`
traces, and the operator Schmidt spectrum by reshaping the dense matrix and
SVDing that — never touching a Pauli label or a symplectic bit. The bound is
`1e-10`.

| case | quantity | gap |
|---|---|---:|
| `n=6`, 4 steps, 572 terms, cut 3 | `pauli_renyi2` | 8.882e-16 |
| | `pauli_shannon` | 8.882e-16 |
| | `pauli_linear` | 0.000e+00 |
| | `op_entanglement` | 0.000e+00 |
| | `hs_weight` | 4.441e-16 |
| `n=8`, 4 steps, 1430 terms, cut 2 | `op_entanglement` | 5.829e-16 |
| `n=10`, 3 steps, 132 terms, cut 5 | `op_entanglement` | 0.000e+00 |

**Every gap is at or below 8.9·10⁻¹⁶** — machine precision, twelve orders inside
the bound. The `hs_weight` row is a fourth, independent identity,
`Σ_P |c_P|² = tr(O†O)/2ⁿ`, which also confirms that untruncated unitary
Heisenberg evolution of a unit Pauli string preserves the spectral weight
exactly.

Depth *and* cut vary across the three rows on purpose: at fixed depth the light
cone makes the evolved operator identical at `n = 6, 8, 10`, and the operator
entanglement across a single cut bond of a 1D chain turns out to be
`n`-independent too, so a fixed `(depth, cut)` would have given three oracles on
one number. The exhaustive `4ⁿ` spectrum runs at `n = 6` only — it costs `16ⁿ`
complex multiplications, measured at 0.6 s for `n = 6` and **99 s for `n = 8`**,
and the module refuses it past `n = 8` outright.

### The Clifford points, with no oracle at all

With `θ_zz` fixed at its Clifford value, both `θ_h = 0` and `θ_h = π/2` make the
whole circuit Clifford, and a single-Pauli seed must stay a single Pauli string:

```text
theta_h=0     terms=1      S_2=-0.000e+00 L=0.000e+00 S_op=-0.000e+00  (bound 0e+00)
theta_h=pi/2  terms=4181   S_2=-0.000e+00 L=0.000e+00 S_op=5.374e-30  (bound 1e-25)
```

`θ_h = 0` is exact in floating point — the X layer really is the identity.
`θ_h = π/2` is Clifford only in *exact* arithmetic: `cos(π/2) = 6.1·10⁻¹⁷`, so
the branch that should cancel survives as dust and the sum keeps 4181 terms with
coefficients around 10⁻⁴⁹. Both diagnostics are quadratic in those, which is why
the bound there is `1e-25` rather than an equality — and both nevertheless come
out at or below 5.4·10⁻³⁰.

## The kick-angle sweep

1D kicked-Ising chain, `n = 16`, 5 Trotter steps, seed `Z_8`, bipartition
`[0,8) | [8,16)` — which crosses **exactly one lattice bond**, the reason this
part uses a chain rather than a heavy-hex sublattice (on heavy-hex the same index
range cuts a topology-dependent number of edges, confounding the reading).
`policy=None` throughout: these curves are exact, nothing is truncated, and **no
convergence panel is owed** for them.

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

*(Nine of the seventeen measured angles; the full table is in the source README
and `theta_sweep.csv`.)*

Reading it:

- **Both diagnostics vanish at both Clifford endpoints and are strictly positive
  at all 15 interior angles** — the cleanest possible statement of what these
  quantities measure. Note that the *term count* does not: 16 796 terms at
  `θ_h → 0⁺`, 4181 at `π/2`, versus 1 exactly at 0. Stored-term count is a
  property of how the coefficients round; the spectrum entropy is a property of
  where the weight actually is.
- The interior is **non-monotone, with structure**: a first local maximum at
  `θ_h ≈ 0.687`, a dip near `0.884`, and the global maximum at `θ_h = 3π/8`
  (`S_2 = 4.68` nats ≈ **108 effective Pauli strings** out of 16 796 stored).
  The two families peak at the same angle here but disagree on the shape in
  between — they are different cost models, and `S_op` never exceeds 1.40 nats
  while `S_2` reaches 4.68.
- `S_op^(2) ≤ S_op` and `S_2 ≤ S_1` at every point, as Rényi entropies must be
  (asserted in the script).

## Depth: exact against truncated

![Both diagnostics against depth, exact and truncated](../assets/b6/depth_sweep.svg)

Same chain at `n = 20`, generic `θ_h = 0.6`, seed `Z_10`, one cut bond. Exact
through depth 6 (208 012 terms, a 462×1715 Schmidt matrix); depth 7 exact is
2.67 M terms and past the guard, so it is truncated-only.

| depth | exact T | exact `S_2` | exact `S_op` | trunc T | trunc `S_2` | trunc `S_op` | kept `Σ\|c\|²` | Schmidt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.56978 | 0.00000 | 2 | 0.56978 | 0.00000 | 1.0000000000 | 1×2 |
| 3 | 132 | 1.54703 | 0.95466 | 70 | 1.54703 | 0.95466 | 1.0000000000 | 9×27 |
| 5 | 16796 | 3.64539 | 1.30311 | 5534 | 3.64539 | 1.30311 | 1.0000000000 | 100×340 |
| 6 | 208012 | 3.91353 | 1.30714 | 31344 | 3.91353 | 1.30714 | 0.9999999964 | 344×1100 |
| 7 | — | — | — | 112727 | 4.60660 | 1.21844 | 0.9999999689 | 827×2569 |

The two diagnostics say different things about the same operator, and that is the
point of plotting both:

- `S_2` **grows steadily with depth** (0.57 → 4.61 nats, i.e. 1.8 → 100 effective
  Pauli strings), and the stored term count grows far faster (2 → 2.67 M exact).
  Pauli propagation's cost tracks the term count; its *achievable accuracy at a
  budget* tracks `S_2`.
- `S_op` **saturates** around 1.3 nats from depth 5 on, and even dips at depth 7.
  That is the Prosen–Pižorn observation in miniature: operator entanglement
  across a fixed cut of a 1D circuit is generated only by the gates crossing that
  cut, so it does not have to grow the way the Pauli spectrum does. **The same
  operator is getting steadily harder for one method and not for the other.**
- Truncation at `min_abs_coeff = 1e-6` reproduces every exact value to five
  decimals while keeping 6.6× fewer terms at depth 6, with 0.999999996 of the
  spectral weight surviving.

## Truncation convergence

![Convergence panel: rows are depth, columns are diagnostic](../assets/b6/convergence_panel.svg)

Depth 6, against the exact value:

| `min_abs_coeff` | terms | kept `Σ\|c\|²` | `S_2` | `S_op` | Schmidt |
|---:|---:|---:|---:|---:|---:|
| 1e-03 | 2358 | 0.9986159568 | 3.91076 | 1.31052 | 107×277 |
| 1e-04 | 7527 | 0.9999718822 | 3.91348 | 1.30720 | 196×563 |
| 1e-05 | 17234 | 0.9999996090 | 3.91353 | 1.30714 | 272×846 |
| 1e-06 | 31344 | 0.9999999964 | 3.91353 | 1.30714 | 344×1100 |
| 1e-07 | 46375 | 1.0000000000 | 3.91353 | 1.30714 | 350×1190 |
| *exact* | 208012 | 1.0000000000 | 3.91353 | 1.30714 | 462×1715 |

```text
|gap| vs exact, S_2 :  2.78e-03, 5.62e-05, 7.82e-07, 7.13e-09, 4.58e-11
|gap| vs exact, S_op:  3.38e-03, 5.47e-05, 7.57e-07, 1.64e-09, 3.70e-11
```

Monotone convergence over five orders of magnitude, ending at 4.6·10⁻¹¹ and
3.7·10⁻¹¹ — and note that at `1e-7` the truncated sum has **4.5× fewer terms**
than the exact one while agreeing with it to eleven digits on both diagnostics.

Depth 7 has no exact reference and must be shown self-converging:

```text
successive drift, S_2 :  1.14e-02, 2.92e-04, 3.95e-06, 6.13e-08
successive drift, S_op:  1.06e-02, 4.05e-04, 8.11e-06, 1.14e-07
```

Successive differences fall by ~1.5 orders per decade of cutoff, so the depth-7
values quoted above are converged to roughly 10⁻⁷. Both statements are asserted
by the script, not eyeballed: the depth-6 sweep must end closer to exact than it
started, and the depth-7 sweep's successive drift must be shrinking.

## Reproducing

```bash
source .venv/bin/activate
python examples/b6_resource_probes/run_b6.py
```

Regenerates both CSVs, the cross-check JSON and all three SVGs in well under a
minute. Nothing here is a performance claim — but one measurement *is*
load-bearing for that being seconds rather than minutes: the script pins
`OMP/OPENBLAS/MKL_NUM_THREADS=1` before importing numpy, because with LAPACK left
to spawn its own pool on a busy shared host the same 462×1715 SVD was observed at
56 s instead of 0.14 s.

The CI gate is 18 tests, numpy-only, under a second:

```bash
pytest python/paulistrings/tests/test_showcase_b6.py
```

## References

1. L. Leone, S. F. E. Oliviero, A. Hamma, "Stabilizer Rényi Entropy",
   Phys. Rev. Lett. **128**, 050402 (2022); arXiv:2106.12587. *State* SRE — the
   quantity this showcase's Pauli-spectrum entropy resembles but is not.
2. L. Leone, L. Bittel, "Stabilizer entropies are monotones for magic-state
   resource theory", Phys. Rev. A **110**, L040403 (2024); arXiv:2404.11652.
   Monotonicity for α ≥ 2, **restricted to pure states**.
3. Y. Shao, S. Cheng, Z. Liu, "Characterizing Pauli Propagation via Operator
   Complexity", arXiv:2510.22311 (2025). Operator Stabilizer Rényi entropy
   (OSE), Definition 1; the truncation-error and Top-K budget bounds.
4. P. Zanardi, "Entanglement of quantum evolutions", Phys. Rev. A **63**,
   040304(R) (2001). Operator Schmidt decomposition / operator entanglement.
5. T. Prosen, I. Pižorn, "Operator space entanglement entropy in transverse
   Ising chain", Phys. Rev. A **76**, 032316 (2007). OSEE as the cost model for
   simulating observables.

**Source for every number on this page:**
[`examples/b6_resource_probes/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/b6_resource_probes/README.md),
with the raw sweeps in `theta_sweep.csv` / `depth_sweep.csv` and the cross-check
in `exact_cross_check.json` next to it.
