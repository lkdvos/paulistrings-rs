# B6 — resource-theoretic probes of the evolved observable

Handoff item B6; adapted spec in
`research/plans/2026-08-31-examples-benchmarks-suite.md` §6 Part B and decision
D12 ("computed in pure Python over the numpy export; no core additions").
Diagnostics: [`resource_probes.py`](resource_probes.py). Script:
[`run_b6.py`](run_b6.py). CI-safe correctness gate:
[`python/paulistrings/tests/test_showcase_b6.py`](../../python/paulistrings/tests/test_showcase_b6.py)
(18 tests, numpy-only, under a second).

Two diagnostics read directly off `PauliSum.x_array()` / `z_array()` /
`coefficients_array()`, both answering "how hard is this operator?" under
different cost models, and both **zero on a single Pauli string**:

| | quantity | cost model it speaks to |
|---|---|---|
| Pauli spectrum | `S_2 = -ln Σ_P p_P²`, `L = 1 - Σ_P p_P²` over `p_P = \|c_P\|²/Σ\|c\|²` | truncation-based **Pauli propagation** (this engine) |
| Operator entanglement | `S_op = -Σ_k λ_k ln λ_k` from the operator Schmidt spectrum across `[0,n/2) \| [n/2,n)` | **matrix-product-operator** methods |

All entropies are in **nats** (natural log), matching the convention of the
reference below that defines the operator quantity.

## What is being computed, and what it is not

### Pauli spectrum

For `O = Σ_P c_P P` in this repo's Hermitian convention, `p_P = |c_P|²/Σ|c|²`
is a probability distribution over the Pauli basis; `S_α` is its Rényi-α
entropy, `L = 1 - Σ p²` the linear (purity) form of the α=2 case. The two are
the same information: `S_2 = -ln(1 - L)` identically (pinned by
`test_renyi2_and_the_linear_variant_are_the_same_information`).

**This is the operator quantity, not the state SRE.** The stabilizer Rényi
entropy of Leone, Oliviero and Hamma (Phys. Rev. Lett. **128**, 050402 (2022);
arXiv:2106.12587) has the same shape but is built on a *pure state* —
`Ξ_P = ⟨ψ|P|ψ⟩²/d`, with the `−log d` normalization that makes it vanish on
stabilizer states — and it is *that* state quantity for which α ≥ 2 was proved
to be a magic monotone, **for pure states**, by Leone and Bittel (Phys. Rev. A
**110**, L040403 (2024); arXiv:2404.11652). **Neither result transfers to what
this showcase computes**, and nothing here is presented as a magic monotone.

The operator-side quantity does have a name in exactly this engine's
literature: Shao, Cheng and Liu, *Characterizing Pauli Propagation via
Operator Complexity* (arXiv:2510.22311, 2025) call it the **Operator
Stabilizer Rényi entropy** (OSE) and define, in their Definition 1,

```
S^α(O) = (α/(1−α)) · ln ‖c²‖_α ,   ‖c²‖_α = (Σ_i c_i^{2α})^{1/α} ,
c_i = 2^{−n} tr(P_i O) ,
```

proving that it is the quantity governing Pauli-propagation truncation error
and the Top-K budget needed for a target accuracy. Their `c_i` **is** this
repo's `c_P` (Pauli strings are orthogonal, `tr(P_i P_j) = 2^n δ_ij`), and
when `Σ c_i² = 1` — i.e. when `O` is Hilbert–Schmidt normalized, which a single
Pauli string is and exact unitary Heisenberg evolution preserves — their
`S^α(O)` is *algebraically identical* to the renormalized `S_α` above.
Truncation breaks `Σ|c|² = 1`, and the two then differ by
`(α/(1−α)) ln(Σ|c|²)`; this showcase always renormalizes (so a truncated curve
stays comparable with the exact curve it converges to) and reports the
surviving weight `Σ|c|²` in every table, so the difference is never hidden.
`pauli_spectrum_renyi_unnormalized` gives the literal formula, and
`test_the_unnormalized_ose_differs_only_by_the_weight_term` pins the offset.

Properties, derived in `resource_probes.py` rather than cited: zero on a single
Pauli string; invariant under Clifford conjugation (a Clifford permutes Pauli
strings up to sign, so the multiset `{p_P}` is unchanged); raised by any
non-Clifford rotation, which splits anticommuting terms into `cos`/`sin`
branches; additive over tensor factors; and **basis-dependent by construction**
— which is the point, since it is a cost model for *this* representation, not a
basis-independent resource measure.

### Operator entanglement

Every Pauli string factorizes across a cut, `P = P_A ⊗ P_B`, so the
coefficient vector reshapes into `M[a,b] = c_P` indexed by the *distinct* left
and right factors (bit masks of the `x`/`z` arrays), and `M`'s SVD **is** the
operator Schmidt decomposition — `{P_a/√(2^{|A|})}` being an orthonormal
operator basis under the Hilbert–Schmidt inner product. Schmidt weights
`λ_k = s_k²/Σ_j s_j²`; `S_op` is their Shannon entropy, `S_op^(2) = -ln Σ λ²`
the α=2 form. This is Zanardi's operator entanglement (Phys. Rev. A **63**,
040304(R) (2001)); its entropy is the operator space entanglement entropy of
Prosen and Pižorn (Phys. Rev. A **76**, 032316 (2007)), introduced there
precisely as the statement that *simulating observables* — unlike simulating
typical states — is efficient for initially local operators.

Unlike the Pauli-spectrum entropy this is **not** Clifford-invariant: a CNOT
across the cut is Clifford and raises it. What is true, and what the showcase
uses, is the weaker statement that a Clifford circuit maps a single Pauli
string to a single Pauli string, whose operator entanglement is zero for any
cut.

**Scaling and guards.** `M` is `(n_left, n_right)` with
`n_left ≤ min(T, 4^cut)`, `n_right ≤ min(T, 4^(n−cut))`; it is dense-allocated
(`complex128`) and SVD'd, so cost is `O(n_left·n_right·min(n_left,n_right))`
time and `16·n_left·n_right` bytes, and `n_left·n_right` can approach `T²` in
the worst case. `operator_schmidt_values` therefore refuses past
`max_entries` (default 4e6 entries = 64 MiB) and reports what it would have
needed; `schmidt_matrix_shape` gives the shape without building it. Every
table below prints the shape, so the guard is visible, not theoretical. In
practice, over the whole depth sweep, the largest was `1027 × 3328`
(3.4 M entries) at depth 7 / `min_abs_coeff = 1e-7`.

## Part 1 — exact dense cross-check

At each size the dense `2^n × 2^n` matrix is rebuilt with `numpy.kron` from the
term labels, and both diagnostics are recomputed by routes that share **no
code** with the array-based probes: the Pauli spectrum by brute force over all
`4^n` traces `tr(P O)/2^n`, and the operator Schmidt spectrum by reshaping the
dense matrix to `(4^{|A|}, 4^{|B|})` and SVDing that (never touching a Pauli
label or a symplectic bit). Every gap is computed; the bound is `1e-10`.

Depth *and* cut vary across the three rows on purpose: at fixed depth the light
cone makes the evolved operator identical at n=6, 8, 10, and the operator
entanglement across a single cut bond of a 1D chain turns out to be
n-independent too, so a fixed `(depth, cut)` would have given three oracles on
one number.

```
n=6 steps=4 terms=572 cut=3 (dense build 0.06 s, exhaustive 4^n spectrum)
    hs_weight                sparse=1.000000000000000 dense=1.000000000000000 gap=4.441e-16
    op_entanglement          sparse=0.923589019147395 dense=0.923589019147395 gap=0.000e+00
    op_entanglement_renyi2   sparse=0.645364148890320 dense=0.645364148890320 gap=1.110e-16
    pauli_linear             sparse=0.919563821513081 dense=0.919563821513081 gap=0.000e+00
    pauli_renyi2             sparse=2.520291222827800 dense=2.520291222827799 gap=8.882e-16
    pauli_shannon            sparse=3.132262905475826 dense=3.132262905475825 gap=8.882e-16
n=8 steps=4 terms=1430 cut=2 (dense build 3.06 s, operator entanglement only)
    hs_weight                sparse=1.000000000000000 dense=1.000000000000000 gap=4.441e-16
    op_entanglement          sparse=0.143012687853494 dense=0.143012687853493 gap=5.829e-16
    op_entanglement_renyi2   sparse=0.064766720030394 dense=0.064766720030394 gap=2.498e-16
n=10 steps=3 terms=132 cut=5 (dense build 4.05 s, operator entanglement only)
    hs_weight                sparse=1.000000000000000 dense=1.000000000000000 gap=2.220e-16
    op_entanglement          sparse=0.954664263083263 dense=0.954664263083263 gap=0.000e+00
    op_entanglement_renyi2   sparse=0.771439945071502 dense=0.771439945071502 gap=3.331e-16
```

(The wall times in that block are incidental — they say only which oracle is
affordable at which size, and they move with host load; every *diagnostic*
value in it is bit-reproducible run to run.)

**Every gap is at or below 8.9e-16** — machine precision, twelve orders inside
the `1e-10` bound. The `hs_weight` row is a fourth, independent identity:
`Σ_P |c_P|² = tr(O†O)/2^n`, which also confirms that untruncated unitary
Heisenberg evolution of a unit Pauli string preserves the spectral weight
exactly.

The exhaustive `4^n` spectrum runs at n=6 only: it costs `16^n` complex
multiplications, measured at 0.6 s for n=6 and **99 s for n=8**, and
`resource_probes.MAX_DENSE_SPECTRUM_N` refuses it past n=8 outright. The dense
operator-entanglement oracle is `O(8^n)` and runs at all three sizes; what caps
its depth at n=10 is the `O(T·4^n)` `numpy.kron` build (132 terms is three
seconds there; 1430 is a minute).

### The Clifford points, with no oracle at all

`theta_zz = −pi/2` is fixed at its Clifford value, so both `theta_h = 0` and
`theta_h = pi/2` make the whole kicked-Ising circuit Clifford, and a
single-Pauli seed must stay a single Pauli string:

```
theta_h=0     terms=1      S_2=-0.000e+00 L=0.000e+00 S_op=-0.000e+00  (bound 0e+00)
theta_h=pi/2  terms=4181   S_2=-0.000e+00 L=0.000e+00 S_op=5.374e-30  (bound 1e-25)
```

`theta_h = 0` is exact in floating point — the X layer really is the identity,
one term survives. `theta_h = pi/2` is Clifford only in *exact* arithmetic:
`cos(pi/2) = 6.1e-17`, so the branch that should cancel survives as dust and
the sum keeps 4181 terms with coefficients around 1e-49. Both diagnostics are
quadratic in those, which is why the bound there is 1e-25 rather than an
equality — and both nevertheless come out at or below 5.4e-30.

## Part 2 — sweeping the kick angle (exact, no truncation)

1D kicked-Ising chain, `n = 16`, 5 Trotter steps, seed `Z_8`, bipartition
`[0,8) | [8,16)` — which crosses **exactly one lattice bond**, the reason this
part uses a chain rather than a heavy-hex sublattice (on heavy-hex the same
index range cuts a topology-dependent number of edges, confounding the
reading). `policy=None` throughout: these curves are exact, nothing is
truncated, and **no convergence panel is owed** for them.

| `theta_h` | terms | `S_2` | `L` | `S_op` | `S_op^(2)` |
|---:|---:|---:|---:|---:|---:|
| 0.00000 | 1 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| 0.09817 | 16796 | 0.24168 | 0.21470 | 0.16777 | 0.07731 |
| 0.19635 | 16796 | 0.85017 | 0.57266 | 0.47767 | 0.30698 |
| 0.29452 | 16796 | 1.51185 | 0.77950 | 0.82211 | 0.64980 |
| 0.39270 | 16796 | 2.12422 | 0.88047 | 1.11686 | 0.97975 |
| 0.49087 | 16796 | 2.85736 | 0.94258 | 1.28895 | 1.16493 |
| 0.58905 | 16796 | 3.58090 | 0.97215 | 1.30927 | 1.15527 |
| 0.68722 | 16796 | 3.93575 | 0.98047 | 1.21522 | 0.94597 |
| 0.78540 | 16793 | 3.65177 | 0.97405 | 1.09869 | 0.70478 |
| 0.88357 | 16796 | 3.25175 | 0.96129 | 1.05827 | 0.62639 |
| 0.98175 | 16796 | 3.35620 | 0.96513 | 1.13876 | 0.77560 |
| 1.07992 | 16796 | 4.02108 | 0.98207 | 1.29282 | 1.09092 |
| 1.17810 | 16796 | 4.67954 | 0.99072 | 1.39224 | 1.34330 |
| 1.27627 | 16796 | 3.83777 | 0.97846 | 1.29211 | 1.20302 |
| 1.37445 | 16796 | 1.88321 | 0.84790 | 0.92336 | 0.68184 |
| 1.47262 | 16796 | 0.48014 | 0.38130 | 0.37776 | 0.18798 |
| 1.57080 | 4181 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

(raw data: [`theta_sweep.csv`](theta_sweep.csv); figure:
[`theta_sweep.svg`](theta_sweep.svg), dotted verticals marking the two Clifford
angles.)

Reading it:

* **Both diagnostics vanish at both Clifford endpoints and are strictly
  positive at all 15 interior angles** — the cleanest possible statement of
  what these quantities measure. Note especially that the *term count* does
  not: 16796 terms at `theta_h → 0⁺`, 4181 at `pi/2`, versus 1 exactly at 0.
  Stored-term count is a property of how the coefficients round; the spectrum
  entropy is a property of where the weight actually is.
* The interior is **non-monotone, with structure**: a first local maximum at
  `theta_h ≈ 0.687` (`S_2 = 3.94`), a dip near `0.884`, and the global maximum
  at `theta_h = 3π/8 ≈ 1.178` (`S_2 = 4.68` nats ≈ **108 effective Pauli
  strings** out of 16796 stored). The two families peak at the same angle here
  but disagree on the shape in between — they are different cost models, and
  `S_op` never exceeds 1.40 nats while `S_2` reaches 4.68.
* `S_op^(2) ≤ S_op` and `S_2 ≤ S_1` at every point, as Rényi entropies must be
  (asserted in the script).

## Part 3 — sweeping depth, exact vs truncated

Same chain at `n = 20`, generic `theta_h = 0.6` (no Clifford shortcut), seed
`Z_10`, one cut bond. Exact through depth 6 (208012 terms, a 462×1715 Schmidt
matrix); depth 7 exact is 2.67 M terms and past the guard, so it is
truncated-only.

| depth | exact T | exact `S_2` | exact `S_op` | trunc T | trunc `S_2` | trunc `S_op` | kept `Σ\|c\|²` | Schmidt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.56978 | 0.00000 | 2 | 0.56978 | 0.00000 | 1.0000000000 | 1×2 |
| 2 | 14 | 1.26989 | 0.62598 | 10 | 1.26989 | 0.62598 | 1.0000000000 | 3×6 |
| 3 | 132 | 1.54703 | 0.95466 | 70 | 1.54703 | 0.95466 | 1.0000000000 | 9×27 |
| 4 | 1430 | 2.52031 | 0.92359 | 588 | 2.52031 | 0.92359 | 1.0000000000 | 30×90 |
| 5 | 16796 | 3.64539 | 1.30311 | 5534 | 3.64539 | 1.30311 | 1.0000000000 | 100×340 |
| 6 | 208012 | 3.91353 | 1.30714 | 31344 | 3.91353 | 1.30714 | 0.9999999964 | 344×1100 |
| 7 | — | — | — | 112727 | 4.60660 | 1.21844 | 0.9999999689 | 827×2569 |

(raw data: [`depth_sweep.csv`](depth_sweep.csv); figure:
[`depth_sweep.svg`](depth_sweep.svg) — the truncated curve is drawn dashed over
the exact one, which is why it is invisible until depth 7.)

The two diagnostics say different things about the same operator, and that is
the point of plotting both:

* `S_2` **grows steadily with depth** (0.57 → 4.61 nats, i.e. 1.8 → 100
  effective Pauli strings), and the stored term count grows far faster
  (2 → 2.67 M exact). Pauli propagation's cost tracks the term count; its
  *achievable accuracy at a budget* tracks `S_2`.
* `S_op` **saturates** around 1.3 nats from depth 5 on, and even dips at depth
  7. That is the Prosen–Pižorn observation in miniature: operator entanglement
  across a fixed cut of a 1D circuit is generated only by the gates crossing
  that cut, so it does not have to grow the way the Pauli spectrum does. The
  same operator is getting steadily harder for one method and not for the
  other.
* Truncation at `min_abs_coeff = 1e-6` reproduces every exact value to five
  decimals while keeping 6.6× fewer terms at depth 6 — with 0.999999996 of the
  spectral weight surviving. The script asserts the exact/truncated gap is
  below 1e-3 at every depth where both exist.

## Part 4 — truncation convergence (plan §7 rule 4)

Part 3's truncated curve is a truncated result, so it ships with a convergence
panel: `min_abs_coeff` swept over `{1e-3 … 1e-7}` at depth 6, where the exact
value is available as a reference line, and at depth 7, which is beyond exact
reach and must be shown self-converging.

**Depth 6, against the exact value:**

| `min_abs_coeff` | terms | kept `Σ\|c\|²` | `S_2` | `S_op` | Schmidt |
|---:|---:|---:|---:|---:|---:|
| 1e-03 | 2358 | 0.9986159568 | 3.91076 | 1.31052 | 107×277 |
| 1e-04 | 7527 | 0.9999718822 | 3.91348 | 1.30720 | 196×563 |
| 1e-05 | 17234 | 0.9999996090 | 3.91353 | 1.30714 | 272×846 |
| 1e-06 | 31344 | 0.9999999964 | 3.91353 | 1.30714 | 344×1100 |
| 1e-07 | 46375 | 1.0000000000 | 3.91353 | 1.30714 | 350×1190 |
| *exact* | 208012 | 1.0000000000 | 3.91353 | 1.30714 | 462×1715 |

```
|gap| vs exact, S_2 :  2.78e-03, 5.62e-05, 7.82e-07, 7.13e-09, 4.58e-11
|gap| vs exact, S_op:  3.38e-03, 5.47e-05, 7.57e-07, 1.64e-09, 3.70e-11
```

Monotone convergence over five orders of magnitude, ending at 4.6e-11 and
3.7e-11 — and note that at `1e-7` the truncated sum has **4.5× fewer terms**
than the exact one while agreeing with it to eleven digits on both
diagnostics.

**Depth 7, self-converged (no exact reference):**

| `min_abs_coeff` | terms | kept `Σ\|c\|²` | `S_2` | `S_op` | Schmidt |
|---:|---:|---:|---:|---:|---:|
| 1e-03 | 4973 | 0.9961022533 | 4.59490 | 1.22941 | 164×450 |
| 1e-04 | 18493 | 0.9998972794 | 4.60630 | 1.21885 | 328×956 |
| 1e-05 | 50822 | 0.9999979933 | 4.60659 | 1.21844 | 585×1740 |
| 1e-06 | 112727 | 0.9999999689 | 4.60660 | 1.21844 | 827×2569 |
| 1e-07 | 205283 | 0.9999999996 | 4.60660 | 1.21844 | 1027×3328 |

```
successive drift, S_2 :  1.14e-02, 2.92e-04, 3.95e-06, 6.13e-08
successive drift, S_op:  1.06e-02, 4.05e-04, 8.11e-06, 1.14e-07
```

Successive differences fall by ~1.5 orders per decade of cutoff, so the depth-7
values quoted in Part 3 are converged to roughly 1e-7. Both statements are
asserted by the script, not eyeballed: the depth-6 sweep must end closer to
exact than it started, and the depth-7 sweep's successive drift must be
shrinking. Figure: [`convergence_panel.svg`](convergence_panel.svg) (rows =
depth, columns = diagnostic; the dashed line in the top row is the exact
value).

## Reproducing

```
source .venv/bin/activate
python examples/b6_resource_probes/run_b6.py
```

Regenerates every artifact in this directory (both CSVs,
[`exact_cross_check.json`](exact_cross_check.json), and all three SVGs) in
well under a minute. Nothing here is a performance claim — but one measurement
*is* load-bearing for that being seconds rather than minutes: the script pins
`OMP/OPENBLAS/MKL_NUM_THREADS=1` before importing numpy, because with LAPACK
left to spawn its own pool on a busy shared host the same 462×1715 SVD was
observed at 56 s instead of 0.14 s. That also keeps the script on the suite's
single-thread default (plan §7 rule 3).

## References

1. L. Leone, S. F. E. Oliviero, A. Hamma, "Stabilizer Rényi Entropy",
   Phys. Rev. Lett. **128**, 050402 (2022); arXiv:2106.12587. *State* SRE —
   the quantity this showcase's Pauli-spectrum entropy resembles but is not.
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
