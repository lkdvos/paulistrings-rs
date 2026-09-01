# B7 — stabilizer preparation in stim, estimation by Pauli propagation

Handoff item B7; adapted spec in
[`research/plans/2026-08-31-examples-benchmarks-suite.md`](../../research/plans/2026-08-31-examples-benchmarks-suite.md)
§6 Part B (decision D13). Capability design note:
[`research/notes/2026-09-01-python-api-extensions.md`](../../research/notes/2026-09-01-python-api-extensions.md)
§A8-ii. Script: [`run_b7.py`](run_b7.py); shared helpers
[`stabilizer_prep.py`](stabilizer_prep.py); CI-safe correctness gate
[`python/paulistrings/tests/test_showcase_b7.py`](../../python/paulistrings/tests/test_showcase_b7.py).

## The idea

Two tools, each useless where the other is strong:

- **stim** runs a Clifford circuit of any depth in polynomial time and cannot
  touch a single non-Clifford gate;
- **Pauli propagation** eats non-Clifford rotations for a living, and would
  gain nothing from re-deriving a Clifford circuit's output.

Chain them *across the observable* and you get something neither has:

```
|0...0> --[ deep Clifford prep U_C ]--> |psi>   --[ NON-Clifford tail U_T ]--> measure O
             (stim: tableau, any depth)              (Pauli propagation, truncated)
```

In the Heisenberg picture the tail is applied to the observable and the state is
only ever needed at the very end:

```
<psi| U_T^dagger O U_T |psi>   with   |psi> = U_C |0...0>
```

`|psi>` is a stabilizer state, so it is fully described by its `n` signed Pauli
generators `s_i·G_i`, and for a Hermitian Pauli string `P`

```
<psi|P|psi> = sigma   if  sigma·P is in the group <s_i G_i>  for some sigma = +-1
            = 0       otherwise
```

— a **group-membership test**, `O(n²/64)` word operations per term after an
`O(n³/64)` GF(2) echelon reduction of the generators
(`crates/paulistrings/src/stabilizer.rs`). Contracting the evolved `m`-term sum
is therefore `O(m·n²/64)`, and **no `2^n` object is ever built**. At the `n = 36`
of this showcase a state vector would be 6.9·10¹⁰ amplitudes, 1.0 TiB.

The three API pieces: `interop.stabilizers_from_stim(...)` (tableau → signed
generator strings), `PauliSum.propagate(tail, policy, direction="heisenberg")`,
and `PauliSum.expectation_stabilizer(generators)`.

## Part 1 — the pipeline

Setup: an open **6×6 square lattice, 36 qubits**. The preparation is the 2D
**cluster state** — `H` on every qubit, then one `CZ` per edge (96 Clifford
gates, emitted as disjoint-support colour layers) — which is a universal
resource state and is not a product state in any local basis, so
`expectation(state=...)` cannot express it at all. `stabilizers_from_stim` reads
36 signed generators out of the tableau in **under a millisecond** (0.99 ms on a
cold first call, 0.49 ms warm).

The observables are two sites' *own* cluster stabilizers
`K_q = X_q ∏_{q' ∈ N(q)} Z_{q'}` — the centre (degree 4) and a corner (degree 2)
— so both read exactly `+1` before the tail is applied. The tail is two
kicked-Ising Trotter steps on the same lattice
(`circuits.heavy_hex_kicked_ising` with the grid edge list: an `rx(θ_h)` layer
then a `ZZ(-π/2)` layer per step, one gate per channel), with `θ_h` swept from
`0` to `π/2`.

| `θ_h` | terms kept | terms untruncated | `⟨K_centre⟩` | `⟨K_corner⟩` | closed-form gap |
|---:|---:|---:|---:|---:|---:|
| 0.00000 | 1 | 16 | +1.000000000000 | +1.000000000000 | 0 |
| 0.19635 | 16 | 626 261 | +0.925328113904 | +0.961939766256 | 1.1e−16 |
| 0.39270 | 16 | 625 720 | +0.728553390593 | +0.853553390593 | 0 |
| 0.58905 | 16 | 626 745 | +0.477953368534 | +0.691341716183 | 5.6e−17 |
| 0.78540 | 16 | 627 990 | +0.250000000000 | +0.500000000000 | 0 |
| 0.98175 | 16 | 627 892 | +0.095269936169 | +0.308658283817 | 0 |
| 1.17810 | 16 | 623 236 | +0.021446609407 | +0.146446609407 | 3.5e−18 |
| 1.37445 | 16 | 625 117 | +0.001448581393 | +0.038060233744 | 2.2e−19 |
| 1.57080 | 1 | 583 529 | +0.000000000000 | +0.000000000000 | 3.8e−33 |

(all 17 points: [`theta_sweep.csv`](theta_sweep.csv); figure:
[`theta_sweep.svg`](theta_sweep.svg)).

### The closed form, which is the real check here

Every point above equals `cos^deg(q)(θ_h)` to **1.1e−16**. That is not a fit — it
is derivable, and the derivation is what makes Part 1 an *exact* showcase rather
than a self-consistent one. Write the two-step circuit as
`U = ZZ₂ X₂ ZZ₁ X₁` (rightmost acts first) and push `K_q` through
`U† · U`:

1. **`ZZ₂`.** At `θ_zz = -π/2` every `Z_qZ_{q'}` rotation is a Clifford, and it
   anticommutes with the running operator exactly on the `deg(q)` edges incident
   to `q`. Composing those, the operator picks up
   `∏_{q'} (Z_q Z_{q'}) = Z_q^{deg(q)} ∏_{q'} Z_{q'}`, which for **even** degree
   is `∏_{q'} Z_{q'}` — cancelling the `Z`s and leaving `±X_q`, weight one.
2. **`X₂`.** `X_q` commutes with every `X` rotation: nothing happens.
3. **`ZZ₁`.** The same Clifford conjugation runs backwards, `±X_q → ±K_q`.
4. **`X₁`.** Each of the `deg(q)` `Z_{q'}` factors anticommutes with its own
   `X_{q'}` rotation and splits,
   `Z_{q'} → cos θ_h · Z_{q'} - sin θ_h · (…)Y_{q'}`, so the sum has `2^deg(q)`
   terms and the all-`cos` branch is `cos^deg(q) · K_q`.

`K_q` is a group element (expectation `+1`); every other branch carries a `Y` on
at least one neighbour and is not, so it contracts to `0`. Hence
`⟨K_q⟩ = cos^deg(q)(θ_h)` — and hence also `⟨K_q⟩ = 0` after a **single** step,
where the evolution stops at `±X_q`. The CI gate asserts both, at every
even-degree site of three lattices. (Odd degree leaves `±Y_q` after step 1,
which does *not* commute with `X₂`, and picks up one extra `cos` factor plus a
sign; those sites are excluded from the closed form, not from the pipeline.)

### The 16-vs-626 261 term counts

The untruncated column is almost entirely **floating-point dust**, not physics:
`cos(-π/2)` is `6.1e-17` rather than an exact zero, so every Clifford `ZZ`
rotation spawns a branch at that scale and an untruncated run accumulates them
without bound (the correction recorded in the plan's §9(b)). At two steps this
sum has exactly **16 coefficients above `1e-16`** and ~626 000 below `1e-190`;
`min_abs_coeff = 1e-12` drops precisely the dust, which is why Part 1's runs are
exact and owe no convergence panel. Verified point by point against the fully
untruncated run — worst gap `3.7e-33`. Part 4 is where truncation actually bites.

### Two more independent checks

- **Clifford endpoints.** At `θ_h ∈ {0, π/2}` the whole pipeline collapses to one
  stabilizer computation, so `oracles.stim_clifford_exact` can answer it on the
  *composed* preparation-plus-tail circuit (`prep_circuit + tail`, via
  `Circuit.__add__`). Gap **0.0e+00** at both, and the values are the exact
  integers `+1` and `0`.
- **The `|0…0⟩` special case.** With the generators `+Z_q`, the stabilizer
  contraction must reproduce `expectation(state="z+")` on the same evolved sum.
  On an evolved `Z_c` (34 terms, `θ_h = 0.6`) both routes give
  `+0.533244389679171`, gap **0.0e+00**.
- **The generators themselves.** All 36 closed-form cluster stabilizers `K_q`
  are `+1` elements of the group stim read out of the tableau (worst deviation
  `0.0e+00`) — two independent descriptions of one state, compared through the
  one operation that is invariant under the choice of generating set.

## Part 2 — preparation depth is free

The Heisenberg side never sees the state, so **one** evolved observable (16
terms, `θ_h = 0.6`) is computed once and contracted against every preparation
below. Depth is grown by appending provably-identity Clifford rounds (`H H`,
`S S S S`, `CNOT CNOT`, `CZ CZ`), which leaves the prepared state — and
therefore every generator and every estimate — invariant by construction:

| preparation | Clifford gates | stim readout | contraction | generators identical | estimate |
|---|---:|---:|---:|:---:|---:|
| cluster, no padding | 96 | 0.49 ms | 0.009 ms | — | +0.464004662796 |
| + 200 identity rounds | 592 | 0.51 ms | 0.008 ms | yes | +0.464004662796 |
| + 2 000 identity rounds | 5 110 | 0.65 ms | 0.009 ms | yes | +0.464004662796 |
| + 20 000 identity rounds | 50 236 | 2.08 ms | 0.009 ms | yes | +0.464004662796 |

A **523× longer** preparation costs 2.1 ms instead of 0.5 ms on the stim side,
the generator strings come back **byte-identical**, the estimate is unchanged to
all 12 printed digits, and the Pauli-propagation cost is not merely similar but
*literally the same computation* — one evolved sum, contracted four times.
(Raw data: [`prep_depth.csv`](prep_depth.csv).)

### The honest caveat

Unstructured deep preparations behave differently, and the showcase reports it
rather than picking friendlier numbers:

| preparation | Clifford gates | mean generator weight | estimate |
|---|---:|---:|---:|
| random Clifford, depth 2 | 94 | 1.7 | +0.000000000000 |
| random Clifford, depth 10 | 470 | 3.1 | +0.000000000000 |
| random Clifford, depth 50 | 2 336 | 10.3 | +0.000000000000 |
| random Clifford, depth 200 | 9 316 | 24.1 | +0.000000000000 |

Exactly zero, and not through truncation: the contraction is a membership test,
and a generic stabilizer state's group holds `2^n` of the `4^n` Pauli strings, so
a term lands in it with probability `2^-n = 1.5e-11` at `n = 36`. A deep
*unstructured* preparation annihilates a low-weight evolved operator term by
term. The informative observables for such a state are its own group elements —
and those work fine:

```
random Clifford prep, depth 50; its own generator G_0 (weight 4) as the observable
  tail depth 0                             -> +1.000000000000   (exactly the sign of G_0)
  tail depth 2, min_abs_coeff=1e-4, 8 200 terms -> -0.215300327096
```

So the pipeline is not restricted to graph states; what matters is that the
observable and the state have group elements in common, which for a *structured*
resource state (cluster, surface-code, GHZ) is the normal case.

## Part 3 — the contraction's cost law, measured

Setup is charged once per call, so it is measured separately as the cost of a
one-term sum and subtracted. Generators are the 1D-chain cluster state at each
`n` (valid at every `n`, built without stim); the terms are 20 000 uniformly
random full-support Pauli strings, since a random Pauli hits about half the
group's pivots and so pays the generic per-term cost the bound describes.

| `n` | `m` | setup (ms) | total (ms) | per term (µs) | `n²/64` | ns per word-op |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 20 000 | 0.018 | 7.983 | 0.3982 | 64 | 6.22 |
| 128 | 20 000 | 0.062 | 18.557 | 0.9247 | 256 | 3.61 |
| 256 | 20 000 | 0.622 | 52.953 | 2.6165 | 1 024 | 2.56 |
| 512 | 20 000 | 2.872 | 190.389 | 9.3758 | 4 096 | 2.29 |
| 1024 | 20 000 | 14.427 | 691.325 | 33.8449 | 16 384 | 2.07 |

Local slopes `d log(per-term time) / d log n`, against the bound's 2.0:

| `n` | 64→128 | 128→256 | 256→512 | 512→1024 |
|---|---:|---:|---:|---:|
| slope | 1.22 | 1.50 | 1.84 | **1.85** |

The exponent climbs towards 2 as `n` grows: below `n ≈ 256` the per-term fixed
cost (the `O(n)` pivot scan, the term decode) dilutes the `n²/64` row work, and
the last decade is within 8% of the bound. Equivalently, the per-word-operation
cost falls from 6.2 ns to 2.1 ns and flattens — one word operation is a load, an
xor pair and a parity, so ~2 ns is the honest steady state and the small-`n` rows
are overhead-dominated, not super-efficient.

Linearity in `m`, at `n = 256`:

| `m` | 1 000 | 5 000 | 20 000 | 100 000 |
|---|---:|---:|---:|---:|
| total (ms) | 2.802 | 13.983 | 52.776 | 260.782 |
| per term (µs) | 2.8017 | 2.7965 | 2.6388 | 2.6078 |

Flat to 7% over two decades of `m` (the residual drift is the fixed setup being
amortized away). Figure: [`scaling.svg`](scaling.svg); raw data
[`scaling.csv`](scaling.csv).

**Load caveat.** Every timing quoted on this page is from the one campaign whose
artifacts are committed here, and untouched code moves by the noise floor between
campaigns — the term counts and expectation values are reproducible to the digit,
the milliseconds are not. These are wall-clock times on a shared 32-core workstation
(`ccqlin038`), single-threaded (`RAYON_NUM_THREADS=1`, asserted). Each entry is
the **minimum of five repeats** after a warm-up, which is the least
load-contaminated estimator available; the suite's single-shot noise floor is
±5–8% single-threaded (CLAUDE.md §Performance discipline). Read the exponents and
the flatness, not the last digit — nothing here is a cross-engine performance
claim.

## Part 4 — truncation convergence (mandatory panel)

Part 1 is exact, so the panel plan §7 rule 4 requires is owed by the *deep* runs.
Same lattice and observable, generic `θ_h = 0.6`, tail depths 4 and 5, sweeping
`min_abs_coeff`:

| `min_abs_coeff` | depth 4: terms | depth 4: `⟨K_c⟩` | depth 5: terms | depth 5: `⟨K_c⟩` |
|---:|---:|---:|---:|---:|
| 1e−2 | 1 004 | +0.390943722001 | 640 | +0.183915036344 |
| 1e−3 | 19 595 | +0.398425325482 | 37 777 | +0.200502716253 |
| 1e−4 | 127 168 | +0.396184858542 | 1 101 631 | +0.203004556159 |
| 1e−5 | 177 612 | +0.396208042360 | 18 261 901 | +0.202105199488 |
| 1e−6 | 177 624 | **+0.396208042360** | 166 786 113 | **+0.202157419139** |

Figure: [`convergence_panel.svg`](convergence_panel.svg). Depth 4 is *converged
inside the grid*: the last two points agree to all 12 digits while the term count
moves by 12, so `+0.3962080424` is a self-converged reference in the plan's D5
sense with the convergence evidence attached. Depth 5 is still moving, by
`5.2e-05` between the last two points — reported as the residual it is, not
rounded into a claim. Its tightest point is 167 million terms, 47 s of
propagation, 27 s of contraction and 10.2 GiB peak RSS, which is the whole reason
`--quick` exists.

Note that the *estimate* converges much faster than the *sum*: between
`1e-4` and `1e-6` at depth 4 the term count grows 1.4× while the answer moves
2.3e-5, because only the terms that land in the stabilizer group contribute at
all. That is a property of the contraction, and it is exactly why the panel is
mandatory rather than optional — the term count is not a proxy for accuracy here.

## Part 5 — validation

Three independent routes, all in [`validation_b7.json`](validation_b7.json):

1. **Dense state vector, numpy only.** The composed preparation-plus-tail circuit
   is run on a `2^n` state vector by `stabilizer_prep.dense_state`, reading the
   gate list off the very `Circuit` object the engine propagates
   (`Circuit.gates`), and the observable is contracted against it by index
   arithmetic. It knows nothing about stabilizer groups, generators, or Pauli
   propagation.
2. **The projector route.** `Π = ∏_i (I + s_i G_i)/2` built from the *generator
   strings* alone (rank-1 assertion included), at `n = 8`. Independent of the
   circuit entirely.
3. **qiskit Aer**, where installed, via `oracles.statevector_expectation`.

| lattice | `n` | tail steps | `θ_h` | terms | PP estimate | dense | gap | projector gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3×4 | 12 | 2 | 0.600 | 16 | +0.464004662796 | +0.464004662796 | 2.2e−15 | — |
| 3×4 | 12 | 3 | 0.600 | 529 | +0.063753121139 | +0.063753121139 | 2.9e−16 | — |
| 3×4 | 12 | 2 | 1.000 | 16 | +0.085221129118 | +0.085221129118 | 5.2e−18 | — |
| 2×4 | 8 | 2 | 0.600 | 10 | −0.464004662796 | −0.464004662796 | 1.4e−15 | 0.0e+00 |
| 2×4 | 8 | 3 | 0.350 | 108 | +0.097463777091 | +0.097463777091 | 3.3e−16 | 0.0e+00 |

Worst gap over every case and route: **2.220e−15**, against a `1e-10` bar.
qiskit Aer's worst gap: `1.55e-15`.

The CI gate (`test_showcase_b7.py`, 36 tests, **0.5 s**) re-runs the dense
cross-check at `n = 6, 8, 12`, the closed form at every even-degree site of three
lattices × eight angles, the one-step-is-zero corollary, both special cases, the
identity-padding invariance, and the dense helpers' own conventions (the
Hermitian-`Y` phase against explicit `numpy.kron` matrices — a `Y` sign error is
invisible to any test that uses only `X` and `Z`). Everything except four
`stim`-gated tests runs in the numpy-only CI job, since `cluster_prep_circuit` is
the `stim`-free spelling of the preparation and
`test_the_two_cluster_preparations_are_the_same_circuit` pins that the two agree
gate for gate.

## Reproducing

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python examples/b7_stabilizer_prep/run_b7.py          # 116 s, 10.2 GiB peak RSS
RAYON_NUM_THREADS=1 python examples/b7_stabilizer_prep/run_b7.py --quick  #  40 s,  1.5 GiB peak RSS
pytest python/paulistrings/tests/test_showcase_b7.py                        # 0.5 s
```

`RAYON_NUM_THREADS=1` must be exported **before** the interpreter starts (see
[`examples/README.md`](../README.md)); the script asserts the pin and refuses to
run unpinned. A full run regenerates every artifact in this directory: three
CSVs, three SVGs, `results_b7.json` (44 `report.RunRecord`s with the standard
provenance block — commit, CPU, library versions, seeds, thread count) and
`validation_b7.json`. `--quick` stops the depth-5 convergence grid one point
early; every other number is identical.
