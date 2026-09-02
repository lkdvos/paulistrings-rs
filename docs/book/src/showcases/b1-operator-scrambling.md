# B1 — Operator scrambling

<p class="lead">Heisenberg-evolve a single-site <code>Z</code> through a kicked Ising
circuit and watch the operator spread. The light cone, OTOC, butterfly
velocity, and two-point function are read off the evolved Pauli sum in one
pass, checked against an independent dense <code>2ⁿ×2ⁿ</code> construction.</p>

## Operator scrambling and the question

A local operator evolved backward through a chaotic circuit,
`O(t) = U(t)† O U(t)`, spreads across a growing set of sites: a light cone
whose edge advances at a butterfly velocity `v_B`. The out-of-time-order
correlator (OTOC) `C(r,t)` tracks how much of the operator anticommutes with
a probe at site `r`, approaching `C → 1` as the signature of full scrambling.
In the Pauli basis this is all one object, `O(t) = Σ_P c_P(t) P`, so support,
OTOC, and two-point function are three sums over the same coefficients — the
question is whether truncation converges to a single, contour-independent
`v_B`, and whether a weight cap, effective elsewhere, survives a 2D quench.

## Running it

```bash
source .venv/bin/activate
python examples/b1_operator_scrambling/run_b1_1d.py     # 1D chain, a few minutes
python examples/b1_operator_scrambling/run_b1_2d.py     # 2D quench, ~15 minutes
```

Evolution is one `propagate` call per Trotter step, truncating after each:

```python
from paulistrings import truncation
from common import observables

evolved = observables.single_z(center, n).propagate(
    step_circuit, truncation.coeff(min_abs_coeff), direction="heisenberg"
)
```

Every quantity below is read from three arrays exported once —
`evolved.x_array()`, `evolved.z_array()`, `evolved.coefficients_array()`:
support `w_q(t)` sums `|c|²` per site from the bit columns (plotted against
`(site, t)` this *is* the light cone); the OTOC `C(r,t) =
2·Σ_{P anti W_r}|c_P|²` tests one symplectic bit per site (`X_r`→`z_r(P)`,
`Z_r`→`x_r(P)`, `Y_r`→`x_r(P)≠z_r(P)`); the two-point function
`G(r,t) = ⟨Z_r,O(t)⟩` is the coefficient of the weight-one string `Z_r`,
found by popcount. `N(t) = Σ_P|c_P(t)|² = ⟨O(t),O(t)⟩` equals 1 for the seed
and is conserved exactly. **Truncation only lowers `N(t)`, and `1 − N(t)` is
exactly the norm discarded** — the diagnostic behind every curve below.

## The 1D chain

Open `n = 61` chain, kicked-Ising entangler `θ_zz = −π/2`, kick angle
`θ_h = 0.9`, seed `Z_30`, 12 Trotter steps, Heisenberg direction. Because this
engine truncates after every channel, `t` successive calls on the one-step
circuit reproduce exactly the (apply-adjoint, truncate) sequence of one call
on the `t`-step circuit, so the whole time series costs **one** 12-step
propagation.

![Light cone heat maps for the 1D chain at four truncation cutoffs](../assets/b1/light_cone_1d.svg)

At `θ_h = π/2` every gate is Clifford, so one Pauli string evolves into one
Pauli string with unit coefficient: its support is the **exact** causal cone,
measured rather than assumed. Over steps 1–8 the sum stays at exactly one
string, weight running `1, 3, 5, …, 15` and cone radius `0, 1, 2, …, 7` —
**1.000 sites per Trotter step**, the structural bound of a single-site `X`
layer followed by one *commuting* `ZZ` layer moving the boundary by at most
one bond.

### Where the cutoff gives out

![Support growth and discarded weight against Trotter step](../assets/b1/support_growth.svg)

Four cutoffs, twelve steps. At the final step:

| `min_abs_coeff` | terms at `t=12` | `N(12)` | support (`w>10⁻⁶`) | front (`w>10⁻⁴`) | wall, 12 steps |
|---:|---:|---:|---:|---:|---:|
| 10⁻³ | 11 900 | 0.0273 | 17 | 8 | 0.3 s |
| 10⁻⁴ | 4 342 184 | 0.7158 | 23 | 10 | 2.2 s |
| 10⁻⁵ | 75 099 815 | 0.9767 | 23 | 11 | 25.4 s |
| 3·10⁻⁶ | 227 673 152 | 0.9953 | 23 | 11 | 63.5 s |

At 10⁻³ the sum *shrinks* after step 9 (59 140 → 11 900 terms) while `N`
collapses to 2.7% — meaningless past step ~8, and `N(t)` is the only signal
that says so. At 3·10⁻⁶ the retained norm stays above 0.999 through step 9.

### The OTOC

![OTOC front and convergence panel](../assets/b1/otoc_1d.svg)

`C(r, t)` at `r = centre + 5`, `t = 12`, as the cutoff tightens: `0.031137`
(10⁻³), `1.003293` (10⁻⁴), `1.346556` (10⁻⁵), `1.368813` (3·10⁻⁶) — the
loosest cutoff wrong by 40×, the two tightest agreeing to 2.2·10⁻² (1.6%
relative). The cone interior saturates at `C ≈ 1.3–1.4`, heading for `C → 1`.

![Headline OTOC against truncation](../assets/b1/convergence_panel_1d.svg)

### Butterfly velocity

Fitting front distance against step (steps 3–12), for every cutoff and
contour level:

| `min_abs_coeff` | `w > 10⁻²` | `w > 10⁻⁴` | `w > 10⁻⁶` |
|---:|---:|---:|---:|
| 10⁻³ | 0.600 | 0.721 | 0.752 |
| 10⁻⁴ | 0.776 | 0.945 | **1.000** |
| 10⁻⁵ | 0.848 | **1.000** | **1.000** |
| 3·10⁻⁶ | 0.903 | **1.000** | **1.000** |

**Verdict at `θ_h = 0.9`: `v_B = 1.000` sites per Trotter step, converged** —
the structural bound measured above. The `w > 10⁻²` column is the
cautionary tale: still climbing (0.776 → 0.848 → 0.903), it reads
`v_B ≈ 0.9` to anyone who picks one contour and stops.

![Front, fits, and v_B against truncation](../assets/b1/butterfly_velocity_1d.svg)

Scanning the kick angle at two cutoffs (`n = 61`, 12 steps, 27 s total, `w >
10⁻⁴` contour, `min_abs_coeff = 10⁻⁵`) gives a velocity rising with kick
strength and saturating the causal bound, **0.327 → 0.503 → 0.776 → 1.000**
for `θ_h = 0.2, 0.4, 0.6, 0.9`. Only the first two are converged
(`N(12) = 1.0000`); at `θ_h = 0.6` and `0.9` velocity moves by 0.05 between
cutoffs (`N(12) = 0.9995` and `0.7158`).

![Velocity against kick angle](../assets/b1/velocity_vs_kick_angle.svg)

## The 2D quench

Open 7×7 lattice (49 sites), seed `Z` at the centre, and a **physical**
Trotter step of `H = J Σ Z_iZ_j + h Σ X_i` with `J = h = 1`, `dt = 0.15`, up
to `t = 1.5` — physical time, not a Floquet period, so the step count
resolves the dynamics rather than merely advancing it.

### 2D is the hard case {#why-2d-is-the-hard-case}

In 1D the causal cone holds `O(t)` sites; in 2D it is an area, `O(t²)`, and
Pauli strings inside it grow as `4^O(t²)`. A weight cap does not fix this: on
a degree-4 lattice one `exp(iπ/4 Z_iZ_j)` layer takes weight-1 to weight 5,
so `max_weight = 4` gives `N = 0.386` at step 2, `0.058` at step 4 (sum never
above two strings); `max_weight = 12` holds `N = 1.0` through step 3, then
`0.523` — the cap deletes the dynamics, not a tail.

So the 2D sweep uses `min_abs_coeff` only, and each cutoff runs until a
**term ceiling** (1.2·10⁸ stored terms) stops it, so the converged window is
measured rather than assumed:

| `min_abs_coeff` | last step | `t` | terms there | `N` there |
|---:|---:|---:|---:|---:|
| 10⁻⁴ | 10 | 1.50 | 2 874 707 | 0.8077 |
| 10⁻⁶ | 7 | 1.05 | 126 017 685 | 0.99977 |
| 10⁻⁷ | 6 | 0.90 | 229 524 102 | 0.999997 |

A cutoff loose enough to reach `t = 1.5` has thrown away 19% of the operator;
one tight enough to keep 99.9997% of it hits the term ceiling at `t = 0.9`.

### Magnetization

![Magnetization and discarded weight for the 2D quench](../assets/b1/quench_observables_2d.svg)

**Verdict: the 2D quench magnetization is converged to ~10⁻³ out to `t = 0.90`
(6 Trotter steps), and not converged beyond it.** The two tightest cutoffs
differ by 0, 0, 6.5·10⁻⁵, 1.3·10⁻⁴, 3.1·10⁻⁴ and 1.2·10⁻³ at steps 1–6; past
step 6 only the two loosest cutoffs reach at all, disagreeing by up to
3.8·10⁻² — shown in the figure's right-hand panel.

![2D light cone maps and radial convergence](../assets/b1/light_cone_2d.svg)

The maps are diamonds, not squares — a Manhattan ball, since one Trotter
step moves the boundary by one bond. Mean per-site weight at step 6,
`min_abs_coeff = 10⁻⁷`, by graph distance 0…6: `0.985, 0.504, 0.0402,
4.3·10⁻⁴, 8.4·10⁻⁷, 1.2·10⁻¹¹, 0` — one to two orders of magnitude per site.

![Infinite-temperature two-point function](../assets/b1/correlator_2d.svg)

The two-point function is a *sensitivity floor*, not an error: a coefficient
`G(r,t)` below `min_abs_coeff` reads back as exactly `0.0`. At step 6 the
only sites with `|G| > 0` are the centre (`G = 0.3477`) and the four
diagonal distance-2 sites (`G = 5.235·10⁻⁵` each, as symmetry requires);
every other site reads exactly zero. A tighter `10⁻¹⁴` probe on a 5×5
lattice shows the centre alone through step 3, so at least part of this is a
real selection rule, unresolved here.

![Headline magnetization against truncation](../assets/b1/convergence_panel_2d.svg)

## Performance

Wall times above are 32-thread on the default Rayon pool, taken on a shared,
loaded workstation (~2× an idle host for the 2D sweep); nothing here is a
benchmark claim. Peak resident memory: **19 GB** for the 2D script; the 1D
script's tightest step holds 2.3·10⁸ strings, ~7 GB in the engine plus about
the same again in the numpy export.

**3D scalability.** On a 3×3×3 cubic lattice (27 sites, 54 bonds), same
quench and term ceiling as the 2D sweep, the converged cutoff (`10⁻⁶`)
reaches `t = 0.90` with 4.9·10⁸ strings — 88 s for that step, growing ~6×
per step; step 7 projects to ~3·10⁹ strings (≈100 GB) and ~10 minutes.
Truncation error is directly visible: `⟨Z_c⟩` must lie in `[−1, 1]`, and the
looser `10⁻⁵` cutoff reports **1.010** and **1.088** at steps 7–8, exactly
where `N` has fallen to 0.97 and 0.91. This 27-site pilot is a **lower**
bound — its centre is 3 bonds from the farthest corner, so by step 4 the
cone already covers the whole lattice.

## Validation

`scrambling.dense_*` builds `U(t)` as an explicit `2ⁿ×2ⁿ` matrix by Kronecker
products and evaluates the same quantities with dense linear algebra — no
`PauliSum`, no engine, no qiskit. At `n = 9`, 4 Trotter steps, untruncated
(1430 terms), against a 10⁻¹⁰ bar, the worst gap between engine and dense is
6.7·10⁻¹⁶ on every coefficient, 3.1·10⁻¹⁵ on `⟨O,O⟩`, 4.0·10⁻¹⁵ on the
support profile, 5.8/5.3/3.8·10⁻¹⁵ on the three OTOC probes, 4.9·10⁻¹⁷ on
the two-point function, and 2.2·10⁻¹⁶ on the probe-average identity. The 2D
script repeats this on a 3×3 lattice with the quench magnetization added:
worst gap 2.1·10⁻¹⁴.

At `min_abs_coeff = 0.05` the truncated leg keeps 54 of 1430 terms, discards
**0.318** of the norm, and its worst per-site weight error is **0.280** — a
ratio of 0.88, licensing `1 − N(t)` as the error proxy used throughout: a
calibration, not a theorem, so the script asserts a factor-of-10 band.

The CI-visible gate (`pytest python/paulistrings/tests/test_showcase_b1.py`)
is numpy-only and runs in under a second. Reference host: Intel Xeon Gold
6244 @ 3.60 GHz (`ccqlin038`), rustc 1.94.0, Python 3.11.11.

---

**Numbers:** computed by
[`run_b1_1d.py`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/b1_operator_scrambling/run_b1_1d.py) /
[`run_b1_2d.py`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/b1_operator_scrambling/run_b1_2d.py),
with full per-`(cutoff, step)` records in `results_1d.json` /
`results_2d.json` and the headline table in
[the README](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/b1_operator_scrambling/README.md).
No number here is from literature.
