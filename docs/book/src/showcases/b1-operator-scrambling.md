# B1 — Operator scrambling, light cones and OTOCs

<p class="lead">Heisenberg-evolve a single-site <code>Z</code> through a kicked
Ising circuit and watch the operator spread. Every quantity here — the light
cone, the out-of-time-order correlator, the butterfly velocity, the
infinite-temperature two-point function — is read off the evolved Pauli sum in
one pass, and every one of them is checked against a dense
<code>2ⁿ×2ⁿ</code> construction that shares no code with the engine.</p>

![Light cone heat maps for the 1D chain at four truncation cutoffs](../assets/b1/light_cone_1d.svg)

*The 1D light cone: per-site weight `w_q(t)` against site and Trotter step, at
four coefficient cutoffs. The support fills the entire causally available cone —
23 sites at step 12 — at every cutoff from 10⁻⁴ down, which is precisely why
support size is a poor probe of truncation quality and the front contour is the
sensitive one.*

## What is computed, and why it is cheap

The evolved operator is a Pauli sum, `O(t) = U(t)† O U(t) = Σ_P c_P(t) P`, seeded
with `O = Z_c` at the centre site. In this repository's Hermitian convention
(`Y → (x=1, z=1)`, no phase) the `c_P` are real. Normalizing the trace inner
product as `⟨A,B⟩ = Tr(A†B)/2ⁿ` makes the Pauli strings orthonormal, so

```text
N(t) = ⟨O(t), O(t)⟩ = Σ_P |c_P(t)|²
```

is conserved by exact unitary evolution and equals 1 for the seed. **Under
truncation `N(t)` only falls, and `1 − N(t)` is exactly the fraction of the
operator that was thrown away.** That is the convergence diagnostic behind every
curve on this page.

Everything else follows from the symplectic bit columns, in one chunked pass:

| quantity | how it is read off |
|---|---|
| support profile `w_q(t) = Σ_{P: P_q ≠ I} \|c_P\|²` | accumulate `\|c\|²` per site from the `x`/`z` bit columns — plotted against `(site, t)` this *is* the light cone |
| OTOC `C(r,t) = 2 · Σ_{P anticommuting with W_r} \|c_P\|²` | one symplectic bit test per site: `X_r` probes `z_r(P)`, `Z_r` probes `x_r(P)`, `Y_r` probes `x_r ≠ z_r` — so one pass gives every site and all three probes at once |
| two-point function `G(r,t) = ⟨Z_r, O(t)⟩` | *literally the coefficient of the weight-one string* `Z_r`; find the weight-one rows by popcount and read them off — no second propagation, no contraction |

The chunking is not incidental: a converged run here reaches 2·10⁸ terms, where
a dense `(terms, n)` bit matrix would be tens of gigabytes.

The OTOC form is derived rather than quoted. `W_r` is a Pauli, so each string
either commutes with it or anticommutes, in which case `[W_r, P] = 2 W_r P`;
writing `A` for the anticommuting part gives `[W_r, O] = 2 W_r A` and, by
unitarity and orthonormality, `C = 4⟨A,A⟩/2 = 2 Σ_anti |c_P|²`. That form needs
no cancellation between large numbers, unlike the equivalent `C = N − F`.

### A free self-test

Average the OTOC over `W_r ∈ {X_r, Y_r, Z_r}`. A string with `P_r = I`
anticommutes with none of them; a string with `P_r ≠ I` anticommutes with
exactly two. So `⅓ Σ_W C_W(r,t) = (4/3) · w_r(t)`, exactly, string by string —
and comparing the two independently accumulated arrays is a machine-precision
self-test of both. Both scripts assert it at every step and report it at
`0.0`–`2·10⁻¹⁶`.

## Validation against a dense construction

`scrambling.dense_*` builds `U(t)` as an explicit `2ⁿ×2ⁿ` matrix by Kronecker
products and evaluates the same quantities with dense linear algebra — no
`PauliSum`, no engine, no qiskit. The support profile comes from the
single-qubit Pauli twirl there, so no Pauli decomposition is involved anywhere
on the reference side.

At `n = 9`, 4 Trotter steps, untruncated (1430 terms), against a 10⁻¹⁰ bar:

| compared quantity | max gap, engine vs dense |
|---|---:|
| every coefficient, `c_P` vs `⟨P, O(t)⟩` | 6.7·10⁻¹⁶ |
| `⟨O,O⟩` (rules out a *missing* term) | 3.1·10⁻¹⁵ |
| support profile `w_q`, vs the dense twirl | 4.0·10⁻¹⁵ |
| OTOC `C_X` / `C_Y` / `C_Z` | 5.8 / 5.3 / 3.8 ·10⁻¹⁵ |
| two-point function `G(r)` | 4.9·10⁻¹⁷ |
| probe-average identity | 2.2·10⁻¹⁶ |

The 2D script repeats the comparison on a 3×3 lattice and adds the quench
magnetization against a dense matrix element, worst gap 2.1·10⁻¹⁴.

The truncated leg calibrates the diagnostic: at `min_abs_coeff = 0.05` the run
keeps 54 of 1430 terms, discards **0.318** of the norm, and its worst per-site
weight error is **0.280** — a ratio of 0.88. The error in `w_q` tracks the norm
that was deleted, which is what licenses `1 − N(t)` as the error proxy
throughout. It is a calibration and not a theorem, so the script asserts a
factor-of-10 band rather than an inequality.

## The 1D chain

Open `n = 61` chain, kicked-Ising entangler `θ_zz = −π/2`, kick angle
`θ_h = 0.9`, seed `Z_30`, 12 Trotter steps, Heisenberg. The whole time series
costs **one** 12-step propagation, because this engine truncates after every
channel: `t` successive calls on the one-step circuit apply exactly the same
(apply-adjoint, truncate) sequence as one call on the `t`-step circuit.

### The exact cone, measured at the Clifford point

At `θ_h = π/2` every gate is Clifford, so one Pauli string evolves into one
Pauli string with unit coefficient and its support is the **exact** causal cone
— measured, not assumed:

| Trotter step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| strings in the sum | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Pauli weight | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 15 |
| cone radius | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |

The radius grows at exactly **1.000 sites per Trotter step** — the structural
bound, since one step is a single-site `X` layer followed by one *commuting*
`ZZ` layer that can move the boundary by at most one bond. The offset (radius
`t − 1`) is real: in the Heisenberg direction the channel list runs in reverse,
so the first layer the `Z`-type seed meets is a `ZZ` layer it commutes through.

### Where the cutoff gives out

![Support growth and discarded weight against Trotter step](../assets/b1/support_growth.svg)

Four cutoffs, twelve steps. At the final step:

| `min_abs_coeff` | terms at `t=12` | `N(12)` | support (`w>10⁻⁶`) | front (`w>10⁻⁴`) | wall, 12 steps |
|---:|---:|---:|---:|---:|---:|
| 10⁻³ | 11 900 | 0.0273 | 17 | 8 | 0.3 s |
| 10⁻⁴ | 4 342 184 | 0.7158 | 23 | 10 | 2.2 s |
| 10⁻⁵ | 75 099 815 | 0.9767 | 23 | 11 | 25.4 s |
| 3·10⁻⁶ | 227 673 152 | 0.9953 | 23 | 11 | 63.5 s |

And the retained norm step by step:

| step | 10⁻³ | 10⁻⁴ | 10⁻⁵ | 3·10⁻⁶ | terms at 3·10⁻⁶ |
|---:|---:|---:|---:|---:|---:|
| 1–4 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 588 (at step 4) |
| 6 | 0.989618 | 0.999941 | 1.000000 | 1.000000 | 56 628 |
| 8 | 0.882807 | 0.995250 | 0.999907 | 0.999991 | 2 555 347 |
| 10 | 0.469623 | 0.946782 | 0.997914 | 0.999684 | 31 700 066 |
| 12 | 0.027345 | 0.715770 | 0.976736 | 0.995313 | 227 673 152 |

Read that as the failure mode of a coefficient cutoff, in real time. At 10⁻³ the
sum *shrinks* after step 9 (59 140 → 11 900 terms) while the norm collapses to
2.7%: every coefficient has fallen below the cutoff and is being deleted, so the
"operator" being propagated is almost entirely gone. That is not a slow
degradation of accuracy — the 10⁻³ curve is meaningless past step ~8, and the
only thing that says so is `N(t)`. At 3·10⁻⁶ the same 12 steps keep 99.53% of
the operator.

### The OTOC

![OTOC front and convergence panel](../assets/b1/otoc_1d.svg)

`C(r, t)` at `r = centre + 5`, `t = 12`, as the cutoff tightens:

| `min_abs_coeff` | 10⁻³ | 10⁻⁴ | 10⁻⁵ | 3·10⁻⁶ |
|---|---:|---:|---:|---:|
| `C(centre+5, 12)` | 0.031137 | 1.003293 | 1.346556 | 1.368813 |

The loosest cutoff is wrong by a factor of 40; the two tightest agree to
2.2·10⁻² (1.6% relative). The cone interior saturates at `C ≈ 1.3–1.4`, heading
for the `C → 1` a fully Haar-scrambled operator would give against a single-site
probe.

![Headline OTOC against truncation](../assets/b1/convergence_panel_1d.svg)

### Butterfly velocity — and why one number is not enough

Fitting front distance against step over steps 3–12, for every cutoff and every
contour level:

| `min_abs_coeff` | `w > 10⁻²` | `w > 10⁻⁴` | `w > 10⁻⁶` |
|---:|---:|---:|---:|
| 10⁻³ | 0.600 | 0.721 | 0.752 |
| 10⁻⁴ | 0.776 | 0.945 | **1.000** |
| 10⁻⁵ | 0.848 | **1.000** | **1.000** |
| 3·10⁻⁶ | 0.903 | **1.000** | **1.000** |

> **Verdict at `θ_h = 0.9`: `v_B = 1.000` sites per Trotter step, converged** —
> and it is the *structural bound* of the Clifford-point measurement above, not
> a dynamical velocity. The front is causally saturated.

The `w > 10⁻²` column is the cautionary tale. It is still climbing
(0.776 → 0.848 → 0.903) and would be reported as `v_B ≈ 0.9` by anyone who
picked one contour, one cutoff, and stopped. A contour sitting *inside* the
front's exponential tail measures the tail's shape, not the front's speed, and
only the cutoff sweep reveals which of the two you have.

![Front, fits, and v_B against truncation](../assets/b1/butterfly_velocity_1d.svg)

To show the readout measuring something rather than a bound, the kick angle is
scanned at two cutoffs (`n = 61`, 12 steps, 27 s for the whole scan):

| `θ_h` | `w > 10⁻⁴` @ 10⁻⁵ | `N(12)` @ 10⁻⁵ | converged? |
|---:|---:|---:|---|
| 0.2 | 0.327 | 1.0000 | yes — both cutoffs agree digit for digit |
| 0.4 | 0.503 | 1.0000 | yes |
| 0.6 | 0.776 | 0.9995 | **no** — velocity moves 0.05 between cutoffs |
| 0.9 | 1.000 | 0.9767 | **no** at 10⁻⁴ (`N = 0.7158` there) |

Now the velocity is a real number that rises with kick strength and saturates
the causal bound — **0.327 → 0.503 → 0.776 → 1.000** — with the convergence
verdict stated per row rather than assumed for the sweep.

![Velocity against kick angle](../assets/b1/velocity_vs_kick_angle.svg)

## The 2D quench

Open 7×7 lattice (49 sites), seed `Z` at the centre, and a **physical** Trotter
step of `H = J Σ Z_iZ_j + h Σ X_i` with `J = h = 1`, `dt = 0.15`, up to `t = 1.5`.
Time is physical here rather than a Floquet period, so the step count resolves
the dynamics instead of merely advancing it.

### Why 2D is the hard case {#why-2d-is-the-hard-case}

In 1D the causal cone holds `O(t)` sites; in 2D it is an area, `O(t²)`, and the
number of Pauli strings inside it grows as `4^O(t²)`. A weight cap was tried and
**measured useless** at this entangling strength — on a degree-4 lattice one
`exp(iπ/4 Z_iZ_j)` layer takes a weight-1 string to weight 5:

| `max_weight` | step | terms | retained `N` |
|---:|---:|---:|---:|
| 4 | 2 | 2 | 0.386399 |
| 4 | 4 | 2 | 0.057691 |
| 8 | 3 | 706 | 0.540273 |
| 8 | 4 | 2 474 | 0.219626 |
| 12 | 3 | 9 826 | 1.000000 |
| 12 | 4 | 457 456 | 0.523203 |

A cap of 4 loses 61% of the operator at the *second* step and the sum never
grows past two strings; a cap of 12 survives three steps and then loses 48%. The
cap is not truncating a tail, it is deleting the dynamics, and it buys no extra
time at all.

So the 2D sweep uses `min_abs_coeff` only — and rather than picking one cutoff,
each cutoff runs until a **term ceiling** (1.2·10⁸ stored terms) stops it. The
converged window is therefore measured, and the wall is a result rather than a
hidden assumption:

| `min_abs_coeff` | last step | `t` | terms there | `N` there |
|---:|---:|---:|---:|---:|
| 10⁻⁴ | 10 | 1.50 | 2 874 707 | 0.8077 |
| 10⁻⁵ | 10 | 1.50 | 146 378 467 | 0.9254 |
| 10⁻⁶ | 7 | 1.05 | 126 017 685 | 0.99977 |
| 10⁻⁷ | 6 | 0.90 | 229 524 102 | 0.999997 |

That table is the whole 2D story in four rows: a cutoff loose enough to reach
`t = 1.5` has thrown away 19% of the operator by the time it gets there, and a
cutoff tight enough to keep 99.9997% of it hits the term ceiling at `t = 0.9`.

### Magnetization, and where it stops being a result

![Magnetization and discarded weight for the 2D quench](../assets/b1/quench_observables_2d.svg)

> **Verdict: the 2D quench magnetization is converged to ~10⁻³ out to `t = 0.90`
> (6 Trotter steps), and not converged beyond it.** The two tightest cutoffs
> differ by 0, 0, 6.5·10⁻⁵, 1.3·10⁻⁴, 3.1·10⁻⁴ and 1.2·10⁻³ at steps 1–6. Past
> step 6 only the two loosest cutoffs reach at all, and they disagree with each
> other by up to 3.8·10⁻².

The `t > 0.9` part of the curve is *shown* but is not a converged result, and
the figure's right-hand panel is how you can tell without being told.

![2D light cone maps and radial convergence](../assets/b1/light_cone_2d.svg)

The maps are diamonds, not squares: the cone is a Manhattan ball, since one
Trotter step moves the boundary by one *bond*. Mean per-site weight at step 6,
`min_abs_coeff = 10⁻⁷`, by graph distance 0…6: `0.985, 0.504, 0.0402, 4.3·10⁻⁴,
8.4·10⁻⁷, 1.2·10⁻¹¹, 0`. Each site outward costs one to two orders of
magnitude, and the fall steepens at the cone edge — which is exactly why a front
read off a fixed contour is so contour-dependent.

![Infinite-temperature two-point function](../assets/b1/correlator_2d.svg)

**The two-point function is the one quantity the cutoff limits as a *sensitivity
floor* rather than as an error.** `G(r,t)` is a single coefficient, so if it
falls below `min_abs_coeff` it is not approximated — it is deleted, and reads
back as exactly `0.0`. At step 6 the only sites with `|G| > 0` are the centre
(`G = 0.3477`) and the four *diagonal* distance-2 sites (`G = 5.235·10⁻⁵` each,
equal to five digits, as the lattice symmetry requires); every other site reads
exactly zero. A tighter probe run at `10⁻¹⁴` on a 5×5 lattice shows the centre
alone through step 3, so at least part of the structure is a real selection rule
— **which part, this run does not resolve**, and it is recorded as an open
question rather than asserted either way.

![Headline magnetization against truncation](../assets/b1/convergence_panel_2d.svg)

## 3D: piloted, and deferred with a measured cost

The plan's 3D item was "only if trivially cheap after the 2D pilot — otherwise
record it as deferred with the projected cost". It is not trivially cheap, so
the cost was **measured** on a 3×3×3 cubic lattice (27 sites, 54 bonds):

| `min_abs_coeff` | step | `t` | terms | `N` | `⟨Z_c⟩` | s/step |
|---:|---:|---:|---:|---:|---:|---:|
| 10⁻⁵ | 6 | 0.90 | 25 374 996 | 0.993928 | 0.98603 | 8.6 |
| 10⁻⁵ | 7 | 1.05 | 73 965 828 | 0.972626 | **1.01024** | 20.9 |
| 10⁻⁵ | 8 | 1.20 | 151 989 465 | 0.911502 | **1.08766** | 57.9 |
| 10⁻⁶ | 5 | 0.75 | 80 401 298 | 0.999882 | 0.96184 | 16.1 |
| 10⁻⁶ | 6 | 0.90 | 493 365 132 | 0.998963 | 0.96635 | 87.8 |

Two things to read off. First, **truncation error becomes visible as an
unphysical answer**: `⟨Z_c⟩` must lie in `[−1, 1]`, and the 10⁻⁵ run reports
1.010 and 1.088 at steps 7–8, where `N` has fallen to 0.97 and 0.91. A curve
leaving the physical range is the loudest convergence failure there is, and it
lands at exactly the `N` the 2D sweep independently identifies as the wall.

Second, the cost. At a converged cutoff the 27-site lattice reaches `t = 0.90`
with 4.9·10⁸ strings — 88 s for that step, growing ~6× per step; step 7 projects
to ~3·10⁹ strings (≈100 GB in the engine alone) and ~10 min. The geometry
explains it: the causal ball holds 13 / 85 / 377 sites at `t = 6` in 1D / 2D /
3D, so 3D at step 6 is doing roughly what 2D would do at step ~12. And this
pilot is a **lower** bound: the centre of a 3×3×3 lattice is 3 bonds from its
farthest corner, so from step 4 on the cone already covers every site.

> **Status: a 3D production showcase is deferred**, and what it needs is not
> more time on this host but a way to bound the *number* of strings rather than
> their size. A weight cap does not do that here. `topn(k)` does, and is
> deliberately not used, because the convergence evidence this showcase rests on
> is a `min_abs_coeff` sweep and a fixed-`k` budget changes what "converged"
> even means. The other honest routes are
> [B2](b2-noisy-verification.md)'s noise channels and out-of-core storage. The
> pilot above is the cost baseline any of them would be measured against.

## Reproducing

```bash
source .venv/bin/activate
python examples/b1_operator_scrambling/run_b1_1d.py     # phase 1 — a few minutes
python examples/b1_operator_scrambling/run_b1_2d.py     # phase 2 — of order fifteen minutes
```

Each script regenerates every SVG and the results JSON next to it.
`results_1d.json` / `results_2d.json` carry the full per-site weight, OTOC and
two-point profiles for every `(cutoff, step)` point — not just the scalars — so
every figure here can be redrawn, or re-analysed at a different contour level,
without rerunning anything.

**These are not laptop runs.** Both peak in the tens of gigabytes of resident
memory: the 2D script was observed at 19 GB, and the 1D script's tightest step
holds 2.3·10⁸ strings, roughly 7 GB in the engine plus about the same again in
the numpy export the analysis pass reads. Lower `STEPS` or drop the tightest
entry of `EPS_GRID` (both module-level constants) to run smaller; the term
ceiling stops a cutoff safely in any case.

The CI-visible correctness gate is numpy-only and runs in under a second:

```bash
pytest python/paulistrings/tests/test_showcase_b1.py
```

## Caveats and sources

- **Wall times here are indicative, not benchmark numbers.** They were taken on
  a shared workstation with other jobs running — the 2D sweep in particular ran
  against a load average above 32 on 32 cores, and its per-step times are
  roughly 2× what the same steps took on an idle host. Nothing in this showcase
  is a performance claim.
- **Threads:** every sweep ran on the default 32-worker Rayon pool, recorded in
  each record's provenance block. No cross-engine timing is claimed, so the
  suite's single-thread rule for *timings* does not bind here.
- Reference host: Intel Xeon Gold 6244 @ 3.60 GHz (`ccqlin038`), rustc 1.94.0,
  Python 3.11.11.

**Source for every number on this page:**
[`examples/b1_operator_scrambling/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/b1_operator_scrambling/README.md),
with the raw records in `results_1d.json` / `results_2d.json` next to it.
