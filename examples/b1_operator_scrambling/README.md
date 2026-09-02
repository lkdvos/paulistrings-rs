# B1 — Operator scrambling

Heisenberg-evolves a single-site `Z` through a kicked-Ising circuit and reads
the light cone, OTOC, butterfly velocity, and two-point function off the
evolved Pauli sum in one pass. At `θ_h = 0.9` the 1D butterfly velocity
converges to the causal bound, `v_B = 1.000` sites/step; the 2D quench
magnetization is converged to ~10⁻³ only out to `t = 0.90` (6 Trotter steps).
Every quantity is checked against an independent dense `2ⁿ×2ⁿ` construction,
worst gap 5.8·10⁻¹⁵ (1D) / 2.1·10⁻¹⁴ (2D).

Full writeup: https://lkdvos.github.io/paulistrings-rs/showcases/b1-operator-scrambling.html

## Run it

```bash
source .venv/bin/activate
python examples/b1_operator_scrambling/run_b1_1d.py     # 1D chain, a few minutes
python examples/b1_operator_scrambling/run_b1_2d.py     # 2D quench, ~15 minutes
pytest python/paulistrings/tests/test_showcase_b1.py    # CI gate, numpy-only, <1 s
```

Both scripts run on the default 32-worker Rayon pool. Lower `STEPS` or drop
the tightest entry of `EPS_GRID` (module-level constants) to run smaller;
the term ceiling stops a cutoff safely in any case. These are not laptop
runs — see peak RSS below.

## Headline results

| quantity | 1D chain (`n=61`, `θ_h=0.9`, 12 steps) | 2D quench (7×7, `dt=0.15`) |
|---|---|---|
| converged window | `v_B = 1.000` at `min_abs_coeff ≤ 10⁻⁵` | `t ≤ 0.90` (6 steps), `N > 0.999` |
| peak terms | 227 673 152 (`t=12`, `3·10⁻⁶`) | 229 524 102 (`t=0.90`, `10⁻⁷`) |
| peak RSS | ~7 GB engine + ~7 GB numpy export | 19 GB |
| wall, 32 threads | 63.5 s (12 steps, `3·10⁻⁶`) | ~15 min (full sweep) |
| dense cross-check, worst gap | 5.8·10⁻¹⁵ | 2.1·10⁻¹⁴ |

The numbers below are printed by the scripts, not persisted to JSON (only
the main sweeps are), so this is their committed record:

- Dense-vs-engine worst gap (`run_b1_1d.py::run_validation`, `n=9`, 4 steps,
  untruncated, `10⁻¹⁰` bar): `6.7·10⁻¹⁶` coefficients, `3.1·10⁻¹⁵` `⟨O,O⟩`,
  `4.0·10⁻¹⁵` support, `5.8/5.3/3.8·10⁻¹⁵` OTOC `X/Y/Z`, `4.9·10⁻¹⁷`
  two-point, `2.2·10⁻¹⁶` probe-average identity. At `min_abs_coeff = 0.05`
  the truncated leg keeps 54/1430 terms, discarding `0.318` of the norm
  against a `0.280` worst per-site weight error.
- Clifford point (`run_b1_1d.py::run_clifford_cone`, `θ_h = π/2`): one
  string throughout, weight `1, 3, …, 15`, radius `0, 1, …, 7` (steps 1–8).
- Weight-cap probe (`run_b1_2d.py::run_weight_cap_probe`, 7×7 lattice):
  `max_weight=4` gives `N = 0.386399`/`0.057691` at steps 2/4; `max_weight=12`
  gives `N = 1.000000`/`0.523203` at steps 3/4.
- 3D pilot (`run_b1_2d.py::run_3d_pilot`, 3×3×3 lattice): at
  `min_abs_coeff = 10⁻⁶`, `t = 0.90` reaches 4.9·10⁸ terms at 88 s/step,
  growing ~6× per step; at `10⁻⁵`, `⟨Z_c⟩` leaves `[−1,1]`, reporting
  `1.01024` (`t=1.05`, `N=0.972626`) and `1.08766` (`t=1.20`, `N=0.911502`).

## Provenance

Recorded on `ccqlin038` (Intel Xeon Gold 6244 @ 3.60 GHz), rustc 1.94.0,
Python 3.11.11, commit `dd7ab7c8`, 32 threads, 2026-08-31. Raw records:
`results_1d.json` / `results_2d.json`; figures: the ten `.svg` files here.
