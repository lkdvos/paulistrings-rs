# B6 — Resource probes

Two diagnostics read directly off `PauliSum.x_array()` / `z_array()` / `coefficients_array()`, both answering "how hard is this operator?" under different cost models: Pauli-spectrum entropy (a magic-adjacent diagnostic for truncation-based Pauli propagation) and operator entanglement (the cost model for matrix-product-operator methods). Diagnostics live in [`resource_probes.py`](resource_probes.py); the script is [`run_b6.py`](run_b6.py); the CI-safe correctness gate is [`test_showcase_b6.py`](../../python/paulistrings/tests/test_showcase_b6.py) (18 tests, numpy-only, under a second).

Full writeup: https://lkdvos.github.io/paulistrings-rs/showcases/b6-resource-probes.html

## Run it

```bash
source .venv/bin/activate
python examples/b6_resource_probes/run_b6.py
```

Regenerates both CSVs, `exact_cross_check.json`, and all three SVGs in well under a minute.

## Headline results

Numbers below are not in the checked-in CSVs/JSON.

The exhaustive dense-spectrum oracle (n=6 only; refused past n=8) took 0.6 s at n=6 and **99 s** at n=8. Pinning `OMP/OPENBLAS/MKL_NUM_THREADS=1` before importing numpy keeps the 462×1715 SVD at 0.14 s; left to spawn its own LAPACK pool on a busy host, the same SVD was measured at **56 s**.

Truncation convergence, `min_abs_coeff` swept at depth 6 (against the exact value) and depth 7 (self-converged, no exact reference):

| depth | `min_abs_coeff` | terms | kept `Σ\|c\|²` | `S_2` | `S_op` |
|---:|---:|---:|---:|---:|---:|
| 6 | 1e-03 | 2358 | 0.9986159568 | 3.91076 | 1.31052 |
| 6 | 1e-04 | 7527 | 0.9999718822 | 3.91348 | 1.30720 |
| 6 | 1e-05 | 17234 | 0.9999996090 | 3.91353 | 1.30714 |
| 6 | 1e-06 | 31344 | 0.9999999964 | 3.91353 | 1.30714 |
| 6 | 1e-07 | 46375 | 1.0000000000 | 3.91353 | 1.30714 |
| 6 | *exact* | 208012 | 1.0000000000 | 3.91353 | 1.30714 |
| 7 | 1e-03 | 4973 | 0.9961022533 | 4.59490 | 1.22941 |
| 7 | 1e-04 | 18493 | 0.9998972794 | 4.60630 | 1.21885 |
| 7 | 1e-05 | 50822 | 0.9999979933 | 4.60659 | 1.21844 |
| 7 | 1e-06 | 112727 | 0.9999999689 | 4.60660 | 1.21844 |
| 7 | 1e-07 | 205283 | 0.9999999996 | 4.60660 | 1.21844 |

```
depth 6, |gap| vs exact, S_2 :  2.78e-03, 5.62e-05, 7.82e-07, 7.13e-09, 4.58e-11
depth 6, |gap| vs exact, S_op:  3.38e-03, 5.47e-05, 7.57e-07, 1.64e-09, 3.70e-11
depth 7, successive drift, S_2 :  1.14e-02, 2.92e-04, 3.95e-06, 6.13e-08
depth 7, successive drift, S_op:  1.06e-02, 4.05e-04, 8.11e-06, 1.14e-07
```

## Provenance

`theta_sweep.csv`, `depth_sweep.csv`, and `exact_cross_check.json` in this directory carry the raw sweeps and cross-check gaps. Cross-check run at commit `285da8bf`, CPU Intel Xeon Gold 6244 @ 3.60GHz, rustc 1.94.0, Python 3.11.11, thread pins `RAYON/OMP/OPENBLAS/MKL_NUM_THREADS=1`.
