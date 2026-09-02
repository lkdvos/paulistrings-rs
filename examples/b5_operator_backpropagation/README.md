# B5 — Hybrid depth reduction

Splits a circuit near the end and back-propagates the observable through the
tail classically in the Heisenberg picture, so a QPU only has to run the
shorter front circuit against a modified observable. The composed
expectation is exactly the full-circuit one; a schema-v1 task file carries
the front circuit and evolved observable to a QPU-side runner. Script:
[`run_b5.py`](run_b5.py); correctness gate:
[`test_showcase_b5.py`](../../python/paulistrings/tests/test_showcase_b5.py).

Full writeup: https://lkdvos.github.io/paulistrings-rs/showcases/b5-operator-backpropagation.html

## Run it

```bash
source .venv/bin/activate
python examples/b5_operator_backpropagation/run_b5.py
pytest python/paulistrings/tests/test_showcase_b5.py
```

## Headline results

| quantity | value |
|---|---|
| round-trip gap (composed vs. full-circuit expectation) | `0.0` |
| qiskit-Aer statevector cross-check gap | 1.665e-16 |
| truncation gap at `min_abs_coeff=1e-6` (64465 terms) | 1.483e-07 |
| truncation gap at `min_abs_coeff=1e-8` (64786 terms) | 1.531e-09 |
| full sweep runtime | ~3 s (laptop-class) |

Depth-vs-term-count sweep (`k` = 0..6 tail steps) is in
[`depth_vs_terms.csv`](depth_vs_terms.csv).

## Provenance

| | |
|---|---|
| host | ccqlin038 (ccq workstation) |
| commit | `94077fa` (2026-08-31) |
| date | 2026-09-01 |
| versions | Python 3.11; packages per `pyproject.toml` `examples` extra (numpy, matplotlib, qiskit, qiskit-aer) |
| artifacts | `task_exact.json`, `task_truncated.json`, `evolved_observable_exact.npz`, `depth_vs_terms.csv`, `depth_vs_terms.svg`, `convergence_panel.svg` |
