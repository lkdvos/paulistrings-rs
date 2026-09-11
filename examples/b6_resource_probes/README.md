# B6 — Resource probes

Two diagnostics read directly off `PauliSum.x_array()` / `z_array()` / `coefficients_array()`, both answering "how hard is this operator?" under different cost models: Pauli-spectrum entropy (a magic-adjacent diagnostic for truncation-based Pauli propagation) and operator entanglement (the cost model for matrix-product-operator methods).

```bash
source .venv/bin/activate
python examples/b6_resource_probes/run_b6.py
pytest examples/tests/test_showcase_b6.py   # CI-safe gate, 18 tests, numpy-only, under a second
```

Regenerates `theta_sweep.csv`, `depth_sweep.csv`, `exact_cross_check.json`, and all three SVGs in well under a minute.

Full writeup, headline numbers, and provenance: https://lkdvos.github.io/paulistrings-rs/showcases/b6-resource-probes.html
