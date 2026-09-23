# B1 — Operator scrambling

Heisenberg-evolves a single-site `Z` through a kicked-Ising circuit and reads the light cone, OTOC, butterfly velocity, and two-point function off the evolved Pauli sum in one pass, on a 1D chain and a 2D quench.

```bash
source .venv/bin/activate
python examples/b1_operator_scrambling/run_b1_1d.py     # 1D chain, a few minutes
python examples/b1_operator_scrambling/run_b1_2d.py     # 2D quench, ~15 minutes
pytest examples/tests/test_showcase_b1.py               # CI gate, numpy-only, <1 s
```

Both scripts run on the default 32-worker Rayon pool.
These are not laptop runs: peak RSS reaches double-digit GB.
Each run rewrites `results_1d.json` / `results_2d.json` and the `.svg` figures next to itself.

Full writeup, headline numbers, and provenance: https://lkdvos.github.io/paulistrings-rs/examples/showcases/b1-operator-scrambling.html
