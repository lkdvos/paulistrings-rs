# B7 — Stabilizer-state preparation

stim prepares a 36-qubit 2D cluster state as a stabilizer tableau; a non-Clifford tail is propagated in the Heisenberg picture; the expectation is contracted against the stabilizer state at `O(m·n²/64)` cost, avoiding the 1.0 TiB state vector a dense check would need.

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python examples/b7_stabilizer_prep/run_b7.py          # 116 s, 10.2 GiB peak RSS
RAYON_NUM_THREADS=1 python examples/b7_stabilizer_prep/run_b7.py --quick  #  40 s,  1.5 GiB peak RSS
pytest examples/tests/test_showcase_b7.py                                 # 36 tests, 0.5 s
```

The run writes `theta_sweep.csv`/`.svg`, `scaling.csv`/`.svg`, `prep_depth.csv`, `convergence_panel.svg`, `results_b7.json`, and `validation_b7.json` next to this README.

Full writeup, headline numbers, and provenance: https://lkdvos.github.io/paulistrings-rs/examples/showcases/b7-stabilizer-prep.html
