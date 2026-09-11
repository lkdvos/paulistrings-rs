# B2 — Noisy circuit verification

Per-gate depolarizing noise added to the 127-qubit heavy-hex kicked-Ising circuit of Benchmark C makes Pauli propagation cheaper, the opposite of how a density-matrix method scales.

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python examples/b2_noisy_verification/run_b2.py           # ~26 min
RAYON_NUM_THREADS=1 python examples/b2_noisy_verification/run_b2.py --quick   # ~1 s, 20 qubits, writes nothing
python examples/b2_noisy_verification/run_b2.py --figures-only                # re-render the SVGs
pytest examples/tests/test_showcase_b2.py                                     # the CI gate
```

`RAYON_NUM_THREADS=1` must be exported before the interpreter starts; the driver refuses to run otherwise.
The full run writes `results.json`, `summary.json`, and three `.svg` figures next to this README.

Full writeup, headline numbers, and provenance: https://lkdvos.github.io/paulistrings-rs/showcases/b2-noisy-verification.html
