# B7 — Stabilizer-state preparation

stim prepares a 36-qubit 2D cluster state as a stabilizer tableau; a
non-Clifford tail is propagated in the Heisenberg picture; the expectation is
contracted against the stabilizer state at `O(m·n²/64)` cost, avoiding the 1.0
TiB state vector a dense check would need. Worst validation gap across dense,
projector, and qiskit Aer routes: 2.220e−15.

Full writeup: https://lkdvos.github.io/paulistrings-rs/showcases/b7-stabilizer-prep.html

## Run it

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python examples/b7_stabilizer_prep/run_b7.py          # 116 s, 10.2 GiB peak RSS
RAYON_NUM_THREADS=1 python examples/b7_stabilizer_prep/run_b7.py --quick  #  40 s,  1.5 GiB peak RSS
pytest python/paulistrings/tests/test_showcase_b7.py                        # 0.5 s
```

## Headline results

| quantity | value |
|---|---:|
| lattice | 6×6 open square, 36 qubits |
| cluster-prep gates | 96 |
| `stabilizers_from_stim` readout | 0.99 ms cold / 0.49 ms warm |
| generator identity under 523× padding | byte-identical |
| contraction slope 512→1024 (bound: 2.0) | 1.85 |
| per-word-op cost at n=1024 | 2.07 ns |
| linearity in m at n=256 | flat to 7% over two decades |
| depth-4 convergence | self-converged, `+0.3962080424` |
| depth-5 tightest point | 167M terms, 47 s propagate, 27 s contract, 10.2 GiB |
| CI gate | `test_showcase_b7.py`, 36 tests, 0.5 s |

## Provenance

- host: `ccqlin038` (Intel Xeon Gold 6244 @ 3.60GHz)
- commit: `13e7e9a7bab4c98fa5bb17b6b9259bd242988a43`
- date: 2026-09-01
- versions: paulistrings 0.1.0, stim 1.16.0, numpy 2.4.6, rustc 1.94.0
- thread count: 1 (`RAYON_NUM_THREADS=1`, asserted by the script)
- artifacts: `run_b7.py`, `stabilizer_prep.py`, `theta_sweep.csv`/`.svg`,
  `scaling.csv`/`.svg`, `prep_depth.csv`, `convergence_panel.svg`,
  `results_b7.json`, `validation_b7.json`
