# B2 — Noisy circuit verification

Per-gate depolarizing noise added to the 127-qubit heavy-hex kicked-Ising circuit of
Benchmark C makes Pauli propagation cheaper: at `p = 3e-2` the tracked set peaks at
651× fewer terms and finishes 1078× faster than at `p = 0`, at the same cutoff, the
opposite of how a density-matrix method scales. The mechanism, convergence sweeps,
channel-generality checks, and a hand-rolled dense Kraus cross-check are recorded in
the full writeup.

Full writeup: https://lkdvos.github.io/paulistrings-rs/showcases/b2-noisy-verification.html

## Run it

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python examples/b2_noisy_verification/run_b2.py           # ~26 min
RAYON_NUM_THREADS=1 python examples/b2_noisy_verification/run_b2.py --quick   # ~1 s, 20 qubits, writes nothing
python examples/b2_noisy_verification/run_b2.py --figures-only                # re-render the SVGs
pytest python/paulistrings/tests/test_showcase_b2.py                          # the CI gate
```

`RAYON_NUM_THREADS=1` must be exported before the interpreter starts; the driver
refuses to run otherwise.

## Headline results

`θ_h = 5π/16`, 20 Trotter steps, `Z_62`, Heisenberg, single-threaded, fixed
`min_abs_coeff = 2⁻¹⁴`:

| p | peak terms | wall (1 thread) | peak vs `p = 0` |
|---|---|---|---|
| 0 | 14 396 463 | 531.0 s | — |
| 3e-2 | 22 105 | 0.49 s | 651× fewer, 1078× faster |

Process RSS peaks at 1.25 GiB across the full driver run, which covers 31 records
and takes 25.7 min single-threaded end to end.

## Provenance

Recorded on ccqlin038 (Intel Xeon Gold 6244 @ 3.60 GHz), rustc 1.94.0, Python
3.11.11, commit `a3d260f`, single-threaded. Raw records are in `results.json`,
sweep verdicts and cited references in `summary.json`, and figures in the three
`.svg` files next to this README.
