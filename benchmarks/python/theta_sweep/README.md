# Benchmark B — Kick-angle sweep

Heavy-hex kicked Ising, 127 qubits, 5 Trotter steps, six kick angles
θ_h ∈ {0, 0.2, π/8, π/4, 3π/8, π/2}, three observables from the utility
experiment (`Z_62`, a weight-10 operator, a weight-17 operator). Heisenberg
picture against `|0…0⟩`, single-threaded, warm timings. Scores truncated
Pauli-sum accuracy against the tightest exact or self-converged reference
reachable at each point, and checks per-layer term-count parity against
`PauliPropagation.jl`.

Full writeup: https://lkdvos.github.io/paulistrings-rs/benchmarks/b-theta-sweep.html

## Run it

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python benchmarks/python/bench_b_theta_sweep.py --validate-convergence
pytest python/paulistrings/tests/test_benchmark_b_sweep.py    # CI gate, 20-qubit sublattice
```

`RAYON_NUM_THREADS=1` must be exported before the interpreter starts.

## Headline results

Numbers quoted on the site page that are not in `results.json` or
`summary.json`:

| metric | value |
|---|---|
| `PauliPropagation.jl` RSS, heaviest parity case (weight-17, `1e-5`) | 67.6 GiB |
| this engine's RSS, same case | ~1.2 GiB |
| `Z_62`'s 19-qubit cone: Pauli-path cost vs. statevector cost | 12.8 s vs. 2.8 s |

## Provenance

| file | what it is |
|---|---|
| `../bench_b_theta_sweep.py` | the driver |
| `results.json` | 236 `report.RunRecord`s (227 paulistrings, 9 PauliPropagation.jl) |
| `summary.json` | references, endpoint checks, time-to-accuracy, parity outcomes |
| `error-vs-*.svg`, `term-count-vs-truncation.svg`, `parity-per-layer-terms.svg` | the site page's figures |

Recorded run: commit `94077fa`, clean worktree, ccqlin038 (Xeon Gold 6244 @
3.60 GHz), rustc 1.94.0, Python 3.11.11, paulistrings 0.1.0, numpy 2.4.6,
qiskit 2.5.2, stim 1.16.0, julia 1.12.6 + PauliPropagation 0.8.2. 39.6 min
end to end.
