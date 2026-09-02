# `examples/` — showcases for the examples & benchmarks suite

Every directory here is a Part B application showcase, plus the shared infrastructure the
showcases and the benchmarks both use.

Part A benchmarks A–E follow `benchmarks/python/` conventions and live there
([`../benchmarks/README.md`](../benchmarks/README.md)). Benchmark D is the exception: its
deliverable is a scaling *sweep* rather than a `pytest-benchmark` entry, so it lives at
[`xxz_chain/`](xxz_chain/).

Rust examples live under `crates/paulistrings/examples/`; this tree is Python-only.

## Shared infrastructure

| module | what it is |
|---|---|
| [`common/circuits.py`](common/circuits.py) | circuit builders: `heavy_hex_kicked_ising` (127q), `xxz_chain_trotter`, `random_su4_staircase`, `qaoa`, `hardware_efficient_ansatz`. Every circuit is built one gate per `Circuit.push`, so truncation points line up across engines. |
| [`common/observables.py`](common/observables.py) | `Z(q)`, the Kim et al. (2023) weight-10/17 operators, sparse XXZ Hamiltonians |
| [`common/oracles.py`](common/oracles.py) | ground truth: `statevector_expectation` (qiskit Aer, `n ≲ 28`), `stim_clifford_exact`, `light_cone_exact`, `load_published_reference`. Each is capability-gated, so the suite runs without the optional deps. |
| [`common/harness.py`](common/harness.py) | `run_propagation` (one `RunRecord` per run: warm timing, term counts, peak RSS, oracle error), `make_policy`/`TruncationSpec`, the timing-discipline gates, and `diff_pauli_sums`/`require_parity`, the blocking cross-engine parity gate. Read its module docstring before writing a new timed script. |
| [`common/report.py`](common/report.py) | `RunRecord`/`Provenance` (commit, CPU, library versions, seeds) and the standard plots |
| [`data/`](data/) | checked-in inputs, every one provenance-tagged: the generated 127-qubit coupling map and the published Kim et al. observable supports — see [`data/README.md`](data/README.md) |
| [`data/references/`](data/references/) | ships with no files; a name `load_published_reference` cannot find is a hard error, never a fabricated number — see [`data/references/README.md`](data/references/README.md) |

`python/paulistrings/interop.py` (circuit importers: stim, qiskit, task-JSON) and
`python/paulistrings/io.py` (`.npz` save/load for a `PauliSum`) are shipped library API rather
than example code; both this tree and `benchmarks/julia/`'s task-JSON schema consume them.

## Showcases (Part B)

| | directory | what it validates | CI vs manual |
|---|---|---|---|
| B1 | [`b1_operator_scrambling/`](b1_operator_scrambling/) | operator scrambling: support growth, butterfly velocity, OTOCs, on a 1D chain and a 2D quench. Every curve carries a truncation-convergence panel. | 1D: CI-safe gate (`test_showcase_b1.py`) + `manual-short` scripts; 2D: `manual-long` |
| B2 | [`b2_noisy_verification/`](b2_noisy_verification/) | per-gate noise on the 127q kicked Ising: the tracked set collapses as noise grows (**651×** fewer peak terms and **1080×** less wall time at `p = 3e-2`, same cutoff), so noise makes this simulation cheaper. Includes a utility-verification demo and a dense density-matrix cross-check of all five noise channels. | CI-safe gate (`test_showcase_b2.py`, numpy-only, ~1 s) + `manual-long` script (~26 min, time-boxed) |
| B5 | [`b5_operator_backpropagation/`](b5_operator_backpropagation/) | hybrid depth reduction: back-propagate an observable through a circuit's tail layers, serialize it with the residual front circuit as a schema-v1 task file, and check the composed expectation against the full-circuit one | CI-safe gate (`test_showcase_b5.py`) + `manual-short` script |
| B6 | [`b6_resource_probes/`](b6_resource_probes/) | resource-theoretic diagnostics over the numpy export: Pauli-spectrum entropy/purity (magic-adjacent, but *not* the pure-state stabilizer Rényi entropy) and operator entanglement across a bipartition | CI-safe gate (`test_showcase_b6.py`, numpy-only, <1s) + `manual-short` script |
| B7 | [`b7_stabilizer_prep/`](b7_stabilizer_prep/) | a capability neither tool has alone: prepare a 36-qubit 2D cluster state in stim, read its signed generators, back-propagate an observable through a non-Clifford tail, and contract against the generators at `O(m·n²/64)` rather than the `2ⁿ` expansion (1.0 TiB here). Carries a derived closed form matched to **1.1e-16** and a dense `n ≤ 12` cross-check to 2.2e-15. | CI-safe gate (`test_showcase_b7.py`, 36 tests, 0.5 s, `stim` only for 4 of them) + `manual-short` script (116 s) |

B3 (variational pre-training) and B4 (QML/QCNN) are design stubs, blocked on a
symbolic-coefficient core redesign.

## Running

```bash
./scripts/setup.sh                                            # one-time: creates .venv, builds the extension
source .venv/bin/activate
pip install -e ".[examples]"                                  # matplotlib, stim, qiskit, qiskit-aer, numpy
maturin develop --release -m crates/paulistrings-py/Cargo.toml # rebuild after any Rust change

RAYON_NUM_THREADS=1 python examples/b1_operator_scrambling/run_b1_1d.py
RAYON_NUM_THREADS=1 python examples/b2_noisy_verification/run_b2.py --quick   # full run: drop --quick
RAYON_NUM_THREADS=1 python examples/b5_operator_backpropagation/run_b5.py
RAYON_NUM_THREADS=1 python examples/b6_resource_probes/run_b6.py
RAYON_NUM_THREADS=1 python examples/b7_stabilizer_prep/run_b7.py            # --quick: 40 s instead of 116 s
RAYON_NUM_THREADS=1 python examples/xxz_chain/run_benchmark_d.py all
```

`RAYON_NUM_THREADS=1` **must be exported before the interpreter starts** — Rayon's global pool is
built once, at the first `propagate`/`propagate_with_stats` call, and is never resized, so setting
the variable from inside a script (after `import paulistrings`) does not reliably reach it.
`common/harness.py`'s module docstring records the measured thread-count facts.

Every CI-visible test (`python/paulistrings/tests/test_showcase_*.py`,
`test_benchmark_*.py`) `importorskip`s `stim`/`qiskit`/`matplotlib`, so
`pytest python/paulistrings/tests` stays green in the numpy-only CI job even without the
`examples` extra installed. The scripts under this directory are not collected by CI; run them
manually as above.

## Cross-engine baseline

The `PauliPropagation.jl` baseline, its task-JSON schema, and the measured semantics divergences
between the two engines are documented in
[`../benchmarks/julia/README.md`](../benchmarks/julia/README.md).
