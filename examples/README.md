# `examples/` — showcases for the examples & benchmarks suite

Root-level index. Adapted plan:
[`research/plans/2026-08-31-examples-benchmarks-suite.md`](../research/plans/2026-08-31-examples-benchmarks-suite.md);
API design note for the capabilities the suite added:
[`research/notes/2026-09-01-python-api-extensions.md`](../research/notes/2026-09-01-python-api-extensions.md).
Part A benchmarks (A–E) fold into `benchmarks/python/` conventions rather than living here — see
[`../benchmarks/README.md`](../benchmarks/README.md) — except **Benchmark D**, which lives at
[`xxz_chain/`](xxz_chain/) because its deliverable is a scaling *sweep*, not a
`pytest-benchmark` entry (see that directory's README for why). Everything under this directory
is a Part B application showcase, plus the shared infrastructure they and the benchmarks both use.

Rust examples (`crates/paulistrings/examples/`) are untouched by this suite; this tree is
Python-only.

## Shared infrastructure

| module | what it is |
|---|---|
| [`common/circuits.py`](common/circuits.py) | circuit builders: `heavy_hex_kicked_ising` (127q, generated coupling map), `xxz_chain_trotter`, `random_su4_staircase`, `qaoa`/`hardware_efficient_ansatz`; every circuit is built one gate per `Circuit.push` (the jl-parity construction rule) |
| [`common/observables.py`](common/observables.py) | `Z(q)`, the Kim et al. (2023) weight-10/17 operators (parameterized by lattice size), sparse XXZ Hamiltonians — built via `PauliSum.from_strings` |
| [`common/oracles.py`](common/oracles.py) | ground-truth references: `statevector_expectation` (qiskit Aer, `n ≲ 28`), `stim_clifford_exact`, `light_cone_exact`, `load_published_reference` — every one capability-gated so the suite runs without the optional deps installed |
| [`common/harness.py`](common/harness.py) | `run_propagation` (one `report.RunRecord` per run: warm timing, term counts via `propagate_with_stats`, peak RSS, oracle error), `make_policy`/`TruncationSpec` (the two jl-comparable truncation knobs), `assert_single_threaded`/`assert_logging_quiet` (timing-discipline gates), `diff_pauli_sums`/`require_parity` (the blocking cross-engine parity gate) — read the module docstring before writing a new timed script, it corrects the thread-pinning facts in the API design note (see below) |
| [`common/report.py`](common/report.py) | `RunRecord`/`Provenance` (commit, CPU, library versions, seeds) and the standard plots (`plot_error_vs_runtime`, `plot_convergence_panel`, `plot_term_count_vs_truncation`, `plot_time_memory_vs_size`) |
| [`data/`](data/) | checked-in inputs, every one provenance-tagged: `heavy_hex_127.edges` (generated from `qiskit-ibm-runtime`'s `FakeSherbrooke` coupling map by [`generate_heavy_hex.py`](data/generate_heavy_hex.py), never hand-typed) and `kim2023_observables.json` (published operator supports, independently re-derived by a stabilizer-relation test, not just transcribed) — see [`data/README.md`](data/README.md) |
| [`data/references/`](data/references/) | ships with **no files**; a name `load_published_reference` doesn't find is a hard error, never a fabricated number — see [`data/references/README.md`](data/references/README.md) |

`python/paulistrings/interop.py` (circuit importers: stim, qiskit, task-JSON) and
`python/paulistrings/io.py` (`.npz` save/load for a `PauliSum`) are shipped library API, not
example code — they live under `python/paulistrings/`, consumed by both this tree and
`benchmarks/julia/`'s task-JSON schema.

## Showcases (Part B)

| | directory | what it validates | CI vs manual |
|---|---|---|---|
| **B1** | [`b1_operator_scrambling/`](b1_operator_scrambling/) | operator scrambling: support growth, butterfly velocity, OTOCs, 1D chain then a 2D quench — every curve carries a truncation-convergence panel | 1D: CI-safe gate (`test_showcase_b1.py`) + `manual-short` scripts; 2D: `manual-long` |
| **B2** | `b2_noisy_verification/` | per-gate noise on the 127q kicked Ising: noise-accelerated truncation, and a "utility verification" demo reproducing Benchmark C's converged answer in the noiseless limit — written by a concurrent branch of this suite; CI gate `python/paulistrings/tests/test_showcase_b2.py` | `manual-short` |
| **B5** | [`b5_operator_backpropagation/`](b5_operator_backpropagation/) | hybrid depth reduction: back-propagate an observable through a circuit's tail layers classically (`direction="heisenberg"`), serialize the evolved observable + residual front circuit as a schema-v1 task file (`PauliSum.from_arrays`/`.npz` + `interop`), and check the composed expectation matches the full-circuit one | CI-safe gate (`test_showcase_b5.py`) + `manual-short` script |
| **B6** | [`b6_resource_probes/`](b6_resource_probes/) | resource-theoretic diagnostics computed read-only over the numpy export: Pauli-spectrum entropy/purity (magic-adjacent, but explicitly *not* the pure-state stabilizer Rényi entropy) and operator entanglement across a bipartition (matrix-product-operator cost model) | CI-safe gate (`test_showcase_b6.py`, numpy-only, <1s) + `manual-short` script |

B3 (variational pre-training), B4 (QML/QCNN) and B7 (stabilizer-prep → PP-estimate) are **design
stubs only** on this branch — see the plan §3/§6 and `research/notes/2026-09-01-python-api-extensions.md`
§A8 for why (blocked on a symbolic-coefficient core redesign, or on phase-2 stabilizer-membership
contraction).

## Running

```bash
./scripts/setup.sh                                            # one-time: creates .venv, builds the extension
source .venv/bin/activate
pip install -e ".[examples]"                                  # matplotlib, stim, qiskit, qiskit-aer, numpy
maturin develop --release -m crates/paulistrings-py/Cargo.toml # rebuild after any Rust change

RAYON_NUM_THREADS=1 python examples/b1_operator_scrambling/run_b1_1d.py
RAYON_NUM_THREADS=1 python examples/b5_operator_backpropagation/run_b5.py
RAYON_NUM_THREADS=1 python examples/b6_resource_probes/run_b6.py
RAYON_NUM_THREADS=1 python examples/xxz_chain/run_benchmark_d.py all
```

`RAYON_NUM_THREADS=1` **must be exported before the interpreter starts** — Rayon's global pool is
built once, at the first `propagate`/`propagate_with_stats` call, and is never resized, so setting
the variable from inside a script (after `import paulistrings`) does not reliably reach it. See
`common/harness.py`'s module docstring for the measured thread-count facts (and a correction to
§A7 of the API design note below).

Every CI-visible test (`python/paulistrings/tests/test_showcase_*.py`,
`test_benchmark_*.py`) `importorskip`s `stim`/`qiskit`/`matplotlib`, so
`pytest python/paulistrings/tests` stays green in the numpy-only CI job even without the
`examples` extra installed. The scripts under this directory are not collected by CI; run them
manually as above.

## Where things are documented

- Adapted execution plan (scope, interface mapping, capability register, decision log):
  [`research/plans/2026-08-31-examples-benchmarks-suite.md`](../research/plans/2026-08-31-examples-benchmarks-suite.md)
- Python API extensions the suite needed (`pauli_rotation`, `propagate_with_stats`, `from_arrays`,
  non-uniform product states, importers, Pauli-channel noise, harness aliases):
  [`research/notes/2026-09-01-python-api-extensions.md`](../research/notes/2026-09-01-python-api-extensions.md)
- Cross-engine (PauliPropagation.jl) baseline, schema, and measured semantics divergences:
  [`../benchmarks/julia/README.md`](../benchmarks/julia/README.md)
