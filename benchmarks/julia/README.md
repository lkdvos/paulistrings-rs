# PauliPropagation.jl baseline

Out-of-CI, subprocess-driven baseline for cross-engine comparisons against **PauliPropagation.jl**.
Nothing here is imported by the Python package or by CI.

Install `julia` (found on `PATH`, at `$JULIA_BINARY`, at `~/.juliaup/bin/julia`, or via `module load julia`).
The environment here pins PauliPropagation.jl 0.8.2, JSON3 1.14.3, BenchmarkTools 1.8.0, resolved for julia 1.12.6.
No extra install step is needed beyond `julia`; `--project=benchmarks/julia` resolves the pinned packages from `Manifest.toml` on first run.

```bash
julia --project=benchmarks/julia benchmarks/julia/runner.jl task.json          # -> stdout JSON
julia --project=benchmarks/julia benchmarks/julia/probes.jl                    # semantics probes
python benchmarks/python/julia_baseline.py --self-test                        # wrapper smoke
pytest benchmarks/python/test_julia_parity.py -q                              # parity gate
```

`runner.jl` reads a task JSON (schema v1) and emits one result JSON line on stdout (or to `-o path`); diagnostics go to stderr.
`../python/julia_baseline.py` is the `subprocess` wrapper other benchmarks call: it builds/validates the task JSON, invokes the runner, parses the result, and skips cleanly with no `julia` on `PATH`.
`probes.jl` prints the semantics-probe table (qubit indexing, Hermitian-Y convention, truncation-boundary agreement) to stdout; each probe's expected value is hand-derived in a comment in that file.
The first Julia run precompiles (~30 s at first use, then cached).
