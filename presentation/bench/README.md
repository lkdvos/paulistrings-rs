# presentation-bench

Baselines and sweeps behind the figures of the talk. A separate Cargo workspace (excluded from the root
workspace) so the measurement features `phase-timing` and `test-utils` never unify into the shipped build.

Workload: 127-qubit heavy-hex kicked Ising, θzz = −π/2, θh = 5π/16, 5 Trotter steps (1355 rotations),
observable Z₆₂, Heisenberg, `CoefficientThreshold(2^-13)` → 1.16·10⁶ peak terms (`src/workload.rs`).

| variant | what it is |
|---|---|
| `naive` | the engine's own direct hash-map path (`engine/direct.rs`) with its size threshold removed: one map for the whole sum, single-threaded |
| `threadmaps` | reconstruction of "per-thread dictionaries merged at the end of every layer" (Rayon chunks → private maps → tree merge → truncate) |
| `mergesort` | reconstruction of "one flat array, parallel mergesort, merge equal keys" |
| `bucketed` | `paulistrings::propagate` with `PropagateOptions { target_bucket_len, min_buckets }` |

All variants use the real `Channel::apply_adjoint` and `TruncationPolicy::keep_term`; `tests/agreement.rs`
gates each against `propagate` to 1e-9.

```bash
cargo test --release --manifest-path presentation/bench/Cargo.toml
cargo build --release --manifest-path presentation/bench/Cargo.toml
presentation/bench/target/release/presentation-bench bucketed --threads 1,32 --reps 5
presentation/bench/scripts/collect_all.sh        # the campaign → presentation/data/*.jsonl
presentation/bench/scripts/ab_targetcpu.sh       # default vs -C target-cpu=native, paired
presentation/bench/scripts/perf_bucket_sweep.sh  # L2/LLC counters for the bucket sweep
```

Output: one JSON line per timed repetition (schema in `src/main.rs`), preceded per cell by the
`cell layer=… n=… layers=… wall_ms=…` line that `scripts/perf-stat.sh` greps.
