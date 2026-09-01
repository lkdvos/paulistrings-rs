# Benchmark A — Clifford point

<p class="lead">The one point in the suite where the exact answer is an integer.
At a Clifford kick angle a single Pauli string evolves into a single Pauli string
with coefficient ±1, so <code>stim</code> can be asked for the truth and the
engine has nowhere to hide.</p>

## Setup

127-qubit heavy-hex kicked Ising, 5 Trotter steps, `θ_h = π/2` — the Clifford
point of the utility experiment — timing the Heisenberg propagation of the
published weight-10 and weight-17 observables against `|0…0⟩`. One
`pytest-benchmark` group per observable. `min_abs_coeff = 1e-8`, single-threaded,
seeded fixtures built outside the timed region.

**Why the cutoff is `1e-8` and not a dyadic.** This engine drops `|c| <= eps`
while `PauliPropagation.jl` keeps `|c| == eps` — a genuine, measured
[cross-engine divergence](../comparisons.md#the-one-real-divergence). Clifford
angles produce exact dyadic coefficients (`sin(π/2) == 1.0`, and `cos(π/2)` is
the tiny residual), so a dyadic cutoff is exactly where that boundary is likely
to be hit bit-for-bit. `1e-8` is far from any dyadic value and nine orders of
magnitude above the `~6.1e-17` residual being truncated away, so it changes
nothing about which branch survives.

## Oracle

`stim`, at the Clifford point, exact: the tableau simulation gives the ±1
integer directly. Every `paulistrings` entry also asserts the Clifford invariant
on its own result — a single term with coefficient exactly ±1.0 — so a
correctness regression fails the *benchmark* run too, not only the dedicated
test file.

## Result

The timing numbers for this benchmark live in
`benchmarks/results/bench_a.json`, which is **gitignored** — the repository
commits no timing table for A, so this page quotes none. What *is* committed, and
what this benchmark actually establishes, is correctness and parity:

- **The exact integers are reproduced bit-exactly.** Benchmark B scores the same
  Clifford endpoints across a full eight-point coefficient sweep and finds a
  worst deviation of **0** for every observable at both endpoints: `+1` for
  weight-10 and `−1` for weight-17 at `θ_h = π/2`, matching A. See
  [Benchmark B §3.2](b-theta-sweep.md#clifford-endpoints).
- **Cross-engine parity is a precondition, not an afterthought.** Before either
  engine's timed entry is allowed to run, both are run once, untimed, and every
  one of the **1355 per-layer term counts** must be identical. The schema-v1 task
  JSON handed to `PauliPropagation.jl` is built from the *same* recorded gate
  list the `paulistrings` side runs, so neither engine gets a transcription of
  the other's circuit.

Regenerate the timings with:

```bash
RAYON_NUM_THREADS=1 pytest benchmarks/python/bench_a_clifford.py \
    --benchmark-only --benchmark-json=benchmarks/results/bench_a.json
```

## The correctness gate

The CI-safe half lives in `python/paulistrings/tests/test_benchmark_a_clifford.py`
and `importorskip`s `stim`, so the numpy-only CI job stays green without it:

```bash
pytest python/paulistrings/tests/test_benchmark_a_clifford.py
```

**Sources:**
[`benchmarks/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/README.md)
§3, the driver's own module docstring in
[`benchmarks/python/bench_a_clifford.py`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/python/bench_a_clifford.py),
and — for the reproduced integers — 
[`benchmarks/python/theta_sweep/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/python/theta_sweep/README.md)
§3.2.
