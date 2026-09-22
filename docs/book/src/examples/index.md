# Examples

## The flagship worked example: a 2D Ising quench {#the-2d-ising-quench}

![Average X magnetization vs time for the 2D Ising quench, 4×4 and 6×6 lattices](../assets/ising-quench/ising_quench.svg)

The site's landing figure is a transverse-field Ising quench on a periodic 4×4 and 6×6 lattice at `J = h = 1`, first-order Trotterized at `δt = 0.05` out to `t = 2`.
The average X magnetization is Heisenberg-propagated through the 40-step circuit and read against `|+…+⟩` at every step, under `coeff(1e-10) & topn(k)` with `k` of 50 000 (4×4) and 200 000 (6×6).
The 6×6 case is `2³⁶` amplitudes, already out of reach for exact diagonalization; the quench runs in ~11–12 s on the reference host.

The honest error bar there is not floating point but the `topn` tie rule: `topn` keeps or drops a magnitude-tied symmetry orbit whole, and an alternative tiebreak that always keeps exactly `k` terms moves the trajectory by up to 1.7% (4×4) and 0.37% (6×6).
The larger lattice is the better-resolved one, because the observable averages over more sites and the truncation error self-averages.

The walkthrough is a Rust one — it is the crate's own worked example, embedded into its rustdoc — and is the hand-off point for Rust users rather than part of this book:
[`crates/paulistrings/docs/examples/ising_2d_quench.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/crates/paulistrings/docs/examples/ising_2d_quench.md), with the source at [`ising_2d_quench.rs`](https://github.com/lkdvos/paulistrings-rs/blob/main/crates/paulistrings/examples/ising_2d_quench.rs).
For the same pattern in Python — one Trotter step at a time, an expectation value per step — see [Incremental propagation](../manual/propagation/incremental.md).

## Contents

- [First propagation](first-propagation.md) — the site's guided walkthrough: build an observable, build a circuit, propagate, validate.
- [Showcases](showcases/index.md) — five measured applications, each checked against an independent oracle.
- [Benchmarks](benchmarks/index.md) — five benchmarks against `PauliPropagation.jl` and exact references, two with negative results.
- [Engine performance](benchmarks/engine-performance.md) — roofline analysis, layer-time breakdown, and partitioned/distributed scaling, all measured.
- [Against other tools](comparisons.md) — where this method fits against state-vector, stabilizer and MPO simulation.
