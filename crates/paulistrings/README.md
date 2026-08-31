# paulistrings

[![docs.rs](https://img.shields.io/docsrs/paulistrings)](https://docs.rs/paulistrings)
[![crates.io](https://img.shields.io/crates/v/paulistrings)](https://crates.io/crates/paulistrings)

<!-- Pitch paragraph is single-sourced with the repo-root README.md — keep the two word-identical. -->
Classical simulation of quantum circuits by Pauli propagation — evolving
operators in the Pauli basis under gates and noise channels, in either
the forward or Heisenberg picture. Aimed at workloads where state-vector
or tensor-network simulators are infeasible (10⁶–10⁸ terms) but the
operator stays sparse in the Pauli basis.

This crate is the pure-Rust core. Python bindings live in the
[`paulistrings-rs`](https://github.com/lkdvos/paulistrings-rs) workspace.

## Quickstart

```rust
use paulistrings::{
    channel::Clifford1Q, BuildAccumulator, Circuit, Direction, PauliString, Phase,
    TruncationPolicy, propagate,
};
use num_complex::Complex64;

// Build the observable Z_0 + 0.5 * X_1 on 2 qubits.
let mut acc = BuildAccumulator::<1>::new(2);
acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
acc.add_term(PauliString::<1>::x(1), Phase::ONE, Complex64::new(0.5, 0.0));
let observable = acc.finalize();

// Heisenberg-evolve through a one-gate circuit: H on qubit 0.
let mut circuit = Circuit::<1>::new(2);
circuit.push(Clifford1Q::h(0));

struct KeepAll;
impl<const W: usize> TruncationPolicy<W> for KeepAll {}

let evolved = propagate(&circuit, observable, &KeepAll, Direction::Heisenberg);
assert_eq!(evolved.len(), 2); // H · Z_0 · H = X_0, plus the unchanged 0.5*X_1
```

## Design

Four pillars in priority order:

1. **Correctness** of the Pauli algebra — symplectic encoding, phase
   tracking, the dedup invariant restored after every layer.
2. **Performance** at 10⁶–10⁸ terms — SoA layout, GF(2)-bucketed
   write-disjoint layers, Rayon-parallel with no global sort.
3. **Extensibility** for research — open [`Channel`] and
   [`TruncationPolicy`] traits.
4. **GPU-readiness** — `#[repr(C)]` `Pod` types, fixed-fanout buffers,
   shared-nothing parallelism that maps onto CUB primitives without
   restructuring.

See the [API documentation](https://docs.rs/paulistrings) for module-level
guides on the bucketed engine, width monomorphization, and the extension
traits. For the system-level design, see
[`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md).

## License

Dual-licensed under MIT or Apache-2.0, at your option.

[`Channel`]: https://docs.rs/paulistrings/latest/paulistrings/channel/trait.Channel.html
[`TruncationPolicy`]: https://docs.rs/paulistrings/latest/paulistrings/truncation/trait.TruncationPolicy.html
