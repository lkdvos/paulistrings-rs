# paulistrings

[![docs.rs](https://img.shields.io/docsrs/paulistrings)](https://docs.rs/paulistrings)
[![crates.io](https://img.shields.io/crates/v/paulistrings)](https://crates.io/crates/paulistrings)

<!-- Pitch paragraph is single-sourced with the repo-root README.md — keep the two word-identical. -->
Library for Pauli propagation built to scale across threads, processes and nodes.
Pauli strings are stored as symplectic bitvectors, and terms are assigned to workers by a hash function that is linear over the bitfield.
As a result, the library can efficiently parallelize the work and compute the communication patterns for a given circuit.
Additionally, for standard gate and noise sets, the amount of communication remains bounded and no global synchronization or reduction is required.

This crate is the pure-Rust core. Python bindings live in the
[`paulistrings-rs`](https://github.com/lkdvos/paulistrings-rs) workspace.

## Building from source and developing

Python is the user interface: install `paulistrings-rs` and read the [documentation site](https://lkdvos.github.io/paulistrings-rs/).
This crate's rustdoc is a developer guide to the engine rather than a user API.
[`CLAUDE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/CLAUDE.md) lists the build, test and benchmark commands, and [`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md) is the design source of truth.

## Design

Four pillars in priority order:

1. **Correctness** of the Pauli algebra — symplectic encoding, phase
   tracking, the dedup invariant restored after every layer.
2. **Performance** at 10⁶–10⁸ terms — SoA layout, GF(2)-bucketed
   write-disjoint layers, Rayon-parallel with no global sort.
3. **Extensibility** for research — open [`Channel`] and
   [`TruncationPolicy`] traits.
4. **GPU-readiness** — structure-of-arrays columns, fixed-fanout buffers,
   shared-nothing parallelism that maps onto CUB primitives without
   restructuring.

See the [API documentation](https://docs.rs/paulistrings) for module-level
guides on the bucketed engine, width monomorphization, and the extension
traits. For the system-level design, see
[`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md).

## License

Dual-licensed under MIT or Apache-2.0, at your option.

[`Channel`]: https://docs.rs/paulistrings/latest/paulistrings/trait.Channel.html
[`TruncationPolicy`]: https://docs.rs/paulistrings/latest/paulistrings/trait.TruncationPolicy.html
