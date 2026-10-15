# paulistrings

[![docs.rs](https://img.shields.io/docsrs/paulistrings)](https://docs.rs/paulistrings)
[![crates.io](https://img.shields.io/crates/v/paulistrings)](https://crates.io/crates/paulistrings)

<!-- Pitch paragraph is single-sourced with the repo-root README.md — keep the two word-identical. -->
Library for Pauli propagation built to scale across threads, processes and nodes.
Pauli strings are stored as symplectic bitvectors, and terms are assigned to workers by a hash function that is linear over the bitfield.
As a result, the library can efficiently parallelize the work and compute the communication patterns for a given circuit.
Additionally, for standard gate and noise sets, the amount of communication remains bounded and no global synchronization or reduction is required.

This crate is the Rust core of the [`paulistrings-rs`](https://github.com/lkdvos/paulistrings-rs) workspace.
Its Python package, `paulistrings-rs`, is the user interface and is documented on the [documentation site](https://lkdvos.github.io/paulistrings-rs/).

## Rust API

The crate root re-exports [`PauliSum`], [`propagate`] and [`propagate_with`], the built-in channels and truncation policies, and the [`Channel`] and [`TruncationPolicy`] extension traits.
The off-by-default features add per-phase timing (`phase-timing`), the MPI transport (`mpi`) and the CUDA backend (`cuda`).
The [API documentation](https://docs.rs/paulistrings) is a developer guide in which each item states the contracts it keeps.
[`CLAUDE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/CLAUDE.md) lists the build, test and benchmark commands.

## License

Dual-licensed under MIT or Apache-2.0, at your option.

[`PauliSum`]: https://docs.rs/paulistrings/latest/paulistrings/struct.PauliSum.html
[`propagate`]: https://docs.rs/paulistrings/latest/paulistrings/fn.propagate.html
[`propagate_with`]: https://docs.rs/paulistrings/latest/paulistrings/fn.propagate_with.html
[`Channel`]: https://docs.rs/paulistrings/latest/paulistrings/trait.Channel.html
[`TruncationPolicy`]: https://docs.rs/paulistrings/latest/paulistrings/trait.TruncationPolicy.html
