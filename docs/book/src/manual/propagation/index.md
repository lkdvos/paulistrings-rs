# Engine and propagation

Propagation is the one thing this library does: take a Pauli sum, push it through a circuit one channel at a time, and truncate after every channel so the sum stays tractable.
Everything else on this site — observables, circuits, measurements — exists to feed that loop or read its output.
This chapter is the loop itself: what `propagate` computes, which knobs change *what* it computes and which only change *how fast*, and how to tell whether the truncated result can be trusted.

## The propagate call

```python
from paulistrings import Circuit, PauliSum, truncation

observable = PauliSum.from_strings({"XIII": 0.25, "IXII": 0.25, "IIXI": 0.25, "IIIX": 0.25}, num_qubits=4)

circuit = Circuit(4)
circuit.rz(0.3, 0)
circuit.cnot(0, 1)

evolved = observable.propagate(circuit, truncation.coeff(1e-10), direction="heisenberg")
print(len(observable), "->", len(evolved), "terms")
```

```text
4 -> 5 terms
```

Three arguments carry the physics and one carries the cost.
`circuit` is the channel list to apply, and it must share `num_qubits` with the sum.
`direction` chooses between `U†OU` and `UOU†`, which are different operators and answer different questions — [Direction](direction.md).
`policy` is the truncation applied after every channel, and it is the only argument that trades accuracy for time — [Truncation](truncation.md).
`propagate` returns a **new** sum and leaves the input untouched, so the same observable can be propagated several ways and compared.
The GIL is released for the duration of the call.

`propagate_with_stats` takes the same arguments and returns `(evolved, stats)`, where `stats` is a `PropagationStats` with per-layer term counts and timings; enabling it does not change the evolved sum — [Stats, memory and logging](settings.md#stats).
The remaining keyword arguments (`engine`, `partitions`, `comm`, bucket sizing) change how the sum is stored and where the work runs, never the operator being computed, and are covered on the [settings](settings.md#engine-selection), [NUMA partitions](partitions.md) and [MPI ranks](mpi.md) pages.
The full signature is in the [Library](../../library/propagate.md).

## The loop {#the-loop}

A **layer** is one applied channel, not a brickwork layer of parallel gates — [One gate, one truncation point](../circuits.md#one-gate-one-channel) is the definition, and it is the sense the word carries throughout this book and in `PropagationStats`.

For each layer the engine does the same three things.
It fans every term out over the channel's outputs: a Clifford gate moves each term to exactly one new Pauli string, a Pauli rotation splits each term into two (the identity part and the generator part), a depolarizing or dephasing channel rescales coefficients in place, and a dense two-qubit unitary can split a term into up to 16 outputs.
It merges terms that landed on the same Pauli string, adding their coefficients.
Then it applies the truncation policy to the merged sum, and only what survives is fed to the next layer.

Truncation runs **after every layer**, not once per `propagate` call.
That single fact has consequences the rest of this chapter returns to: fusing two gates into one channel changes the answer, noise makes a fixed threshold cheaper to satisfy at every depth, and a circuit can be split into consecutive `propagate` calls with no change in the result.

The order in which the layers are applied depends on `direction`: forward walks the circuit as written, Heisenberg walks it in reverse and applies each channel's adjoint.
[Direction](direction.md#push-order) states that rule once and precisely; the [Incremental propagation](incremental.md#the-heisenberg-ordering-trap) page is where it bites.
How the engine makes each layer fast — the bucketed layout and the write-disjoint parallel decomposition — is on [Inside the engine](engine.md).

## Cost is terms, not qubits {#cost-model}

The cost of a propagation is not the qubit count.
It is the number of Pauli strings the operator spreads over, which grows with circuit depth and entangling strength until truncation holds it.
A local observable starts as a handful of terms and only spreads as far as its causal cone, so 127 qubits and a shallow circuit can be cheap while 12 qubits and a deep one are not.
The qubit count enters only through the number of channels a circuit contains and through the per-term storage width — [Estimate the memory a run needs](settings.md#estimate-the-memory-a-run-needs).

Term growth is the cost signal to watch.
`propagate_with_stats` reports it per layer in `terms_out`, and a short prefix of the circuit is usually enough to see whether the growth is saturating or still exponential.
What every truncated result then has to report is how much of the operator the truncation deleted, and whether the answer still moves when the cutoff is tightened — [Validating a result](validation.md).

## Always pass direction

The Python binding defaults `direction=None` to `"forward"`, but the two pictures compute different operators, and most examples on this site use `"heisenberg"`.
A result computed in the wrong picture is a plausible number with no error raised, so treat the default as a one-off exploration convenience and pass `direction` explicitly in anything you keep — [Direction](direction.md#which-state-the-label-is-read-against) has the two numbers side by side.

## In this chapter

- [Direction](direction.md) — `U†OU` versus `UOU†`, which state the expectation label then denotes, and push order.
- [Truncation](truncation.md) — why the policy is the accuracy knob, which policy to reach for, and why a truncated sum has no variational bound.
- [Validating a result](validation.md) — the cutoff sweep, the retained norm, the plateau criterion, and exact references.
- [Incremental propagation](incremental.md) — splitting a circuit is free, which makes a time series one pass, with the Heisenberg-ordering trap.
- [Stats, memory and logging](settings.md) — `PropagationStats`, the bytes-per-term arithmetic, logging, thread count and engine selection.
- [Inside the engine](engine.md) — the bucketed layout, the GF(2) hash, and write-disjoint cosets.
- [NUMA partitions](partitions.md) — one pinned pool and one share of the sum per NUMA domain.
- [MPI ranks](mpi.md) — the same split across processes, for capacity.

[First propagation](../../examples/first-propagation.md) runs this whole chapter once on four qubits; [Benchmark D](../../examples/benchmarks/d-xxz-chain.md) is the cost model measured, with term growth against depth and chain length.
