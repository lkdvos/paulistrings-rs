# B7 — Stabilizer-state preparation

<p class="lead">stim prepares a 36-qubit 2D cluster state as a stabilizer
tableau; a non-Clifford tail is propagated in the Heisenberg picture; the
expectation is contracted against the stabilizer state at
<code>O(m·n²/64)</code> cost, never building the 1.0 TiB state vector the
same check would need as a dense array.</p>

## The idea

stim runs a Clifford circuit of any depth in polynomial time and cannot touch
a non-Clifford gate; Pauli propagation eats non-Clifford rotations for a
living and gains nothing from re-deriving a Clifford circuit's output.
Chaining them across the observable gets both:

```text
|0...0> --[ deep Clifford prep U_C ]--> |psi>   --[ non-Clifford tail U_T ]--> measure O
```

In the Heisenberg picture the tail acts on the observable and the state is
only needed at the very end, `⟨ψ|U_T† O U_T|ψ⟩` with `|ψ⟩ = U_C|0...0⟩`. A
stabilizer state is fully described by its `n` signed Pauli generators
`s_i·G_i`, and for a Hermitian Pauli string `P`

```text
⟨ψ|P|ψ⟩ = σ    if σ·P is in the group ⟨s_i G_i⟩ for some σ = ±1
        = 0    otherwise
```

— a group-membership test, `O(n²/64)` word operations per term after an
`O(n³/64)` GF(2) echelon reduction of the generators. Contracting the evolved
`m`-term sum is `O(m·n²/64)`, with no `2^n` object ever built. At `n = 36` a
state vector would hold 6.9·10¹⁰ amplitudes, 1.0 TiB.

Three API calls carry the whole pipeline: `interop.stabilizers_from_stim(...)`
turns a stim tableau into signed generator strings,
`PauliSum.propagate(tail, policy, direction="heisenberg")` evolves the
observable, and `PauliSum.expectation_stabilizer(generators)` contracts it.

## Results

The setup is an open 6×6 square lattice, 36 qubits. Preparation is the 2D
cluster state — `H` on every qubit then one `CZ` per edge, 96 Clifford gates —
a universal resource state that is not a product state in any local basis.
`stabilizers_from_stim` reads the 36 signed generators out of the tableau in
under a millisecond. The observables are two sites' own cluster stabilizers,
`K_q = X_q ∏_{q'∈N(q)} Z_{q'}`, at the lattice centre (degree 4) and a corner
(degree 2); both read exactly `+1` before the tail runs. The tail is two
kicked-Ising Trotter steps (`rx(θ_h)` then `ZZ(-π/2)` per step), with `θ_h`
swept from `0` to `π/2`:

| `θ_h` | terms kept | `⟨K_centre⟩` | `⟨K_corner⟩` | closed-form gap |
|---:|---:|---:|---:|---:|
| 0.000 | 1 | +1.000000000000 | +1.000000000000 | 0 |
| 0.393 | 16 | +0.728553390593 | +0.853553390593 | 0 |
| 0.785 | 16 | +0.250000000000 | +0.500000000000 | 0 |
| 1.178 | 16 | +0.021446609407 | +0.146446609407 | 3.5e−18 |
| 1.571 | 1 | +0.000000000000 | +0.000000000000 | 3.8e−33 |

(all 17 sweep points: `theta_sweep.csv`)

![Expectation of the two cluster stabilizers against the kick angle, with the closed-form curve overlaid](../assets/b7/theta_sweep.svg)

Preparation depth is free on the Heisenberg side: it never sees the state, so
one evolved observable is contracted against the same cluster state prepared
through four circuits of growing depth (96 to 50 236 Clifford gates, via
provably-identity padding rounds). The generator strings come back
byte-identical at every depth and the estimate is unchanged to all twelve
printed digits — a 523× longer preparation costs 2.1 ms instead of 0.5 ms on
the stim side and nothing on the propagation side, because it is literally the
same evolved sum contracted four times (`prep_depth.csv`).

### Scope of the check

Preparation-depth invariance holds for *structured* preparations. A generic
stabilizer state's group holds `2^n` of the `4^n` Pauli strings, so a random
low-weight operator term lands in the group with probability `2^-n`
(1.5·10⁻¹¹ at `n = 36`) — an unstructured deep Clifford preparation drives the
estimate to exactly zero, by membership, not by truncation. The pipeline is
not restricted to graph states, but the observable and the state need group
elements in common; for a structured resource state (cluster, surface-code,
GHZ) that is the normal case, and such a state's own generators remain
estimable regardless of preparation depth.

### The closed-form check

Every point in the sweep equals `cos^deg(q)(θ_h)` to 1.1e−16, derived rather
than fit: pushing `K_q` through the two-step circuit `U = ZZ₂X₂ZZ₁X₁`, the
first `ZZ` layer is Clifford at `θ_zz = -π/2` and turns `K_q` into `±X_q`
(weight one) whenever `deg(q)` is even; `X_q` commutes with the `X` rotations;
the second `ZZ` layer reverses the first; the second `X` layer splits each
`Z_{q'}` factor into a `cos`/`sin` pair, and only the all-`cos` branch survives
contraction because every other branch carries a `Y` and contracts to zero.
Odd-degree sites pick up one extra `cos` factor and a sign and are excluded
from the closed form, not from the pipeline.

## Performance

Contraction cost was measured directly against the `O(m·n²/64)` bound, on a
1D-chain cluster state at each `n`, 20 000 random full-support terms, setup
cost subtracted:

| `n` | per term (µs) | `n²/64` | ns per word-op |
|---:|---:|---:|---:|
| 64 | 0.398 | 64 | 6.22 |
| 256 | 2.617 | 1 024 | 2.56 |
| 1024 | 33.845 | 16 384 | 2.07 |

Local slope `d log(per-term time) / d log n` climbs from 1.22 (64→128) to
**1.85** (512→1024) against the bound's 2.0 — below `n ≈ 256` a per-term fixed
cost (pivot scan, term decode) dilutes the `n²/64` row work, and the last
decade sits within 8% of the bound. Per-word-operation cost falls from 6.2 ns
to 2.1 ns and flattens near the cost of a load, an xor pair and a parity.
Linearity in `m` at `n = 256` holds flat to 7% over two decades, from 1 000 to
100 000 terms (`scaling.csv`).

![Per-term contraction time against n, and linearity against m](../assets/b7/scaling.svg)

The full run takes 116 s and 10.2 GiB peak RSS, single-threaded
(`RAYON_NUM_THREADS=1`), the minimum of five repeats after warm-up. Every
timing here is one campaign's numbers on a shared workstation; term counts and
expectation values reproduce to the digit, milliseconds do not — read the
exponents and the flatness, not the last digit.

## Validation

Three independent routes agree: a dense `2^n` state-vector reference built
straight off the `Circuit` object the engine propagates, a projector
`Π = ∏_i(I + s_i G_i)/2` built from the generator strings alone at `n = 8`,
and qiskit Aer where installed. Worst gap over every case and route:
**2.220e−15**, against a `1e-10` bar; qiskit Aer's worst gap is 1.55e-15.

A truncation-convergence panel (tail depths 4 and 5, sweeping
`min_abs_coeff`) shows depth 4 self-converged inside the grid — the last two
points agree to all twelve digits — while depth 5 is still moving by 5.2e-05
between its last two points, reported as the residual it is. The estimate
converges faster than the sum: between `1e-4` and `1e-6` at depth 4 the term
count grows 1.4× while the answer moves 2.3e-5, because only terms landing in
the stabilizer group contribute at all.

![Truncation convergence at tail depths 4 and 5](../assets/b7/convergence_panel.svg)

## Reproducing

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python examples/b7_stabilizer_prep/run_b7.py          # 116 s, 10.2 GiB peak RSS
RAYON_NUM_THREADS=1 python examples/b7_stabilizer_prep/run_b7.py --quick  #  40 s,  1.5 GiB peak RSS
pytest python/paulistrings/tests/test_showcase_b7.py                        # 0.5 s
```

`RAYON_NUM_THREADS=1` must be exported before the interpreter starts; the
script asserts the pin and refuses to run unpinned. A full run regenerates
every artifact in the showcase directory.

**Numbers:** every value on this page is sourced from
[`examples/b7_stabilizer_prep/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/b7_stabilizer_prep/README.md)
and its committed artifacts — `theta_sweep.csv`, `scaling.csv`,
`prep_depth.csv`, `validation_b7.json`, `results_b7.json`.
