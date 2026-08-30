A worked end-to-end simulation: classical computation of the average X
magnetization following a sudden quench in a 2D transverse-field Ising
model. The lattice is small enough to fit in a single 64-bit Pauli word
(`W = 1`) but large enough that exact diagonalization is already painful
(`2^36` ≈ 7 × 10¹⁰ amplitudes for the 6×6 case). Pauli propagation with
modest truncation gets the answer in seconds-to-minutes.

# Setup

Hamiltonian, on an `Lx × Ly` lattice with periodic boundary conditions:

```text
H = -J · Σ_⟨i,j⟩ Z_i Z_j  -  h · Σ_i X_i
```

with `J = h = 1` (the model is near its 2D quantum critical point — the
exact critical ratio is `h/J ≈ 3.044` in 2D, but `J = h` is a convenient
non-trivial point). The initial state is `|+⟩^⊗N`, a `+1` eigenstate of
every single-qubit `X`, and the observable is the average X magnetization:

```text
O = (1/N) · Σ_i X_i
```

The Trotter decomposition is first-order with step `δt = 0.05`:

```text
U(δt) ≈ exp(-i·δt·h·Σ X_i) · exp(-i·δt·J·Σ Z_i Z_j)
```

Evolve to `t_max = 2.0 / J` over 40 steps.

# Strategy

We Heisenberg-evolve the **observable** `O(t) = U†(t) · O · U(t)` rather
than the state, then read off `⟨+...+| O(t) |+...+⟩` from the resulting
Pauli sum. This is the natural workload for Pauli propagation: the
observable starts as `N` weight-1 terms and grows under the Trotter
circuit; the wave function is never stored.

The four ingredients we need are: (1) the per-step `Circuit`, (2) the
initial observable as a `PauliSum`, (3) the propagation loop, and
(4) the observable-to-scalar projection.

## Per-step circuit

`PauliRotation` represents `exp(-i·θ·P/2)`, so the angle for a single
`Z_i Z_j` bond at coupling `J` and time step `dt` is `θ = 2 · J · dt`,
and likewise `θ = 2 · h · dt` for an `X_i` site. The generator for a
`Z_i Z_j` bond is built by multiplying single-qubit `Z`s — the engine
returns the right `x` / `z` bitmasks plus an `i^k` phase (which is
`+1` for two distinct Pauli `Z`s, but we go through `mul_assign` so the
pattern generalises to mixed generators).

```rust,no_run
use paulistrings::{Circuit, PauliString};
use paulistrings::channel::PauliRotation;

fn qubit_index(x: usize, y: usize, lx: usize) -> u32 {
    (y * lx + x) as u32
}

/// One Trotter step `U(dt) = exp(-i·dt·h·ΣX) · exp(-i·dt·J·ΣZZ)`.
fn trotter_step(lx: usize, ly: usize, dt: f64, j_coupling: f64, h: f64) -> Circuit<1> {
    let n = lx * ly;
    assert!(n <= 64, "PauliString<1> covers up to 64 qubits");
    let mut circuit = Circuit::<1>::new(n);

    // ZZ rotations on every nearest-neighbour bond (PBC).
    for y in 0..ly {
        for x in 0..lx {
            let i = qubit_index(x, y, lx);
            let right = qubit_index((x + 1) % lx, y, lx);
            let down = qubit_index(x, (y + 1) % ly, lx);
            for partner in [right, down] {
                let mut gen = PauliString::<1>::z(i);
                gen.mul_assign(&PauliString::<1>::z(partner));
                circuit.push(PauliRotation::new(gen, 2.0 * j_coupling * dt));
            }
        }
    }

    // Transverse-field X rotations on every site.
    for site in 0..n as u32 {
        let gen = PauliString::<1>::x(site);
        circuit.push(PauliRotation::new(gen, 2.0 * h * dt));
    }

    circuit
}
```

Push order matters under `Direction::Heisenberg`: the engine iterates
channels in reverse and applies each one's adjoint. With ZZ pushed first
and X pushed last, the per-step state-evolution operator is
`U_X · U_ZZ`, and Heisenberg evolution of `O` correctly computes
`U_ZZ† · U_X† · O · U_X · U_ZZ`.

## Initial observable

`(1/N) · Σ_i X_i` is `N` weight-1 Pauli terms with equal coefficient.
`BuildAccumulator` is the standard ingestion path: insert terms one at
a time, call `.finalize()` to get a sorted-and-deduplicated `PauliSum`.

```rust,no_run
use paulistrings::{BuildAccumulator, PauliString, PauliSum, Phase};
use num_complex::Complex64;

fn x_magnetization(lx: usize, ly: usize) -> PauliSum<1> {
    let n = lx * ly;
    let inv_n = Complex64::new(1.0 / n as f64, 0.0);
    let mut acc = BuildAccumulator::<1>::new(n);
    for site in 0..n as u32 {
        acc.add_term(PauliString::<1>::x(site), Phase::ONE, inv_n);
    }
    acc.finalize()
}
```

## Reading the result

For a single-qubit Pauli `P`, `⟨+|P|+⟩` equals `1` when `P ∈ {I, X}` and
`0` when `P ∈ {Y, Z}`. By tensoring,

```text
⟨+...+| P_total |+...+⟩  =  ∏_i ⟨+|P_i|+⟩
                         =  1[z-part of P_total is all zero]
```

So the expectation value is the sum of `Re(coeff)` over `PauliSum` terms
whose `z`-bitmask is zero — a linear-time pass over the output sum. That is
exactly [`ProductState::XPlus`](crate::ProductState), so it is one call:

```rust,no_run
use paulistrings::{PauliSum, ProductState};

fn expectation_plus_state(sum: &PauliSum<1>) -> f64 {
    // The observable is Hermitian, so the imaginary part is zero.
    sum.expectation_product_state(ProductState::XPlus).re
}
```

The other two uniform product states work the same way:
[`ZPlus`](crate::ProductState::ZPlus) for `|0…0⟩` (keep terms with no `x` bits)
and [`YPlus`](crate::ProductState::YPlus) for `|+i…+i⟩` (keep terms whose `x` and
`z` words agree, i.e. every factor is `I` or `Y`). For an observable against a
state that is itself a Pauli sum, use
[`PauliSum::overlap`](crate::PauliSum::overlap) instead.

Until v0.2 B.10 this had to be hand-rolled against the raw SoA columns, because
the crate exposed no expectation-value API at all.

## Putting it together

The propagation entry point is [`propagate`](crate::propagate) with
[`Direction::Heisenberg`](crate::Direction::Heisenberg). We build the
per-step circuit once, then call `propagate` 40 times, recording the
expectation after each step.

```rust,no_run
use paulistrings::{propagate, Circuit, Direction, PauliSum};
use paulistrings::truncation::{And, CoefficientThreshold, TopN};

# fn trotter_step(_lx: usize, _ly: usize, _dt: f64, _j: f64, _h: f64) -> Circuit<1> { todo!() }
# fn x_magnetization(_lx: usize, _ly: usize) -> PauliSum<1> { todo!() }
# fn expectation_plus_state(_sum: &PauliSum<1>) -> f64 { todo!() }
let (lx, ly) = (4, 4);
let (j_coupling, h) = (1.0, 1.0);
let dt = 0.05;
let steps = 40;

let trotter = trotter_step(lx, ly, dt, j_coupling, h);
let policy = And(CoefficientThreshold(1e-10), TopN(50_000));

let mut observable = x_magnetization(lx, ly);
let mut series = vec![(0.0, expectation_plus_state(&observable))];

for k in 1..=steps {
    observable = propagate(&trotter, observable, &policy, Direction::Heisenberg);
    series.push((k as f64 * dt, expectation_plus_state(&observable)));
}
```

# Truncation

The 4×4 lattice grows from `N = 16` weight-1 terms at `t = 0` to ~10⁴–10⁵
terms within a handful of Trotter steps. The 6×6 grows much faster. Two
policies, composed with [`And`](crate::truncation::And), keep this
tractable:

| Lattice | `CoefficientThreshold` | `TopN`     |
|---------|------------------------|------------|
| 4×4     | `1e-10`                | `50_000`   |
| 6×6     | `1e-10`                | `200_000`  |

The first drops floating-point noise (and exact cancellations); the second
caps the sum at no more than `n` terms (v0.3: it can end up holding fewer —
see below). Loosening either preserves the curves at early times
and only changes long-time behaviour (where the answer is already
truncation-limited).

### How sensitive is the answer to *which* terms `TopN` keeps?

More than one might expect, and worth knowing before reading precision into
these curves. `TopN` has to cut through groups of terms with **exactly equal**
coefficient magnitude — unavoidable here, because lattice symmetry on a periodic
lattice relates many Pauli strings and gives them identical weights. Which
members of a tie group survive — and, as of v0.3, whether the tie group
survives at all — is a real design choice, not an implementation accident.

Three variants have existed:

- **(a) pre-v0.2, unspecified order.** Ties at the cutoff were broken by
  `select_nth_unstable_by`'s implementation-defined order — not reproducible
  across Rust versions or even runs.
- **(b) v0.2, key-ascending tiebreak.** Made the choice deterministic: within
  a tie group at the cutoff, keep the `n` smallest-key terms. This was the
  reference used for every CSV committed before v0.3. It always keeps
  exactly `n` terms (once the sum has at least `n`).
- **(c) v0.3, whole-group discard.** `TopN(n)` now keeps *at most* `n`
  terms: if the tie group straddling the cutoff would fit entirely within
  `n`, it is kept whole; if it would not fit, the *entire* group is
  discarded rather than split. An all-tied sum is wiped to empty. The
  retained count is therefore `≤ n`, and can be strictly less than `n`
  whenever the boundary group doesn't fit.

Each transition moves the trajectory by:

| Lattice | (a) → (b) max Δ⟨X_avg⟩ | (a) → (b) max relative | (b) → (c) max Δ⟨X_avg⟩ | (b) → (c) max relative |
|---------|------------------------|-------------------------|------------------------|-------------------------|
| 4×4     | 0.0053                 | 1.8%                    | 0.00492                | 1.69%                   |
| 6×6     | 0.00037                | 0.12%                   | 0.00120                | 0.37%                   |

(final `⟨X_avg⟩` at `t = 2.0`, (b) → (c): 4×4 `0.304114 → 0.308620`, 6×6
`0.323626 → 0.322427`.)

Why (c) is the choice going forward: whole-group discard is the physically
defensible rule. Truncation should commute with the lattice's symmetry group
— a tie group here *is* a symmetry orbit, so keeping part of one and
discarding the rest breaks a symmetry the exact operator has, while
discarding the whole orbit does not. The tradeoff is that `TopN(n)` no
longer guarantees exactly `n` terms: retention is now "at most `n`," and can
land measurably below `n` when the boundary group is large relative to what's
left of the budget — which is also why the 6×6 quench got faster under (c)
(see below): a smaller retained sum every layer is less work every layer.

Notice the 6×6 sensitivity from (b) to (c) (0.00120, 0.37%) is *larger* than
the 6×6 sensitivity from (a) to (b) (0.00037, 0.12%) — about 3× as large.
Going from "some deterministic tiebreak" to "no tiebreak, discard the
group" moved the answer more than going from "no determinism at all" to
"some deterministic tiebreak" did. Tie handling matters more than the (a)→(b)
numbers alone suggested.

So the 4×4 curve is good to a couple percent at `TopN(50_000)`, not more,
and the 6×6 remains better resolved than the 4×4 despite being the larger
problem — the observable is an average over more sites, so the truncation
error self-averages. Treat the spread across truncation-tiebreak choices as
the honest error bar; it is larger than any floating-point effect.

Wall-clock note: under the v0.3 rule the 6×6 run also got noticeably
faster — about 11–12 s versus roughly 25–28 s under (b) — because whole-group
discard keeps the propagated sum strictly below the `TopN` cap whenever a
boundary group straddles it, so later layers do less work. The 4×4 run stays
under 2 s in both cases (its sum rarely approaches `TopN(50_000)`).

# Result

![Average X magnetization vs time for the 2D Ising quench, 4×4 and 6×6 lattices](https://raw.githubusercontent.com/lkdvos/paulistrings-rs/main/crates/paulistrings/docs/examples/img/ising_quench.svg)

`⟨X_avg⟩` starts at `+1.0` (every spin in the `|+⟩` eigenstate of `X`)
and decays as the `ZZ` coupling mixes `X` with `Y` and `Z` components.
The two lattice sizes track each other closely — finite-size effects
are visible but small in this time window — consistent with the
short-time evolution being dominated by local dynamics.

# Reproduce

From the repo root:

```bash
cargo run --example ising_2d_quench --release
python crates/paulistrings/examples/plot_ising_quench.py
```

The first command writes
`crates/paulistrings/examples/output/ising_{4x4,6x6}.csv`. The second
reads those and writes
`crates/paulistrings/docs/examples/img/ising_quench.svg` — the file
embedded above. Both files are committed to the repo so docs.rs and the
GitHub Pages preview render the plot without re-running the simulation.

Full Rust source (with CSV output and per-step progress logging):
[`crates/paulistrings/examples/ising_2d_quench.rs`](https://github.com/lkdvos/paulistrings-rs/blob/main/crates/paulistrings/examples/ising_2d_quench.rs).
