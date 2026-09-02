# Julia baseline: amplitude_damping was transposed relative to the unitary channels

This note records a real bug the PauliPropagation.jl cross-engine baseline caught: `AmplitudeDamping`
had `apply`/`apply_adjoint` swapped relative to the convention every other channel follows. Moved from
`benchmarks/julia/README.md` on 2026-09-01, where it lived under a "RESOLVED" section; the README now
keeps only a one-line current-state statement (amplitude damping semantics agree between engines, and
the parity gate covers it) and points here for the history.

## RESOLVED: `amplitude_damping` was transposed relative to the unitary channels

**Fixed in the core; this section is kept as the record of a real bug this baseline caught.** Until
the fix, `AmplitudeDamping::apply` and `::apply_adjoint` in `channel/noise.rs` were swapped relative
to the convention every other channel follows, so `direction="heisenberg"` applied the Schrödinger
channel `Φ` instead of its dual `Φ†`.

What was measured at the time (γ = 0.3, single `amplitude_damping` gate, 8-term 3-qubit observable):

| | map applied to the qubit's Pauli |
|---|---|
| jl, `heisenberg=true` | `I → I`, `X,Y → √(1-γ)·same`, `Z → (1-γ)Z + γI` |
| this engine, `direction="heisenberg"` (**before the fix**) | `I → I + γZ`, `X,Y → √(1-γ)·same`, `Z → (1-γ)Z` |
| this engine, `direction="forward"` (**before the fix**) | identical to jl's `heisenberg=true` |

Why that was an inconsistency, not a choice:

* For unitary channels, `Channel::apply` in this core is the **Schrödinger** conjugation `U P U†` —
  `channel/clifford.rs` documents `S: X → Y`, i.e. `S X S†` (and `S† X S = -Y`, which is what
  `direction="heisenberg"` produces; probes.jl §P2 confirms jl agrees). So for unitaries
  `apply_adjoint` — what `direction="heisenberg"` calls — is the Heisenberg dual.
* `AmplitudeDamping` had it the other way round: `apply` was the **Heisenberg dual** `Φ†`
  (`I → I`, `Z → (1-γ)Z + γI`) and `apply_adjoint` was `Φ` itself (`I → I + γZ`).
* The Heisenberg dual of a trace-preserving channel is necessarily **unital** (`Φ†(I) = I`, because
  `Φ` preserves trace), so a Heisenberg map sending `I → I + γZ` cannot be a dual at all. Physically:
  `⟨Z⟩` for a qubit already in `|0⟩` — the fixed point of amplitude damping — decayed to `1-γ`
  instead of staying at `1`.
* `Depolarizing`, `Dephasing`, `PauliChannel` and `Depolarizing2Q` are self-adjoint, so the swap was
  invisible for them. `AmplitudeDamping` is the only built-in that exposes it.

The fix swapped the two bodies, so `apply` is now `Φ` (`I → I + γZ`, `Z → (1-γ)Z`) and
`apply_adjoint` is `Φ†` (`I → I`, `Z → (1-γ)Z + γI`), with the Kraus derivation written out in
`channel/noise.rs`. Measured after the fix, same fixture:

| direction | terms (rust / jl) | max coefficient \|Δ\| vs jl `heisenberg=true` |
|---|---|---|
| heisenberg | 9 / 9, labels identical | **0** (bit-exact on all 9) |
| forward | 11 / — | the transpose: no `III`, plus `ZXI ZZI ZIY` from the non-unital `I → I + γZ` |

`test_amplitude_damping_heisenberg_is_the_unital_dual` now pins the fixed orientation from both
sides, and `amplitude_damping` is back in `VOCAB_CASES` (the term-by-term sweep in
`benchmarks/julia/README.md`'s parity gate result). `direction="forward"` still cannot be compared
against jl — see that file's "Known gaps".
