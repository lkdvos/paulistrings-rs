# Partitioned engine, phase 1 — decision log

Running log of gates and decisions while the partitioned (NUMA/multi-node) engine lands on branch
`partitioned-engine`. Plan: `~/.claude/plans/i-would-like-to-cheerful-mitten.md` (user-approved
2026-09-08). Numbers here come from `benchmarks/results/2026-09-08-ccqlin038/` (gitignored).

## 2026-09-08 — S7a `ExtraRows` hook on `fill_coset`: accepted

`scripts/ab-compare.sh s7a-extrarows --a e7de227 --b . --probe '--n 1000000 --qubits 128 --layers
rotation_zz,cnot,su4 --threads 1,16 --reps 8' --pairs 3 --order abba`, load 4–7, `RUST_LOG` unset.
The B side adds the zero-cost `ExtraRows` generic (`NoExtra` on the existing path), the
`apply_layer_bucketed_with` wrapper level, and `pub(super)` visibility; no merge-kernel change.

| cell | wall median Δ% | pairs | work counters (`rows_gathered/sorted/id`, `runs`, `cosets`) |
|---|---|---|---|
| rotation_zz 1t | +3.8 | 3/3 up | identical |
| rotation_zz 16t | +4.4 | 3/3 up | identical |
| cnot 1t | +1.7 | mixed | identical |
| cnot 16t | −17.9 | 3/3 down | identical |
| su4 1t | −3.0 | 3/3 down | identical |
| su4 16t | −18.8 | mixed | identical |

Verdict: **layout artifact, accepted.** Every work counter is bit-identical between sides, and the
consistent deltas have opposite signs on different layers (rotation up, su4/cnot down), inside the
calibrated ±4–7% LTO layout band (`research/notes/2026-09-01-large-m-campaign-log.md`). The fallback
(a textual `fill_coset_recv` copy) would perturb layout just the same. Re-check with the full
`--partitions 1` path after S9 lands, as the plan requires.

## 2026-09-08 — pre-existing flaky stack overflow in the debug test binary

`cargo test -p paulistrings` (debug, full parallelism) aborts with `fatal runtime error: stack
overflow` on an unnamed Rayon worker in roughly 1 run in 4; reproduced on the scaffold commit
`e7de227` (before any partitioned code) and never with `RUST_MIN_STACK=16777216`. Worked around by
`.cargo/config.toml` `[env] RUST_MIN_STACK = "16777216"`; root cause (leading hypothesis: Rayon's
adaptive split depth under stealing × debug frame sizes) deferred to the phase-4 cleanup.
