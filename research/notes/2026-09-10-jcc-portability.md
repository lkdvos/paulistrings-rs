# JCC padding across node types — the flag is a tax off Skylake

Measured 2026-09-10 on `ccq`, jobs 7018021 (rome), 7018022 (genoa), 7018023 (icelake), all
COMPLETED 0:0. Commit `00c07d7`. `scripts/slurm/jcc-portability.sbatch`, one exclusive node
each, 7 pairs `abba` per cell, `--reps 20`, `RUST_LOG` unset, governor `performance`.
Raw logs: `benchmarks/results/slurm-70180{21,22,23}.out`.

Companion to `2026-09-10-hot-path-code-size.md`, which established that the reference
workstation (ccqlin038, Cascade Lake) pays the JCC erratum (SKX102) almost in full and that
`-Cllvm-args=-x86-branches-within-32B-boundaries` recovers **45.8% -> 98.0% DSB residency,
worth −9..−13% wall**. That flag currently applies to **all** `x86_64`. This note asks what
it does on the parts that do not have the erratum — which is every node type on `ccq`.

## The parts

The job prints its own verdict from `/proc/cpuinfo`; all three confirmed **not affected**:

| node | CPU | family/model | erratum |
|---|---|---|---|
| rome | AuthenticAMD Zen2 | 23 / 49 | no |
| genoa | AuthenticAMD Zen4 | 25 / 17 | no |
| icelake | GenuineIntel Ice Lake-SP | 6 / 106 | no |

Padding cost in code size, identical on all three (same commit, same target):
**2 044 024 B padded vs 2 003 272 B unpadded, +40 752 B = +2.03%.**

## Result: 13 of 13 direction-consistent phase results say the flag is a tax

Wall time mostly cannot resolve an effect this small — only 2 of 12 single-thread cells are
direction-consistent on wall, and they point opposite ways. The phase counters can, and they
are unanimous. Every 7/7 phase result at 1 thread, across all three parts:

| part | layer | phase | Δ% |
|---|---|---|---:|
| rome | trotter | sort | **+3.78** |
| rome | cnot | sort | +2.25 |
| icelake | rotation_zz | sort | +2.20 |
| icelake | su4 | merge | +1.98 |
| icelake | trotter | merge | +1.72 |
| icelake | cnot | merge | +1.41 |
| icelake | rotation_zz | merge | +1.15 |
| rome | su4 | merge | +1.14 |
| rome | su4 | gather | +1.12 |
| icelake | cnot | sort | +1.05 |
| genoa | su4 | gather | +0.86 |
| icelake | su4 | gather | +0.76 |
| genoa | cnot | sort | +0.60 |

**Padded slower: 13. Padded faster: 0.** Magnitudes +0.6% to +3.8%, clustered ~1–2% — the
right order for a +2.03% instruction tax landing on whichever phase is branchiest in a given
layer (the sort for the sparse layers, the gather and merge for su4).

The one contrary wall result — genoa `rotation_zz`, −1.52% at 7/7 — has **no** direction-
consistent phase movement behind it (gather, sort and merge all ns). A wall win with no
mechanism is a layout draw, not a benefit, and it is not evidence that padding helps Zen4.
The only other consistent wall cell, icelake `cnot` at **+1.13%, 7/7**, agrees with the phase
story and is backed by sort +1.05% and merge +1.41%, both 7/7.

## The multi-thread half of this campaign failed, by construction

Every multi-thread cell used the same `--n 1000000` as the single-thread cell, so at 64–128
threads there are only ~10–20k terms per worker and the measurement is scheduling noise:
rome `rotation_zz` at 128 threads spans −33.31% to +32.55%. **Zero** multi-thread wall cells
are direction-consistent.

The one exception is `su4`, which closes at ~1.4e7 terms from `--n 1e6` and therefore keeps
~150k terms per worker: genoa's su4 at 96 threads gives `gather_ns` −7.31% (7/7). That single
cell is the only usable multi-thread datum here and it is not enough to claim a direction.

**A multi-thread campaign needs `--n` scaled with the core count** — roughly `3e7` on a
96-core node to match the per-thread work of the 1-thread cell at `1e6`. The template should
be fixed before it is rerun for that purpose.

## What follows

The asymmetry is large and one-directional:

- Skylake-derived Intel (the reference workstation): **−9..−13% wall**.
- Everything else measured (Zen2, Zen4, Ice Lake): **+0.6..+3.8% per phase**, roughly +1% wall
  where wall resolves at all.

So the flag should be **narrowed to the parts that have the erratum** rather than applied to
all `x86_64`. The cost of not narrowing it is ~1% on the hardware most jobs actually run on;
the cost of dropping it is 9–13% on the reference host. Neither is acceptable to eat, and
they need not be.

Mechanism options, none free:

1. **`build.rs` probe of the build host's `/proc/cpuinfo`.** Correct exactly when the build
   host is the run host — which the `scripts/slurm/*.sbatch` templates guarantee, since they
   build in-job on the target node. Wrong for a wheel built on the workstation and run on
   `ccq`, and wrong for any cross-compile. The crate currently has a `build.rs` only for the
   `mpi` feature, and the "default build must be byte-identical" rule in CLAUDE.md means a
   host-dependent default needs care.
2. **A named cargo profile or documented per-site `RUSTFLAGS`.** Explicit and cross-compile
   safe; relies on the operator getting it right, and an exported `RUSTFLAGS` **replaces**
   the config's list wholesale, which is exactly the footgun `scripts/profile.sh` already hit.
3. **Leave it on and document the ~1% cluster tax.** Simplest, and defensible while the
   reference host is where optimization work happens.

**Resolved 2026-09-11: default off, measurement hosts opt in.** The flag is out of
`.cargo/config.toml`, so the shipped build is the portable one and no site pays a tax it
did not ask for. The failure mode that creates — the reference host silently losing
9-13% and corrupting an A/B — is closed by `scripts/jcc-rustflags.sh`, which detects the
erratum from `/proc/cpuinfo` and is sourced by every script that builds a binary it then
measures (`ab-compare.sh`, `bench-campaign.sh`, `perf-stat.sh`, `bandwidth.sh`,
`profile.sh`). Detection reads the CPU rather than a hostname, so it is correct on
uncalibrated nodes too.

Rejected: a `build.rs` host probe (same commit would produce different binaries on
different hosts — a real hazard for a methodology built on binary-vs-binary A/B), and
leaving it on with the tax documented (every cluster job pays ~1% forever on the strength
of operators reading docs).

Consequence for this template: the arms inverted. PADDED is now the arm that adds a flag
and UNPADDED is the plain default build.
