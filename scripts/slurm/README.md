# Slurm templates for the partitioned-engine measurements

Batch scripts that run the partitioned engine's measurements on an exclusive Rusty node rather than
the shared reference workstation. Submitting is a manual step (`sbatch`), by design: allocating
cluster resources is a user check-in point.

| script | what it runs | when |
|---|---|---|
| `ab-campaign.sbatch` | `scripts/ab-compare.sh` paired A/B cells: either a code A/B (`A_REV=<sha>` vs the working tree) or the runtime-knob A/B P=1 vs P=`<numa nodes>` on one binary | in-process partitioning |
| `jcc-portability.sbatch` | the shipping `-Cllvm-args=-x86-branches-within-32B-boundaries` padding, paired against an unpadded build of the same commit, at 1 thread and full physical cores | deciding whether the JCC-erratum flag helps or costs on a given node type — **and** the quiet-box multi-thread campaign |
| `mpi-ranks.sbatch` | the multi-rank differential test (`tests/mpi_ranks.rs`) at one rank per NUMA domain under `srun --cpu-bind=ldoms --mpi=pmix`, then `phase_breakdown --mpi` across the allocation | distributed runs (needs the `mpi` cargo feature) |

## Node choice

Partition `ccq` (the user is at CCQ; `scc` is the Scientific Computing Core's). Snapshot of `sinfo -p ccq` on 2026-09-08: 43 idle **icelake** (2 × 32 cores, rocky9), 4 idle rome
(2 × 64), 11 genoa (2 × 48, mostly allocated). The templates default to
`--constraint=icelake&rocky9` for availability and because rocky9 matches the module stack the
workstation uses; override with `sbatch --constraint='genoa&rocky9' ...` to get more NUMA domains per
socket (P = 4 or 8 if the BIOS exposes NPS4). The scripts never hard-code CPU lists: `ab-campaign.sbatch`
reads `/sys/devices/system/node/node*/cpulist` on the node and builds the `--partition-cpus` string
from it, so a node with N NUMA domains runs P = N, and `mpi-ranks.sbatch` lets `--cpu-bind=ldoms`
do the same job for it.

## One-time setup: a toolchain on the shared filesystem

`~/.cargo` and `~/.rustup` on the CCQ workstations are symlinks into the local NVMe `/home`, which
cluster nodes do not mount — rustup's `cargo` proxy dangles there and the first jobs died with
`cargo: command not found`. Run once, on a host with network access:

```bash
scripts/slurm/setup-shared-toolchain.sh     # ~750 MB under $HOME/.local/rust-shared
```

It installs the `rust-toolchain.toml` channel and pre-fetches the crate registry there; the templates
export `RUSTUP_HOME`/`CARGO_HOME` accordingly and fail fast (exit 3) if `cargo` is still unusable.
Re-run it after changing the toolchain pin or the dependency set.

## Usage

The workstation environment exports `SBATCH_RESERVATION=rocky9` (a leftover of the rocky9 migration;
that reservation no longer exists), which makes every `sbatch` fail with "Requested reservation is
invalid". Unset it for the submission (`env -u SBATCH_RESERVATION sbatch ...`, or `unset
SBATCH_RESERVATION` once in the shell).

```bash
# P=1 vs P=<numa> runtime-knob campaign on one binary:
env -u SBATCH_RESERVATION sbatch scripts/slurm/ab-campaign.sbatch
# same, choosing layers / sizes / pairs:
LAYERS="su4 rotation_local rotation_remote gu2q" NS="1000000 3000000" PAIRS=5 sbatch scripts/slurm/ab-campaign.sbatch
# code A/B of the untouched path (P=1 both sides): baseline sha vs the working tree
A_REV=e7de227 LAYERS="rotation_zz cnot su4" sbatch scripts/slurm/ab-campaign.sbatch
# more NUMA domains:
sbatch --constraint='genoa&rocky9' scripts/slurm/ab-campaign.sbatch
```

## The MPI ranks job

`mpi-ranks.sbatch` builds with `--features mpi` and runs the differential net across the allocation
at **one rank per NUMA domain** (`D = 1`). The engine holds exactly one partition per process and
reads its placement from the launcher's affinity mask, so `--cpu-bind=ldoms` is what pins the run —
there is no domains-per-rank knob and no `--partition-cpus` string to build. The rank count must be
a power of two (a partition is named by `log2(P)` GF(2) rows), so the script rounds
`nodes × domains` down and prints what it picked: two icelake nodes of two domains each give four
ranks. The net having passed, it then runs `phase_breakdown --mpi` at the same rank count — `LAYERS`
and `N` choose the cells — and each rank writes its own `.rank<N>` sidecar.

```bash
env -u SBATCH_RESERVATION sbatch scripts/slurm/mpi-ranks.sbatch
env -u SBATCH_RESERVATION sbatch --nodes=4 scripts/slurm/mpi-ranks.sbatch          # 8 ranks
env -u SBATCH_RESERVATION sbatch --constraint='genoa&rocky9' scripts/slurm/mpi-ranks.sbatch
```

The build needs `libclang` for rsmpi's bindgen, which is why the template loads `llvm/19.1.7`
alongside `openmpi/5.0.6` and exports `LIBCLANG_PATH`. Run `cargo fetch` once on a login host after
adding the feature — the job builds `--offline` against the shared registry cache. On the
workstation the same net runs without Slurm through `scripts/mpi-test.sh --ranks 2,4`.

Jobs build the commit `PS_REV` (default: `HEAD` when the job starts) in a git worktree, never the
live checkout, so editing the tree while jobs are queued is safe; pin with
`PS_REV=$(git rev-parse HEAD) sbatch ...` when the branch may move before the job starts.

Output lands where `ab-compare.sh` always puts it, `benchmarks/results/<date>-<nodename>/`
(gitignored, on the shared home filesystem), plus `benchmarks/results/slurm-<jobid>.out` with the
node's topology dump (`lscpu`, `numactl -H`) as provenance. Results are **per host**: a node's numbers
are not comparable with ccqlin038's, but P=1 vs P=N on the same node is the comparison that means
something. Roofline percentages need that node's ceilings: run `crates/membench` there
(`MEMBENCH=1 sbatch ...` adds a short node-local / all-core pass) and add a `host-topology.sh` arm
for the node type before quoting them.

Hardware counters (`scripts/perf-stat.sh`, uncore IMC bandwidth, remote-load share) are not part of
the templates: they need `perf_event_paranoid` low enough on the node, which is typically not the
case on the cluster. Run the counter pass on the workstation for the cells that matter.

The repo's `target` is likewise a symlink into the workstation's local `/home` (dangling on the
nodes), which is why `ab-compare.sh` honours `CARGO_TARGET_DIR` for its worktrees and binaries.
Builds happen on the node into a job-private `CARGO_TARGET_DIR` under the node's local scratch
(`$TMPDIR`), with `cargo --offline` against the shared `~/.cargo` registry cache — so run any
`cargo fetch`/build once on a login host first if dependencies changed.

## JCC-erratum padding across node types

`.cargo/config.toml` carries `-Cllvm-args=-x86-branches-within-32B-boundaries` for **all**
`x86_64`. On the reference workstation (ccqlin038, Cascade Lake) it is worth −9..−13% wall: the
JCC erratum (SKX102) excludes any 32-byte fetch window whose jump crosses or ends on the boundary
from the decoded-uop cache, and the padding takes DSB residency from 45.8% to 98.0%
(`research/notes/2026-09-10-hot-path-code-size.md`).

**The erratum is Skylake-derived Intel only.** AMD Zen (rome, genoa) and Ice Lake and later do not
have it and pay ~2% extra instructions for nothing. `ccq` is mostly rome and genoa, so the shipping
default is unvalidated on the hardware most jobs actually run on. `jcc-portability.sbatch` settles
it per node type; the job prints whether the erratum applies to the part it landed on, and refuses
to run if the two builds come out byte-identical (i.e. the override silently failed).

```bash
for c in rome genoa icelake; do
  ISA_TAG=$c env -u SBATCH_RESERVATION sbatch --constraint="$c&rocky9" \
    scripts/slurm/jcc-portability.sbatch
done
```

Read it as: side B (padded) consistently lower ⇒ the shipping config is right for that part. Side B
consistently *higher* ⇒ narrow the flag from `cfg(target_arch = "x86_64")` to something that
excludes it. There is no cargo cfg for a CPU model, so narrowing means either a `target-cpu`-keyed
profile, a documented per-site `RUSTFLAGS` (remembering that an exported `RUSTFLAGS` **replaces**
the config's list wholesale), or a `build.rs` that emits the flag conditionally.

### The multi-thread campaign

The same template runs it, via `N_PER_THREAD`, which keeps the per-worker work fixed
instead of the total. A fixed `--n` across thread counts starves the workers: jobs
7018021-3 ran that way and produced **zero** direction-consistent multi-thread wall cells,
with spreads to −33%..+33% at ~10–20k terms per worker.

```bash
# sparse layers, per-worker work matched to the 1-thread cell
N_PER_THREAD=1000000 LAYERS="rotation_zz cnot trotter" ISA_TAG=genoa \
  env -u SBATCH_RESERVATION sbatch --constraint='genoa&rocky9' \
  scripts/slurm/jcc-portability.sbatch

# su4 separately, unscaled — it closes at ~14x --n, so scaling it asks for ~1.4e9 terms.
# The job refuses (exit 6) if su4 appears in LAYERS with N_PER_THREAD set.
LAYERS=su4 NS=1000000 ISA_TAG=genoa env -u SBATCH_RESERVATION sbatch \
  --constraint='genoa&rocky9' scripts/slurm/jcc-portability.sbatch
```

Hardware counters do not work on cluster nodes, so this job is wall-clock paired only — no DSB
share and no branch-miss attribution. Counter work stays on the workstation.
