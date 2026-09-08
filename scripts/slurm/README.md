# Slurm templates for the partitioned-engine measurements

Batch scripts that move the *quiet-box* measurement steps of the partitioned-engine plan off the
reference workstation and onto an exclusive Rusty node. Submitting is a manual step (`sbatch`), by
design: allocating cluster resources is a user check-in point.

| script | what it runs | when |
|---|---|---|
| `ab-campaign.sbatch` | `scripts/ab-compare.sh` paired A/B cells: either a code A/B (`A_REV=<sha>` vs the working tree) or the runtime-knob A/B P=1 vs P=`<numa nodes>` on one binary | phase 2 (and the post-S9 re-check) |
| `mpi-ranks.sbatch` | the multi-rank differential test and the D-domains-vs-ranks probe comparison under `srun --mpi=pmix` | phase 3 (needs the `mpi` cargo feature; placeholder until it lands) |

## Node choice

Partition `ccq` (the user is at CCQ; `scc` is the Scientific Computing Core's). Snapshot of `sinfo -p ccq` on 2026-09-08: 43 idle **icelake** (2 × 32 cores, rocky9), 4 idle rome
(2 × 64), 11 genoa (2 × 48, mostly allocated). The templates default to
`--constraint=icelake&rocky9` for availability and because rocky9 matches the module stack the
workstation uses; override with `sbatch --constraint='genoa&rocky9' ...` to get more NUMA domains per
socket (P = 4 or 8 if the BIOS exposes NPS4). The scripts never hard-code CPU lists: they read
`/sys/devices/system/node/node*/cpulist` on the node and build the `--partition-cpus` string from it,
so a node with N NUMA domains runs P = N.

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
env -u SBATCH_RESERVATION sbatch scripts/slurm/ab-campaign.sbatch
# P=1 vs P=<numa> runtime-knob campaign on one binary (the phase-2 headline):
# same, choosing layers / sizes / pairs:
LAYERS="su4 rotation_local rotation_remote gu2q" NS="1000000 3000000" PAIRS=5 sbatch scripts/slurm/ab-campaign.sbatch
# code A/B of the untouched path (P=1 both sides): baseline sha vs the working tree
A_REV=e7de227 LAYERS="rotation_zz cnot su4" sbatch scripts/slurm/ab-campaign.sbatch
# more NUMA domains:
sbatch --constraint='genoa&rocky9' scripts/slurm/ab-campaign.sbatch
```

Jobs build the commit `PS_REV` (default: `HEAD` when the job starts) in a git worktree, never the
live checkout, so editing the tree while jobs are queued is safe; pin with
`PS_REV=$(git rev-parse HEAD) sbatch ...` when the branch may move before the job starts.

Output lands where `ab-compare.sh` always puts it, `benchmarks/results/<date>-<nodename>/`
(gitignored, on the shared home filesystem), plus `benchmarks/results/slurm-<jobid>.out` with the
node's topology dump (`lscpu`, `numactl -H`) as provenance. Results are **per host**: a node's numbers
are not comparable with ccqlin038's, but P=1 vs P=N on the same node is exactly the comparison the
plan needs. Roofline percentages need that node's ceilings: run `crates/membench` there
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
