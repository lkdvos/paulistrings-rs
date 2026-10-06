# Slurm templates for the partitioned-engine measurements

Batch scripts that run the partitioned engine's measurements on an exclusive Rusty node rather than
the shared reference workstation. Submitting is a manual step (`sbatch`), by design: allocating
cluster resources is a user check-in point.

| script | what it runs | when |
|---|---|---|
| `ab-campaign.sbatch` | `scripts/ab-compare.sh` paired A/B cells: either a code A/B (`A_REV=<sha>` vs the working tree) or the runtime-knob A/B P=1 vs P=`<numa nodes>` on one binary | in-process partitioning |
| `gpu-devices.sbatch` | one 4 × A100 node (`gpu` partition): the single-device CUDA net, `tests/propagate_gpu_partitioned.rs` at one partition per GPU, then `phase_breakdown --device 0` and `--device 0,1,2,3` on the device cells | the CUDA backend, one process (needs the `cuda` cargo feature) |
| `mpi-gpu.sbatch` | one 4-GPU node by default (`--gpus-per-node=4 --ntasks-per-node=4`, every node GPU visible to every task), one GPU per MPI rank over NCCL: diagnostics preamble, `tests/mpi_ranks.rs` with `NCCL_DEBUG` on, then `phase_breakdown --mpi --device auto` at `--n 4e6 × ranks` | GPU per rank (needs `mpi` and `cuda`) |
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

## The GPU jobs

Both run on partition `gpu`, whose `scontrol show partition gpu` reads `Exclusive=NO OverSubscribe=NO` (exclusivity is the job's choice, not forced) under QoS `gpu` (at most 24 GPUs and 432 CPUs per user).
The templates do not ask for `--exclusive`, so they schedule sooner; a job holding all four GPUs of a node already keeps other GPU jobs off it, and for a quiet-host timing campaign add `--exclusive` on the `sbatch` line.
The 4 × A100-SXM4-80GB NVLink nodes are `--constraint='a100-80gb&rocky9'` (the default; `rocky9` because workergpu038–040 are still rocky8 and `modules/2.4-20250724` is the rocky9 stack, ); the 4 × H100-SXM5 genoa nodes are `--constraint=h100-sxm5`.
The device-to-device templates exclude workergpu062, the one A100 node with two GPUs, whose link a `--gres=gpu:2` request could otherwise land on.
Run `scripts/slurm/setup-shared-toolchain.sh` once after pulling, with `module load openmpi/5.0.6 llvm/19.1.7` and `LIBCLANG_PATH` set, so its offline checks cover `cuda` and `mpi,cuda` and the registry holds `cudarc`.

```bash
# one 4 x A100 node: device nets, then the probe on one GPU and on all four
env -u SBATCH_RESERVATION sbatch scripts/slurm/gpu-devices.sbatch
# the same on an H100-SXM5 node
env -u SBATCH_RESERVATION sbatch --constraint=h100-sxm5 scripts/slurm/gpu-devices.sbatch
# one GPU per rank: 1 node x 2 ranks, 1 node x 4 (the default), 2 nodes x 8, H100-SXM5, or the net alone
env -u SBATCH_RESERVATION sbatch --ntasks-per-node=2 --gpus-per-node=2 scripts/slurm/mpi-gpu.sbatch
env -u SBATCH_RESERVATION sbatch scripts/slurm/mpi-gpu.sbatch
env -u SBATCH_RESERVATION sbatch --nodes=2 scripts/slurm/mpi-gpu.sbatch
env -u SBATCH_RESERVATION sbatch --constraint=h100-sxm5 scripts/slurm/mpi-gpu.sbatch
PROBE=0 env -u SBATCH_RESERVATION sbatch scripts/slurm/mpi-gpu.sbatch
```

`gpu-devices.sbatch` writes `benchmarks/results/<date>-<node>/gpu-<job>-dev0.jsonl` and `gpu-<job>-dev0123.jsonl`, plus `gpu-<job>-topo.txt` (`nvidia-smi topo -m`), `gpu-<job>-p2p.txt` (`nvidia-smi topo -p2p` and `nvlink -s`, on two or more GPUs) and the clock, power and temperature dumps at start and end.
`mpi-gpu.sbatch` writes one sidecar per rank, `benchmarks/results/<date>-mpi-gpu/mpi-gpu-<job>-r<ranks>.jsonl.rank<N>`, each row carrying `rank`, `ranks` and the rank's `device`.
It asks for 16 GB per CPU (128 GB per rank): the probe replicates its input in host memory on every rank, above the partition's default of about 250 GB per four-rank node.
Render either directory with `scripts/perf-viz.py <dir>/<prefix>` for the phase charts, and read the numbers for `research/HARDWARE.md` straight from the sidecars: per row `n`, `wall_ns / layers`, the device phases (`gather_ns`, `merge_ns`, `compact_ns`, `coset_loop_ns`, `h2d_ns`, `d2h_ns`), and on a multi-device or rank row `export_ns`, `exchange_ns`, `chunk_wait_ns`, `bytes_exported` and `vmhwm_kb` (medians over ranks, as the MPI weak-scaling table does).
The table skeletons are under the cluster GPU sections of `research/HARDWARE.md`.

`mpi-gpu.sbatch` exposes every node GPU to every task (`--gpus-per-node`/`--ntasks-per-node`, not `--gpus-per-task=1`): the per-task device cgroup otherwise makes NCCL fall back to SHM through host memory instead of P2P/NVLink.
Its preamble logs `nvidia-smi topo -m`, `ibv_devinfo -l`, the `nvidia_peermem`/`nv_peer_mem` module state and `ip -br addr`, and `tests/mpi_ranks.rs` runs with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,NET`, so the transport NCCL picked is in the job log.

## Bounded steps

Every `srun` in the GPU templates runs through `bounded <secs> srun ...` from `scripts/slurm/bounded.sh`: `timeout` sends TERM at the bound and KILL 30 s later, the step's status is reported, and a failed step lists the compute processes left on the node's GPUs (`leftover_gpu_processes`, an `--overlap` step so it runs beside whatever is stuck).
A hung rank (a device wait that never ends, an `MPI_Abort` that never returns) therefore costs one step's bound, not the allocation's time limit; `NET_BOUND` and `PROBE_BOUND` set the bounds in seconds.
The differential net of `mpi-gpu.sbatch` runs under `PAULISTRINGS_NCCL_TIMEOUT_S=60` (`NCCL_TIMEOUT_S` overrides), since its exchanges take milliseconds and a 300 s wait is the engine's production bound, not a test's.
A failed net is reported, the probe is skipped and the job exits non-zero.
`tests/mpi_ranks.rs` itself arms a watchdog before `MPI_Abort` and aborts the process 15 s later if the launcher has not ended it.

## JCC-erratum padding across node types

`-Cllvm-args=-x86-branches-within-32B-boundaries` is worth −9..−13% wall on the reference
workstation (ccqlin038, Cascade Lake): the
JCC erratum (SKX102) excludes any 32-byte fetch window whose jump crosses or ends on the boundary
from the decoded-uop cache, and the padding takes DSB residency from 45.8% to 98.0%
(`research/FINDINGS.md`).

**The erratum is Skylake-derived Intel only**, and the campaign settled it: across rome (Zen2),
genoa (Zen4) and icelake (Ice Lake-SP), **13 of 13 direction-consistent phase results at 1 thread
show the padded build slower** (+0.6..+3.8%). So the flag is **not** in `.cargo/config.toml` — the
shipped default is portable, and hosts with the erratum opt in via `scripts/jcc-rustflags.sh`,
which every measurement script sources. Cluster jobs therefore get the right build automatically.

## The operator Loschmidt echo jobs (`examples/b8_ole`)

| script | what it runs |
|---|---|
| `ole-ppmc-disbatch.sbatch` | independent PP-MC trajectories as disBatch tasks inside one allocation, `--ntasks` concurrent tasks of `--cpus-per-task` threads |
| `ole-mpi.sbatch` | one trajectory or deterministic run at a time with the sum distributed one partition per NUMA domain (`--mpi`), seeds sequential |

Both run the Python driver from `./.venv-mpi` in the checkout the job is submitted from, built by `scripts/sync-mpi-venv.sh` as in CLAUDE.md (Commands); the MPI build also serves the non-MPI disBatch runs.
That install is non-editable, so the venv pins the commit it was built from; re-run the script after changing code.
A git worktree does not share the main checkout's venvs, so build one per worktree.
Budget about 3 × 32 B × cache per trajectory, counting the transient growth of a rotation before collapse; a ccq rome or icelake node has 1 TB and a genoa node 1.5 TB.

```bash
# paper replication: 6 alphas x 800 trajectories at cache 5e8, 10 seeds per task, 8 x 16-thread tasks per rome node
python examples/b8_ole/make_tasks.py --seeds 0:800 --block 10 --threads 16 --out benchmarks/results/ole/ppmc -- --policy ppmc --cache 5e8 > benchmarks/results/ole-tasks.txt
env -u SBATCH_RESERVATION TASKS=benchmarks/results/ole-tasks.txt sbatch scripts/slurm/ole-ppmc-disbatch.sbatch
env -u SBATCH_RESERVATION TASKS=benchmarks/results/ole-tasks.txt sbatch --nodes=4 --ntasks=32 scripts/slurm/ole-ppmc-disbatch.sbatch

# beyond one node: cache 3e10 over 8 genoa nodes (16 ranks), 4 trajectories per alpha
env -u SBATCH_RESERVATION CACHE=3e10 ALPHAS=0.05,0.1,0.15 SEEDS=0:4 sbatch --nodes=8 scripts/slurm/ole-mpi.sbatch

# deterministic baseline at a huge budget, with the exact overlap
env -u SBATCH_RESERVATION POLICY=approx_topn CACHE=2e10 SEEDS=0:1 EXACT=1 ALPHAS=0.05 sbatch --nodes=8 scripts/slurm/ole-mpi.sbatch
```

A sweep that outruns its time limit continues with `RESUME=benchmarks/results/disbatch-<jobid>-_status.txt` on the same `TASKS`.
`examples/b8_ole/aggregate.py benchmarks/results/ole --plot ...` summarizes every run.
