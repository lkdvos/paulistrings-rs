# Running across MPI ranks

A distributed run is a [partitioned run](numa.md) with one partition per MPI process: each rank holds
a disjoint share of the terms, selected by designated rows of the GF(2) hash, and a layer exchanges
only the rows that cross a rank boundary. Everything on the NUMA page still applies — the same layer
loop, the same truncation rules, the same two things that change. What is new is the transport
(point-to-point MPI instead of in-process channels) and the launch.

It is behind the off-by-default `mpi` cargo feature, so the default wheel cannot do it.
`paulistrings.mpi_available()` says whether this build can, and `comm=` in a build without it raises
`RuntimeError`.

## Building it

The feature needs an MPI installation (for `mpicc`, which rsmpi's build script probes) and a
`libclang` for its bindgen. On a Flatiron host:

```bash
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7 python-mpi/3.12.9
export LIBCLANG_PATH=$(llvm-config --libdir)

python3 -m venv --system-site-packages .venv-mpi   # so the module's mpi4py is visible
.venv-mpi/bin/pip install maturin pytest numpy
VIRTUAL_ENV=$PWD/.venv-mpi .venv-mpi/bin/maturin develop --release --features mpi \
    -m crates/paulistrings-py/Cargo.toml
```

`mpi4py` must be built against the *same* MPI the extension links; the boundary checks the width of
`MPI_Comm` and refuses a mismatch rather than corrupting a handle. The built extension carries an
rpath to that MPI's library directory, so `import paulistrings` works from a shell with no modules
loaded.

`scripts/mpi-test.sh --ranks 2,4 --python` builds the extension into that venv and runs
`python/paulistrings/tests/test_mpi.py` under `mpirun` at each rank count.

## Python

```python
import mpi4py
mpi4py.rc.thread_level = "serialized"      # before mpi4py.MPI is imported
from mpi4py import MPI

import numpy as np
import paulistrings
from paulistrings import Circuit, PauliSum, truncation

comm = MPI.COMM_WORLD
observable = build_observable(seed=20260908)   # the SAME terms on every rank
circuit = build_circuit()

evolved = observable.propagate(
    circuit,
    truncation.approx_topn(1_000_000),
    direction="heisenberg",
    comm=comm,
)
if comm.Get_rank() == 0:
    print(len(evolved), evolved.expectation("z+"))
```

`comm` and `partitions` are alternatives — passing both is a `ValueError`. A distributed run places
one partition per *process*, so the placement is the launcher's job.

### `result="gather"` versus `result="local"`

| `result` | what each rank gets back |
|---|---|
| `"gather"` (default) | rank 0 the whole evolved sum; every other rank an **empty** `PauliSum` of the same `num_qubits`, so downstream code still type-checks |
| `"local"` | this rank's own share |

The shares are disjoint and cover the result, so reductions over `comm` are exact:

```python
local = observable.propagate(circuit, policy, comm=comm, result="local")
terms = comm.allreduce(len(local))
value = comm.allreduce(local.expectation("z+"))
```

Prefer `"local"` when the answer is a scalar: gathering materializes the whole sum on rank 0, and
that rank transiently holds one extra rank's copy on top of it.

### Launching

One rank per NUMA domain, bound to it — that is the placement the in-process engine would have made
with explicit CPU lists, done by the launcher instead:

```bash
mpirun -n 4 --map-by ppr:1:numa --bind-to numa python script.py
srun --ntasks-per-node=4 --cpu-bind=ldoms --mpi=pmix python script.py
```

Each rank's Rayon pool sizes itself from the CPUs the launcher left in its affinity mask, so an
unbound launch gives every rank a pool over the whole node and they fight. `RAYON_NUM_THREADS` is
ignored, as in any partitioned run.

## The three requirements

**Thread level at least `MPI_THREAD_SERIALIZED`.** The layer loop runs inside a pinned Rayon pool, so
MPI is called from a pool worker rather than the process's main thread — and not necessarily the same
worker on every layer. Only ever one at a time, which is exactly `SERIALIZED`; `FUNNELED` would be a
false claim. Set `mpi4py.rc.thread_level` *before* `from mpi4py import MPI` (mpi4py's default,
`"multiple"`, is also fine). A weaker level is a `RuntimeError` at the boundary.

**A power-of-two rank count.** A partition is named by `log2(P)` GF(2) hash rows, so `mpirun -n 3` is
a `ValueError`.

**A replicated input.** Every rank must call `propagate` with the same observable, circuit, policy,
direction and options. The scatter is a *local filter* of a sum every rank already holds, not a
distribution of rank 0's copy — build the terms from the same seed, or load the same file, on every
rank. Nothing checks the terms; the run's *shape* is checked, and a group that disagrees about the
circuit aborts with a fingerprint mismatch rather than hanging.

`propagate` is collective: every rank of `comm` must reach it, in the same order. A rank that skips
one hangs the rest.

## Stats

`propagate_with_stats(..., comm=comm)` fills `PropagationStats.partition` with a `PartitionStats`
whose `rank` and `size` name the group (both are `None` for a run that is not distributed). Its
per-layer lists hold **this rank's entry only** — gathering the group's counters would mean a
collective per layer for a diagnostic — so `terms_in[k]` is a one-element list, `rows_exported[k]` is
what this rank sent, and `imbalance[k]` is always `1.0`. Reduce over `comm` for the group's picture:

```python
_, stats = observable.propagate_with_stats(circuit, policy, comm=comm)
exported = comm.allreduce(sum(stats.partition.rows_exported))
```

## Limits

- **Exact `topn` is unavailable**, as in any partitioned run: choosing the `n`-th largest magnitude
  across ranks is a distributed selection. `truncation.approx_topn(n)` all-reduces its octave
  histogram and retains exactly the set a single-process run would have.
- **Homogeneous groups only.** The wire carries raw host bytes, so a mixed-endianness job is silently
  wrong.
- **One partition per rank.** The hybrid — several NUMA domains inside one rank — is not implemented.
- **No committed measurement yet.** The transport works and is tested at 1, 2 and 4 ranks; no scaling
  campaign has been run. Treat it as a facility to try, not as a documented speedup.
