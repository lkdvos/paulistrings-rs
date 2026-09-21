# MPI ranks

Needs a from-source build with the `mpi` build option — see [Installation](../installation.md).

<!-- doctest: skip -->
```python
import mpi4py
mpi4py.rc.thread_level = "serialized"      # before `from mpi4py import MPI`
from mpi4py import MPI

import paulistrings
from paulistrings import truncation

comm = MPI.COMM_WORLD
observable = build_observable()   # the SAME terms on every rank
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

Check `paulistrings.mpi_available()` before relying on `comm=` — a build without the feature raises `RuntimeError` on it.
`comm` and `partitions` are alternatives; passing both is a `ValueError`.
Every rank must call `propagate` with the same observable, circuit, policy, direction and options — the scatter is a local filter of a sum every rank already holds, not a distribution of rank 0's copy.

Get each rank's own share back instead of a gather, when the answer is a scalar:

<!-- doctest: skip -->
```python
local = observable.propagate(circuit, policy, comm=comm, result="local")
value = comm.allreduce(local.expectation("z+"))
```

`result="gather"` (the default) gives rank 0 the whole evolved sum and every other rank an empty `PauliSum`; `result="local"` gives each rank its own disjoint share, which reductions over `comm` treat as exact.

Launch one rank per NUMA domain, bound to it:

```bash
mpirun -n 4 --map-by ppr:1:numa --bind-to numa python script.py
```

Stats work the same way, reduced over `comm` yourself:

<!-- doctest: skip -->
```python
_, stats = observable.propagate_with_stats(circuit, policy, comm=comm)
exported = comm.allreduce(sum(stats.partition.rows_exported))
```

See [MPI ranks](../explanation/mpi.md) for what this costs and buys, the rank-count and thread-level requirements, and [Installation](../installation.md) for the from-source MPI build steps.
