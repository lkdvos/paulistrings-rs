# Module helpers

| Call | Returns |
|---|---|
| `paulistrings.numa_nodes()` | `list[list[int]]`, the NUMA nodes this process may run on, one CPU-index list per node, ascending node order; intersected with the process's CPU affinity mask |
| `paulistrings.mpi_available()` | `bool`, whether this build can run `PauliSum.propagate(comm=...)` (compiled with the `mpi` feature) |
| `paulistrings.DEFAULT_SMALL_SUM_THRESHOLD` | `int`, the default `small_sum_threshold` `propagate`/`propagate_with_stats` use when the kwarg is omitted |
| `paulistrings.reset_log_cache()` | drops pyo3-log's cached per-logger effective level |

## Logging gotcha

`pyo3-log` caches each logger's effective level the first time it is consulted.
Call `paulistrings.reset_log_cache()` after changing a Python log level mid-process (e.g. `logging.getLogger("paulistrings.propagate").setLevel(logging.DEBUG)`), or the new level is not picked up.

See [Propagation stats and logs](../how-to/read-propagation-stats-and-logs.md) for the full recipe.
