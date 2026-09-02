# `examples/data/references/` — published reference values

**This directory ships with no data files.** Nothing has been fetched into the
repo, and no reference number in this suite may be written down from
recollection: every numeric claim is computed by an oracle or loaded from a
provenance-tagged reference file.

Files here are read by
`examples/common/oracles.py::load_published_reference(name)`, which **refuses
any file whose header does not record `source`, `method` and `accuracy`**. That
refusal is the point of the loader: a value without a traceable origin, a stated
method, and a stated accuracy cannot reach a plot or a test through it. The
other three oracles in that module (`statevector_expectation`,
`stim_clifford_exact`, `light_cone_exact`) *compute* their references and need
nothing from here.

## Required header

Two formats, one requirement. Extra fields are kept and returned verbatim —
`retrieved`, `doi`, `figure`, `notes`, `license` are all worth recording.

### CSV — leading `# key: value` comment lines, then a header row

```
# source: https://github.com/tbegusic/arxiv-2308.05077-data (exact.csv)
# method: <what produced the numbers, e.g. "belief-propagation + Pauli, converged">
# accuracy: <stated or estimated error, e.g. "1e-3 absolute">
# retrieved: 2026-09-01
# doi: 10.5281/zenodo.10223349
theta_h,observable,value
0.0,weight_1_z62,1.0
```

`load_published_reference` returns the rows as dicts keyed by the header row;
`PublishedReference.column("value")` converts one column to a NumPy array.

### JSON — a top-level `"provenance"` object, payload under `"data"`

```json
{
  "provenance": {
    "source": "…",
    "method": "…",
    "accuracy": "…",
    "retrieved": "2026-09-01"
  },
  "data": [{"steps": 20, "theta_h": 1.0, "value": 0.0}]
}
```

A JSON `"data"` that is a list surfaces as `rows`; anything else surfaces
unchanged as `payload`.

## Upstream pointers (not fetched)

For the 127-qubit kicked-Ising benchmarks (A–C), two public sources exist. Both are recorded in `examples/data/README.md` as well;
neither has been retrieved.

| What | Where | Note |
|---|---|---|
| Kim et al. (2023) experimental values with error bars | <https://doi.org/10.6084/m9.figshare.22500355> | Refuses automated fetch |
| Begušić, Gray & Chan (arXiv:2308.05077) converged exact values, θ_h grid of π/32, all four observables | <https://github.com/tbegusic/arxiv-2308.05077-data> (`exact.csv`); Zenodo [10.5281/zenodo.10223349](https://doi.org/10.5281/zenodo.10223349) | Would drop in here as `begusic2023_exact.csv` with the header above |

Benchmark C does not block on either: its reference is self-converged with
documented convergence evidence, and published values are used only if they can
be obtained with clean provenance.

## Adding a file

1. Fetch it, and record the URL and retrieval date in the header — the header is
   the citation, so it must be written from the fetch, not from memory.
2. State the `method` and `accuracy` the *source* claims. If the source states
   neither, it is not a reference file; treat it as data to be reproduced by an
   oracle instead.
3. Note the file in this README's table in the same commit, and check that
   `test_examples_oracles.py::test_references_directory_ships_a_readme_and_no_data_files`
   is updated deliberately rather than deleted — it is the tripwire that keeps
   an untagged file from appearing here unnoticed.
