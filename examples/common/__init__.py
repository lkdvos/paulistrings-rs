"""Shared infrastructure for the `examples/` showcase suite.

See `research/plans/2026-08-31-examples-benchmarks-suite.md` (file-layout
reconciliation, §4) for the intended contents of this package:
`circuits.py`, `observables.py`, `oracles.py`, `harness.py` and `report.py`; of
those, `circuits.py`, `observables.py` and `report.py` are present so far. This
module intentionally stays empty — it is a package marker only, so that
importing `common.circuits` does not drag in `report.py`'s plotting deps.
"""
