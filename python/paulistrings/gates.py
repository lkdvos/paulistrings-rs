"""Gate factories — re-export of the compiled ``gates`` submodule."""

from ._paulistrings import gates as _gates

h = _gates.h
cnot = _gates.cnot
rz = _gates.rz

__all__ = ["h", "cnot", "rz"]
