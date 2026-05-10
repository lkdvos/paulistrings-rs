"""Gate factories — re-export of the compiled ``gates`` submodule."""

from ._paulistrings import gates as _gates

h = _gates.h
s = _gates.s
x = _gates.x
y = _gates.y
z = _gates.z
cnot = _gates.cnot
cz = _gates.cz
swap = _gates.swap
rz = _gates.rz
rx = _gates.rx
ry = _gates.ry

__all__ = ["h", "s", "x", "y", "z", "cnot", "cz", "swap", "rz", "rx", "ry"]
