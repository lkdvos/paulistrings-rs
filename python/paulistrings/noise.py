"""Noise channel factories — re-export of the compiled ``noise`` submodule."""

from ._paulistrings import noise as _noise

depolarize = _noise.depolarize
dephase = _noise.dephase
amplitude_damping = _noise.amplitude_damping
pauli_channel = _noise.pauli_channel
depolarize2 = _noise.depolarize2

__all__ = [
    "depolarize",
    "dephase",
    "amplitude_damping",
    "pauli_channel",
    "depolarize2",
]
