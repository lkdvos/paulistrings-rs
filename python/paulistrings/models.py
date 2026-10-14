"""Hamiltonians of standard spin models as ``PauliTerms``.

Terms are grouped by type; each group commutes internally, so a Trotter step's error comes only from between groups.
Groups with a zero coupling are omitted, since ``Circuit.trotter`` would turn them into zero-angle rotations.
"""

from __future__ import annotations

from ._paulistrings import PauliTerms, pt

__all__ = ["transverse_ising", "xxz"]


def _nonzero_sum(groups: list[tuple[float, PauliTerms]]) -> PauliTerms:
    return sum((c * terms for c, terms in groups if c != 0), PauliTerms())


def transverse_ising(bonds, sites, J: float = 1.0, h: float = 1.0) -> PauliTerms:
    """``H = -J Σ_{(j, j') in bonds} Z_j Z_j' - h Σ_{j in sites} X_j``, bonds before fields."""
    return _nonzero_sum([(-J, pt("ZZ", bonds)), (-h, pt("X", sites))])


def xxz(bonds, J: float = 1.0, Jz: float = 1.0, sites=(), h: float = 0.0) -> PauliTerms:
    """``H = J Σ (X_j X_j' + Y_j Y_j') + Jz Σ Z_j Z_j' - h Σ_{j in sites} Z_j``, ordered XX, YY, ZZ, fields."""
    return _nonzero_sum(
        [(J, pt("XX", bonds)), (J, pt("YY", bonds)), (Jz, pt("ZZ", bonds)), (-h, pt("Z", sites))]
    )