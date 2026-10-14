"""Bond lists of common lattices, for ``pt(label, bonds)``.

Sites are numbered from 0; a bond is a pair of sites coupled by a two-site term.
"""

from __future__ import annotations

__all__ = ["chain", "square"]


def chain(L: int, *, periodic: bool = False) -> list[tuple[int, int]]:
    """Bonds ``(j, j + 1)`` of an ``L``-site chain, plus ``(L - 1, 0)`` when ``periodic``.

    A periodic chain needs ``L >= 3``: at ``L = 2`` the closing bond repeats ``(0, 1)``.
    """
    if periodic and L < 3:
        raise ValueError(f"a periodic chain needs L >= 3, got L = {L}")
    bonds = [(j, j + 1) for j in range(L - 1)]
    if periodic:
        bonds.append((L - 1, 0))
    return bonds


def square(lx: int, ly: int, *, periodic: bool = False) -> list[tuple[int, int]]:
    """Nearest-neighbour bonds of an ``lx`` by ``ly`` square lattice, site ``(x, y)`` at index ``y * lx + x``.

    Bonds are listed site by site in index order, each site's right neighbour before its lower one.
    A periodic lattice needs ``lx, ly >= 3``: a side of length 2 repeats its bonds.
    """
    if periodic and min(lx, ly) < 3:
        raise ValueError(f"a periodic square lattice needs lx, ly >= 3, got {lx} x {ly}")
    bonds = []
    for y in range(ly):
        for x in range(lx):
            j = y * lx + x
            if periodic or x + 1 < lx:
                bonds.append((j, y * lx + (x + 1) % lx))
            if periodic or y + 1 < ly:
                bonds.append((j, ((y + 1) % ly) * lx + x))
    return bonds