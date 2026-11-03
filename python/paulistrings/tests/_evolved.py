"""Copying forms of the in-place ``PauliSum.propagate``, for tests that compare an input with its evolution."""


def propagated(sum_, *args, **kwargs):
    """``sum_`` propagated by ``PauliSum.propagate(*args, **kwargs)``, leaving ``sum_`` untouched."""
    out = sum_.copy()
    out.propagate(*args, **kwargs)
    return out


def propagated_with_stats(sum_, *args, **kwargs):
    """``(evolved, stats)`` from ``PauliSum.propagate_with_stats``, leaving ``sum_`` untouched."""
    out = sum_.copy()
    stats = out.propagate_with_stats(*args, **kwargs)
    return out, stats
