"""Truncation policy factories — re-export of the compiled ``truncation`` submodule."""

from ._paulistrings import truncation as _truncation

coeff = _truncation.coeff
weight = _truncation.weight
topn = _truncation.topn
approx_topn = _truncation.approx_topn

__all__ = ["coeff", "weight", "topn", "approx_topn"]
