"""Python-side benchmarks comparing paulistrings against reference libraries.

Run with: pytest benchmarks/python --benchmark-only
"""
import pytest


@pytest.mark.benchmark(group="placeholder")
def test_placeholder(benchmark):
    benchmark(lambda: 1 + 1)
