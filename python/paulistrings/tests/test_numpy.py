"""Slice 10.5 — NumPy interop on bulk SoA storage.

Pin the array shapes, dtypes, and bit layout so downstream NumPy code can
rely on the contract.
"""

import numpy as np
import pytest

from paulistrings import PauliSum


def test_coefficients_array_shape_and_dtype():
    s = PauliSum.from_strings({"XII": 1.0, "ZII": 0.5}, num_qubits=3)
    arr = s.coefficients_array()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.complex128
    assert arr.shape == (2,)
    # Same lex ordering as .coefficients(): ZII (key (0,1)) before XII (key (1,0)).
    np.testing.assert_array_equal(arr, np.array([0.5 + 0j, 1.0 + 0j]))


def test_x_z_array_shape_for_w1():
    # 3-qubit sum → width 1.
    s = PauliSum.from_strings({"XII": 1.0, "ZII": 0.5}, num_qubits=3)
    assert s.width == 1
    x = s.x_array()
    z = s.z_array()
    assert x.shape == (2, 1)
    assert z.shape == (2, 1)
    assert x.dtype == np.uint64
    assert z.dtype == np.uint64
    # Row 0 = ZII = (x=0, z=1); row 1 = XII = (x=1, z=0).
    np.testing.assert_array_equal(x, np.array([[0], [1]], dtype=np.uint64))
    np.testing.assert_array_equal(z, np.array([[1], [0]], dtype=np.uint64))


def test_x_array_layout_for_w2_word_boundary():
    # 65 qubits → width 2; X on qubit 64 lands in word 1, bit 0.
    s_str = "I" * 64 + "X"
    s = PauliSum.from_strings({s_str: 1.0}, num_qubits=65)
    assert s.width == 2
    x = s.x_array()
    z = s.z_array()
    assert x.shape == (1, 2)
    assert z.shape == (1, 2)
    np.testing.assert_array_equal(x, np.array([[0, 1]], dtype=np.uint64))
    np.testing.assert_array_equal(z, np.array([[0, 0]], dtype=np.uint64))


def test_arrays_on_empty_sum():
    s = PauliSum(4)
    assert s.coefficients_array().shape == (0,)
    assert s.x_array().shape == (0, 1)
    assert s.z_array().shape == (0, 1)


def test_arrays_are_independent_copies():
    # Mutating the returned array must not corrupt the underlying sum.
    s = PauliSum.from_strings({"X": 2.0}, num_qubits=1)
    arr = s.coefficients_array()
    arr[0] = 99 + 0j
    # Re-read: the original coefficient must be unchanged.
    assert s.coefficients_array()[0] == 2 + 0j


def test_width_getter_matches_picked_monomorphization():
    assert PauliSum(64).width == 1
    assert PauliSum(65).width == 2
    assert PauliSum(128).width == 2
    assert PauliSum(129).width == 4
    assert PauliSum(257).width == 8
    assert PauliSum(513).width == 16
