"""``PauliSum.from_arrays`` and ``paulistrings.io`` (``.npz`` save/load).

Design source: ``research/notes/2026-09-01-python-api-extensions.md`` §A3.
``from_arrays`` is the inverse of ``x_array`` / ``z_array`` /
``coefficients_array``; ingest routes through the same `BuildAccumulator`
`from_strings` uses (duplicate keys sum, exact zeros drop). ``io.save`` /
``io.load`` wrap that round-trip in a small versioned ``.npz`` container so
an evolved observable can cross a process boundary (the B5 motivating flow).
"""

import math

import numpy as np
import pytest

from paulistrings import Circuit, PauliSum, io


def _as_dict(sum_):
    """{(x_words, z_words): coeff}, so comparisons do not depend on ordering."""
    xs = sum_.x_array()
    zs = sum_.z_array()
    cs = sum_.coefficients_array()
    return {
        (tuple(int(v) for v in xs[i]), tuple(int(v) for v in zs[i])): complex(cs[i])
        for i in range(len(sum_))
    }


def _assert_close(a, b, tol=1e-12):
    da, db = _as_dict(a), _as_dict(b)
    assert set(da) == set(db), f"different keys:\n{sorted(da)}\nvs\n{sorted(db)}"
    for k in da:
        assert abs(da[k] - db[k]) < tol, f"{k}: {da[k]} vs {db[k]}"


# ---- round-trip against from_strings ----


def test_from_arrays_round_trips_a_from_strings_twin_w1():
    s = PauliSum.from_strings({"XII": 1.0, "ZII": 0.5}, num_qubits=3)
    rebuilt = PauliSum.from_arrays(
        s.x_array(), s.z_array(), s.coefficients_array(), s.num_qubits
    )
    assert rebuilt.num_qubits == s.num_qubits
    _assert_close(rebuilt, s)


def test_from_arrays_round_trips_a_from_strings_twin_w2_word_boundary():
    # 65 qubits -> width 2; qubit 64 lands in word 1, bit 0.
    terms = {"I" * 64 + "X": 1.5, "Z" + "I" * 64: -2.0}
    s = PauliSum.from_strings(terms, num_qubits=65)
    assert s.width == 2
    rebuilt = PauliSum.from_arrays(
        s.x_array(), s.z_array(), s.coefficients_array(), s.num_qubits
    )
    _assert_close(rebuilt, s)


def test_from_arrays_matches_hand_built_terms():
    # Hand-derived symplectic keys: X(0) -> x=0b001=1, z=0; Z(1) -> x=0, z=0b010=2.
    x = np.array([[1], [0]], dtype=np.uint64)
    z = np.array([[0], [2]], dtype=np.uint64)
    coeffs = np.array([1.0 + 0j, 0.5 + 0j], dtype=np.complex128)
    got = PauliSum.from_arrays(x, z, coeffs, num_qubits=3)
    want = PauliSum.from_strings({"XII": 1.0, "IZI": 0.5}, num_qubits=3)
    _assert_close(got, want)


# ---- dedup and zero-dropping (BuildAccumulator semantics) ----


def test_from_arrays_sums_duplicate_rows():
    x = np.array([[1], [1]], dtype=np.uint64)
    z = np.array([[0], [0]], dtype=np.uint64)
    coeffs = np.array([1.0 + 0j, 2.5 - 1.0j], dtype=np.complex128)
    got = PauliSum.from_arrays(x, z, coeffs, num_qubits=1)
    assert len(got) == 1
    assert got.coefficients() == [3.5 - 1.0j]


def test_from_arrays_drops_exact_zero_after_summing():
    x = np.array([[1], [1]], dtype=np.uint64)
    z = np.array([[0], [0]], dtype=np.uint64)
    coeffs = np.array([1.0 + 0j, -1.0 + 0j], dtype=np.complex128)
    got = PauliSum.from_arrays(x, z, coeffs, num_qubits=1)
    assert len(got) == 0


def test_from_arrays_on_empty_arrays():
    x = np.zeros((0, 1), dtype=np.uint64)
    z = np.zeros((0, 1), dtype=np.uint64)
    coeffs = np.zeros((0,), dtype=np.complex128)
    got = PauliSum.from_arrays(x, z, coeffs, num_qubits=4)
    assert len(got) == 0
    assert got.num_qubits == 4


# ---- width-band coverage ----


@pytest.mark.parametrize("num_qubits", [3, 200])
def test_from_arrays_width_band_coverage(num_qubits):
    s_str = "I" * (num_qubits - 1) + "X"
    s = PauliSum.from_strings({s_str: 2.0}, num_qubits=num_qubits)
    rebuilt = PauliSum.from_arrays(
        s.x_array(), s.z_array(), s.coefficients_array(), s.num_qubits
    )
    _assert_close(rebuilt, s)


def test_from_arrays_narrower_array_is_zero_padded():
    # width for num_qubits=65 is 2 words; supply only 1 word (all bits fit in
    # word 0 anyway, since qubit 3 < 64) and expect it to be accepted and
    # zero-padded to width 2.
    x = np.array([[1 << 3]], dtype=np.uint64)  # X on qubit 3
    z = np.array([[0]], dtype=np.uint64)
    coeffs = np.array([1.0 + 0j], dtype=np.complex128)
    got = PauliSum.from_arrays(x, z, coeffs, num_qubits=65)
    assert got.width == 2
    want = PauliSum.from_strings({"I" * 3 + "X" + "I" * 61: 1.0}, num_qubits=65)
    _assert_close(got, want)


# ---- error cases ----


def test_from_arrays_rejects_x_z_shape_mismatch():
    x = np.zeros((2, 1), dtype=np.uint64)
    z = np.zeros((3, 1), dtype=np.uint64)
    coeffs = np.zeros((2,), dtype=np.complex128)
    with pytest.raises(ValueError, match="shape"):
        PauliSum.from_arrays(x, z, coeffs, num_qubits=4)


def test_from_arrays_rejects_coefficients_length_mismatch():
    x = np.zeros((2, 1), dtype=np.uint64)
    z = np.zeros((2, 1), dtype=np.uint64)
    coeffs = np.zeros((3,), dtype=np.complex128)
    with pytest.raises(ValueError, match="length"):
        PauliSum.from_arrays(x, z, coeffs, num_qubits=4)


def test_from_arrays_rejects_wrong_dtype_for_x():
    x = np.zeros((1, 1), dtype=np.int64)  # signed, not uint64
    z = np.zeros((1, 1), dtype=np.uint64)
    coeffs = np.zeros((1,), dtype=np.complex128)
    with pytest.raises((TypeError, ValueError)):
        PauliSum.from_arrays(x, z, coeffs, num_qubits=4)


def test_from_arrays_rejects_wrong_dtype_for_coefficients():
    x = np.zeros((1, 1), dtype=np.uint64)
    z = np.zeros((1, 1), dtype=np.uint64)
    coeffs = np.zeros((1,), dtype=np.int32)
    with pytest.raises(TypeError):
        PauliSum.from_arrays(x, z, coeffs, num_qubits=4)


def test_from_arrays_accepts_real_float_coefficients():
    x = np.array([[1]], dtype=np.uint64)
    z = np.array([[0]], dtype=np.uint64)
    coeffs = np.array([2.0], dtype=np.float64)
    got = PauliSum.from_arrays(x, z, coeffs, num_qubits=1)
    assert got.coefficients() == [2.0 + 0j]


def test_from_arrays_rejects_bit_beyond_num_qubits():
    # Bit 3 set, but num_qubits=3 only allows qubits 0..2.
    x = np.array([[1 << 3]], dtype=np.uint64)
    z = np.array([[0]], dtype=np.uint64)
    coeffs = np.array([1.0 + 0j], dtype=np.complex128)
    with pytest.raises(ValueError, match="qubit"):
        PauliSum.from_arrays(x, z, coeffs, num_qubits=3)


def test_from_arrays_rejects_z_bit_beyond_num_qubits():
    x = np.array([[0]], dtype=np.uint64)
    z = np.array([[1 << 3]], dtype=np.uint64)
    coeffs = np.array([1.0 + 0j], dtype=np.complex128)
    with pytest.raises(ValueError, match="qubit"):
        PauliSum.from_arrays(x, z, coeffs, num_qubits=3)


def test_from_arrays_rejects_num_qubits_above_1024():
    x = np.zeros((0, 1), dtype=np.uint64)
    z = np.zeros((0, 1), dtype=np.uint64)
    coeffs = np.zeros((0,), dtype=np.complex128)
    with pytest.raises(ValueError, match="1024"):
        PauliSum.from_arrays(x, z, coeffs, num_qubits=1025)


def test_from_arrays_rejects_array_wider_than_band_width():
    # num_qubits=3 picks width 1; a 2-word array is too wide to be a
    # zero-padding case even though the extra word is all zero.
    x = np.zeros((1, 2), dtype=np.uint64)
    z = np.zeros((1, 2), dtype=np.uint64)
    coeffs = np.array([1.0 + 0j], dtype=np.complex128)
    with pytest.raises(ValueError, match="width"):
        PauliSum.from_arrays(x, z, coeffs, num_qubits=3)


# ---- paulistrings.io: .npz save/load ----


def test_io_save_load_round_trip(tmp_path):
    s = PauliSum.from_strings({"XII": 1.0, "ZII": 0.5, "YII": -0.25j}, num_qubits=3)
    path = tmp_path / "sum.npz"
    io.save(path, s)
    loaded = io.load(path)
    assert loaded.num_qubits == s.num_qubits
    _assert_close(loaded, s)


def test_io_save_load_round_trip_w2(tmp_path):
    s = PauliSum.from_strings({"I" * 64 + "X": 1.0}, num_qubits=65)
    path = tmp_path / "sum_w2.npz"
    io.save(path, s)
    loaded = io.load(path)
    assert loaded.width == 2
    _assert_close(loaded, s)


def test_io_load_accepts_str_path(tmp_path):
    s = PauliSum.from_strings({"X": 1.0}, num_qubits=1)
    path = tmp_path / "sum.npz"
    io.save(str(path), s)
    loaded = io.load(str(path))
    _assert_close(loaded, s)


def test_io_load_rejects_missing_format_field(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        num_qubits=np.int64(1),
        x=np.zeros((0, 1), dtype=np.uint64),
        z=np.zeros((0, 1), dtype=np.uint64),
        coefficients=np.zeros((0,), dtype=np.complex128),
    )
    with pytest.raises(ValueError, match="format"):
        io.load(path)


def test_io_load_rejects_unknown_format_version(tmp_path):
    path = tmp_path / "future.npz"
    np.savez_compressed(
        path,
        format="paulistrings-npz-v2",
        num_qubits=np.int64(1),
        x=np.zeros((0, 1), dtype=np.uint64),
        z=np.zeros((0, 1), dtype=np.uint64),
        coefficients=np.zeros((0,), dtype=np.complex128),
    )
    with pytest.raises(ValueError, match="format"):
        io.load(path)


def test_io_save_stamps_the_current_format_version():
    assert io.FORMAT == "paulistrings-npz-v1"


# ---- B5 motivating flow: propagate -> save -> load -> propagate further ----


def test_b5_save_load_mid_propagation_matches_doing_it_in_one_go(tmp_path):
    num_qubits = 4
    initial = PauliSum.from_strings({"XIII": 1.0, "ZIII": 0.3}, num_qubits=num_qubits)

    first_half = Circuit(num_qubits)
    first_half.rz(0.37, 0)
    first_half.cnot(0, 1)
    first_half.h(1)

    second_half = Circuit(num_qubits)
    second_half.rx(0.71, 1)
    second_half.cnot(1, 2)
    second_half.rz(0.13, 2)

    # One go: propagate through both halves back-to-back.
    one_go = initial.propagate(first_half).propagate(second_half)

    # Split across a file boundary: propagate the first half, save, load,
    # then propagate the second half on the reloaded sum.
    mid = initial.propagate(first_half)
    path = tmp_path / "mid.npz"
    io.save(path, mid)
    reloaded = io.load(path)
    split = reloaded.propagate(second_half)

    _assert_close(split, one_go)
