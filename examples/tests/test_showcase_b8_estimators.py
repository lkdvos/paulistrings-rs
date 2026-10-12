"""B8 estimators in `ole.py` against hand values, a materialized rotation, and their own MPI split."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b8_ole"))

import ole  # noqa: E402

ps = pytest.importorskip("paulistrings")

DELTA = 0.3


def _sum(terms, num_qubits=None):
    return ps.PauliSum.from_strings({label: complex(c) for label, c in terms}, num_qubits=num_qubits)


def _rand_sum_on(m: int, num_qubits: int, window, seed: int):
    """`m` random strings supported on `window`, complex coefficients."""
    rng = np.random.default_rng(seed)
    terms = {}
    for _ in range(m):
        label = ["I"] * num_qubits
        for q in window:
            label[q] = "IXYZ"[rng.integers(4)]
        key = "".join(label)
        terms[key] = terms.get(key, 0) + complex(*rng.normal(size=2))
    return ps.PauliSum.from_strings(terms, num_qubits=num_qubits)


def _materialized(A, sites, delta, axis):
    """`2^-n Tr(A^dagger V^dagger A V)` by brute force: `V^dagger A V` propagated, then `overlap`."""
    circuit = ps.Circuit(A.num_qubits)
    for q in sites:
        (circuit.rx if axis == "x" else circuit.rz)(2.0 * delta, q)
    return A.overlap(A.propagate(circuit, direction="heisenberg"))


@pytest.mark.parametrize(
    "label,axis,site,want",
    [
        ("X", "z", 0, "c"),
        ("Y", "z", 0, "c"),
        ("Z", "z", 0, 1.0),
        ("Z", "x", 0, "c"),
        ("Y", "x", 0, "c"),
        ("X", "x", 0, 1.0),
        # A generator off the string's support is invisible.
        ("XI", "z", 1, 1.0),
    ],
)
def test_single_qubit_hand_values(label, axis, site, want):
    want = math.cos(2 * DELTA) if want == "c" else want
    assert ole.rotated_overlap(_sum([(label, 1.0)]), [site], DELTA, axis) == pytest.approx(want, abs=1e-15)


def test_two_qubit_class_hand_values():
    """`XX` and `YY` form one class on sites `{0, 1}` (Z axis), with `R[XX, YY] = sin^2 2 delta`; on the X axis `ZZ`/`YY` and `ZY`/`YZ` pair up."""
    s, c = math.sin(2 * DELTA), math.cos(2 * DELTA)
    cases = [
        ([("XX", 0.7), ("YY", -0.4)], "z", 0.65 * c * c - 0.56 * s * s),
        ([("XX", 1.0), ("YY", 1.0)], "z", 2.0),
        ([("XY", 1.0), ("YX", -1.0)], "z", 2.0),
        ([("XX", 1.0), ("YY", -1.0)], "z", 2.0 * math.cos(4 * DELTA)),
        ([("ZZ", 1.0), ("YY", 1.0)], "x", 2.0),
        ([("ZY", 1.0), ("YZ", -1.0)], "x", 2.0),
        ([("ZY", 1.0), ("YZ", 1.0)], "x", 2.0 * math.cos(4 * DELTA)),
    ]
    for terms, axis, want in cases:
        got = ole.rotated_overlap(_sum(terms), [0, 1], DELTA, axis)
        assert got == pytest.approx(want, abs=1e-14), (terms, axis)


@pytest.mark.parametrize("num_qubits,window", [(8, [1, 2, 3, 4, 6]), (70, [62, 63, 64, 65, 67])])
def test_agrees_with_materializing_the_rotation(num_qubits, window):
    """Complex coefficients, so the sign of every off-diagonal entry shows in the imaginary part too."""
    for seed, sites in [(1, [window[0], window[2]]), (2, window), (3, [window[1]])]:
        A = _rand_sum_on(300, num_qubits, window, seed)
        for axis in "zx":
            want = _materialized(A, sites, DELTA, axis)
            got = ole.rotated_overlap_complex(A, sites, DELTA, axis)
            assert abs(got - want) < 1e-10 * max(abs(want), 1.0), (seed, axis, sites)


def test_histogram_hand_values_and_the_diagonal_limit():
    """With no class of two, the overlap is the histogram's diagonal sum: `a X0 + b Z0 + c Y0Y1 + d I`."""
    a, b, c, d = 0.5, -0.25, 0.75, 0.125
    A = _sum([("XI", a), ("ZI", b), ("YY", c), ("II", d)])
    np.testing.assert_array_equal(ole.anticommute_histogram(A, [0, 1], "z"), [b * b + d * d, a * a, c * c])
    np.testing.assert_array_equal(ole.anticommute_histogram(A, [0, 1], "x"), [a * a + d * d, b * b, c * c])
    for axis in "zx":
        w = ole.anticommute_histogram(A, [0, 1], axis)
        diagonal = ole.diagonal_echo(w, DELTA) * w.sum()
        assert diagonal == pytest.approx(ole.rotated_overlap(A, [0, 1], DELTA, axis), abs=1e-15)
    # X_q under Z rotations keeps cos 2 delta of itself.
    w = ole.anticommute_histogram(_sum([("X", 1.0)]), [0], "z")
    assert ole.diagonal_echo(w, DELTA) == pytest.approx(math.cos(2 * DELTA), abs=1e-15)


def test_diagonal_echo_resums_the_moment_series():
    hist = [0.1, 0.2, 0.3, 0.25, 0.15]
    for order in (2, 4, 8):
        assert ole.truncated_series(hist, 0.3, order) != pytest.approx(ole.diagonal_echo(hist, 0.3), abs=1e-12)
    assert ole.truncated_series(hist, 0.3, 24) == pytest.approx(ole.diagonal_echo(hist, 0.3), abs=1e-12)
    # Full scrambling: n_P ~ Binomial(35, 1/2), App. C 1 c.
    full = [math.comb(35, n) / 2**35 for n in range(36)]
    assert ole.diagonal_echo(full, 0.3) == pytest.approx(ole.spec()["references"]["full_scrambling"]["value"], abs=1e-4)


def _split(A, keep):
    x, z, c = A.to_arrays()
    return [ps.PauliSum.from_arrays(x[rows], z[rows], c[rows], A.num_qubits) for rows in (keep, ~keep)]


@pytest.mark.parametrize("axis", "xz")
def test_shares_that_keep_classes_whole_add_up(axis):
    """What `comm=` relies on: split by a bit `V` never flips, each share's values add to the whole sum's."""
    A = _rand_sum_on(2000, 70, [0, 5, 63, 64, 66, 68], 9)
    sites = [5, 63, 64]
    x, z, _ = A.to_arrays()
    # The z-bit of qubit 64 under the X axis, the x-bit of qubit 0 under the Z axis.
    keep = (z[:, 1] & 1).astype(bool) if axis == "x" else (x[:, 0] & 1).astype(bool)
    shares = _split(A, keep)
    assert all(len(s) for s in shares)
    hist = sum(ole.anticommute_histogram(s, sites, axis) for s in shares)
    np.testing.assert_allclose(hist, ole.anticommute_histogram(A, sites, axis), rtol=1e-12)
    total = sum(ole.rotated_overlap(s, sites, DELTA, axis) for s in shares)
    assert total == pytest.approx(ole.rotated_overlap(A, sites, DELTA, axis), rel=1e-12)


def test_a_class_split_across_shares_does_not_add_up():
    """The converse: split on a flipped bit and the cross-share pairs are lost."""
    A = _sum([("XX", 1.0), ("YY", -1.0)])
    x, z, _ = A.to_arrays()
    shares = _split(A, (z[:, 0] & 1).astype(bool))
    total = sum(ole.rotated_overlap(s, [0, 1], DELTA, "z") for s in shares)
    assert total != pytest.approx(ole.rotated_overlap(A, [0, 1], DELTA, "z"), abs=1e-6)


def test_a_repeated_site_is_rejected():
    with pytest.raises(ValueError, match="listed twice"):
        ole.rotated_overlap(_sum([("XX", 1.0)]), [1, 1], DELTA, "z")


def test_an_out_of_range_site_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        ole.anticommute_histogram(_sum([("XX", 1.0)]), [2], "x")


def test_an_unknown_axis_is_rejected():
    with pytest.raises(ValueError, match="axis"):
        ole.anticommute_histogram(_sum([("XX", 1.0)]), [0], "y")


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("axis", "zx")
def test_zero_angle_is_the_squared_norm(seed, axis):
    A = _rand_sum_on(120, 8, [0, 1, 2, 5, 7], seed)
    norm = float(np.sum(np.abs(A.coefficients_array()) ** 2))
    assert ole.rotated_overlap(A, [0, 2, 5, 7], 0.0, axis) == pytest.approx(norm, rel=1e-12)


def test_empty_sum():
    A = ps.PauliSum(4)
    np.testing.assert_array_equal(ole.anticommute_histogram(A, [0, 1], "x"), [0.0, 0.0, 0.0])
    assert ole.rotated_overlap(A, [0, 1], DELTA, "x") == 0.0


def _random_case(num_qubits: int, big_class: bool):
    """4000 strings on a random 12-qubit window plus stray qubits; with `big_class`, every other string holds X or Y on all 12 sites, one class of up to 2^12 under the Z axis."""
    rng = np.random.default_rng(num_qubits)
    window = sorted(rng.choice(num_qubits, 12, replace=False).tolist())
    sites = window if big_class else window[::2]
    terms = {}
    for i in range(4000):
        label = ["I"] * num_qubits
        for q in window:
            label[q] = rng.choice(list("XY") if big_class and i % 2 == 0 else list("IXYZ"))
        for q in rng.choice(num_qubits, 3):
            if q not in window:
                label[q] = "IXYZ"[rng.integers(4)]
        key = "".join(label)
        terms[key] = terms.get(key, 0) + complex(*rng.normal(size=2))
    return ps.PauliSum.from_strings(terms, num_qubits=num_qubits), sites


# `(num_qubits, big_class, axis, rotated_overlap, sum_n n w_n)` at delta 0.3, as the library's Rust estimators computed them.
RUST_VALUES = [
    (12, False, "z", 4593.332499260803, 23424.086658087588),
    (12, False, "x", 4552.698258155332, 23762.99327141203),
    (12, True, "z", 1748.7503390208758, 71269.42342594643),
    (12, True, "x", 2637.9523350704894, 47738.33611948646),
    (70, False, "z", 4787.217502779087, 24439.339328955928),
    (70, False, "x", 4727.985585555442, 24996.188050514065),
    (70, True, "z", 1781.8993596760563, 74297.55268337179),
    (70, True, "x", 2754.0481047850994, 49365.4388520283),
    (130, False, "z", 4713.750970390883, 24215.094505571265),
    (130, False, "x", 4657.059120751598, 24670.228702687),
    (130, True, "z", 1779.1405847570218, 72880.09910443037),
    (130, True, "x", 2673.2612932112606, 49248.75633471048),
    (300, False, "z", 4663.912397229757, 24551.341283166214),
    (300, False, "x", 4638.390221454131, 24752.86235025574),
    (300, True, "z", 1747.6534948096407, 73146.21809151606),
    (300, True, "x", 2684.144715617881, 48832.90216697548),
]


@pytest.mark.parametrize("num_qubits,big_class,axis,overlap,first_moment", RUST_VALUES)
def test_random_sums_reproduce_the_rust_estimators(num_qubits, big_class, axis, overlap, first_moment):
    A, sites = _random_case(num_qubits, big_class)
    hist = ole.anticommute_histogram(A, sites, axis)
    assert hist @ np.arange(len(hist)) == pytest.approx(first_moment, rel=1e-12)
    assert ole.rotated_overlap(A, sites, 0.3, axis) == pytest.approx(overlap, rel=1e-12)
    if hasattr(A, "rotated_overlap"):
        assert ole.rotated_overlap(A, sites, 0.3, axis) == pytest.approx(A.rotated_overlap(sites, 0.3, axis=axis), rel=1e-12)
        np.testing.assert_allclose(hist, A.anticommute_histogram(sites, axis=axis), rtol=1e-12, atol=1e-12 * hist.sum())
