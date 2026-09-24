"""``device=`` and ``GpuPauliSum`` on the propagate surface — the CUDA path.

The Rust side is covered by ``crates/paulistrings/tests/propagate_gpu.rs``,
the differential net of the device layer against the host engine. This file
checks the Python boundary: that ``device=0``, a device list and a resident
``GpuPauliSum`` agree with the host ``propagate`` under every lowerable policy,
that the placement kwargs are mutually exclusive, that the unsupported requests
raise the documented exceptions, and that the stats record names the devices.
A device list may repeat an ordinal, which is how the multi-device path runs on
a one-GPU box.

Everything that needs a device is skipped when ``cuda_available()`` is false;
the two tests at the top run on every build. Build and run with::

    maturin develop --release --features cuda -m crates/paulistrings-py/Cargo.toml
    pytest python/paulistrings/tests/test_cuda.py
"""

import numpy as np
import pytest

import paulistrings
from paulistrings import Circuit, GpuPauliSum, PauliSum, truncation

needs_cuda = pytest.mark.skipif(
    not paulistrings.cuda_available(),
    reason="no CUDA device, or built without the cuda feature; "
    "rebuild with `maturin develop --release --features cuda`",
)

# One per width band the bindings monomorphize: 8 -> W=1, 68 -> W=2, 130 -> W=4.
WIDTHS = [8, 68, 130]

# Coefficients agree to floating point, not bit for bit: the device sums equal
# keys in a different order (ARCHITECTURE.md §Determinism).
TOL = 1e-9

NUM_TERMS = 4_000

# Qubits the circuit touches; the observable's support spans every word.
_WINDOW = 6


def _observable(num_qubits, terms=NUM_TERMS, seed=20260923):
    """A seeded random sum: keys spread over every word, complex coefficients."""
    rng = np.random.default_rng(seed + num_qubits)
    words = (num_qubits + 63) // 64
    x = rng.integers(0, 1 << 63, size=(terms, words), dtype=np.uint64)
    z = rng.integers(0, 1 << 63, size=(terms, words), dtype=np.uint64)
    tail = num_qubits - 64 * (words - 1)
    last = np.uint64((1 << tail) - 1) if tail < 64 else np.uint64((1 << 64) - 1)
    x[:, words - 1] &= last
    z[:, words - 1] &= last
    coeffs = rng.normal(size=terms) + 1j * rng.normal(size=terms)
    return PauliSum.from_arrays(x, z, coeffs, num_qubits=num_qubits)


def _circuit(num_qubits):
    """Generic-angle rotations, a multi-qubit generator, Cliffords, a dense
    two-qubit unitary and noise: every prepared-table shape the device runs."""
    circuit = Circuit(num_qubits)
    for q, angle in enumerate((0.37, 0.81, 1.13, 0.59)):
        getattr(circuit, ("rz", "rx", "ry", "rz")[q])(angle, q)
    circuit.pauli_rotation("XYZ", [0, 2, 4], 0.43)
    circuit.cnot(0, 1)
    circuit.h(3)
    theta = 0.7
    c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
    xx = np.array(
        [[c, 0, 0, s], [0, c, s, 0], [0, s, c, 0], [s, 0, 0, c]], dtype=np.complex128
    )
    circuit.unitary_2q(2, _WINDOW - 1, xx)
    circuit.depolarize(0.02, [3])
    circuit.amplitude_damping(0.05, [1])
    return circuit


def _sorted_arrays(sum_):
    """``(x, z, coefficients)`` sorted by key, so two sums compare regardless
    of bucket count or storage order."""
    xs, zs, cs = sum_.x_array(), sum_.z_array(), sum_.coefficients_array()
    order = np.lexsort(np.hstack([xs, zs]).T[::-1])
    return xs[order], zs[order], cs[order]


def _assert_terms_close(got, want, tol=TOL):
    assert got.num_qubits == want.num_qubits
    assert len(got) == len(want)
    gx, gz, gc = _sorted_arrays(got)
    wx, wz, wc = _sorted_arrays(want)
    assert np.array_equal(gx, wx) and np.array_equal(gz, wz), "different key sets"
    assert np.allclose(gc, wc, rtol=0, atol=tol)


# --------------------------------------------------------------------------
# Every build


def test_cuda_available_is_a_bool():
    assert isinstance(paulistrings.cuda_available(), bool)
    assert "GpuPauliSum" in paulistrings.__all__


@pytest.mark.skipif(paulistrings.cuda_available(), reason="a CUDA device is available")
def test_device_without_cuda_is_a_runtime_error():
    """Without the feature, or with it but no device, a device request is a
    ``RuntimeError`` rather than a silent host run."""
    s, c = _observable(8, terms=16), _circuit(8)
    with pytest.raises(RuntimeError, match="CUDA"):
        s.propagate(c, device=0)
    with pytest.raises(RuntimeError, match="CUDA"):
        s.propagate_with_stats(c, device=0)
    with pytest.raises(RuntimeError, match="CUDA"):
        s.to_device()


# --------------------------------------------------------------------------
# device= against the host engine


@needs_cuda
@pytest.mark.parametrize("num_qubits", WIDTHS)
@pytest.mark.parametrize("direction", ["forward", "heisenberg"])
def test_device_matches_host(num_qubits, direction):
    s, c = _observable(num_qubits), _circuit(num_qubits)
    want = s.propagate(c, direction=direction)
    got = s.propagate(c, direction=direction, device=0)
    assert len(got) > len(s), "the circuit must actually fan out"
    _assert_terms_close(got, want)


@needs_cuda
def test_device_leaves_the_input_untouched():
    s, c = _observable(8), _circuit(8)
    before = _sorted_arrays(s)
    s.propagate(c, device=0)
    after = _sorted_arrays(s)
    for a, b in zip(before, after):
        assert np.array_equal(a, b)


@needs_cuda
def test_bucket_knobs_pass_through():
    s, c = _observable(68), _circuit(68)
    want = s.propagate(c, direction="heisenberg")
    got = s.propagate(c, direction="heisenberg", device=0, target_bucket_len=64, min_buckets=4)
    _assert_terms_close(got, want)


# A random key has weight about 3n/4, so these weight cuts bite at every width.
POLICIES = {
    "coeff": lambda n: truncation.coeff(0.05),
    "weight": lambda n: truncation.weight(n * 7 // 10),
    "approx_topn": lambda n: truncation.approx_topn(3_000),
    "coeff & approx_topn": lambda n: truncation.coeff(0.01) & truncation.approx_topn(5_000),
    "weight | coeff": lambda n: truncation.weight(n * 6 // 10) | truncation.coeff(0.5),
}


@needs_cuda
@pytest.mark.parametrize("name", list(POLICIES))
@pytest.mark.parametrize("num_qubits", [8, 68])
def test_policies_match_host_term_for_term(name, num_qubits):
    """Every Python policy lowers to the device, and the retained set is the
    host's exactly — the lowering is not silently ``Keep``."""
    policy = POLICIES[name](num_qubits)
    s, c = _observable(num_qubits), _circuit(num_qubits)
    untruncated = s.propagate(c, direction="heisenberg")
    want = s.propagate(c, policy, direction="heisenberg")
    got = s.propagate(c, policy, direction="heisenberg", device=0)
    assert len(want) < len(untruncated), f"{name} must actually truncate"
    _assert_terms_close(got, want)


# --------------------------------------------------------------------------
# The resident sum


@needs_cuda
@pytest.mark.parametrize("num_qubits", [8, 68])
def test_resident_sum_steps_like_two_host_calls(num_qubits):
    s, c = _observable(num_qubits, terms=500), _circuit(num_qubits)
    policy = truncation.coeff(1e-6)
    want = s.propagate(c, policy, direction="heisenberg").propagate(
        c, policy, direction="heisenberg"
    )

    resident = s.to_device(0)
    assert resident.propagate(c, policy, direction="heisenberg") is None
    resident.propagate(c, policy, direction="heisenberg")
    got = resident.to_host()
    _assert_terms_close(got, want)
    # `to_host` copies: the resident sum is still there and still the same.
    _assert_terms_close(resident.to_host(), want)
    assert len(resident) == len(want)


@needs_cuda
def test_resident_sum_surface():
    s = _observable(68, terms=100)
    resident = s.to_device()
    assert isinstance(resident, GpuPauliSum)
    assert len(resident) == len(s)
    assert resident.num_qubits == 68
    assert resident.device == 0
    assert resident.num_buckets >= 1
    assert repr(resident) == f"GpuPauliSum(num_qubits=68, terms={len(s)}, device=0)"
    _assert_terms_close(resident.to_host(), s)
    with pytest.raises(TypeError):
        GpuPauliSum()


@needs_cuda
def test_resident_stats_cover_this_call_only():
    s, c = _observable(8, terms=200), _circuit(8)
    resident = s.to_device()
    resident.propagate(c)
    stats = resident.propagate_with_stats(c)
    assert stats.layers == len(c)
    assert stats.final_terms == len(resident)
    assert stats.partition.devices == [0]
    assert stats.partition.partitions == 1


@needs_cuda
def test_resident_width_mismatch_is_the_host_error():
    resident = _observable(8, terms=16).to_device()
    with pytest.raises(ValueError, match="num_qubits"):
        resident.propagate(Circuit(9))
    with pytest.raises(ValueError, match="num_qubits"):
        _observable(8, terms=16).propagate(Circuit(9), device=0)


@needs_cuda
def test_resident_exact_topn_is_not_implemented():
    resident = _observable(8, terms=16).to_device()
    with pytest.raises(NotImplementedError, match="approx_topn"):
        resident.propagate(_circuit(8), truncation.topn(10))


@needs_cuda
def test_to_device_rejects_a_bad_ordinal():
    s = _observable(8, terms=16)
    with pytest.raises(ValueError):
        s.to_device(-1)
    with pytest.raises(ValueError, match="CUDA device"):
        s.to_device(1 << 20)


# --------------------------------------------------------------------------
# Stats


@needs_cuda
def test_stats_name_the_device():
    s, c = _observable(68), _circuit(68)
    want, host_stats = s.propagate_with_stats(c, direction="heisenberg")
    got, stats = s.propagate_with_stats(c, direction="heisenberg", device=0)
    _assert_terms_close(got, want)
    assert stats.layers == len(c)
    assert stats.terms_in == host_stats.terms_in
    assert stats.terms_out == host_stats.terms_out
    assert stats.final_terms == len(got)
    part = stats.partition
    assert part.partitions == 1
    assert part.devices == [0]
    assert part.rank is None and part.size is None
    assert all(len(row) == 1 for row in part.terms_in)
    assert part.local == [True] * len(c)
    assert "devices=[0]" in repr(part)
    assert host_stats.partition is None


@needs_cuda
def test_zero_layer_circuit_on_device():
    s = _observable(8, terms=64)
    got, stats = s.propagate_with_stats(Circuit(8), device=0)
    _assert_terms_close(got, s)
    assert stats.layers == 0
    assert stats.partition.partitions == 1
    assert stats.partition.devices == [0]


# --------------------------------------------------------------------------
# Errors


@needs_cuda
def test_exact_topn_is_not_implemented():
    s, c = _observable(8, terms=64), _circuit(8)
    with pytest.raises(NotImplementedError, match="device=0"):
        s.propagate(c, truncation.topn(10), device=0)
    with pytest.raises(NotImplementedError, match="topn"):
        s.propagate(c, truncation.coeff(0.1) | truncation.topn(10), device=0)


@needs_cuda
def test_device_conflicts_are_value_errors():
    s, c = _observable(8, terms=64), _circuit(8)
    with pytest.raises(ValueError, match="alternatives"):
        s.propagate(c, device=0, partitions=2)
    with pytest.raises(ValueError, match="alternatives"):
        s.propagate(c, device=[0, 0], partitions=2)
    with pytest.raises(ValueError, match="local"):
        s.propagate(c, device=0, result="local")
    with pytest.raises(ValueError, match="local"):
        s.propagate(c, device=[0, 0], result="local")
    with pytest.raises(ValueError, match="partition_row_blocks"):
        s.propagate(c, device=0, partition_row_blocks=[[0, 1, 2, 3], [4, 5, 6, 7]])


@pytest.mark.skipif(paulistrings.mpi_available(), reason="built with the mpi feature")
def test_device_with_comm_names_the_missing_feature():
    s, c = _observable(8, terms=64), _circuit(8)
    with pytest.raises(RuntimeError, match="mpi feature"):
        s.propagate(c, device=0, comm=object())


@needs_cuda
def test_device_spellings():
    s, c = _observable(8, terms=64), _circuit(8)
    want = s.propagate(c)
    _assert_terms_close(s.propagate(c, device=[0]), want)
    with pytest.raises(TypeError, match="bool"):
        s.propagate(c, device=True)
    with pytest.raises(ValueError, match="auto"):
        s.propagate(c, device="gpu")
    with pytest.raises(ValueError):
        s.propagate(c, device=-1)
    with pytest.raises(ValueError):
        s.propagate(c, device=[])
    with pytest.raises(TypeError):
        s.propagate(c, device=1.5)


def _auto():
    """``device="auto"``'s stats on a small run, and the host answer."""
    s, c = _observable(8, terms=64), _circuit(8)
    got, stats = s.propagate_with_stats(c, device="auto")
    _assert_terms_close(got, s.propagate(c))
    return stats


@needs_cuda
def test_auto_on_one_device_is_device_zero():
    stats = _auto()
    if len(stats.partition.devices) > 1:
        pytest.skip("more than one CUDA device is visible")
    assert stats.partition.devices == [0]
    assert stats.partition.partitions == 1


@needs_cuda
def test_auto_on_several_devices_takes_a_power_of_two():
    stats = _auto()
    devices = stats.partition.devices
    if len(devices) == 1:
        pytest.skip("only one CUDA device is visible")
    k = len(devices)
    assert k & (k - 1) == 0
    assert devices == list(range(k))
    assert stats.partition.partitions == k


# --------------------------------------------------------------------------
# Several device partitions: one per list entry, repeats sharing a device

MULTI = [[0, 0], [0, 0, 0, 0]]


def _haar_4(rng):
    """A Haar-random two-qubit unitary: the QR of a complex Gaussian, phases fixed."""
    m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    q, r = np.linalg.qr(m)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _dense_circuit(num_qubits, layers=3, seed=20260923):
    """Brickwork layers of Haar-random two-qubit gates: every delta is a dense PTM row, so rows cross partitions on most gates."""
    rng = np.random.default_rng(seed)
    circuit = Circuit(num_qubits)
    for layer in range(layers):
        for q in range(layer % 2, num_qubits - 1, 2):
            circuit.unitary_2q(q, q + 1, _haar_4(rng))
    return circuit


def _trotter_circuit(num_qubits, steps=3, dt=0.1, j=1.0, h=0.8):
    """First-order Trotter steps of the transverse-field Ising chain: ``ZZ`` rotations on every bond, then ``X`` on every site."""
    circuit = Circuit(num_qubits)
    for _ in range(steps):
        for q in range(num_qubits - 1):
            circuit.pauli_rotation("ZZ", [q, q + 1], 2 * j * dt)
        for q in range(num_qubits):
            circuit.rx(2 * h * dt, q)
    return circuit


def _real_observable(num_qubits, terms, seed=20260924):
    s = _observable(num_qubits, terms=terms, seed=seed)
    return PauliSum.from_arrays(
        s.x_array(), s.z_array(), s.coefficients_array().real, num_qubits=num_qubits
    )


@needs_cuda
@pytest.mark.parametrize("devices", MULTI, ids=["2", "4"])
@pytest.mark.parametrize("direction", ["forward", "heisenberg"])
def test_device_list_matches_host_on_a_dense_circuit(devices, direction):
    s, c = _observable(8, terms=1_000), _dense_circuit(8)
    want = s.propagate(c, direction=direction)
    got = s.propagate(c, direction=direction, device=devices)
    assert len(got) > len(s), "the circuit must actually fan out"
    _assert_terms_close(got, want)


@needs_cuda
@pytest.mark.parametrize("devices", MULTI, ids=["2", "4"])
@pytest.mark.parametrize("direction", ["forward", "heisenberg"])
def test_device_list_matches_host_on_a_truncated_trotter_circuit(devices, direction):
    """``approx_topn`` all-reduces its histogram over the partitions, so the retained set is the host's exactly."""
    s, c = _real_observable(68, terms=2_000), _trotter_circuit(68)
    policy = truncation.approx_topn(3_000)
    looser = s.propagate(c, truncation.approx_topn(12_000), direction=direction)
    want = s.propagate(c, policy, direction=direction)
    got = s.propagate(c, policy, direction=direction, device=devices)
    assert len(want) < len(looser), "approx_topn must actually truncate"
    _assert_terms_close(got, want)


@needs_cuda
@pytest.mark.parametrize("devices", MULTI, ids=["2", "4"])
def test_device_list_stats_name_every_partition(devices):
    s, c = _observable(8, terms=1_000), _dense_circuit(8)
    want, host_stats = s.propagate_with_stats(c, direction="heisenberg")
    got, stats = s.propagate_with_stats(c, direction="heisenberg", device=devices)
    _assert_terms_close(got, want)
    assert stats.terms_in == host_stats.terms_in
    assert stats.terms_out == host_stats.terms_out
    part = stats.partition
    assert part.partitions == len(devices)
    assert part.devices == devices
    assert part.rank is None and part.size is None
    assert all(len(row) == len(devices) for row in part.terms_in)
    assert [sum(row) for row in part.terms_in] == stats.terms_in
    assert sum(part.rows_exported) > 0, "a dense circuit must move rows between partitions"
    assert f"devices={devices}" in repr(part)


@needs_cuda
def test_device_list_runs_twice_on_the_cached_runtime():
    s, c = _observable(8, terms=500), _dense_circuit(8, layers=2)
    want = s.propagate(c, direction="heisenberg")
    for _ in range(2):
        _assert_terms_close(s.propagate(c, direction="heisenberg", device=[0, 0]), want)


@needs_cuda
def test_device_list_zero_layer_circuit():
    s = _observable(8, terms=64)
    got, stats = s.propagate_with_stats(Circuit(8), device=[0, 0])
    _assert_terms_close(got, s)
    assert stats.layers == 0
    assert stats.partition.partitions == 2
    assert stats.partition.devices == [0, 0]


@needs_cuda
def test_a_row_seed_moves_rows_but_not_the_answer():
    s, c = _observable(8, terms=1_000), _dense_circuit(8)
    want = s.propagate(c)
    splits = []
    for seed in (1, 2):
        got, stats = s.propagate_with_stats(c, device=[0, 0], partition_row_seed=seed)
        _assert_terms_close(got, want)
        splits.append(stats.partition.terms_in[0])
    assert splits[0] != splits[1], "two seeds drew the same partition rows"


@needs_cuda
def test_partition_row_blocks_cut_the_device_partitions():
    """A cut labels a key by the XOR of the blocks it has odd Z-weight in, so a single ``Z`` on qubit ``q`` lands on the partition whose block holds ``q``; ``rz`` commutes with every term, so the split is visible in the first layer's counts."""
    terms = {"I" * q + "Z" + "I" * (7 - q): 1.0 + q for q in range(8)}
    s = PauliSum.from_strings(terms, num_qubits=8)
    c = Circuit(8)
    c.rz(0.3, 0)
    got, stats = s.propagate_with_stats(c, device=[0, 0], partition_row_blocks=[[0], list(range(1, 8))])
    _assert_terms_close(got, s.propagate(c))
    assert stats.partition.terms_in[0] == [1, 7]


@needs_cuda
@pytest.mark.parametrize("direction", ["forward", "heisenberg"])
def test_partition_row_blocks_match_host_on_a_dense_circuit(direction):
    s, c = _observable(8, terms=1_000), _dense_circuit(8)
    want = s.propagate(c, direction=direction)
    got = s.propagate(c, direction=direction, device=[0, 0], partition_row_blocks=[[0, 1, 2, 3], [4, 5, 6, 7]])
    _assert_terms_close(got, want)


@needs_cuda
def test_device_list_errors():
    s, c = _observable(8, terms=64), _circuit(8)
    with pytest.raises(ValueError, match="power of two"):
        s.propagate(c, device=[0, 0, 0])
    with pytest.raises(ValueError, match="CUDA device"):
        s.propagate(c, device=[0, 1 << 20])
    with pytest.raises(ValueError, match="partition_row_blocks"):
        s.propagate(c, device=[0, 0], partition_row_blocks=[list(range(8))])
    with pytest.raises(ValueError, match="alternatives"):
        s.propagate(c, device=[0, 0], partition_row_seed=1, partition_row_blocks=[[0], [1]])
    with pytest.raises(NotImplementedError, match="approx_topn"):
        s.propagate(c, truncation.topn(10), device=[0, 0])
