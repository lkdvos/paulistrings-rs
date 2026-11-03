"""Coefficient dtypes: `float64`, `float32`, `complex128` (the default) and `complex64`."""

import numpy as np
import pytest

import paulistrings
from paulistrings import Circuit, PauliSum, gates, noise, p, truncation

DTYPES = [np.float64, np.float32, np.complex128, np.complex64]
REAL = [np.float64, np.float32]
COMPLEX = [np.complex128, np.complex64]


def rtol_of(dtype):
    return 1e-12 if np.dtype(dtype) in (np.float64, np.complex128) else 1e-5


def scalar_type(dtype):
    return complex if np.dtype(dtype).kind == "c" else float


def hamiltonian(dtype=None):
    kwargs = {} if dtype is None else {"dtype": dtype}
    return PauliSum.from_strings({"ZIII": 1.0, "IXXI": 0.5, "IIYZ": -0.25, "IIII": 0.125}, **kwargs)


def noisy_circuit():
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    circuit = Circuit(4)
    circuit.append(gates.h(0))
    circuit.append(gates.cnot(0, 1))
    circuit.append(gates.rx(0.3, 2))
    circuit.append(gates.ry(-0.7, 3))
    circuit.append(gates.rz(1.1, 1))
    circuit.append(gates.pauli_rotation("XYZ", [0, 2, 3], 0.4))
    circuit.append(gates.unitary_2q(1, 2, q))
    circuit.append(gates.s(3))
    circuit.append(noise.depolarize(0.05, 0))
    circuit.append(noise.dephase(0.1, 2))
    circuit.append(noise.amplitude_damping(0.2, 1))
    circuit.append(noise.pauli_channel(0.01, 0.02, 0.03, 3))
    circuit.append(gates.cz(2, 3))
    return circuit


def as_dict(pauli_sum):
    x, z, c = pauli_sum.to_arrays()
    return {(tuple(xr), tuple(zr)): complex(cv) for xr, zr, cv in zip(x, z, c)}


def test_the_default_dtype_is_complex128():
    assert PauliSum(3).dtype == np.complex128
    assert hamiltonian().dtype == np.complex128
    x, z, c = hamiltonian().to_arrays()
    assert PauliSum.from_arrays(x, z, c.real.copy(), 4).dtype == np.complex128
    assert (p("XY") * 2.0).dtype == np.complex128


@pytest.mark.parametrize("spelling", [np.float64, "float64", float, np.dtype("float64"), "f8"])
def test_dtype_takes_any_numpy_spelling(spelling):
    assert PauliSum(2, dtype=spelling).dtype == np.float64


def test_complex_spelling_is_complex128():
    assert PauliSum(2, dtype=complex).dtype == np.complex128


@pytest.mark.parametrize("dtype", [np.int64, np.float16, "bool", np.clongdouble])
def test_an_unsupported_dtype_is_a_type_error(dtype):
    with pytest.raises(TypeError, match="float64, float32, complex128 or complex64"):
        PauliSum(2, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_every_constructor_takes_dtype(dtype):
    assert PauliSum(5, dtype=dtype).dtype == dtype
    assert hamiltonian(dtype).dtype == dtype
    by_lists = PauliSum.from_strings(["XX", "ZZ"], [1.0, 2.0], dtype=dtype)
    assert by_lists.dtype == dtype
    x, z, c = hamiltonian().to_arrays()
    assert PauliSum.from_arrays(x, z, c, 4, dtype=dtype).dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_arrays_and_lists_come_back_in_the_sums_dtype(dtype):
    h = hamiltonian(dtype)
    assert h.coefficients_array().dtype == dtype
    assert h.to_arrays()[2].dtype == dtype
    assert all(type(c) is scalar_type(dtype) for c in h.coefficients())
    np.testing.assert_allclose(
        sorted(np.asarray(h.coefficients()).real), [-0.25, 0.125, 0.5, 1.0]
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_scalars_are_float_for_a_real_dtype_and_complex_otherwise(dtype):
    h = hamiltonian(dtype)
    kind = scalar_type(dtype)
    assert type(h.expectation("z+")) is kind
    assert type(h.expectation("0101")) is kind
    assert type(h.expectation_stabilizer(["ZIII", "IZII", "IIZI", "IIIZ"])) is kind
    assert type(h.identity_coefficient()) is kind
    assert type(h.overlap(h)) is kind
    assert h.expectation("z+") == pytest.approx(1.125)
    assert h.identity_coefficient() == pytest.approx(0.125)


@pytest.mark.parametrize("dtype", REAL)
def test_a_real_dtype_refuses_an_imaginary_coefficient(dtype):
    with pytest.raises(ValueError, match="imaginary"):
        PauliSum.from_strings({"XX": 1.0 + 0.5j}, dtype=dtype)
    with pytest.raises(ValueError, match="imaginary"):
        PauliSum.from_strings(["XX"], [0.5j], dtype=dtype)
    x, z, _ = hamiltonian().to_arrays()
    with pytest.raises(ValueError, match="imaginary"):
        PauliSum.from_arrays(x, z, np.full(len(x), 1j), 4, dtype=dtype)


@pytest.mark.parametrize("dtype", REAL)
def test_a_real_dtype_accepts_complex_values_with_no_imaginary_part(dtype):
    h = PauliSum.from_strings({"XX": 1.0 + 0j}, dtype=dtype)
    assert h.coefficients() == [1.0]
    x, z, c = hamiltonian().to_arrays()
    assert PauliSum.from_arrays(x, z, c, 4, dtype=dtype).dtype == dtype


@pytest.mark.parametrize("source", [np.float64, np.float32, np.complex128, np.complex64])
@pytest.mark.parametrize("dtype", DTYPES)
def test_from_arrays_reads_every_coefficient_array_dtype(source, dtype):
    x, z, c = hamiltonian().to_arrays()
    built = PauliSum.from_arrays(x, z, c.real.astype(source), 4, dtype=dtype)
    assert as_dict(built) == pytest.approx(as_dict(hamiltonian()))


@pytest.mark.parametrize("source", DTYPES)
@pytest.mark.parametrize("target", DTYPES)
def test_astype_round_trips_a_real_valued_sum(source, target):
    h = hamiltonian(source)
    converted = h.astype(target)
    assert converted.dtype == target
    assert h.dtype == source
    assert as_dict(converted) == pytest.approx(as_dict(h), rel=rtol_of(target))
    assert as_dict(converted.astype(source)) == pytest.approx(as_dict(h), rel=rtol_of(target))


@pytest.mark.parametrize("target", REAL)
def test_astype_refuses_to_drop_an_imaginary_part(target):
    h = PauliSum.from_strings({"XX": 1.0, "ZZ": 0.5j})
    with pytest.raises(ValueError, match="imaginary"):
        h.astype(target)


def test_astype_of_complex64_keeps_the_imaginary_part():
    h = PauliSum.from_strings({"XX": 1.0, "ZZ": 0.5j})
    assert as_dict(h.astype(np.complex64)) == as_dict(h)


@pytest.mark.parametrize(
    "a, b, result",
    [
        (np.float32, np.float32, np.float32),
        (np.float32, np.float64, np.float64),
        (np.float32, np.complex64, np.complex64),
        (np.float32, np.complex128, np.complex128),
        (np.float64, np.complex64, np.complex128),
        (np.complex64, np.complex128, np.complex128),
        (np.float64, np.float64, np.float64),
    ],
)
def test_binary_operations_promote_as_numpy_does(a, b, result):
    assert np.result_type(a, b) == result
    x, y = hamiltonian(a), hamiltonian(b) * 2.0
    for combined in (x + y, y + x, x - y, y - x):
        assert combined.dtype == result
    assert as_dict(x + y) == pytest.approx(as_dict(hamiltonian() * 3.0), rel=1e-6)
    assert type(x.overlap(y)) is scalar_type(result)
    assert x.overlap(y) == pytest.approx(2.0 * hamiltonian().overlap(hamiltonian()), rel=1e-6)


@pytest.mark.parametrize("a", DTYPES)
@pytest.mark.parametrize("b", DTYPES)
def test_in_place_operations_keep_the_left_dtype(a, b):
    x = hamiltonian(a)
    x += hamiltonian(b)
    assert x.dtype == a
    x -= hamiltonian(b) * 0.5
    assert x.dtype == a
    assert as_dict(x) == pytest.approx(as_dict(hamiltonian() * 1.5), rel=1e-6)


@pytest.mark.parametrize("dtype", REAL)
def test_in_place_onto_a_real_sum_refuses_an_imaginary_part(dtype):
    x = hamiltonian(dtype)
    with pytest.raises(ValueError, match="imaginary"):
        x += PauliSum.from_strings({"XXII": 1j})
    assert as_dict(x) == as_dict(hamiltonian())


@pytest.mark.parametrize("dtype", REAL)
def test_scaling_a_real_sum_by_an_imaginary_number_raises(dtype):
    h = hamiltonian(dtype)
    with pytest.raises(ValueError, match="imaginary"):
        h * 1j
    with pytest.raises(ValueError, match="imaginary"):
        h *= 2 + 1j
    assert (h * (2 + 0j)).dtype == dtype
    assert (3 * h).dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_every_dtype_propagates_a_hermitian_observable_like_complex128(dtype):
    circuit = noisy_circuit()
    policy = truncation.coeff(1e-9)
    reference = hamiltonian().propagate(circuit, policy, direction="heisenberg")
    evolved = hamiltonian(dtype).propagate(circuit, policy, direction="heisenberg")
    assert evolved.dtype == dtype
    rtol = rtol_of(dtype)
    for state in ("z+", "x+", "0110", "+-rl"):
        assert evolved.expectation(state) == pytest.approx(
            reference.expectation(state).real, rel=rtol, abs=rtol
        )
    got, want = as_dict(evolved), as_dict(reference)
    big = {k for k, v in want.items() if abs(v) > 1e-4}
    assert big <= got.keys()
    for key in big:
        assert got[key] == pytest.approx(want[key], rel=rtol, abs=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "policy",
    [truncation.topn(20), truncation.approx_topn(20), truncation.weight(2) & truncation.coeff(1e-3)],
    ids=["topn", "approx_topn", "weight_and_coeff"],
)
def test_truncation_policies_run_on_every_dtype(dtype, policy):
    evolved = hamiltonian(dtype).propagate(noisy_circuit(), policy)
    reference = hamiltonian().propagate(noisy_circuit(), policy)
    assert evolved.dtype == dtype
    assert len(evolved) == len(reference)


@pytest.mark.parametrize("dtype", DTYPES)
def test_propagate_with_stats_keeps_the_dtype(dtype):
    evolved, stats = hamiltonian(dtype).propagate_with_stats(noisy_circuit())
    assert evolved.dtype == dtype
    assert stats.final_terms == len(evolved)


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_partitioned_sum_keeps_the_dtype(dtype):
    circuit = noisy_circuit()
    split = hamiltonian(dtype).partition(partitions=[[0]])
    assert split.dtype == dtype
    split.propagate(circuit, direction="heisenberg")
    reference = hamiltonian().propagate(circuit, direction="heisenberg")
    value = split.expectation("z+")
    assert type(value) is scalar_type(dtype)
    assert value == pytest.approx(reference.expectation("z+").real, rel=rtol_of(dtype))
    assert split.gather().dtype == dtype
    assert split.local().dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_partitioned_propagate_keeps_the_dtype(dtype):
    evolved = hamiltonian(dtype).propagate(noisy_circuit(), partitions=[[0], [1]])
    assert evolved.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_save_and_load_keep_the_dtype(tmp_path, dtype):
    path = tmp_path / "h.npz"
    paulistrings.io.save(path, hamiltonian(dtype))
    loaded = paulistrings.io.load(path)
    assert loaded.dtype == dtype
    assert as_dict(loaded) == as_dict(hamiltonian(dtype))


def test_with_hash_keeps_the_dtype():
    assert hamiltonian(np.float32).with_hash(seed=3).dtype == np.float32
