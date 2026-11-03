"""Coefficient dtypes: `float64`, `float32`, `complex128` (the default) and `complex64`."""

import numpy as np
import pytest

import paulistrings
from paulistrings import Circuit, PauliSum, gates, noise, p, truncation

from ._evolved import propagated

DTYPES = [np.float64, np.float32, np.complex128, np.complex64]
REAL = [np.float64, np.float32]


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
    for gate in (
        gates.h(0),
        gates.cnot(0, 1),
        gates.rx(0.3, 2),
        gates.ry(-0.7, 3),
        gates.rz(1.1, 1),
        gates.pauli_rotation("XYZ", [0, 2, 3], 0.4),
        gates.unitary_2q(1, 2, q),
        gates.s(3),
        noise.depolarize(0.05, 0),
        noise.dephase(0.1, 2),
        noise.amplitude_damping(0.2, 1),
        noise.pauli_channel(0.01, 0.02, 0.03, 3),
        gates.cz(2, 3),
    ):
        circuit.append(gate)
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


@pytest.mark.parametrize(
    "spelling, dtype",
    [(np.float64, np.float64), ("f8", np.float64), (float, np.float64), (complex, np.complex128)],
)
def test_dtype_takes_any_numpy_spelling(spelling, dtype):
    assert PauliSum(2, dtype=spelling).dtype == dtype


def test_an_unsupported_dtype_is_a_type_error():
    for dtype in (np.int64, np.float16, "bool", np.clongdouble):
        with pytest.raises(TypeError, match="float64, float32, complex128 or complex64"):
            PauliSum(2, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_constructors_arrays_and_scalars_carry_the_dtype(dtype):
    h = hamiltonian(dtype)
    x, z, c = hamiltonian().to_arrays()
    for built in (
        h,
        PauliSum(5, dtype=dtype),
        PauliSum.from_strings(["XX", "ZZ"], [1.0, 2.0], dtype=dtype),
        PauliSum.from_arrays(x, z, c, 4, dtype=dtype),
        PauliSum.from_arrays(x, z, c.real.astype(dtype), 4, dtype=np.float64).astype(dtype),
        h.with_hash(seed=3),
    ):
        assert built.dtype == dtype
    assert h.coefficients_array().dtype == dtype
    assert h.to_arrays()[2].dtype == dtype
    kind = scalar_type(dtype)
    assert all(type(v) is kind for v in h.coefficients())
    for value in (
        h.expectation("z+"),
        h.expectation("0101"),
        h.expectation_stabilizer(["ZIII", "IZII", "IIZI", "IIIZ"]),
        h.identity_coefficient(),
        h.overlap(h),
    ):
        assert type(value) is kind
    assert h.expectation("z+") == pytest.approx(1.125)


@pytest.mark.parametrize("dtype", REAL)
def test_a_real_dtype_refuses_every_imaginary_part(dtype):
    x, z, _ = hamiltonian().to_arrays()
    h = hamiltonian(dtype)
    refusals = [
        lambda: PauliSum.from_strings({"XX": 1.0 + 0.5j}, dtype=dtype),
        lambda: PauliSum.from_strings(["XX"], [0.5j], dtype=dtype),
        lambda: PauliSum.from_arrays(x, z, np.full(len(x), 1j), 4, dtype=dtype),
        lambda: PauliSum.from_strings({"XX": 1.0, "ZZ": 0.5j}).astype(dtype),
        lambda: h * 1j,
    ]
    for refusal in refusals:
        with pytest.raises(ValueError, match="imaginary"):
            refusal()
    assert PauliSum.from_strings({"XX": 1.0 + 0j}, dtype=dtype).coefficients() == [1.0]
    assert (h * (2 + 0j)).dtype == dtype


@pytest.mark.parametrize("target", DTYPES)
def test_astype_round_trips(target):
    imaginary = 0.25j if np.dtype(target).kind == "c" else 0
    h = PauliSum.from_strings({"XX": 1.0, "ZZ": 0.5 + imaginary})
    converted = h.astype(target)
    assert converted.dtype == target and h.dtype == np.complex128
    assert as_dict(converted.astype(np.complex128)) == pytest.approx(as_dict(h), rel=rtol_of(target))


@pytest.mark.parametrize(
    "a, b",
    [(np.float32, np.float64), (np.float32, np.complex64), (np.float64, np.complex64), (np.complex64, np.complex128)],
)
def test_binary_operations_promote_as_numpy_does(a, b):
    result = np.result_type(a, b)
    x, y = hamiltonian(a), hamiltonian(b) * 2.0
    for combined in (x + y, y - x):
        assert combined.dtype == result
    assert as_dict(x + y) == pytest.approx(as_dict(hamiltonian() * 3.0), rel=1e-6)
    assert type(x.overlap(y)) is scalar_type(result)
    x += y
    assert x.dtype == result


@pytest.mark.parametrize("dtype", DTYPES)
def test_every_dtype_propagates_a_hermitian_observable_like_complex128(dtype):
    circuit = noisy_circuit()
    policy = truncation.coeff(1e-9)
    reference = hamiltonian()
    reference.propagate(circuit, policy, direction="heisenberg")
    evolved = hamiltonian(dtype)
    evolved.propagate(circuit, policy, direction="heisenberg")
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
def test_every_propagation_path_and_save_keep_the_dtype(dtype, tmp_path):
    circuit = noisy_circuit()
    policy = truncation.weight(2) & truncation.approx_topn(20)
    assert propagated(hamiltonian(dtype), circuit, policy).dtype == dtype
    evolved = hamiltonian(dtype)
    stats = evolved.propagate_with_stats(circuit)
    assert evolved.dtype == dtype and stats.final_terms == len(evolved)
    assert propagated(hamiltonian(dtype), circuit, partitions=[[0], [1]]).dtype == dtype

    split = hamiltonian(dtype).partition(partitions=[[0]])
    assert split.dtype == dtype
    split.propagate(circuit, direction="heisenberg")
    value = split.expectation("z+")
    assert type(value) is scalar_type(dtype)
    reference = hamiltonian()
    reference.propagate(circuit, direction="heisenberg")
    assert value == pytest.approx(reference.expectation("z+").real, rel=rtol_of(dtype))
    assert split.gather().dtype == dtype and split.local().dtype == dtype

    path = tmp_path / "h.npz"
    paulistrings.io.save(path, hamiltonian(dtype))
    loaded = paulistrings.io.load(path)
    assert loaded.dtype == dtype and as_dict(loaded) == as_dict(hamiltonian(dtype))
