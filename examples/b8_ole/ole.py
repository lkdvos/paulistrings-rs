"""Operator Loschmidt echo on the 56-qubit ibm_boston heavy-hex patch (arXiv:2607.25998).

Circuits come from the vendored Quantum Advantage Tracker QASM; `echo_half` rebuilds the measured half for any `(L, eta)`.
Estimators, in numpy over `PauliSum.to_arrays()`: the paper's diagonal OLE from an anticommutation histogram, and the exact `rotated_overlap`.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "ole56"
SPEC_PATH = HERE / "spec.json"

_OP = re.compile(r"^(\w+)(?:\(([^)]*)\))?\s+(q\[\d+\](?:,\s*q\[\d+\])*);$")
_INVERSE = {"rx": "rx", "rz": "rz", "cz": "cz", "s": "sdg", "sdg": "s", "sx": "sxdg", "sxdg": "sx"}


@lru_cache(maxsize=1)
def spec() -> dict:
    return json.loads(SPEC_PATH.read_text())


def _angle(expr: str) -> float:
    if not re.fullmatch(r"[0-9.+\-*/ epi]+", expr):
        raise ValueError(f"unexpected angle expression {expr!r}")
    return float(eval(expr, {"__builtins__": {}}, {"pi": math.pi}))


def qasm_path(alpha: float) -> Path:
    return DATA / f"operator_loschmidt_echo_56x1488_alpha_{alpha:.2f}.qasm.gz"


@lru_cache(maxsize=None)
def qasm_ops(alpha: float) -> tuple:
    """The tracker circuit as `(name, angle | None, qubits)` in time order, device numbering."""
    ops = []
    with gzip.open(qasm_path(alpha), "rt") as fh:
        for line in fh:
            m = _OP.match(line.strip())
            if not m or m.group(1) in ("barrier", "qubit"):
                continue
            name, arg, qs = m.groups()
            qubits = tuple(int(q) for q in re.findall(r"q\[(\d+)\]", qs))
            ops.append((name, _angle(arg) if arg else None, qubits))
    return tuple(ops)


def split_echo(ops) -> tuple[list, list, list]:
    """`(first half, V_delta block, second half)`; the block is the basis change, `rz(2 delta)` layer and its undo."""
    lo = min(i for i, o in enumerate(ops) if o[0] in ("sdg", "sxdg"))
    hi = max(i for i, o in enumerate(ops) if o[0] in ("sx", "s"))
    return list(ops[:lo]), list(ops[lo : hi + 1]), list(ops[hi + 1 :])


def invert(ops) -> list:
    return [(_INVERSE[n], -a if a is not None else None, q) for n, a, q in reversed(ops)]


def eta_of_alpha(alpha: float) -> float:
    return _angle(spec()["eta_per_alpha"]) * alpha


@lru_cache(maxsize=1)
def floquet_layer() -> tuple:
    """One unscattered forward Floquet layer (3 CZ colours, 62 CZ), taken from the tracker's `alpha = 0` circuit."""
    _, _, second = split_echo(qasm_ops(0.0))
    layers = 2 * spec()["tracker_L"]
    size, rem = divmod(len(second), layers)
    assert rem == 0
    return tuple(second[:size])


def echo_half(L: int, eta: float) -> list:
    """The measured half `C` in time order: `L` scattered forward layers, then `L` unscattered inverse layers.

    `A = C^dagger O C` is what PP-MC Heisenberg-propagates, and `S_delta = 2^-n Tr(A V^dagger A V)`.
    Scattering shifts every `rx` on a scattering qubit by `-eta` (App. A 2 c).
    """
    scattering = set(spec()["scattering_qubits"])
    base = floquet_layer()
    scattered = [
        (n, a - eta if n == "rx" and q[0] in scattering else a, q) for n, a, q in base
    ]
    return scattered * L + invert(base) * L


def qubit_index() -> dict[int, int]:
    """Device qubit -> dense index `0..55` (one `u64` word)."""
    return {q: i for i, q in enumerate(spec()["device_qubits"])}


def to_circuit(ops, index: dict[int, int] | None = None):
    """A `paulistrings.Circuit`, one gate per channel, so collapse checks fall after every gate as in the paper.

    Quarter-turn `rx`/`rz` snap to exact Cliffords inside `PauliRotation` itself; nothing here branches.
    """
    from paulistrings import Circuit

    index = qubit_index() if index is None else index
    circuit = Circuit(len(index))
    for name, angle, qs in ops:
        q = [index[x] for x in qs]
        if name == "rx":
            circuit.rx(angle, q[0])
        elif name == "rz":
            circuit.rz(angle, q[0])
        elif name == "cz":
            circuit.cz(q[0], q[1])
        elif name == "s":
            circuit.s(q[0])
        elif name == "sdg":
            circuit.sdg(q[0])
        elif name == "sx":
            circuit.rx(math.pi / 2, q[0])
        elif name == "sxdg":
            circuit.rx(-math.pi / 2, q[0])
        else:
            raise ValueError(f"unsupported gate {name!r}")
    return circuit


def observable(index: dict[int, int] | None = None):
    """`O = prod_{q in V_O} Z_q` with coefficient 1."""
    sys.path.insert(0, str(HERE.parent))
    from common.observables import pauli_sum_from_support

    index = qubit_index() if index is None else index
    support = {index[q]: "Z" for q in spec()["observable_qubits"]}
    return pauli_sum_from_support(support, len(index))


def perturbation_sites(index: dict[int, int] | None = None) -> list[int]:
    index = qubit_index() if index is None else index
    return [index[q] for q in spec()["perturbation_qubits"]]


def diagonal_moments(hist, orders: int) -> np.ndarray:
    """`C_2m,diag / sum_n w_n` for `m = 0..orders` via Eq. (C34): `4^m sum_n w_n 2^-n sum_k binom(n,k) (2k - n)^{2m}`."""
    w = np.asarray(hist, dtype=float) / np.sum(hist)
    out = np.zeros(orders + 1)
    for n, wn in enumerate(w):
        if wn == 0.0:
            continue
        k = np.arange(n + 1)
        binom = np.array([math.comb(n, int(j)) for j in k], dtype=float) / 2.0**n
        for m in range(orders + 1):
            out[m] += wn * 4.0**m * np.dot(binom, (2.0 * k - n) ** (2 * m))
    return out


def truncated_series(hist, delta: float, order_2m: int) -> float:
    """`sum_{m <= order_2m / 2} (-1)^m delta^{2m} / (2m)! C_2m,diag`, the truncation Fig. 17(b) scans."""
    moments = diagonal_moments(hist, order_2m // 2)
    return float(
        sum((-1) ** m * delta ** (2 * m) / math.factorial(2 * m) * c for m, c in enumerate(moments))
    )


# Estimators on a propagated `A`, with `V = prod_{q in sites} exp(-i delta G_q)` and `G_q` the single-qubit `X` (`axis="x"`) or `Z`.
# With `comm`, `A` is a `result="local"` share and the per-rank values are summed.

_BYTE_POPCOUNT = np.array([bin(b).count("1") for b in range(256)], dtype=np.uint8)
_PAIR_CHUNK = 1 << 22


def _popcount(words: np.ndarray) -> np.ndarray:
    """Set bits per row of a `uint64` array of shape `(..., W)`."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(words).sum(axis=-1, dtype=np.int64)
    as_bytes = np.ascontiguousarray(words).view(np.uint8)
    return _BYTE_POPCOUNT[as_bytes].sum(axis=-1, dtype=np.int64)


def _site_mask(sites, num_qubits: int, width: int) -> np.ndarray:
    sites = [int(q) for q in sites]
    if len(set(sites)) != len(sites):
        raise ValueError(f"a site is listed twice in {sites}")
    mask = np.zeros(width, dtype=np.uint64)
    for q in sites:
        if not 0 <= q < num_qubits:
            raise ValueError(f"site {q} is outside the {num_qubits}-qubit register")
        mask[q // 64] |= np.uint64(1 << (q % 64))
    return mask


def _halves(x: np.ndarray, z: np.ndarray, axis: str) -> tuple[np.ndarray, np.ndarray]:
    """`(selector, flip)`: `G_q` anticommutes with a term iff its selector bit is set on `q`, and `V` then flips its other half there."""
    if axis == "z":
        return x, z
    if axis == "x":
        return z, x
    raise ValueError(f"axis must be 'x' or 'z', got {axis!r}")


def _allreduce(value, comm):
    return value if comm is None else comm.allreduce(value)


def anticommute_histogram(A, sites, axis: str = "x", comm=None) -> np.ndarray:
    """`w[n] = sum |c_P|^2` over the strings `P` anticommuting with exactly `n` of the `G_q`; length `len(sites) + 1`."""
    x, z, c = A.to_arrays()
    mask = _site_mask(sites, A.num_qubits, x.shape[1])
    selector, _ = _halves(x, z, axis)
    counts = _popcount(selector & mask)
    hist = np.bincount(counts, weights=np.abs(c) ** 2, minlength=len(sites) + 1).astype(float)
    return _allreduce(hist, comm)


def diagonal_echo(hist, delta: float) -> float:
    """`sum_n w_n cos(2 delta)^n / sum_n w_n`: the class-diagonal part of `rotated_overlap`, normalized by `sum |c|^2`."""
    w = np.asarray(hist, dtype=float)
    return float(np.dot(w, np.cos(2.0 * delta) ** np.arange(len(w))) / w.sum())


def rotated_overlap_complex(A, sites, delta: float, axis: str = "x") -> complex:
    """`2^-n Tr(A^dagger V^dagger A V)` of one sum (or share), without materializing `V^dagger A V`.

    Strings agreeing off the flipped bits on their anticommuting sites form a class that `V^dagger . V` mixes; each class of `k` strings costs `k^2` pairs.
    """
    x, z, c = A.to_arrays()
    mask = _site_mask(sites, A.num_qubits, x.shape[1])
    selector, flip = _halves(x, z, axis)
    k = selector & mask
    moving = _popcount(k) > 0
    fixed = float(np.sum(np.abs(c[~moving]) ** 2))
    selector, flip, k, c = selector[moving], flip[moving], k[moving], c[moving]
    if len(c) == 0:
        return complex(fixed)
    # Group by class: the selector half and the flip half with its moving bits cleared.
    key = np.concatenate([selector, flip & ~k], axis=1)
    order = np.lexsort(key.T[::-1])
    key, flip, k, c = key[order], (flip & k)[order], k[order], c[order]
    new_class = np.ones(len(c), dtype=bool)
    new_class[1:] = np.any(key[1:] != key[:-1], axis=1)
    starts = np.flatnonzero(new_class)
    sizes = np.diff(np.append(starts, len(c)))
    member_size = np.repeat(sizes, sizes)
    member_start = np.repeat(starts, sizes)
    sites_k = _popcount(k)

    sin, cos = math.sin(2.0 * delta), math.cos(2.0 * delta)
    powers = np.arange(len(sites) + 1)
    cos_pow, sin_pow = cos**powers, sin**powers
    total = 0.0j
    # Every ordered pair (p, q) of one class, in chunks of about `_PAIR_CHUNK` pairs.
    bounds = np.searchsorted(np.cumsum(member_size), np.arange(_PAIR_CHUNK, member_size.sum(), _PAIR_CHUNK))
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(c)]):
        if lo == hi:
            continue
        pair_counts = member_size[lo:hi]
        p = np.repeat(np.arange(lo, hi), pair_counts)
        offset = np.arange(len(p)) - np.repeat(np.cumsum(pair_counts) - pair_counts, pair_counts)
        q = member_start[p] + offset
        d = flip[p] ^ flip[q]
        flipped = _popcount(d)
        # R[P, Q] is `-sin 2 delta` where P holds Y against Q's X (Z axis), or P holds Z against Q's Y (X axis).
        negative = _popcount((flip[p] if axis == "z" else flip[q]) & d) % 2 == 1
        magnitude = cos_pow[sites_k[p] - flipped] * sin_pow[flipped]
        total += np.sum(np.conj(c[p]) * c[q] * np.where(negative, -magnitude, magnitude))
    return complex(total + fixed)


def rotated_overlap(A, sites, delta: float, axis: str = "x", comm=None) -> float:
    """The operator Loschmidt echo `2^-n Tr(A V^dagger A V)`: the real part of `rotated_overlap_complex`, which is the echo itself for a Hermitian `A`.

    With `comm`, every class must lie on one rank; `run_b8.py` ensures it by excluding the flipped bits of `sites` from the partition rows.
    """
    return _allreduce(rotated_overlap_complex(A, sites, delta, axis).real, comm)
