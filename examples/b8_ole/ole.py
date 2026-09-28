"""Operator Loschmidt echo on the 56-qubit ibm_boston heavy-hex patch (arXiv:2607.25998).

Circuits come from the vendored Quantum Advantage Tracker QASM; `echo_half` rebuilds the measured half for any `(L, eta)`.
Estimators: the paper's diagonal OLE from an anticommutation histogram, and the exact `rotated_overlap`.
"""

from __future__ import annotations

import gzip
import json
import math
import re
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


def _quarter_turns(angle: float) -> int | None:
    """`k` if `angle` is `k * pi/2` modulo `2 pi` to rounding, else `None`."""
    k = angle / (math.pi / 2)
    return int(round(k)) % 4 if abs(k - round(k)) < 1e-9 else None


def to_circuit(ops, num_qubits: int | None = None, index: dict[int, int] | None = None, snap_cliffords: bool = True):
    """A `paulistrings.Circuit`, one gate per channel, so collapse checks fall after every gate as in the paper.

    With `snap_cliffords`, rotations by a multiple of `pi/2` become exact Cliffords (up to global phase).
    As rotations they branch: `cos(pi/2)` rounds to `6e-17`, so each `rz(pi/2)` would add a near-zero copy of every anticommuting string, which fills the PP-MC cache.
    """
    from paulistrings import Circuit

    index = qubit_index() if index is None else index
    circuit = Circuit(len(index) if num_qubits is None else num_qubits)
    for name, angle, qs in ops:
        q = [index[x] for x in qs]
        k = _quarter_turns(angle) if snap_cliffords and name in ("rx", "rz") else None
        if k is not None:
            if name == "rx" and k:
                circuit.h(q[0])
            if k == 1:
                circuit.s(q[0])
            elif k == 2:
                circuit.z(q[0])
            elif k == 3:
                circuit.sdg(q[0])
            if name == "rx" and k:
                circuit.h(q[0])
        elif name == "rx":
            circuit.rx(angle, q[0])
        elif name == "rz":
            circuit.rz(angle, q[0])
        elif name == "cz":
            circuit.cz(q[0], q[1])
        elif name == "s":
            circuit.s(q[0])
        elif name == "sdg":
            circuit.sdg(q[0])
        elif name in ("sx", "sxdg"):
            circuit.h(q[0])
            circuit.s(q[0]) if name == "sx" else circuit.sdg(q[0])
            circuit.h(q[0])
        else:
            raise ValueError(f"unsupported gate {name!r}")
    return circuit


def observable(index: dict[int, int] | None = None):
    """`O = prod_{q in V_O} Z_q` with coefficient 1."""
    from paulistrings import PauliSum

    index = qubit_index() if index is None else index
    chars = ["I"] * len(index)
    for q in spec()["observable_qubits"]:
        chars[index[q]] = "Z"
    return PauliSum.from_strings({"".join(chars): 1.0}, num_qubits=len(index))


def perturbation_sites(index: dict[int, int] | None = None) -> list[int]:
    index = qubit_index() if index is None else index
    return [index[q] for q in spec()["perturbation_qubits"]]


def diagonal_echo(hist, delta: float) -> float:
    """`sum_n w_n cos(2 delta)^n / sum_n w_n`: the paper's diagonal OLE with every moment `C_2m,diag` resummed."""
    w = np.asarray(hist, dtype=float)
    norm = w.sum()
    return float(np.dot(w, np.cos(2 * delta) ** np.arange(len(w))) / norm) if norm > 0 else float("nan")


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
