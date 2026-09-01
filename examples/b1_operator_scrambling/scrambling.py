"""Operator-spreading diagnostics read off a truncated `PauliSum`.

Handoff item B1; see `research/plans/2026-08-31-examples-benchmarks-suite.md`
§6 Part B ("B1 scrambling/OTOC"). Consumed by `run_b1_1d.py`, `run_b1_2d.py`
and `python/paulistrings/tests/test_showcase_b1.py`.

Everything here is **read-only** over the numpy export
(`x_array`/`z_array`/`coefficients_array`), the same discipline as
`examples/b6_resource_probes/probes.py` (plan §8, D12): no core change is
needed to measure operator spreading, only an honest reading of the evolved
sum.

The one quantity everything is built from
----------------------------------------
Let the Heisenberg-evolved operator be

    O(t) = U(t)^dagger O U(t) = sum_P c_P(t) P

with `P` running over Pauli strings in the repo's Hermitian convention
(CLAUDE.md §Known gaps: `Y -> (x=1, z=1)`, no phase, so a Hermitian operator
has real coefficients). Normalize the trace inner product as

    <A, B> = Tr(A^dagger B) / 2^n     ==>     <P, Q> = delta_PQ

so the Pauli strings are an orthonormal basis and

    N(t) = <O(t), O(t)> = sum_P |c_P(t)|^2

is the (squared) Hilbert-Schmidt norm. Under exact unitary evolution `N` is
conserved; the seed `O = Z_c` used throughout this showcase has `N(0) = 1`.
**Under truncation `N(t)` only decreases, and `1 - N(t)` is the fraction of the
operator the truncation threw away** — the single most useful convergence
diagnostic in this whole showcase, reported next to every curve.

Per-site weight (the light cone)
--------------------------------
`support_profile` returns

    w_q(t) = sum_{P : P_q != I} |c_P(t)|^2

the operator weight that acts non-trivially on qubit `q`. `w_q(t)` is the
light-cone heat map; `support_size` counts the sites where it clears a floor.
At `t = 0`, `w_q = delta_{q,c}`.

The OTOC, derived
-----------------
The infinite-temperature squared commutator of the evolved operator with a
probe Pauli `W_r` supported on site `r` is

    C(r,t) = (1/2) <[W_r, O(t)], [W_r, O(t)]>.

Take it apart term by term. `W_r` is a Pauli, so for each string `P` either
`[W_r, P] = 0` (they commute) or `W_r P = -P W_r`, in which case

    [W_r, P] = W_r P - P W_r = 2 W_r P.

Writing `A = sum_{P : {W_r,P} = 0} c_P P` for the anticommuting part,

    [W_r, O(t)] = 2 W_r A,

and since `W_r` is unitary, `<2 W_r A, 2 W_r A> = 4 <A, A>`. Orthonormality of
the Pauli basis then gives the working formula this module implements:

    C(r,t) = 2 * sum_{P anticommuting with W_r} |c_P(t)|^2.          (*)

Equivalently, with the more familiar OTOC `F(r,t) = <W_r O(t) W_r, O(t)> =
sum_P s_P |c_P|^2` (where `s_P = +1` if `P` commutes with `W_r` and `-1` if it
anticommutes), splitting `N = sum_comm + sum_anti` gives `F = N - 2 sum_anti`
and hence `C(r,t) = N(t) - F(r,t)` — the form quoted in the handoff. The two
are the same statement; (*) is what the code evaluates because it needs no
cancellation between large terms.

**Normalization.** (*) is written for the *unnormalized* evolved operator, so
under truncation `C` inherits the lost norm: `0 <= C <= 2 N(t) <= 2`, with the
right-hand equality when every remaining string anticommutes with the probe (as
at `t = 0`, where `C(c,0) = 2`). Both `C` and `N` are reported; `C/N` is available for the reader who wants the norm-restored
version, but nothing here silently divides by a shrinking `N` — a curve that
is drifting because the truncation is losing norm must look like it.

For a single-site probe the (anti)commutation test is one symplectic bit. With
`P` carrying bits `(x_q, z_q)` at site `q` and `W` carrying `(a, b)`, they
anticommute at that site iff `x_q b + z_q a = 1 (mod 2)`, so

    W = X_r  ->  anticommutes iff z_r(P) = 1
    W = Z_r  ->  anticommutes iff x_r(P) = 1
    W = Y_r  ->  anticommutes iff x_r(P) XOR z_r(P) = 1

which is why one chunked pass over the bit columns yields `C(r,t)` for every
site `r` and all three probes at once (`site_sums`).

**A cross-check that costs nothing.** Averaging (*) over the three probes
`W_r in {X_r, Y_r, Z_r}`: a string with `P_r = I` anticommutes with none of
them, and a string with `P_r != I` anticommutes with exactly two, so

    (1/3) sum_W C_W(r,t) = (2/3) * 2 * sum_{P_r != I} |c_P|^2 = (4/3) w_r(t).

The probe-averaged OTOC *is* the support profile, up to `4/3`. That identity is
exact term by term, so `probe_average_gap` comparing the two independently
computed arrays is a machine-precision self-test of both implementations, and
the showcase scripts assert it.

Dense reference path
--------------------
`dense_*` builds the evolution operator as an explicit `2^n x 2^n` matrix by
Kronecker products and evaluates the same three quantities with dense linear
algebra: no `PauliSum`, no engine, no qiskit. That is the *independent* path
the small-`n` validation compares against, per plan §7 rule 1. The support
profile is obtained there from the single-qubit Pauli twirl

    T_q(O) = (1/4) sum_{g in {I,X,Y,Z}_q} g O g,

which projects onto the strings carrying identity at `q` (for `P_q != I` two of
the four conjugations flip the sign and the sum cancels), so
`w_q = <O,O> - <T_q(O), T_q(O)>` with no Pauli decomposition anywhere.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "PAULI_MATRICES",
    "SiteSums",
    "chain_edges",
    "cubic_lattice_edges",
    "cubic_lattice_index",
    "dense_coefficient",
    "dense_hs_norm",
    "dense_otoc",
    "dense_pauli",
    "dense_support_profile",
    "dense_unitary",
    "front_position",
    "front_velocity",
    "hs_norm",
    "otoc_from_sums",
    "otoc_profile",
    "probe_average_gap",
    "single_pauli_coefficients",
    "site_sums",
    "square_lattice_coords",
    "square_lattice_distances",
    "square_lattice_edges",
    "square_lattice_index",
    "support_profile",
    "support_size",
]

#: Rows of the numpy export processed per chunk. The chunked loop exists
#: because a converged real-time run reaches ~10^8 terms: a dense `(terms,
#: n_qubits)` bit matrix at that size is tens of GiB, while the chunk below is
#: a few MiB and the accumulators are `O(n_qubits)`.
CHUNK_ROWS = 1 << 16


# --------------------------------------------------------------------------
# Lattices
# --------------------------------------------------------------------------


def chain_edges(n: int) -> list[tuple[int, int]]:
    """Open 1D chain: bonds `(i, i+1)` for `i = 0 .. n-2`."""
    if n < 2:
        raise ValueError(f"n must be >= 2 for a chain with at least one bond, got {n}")
    return [(i, i + 1) for i in range(n - 1)]


def square_lattice_index(rows: int, cols: int, r: int, c: int) -> int:
    """Row-major site index of lattice coordinate `(r, c)`."""
    if not 0 <= r < rows or not 0 <= c < cols:
        raise ValueError(f"({r}, {c}) is outside a {rows}x{cols} lattice")
    return r * cols + c


def square_lattice_coords(rows: int, cols: int, q: int) -> tuple[int, int]:
    """Inverse of `square_lattice_index`."""
    if not 0 <= q < rows * cols:
        raise ValueError(f"site {q} is outside a {rows}x{cols} lattice")
    return divmod(q, cols)


def square_lattice_edges(rows: int, cols: int) -> list[tuple[int, int]]:
    """Open square-lattice bonds, row-major site numbering.

    `circuits.py` ships heavy-hex and chain-shaped topologies but no square
    lattice; the plan allows building one in the showcase ("build edges in the
    script if circuits.py lacks square -- keep it simple"), and it is exactly
    this: every horizontal and vertical nearest-neighbour pair, each once, as
    `(lo, hi)`. `circuits.heavy_hex_kicked_ising(..., edges=...)` then runs the
    same kicked-Ising step structure on it, so the 1D and 2D showcases share
    one circuit builder and one truncation schedule.
    """
    if rows < 1 or cols < 1:
        raise ValueError(f"rows and cols must be >= 1, got {rows}x{cols}")
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            here = square_lattice_index(rows, cols, r, c)
            if c + 1 < cols:
                edges.append((here, square_lattice_index(rows, cols, r, c + 1)))
            if r + 1 < rows:
                edges.append((here, square_lattice_index(rows, cols, r + 1, c)))
    return [(min(a, b), max(a, b)) for a, b in edges]


def cubic_lattice_edges(nx: int, ny: int, nz: int) -> list[tuple[int, int]]:
    """Open cubic-lattice bonds, with site index `(i, j, k) -> (i*ny + j)*nz + k`.

    The 3D counterpart of `square_lattice_edges`, used only by the 3D pilot: it
    exists so the projected cost in the B1 README is a measured number on a
    real degree-6 lattice rather than an extrapolation of a formula.
    """
    if min(nx, ny, nz) < 1:
        raise ValueError(f"lattice dimensions must be >= 1, got {nx}x{ny}x{nz}")

    def index(i: int, j: int, k: int) -> int:
        return (i * ny + j) * nz + k

    edges: list[tuple[int, int]] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                here = index(i, j, k)
                if i + 1 < nx:
                    edges.append((here, index(i + 1, j, k)))
                if j + 1 < ny:
                    edges.append((here, index(i, j + 1, k)))
                if k + 1 < nz:
                    edges.append((here, index(i, j, k + 1)))
    return [(min(a, b), max(a, b)) for a, b in edges]


def cubic_lattice_index(nx: int, ny: int, nz: int, i: int, j: int, k: int) -> int:
    """Site index of cubic-lattice coordinate `(i, j, k)`."""
    if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
        raise ValueError(f"({i}, {j}, {k}) is outside a {nx}x{ny}x{nz} lattice")
    return (i * ny + j) * nz + k


def square_lattice_distances(rows: int, cols: int, center: int) -> np.ndarray:
    """Graph distance from `center` to every site, as a flat `(rows*cols,)` array.

    On the square lattice the graph distance is the Manhattan distance, which
    is also the strict causal radius of the kicked-Ising step used here (one
    commuting ZZ layer per Trotter step moves the operator boundary by at most
    one bond).
    """
    r0, c0 = square_lattice_coords(rows, cols, center)
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return (np.abs(rr - r0) + np.abs(cc - c0)).reshape(-1).astype(np.int64)


# --------------------------------------------------------------------------
# Chunked reading of the numpy export
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteSums:
    """Per-site coefficient-weight sums of one evolved `PauliSum`.

    Every array is indexed by qubit and every entry is a sum of `|c_P|^2` over
    the strings satisfying the stated condition at that qubit:

    - `x` — strings with `x_q = 1` (a factor of `X` or `Y` at `q`)
    - `z` — strings with `z_q = 1` (a factor of `Z` or `Y` at `q`)
    - `xz` — strings with `x_q = z_q = 1` (a factor of `Y` at `q`)

    `norm` is `sum_P |c_P|^2` over all strings (the squared Hilbert-Schmidt
    norm), and `terms` the number of stored strings. The derived quantities
    (`occupied`, `xor`) follow by inclusion-exclusion, which is why only three
    bit columns are accumulated rather than four.
    """

    num_qubits: int
    terms: int
    norm: float
    x: np.ndarray
    z: np.ndarray
    xz: np.ndarray

    @property
    def occupied(self) -> np.ndarray:
        """Weight on strings with a non-identity factor at `q`: `x | z`."""
        return self.x + self.z - self.xz

    @property
    def xor(self) -> np.ndarray:
        """Weight on strings whose factor at `q` is `X` or `Z` (not `Y`, not `I`)."""
        return self.x + self.z - 2.0 * self.xz


def _bit_columns(words: np.ndarray, num_qubits: int) -> np.ndarray:
    """Unpack a `(rows, W)` array of `u64` words into a `(rows, num_qubits)` bit
    matrix of `uint8`, with column `q` holding bit `q` of the packed row.

    Uses the byte view plus `np.unpackbits(..., bitorder="little")`: on a
    little-endian machine byte `j` of word `w` holds packed bits `8j .. 8j+7`,
    so the unpacked column order is exactly ascending qubit index. The
    byte-order assumption is asserted at import.
    """
    if words.size == 0:
        return np.zeros((words.shape[0], num_qubits), dtype=np.uint8)
    bits = np.unpackbits(
        np.ascontiguousarray(words).view(np.uint8), axis=1, bitorder="little"
    )
    return bits[:, :num_qubits]


if sys.byteorder != "little":  # pragma: no cover - every target host is x86/ARM LE
    raise RuntimeError(
        "scrambling.py unpacks the symplectic key with np.unpackbits over a byte "
        "view, which assumes little-endian u64 words"
    )


def site_sums(pauli_sum: Any, *, chunk_rows: int = CHUNK_ROWS) -> SiteSums:
    """One chunked pass over `pauli_sum`, returning every per-site weight sum.

    This is the only function in the module that touches the export, and every
    other diagnostic (`support_profile`, `otoc_profile`, `hs_norm`) is a
    closed-form combination of its output — so a time series costs one pass per
    time point, not one pass per quantity.
    """
    n = int(pauli_sum.num_qubits)
    coeffs = np.asarray(pauli_sum.coefficients_array())
    total = int(coeffs.shape[0])
    x_acc = np.zeros(n, dtype=np.float64)
    z_acc = np.zeros(n, dtype=np.float64)
    xz_acc = np.zeros(n, dtype=np.float64)
    norm = 0.0
    if total == 0:
        return SiteSums(num_qubits=n, terms=0, norm=0.0, x=x_acc, z=z_acc, xz=xz_acc)

    xs = np.asarray(pauli_sum.x_array())
    zs = np.asarray(pauli_sum.z_array())
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")

    for lo in range(0, total, chunk_rows):
        hi = min(lo + chunk_rows, total)
        p = np.abs(coeffs[lo:hi]) ** 2
        norm += float(p.sum())
        xb = _bit_columns(xs[lo:hi], n).astype(np.float64)
        zb = _bit_columns(zs[lo:hi], n).astype(np.float64)
        x_acc += p @ xb
        z_acc += p @ zb
        xz_acc += p @ (xb * zb)

    return SiteSums(
        num_qubits=n, terms=total, norm=norm, x=x_acc, z=z_acc, xz=xz_acc
    )


def hs_norm(pauli_sum: Any) -> float:
    """`sum_P |c_P|^2` — the squared Hilbert-Schmidt norm, `<O, O>`.

    Conserved by exact unitary evolution; the shortfall from its initial value
    is exactly the operator weight truncation discarded.
    """
    coeffs = np.asarray(pauli_sum.coefficients_array())
    if coeffs.size == 0:
        return 0.0
    return float(np.sum(np.abs(coeffs) ** 2))


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def support_profile(pauli_sum: Any, *, sums: SiteSums | None = None) -> np.ndarray:
    """Per-site operator weight `w_q = sum_{P_q != I} |c_P|^2`, shape `(n,)`.

    Pass `sums` to reuse an existing `site_sums` pass.
    """
    s = sums if sums is not None else site_sums(pauli_sum)
    return s.occupied


def support_size(profile: np.ndarray, floor: float = 1e-6) -> int:
    """Number of sites whose weight exceeds `floor` (strictly).

    The floor is what makes "support size" a number rather than a tautology: a
    generic evolved operator has a tiny but nonzero weight on every site inside
    the causal cone, so the answer without a floor is just the cone size. The
    default `1e-6` is an absolute threshold on a seed normalized to
    `sum_P |c_P|^2 = 1`; the showcase scripts report the floor next to every
    number and sweep it where the conclusion could depend on it.
    """
    if floor < 0.0:
        raise ValueError(f"floor must be non-negative, got {floor}")
    return int(np.count_nonzero(np.asarray(profile) > floor))


def otoc_profile(
    pauli_sum: Any, probe: str = "X", *, sums: SiteSums | None = None
) -> np.ndarray:
    """The squared commutator `C(r) = 2 sum_{P anti W_r} |c_P|^2` for every site `r`.

    `probe` is the single-site Pauli `W_r` (`"X"`, `"Y"` or `"Z"`), applied at
    every site in turn: the returned array's entry `r` is the OTOC of the
    evolved operator with `W` placed on site `r`. See the module docstring for
    the derivation of the factor 2 and for the symplectic-bit rule behind each
    branch.

    Unnormalized (module docstring, "Normalization"): divide by
    `hs_norm(pauli_sum)` for the norm-restored version if that is what you
    want, and say so when you do.
    """
    return otoc_from_sums(sums if sums is not None else site_sums(pauli_sum), probe)


def otoc_from_sums(s: SiteSums, probe: str = "X") -> np.ndarray:
    """`otoc_profile` for an already-computed `site_sums` pass."""
    key = probe.upper()
    if key == "X":
        anti = s.z  # {X_r, P} = 0  <=>  z_r(P) = 1
    elif key == "Z":
        anti = s.x  # {Z_r, P} = 0  <=>  x_r(P) = 1
    elif key == "Y":
        anti = s.xor  # {Y_r, P} = 0  <=>  x_r(P) != z_r(P)
    else:
        raise ValueError(f"probe must be 'X', 'Y' or 'Z', got {probe!r}")
    return 2.0 * anti


def probe_average_gap(sums: SiteSums) -> float:
    """Max violation of the exact identity `mean_W C_W(r) = (4/3) w_r`.

    Zero to floating-point rounding for any sum; a nonzero value means one of
    the two independently accumulated bit columns is wrong (module docstring,
    "A cross-check that costs nothing"). Returned as a single number so a
    script or test can assert on it directly.
    """
    average = (
        otoc_from_sums(sums, "X")
        + otoc_from_sums(sums, "Y")
        + otoc_from_sums(sums, "Z")
    ) / 3.0
    return float(np.max(np.abs(average - (4.0 / 3.0) * sums.occupied)))


def single_pauli_coefficients(pauli_sum: Any, axis: str = "Z") -> np.ndarray:
    """Coefficients of the `n` weight-one strings on the given axis, shape `(n,)`.

    Entry `r` is `c_{W_r}` where `W_r` is `axis` on site `r` and identity
    elsewhere — i.e. the infinite-temperature two-point function of the evolved
    operator,

        G(r, t) = <W_r, O(t)> = Tr(W_r O(t)) / 2^n,

    which for `O = Z_c` and `axis = "Z"` is the standard dynamical correlator
    `Tr(Z_r U^dagger Z_c U)/2^n`. Sites whose string is not in the sum get
    exactly `0.0`, and that zero is a **sensitivity floor, not an
    approximation**: this is a single coefficient, so a value below the run's
    `min_abs_coeff` was deleted rather than approximated. A zero therefore means
    "absent, or below the cutoff", and only a tighter cutoff can tell those two
    apart -- unlike `support_profile`, where truncation shows up as a small
    error in a sum of many terms.

    Real part only: every observable in this showcase is Hermitian, so these
    coefficients are real, and a non-negligible imaginary part is a bug — the
    caller gets a `ValueError` rather than a silently discarded imaginary part.
    """
    n = int(pauli_sum.num_qubits)
    key = axis.upper()
    if key not in ("X", "Y", "Z"):
        raise ValueError(f"axis must be 'X', 'Y' or 'Z', got {axis!r}")
    out = np.zeros(n, dtype=np.float64)
    coeffs = np.asarray(pauli_sum.coefficients_array())
    if coeffs.size == 0:
        return out

    xs = np.asarray(pauli_sum.x_array())
    zs = np.asarray(pauli_sum.z_array())
    # A weight-one string on `axis` has exactly one populated bit in the
    # relevant half-key and none in the other (for Y, one bit in both halves at
    # the same position).
    x_pop = np.bitwise_count(xs).sum(axis=1)
    z_pop = np.bitwise_count(zs).sum(axis=1)
    if key == "X":
        rows = np.flatnonzero((x_pop == 1) & (z_pop == 0))
        carrier = xs
    elif key == "Z":
        rows = np.flatnonzero((x_pop == 0) & (z_pop == 1))
        carrier = zs
    else:
        rows = np.flatnonzero((x_pop == 1) & (z_pop == 1) & np.all(xs == zs, axis=1))
        carrier = xs
    for row in rows:
        bits = _bit_columns(carrier[row : row + 1], n)[0]
        site = int(np.flatnonzero(bits)[0])
        value = complex(coeffs[row])
        if abs(value.imag) > 1e-9 * max(1.0, abs(value.real)):
            raise ValueError(
                f"the weight-one {key} string on site {site} has coefficient {value!r}, "
                "which is not real: the evolved operator is not Hermitian, so something "
                "upstream (a convention, or the seed observable) is wrong"
            )
        out[site] = value.real
    return out


# --------------------------------------------------------------------------
# Light-cone front and butterfly velocity
# --------------------------------------------------------------------------


def front_position(
    profile: np.ndarray,
    center: int,
    threshold: float,
    *,
    distances: np.ndarray | None = None,
) -> float:
    """Outermost distance from `center` at which the weight clears `threshold`.

    `distances` supplies the lattice metric (from `square_lattice_distances`,
    say); the default is the 1D chain metric `|q - center|`. Returns `0.0` when
    no site clears the threshold, `nan` never — a caller distinguishing "front
    at the origin" from "nothing above threshold" should check the profile.

    This is a *contour* readout, and the contour level is a real choice: a
    ballistic front `w ~ exp(-(x - v t)/xi)` has a threshold-independent
    asymptotic velocity but a threshold-dependent apparent one at finite time.
    That is why `front_velocity` is called at several thresholds in the
    showcase, and the spread across them is reported as the systematic.
    """
    profile = np.asarray(profile)
    if distances is None:
        distances = np.abs(np.arange(profile.shape[0]) - int(center))
    distances = np.asarray(distances)
    if distances.shape != profile.shape:
        raise ValueError(
            f"distances has shape {distances.shape}, profile has {profile.shape}"
        )
    above = profile > threshold
    if not np.any(above):
        return 0.0
    return float(np.max(distances[above]))


def front_velocity(
    times: Sequence[float], fronts: Sequence[float]
) -> tuple[float, float]:
    """Least-squares `(slope, intercept)` of `fronts` against `times`.

    The slope is the butterfly velocity in sites per unit of whatever `times`
    is measured in (Trotter steps here, unless the caller divides by `dt`).
    Needs at least two points and at least two distinct times.
    """
    t = np.asarray(times, dtype=np.float64)
    d = np.asarray(fronts, dtype=np.float64)
    if t.shape != d.shape:
        raise ValueError(f"times has shape {t.shape}, fronts has {d.shape}")
    if t.size < 2 or np.unique(t).size < 2:
        raise ValueError("front_velocity needs at least two points at distinct times")
    slope, intercept = np.polyfit(t, d, 1)
    return float(slope), float(intercept)


# --------------------------------------------------------------------------
# Dense reference path (independent of the engine)
# --------------------------------------------------------------------------

#: Single-qubit Pauli matrices in the Hermitian convention -- `Y` really is the
#: Hermitian `[[0, -i], [i, 0]]`, matching `(x=1, z=1)` with no phase factor.
PAULI_MATRICES: Mapping[str, np.ndarray] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def dense_pauli(label: str) -> np.ndarray:
    """A Pauli string as a dense `2^n x 2^n` matrix.

    Kronecker order: character 0 of the label is the **most significant**
    tensor factor. That is a free choice -- every quantity computed from it
    here is a trace, hence basis-independent -- but it must be the same choice
    for the operator and for the circuit, so `dense_unitary` uses it too.
    """
    if not label:
        raise ValueError("label must be non-empty")
    matrix = np.ones((1, 1), dtype=complex)
    for ch in label:
        try:
            factor = PAULI_MATRICES[ch]
        except KeyError:
            raise ValueError(f"unexpected Pauli character {ch!r} in {label!r}") from None
        matrix = np.kron(matrix, factor)
    return matrix


def _embedded_pauli(pauli: str, qubits: Sequence[int], n: int) -> np.ndarray:
    label = ["I"] * n
    for ch, q in zip(pauli, qubits, strict=True):
        label[int(q)] = ch
    return dense_pauli("".join(label))


def _gate_unitary(gate: Mapping[str, Any], n: int) -> np.ndarray:
    """`exp(-i theta P / 2)` for one gate object of the suite's gate vocabulary.

    Every gate the B1 circuits emit is a Pauli rotation (`rx`/`ry`/`rz` are the
    weight-one cases), and for a Pauli `P` with `P^2 = 1`

        exp(-i theta P / 2) = cos(theta/2) I - i sin(theta/2) P

    exactly, so one branchless formula covers them all. Anything else raises:
    this reference deliberately has no fallback that could quietly disagree
    with the engine.
    """
    name = str(gate["name"])
    qubits = [int(q) for q in gate["qubits"]]
    if name in ("rx", "ry", "rz"):
        pauli = name[1].upper()
    elif name == "pauli_rotation":
        pauli = str(gate["pauli"]).upper()
    else:
        raise ValueError(
            f"the dense reference in scrambling.py handles Pauli rotations only "
            f"(rx/ry/rz/pauli_rotation), got gate {name!r}"
        )
    theta = float(gate["theta"])
    p = _embedded_pauli(pauli, qubits, n)
    return np.cos(theta / 2.0) * np.eye(2**n, dtype=complex) - 1j * np.sin(
        theta / 2.0
    ) * p


def dense_unitary(spec: Any) -> np.ndarray:
    """The full `2^n x 2^n` unitary of a `CircuitSpec`-like gate list.

    Gates are composed in circuit order: gate 0 acts first, so
    `U = G_last ... G_1 G_0`. Only `n <= ~12` is sane; nothing here guards the
    size, because the callers are the small-`n` validation paths.
    """
    n = int(spec.num_qubits)
    u = np.eye(2**n, dtype=complex)
    for gate in spec.gates:
        u = _gate_unitary(gate, n) @ u
    return u


def dense_heisenberg(spec: Any, observable_label: str) -> np.ndarray:
    """`U^dagger O U` as a dense matrix, for `O` the given Pauli string.

    Matches `PauliSum.propagate(circuit, None, direction="heisenberg")`, which
    is exactly what the small-`n` validation compares against.
    """
    n = int(spec.num_qubits)
    if len(observable_label) != n:
        raise ValueError(
            f"observable label has length {len(observable_label)}, expected {n}"
        )
    u = dense_unitary(spec)
    o = dense_pauli(observable_label)
    return u.conj().T @ o @ u


def dense_hs_norm(operator: np.ndarray) -> float:
    """`<O, O> = Tr(O^dagger O) / 2^n`."""
    dim = operator.shape[0]
    return float(np.real(np.trace(operator.conj().T @ operator)) / dim)


def dense_coefficient(operator: np.ndarray, label: str) -> complex:
    """`<P, O> = Tr(P^dagger O) / 2^n` for the Pauli string `label`.

    `P` is Hermitian, so this is `Tr(P O)/2^n`, evaluated as an elementwise
    contraction rather than a matrix product.
    """
    p = dense_pauli(label)
    dim = operator.shape[0]
    if p.shape != operator.shape:
        raise ValueError(f"label {label!r} gives shape {p.shape}, operator {operator.shape}")
    return complex(np.sum(p * operator.T) / dim)


def dense_otoc(operator: np.ndarray, probe: str, site: int, n: int) -> float:
    """`C(r) = (1/2) <[W_r, O], [W_r, O]>` computed with dense matrices.

    No Pauli decomposition is involved: the commutator is formed explicitly and
    contracted, which is what makes this an independent check of
    `otoc_profile`'s bit-mask derivation.
    """
    w = _embedded_pauli(probe.upper(), [site], n)
    commutator = w @ operator - operator @ w
    return 0.5 * dense_hs_norm(commutator)


def dense_support_profile(operator: np.ndarray, n: int) -> np.ndarray:
    """Per-site weight `w_q` from the single-qubit Pauli twirl.

    `w_q = <O,O> - <T_q(O), T_q(O)>` with
    `T_q(O) = (1/4) sum_{g in {I,X,Y,Z}_q} g O g` (module docstring, "Dense
    reference path"). Again no Pauli decomposition: the projector is built from
    four conjugations.
    """
    total = dense_hs_norm(operator)
    out = np.zeros(n, dtype=np.float64)
    for q in range(n):
        twirled = np.zeros_like(operator)
        for g in "IXYZ":
            m = _embedded_pauli(g, [q], n) if g != "I" else None
            twirled += operator if m is None else m @ operator @ m
        twirled /= 4.0
        out[q] = total - dense_hs_norm(twirled)
    return out
