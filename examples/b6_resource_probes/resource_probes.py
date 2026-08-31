"""Resource-theoretic probes read off an evolved Pauli sum (showcase B6).

Handoff item B6; adapted spec in
`research/plans/2026-08-31-examples-benchmarks-suite.md` §6 Part B and
decision D12 ("B6 computed in pure Python over the numpy export -- read-only
diagnostics; no core additions"). Nothing in this module touches the Rust
core: every function consumes `PauliSum.x_array()` / `z_array()` /
`coefficients_array()` (or a dense matrix built from them) and returns plain
floats or numpy arrays.

Two families of diagnostic, plus the dense oracles that cross-check them.

------------------------------------------------------------------------
1. Pauli-spectrum (Rényi participation) entropies
------------------------------------------------------------------------

For `O = sum_P c_P P` over Pauli strings `P` (this repo's Hermitian
convention: `c_P` multiplies the literal string, `Y -> (x=1, z=1)` with no
phase), define the **Pauli spectrum**

    p_P = |c_P|^2 / sum_Q |c_Q|^2 .

`p` is a probability distribution over the Pauli basis, and its Rényi-α
entropy

    S_α(O) = (1 / (1 - α)) · ln( sum_P p_P^α )                       (nats)

is what `pauli_spectrum_renyi` returns. `α = 2` gives the collision entropy
`S_2 = -ln sum_P p_P^2` (`pauli_spectrum_renyi2`); the *linear* variant is
the same information without the logarithm,

    L(O) = 1 - sum_P p_P^2 = 1 - exp(-S_2) ,

so `L` and `S_2` are monotonically related by construction and either one
determines the other (`pauli_spectrum_linear`, and `S_2 = -ln(1 - L)` is
asserted in the test file).

**What this is, precisely.** This is the *operator* Pauli-spectrum entropy,
not the state stabilizer Rényi entropy (SRE). The SRE of Leone, Oliviero and
Hamma [1] is built from the same shape of formula but on a *pure state*:
`Ξ_P = <ψ|P|ψ>^2 / d` with the `-log d` normalization that makes it vanish
on stabilizer states, and it is that state quantity for which α ≥ 2 was later
proved to be a magic monotone (for pure states, under stabilizer protocols)
by Leone and Bittel [2]. **Neither monotonicity result applies to what this
module computes**, and nothing here should be described as a magic monotone.

The operator-side quantity does have a name and a purpose in the literature
this repo's engine belongs to: Shao, Cheng and Liu [3] call it the *Operator
Stabilizer Rényi entropy* (OSE) and define (their Definition 1)

    S^α(O) = (α / (1 - α)) · ln || c^2 ||_α ,
    || c^2 ||_α = ( sum_i c_i^{2α} )^{1/α} ,   c_i = 2^{-n} tr(P_i O) ,

proving that it is exactly the quantity governing Pauli-propagation
truncation error and the Top-K budget needed for a target accuracy. Their
`c_i = 2^{-n} tr(P_i O)` **is** this repo's `c_P` (Pauli strings are
orthogonal, `tr(P_i P_j) = 2^n δ_ij`), and when `sum_i c_i^2 = 1` -- i.e.
when `O` is Hilbert-Schmidt normalized, `tr(O^2) = 2^n`, which a single Pauli
string satisfies and *exact* unitary Heisenberg evolution preserves --
their `S^α(O)` is algebraically identical to the `S_α` above:

    (α/(1-α)) ln (sum c^{2α})^{1/α} = (1/(1-α)) ln sum p^α    when sum c^2 = 1.

Under truncation `sum_P |c_P|^2 < 1`, and the two differ by
`(α/(1-α)) ln(sum c^2)`. This module always renormalizes (`p` sums to 1 by
construction), which is the choice that keeps a *truncated* curve comparable
with the exact curve it is supposed to converge to; `hilbert_schmidt_weight`
reports the discarded weight `sum_P |c_P|^2` separately so the difference is
never hidden. `pauli_spectrum_renyi_unnormalized` gives the literal [3]
formula for anyone who wants it.

**Properties, derived here rather than cited.**

* *Zero on a single Pauli string.* One term means `p = (1)`, so
  `sum p^α = 1` and `S_α = 0` for every α, and `L = 0`.
* *Invariant under Clifford conjugation.* A Clifford `U` maps each Pauli
  string to `± another Pauli string`, bijectively, so `U^† O U` has the same
  multiset `{|c_P|}` up to a permutation of labels and signs. Every function
  of `{p_P}` alone -- all of the above -- is therefore unchanged. In
  particular, a Clifford circuit applied to a single Pauli string keeps the
  diagnostics at exactly `0`.
* *Grows under non-Clifford gates.* A non-Clifford rotation
  `exp(-iθ P_g / 2)` splits each anticommuting term into two
  (`cos θ`, `sin θ` branches), spreading `p` over more Paulis, which strictly
  raises `S_α` unless `θ` is at a Clifford angle. The θ_h sweep in
  `run_b6.py` is exactly this statement, measured.
* *Additive over tensor factors* (immediate from `p` factorizing), and
  *basis-dependent by construction*: it is a property of the Pauli-basis
  representation, which is the point -- it is a cost model for *this* engine,
  not a basis-independent resource measure.

------------------------------------------------------------------------
2. Operator entanglement across a bipartition
------------------------------------------------------------------------

Split the qubits at `cut`: `A = [0, cut)`, `B = [cut, n)`. Every Pauli string
factorizes across the cut, `P = P_A ⊗ P_B`, so the coefficient vector
reshapes into a matrix indexed by the *distinct* left and right factors,

    M[a, b] = c_P   for the unique P with P_A = a, P_B = b   (0 otherwise),

and `O = sum_{a,b} M[a,b] · a ⊗ b`. Because `{P_a / sqrt(2^{|A|})}` is an
orthonormal basis of operators on `A` under the Hilbert-Schmidt inner
product (and likewise on `B`), the SVD `M = U diag(s) V^†` *is* the operator
Schmidt decomposition of `O` (Zanardi [4]; the entropy of the resulting
spectrum is the operator space entanglement entropy of Prosen and Pižorn
[5]). The Schmidt weights are

    λ_k = s_k^2 / sum_j s_j^2 ,

and `operator_entanglement_entropy` returns `-sum_k λ_k ln λ_k` (nats),
`operator_entanglement_renyi2` returns `-ln sum_k λ_k^2`. `sum_j s_j^2 =
||M||_F^2 = sum_P |c_P|^2` is the same Hilbert-Schmidt weight as above, so
the same renormalization remark applies verbatim.

A single Pauli string gives a rank-1 `M`, hence `λ = (1)` and zero operator
entanglement; a Clifford circuit keeps `M` rank-1 *only if* it does not mix
the factorization, so unlike the Pauli-spectrum entropy this diagnostic is
**not** Clifford-invariant in general -- a CNOT across the cut is Clifford and
raises it. What *is* true, and is what the showcase uses, is that a Clifford
circuit applied to a single Pauli string leaves a single Pauli string, whose
operator entanglement is zero for any cut. Stated this way because the
stronger claim is false.

**Cost and guards.** `M` has shape `(n_left, n_right)` with
`n_left <= min(T, 4^cut)` and `n_right <= min(T, 4^(n-cut))` for a `T`-term
sum; it is dense-allocated (`complex128`) and SVD'd, so the cost is
`O(n_left · n_right · min(n_left, n_right))` time and
`16 · n_left · n_right` bytes. Both factor counts grow with operator
spreading, and `n_left · n_right` can approach `T^2` in the worst case, so
`operator_schmidt_values` refuses to allocate past `max_entries`
(default `MAX_SCHMIDT_ENTRIES`, 4e6 entries = 64 MiB) and says what it would
have needed. `schmidt_matrix_shape` reports the shape without building it.

------------------------------------------------------------------------
3. Dense oracles (small `n` only)
------------------------------------------------------------------------

`dense_matrix` builds the literal `2^n x 2^n` matrix by summing
`c_P · (σ ⊗ σ ⊗ ...)` with `numpy.kron`, qubit `0` as the *leftmost* (most
significant) tensor factor. From it:

* `dense_pauli_spectrum_probabilities` recovers every one of the `4^n`
  coefficients as `c_P = tr(P O) / 2^n` by brute force, with no reference to
  the sparse representation's key layout. Cost is `16^n` -- fine at `n = 6`
  (17 M complex mults), hopeless by `n = 10`, hence `MAX_DENSE_SPECTRUM_N`.
* `dense_operator_schmidt_spectrum` reshapes `O` to
  `(2^|A|, 2^|B|, 2^|A|, 2^|B|)`, regroups to `(4^|A|, 4^|B|)` and SVDs
  that -- the textbook operator Schmidt decomposition, sharing no code path
  with the Pauli-factor construction above. Cost is `O(8^n)`, fine to
  `n = 10`.

`run_b6.py` Part 1 asserts agreement to `1e-10`.

------------------------------------------------------------------------
References
------------------------------------------------------------------------

[1] L. Leone, S. F. E. Oliviero, A. Hamma, "Stabilizer Rényi Entropy",
    Phys. Rev. Lett. 128, 050402 (2022); arXiv:2106.12587. *State* SRE.
[2] L. Leone, L. Bittel, "Stabilizer entropies are monotones for magic-state
    resource theory", Phys. Rev. A 110, L040403 (2024); arXiv:2404.11652.
    Monotonicity for α >= 2, **pure states**.
[3] Y. Shao, S. Cheng, Z. Liu, "Characterizing Pauli Propagation via Operator
    Complexity", arXiv:2510.22311 (2025). Operator Stabilizer Rényi entropy
    (OSE), Definition 1; truncation-error bounds.
[4] P. Zanardi, "Entanglement of quantum evolutions", Phys. Rev. A 63,
    040304(R) (2001). Operator Schmidt decomposition / operator entanglement.
[5] T. Prosen, I. Pižorn, "Operator space entanglement entropy in transverse
    Ising chain", Phys. Rev. A 76, 032316 (2007). Operator space entanglement
    entropy (OSEE) as the cost model for simulating observables.
"""

from __future__ import annotations

import functools
import itertools
from typing import Iterable

import numpy as np

__all__ = [
    "MAX_SCHMIDT_ENTRIES",
    "MAX_DENSE_SPECTRUM_N",
    "PAULI_MATRICES",
    "hilbert_schmidt_weight",
    "pauli_spectrum_probabilities",
    "pauli_spectrum_renyi",
    "pauli_spectrum_renyi2",
    "pauli_spectrum_renyi_unnormalized",
    "pauli_spectrum_linear",
    "pauli_spectrum_shannon",
    "schmidt_matrix_shape",
    "schmidt_matrix",
    "operator_schmidt_values",
    "operator_schmidt_spectrum",
    "operator_entanglement_entropy",
    "operator_entanglement_renyi2",
    "dense_matrix",
    "dense_pauli_spectrum_probabilities",
    "dense_operator_schmidt_spectrum",
    "renyi_entropy",
]

#: Largest `n_left · n_right` `operator_schmidt_values` will allocate.
#: 4e6 `complex128` entries is 64 MiB; the SVD of a matrix that size costs a
#: few seconds. Callers wanting more must pass `max_entries` explicitly and
#: own the consequences.
MAX_SCHMIDT_ENTRIES = 4_000_000

#: Largest `n` for the brute-force `4^n`-Pauli dense spectrum. The loop is
#: `16^n` complex multiplications: 1.7e7 at n=6 (well under a second), 4.3e9
#: at n=8 (tens of seconds), 1.2e12 at n=10 (hopeless).
MAX_DENSE_SPECTRUM_N = 8

#: Single-qubit Paulis in the Hermitian convention (`Y` real-antisymmetric
#: times `i`, i.e. the physical `Y`), indexed by character.
PAULI_MATRICES = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}

_PAULI_CHARS = ("I", "X", "Y", "Z")


# ---------------------------------------------------------------------------
# Generic entropy helpers
# ---------------------------------------------------------------------------


def renyi_entropy(probabilities: Iterable[float], alpha: float = 2.0) -> float:
    """Rényi-α entropy of a probability vector, in nats.

    `alpha == 1` is evaluated as the Shannon limit `-sum p ln p` (the α → 1
    limit of the Rényi family, not a separate definition); `alpha == inf`
    gives the min-entropy `-ln max p`. Entries that are exactly zero
    contribute nothing in every case, which is the standard convention and
    also what keeps the `0 · ln 0` term out of the sum.

    The input is required to be a normalized, non-negative distribution: this
    function never renormalizes silently, because the whole point of
    `hilbert_schmidt_weight` existing separately is that a lost norm is a
    reportable quantity, not something to absorb.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.size == 0:
        raise ValueError("cannot take the entropy of an empty distribution")
    if np.any(p < -1e-15):
        raise ValueError(f"probabilities must be non-negative, got min {p.min()!r}")
    total = float(p.sum())
    if not np.isfinite(total) or abs(total - 1.0) > 1e-9:
        raise ValueError(f"probabilities must sum to 1, got {total!r}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha!r}")

    nz = p[p > 0]
    if np.isinf(alpha):
        return float(-np.log(nz.max()))
    if abs(alpha - 1.0) < 1e-12:
        return float(-(nz * np.log(nz)).sum())
    return float(np.log((nz**alpha).sum()) / (1.0 - alpha))


# ---------------------------------------------------------------------------
# 1. Pauli-spectrum entropies
# ---------------------------------------------------------------------------


def _coefficients(pauli_sum) -> np.ndarray:
    coefficients = np.asarray(pauli_sum.coefficients_array())
    if coefficients.size == 0:
        raise ValueError(
            "the Pauli sum is empty: every diagnostic here is a property of the "
            "normalized coefficient distribution, which does not exist for a sum "
            "with no terms (truncation may have removed everything)"
        )
    return coefficients


def hilbert_schmidt_weight(pauli_sum) -> float:
    """`sum_P |c_P|^2 = tr(O^† O) / 2^n`, the un-normalized spectral weight.

    Exactly `1` for a single Pauli string with unit coefficient, and preserved
    by exact unitary Heisenberg evolution (each channel is an orthogonal
    rotation of the coefficient vector); anything less is weight thrown away
    by truncation. Reported alongside every diagnostic so the renormalization
    the entropies perform is visible rather than implicit.
    """
    return float((np.abs(_coefficients(pauli_sum)) ** 2).sum())


def pauli_spectrum_probabilities(pauli_sum) -> np.ndarray:
    """`p_P = |c_P|^2 / sum_Q |c_Q|^2`, in the sum's own (unspecified) order.

    Storage order is bucketed and free to change (CLAUDE.md §Determinism
    policy), so this is a multiset, not a labelled vector -- every consumer
    here is a symmetric function of it. Terms with an exactly-zero coefficient
    stay in the vector as `0.0` and contribute nothing to any entropy.
    """
    weights = np.abs(_coefficients(pauli_sum)) ** 2
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(
            "every coefficient in the Pauli sum is zero, so there is no spectrum "
            "to normalize"
        )
    return weights / total


def pauli_spectrum_renyi(pauli_sum, alpha: float = 2.0) -> float:
    """Rényi-α entropy (nats) of the Pauli spectrum. See the module docstring."""
    return renyi_entropy(pauli_spectrum_probabilities(pauli_sum), alpha)


def pauli_spectrum_renyi2(pauli_sum) -> float:
    """`S_2 = -ln sum_P p_P^2`, the α=2 Pauli-spectrum entropy (nats).

    Zero for a single Pauli string; unchanged by Clifford conjugation; grows
    when a non-Clifford rotation spreads coefficient weight. Equal to the
    Operator Stabilizer Rényi entropy `S^2(O)` of arXiv:2510.22311 whenever
    `hilbert_schmidt_weight(O) == 1`.
    """
    return pauli_spectrum_renyi(pauli_sum, 2.0)


def pauli_spectrum_linear(pauli_sum) -> float:
    """`L = 1 - sum_P p_P^2`, the linear variant.

    The same content as `pauli_spectrum_renyi2` with the log removed:
    `S_2 = -ln(1 - L)` identically. Bounded in `[0, 1)`, zero for a single
    Pauli string, and often the more readable of the two when the spectrum is
    close to a single term.
    """
    p = pauli_spectrum_probabilities(pauli_sum)
    return float(1.0 - (p**2).sum())


def pauli_spectrum_shannon(pauli_sum) -> float:
    """`-sum_P p_P ln p_P` (nats), the α → 1 participation entropy."""
    return pauli_spectrum_renyi(pauli_sum, 1.0)


def pauli_spectrum_renyi_unnormalized(pauli_sum, alpha: float = 2.0) -> float:
    """The literal OSE of arXiv:2510.22311 Definition 1, without renormalizing:

        S^α(O) = (α / (1 - α)) · ln ( sum_i c_i^{2α} )^{1/α} .

    Equals `pauli_spectrum_renyi(pauli_sum, alpha)` exactly when
    `hilbert_schmidt_weight(pauli_sum) == 1`, and differs from it by
    `(α/(1-α)) · ln(hilbert_schmidt_weight)` otherwise -- so for a truncated
    sum, where weight has been discarded, this one drifts with the *amount*
    of surviving weight and the renormalized one does not. Provided for
    completeness; the showcase plots the renormalized version.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha!r}")
    if abs(alpha - 1.0) < 1e-12:
        raise ValueError(
            "the unnormalized OSE formula has a 1/(1-alpha) pole at alpha=1; use "
            "pauli_spectrum_shannon for the renormalized alpha -> 1 limit"
        )
    weights = np.abs(_coefficients(pauli_sum)) ** 2
    moment = float((weights**alpha).sum())
    if moment <= 0.0:
        raise ValueError("every coefficient in the Pauli sum is zero")
    return float(alpha / (1.0 - alpha) * np.log(moment ** (1.0 / alpha)))


# ---------------------------------------------------------------------------
# 2. Operator entanglement
# ---------------------------------------------------------------------------


def _bit_mask(num_qubits: int, lo: int, hi: int) -> np.ndarray:
    """Word-packed `uint64` mask selecting symplectic bits `[lo, hi)`.

    Matches the storage layout `x_array()` / `z_array()` expose: bit `b` of
    qubit index `b` lives in word `b // 64` at position `b % 64`.
    """
    words = (num_qubits + 63) // 64
    mask = np.zeros(words, dtype=np.uint64)
    for bit in range(lo, hi):
        mask[bit // 64] |= np.uint64(1) << np.uint64(bit % 64)
    return mask


def _normalize_cut(pauli_sum, cut: int | None) -> int:
    n = pauli_sum.num_qubits
    if cut is None:
        cut = n // 2
    if not 1 <= cut <= n - 1:
        raise ValueError(
            f"cut must be a nontrivial bipartition boundary in 1..{n - 1}, got {cut}"
        )
    return cut


def _factor_indices(pauli_sum, cut: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Per-term indices into the distinct left/right Pauli factors.

    A term's left factor is the `(x, z)` bit pair restricted to qubits
    `[0, cut)` and its right factor the restriction to `[cut, n)`; the pair of
    the two is the full symplectic key, which the bucketed storage guarantees
    is unique, so `(left_index, right_index)` is unique per term as well.
    """
    n = pauli_sum.num_qubits
    terms = len(pauli_sum)
    if terms == 0:
        raise ValueError(
            "the Pauli sum is empty: there is no operator Schmidt matrix for a sum "
            "with no terms (truncation may have removed everything)"
        )
    xs = np.asarray(pauli_sum.x_array()).reshape(terms, -1)
    zs = np.asarray(pauli_sum.z_array()).reshape(terms, -1)
    lo = _bit_mask(n, 0, cut)
    hi = _bit_mask(n, cut, n)
    left_keys = np.concatenate([xs & lo, zs & lo], axis=1)
    right_keys = np.concatenate([xs & hi, zs & hi], axis=1)
    left_unique, left_index = np.unique(left_keys, axis=0, return_inverse=True)
    right_unique, right_index = np.unique(right_keys, axis=0, return_inverse=True)
    return (
        np.asarray(left_index).reshape(-1),
        np.asarray(right_index).reshape(-1),
        left_unique.shape[0],
        right_unique.shape[0],
    )


def schmidt_matrix_shape(pauli_sum, cut: int | None = None) -> tuple[int, int]:
    """`(n_left, n_right)` -- the shape `schmidt_matrix` would allocate.

    Cheap relative to the SVD (two `np.unique` passes over the key columns),
    so callers can size-check a run before committing to it.
    """
    cut = _normalize_cut(pauli_sum, cut)
    _, _, n_left, n_right = _factor_indices(pauli_sum, cut)
    return n_left, n_right


def schmidt_matrix(
    pauli_sum, cut: int | None = None, *, max_entries: int = MAX_SCHMIDT_ENTRIES
) -> np.ndarray:
    """The operator Schmidt matrix `M[a, b] = c_{a ⊗ b}` for the cut at `cut`.

    Rows index distinct left (`[0, cut)`) Pauli factors, columns distinct
    right factors, both in `np.unique` order (irrelevant: only the singular
    values are ever used). Raises before allocating if the result would
    exceed `max_entries`.
    """
    cut = _normalize_cut(pauli_sum, cut)
    left_index, right_index, n_left, n_right = _factor_indices(pauli_sum, cut)
    entries = n_left * n_right
    if entries > max_entries:
        raise ValueError(
            f"the operator Schmidt matrix for cut={cut} would be "
            f"{n_left} x {n_right} = {entries} complex128 entries "
            f"({entries * 16 / 2**20:.0f} MiB), over the {max_entries}-entry guard. "
            "Tighten the truncation, shrink the circuit, or raise max_entries "
            "deliberately -- the SVD cost grows as n_left · n_right · "
            "min(n_left, n_right)."
        )
    matrix = np.zeros((n_left, n_right), dtype=np.complex128)
    # Additive rather than assigning: the (left, right) index pairs are unique
    # by construction (see `_factor_indices`), so this is only defensive, but
    # it costs nothing and cannot silently drop a term if that ever changes.
    np.add.at(matrix, (left_index, right_index), _coefficients(pauli_sum))
    return matrix


def operator_schmidt_values(
    pauli_sum, cut: int | None = None, *, max_entries: int = MAX_SCHMIDT_ENTRIES
) -> np.ndarray:
    """Singular values `s_k` of the operator Schmidt matrix, descending.

    `sum_k s_k^2 == hilbert_schmidt_weight(pauli_sum)` (Frobenius norm of
    `M`), which the test file pins.
    """
    matrix = schmidt_matrix(pauli_sum, cut, max_entries=max_entries)
    return np.linalg.svd(matrix, compute_uv=False)


def operator_schmidt_spectrum(
    pauli_sum, cut: int | None = None, *, max_entries: int = MAX_SCHMIDT_ENTRIES
) -> np.ndarray:
    """Normalized operator Schmidt weights `λ_k = s_k^2 / sum_j s_j^2`."""
    values = operator_schmidt_values(pauli_sum, cut, max_entries=max_entries)
    return _normalized_squares(values)


def _normalized_squares(singular_values: np.ndarray) -> np.ndarray:
    weights = np.asarray(singular_values, dtype=float) ** 2
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("the operator Schmidt matrix is identically zero")
    return weights / total


def operator_entanglement_entropy(
    pauli_sum, cut: int | None = None, *, max_entries: int = MAX_SCHMIDT_ENTRIES
) -> float:
    """`-sum_k λ_k ln λ_k` (nats) for the bipartition at `cut`.

    The operator space entanglement entropy of Prosen and Pižorn (PRA 76,
    032316), computed from the Pauli-basis Schmidt matrix. Zero for a single
    Pauli string (rank-1 `M`).
    """
    spectrum = operator_schmidt_spectrum(pauli_sum, cut, max_entries=max_entries)
    return renyi_entropy(spectrum, 1.0)


def operator_entanglement_renyi2(
    pauli_sum, cut: int | None = None, *, max_entries: int = MAX_SCHMIDT_ENTRIES
) -> float:
    """`-ln sum_k λ_k^2` (nats): the α=2 operator entanglement entropy."""
    spectrum = operator_schmidt_spectrum(pauli_sum, cut, max_entries=max_entries)
    return renyi_entropy(spectrum, 2.0)


# ---------------------------------------------------------------------------
# 3. Dense oracles
# ---------------------------------------------------------------------------


def _kron_string(label: str) -> np.ndarray:
    """`σ_{label[0]} ⊗ σ_{label[1]} ⊗ ... ` -- qubit 0 leftmost/most significant.

    The ordering is a *choice* fixed here and used consistently by both dense
    oracles: bit `0` of a row index is qubit `0`'s, so the bipartition
    `[0, cut)` is the high-order half of the index and the reshape in
    `dense_operator_schmidt_spectrum` lines up with `schmidt_matrix`'s
    `_bit_mask(n, 0, cut)` rows. (Note this is the opposite of qiskit's
    little-endian labelling; nothing here talks to qiskit.)
    """
    return functools.reduce(np.kron, (PAULI_MATRICES[ch] for ch in label))


def dense_matrix(terms) -> np.ndarray:
    """`sum_P c_P · P` as a dense `2^n x 2^n` `complex128` matrix.

    `terms` is a sequence of `(label, coefficient)` pairs -- e.g. from
    `examples/common/oracles.py::pauli_terms`, which decodes a `PauliSum` via
    its label export. Building the reference this way, from labels and
    `numpy.kron`, keeps it independent of the bit-twiddling in
    `_factor_indices`: the two share the numpy export and nothing else.
    """
    terms = list(terms)
    if not terms:
        raise ValueError("cannot build a dense matrix from an empty term list")
    lengths = {len(label) for label, _ in terms}
    if len(lengths) != 1:
        raise ValueError(f"Pauli labels have mixed lengths {sorted(lengths)}")
    n = lengths.pop()
    dim = 1 << n
    out = np.zeros((dim, dim), dtype=np.complex128)
    for label, coefficient in terms:
        out += complex(coefficient) * _kron_string(label)
    return out


def dense_pauli_spectrum_probabilities(
    matrix: np.ndarray, num_qubits: int, *, max_qubits: int = MAX_DENSE_SPECTRUM_N
) -> np.ndarray:
    """Brute-force `p_P` over all `4^n` Paulis from a dense matrix.

    `c_P = tr(P O) / 2^n` for every one of the `4^n` strings, in
    `itertools.product("IXYZ", repeat=n)` order, normalized to a probability
    vector. Deliberately the dumbest possible implementation (build `P`,
    contract, divide) so it can be trusted as an oracle; the `16^n` cost is
    why `max_qubits` exists.
    """
    if num_qubits > max_qubits:
        raise ValueError(
            f"the brute-force {4**num_qubits}-Pauli spectrum at n={num_qubits} costs "
            f"~16^n = {16**num_qubits:.2e} complex multiplications, past the "
            f"n<={max_qubits} guard"
        )
    dim = 1 << num_qubits
    if matrix.shape != (dim, dim):
        raise ValueError(f"expected a {dim}x{dim} matrix for n={num_qubits}, got {matrix.shape}")
    weights = np.empty(4**num_qubits, dtype=float)
    for index, chars in enumerate(itertools.product(_PAULI_CHARS, repeat=num_qubits)):
        pauli = _kron_string("".join(chars))
        # tr(P O) without forming the product: sum_{i,j} P[i,j] O[j,i].
        coefficient = np.einsum("ij,ji->", pauli, matrix) / dim
        weights[index] = abs(coefficient) ** 2
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("the dense matrix is traceless against every Pauli string")
    return weights / total


def dense_operator_schmidt_spectrum(
    matrix: np.ndarray, num_qubits: int, cut: int
) -> np.ndarray:
    """Operator Schmidt weights `λ_k` from the dense matrix, by reshape + SVD.

    The textbook route: view `O` as a vector in `H_A ⊗ H_A^* ⊗ H_B ⊗ H_B^*`,
    regroup the `A` and `B` index pairs into a `(4^|A|, 4^|B|)` matrix and
    SVD. Shares no code with `schmidt_matrix` -- it never looks at a Pauli
    label or a symplectic bit -- which is what makes it an independent check
    of it. Cost `O(8^n)`.
    """
    if not 1 <= cut <= num_qubits - 1:
        raise ValueError(f"cut must be in 1..{num_qubits - 1}, got {cut}")
    dim = 1 << num_qubits
    if matrix.shape != (dim, dim):
        raise ValueError(f"expected a {dim}x{dim} matrix for n={num_qubits}, got {matrix.shape}")
    da = 1 << cut
    db = 1 << (num_qubits - cut)
    # Row index -> (a_row, b_row), column index -> (a_col, b_col); qubit 0 is
    # the most significant bit, so the A block is the leading factor.
    tensor = matrix.reshape(da, db, da, db).transpose(0, 2, 1, 3).reshape(da * da, db * db)
    values = np.linalg.svd(tensor, compute_uv=False)
    return _normalized_squares(values)
