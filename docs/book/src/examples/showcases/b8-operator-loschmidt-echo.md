# B8 — Operator Loschmidt echo

*Source: `examples/b8_ole/`*

<p class="lead">The 56-qubit operator Loschmidt echo of arXiv:2607.25998 on ibm_boston, estimated by the paper's hybrid Pauli-propagation Monte Carlo (PP-MC), with caches that extend past one node through the MPI engine.</p>

## The circuit

The circuit is the Quantum Advantage Tracker's `operator_loschmidt_echo_56x1488`: a CZ kicked-Ising Floquet echo `U V_δ U†` on a heavy-hex patch.
Its 11 scattering qubits carry an `rx` angle shifted by `η = 3πα/2`.
`echo_half(L, η)` rebuilds the measured half `C` for any depth and reproduces the six vendored tracker files gate for gate.
The echo is `S_δ = 2⁻ⁿ Tr(A V_δ† A V_δ)`, where `A = C†OC`, `O = Z^{⊗12}`, and `V_δ = ⊗ e^{-iδX}` on 35 qubits with `δ = 0.3`.

## PP-MC

<!-- doctest: skip -->
```python
from paulistrings import truncation
policy = truncation.collapse_sample(cache=500_000_000, seed=s)
A = O.propagate(C, policy, direction="heisenberg")
S = diagonal_echo(A.anticommute_histogram(sites, axis="x"), 0.3)
```

- `collapse_sample` propagates exactly until a gate leaves more than `cache` strings.
- It then draws one string with probability `b_P²/Σb²` and restarts from it.
- The last block is kept whole.
- The diagonal estimator `Σ b_P² cos(2δ)^{n_P} / Σ b²` is the resummed moment series of the paper's Eq. (C34).
- `rotated_overlap` gives the exact echo of the same `A`, off-diagonal terms included.
- Under `comm=` both read-outs are collective, and `collapse_sample` draws the same string whatever the thread count.

## Cost

One L = 6 trajectory at η = 9π/40 on 8 threads of ccqlin038 (Xeon Gold 6244):

| cache | wall | peak RSS |
|---|---|---|
| 5e6 | 13 s | 1.0 GB |
| 5e7 | 131 s | 3.9 GB |

For comparison, the paper's Table VI reports about 8 minutes on 10 cores and 8–10 GB at cache 5e7.
`scripts/slurm/ole-ppmc-disbatch.sbatch` runs the 800-trajectory sweep, and `scripts/slurm/ole-mpi.sbatch` runs caches beyond one node's memory.
