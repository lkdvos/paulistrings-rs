# B8 — Operator Loschmidt echo

*Source: `examples/b8_ole/`*

<p class="lead">The 56-qubit operator Loschmidt echo of arXiv:2607.25998 on ibm_boston, estimated by a hybrid Pauli-propagation Monte Carlo (PP-MC) over pairs of independent sampled trajectories, with caches that extend past one node through the MPI engine.</p>

## The circuit

The circuit is the Quantum Advantage Tracker's `operator_loschmidt_echo_56x1488`: a CZ kicked-Ising Floquet echo `U V_δ U†` on a heavy-hex patch.
Its 11 scattering qubits carry an `rx` angle shifted by `η = 3πα/2`.
`echo_half(L, η)` rebuilds the measured half `C` for any depth and reproduces the six vendored tracker files gate for gate.
The echo is `S_δ = 2⁻ⁿ Tr(A V_δ† A V_δ)`, where `A = C†OC`, `O = Z^{⊗12}`, and `V_δ = ⊗ e^{-iδX}` on 35 qubits with `δ = 0.3`.

## PP-MC

<!-- doctest: skip -->
```python
import ole
from paulistrings import propagated, truncation
def trajectory(seed):
    policy = truncation.collapse_sample(cache=500_000_000, samples=100_000_000, seed=seed)
    return propagated(O, C, policy, direction="heisenberg")

A, A2 = trajectory(2 * s), trajectory(2 * s + 1)
S = ole.diagonal_echo(ole.anticommute_histogram(A, sites, "x", B=A2), 0.3, normalize=False)
```

- `collapse_sample` propagates exactly until a gate leaves more than `cache` strings.
- It then replaces the sum by `samples` systematic draws by `|b_P|`, each worth `Σ|b|/samples`, so every collapse and the final `A` are unbiased.
- The echo is quadratic in `A`, so each estimate pairs two independent trajectories; its expectation is the exact echo.
- The diagonal estimator `Σ b_P b'_P cos(2δ)^{n_P}` is the resummed moment series of the paper's Eq. (C34), unnormalized since `‖A‖₂ = 1`.
- `ole.rotated_overlap(A, ..., B=A2)` gives the exact echo of the pair, off-diagonal terms included.
- The paper's PP-MC instead restarts from one string drawn by `b_P²` at coefficient 1 and normalizes a single trajectory's diagonal echo.
- Both estimators are numpy over `A.to_arrays()`; under MPI each rank reads its share of a `PauliSum.partition(comm=...)` split (`.local()`) and `comm.allreduce` sums the values.
- `collapse_sample` draws the same strings whatever the thread count.

## Cost

One L = 6 trajectory of the collapse-to-one rule this example used to run, at η = 9π/40 on 8 threads of ccqlin038 (Xeon Gold 6244):

| cache | wall | peak RSS |
|---|---|---|
| 5e6 | 13 s | 1.0 GB |
| 5e7 | 131 s | 3.9 GB |

For comparison, the paper's Table VI reports about 8 minutes on 10 cores and 8–10 GB at cache 5e7.
`scripts/slurm/ole-ppmc-disbatch.sbatch` runs the 800-trajectory sweep, and `scripts/slurm/ole-mpi.sbatch` runs caches beyond one node's memory.
