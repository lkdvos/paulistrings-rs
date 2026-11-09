# B8 — operator Loschmidt echo on 56 heavy-hex qubits

Estimates the operator Loschmidt echo of arXiv:2607.25998 by a hybrid Pauli-propagation Monte Carlo (PP-MC) in the spirit of its App. C 2, with caches beyond one node and deterministic truncation baselines.

## The problem

The circuit is the tracker's `operator_loschmidt_echo_56x1488`: a CZ kicked-Ising Floquet echo on 56 qubits of ibm_boston, with 11 scattering qubits whose `rx` angle is shifted by `η = 3πα/2`.
The files for α ∈ {0, 0.05, …, 0.25} are vendored under `examples/data/ole56/`.
`ole.echo_half(L, η)` rebuilds the measured half `C` of the echo for any `(L, η)` and reproduces every vendored circuit gate for gate.
The echo is `S_δ = 2⁻ⁿ Tr(A V_δ† A V_δ)`, where `A = C†OC`, `O = Z^{⊗12}` on the fast loop, and `V_δ = ⊗ e^{-iδX}` on the 35 perturbation qubits, with δ = 0.3.

## The estimators

The estimators in `ole.py` are numpy over `PauliSum.to_arrays()`.
`ole.anticommute_histogram` gives `w_n = Σ_{n_P = n} b_P²`, where `n_P` counts the perturbation sites whose `X` anticommutes with `P`; given a second sum `B`, it gives `w_n = Σ_{n_P = n} Re(b_P b'_P)` instead.
The paper's diagonal OLE is `ole.diagonal_echo`, `S_diag = Σ_n w_n cos(2δ)ⁿ / Σ_n w_n`, which is the closed form of its moment series Eq. (C34).
`truncated_series` gives the truncated sum that Fig. 17(b) scans.
`ole.rotated_overlap` is the exact `S_δ` of the propagated `A`, off-diagonal terms included: strings that `V_δ† · V_δ` mixes form classes, and each class of `k` strings costs `k²` pairs; given `B`, it is `2⁻ⁿ Re Tr(A V_δ† B V_δ)`.
Under `--mpi` each rank evaluates both on its `result="local"` share and the values are summed over ranks, which is exact for the overlap because the partition rows exclude the probe sites' x-bits.

## PP-MC

`truncation.collapse_sample(cache, samples, seed)` propagates exactly until a gate leaves more than `cache` strings.
It then replaces the sum by `samples` systematic draws in proportion to `|b_P|`, each worth `Σ|b|/samples` with the sign of its coefficient, so every collapse is unbiased and so is the trajectory's final `A`.
The echo is quadratic in `A`, so a trajectory's own echo is biased; `run_b8.py --policy ppmc` pairs consecutive seeds into two independent trajectories `A`, `A'` and records the bilinear echoes `Σ_n w_n cos(2δ)ⁿ` (with `w_n` from both sums) and `2⁻ⁿ Re Tr(A V_δ† A' V_δ)`, whose expectations are the exact diagonal and full echoes.
They are not divided by a norm: `‖A‖₂ = 1` exactly for the untruncated unitary evolution, and a ratio of estimates would be biased.
The reported value is the mean over pairs with its standard error.
This differs from the paper's PP-MC, which restarts from one string drawn by `b_P²` at coefficient 1 and normalizes the diagonal echo of a single trajectory; that estimator drops the interference between branches.
The paper runs 800 trajectories at cache 5e8 (Table VI).

## Running

```bash
python examples/b8_ole/run_b8.py --alpha 0.15 --L 6 --policy ppmc --cache 5e8 --samples 1e8 --seeds 0:20 --out results/ppmc
python examples/b8_ole/run_b8.py --alpha 0.15 --L 3 --policy approx_topn --cache 1e9 --exact-overlap --out results/det
python examples/b8_ole/aggregate.py results/ppmc results/det --csv summary.csv --plot ole_vs_eta.svg
```

- Cluster sweeps: `make_tasks.py` writes a disBatch task file for `scripts/slurm/ole-ppmc-disbatch.sbatch`.
- Caches beyond one node: `scripts/slurm/ole-mpi.sbatch` runs `--mpi` with one partition per NUMA domain.

Split runs (`--mpi`, `--partitions`) draw partition rows on x-bits only, because CZ and `rz` never change a string's x-bits and so only `rx` gates exchange.
At 4 ranks and cache 2e7 this used 2.3× less memory per rank and ran 1.6–2× faster than the default rows.
`--policy approx_topn --cache M` is the deterministic, unsampled baseline; report its retained norm next to its value.
Coefficient thresholds suit the echo badly: the forward half spreads the coefficients thin before the backward half refocuses them, so `--policy coeff` either keeps nothing or keeps everything.
`--exact-overlap` adds the exact `S_δ` next to the diagonal one.
A ppmc task holds both trajectories of a pair at once, one more final sum than a single trajectory.

## References

`spec.json` records the qubit sets, which are asserted against the QASM by `examples/tests/test_showcase_b8.py`.
It also holds the paper's Table I (experiment and belief propagation at η = 9π/40, L = 2..6) and the full-scrambling value 0.04083.
The PP-MC curve of Fig. 1(e) and Fig. 21 is published only as a figure.
