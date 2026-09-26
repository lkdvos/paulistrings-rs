# B8 — operator Loschmidt echo on 56 heavy-hex qubits

Reproduces the hybrid Pauli-propagation Monte Carlo (PP-MC) estimate of the operator Loschmidt echo in arXiv:2607.25998 (App. C 2), and extends it with caches beyond one node and with deterministic truncation baselines.

## The problem

The circuit is the tracker's `operator_loschmidt_echo_56x1488`: a CZ kicked-Ising Floquet echo on 56 qubits of ibm_boston, with 11 scattering qubits whose `rx` angle is shifted by `η = 3πα/2`.
The files for α ∈ {0, 0.05, …, 0.25} are vendored under `examples/data/ole56/`.
`ole.echo_half(L, η)` rebuilds the measured half `C` of the echo for any `(L, η)` and reproduces every vendored circuit gate for gate.
The echo is `S_δ = 2⁻ⁿ Tr(A V_δ† A V_δ)`, where `A = C†OC`, `O = Z^{⊗12}` on the fast loop, and `V_δ = ⊗ e^{-iδX}` on the 35 perturbation qubits, with δ = 0.3.

## The estimators

`anticommute_histogram` gives `w_n = Σ_{n_P = n} b_P²`, where `n_P` counts the perturbation sites whose `X` anticommutes with `P`.
The paper's diagonal OLE is `S_diag = Σ_n w_n cos(2δ)ⁿ / Σ_n w_n`, which is the closed form of its moment series Eq. (C34).
`truncated_series` gives the truncated sum that Fig. 17(b) scans.
`rotated_overlap` is the exact `S_δ` of the propagated `A`, off-diagonal terms included.

## PP-MC

`truncation.collapse_sample(cache, seed)` propagates exactly until a gate leaves more than `cache` strings.
It then draws one string with probability `b_P² / Σ b²`, and propagation restarts from that string with coefficient 1.
The last block is kept whole.
One seed is one trajectory, and the reported value is the mean over trajectories with its standard error.
The paper runs 800 trajectories at cache 5e8 (Table VI).

## Running

```bash
python examples/b8_ole/run_b8.py --alpha 0.15 --L 6 --policy ppmc --cache 5e8 --seeds 0:20 --out results/ppmc
python examples/b8_ole/run_b8.py --alpha 0.15 --L 3 --policy coeff --eps 1e-7 --exact-overlap --out results/det
python examples/b8_ole/aggregate.py results/ppmc results/det --csv summary.csv --plot ole_vs_eta.svg
```

- Cluster sweeps: `make_tasks.py` writes a disBatch task file for `scripts/slurm/ole-ppmc-disbatch.sbatch`.
- Caches beyond one node: `scripts/slurm/ole-mpi.sbatch` runs `--mpi` with one partition per NUMA domain.

`--policy approx_topn --cache M` and `--policy coeff --eps ε` are the deterministic, unsampled baselines.
`--exact-overlap` adds the exact `S_δ` next to the diagonal one.

## References

`spec.json` records the qubit sets, which are asserted against the QASM by `examples/tests/test_showcase_b8.py`.
It also holds the paper's Table I (experiment and belief propagation at η = 9π/40, L = 2..6) and the full-scrambling value 0.04083.
The PP-MC curve of Fig. 1(e) and Fig. 21 is published only as a figure.
