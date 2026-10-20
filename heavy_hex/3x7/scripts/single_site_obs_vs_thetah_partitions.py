#!/usr/bin/env python3
"""
Test NUMA partitioning on the single-site kicked-Ising run.

Runs the same computation as single_site_obs_vs_thetah.py:
<Z>_k = <0...0| (U^k)^dag Z_site U^k |0...0> for k = 1, ..., steps, where U is one
Trotter step of H = -J sum_<i,j> Z_i Z_j - h sum_i X_i on the heavy-hex bonds:
R_ZZ(-2 J dt) on every bond, then R_X(-2 h dt) on every qubit.

After every gate, terms with |c| <= eps are dropped. The time series is computed once per
cutoff eps, and the whole sweep is repeated for P = 1, 2, 4, ... NUMA
partitions.

Usage:
    python3 heavy_hex/3x7/scripts/single_site_obs_vs_thetah_partitions.py --config heavy_hex/3x7/3x7_partitions.json

Config:
    rows, width  heavy-hex size: odd rows, width = 3 (mod 4)
    J, h, dt     couplings and time step
    steps        number of Trotter steps
    site         measured qubit index
    cutoffs      list of truncation cutoffs eps

Output:
    heavy_hex/3x7/data/single_site_obs_vs_thetah_partitions.npz
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import paulistrings
from paulistrings import Circuit, PauliString, truncation

CONFIG_KEYS = {"rows", "width", "J", "h", "dt", "steps", "site", "cutoffs"}
OUTPUT = Path("data") / "single_site_obs_vs_thetah_partitions.npz"


def heavy_hex(rows, width):
    """
    Heavy-hex bond list in IBM numbering.

    Each row is a chain of `width` qubits; the first row loses its last qubit and
    the last row its first. Between rows r and r+1 a bridge qubit sits at columns
    0, 4, 8, ... (r even) or 2, 6, 10, ... (r odd). Qubits are numbered row by
    row, each row followed by its bridges.
    """
    if rows % 2 == 0 or width % 4 != 3:
        raise ValueError("heavy_hex needs odd rows and width = 3 (mod 4)")
    qubit, bridge, n = {}, {}, 0
    for r in range(rows):
        for c in range(width):
            if (r, c) not in {(0, width - 1), (rows - 1, 0)}:
                qubit[r, c] = n
                n += 1
        if r < rows - 1:
            for c in range(2 * (r % 2), width, 4):
                bridge[r, c] = n
                n += 1
    bonds = [
	    (q, qubit[r, c + 1]) for (r, c), q in qubit.items() if (r, c + 1) in qubit
	   ]
    for (r, c), b in bridge.items():
        bonds += [(qubit[r, c], b), (b, qubit[r + 1, c])]
        
    return n, bonds


def time_series(observable, step, policy, steps, P):
    """
    Per-step results for k = 1..steps, evaluated in |0...0>, with the Pauli sum
    split over P NUMA nodes (P = 1 is an unpartitioned run).

    Returns a dict of arrays: <Z>_k, term count, retained norm, wall time of
    the step in seconds, and rows exported between nodes during the step.
    """
    out = {
	    "expectation": np.zeros(steps), 
	    "terms": np.zeros(steps, dtype=int),
        "retained": np.zeros(steps), 
        "seconds": np.zeros(steps),
        "exported": np.zeros(steps, dtype=int)
    }

    for k in range(1, steps + 1):
        # Apply one more Trotter step in the Heisenberg picture
        start = time.perf_counter()
        observable, stats = observable.propagate_with_stats(
            step, policy, direction="heisenberg", partitions=P
        )
        out["seconds"][k - 1] = time.perf_counter() - start

        # <Z>_k is evaluated in |0...0>, the state before the quench.
        # "retained" is the sum of |c|^2. It starts at 1 for a single Pauli
        # string, and only truncation lowers it.
        out["expectation"][k - 1] = observable.expectation("z+").real
        out["terms"][k - 1] = len(observable)
        out["retained"][k - 1] = observable.overlap(observable).real
        
        if stats.partition is not None:
            out["exported"][k - 1] = sum(stats.partition.rows_exported)
        print(
	        f"{k:3d} {out['expectation'][k - 1]:+14.9f} {out['terms'][k - 1]:10d} "
	        f"{out['retained'][k - 1]:9.6f} {out['seconds'][k - 1]:9.3f} "
	        f"{out['exported'][k - 1]:11d}"
	      )
              
    return out


def save_results(path, partition_counts, cutoffs, results, parameters):
    """
    Save the time series for plotting.

    results[i][j] is the dict from time_series for partition_counts[i] and
    cutoffs[j]. Each quantity is stored as an array of shape
    (len(partition_counts), len(cutoffs), steps), where the last index is k - 1.
    """
    arrays = {
	    key: np.array([[run[key] for run in runs] for runs in results])
      for key in results[0][0]
    }
    steps = arrays["expectation"].shape[-1]
    np.savez(
	    path, 
	    partition_counts=np.array(partition_counts), 
	    cutoffs=np.array(cutoffs),
        k=np.arange(1, steps + 1), 
        **arrays, **parameters
    )


def load_config(path):
    """Read the run parameters from a JSON file"""
    with open(path) as f:
        config = json.load(f)
    missing, unknown = CONFIG_KEYS - set(config), set(config) - CONFIG_KEYS
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    return config


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="JSON file with the run parameters")
    parser.add_argument("--output-dir", default=None, help="directory for the .npz")
    args = parser.parse_args()
    config = load_config(args.config)

    rows, width = config["rows"], config["width"]
    J, h, dt = config["J"], config["h"], config["dt"]
    steps, site, cutoffs = config["steps"], config["site"], config["cutoffs"]
    output = Path(args.output_dir) / OUTPUT.name if args.output_dir else Path(args.config).parent / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)

    nodes = len(paulistrings.numa_nodes())
    partition_counts = [2**m for m in range(min(nodes.bit_length(), 7))]

    n, bonds = heavy_hex(rows, width)

    # One Trotter step, with R_ZZ(-2 J dt) on every bond, then R_X(-2 h dt) on every qubit.
    step = Circuit(n)
    for i, j in bonds:
        step.pauli_rotation("ZZ", [i, j], -2 * J * dt)
    for q in range(n):
        step.rx(-2 * h * dt, q)
    parameters = dict(rows=rows, width=width, J=J, h=h, dt=dt, site=site)

    results = []
    for P in partition_counts:
        # warm-up
        time_series(1.0 * PauliString.z(site, n), step, truncation.coeff(cutoffs[0]), steps, P)
        runs = []
        for eps in cutoffs:
            print(f"\nP = {P}, coeff({eps:g})")
            observable = 1.0 * PauliString.z(site, n)
            runs.append(
	            time_series(observable, step, truncation.coeff(eps), steps, P)
	          )
        results.append(runs)
        save_results(output, partition_counts[:len(results)], cutoffs, results, parameters)

if __name__ == "__main__":
    main()