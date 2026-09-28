"""PP-MC trajectories and deterministic baselines for the 56-qubit OLE; one JSON per seed.

    python run_b8.py --alpha 0.15 --L 6 --policy ppmc --cache 5e8 --seeds 0:40 --out results/ppmc
    mpirun -n 8 python run_b8.py --mpi --alpha 0.05 --L 6 --policy ppmc --cache 8e9 --seeds 0:4 --out results/mpi
    python run_b8.py --alpha 0.15 --L 3 --policy coeff --eps 1e-6 --exact-overlap --out results/exact
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ole  # noqa: E402


def _seeds(text: str) -> list[int]:
    if ":" in text:
        lo, hi = (int(x) for x in text.split(":"))
        return list(range(lo, hi))
    return [int(x) for x in text.split(",")]


def _git_rev() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_policy(args, seed: int):
    from paulistrings import truncation

    parts = []
    if args.policy == "ppmc":
        parts.append(truncation.collapse_sample(int(args.cache), seed))
    elif args.policy == "approx_topn":
        parts.append(truncation.approx_topn(int(args.cache)))
    if args.eps is not None:
        parts.append(truncation.coeff(args.eps))
    if args.max_weight is not None:
        parts.append(truncation.weight(args.max_weight))
    policy = None
    for p in parts:
        policy = p if policy is None else policy & p
    return policy


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--alpha", type=float, help="tracker scattering parameter; eta = 3 pi alpha / 2")
    g.add_argument("--eta", type=float, help="scattering strength eta in radians")
    ap.add_argument("--L", type=int, default=6, help="Floquet layers per half of U")
    ap.add_argument("--delta", type=float, default=None, help="probe strength (default: spec, 0.3)")
    ap.add_argument("--policy", choices=["ppmc", "approx_topn", "coeff", "none"], default="ppmc")
    ap.add_argument("--cache", type=float, default=5e8, help="PP-MC cache size, or the ApproxTopN budget")
    ap.add_argument("--eps", type=float, default=None, help="additional coefficient threshold")
    ap.add_argument("--max-weight", type=int, default=None)
    ap.add_argument("--seeds", default="0", help="'lo:hi' or comma list; one trajectory per seed")
    ap.add_argument("--exact-overlap", action="store_true", help="also compute the exact rotated_overlap S_delta")
    ap.add_argument("--no-snap-cliffords", action="store_true", help="import rz(k pi/2) as branching rotations, as before the fix")
    ap.add_argument("--mpi", action="store_true", help="one partition per MPI rank (needs the `mpi` build and mpi4py)")
    ap.add_argument("--partitions", default=None, help="in-process NUMA partitions: an int or 'auto'")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default=None, help="subdirectory name; default derived from the configuration")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    delta = ole.spec()["delta"] if args.delta is None else args.delta
    eta = ole.eta_of_alpha(args.alpha) if args.alpha is not None else args.eta

    comm = None
    rank = 0
    if args.mpi:
        import mpi4py

        mpi4py.rc.thread_level = "serialized"
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    snap = not args.no_snap_cliffords
    circuit = ole.to_circuit(ole.echo_half(args.L, eta), snap_cliffords=snap)
    obs = ole.observable()
    sites = ole.perturbation_sites()

    tag = args.tag or (
        f"L{args.L}_eta{eta:.4f}_d{delta:g}_{args.policy}"
        + (f"_M{args.cache:.0e}" if args.policy in ("ppmc", "approx_topn") else "")
        + (f"_eps{args.eps:.0e}" if args.eps is not None else "")
        + ("_snap" if snap else "")
    )
    outdir = args.out / tag
    if rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)

    # Partition rows on x-bits only: CZ and rz never change a string's x-bits, so only rx exchanges.
    # The exact overlap additionally needs the probe sites' x-bits unread.
    exclude = {"z": list(range(obs.num_qubits))}
    if args.exact_overlap:
        exclude["x"] = sites
    kwargs = {}
    if comm is not None:
        kwargs.update(comm=comm, result="local", partition_row_exclude=exclude)
    elif args.partitions is not None:
        kwargs["partitions"] = args.partitions if args.partitions == "auto" else int(args.partitions)
        kwargs["partition_row_exclude"] = exclude

    for seed in _seeds(args.seeds):
        t0 = time.perf_counter()
        result, stats = obs.propagate_with_stats(
            circuit, build_policy(args, seed), direction="heisenberg", **kwargs
        )
        t_prop = time.perf_counter() - t0
        hist = result.anticommute_histogram(sites, axis="x", comm=comm)
        record = {
            "config": {
                "L": args.L,
                "eta": eta,
                "alpha": args.alpha,
                "delta": delta,
                "policy": args.policy,
                "cache": int(args.cache) if args.policy in ("ppmc", "approx_topn") else None,
                "eps": args.eps,
                "max_weight": args.max_weight,
                "seed": seed,
                "mpi_ranks": comm.Get_size() if comm is not None else 1,
                "partitions": args.partitions,
                "snap_cliffords": snap,
            },
            "S_diag": ole.diagonal_echo(hist, delta),
            "norm": float(sum(hist)),
            "hist": [float(w) for w in hist],
            "collapses": getattr(stats, "collapses", None),
            "final_terms": getattr(stats, "final_terms", None),
            "propagate_s": t_prop,
        }
        if args.exact_overlap:
            t1 = time.perf_counter()
            raw = result.rotated_overlap(sites, delta, axis="x", comm=comm)
            record["S_exact"] = raw / record["norm"] if record["norm"] > 0 else math.nan
            record["overlap_s"] = time.perf_counter() - t1
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if comm is not None:
            from mpi4py import MPI

            peak_kb = comm.allreduce(peak_kb, op=MPI.MAX)
            record["final_terms"] = comm.allreduce(len(result), op=MPI.SUM)
        elif record["final_terms"] is None:
            record["final_terms"] = len(result)
        record["peak_rss_kb_max_rank"] = peak_kb
        if rank == 0:
            record["provenance"] = {
                "git_rev": _git_rev(),
                "host": socket.gethostname(),
                "python": platform.python_version(),
                "rayon_threads": os.environ.get("RAYON_NUM_THREADS"),
                "slurm_job": os.environ.get("SLURM_JOB_ID"),
                "argv": sys.argv,
            }
            (outdir / f"seed_{seed:06d}.json").write_text(json.dumps(record, indent=1))
            print(
                f"seed {seed}: S_diag={record['S_diag']:.5f}"
                + (f" S_exact={record['S_exact']:.5f}" if "S_exact" in record else "")
                + f" collapses={record['collapses']} terms={record['final_terms']} {t_prop:.1f}s",
                flush=True,
            )
        # The result keeps its peak bucket capacity; drop it before the next seed propagates.
        del result, stats
    return 0


if __name__ == "__main__":
    sys.exit(main())
