"""Per-partition truncation against the global budget on the heavy-hex kicked Ising point of Benchmark C (issue #12, part 3).

At a total budget `N` and `P` partitions it runs `approx_topn(N)` (partition-exact, the baseline), `partition_topn(N/P)` and `partition_approx_topn(N/P)`, and scores each against Benchmark C's reference for `Z_62`.
One JSON line per run: the expectation value, its error against the reference and its difference from `approx_topn(N)`, wall time, and the largest share any partition held after any layer.
`topn(N)` and `approx_topn(N)` unpartitioned anchor each budget; the per-partition policies repeat over `--row-seeds`, since which strings share a partition is what they are sensitive to.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "examples"))

import paulistrings  # noqa: E402
from common import circuits, observables  # noqa: E402
from paulistrings import truncation  # noqa: E402

THETAS = {"7pi/32": 7 * math.pi / 32, "5pi/16": 5 * math.pi / 16}
REFERENCES = _REPO_ROOT / "benchmarks/python/deep_trotter/summary.json"


def cpu_lists(partitions: int) -> list[list[int]]:
    """The affinity mask cut into `partitions` contiguous CPU lists."""
    cpus = sorted(os.sched_getaffinity(0))
    per = len(cpus) // partitions
    if per == 0:
        raise SystemExit(f"{partitions} partitions need at least {partitions} CPUs, have {len(cpus)}")
    return [cpus[r * per : (r + 1) * per] for r in range(partitions)]


def run(circuit, observable, policy, partitions: int, row_seed: int | None) -> dict:
    kwargs = {} if partitions == 1 else {"partitions": cpu_lists(partitions), "partition_row_seed": row_seed}
    start = time.perf_counter()
    evolved, stats = paulistrings.propagated_with_stats(observable, circuit, policy, direction="heisenberg", **kwargs)
    wall = time.perf_counter() - start
    value = evolved.expectation("z+")
    part = stats.partition
    return {
        "value": value.real,
        "wall_s": wall,
        "final_terms": stats.final_terms,
        "peak_terms": stats.peak_terms,
        "max_partition_terms": max(max(layer) for layer in part.terms_out),
        "max_imbalance": max(part.imbalance),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 9])
    parser.add_argument("--thetas", nargs="+", default=list(THETAS), choices=list(THETAS))
    parser.add_argument("--budgets", type=float, nargs="+", default=[1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7])
    parser.add_argument("--partitions", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--row-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    references = json.loads(REFERENCES.read_text())["references"]
    observable = observables.canonical_z_127()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as out:

        def emit(record: dict) -> None:
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(
                f"{record['steps']:>2} {record['theta']:>6} N={record['budget']:<9.0e} P={record['partitions']:<2} "
                f"seed={record['row_seed']} {record['policy']:<22} err={record['error']:+.3e} "
                f"vs-global={record['delta_vs_global']:+.3e} max/part={record['max_partition_terms']} "
                f"{record['wall_s']:.1f}s",
                flush=True,
            )

        for steps in args.steps:
            for theta_label in args.thetas:
                key = f"{theta_label}@{steps}steps"
                if key not in references:
                    raise SystemExit(f"no Benchmark C reference for {key}")
                ref = references[key]
                circuit = circuits.heavy_hex_kicked_ising(127, trotter_steps=steps, theta_h=THETAS[theta_label])
                for budget in map(int, args.budgets):
                    point = {
                        "steps": steps,
                        "theta": theta_label,
                        "budget": budget,
                        "reference": ref["reference_value"],
                        "reference_exact": ref["reference_exact"],
                        "reference_uncertainty": ref["reference_uncertainty"],
                        "paulistrings": paulistrings.__version__,
                    }
                    global_value = None

                    def record(policy_name, policy, partitions, row_seed):
                        result = run(circuit, observable, policy, partitions, row_seed)
                        result["error"] = result["value"] - ref["reference_value"]
                        result["delta_vs_global"] = (
                            0.0 if global_value is None else result["value"] - global_value
                        )
                        emit({**point, "policy": policy_name, "partitions": partitions, "row_seed": row_seed, **result})
                        return result["value"]

                    global_value = record("approx_topn", truncation.approx_topn(budget), 1, None)
                    record("topn", truncation.topn(budget), 1, None)
                    for partitions in args.partitions:
                        share = budget // partitions
                        record("approx_topn", truncation.approx_topn(budget), partitions, args.row_seeds[0])
                        for seed in args.row_seeds:
                            record("partition_topn", truncation.partition_topn(share), partitions, seed)
                            record("partition_approx_topn", truncation.partition_approx_topn(share), partitions, seed)


if __name__ == "__main__":
    main()
