"""Write a disBatch task file: one line per (alpha, seed block), each line one `run_b8.py` call.

    python make_tasks.py --alphas 0,0.05,0.1,0.15,0.2,0.25 --seeds 0:800 --block 10 \
        --threads 16 --cache 5e8 --out results/ppmc > tasks.txt
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alphas", default="0,0.05,0.1,0.15,0.2,0.25")
    ap.add_argument("--L", default="6", help="comma list of Floquet layer counts")
    ap.add_argument("--seeds", default="0:800")
    ap.add_argument("--block", type=int, default=10, help="seeds per task, amortizing import and circuit build")
    ap.add_argument("--threads", type=int, required=True, help="RAYON_NUM_THREADS per task")
    ap.add_argument("--python", default="python")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="passed through to run_b8.py after `--`")
    args, run_args = ap.parse_known_args(argv)
    lo, hi = (int(x) for x in args.seeds.split(":"))
    passthrough = [a for a in (args.extra or []) if a != "--"] + run_args
    script = HERE / "run_b8.py"
    for L in args.L.split(","):
        for alpha in args.alphas.split(","):
            for s in range(lo, hi, args.block):
                cmd = [
                    args.python, str(script), "--alpha", alpha, "--L", L,
                    "--seeds", f"{s}:{min(s + args.block, hi)}", "--out", str(args.out), *passthrough,
                ]
                print(f"RAYON_NUM_THREADS={args.threads} " + " ".join(shlex.quote(c) for c in cmd))
    return 0


if __name__ == "__main__":
    sys.exit(main())
