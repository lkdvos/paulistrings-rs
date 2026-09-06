#!/usr/bin/env bash
# Paired A/B: default build vs RUSTFLAGS="-C target-cpu=native", alternated
# abba, N pairs, single thread. Writes targetcpu_{default,native}.jsonl and a
# paired-delta table targetcpu_ab.md (acceptance: direction consistency across
# every pair, median delta as effect size — benchmarks/PROFILING.md).
set -euo pipefail
cd "$(dirname "$0")/../../.."
unset RUST_LOG
A=presentation/bench/target/release/presentation-bench
N=presentation/bench/target-native/release/presentation-bench
D=presentation/data; PAIRS=${PAIRS:-5}; EPS=${EPS:-2.44140625e-4}; STEPS=${STEPS:-10}
VARIANTS=${VARIANTS:-"bucketed naive"}
[[ -x $A && -x $N ]] || { echo "build both binaries first (collect_all.sh build)"; exit 1; }
for f in default native; do fa=$D/targetcpu_$f.jsonl; [[ -s $fa ]] || echo "# provenance: $(date -Is) host=$(hostname -s) commit=$(git rev-parse --short HEAD) pairs=$PAIRS order=abba" > "$fa"; done
for v in $VARIANTS; do
  for ((p=0; p<PAIRS; p++)); do
    if (( p % 2 == 0 )); then order="A N N A"; else order="N A A N"; fi
    for side in $order; do
      if [[ $side == A ]]; then bin=$A; tag=default; else bin=$N; tag=native; fi
      $bin $v --threads 1 --reps 1 --steps $STEPS --eps $EPS --tag $tag --json-out $D/targetcpu_$tag.jsonl | grep '^cell' | sed "s/^/$tag pair=$p /"
    done
  done
done
python3 - "$D" <<'PY'
import json, sys, statistics as st
D=sys.argv[1]
def load(p):
    rows=[json.loads(l) for l in open(p) if not l.startswith('#')]
    out={}
    for r in rows: out.setdefault(r['layer'],[]).append(r['wall_ns'])
    return out
a=load(f"{D}/targetcpu_default.jsonl"); b=load(f"{D}/targetcpu_native.jsonl")
lines=["# target-cpu=native vs default, paired per run (1 thread)","","| variant | pairs | delta % per pair | same sign | median delta % | verdict |","|---|---|---|---|---|---|"]
for v in a:
    n=min(len(a[v]),len(b[v])); d=[100*(b[v][i]-a[v][i])/a[v][i] for i in range(n)]
    same=sum(1 for x in d if x<0)==n or sum(1 for x in d if x>0)==n
    verdict="consistent" if same else "no consistent change"
    lines.append(f"| {v} | {n} | {', '.join(f'{x:+.1f}' for x in d)} | {'yes' if same else 'no'} | {st.median(d):+.1f} | {verdict} |")
open(f"{D}/targetcpu_ab.md","w").write("\n".join(lines)+"\n"); print("\n".join(lines))
PY
