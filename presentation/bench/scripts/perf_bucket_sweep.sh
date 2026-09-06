#!/usr/bin/env bash
# Hardware counters for the bucket-size sweep (1 thread): L2 and LLC demand
# misses per instruction, IPC. Merges `perf stat -x,` output with the cell's
# JSON into bucket_sweep_perf.jsonl. Cascade Lake event names; falls back to
# mem_load_retired.* if l2_rqsts.* is unavailable.
set -euo pipefail
cd "$(dirname "$0")/../../.."
unset RUST_LOG
B=presentation/bench/target/release/presentation-bench
D=presentation/data; f=$D/bucket_sweep_perf.jsonl; EPS=${EPS:-2.44140625e-4}; STEPS=${STEPS:-10}; REPS=${REPS:-3}
THREADS=${THREADS:-1}
if perf list 2>/dev/null | grep -q 'l2_rqsts.references'; then
  EV="duration_time,cycles,instructions,l2_rqsts.references,l2_rqsts.miss,LLC-loads,LLC-load-misses"
else
  EV="duration_time,cycles,instructions,mem_load_retired.l2_hit,mem_load_retired.l2_miss,LLC-loads,LLC-load-misses"
fi
[[ -s $f ]] || echo "# provenance: $(date -Is) host=$(hostname -s) commit=$(git rev-parse --short HEAD) events=$EV reps=$REPS threads=$THREADS" > "$f"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for cell in "256 128" "1024 128" "4096 128" "16384 64" "65536 16" "262144 16"; do set -- $cell
  perf stat -x, -o "$tmp/c.csv" -e "$EV" -- $B bucketed --threads $THREADS --reps $REPS --steps $STEPS --eps $EPS --target-bucket-len $1 --min-buckets $2 > "$tmp/out.txt"
  grep '^cell' "$tmp/out.txt" | sed "s/^/target=$1 min=$2 /"
  python3 - "$tmp/c.csv" "$tmp/out.txt" "$f" <<'PY'
import sys, json
csv, out, dst = sys.argv[1:4]
ctr={}
for line in open(csv):
    if line.startswith('#') or not line.strip(): continue
    p=line.split(',')
    try: ctr[p[2]]=float(p[0])
    except ValueError: pass
rows=[json.loads(l) for l in open(out) if l.startswith('{')]
# The counters cover warm-up + all reps; attribute per-instruction rates (scale-free) and totals.
l2r=ctr.get('l2_rqsts.references'); l2m=ctr.get('l2_rqsts.miss')
if l2r is None and 'mem_load_retired.l2_hit' in ctr:
    l2m=ctr['mem_load_retired.l2_miss']; l2r=ctr['mem_load_retired.l2_hit']+l2m
llc=ctr.get('LLC-loads'); llcm=ctr.get('LLC-load-misses'); ins=ctr.get('instructions'); cyc=ctr.get('cycles')
agg={"cycles":cyc,"instructions":ins,"l2_refs":l2r,"l2_misses":l2m,"llc_loads":llc,"llc_load_misses":llcm,
     "l2_miss_rate": (l2m/l2r if l2r else None), "llc_miss_rate": (llcm/llc if llc else None),
     "ipc": (ins/cyc if cyc else None), "l2_misses_per_kinst": (1000*l2m/ins if ins else None),
     "llc_misses_per_kinst": (1000*llcm/ins if ins else None), "perf_reps_including_warmup": len(rows)+1}
with open(dst,'a') as fh:
    for r in rows:
        r.pop('terms_out',None); r.update(agg); fh.write(json.dumps(r)+"\n")
PY
done
echo done
