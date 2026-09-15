#!/usr/bin/env bash
# Draft mode (default): report status, always exit 0 -- a pending figure is
# expected right now and must never fail a scaffold build.
# --release: exit 1 if any figure-manifest.json asset is not "real-asset", or
# if any [slot: ...] marker remains in notes.md. Use before an actual delivery
# dry run, never as part of routine scaffold iteration.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

release=false
[[ "${1:-}" == "--release" ]] && release=true

python3 - "$release" <<'PY'
import json, re, sys
release = sys.argv[1].lower() == "true"

manifest = json.load(open("figures/figure-manifest.json"))
pending = [a for a in manifest["assets"] if a["status"] != "real-asset"]

print(f"figures: {len(manifest['assets']) - len(pending)}/{len(manifest['assets'])} real")
for a in pending:
    print(f"  pending: {a['path']} (slide {a['used_by_slide']}, status={a['status']})")

notes = open("notes.md").read()
slots = re.findall(r"\[slot:[^\]]*\]", notes)
# notes.md itself only ever names slot categories in prose; the actual count
# that matters for release is in the .typ sources, so check those too.
import glob
slot_hits = []
for path in glob.glob("sections/*.typ"):
    text = open(path).read()
    for m in re.finditer(r'"\[slot:[^"]*\]"', text):
        slot_hits.append((path, m.group(0)))

print(f"open [slot: ...] markers in sections/*.typ: {len(slot_hits)}")

if release and (pending or slot_hits):
    print("RELEASE CHECK FAILED: pending figures or unresolved slots remain.")
    print("This is expected while benchmark data is intentionally pending --")
    print("it is not a defect in the scaffold itself.")
    sys.exit(1)

print("draft check OK" if not release else "RELEASE CHECK PASSED")
PY
