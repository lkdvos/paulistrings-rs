#!/usr/bin/env python3
"""Execute every ```python fence in docs/book/src against the built extension.

Blocks in one page share a namespace and run in document order, so a page can
build an observable in one fence and use it in the next. A `<!-- doctest: skip
-->` line immediately before a fence excludes it (MPI/NUMA snippets that need
hardware this runner does not have).

    source .venv/bin/activate
    python scripts/test-doc-snippets.py            # run every page
    python scripts/test-doc-snippets.py --list      # show run/skip without executing
    python scripts/test-doc-snippets.py FILE ...    # only these pages
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOK_SRC = REPO_ROOT / "docs" / "book" / "src"
SKIP_MARKER = "<!-- doctest: skip -->"

FENCE_RE = re.compile(
    r"^```python\s*$\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL
)


def extract_blocks(text: str) -> list[tuple[str, bool]]:
    """Return (code, skip) for every python fence in document order."""
    blocks = []
    for match in FENCE_RE.finditer(text):
        preceding = text[: match.start()]
        skip = preceding.rstrip().endswith(SKIP_MARKER)
        blocks.append((match.group(1), skip))
    return blocks


def run_file(path: Path) -> tuple[int, int, str | None]:
    """Returns (run_count, skip_count, error or None)."""
    blocks = extract_blocks(path.read_text())
    run_blocks = [code for code, skip in blocks if not skip]
    skip_count = len(blocks) - len(run_blocks)
    if not run_blocks:
        return 0, skip_count, None

    script = "\n\n".join(run_blocks)
    with tempfile.TemporaryDirectory() as tmp:
        # A page's snippet (e.g. io.save) may write a file; do it off-tree.
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp,
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        return len(run_blocks), skip_count, result.stderr
    return len(run_blocks), skip_count, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="specific markdown files (default: all under docs/book/src)")
    parser.add_argument("--list", action="store_true", help="list run/skip counts per file, don't execute")
    args = parser.parse_args()

    paths = [Path(f) for f in args.files] if args.files else sorted(BOOK_SRC.rglob("*.md"))
    paths = [p for p in paths if extract_blocks(p.read_text())]

    if not paths:
        print("no ```python fences found")
        return 0

    failures = []
    skip_only_pages = 0
    total_run = total_skip = 0
    for path in paths:
        rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
        blocks = extract_blocks(path.read_text())
        run_blocks = [c for c, skip in blocks if not skip]
        skip_count = len(blocks) - len(run_blocks)
        if args.list:
            print(f"{rel}: {len(run_blocks)} run, {skip_count} skip")
            continue

        run_count, skip_count, error = run_file(path)
        total_run += run_count
        total_skip += skip_count
        if error is not None:
            failures.append((rel, error))
            print(f"FAIL {rel} ({run_count} block(s))")
        elif run_count:
            print(f"ok   {rel} ({run_count} block(s), {skip_count} skipped)")
        else:
            skip_only_pages += 1
            print(f"skip {rel} ({skip_count} block(s), 0 run)")

    if args.list:
        return 0

    passed_pages = len(paths) - len(failures) - skip_only_pages
    print(
        f"\n{passed_pages}/{len(paths)} pages passed, {skip_only_pages} fully skipped, "
        f"{len(failures)} failed, {total_run} block(s) run, {total_skip} skipped"
    )
    for rel, error in failures:
        print(f"\n--- {rel} ---\n{error}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
