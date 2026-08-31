# Research

Negative-result records, hardware fact sheets, and forward design notes.
Nothing here is load-bearing for the build — design documentation lives in
`ARCHITECTURE.md`, operational procedure in `benchmarks/PROFILING.md`.

Naming convention: `YYYY-MM-DD-short-slug.md`. Keep the date so the timeline
of measurement is preserved.

**Check the negative-result notes before re-attempting an optimization idea**
— each records what was tried, the measured reason it lost, and what a future
attempt would need to do differently:

- `notes/2026-08-26-why-s5-concatenation-fails.md` — support-bit bucket
  concatenation cannot replace a sort (four-term counterexample).
- `notes/2026-08-30-static-coset-placement.md` — static coset→worker
  assignment loses 1.25–1.9× to work-stealing.
- `notes/2026-08-31-v0.6-results.md` — three rejected gather/merge variants
  (recompute-in-merge borrow, segment-copy merge, interleaved key layout).

Fact sheets and open items:

- `notes/2026-08-30-bandwidth-ceiling-ccqlin038.md` — the reference host's
  measured DRAM ceiling (the denominator for every roofline claim). Rerun
  only after hardware changes.
- `notes/2026-08-31-local-ptm-generalization.md` — design sketch for
  supporting custom channels on more than two qubits.
- `notes/2026-08-31-python-test-triage.md` — the first execution of the
  Python test suite, and the Y-phase convention conflict it exposed
  (resolved: parsers now use the core's Hermitian convention).
