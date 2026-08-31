# Static coset→worker placement — negative result

Host: ccqlin038 (2× Xeon Gold 6244, 16c/32t, 2 NUMA nodes, `powersave`
governor, shared box). Measured 2026-08-30 during the engine-optimization
campaign; extracted from that campaign's results note when the historical
notes were pruned.

## Context (diagnostics that motivated the attempt)

- gu2q IPC drops 2.32 → 1.21 and cycles/string rises 1636 → 3205 going
  8t → 32t (hyperthread sharing + memory pressure).
- A Trotter step at 32t drives **36.3 GB/s attributable DRAM traffic**
  (15.3 read + 21.3 write) against the measured ~45–49 GB/s box ceiling
  (see `2026-08-30-bandwidth-ceiling-ccqlin038.md`) — genuinely near the
  memory wall — with traffic *balanced across sockets* (S0 ≈ S1), i.e.
  pages are spread independent of which socket touches them.

## The experiment — reverted, do not re-attempt without new data

Static contiguous coset→worker assignment via `rayon::broadcast` (stable
worker index, one Mutex'd mega-slice per worker, opt-in env var) measured
**1.25–1.9× slower** than Rayon work-stealing in 7 of 8 probe cells at
16/32t: rotation 1.90×/0.93×, gu2q 1.59×/1.64×, trotter 1.41×/1.25×,
cnot 1.53×/1.06×.

**Why:** stragglers with no stealing cost far more than page locality
recovers. The code never landed in the tree (worktree discarded); the A/B
JSONs lived only in that session's scratchpad.

## If placement is ever revisited

A future variant would need work-stealing *within* a socket plus a stable
socket-level partition (or block-cyclic assignment), plus NUMA-aware first
touch of the bucket columns. Recorded as future work, not attempted. Near
the memory wall, the higher-leverage directions are traffic reduction
(e.g. in-place cos-scaling of the id stream rather than copying it through
the gather run) or real NUMA partitioning — not scheduling.
