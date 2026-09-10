# Re-calibrating two tuned constants on the padded build — one survives, one survives for a different reason, and the gather gets 3.4%

Four separately-recorded performance conclusions in this repo dissolved on re-measurement in the
week of 2026-09-10, all traceable to one artifact: branch alignment against 32-byte boundaries
(the JCC erratum, `2026-09-10-hot-path-code-size.md`; the `merge.rs` `#[inline]` folklore,
`2026-09-10-inline-set-repost.md`). `RADIX_MIN_REST_STREAMS = 8`
(`2026-09-01-sort-kernel.md`) and `GATHER_OUTPUT_MAJOR_MIN_R = 3` (`2026-09-01-bucket-cliff.md`)
were calibrated in the same regime, and `06777e3` / `4ee8033` then changed the gather and merge
cost structure underneath both. This is the re-calibration.

**Both constants keep their value.** Neither sweep produced a direction-consistent winner at any
other setting, and both incumbents' *numbers* reproduce on the padded build to within a point or
two. What did not survive is one of the two justifications: the radix gate's predictor — the
rest-stream count — is measurably not a predictor in the band it was left unmeasured in. `cnot`
and `gu2q` both have exactly **3** rest streams and want **opposite** kernels, by −5.5% and
+13.2%. No threshold on that quantity can separate them.

**One change shipped, and it came out of the setup rather than the sweep** (`2b949e6`). Making
the two arms of the gather gate comparable meant converting `gather_local_output_major` to the
branchless `push_if` filter that `4ee8033` gave only the input-major arm. That is worth
**−4.22% wall / −10.68% gather on `su4`, 11/11 pairs** — on the one layer where the *filter*
provably cannot fire, because a dense PTM has no zero amplitudes. The win is the three
`Vec::push` capacity checks per row, not the branch.

---

## 0. Protocol and provenance

Host ccqlin038 (reference host, 2× Xeon Gold 6244 Cascade Lake-SP, governor `powersave`,
microcode `0x5003901`), 2026-09-10, rustc 1.94.0, working tree `1d28d46` → `2b949e6`.

- Cell throughout: `--n 1000000 --qubits 128 --threads 1 --reps 20`, `taskset -c 6`, `RUST_LOG`
  unset. Exceptions: `su4 --reps 2` and `tfim_step --reps 6` (both cells are minutes per run
  otherwise), and `tfim_step --truncation coeff:1.220703125e-4` because it does not converge
  under `keep`.
- A constant is a compile-time `const`, so **all nine binaries were built up front** into one
  `CARGO_TARGET_DIR` and copied out before any timing; no build ever ran during a measurement.
  The build is reproducible byte-identical (two independent builds of the same source `cmp`
  equal), which is worth knowing on a fat-LTO / `codegen-units = 1` profile.
- `env -u RUSTFLAGS` on every build, so `.cargo/config.toml`'s
  `-Cllvm-args=-x86-branches-within-32B-boundaries` applied intact. **Verified per arm**: DSB
  share 95.9–97.6% on every binary in this note (`idq.dsb_uops / (idq.dsb_uops +
  idq.mite_uops)`, `perf stat` on the pinned process), against the pre-padding 45.8%.
- Alternated `abba`, 7 pairs, `python3 scripts/ab-report.py a.jsonl b.jsonl --all-phases`.
  Acceptance is direction consistency across every pair; median Δ% is the effect size.
- **LTO discriminator** on every campaign: `terms_in`, `terms_out`, `rows_gathered`,
  `rows_sorted`, `rows_id`, `cosets`, `runs`, `layers` bit-identical between sides. This holds
  literally everywhere in this note, including the radix cells — a threshold that changes which
  *sort kernel* runs does not change what is sorted, only how, so even `rows_sorted` is
  invariant. Nothing here is a work-count difference in disguise.
- Load 1.3–6.6 throughout (a concurrent Julia workload and a second agent session on the box).
  Not a quiet box; the `abba` pairing is the mitigation, and one campaign (§2.3) needed 11 pairs
  because of it.
- Raw artefacts for the two confirmation campaigns are in the gitignored
  `benchmarks/results/2026-09-10-ccqlin038/om-pushif-confirm{,2}-*`; the sweep campaigns were
  driven by a scratch alternator over the prebuilt binaries and are reproducible from §1's
  binary matrix.

## 1. The structural result: these constants have four settings, not fourteen

Before timing anything, an instrumented build printed each layer's realized plan, and the probe's
`runs` / `cosets` counters give the coset width `m = 2^r` exactly. Both are deterministic; no
noise, no protocol.

| probe layer | plan | rest streams | `m` | `r` |
|---|---|---:|---:|---:|
| `rotation_zz` | `Rotation` | — (not this path) | 2 | 1 |
| `trotter` | `Local` | **1** | 2 | 1 |
| `tfim_step`, `heavyhex_step` | `Local` | **1** | — | — |
| `cnot` | `Local` | **3** | 4 | 2 |
| `gu2q` (sqrt-SWAP) | `Local` | **3** | 4 | 2 |
| `su4` (Haar SU(4)) | `Local` | **15** | 16 | 4 |
| `depolarizing` | key-preserving | — (no gather) | — | — |

Two immediate consequences, and they are why this exercise cost nine binaries rather than
twenty-two:

- `RADIX_MIN_REST_STREAMS` has **four** distinct settings — `≤1`, `2..=3`, `4..=15`, `≥16` — not
  fourteen. The incumbent 8 is interior to a plateau six values wide; the whole "unmeasured
  `2..8` band" of `2026-09-01-sort-kernel.md` §6 is the *single* behavioural step at 3→4.
- `GATHER_OUTPUT_MAJOR_MIN_R` has **four**: `≤1` (every `Local` layer output-major), `2`
  (`cnot`/`gu2q`/`su4`), `3..=4` (`su4` only), `≥5` (none).
- `gu2q` is **3** rest streams, not the "~4" the task's framing assumed and not the "runner-up at
  3" being a near-miss — it is squarely in the unmeasured band, and so is `cnot`, which nobody
  had counted.

Binary matrix built for the sweeps (all `phase-timing`, all padded):

| name | `GATHER_OUTPUT_MAJOR_MIN_R` | `RADIX_MIN_REST_STREAMS` | output-major arm |
|---|---:|---:|---|
| `base_r3` | 3 | 8 | branchy (incumbent) |
| `pi_r1` / `pi_r2` / `pi_r3` / `pi_r5` | 1 / 2 / 3 / 5 | 8 | `push_if` |
| `br_r2` | 2 | 8 | branchy |
| `rx1` / `rx2` / `rx16` | 3 | 1 / 2 / 16 | branchy |

## 2. `GATHER_OUTPUT_MAJOR_MIN_R`: unchanged at 3

### 2.1 First, make the two arms comparable

`4ee8033` converted only `gather_local_input_major` to the branchless filter. Sweeping the gate
in that state compares an optimized arm against an unoptimized one and measures the asymmetry,
not the threshold. So the same change was mirrored into `gather_local_output_major` (the rest
stream and the sparse identity stream both) and measured in isolation first.

`base_r3` → `pi_r3`, i.e. nothing but the filter, at the incumbent threshold. `su4` is the only
layer that reaches output-major at `MIN_R = 3`; the other three are the untouched-path control.

| layer | path taken | wall Δ% | gather Δ% | sort Δ% | merge Δ% |
|---|---|---:|---:|---:|---:|
| **`su4`** | output-major | **−3.38** (7/7) | **−9.74** (7/7) | ns | ns |
| `cnot` | input-major | −0.33 (3/7 neg) | ns | ns | ns |
| `gu2q` | input-major | −0.09 (5/7 neg) | ns | ns | ns |
| `rotation_zz` | rotation gather | −0.31 (4/7 neg) | ns | ns | ns |

**The `su4` win is the interesting part, because the filter cannot be doing it.** Output-major is
only reachable at `r ≥ 3`, which in practice means a dense two-qubit PTM, whose `d.amp[s]` never
vanishes: no row is ever discarded and the `a == ZERO` branch is perfectly predictable. What
`push_if` removes on a dense PTM is the *other* half of `4ee8033` — the three `Vec::push`
capacity checks and length increments, on 394M rows:

| `su4`, `--n 300000 --reps 2` | `base_r3` | `pi_r3` |
|---|---:|---:|
| cycles | 18.94e9 | **18.19e9** (−4.0%) |
| instructions | 59.82e9 | **57.47e9** (−3.9%) |
| IPC | 3.16 | 3.16 |
| `branch-misses` | 28.8M | **14.7M** |
| DSB share | 97.6% | 97.5% |

Instructions and mispredicts both fall, IPC is flat: this is work removal, not a speculation
effect, notwithstanding the halved miss count (which is the capacity-check branch, taken once per
`Vec` growth and mispredicted at every growth). The three control layers moving 0.1–0.3% with
inconsistent sign is also the **layout control** this repo used to have to argue about: on the
padded build, editing this module no longer moves an untouched hot path.

Counters bit-identical, byte-identity tripwires (`layer_fingerprints_are_stable`, the thread- and
bucket-count tests) all pass — the emitted row sequence is unchanged, only the branch is gone.

### 2.2 The sweep, both arms in final form

Four settings, three of them measured against `pi_r3`:

| value | who changes gather order | cell | wall Δ% vs 3 | gather Δ% |
|---:|---|---|---:|---:|
| 1 | `trotter` → output-major | `trotter` | **+4.61** (7/7) | +11.39 (7/7) |
| 2 | `cnot`, `gu2q` → output-major | `cnot` | **+23.31** (7/7) | +43.12 (7/7) |
| 2 | " | `gu2q` | **+15.53** (7/7) | +48.12 (7/7) |
| **3** | — | — | incumbent | — |
| 5 | `su4` → input-major | `su4`, 1 thread | −0.31 (6/7 neg) | ns (4/7 neg) |
| 5 | " | `su4`, 16 threads | +6.86 (2/7 neg) | **+69.25** (7/7) |

(The `2` rows use `pi_r2`; `1` is `pi_r2` → `pi_r1` so only `trotter` moves. The 16-thread cell is
`taskset -c 0-15`, otherwise identical.)

Downward is decisively wrong, and by much more than the deterministic instrument predicted (§4).
Upward is the one that needed care: **at `r = 4` and one thread the two orders are
indistinguishable** — −0.31%, 6/7 negative, which is a lean toward input-major that does not clear
the bar, and exactly what `2026-09-01-bucket-cliff.md` §1.2 says it should be: each delta owns a
coordinate, both orders emit one contiguous ascending block, and input-major's one pass over the
input is then a hair cheaper than output-major's `2^r`. The entire
justification for keeping `su4` on output-major is therefore the *multi-threaded* gather, and it
holds: at 16 threads input-major's gather is +69% (7/7), the sixteen open write streams plus the
swapped coset overflowing L2 as recorded. Wall at 16 threads is sign-inconsistent, which is what
this box's ±10–26% does to a single-digit wall effect and is neither evidence for nor against.

**Verdict: 3 stands.** It is the only value that is not worse somewhere, and it is now bracketed
by measurement on both sides rather than by one 32-thread number on one side and a deterministic
comparison count on the other.

### 2.3 Confirming the shipped change independently

Re-run as a fresh `scripts/ab-compare.sh` campaign against the committed parent, twice:

| campaign | pairs | wall Δ% | gather Δ% |
|---|---:|---:|---:|
| `om-pushif-confirm` | 7 | −4.22 (**6/7**, one disturbed pair at +5.65) | −10.19 (6/7) |
| `om-pushif-confirm2` | 11 | **−4.22 (11/11)** | **−10.68 (11/11)** |

The first campaign's outlier is a single B run 10% above every other B run in the same campaign,
on a box with a neighbour; extending to 11 pairs resolves it, and the median is identical to two
decimal places across three independent campaigns (−3.38 / −4.22 / −4.22). Reported as
**−4.22%**.

## 3. `RADIX_MIN_REST_STREAMS`: unchanged at 8, but the reason changed

| value | effect | cell | wall Δ% vs 8 | sort Δ% |
|---:|---|---|---:|---:|
| 1 | 1-stream `Local` layers join | `trotter` | +0.63 (1/7 neg) | ns (2/7 neg) |
| 1 | " | `tfim_step` | +0.55 (1/7 neg) | **+6.14** (7/7) |
| 2 or 3 | `cnot`, `gu2q` join | **`cnot`** | **−5.48** (7/7) | **−22.66** (7/7) |
| 2 or 3 | " | **`gu2q`** | **+13.20** (7/7) | **+34.27** (7/7) |
| **4..=15** | incumbent | — | — | — |
| 16 | nobody; radix off | `su4` | **+17.20** (7/7) | **+31.42** (7/7) |

Three things, in order of how much they change the picture.

**The `su4` justification reproduces intact.** Turning the kernel off costs +17.20% wall and
+31.42% sort, i.e. the radix is worth −14.7% / −23.9% there, against the originally recorded
−15.2% layer / −25.4% sort at the same cell. That is inside the noise of the two protocols. This
one did *not* dissolve on the padded build — unlike the `#[inline]` set in the same module, and
unlike the three other conclusions of that week. The gather and merge rewrites of `06777e3` /
`4ee8033` did not move it either.

**The unmeasured band is measured, and it splits.** `cnot` and `gu2q` have **the same** rest-stream
count — 3 — and land on opposite sides by 19 percentage points of wall, with the sort phase moving
−22.7% and +34.3% respectively. Both are 7/7. This is not a threshold that was set slightly wrong;
it is a threshold on a quantity that carries no signal in that band. Adopting 2 or 3 would buy
`cnot` 5.5% and pay `gu2q` 13.2%.

The mechanism is visible in the incumbent's own per-row sort cost, and it is presortedness, not
stream count: at the same `m = 4` and the same 3 streams, `cnot` sorts at **11.4 ns/row** and
`gu2q` at **7.2 ns/row**. `gu2q` is sqrt-SWAP, whose delta masks are Pauli-structured
(`{XX, ZZ, YY}`) and, at `r = 2` with 4 coset coordinates, each own a coordinate — so a `gu2q` run
receives one contiguous ascending block per delta and driftsort merges three natural runs at close
to `log₂ 3 + 1 ≈ 2.6` comparisons, which is already cheap enough that the radix's two fixed passes
lose. `cnot`'s deltas do not distribute that way and its run arrives interleaved. The rest-stream
count is at best a proxy for that, and at 15 streams the proxy is safe only because *no* draw of
15 streams is presorted.

**The recorded sparse-PTM disaster does not reproduce as a layer effect.** `2026-09-01-sort-kernel.md`
records +130…+165% for the radix on a one-stream run; that was a microbench on a large synthetic
run. On the engine's actual 1-stream `Local` layers the sort phase loses only **+6.1%**
(`tfim_step`, 7/7) and wall does not move, because those layers' gather runs are 23 rows
(`tfim_step`) and 116 rows (`trotter`) — the radix falls back or is simply not amortizing anything
either way at that size. The low end of the gate is therefore much flatter than the note implies.
It does not change the verdict, because the `2..=3` step is already disqualifying, but it means the
gate's *conservatism* was buying less safety than believed.

**Verdict: 8 stands** — as the midpoint of the `4..=15` plateau, which is the only setting that
takes `su4` and leaves both 3-stream layers alone. Its doc comment is corrected: it is no longer
"conservative pending measurement of `2..8`", it is "the only value that works, because the
predictor is wrong below 4."

## 4. Correction to `2026-09-01-bucket-cliff.md` §1.2's instrument

§1.2 rejected retuning `GATHER_OUTPUT_MAJOR_MIN_R` downward on *deterministic* evidence:
comparisons per row for output-major vs input-major at `r = 1/2/0` are −14% / −6% / +7%,
"inconsistent and small". The conclusion is right and this note confirms it by timing. **The
instrument's sign is not right**, and that is worth recording where the comparison-count method is
used again:

| `r` | comparisons/row, output- vs input-major (§1.2) | measured wall, output- vs input-major (this note) |
|---:|---:|---:|
| 1 | −14% | **+4.61%** (`trotter`) |
| 2 | −6% | **+23.31% / +15.53%** (`cnot` / `gu2q`) |
| 4 | 0% | 0% at 1 thread; output-major −41% gather at 16 threads |

At `r ≤ 2` the comparison count says output-major should be *better* and the clock says it is
much worse, by up to 23%. The reason is that the choice does not trade sort against sort — it
trades sort against **gather**, and the gather is where the difference lands (+43…+48% at `r = 2`,
+11% at `r = 1`) while `sort_ns` barely moves at all (−0.3% / +0.4%, sign-inconsistent). The
comparison count is a good instrument for the *bucket-count* question §1 is actually about, where
both sides gather identically; it is the wrong instrument for the gather-order question, where the
phase it does not measure is the phase that moves. §1.2's own first bullet says as much for `r = 4`
("a pure gather-cost question") — it just does not extend the point downward.

## 5. Negatives, in full

- `GATHER_OUTPUT_MAJOR_MIN_R = 1`: +4.61% (`trotter`), 7/7. Rejected.
- `GATHER_OUTPUT_MAJOR_MIN_R = 2`: +23.31% / +15.53% (`cnot` / `gu2q`), 7/7 both. Rejected. Note
  this is the setting the branchless-arm asymmetry could plausibly have flipped — it does not come
  close. The branchless output-major arm *is* better than the branchy one on `gu2q` under
  output-major (18.33e9 vs 19.19e9 DSB uops, −4.5%), and output-major is still 15.5% worse than
  input-major there.
- `GATHER_OUTPUT_MAJOR_MIN_R = 5`: 1-thread null, 16-thread gather +69% (7/7). Rejected.
- `RADIX_MIN_REST_STREAMS = 1`: sort +6.1% on `tfim_step` (7/7), wall null. Rejected (and much
  milder than recorded).
- `RADIX_MIN_REST_STREAMS = 2` or `3`: `cnot` −5.48%, `gu2q` +13.20%, both 7/7. Rejected as
  sign-split across two layers with identical predictor values.
- `RADIX_MIN_REST_STREAMS = 16`: `su4` +17.20% (7/7). Rejected.
- No sweep value beat its incumbent on a direction-consistent basis with a supporting phase
  counter. With four candidate settings per constant this is not a multiple-comparisons escape —
  there was nothing to escape from.

## 6. What to do next

1. **A presortedness predictor at plan time.** `cnot` at −5.5% is a real, reproducible win that the
   current gate cannot express. The quantity that separates it from `gu2q` is how many maximal
   ascending runs a gather run arrives as, which is a deterministic function of the plan: the
   multiset of coset coordinates the rest deltas map to (`coords[e]` for `e ≥ rest_start`). If
   every delta owns a distinct coordinate, each run is a `k`-way merge of `k` clean blocks and the
   comparison sort is near its floor; if `d` deltas share a coordinate, their rows interleave row
   by row and it is not. `examples/delta_span_diagnostics.rs` already computes the run counts
   offline — the work is confirming the correspondence and then computing the same thing in
   `DeltaPlan::new`, which is once per layer. Gate on that instead of on the stream count, and
   `cnot` and `su4` both take the radix while `gu2q` and the 1-stream layers do not.
2. **The remaining `Vec::push` sites in the gather.** §2.1's win was capacity checks on a path
   where nothing is filtered. The rotation gather (`DeltaPlan::Rotation`, three `push` /
   `id_coeff.push` sites) has the same structure and covers `rotation_zz`, `trotter`, `tfim_step`
   and `heavyhex_step` — every kicked-Ising layer in the benchmark suite. Same change, same
   `cap`+1 reservation argument, and `rotation_zz`'s gather is 45% of its layer.
3. **Multi-thread the radix gate.** Everything in §3 is 1 thread. `2026-09-01-sort-kernel.md`
   §6 risk 2 already flags that the kernel's scratch grows 16 B/row and that its win shrinks as the
   layer approaches the write ceiling (−30.3% at `m` = 9884 down to −10.5% at `m` = 9.9e5 at 8
   threads). The `su4` +17.2% here should be re-taken at 16 threads with measured bandwidth
   alongside, per that note's own instruction.
4. **Nothing further on either constant.** Both are now bracketed by measurement on every side
   they have. Re-open only if a new channel shape lands outside {1, 3, 15} rest streams or
   {1, 2, 4} coset rank.

## 7. Reproduction

```bash
# The shipped change, against its parent.
env -u RUSTFLAGS scripts/ab-compare.sh om-pushif --a 00c07d7 --b 2b949e6 \
  --pairs 11 --order abba --features phase-timing \
  --probe '--n 1000000 --threads 1 --reps 2 --layers su4 --qubits 128'

# The sweeps: edit the const, build into its own CARGO_TARGET_DIR, copy the
# binary out, and alternate the prebuilt binaries abba. Nine binaries, §1's
# matrix. Every cell is
#   taskset -c 6 <bin> --n 1000000 --qubits 128 --threads 1 --reps 20 \
#     --layers <L> --json-out <side>.jsonl
# with --reps 2 for su4 and --reps 6 --truncation coeff:1.220703125e-4 for
# tfim_step, then
#   python3 scripts/ab-report.py a.jsonl b.jsonl --all-phases

# The structural table of §1 needs no timing: the rest-stream counts come from
# an eprintln in DeltaPlan::new, and m = runs / cosets from the probe sidecar.
```
