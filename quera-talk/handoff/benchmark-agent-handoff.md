# Agent handoff: generate the data for the QuEra technical talk

## Mission

Work in Lukas Devos's `paulistrings-rs` repository and on his Slurm cluster to produce reproducible, slide-ready benchmark data for **Pauli propagation at scale: “Wie niet sterk is, moet slim zijn”**.
The main deliverable is the data supporting a recurring figure that reveals implementation improvements and ultimately extends the calculation beyond single-node memory capacity.
Implement the measurement plumbing, prepare and execute the required cluster jobs, analyze the results, and deliver a complete evidence package.
This is a measurement and presentation-support task, not an open-ended algorithm-optimization project.

The companion `quera-talk-outline.md` contains the current slide plan.
The requirements here remain sufficient to execute the data campaign if that document is unavailable.
Source paths below were checked in the attached `paulistrings-rs-main.zip` snapshot; inspect the live checkout before relying on them.
The ZIP does not establish the current Git revision.
No benchmark run or cluster submission was performed while writing this handoff.

## Orchestration layer: minimize total agent cost

This section defines how the receiving agent should turn the handoff into delegated work.
Use a main orchestrator and bounded subagents, choosing cheaper models and lower reasoning effort for routine tasks.
Optimize total cost to an accepted result, including retries and integration, rather than minimizing the cost of each individual call.
Preserve every scientific and measurement requirement below.
Do not spawn agents merely because slots are available.

### Main-agent responsibilities

The main agent owns the shared benchmark contract, dependency graph, integration, resource budget, cluster submission, and final evidence assessment.
It reads this complete handoff once, inspects applicable repository instructions, and produces a small shared contract before implementation begins.
It should use moderate reasoning by default and reserve high effort for unresolved numerical semantics, MPI timing, or conflicting evidence.
It must not independently repeat a completed subagent inspection unless a concrete inconsistency needs resolution.

Only the main agent submits, resubmits, or cancels cluster jobs and changes the campaign manifest's execution state.
Subagents can prepare and validate job scripts but cannot allocate resources independently.
A job-submission ledger prevents duplicate submissions after retries or context loss.
This is an internal coordination rule, not a new user-approval requirement.

Use these compact working files in the campaign workspace:

| File | Purpose |
| --- | --- |
| `contract.md` | Frozen scientific task, timing semantics, schema version, variant definitions, and immutable constraints |
| `tasks.json` | Task IDs, dependencies, owners, file boundaries, status, output pointers, and acceptance result |
| `decisions.md` | Short decisions with rationale and source pointers; no chronological transcript |
| `job-ledger.jsonl` | Configuration/revision identity, submitted job ID, resource request, state, and output path |
| `agent-results/<task-id>.md` | Compact completion report for each delegated task |

Treat `contract.md` as a concise index and a statement of agreed choices, not a copy of this handoff.
Link to longer specifications when needed.
Do not let it silently weaken the requirements in this document.

### Model and effort routing

Use the cheapest available model that can reliably satisfy the task's acceptance criteria.
Select from the models actually exposed in the receiving environment; do not assume a named model exists or has a particular current price.
Record the selected model and effort per task when the orchestration tools support them.

| Work type | Default tier | Effort | Escalation trigger |
| --- | --- | --- | --- |
| File discovery, provenance extraction, result inventory | Cheap/fast | Low | Conflicting sources or ambiguous variant identity |
| Narrow scripts, schemas, normalization, plot builds | Cheap/fast coding | Low–medium | Failed meaningful acceptance check after one targeted repair |
| Julia wrapper and existing benchmark integration | Standard coding | Medium | Numerical/truncation mismatch or API behavior uncertainty |
| Rust gate tracing and MPI timing implementation | Strong coding | Medium | Ownership, concurrency, collective ordering, or default-path regression |
| Independent timing/semantics review | Strong reasoning/coding | High, narrowly scoped | Resolve a specific dispute; avoid a whole-repository review |
| Slurm status collection | Script/tool, no subagent preferred | None | Concrete failed-job diagnosis |
| Final narrative/evidence packaging | Cheap/fast | Low | Unsupported claim or incompatible comparison discovered |

Do not assign delicate MPI or ownership changes to a weak model just to reduce initial token use.
Do not use the strongest model to copy metadata, watch queues, or regenerate figures.
If effort controls are unavailable, express the same constraints through a narrow task and explicit stop conditions.

### Task graph and bounded assignments

T01–T03 are independent read-only reconnaissance and can run together.
After those reports, the main agent freezes `contract.md` and the schema before dispatching implementation.
Do not launch every task simultaneously.

| ID | Assignment and required output | Dependencies | Default tier/effort | Write boundary |
| --- | --- | --- | --- | --- |
| T01 | Inventory real baseline revisions/configurations and existing datasets; draft `variants.json` with provenance/confounders | None | Cheap, low | Own report and variant draft only |
| T02 | Inspect benchmark builders/reference material; recommend one exact shared task and document source/convention alignment | None | Standard, medium | Own report and task draft only |
| T03 | Inspect cluster/toolchain availability; select a uniform CPU class and physical-core-only mapping; report limits, reusable launch paths, missing inputs | None | Cheap, low | Own report only; no submission |
| T04 | Implement opt-in Rust gate tracing and focused tests, covering persistent unpartitioned/partitioned loops and necessary binding exposure | Main contract | Strong coding, medium | Explicitly assigned core/binding files and tests |
| T05 | Extend Julia/external wrappers to the agreed task and timing schema; preserve parity and warm-timing rules | Main contract | Standard coding, medium | External wrapper/harness files, not T04 files |
| T06 | Implement schema validation, trace normalization, and plot-table generation with synthetic fixtures | Main schema | Cheap coding, low–medium | Analysis/schema tools and their tests |
| T07 | Prepare resumable campaign driver, constrained Slurm scripts, hardware/physical-core preflight, and metadata capture | Contract, T03; final integration after T04–T05 | Standard coding, medium | Campaign/job scripts, no core files |
| T08 | Review timing semantics, counts, policy boundaries, MPI aggregation, and measurement perturbation | T04–T07 | Strong reviewer, high | Read-only findings tied to changed code |
| T09 | Create recurring plot stages and scaling/hash/accuracy figures from normalized tables | T06; final outputs after real data | Cheap coding, low–medium | Plot scripts and generated figures |
| T10 | Validate collected evidence and assemble E0–E9 mapping, headline table, README, reproduction commands | Accepted data, T09 | Cheap, low; escalate disputed claims | Final documentation, not raw data |

The main agent integrates T04–T07, runs the required acceptance gates, and executes the staged campaign.
T09 can develop against a tiny explicitly synthetic fixture while real jobs run, but synthetic values must never enter delivered result tables or final figures.
A bounded failed-job diagnosis can be delegated using the packet below; do not create a permanent queue-watching agent.

T04 is intentionally one owned task because loop changes, trace records, and bindings can otherwise collide.
Split it only if the main agent first defines an interface and non-overlapping file ownership, and the subtask is large enough to justify the coordination cost.
No recursive delegation by default.
A subagent should report a need to split or escalate rather than autonomously creating an agent tree.

### Copyable dispatch packet

Send only this packet, the applicable instructions, and the relevant excerpts or file pointers.
Do not fork the entire conversation or resend the full handoff to every worker when the tools allow a fresh context.
If context inheritance cannot be controlled, still specify the narrow scope clearly.

```text
Task ID: <Txx>
Objective: <one concrete result>
Model/effort: <available model and setting>
Contract: <path, version/hash, relevant clauses>
Inputs: <specific files, predecessor outputs, immutable revision>
Read first: <applicable local instructions and only relevant design sections>
Allowed writes: <exact files/directories or assigned worktree>
Do not change: <interfaces/semantics/files outside ownership>
Deliverables: <paths and formats>
Acceptance: <meaningful checks and observable completion conditions>
Budget: <bounded inspection/implementation scope; no broad exploration>
Escalate if: <specific ambiguity, failed check, or permission/access boundary>
Return: <report format below>
```

For each implementation task, include the relevant schema excerpt and exact call/interface contract instead of expecting the worker to infer them from neighboring code.
For read-only tasks, explicitly forbid mutations.
For cluster-script tasks, explicitly forbid submission and provide the resource bounds chosen by the main agent.
Subagents remain subject to all applicable instructions and permissions even when only a subset of the project context is forwarded.

### Shared-workspace and integration rules

Assign disjoint write sets before starting parallel workers.
Prefer a separate worktree per implementation task when source edits overlap or the tools support clean patch integration.
In a shared worktree, prohibit edits outside the assigned set and have the main agent perform cross-cutting interface changes.
A worker encountering an unexpected change stops editing the affected file and reports it rather than overwriting it.

Integrate accepted tasks in dependency order.
Reuse task-level test evidence and run the repository-required integration gates once for the assembled revision, unless a failure or new change requires another run.
Keep raw measurements immutable and keyed by the exact integrated build.
Code changed after job submission is a new variant; never silently combine its measurements with the previous revision.

### Compact return format

Target at most 300 words, plus artifact pointers and short test summaries.
Keep long logs in files and return only the relevant failure excerpt.

```text
Task: Txx
Status: done | blocked | needs_review
Outputs: <paths, revision/patch if applicable>
Changes/findings: <up to five concise points>
Acceptance evidence: <commands/checks and their outcomes>
Assumptions/limitations: <only material items>
Next dependency: <what the main agent can now start>
```

Do not return the full source, a long chain of reasoning, or a repeated task specification.
Read predecessor reports and precise code regions rather than rediscovering the whole repository.
Preserve useful findings in files so retries and compacted contexts do not repeat expensive searches.

### Escalation and stopping

Permit one focused repair for a routine failed check before escalating with the concrete failure and attempted fix.
Escalate immediately for uncertain numerical semantics, collective ordering, ownership safety, inconsistent reference data, or a permission/access boundary.
Escalation can mean the main agent resolves the issue, gives a stronger model the same bounded task, or requests one targeted review.
Do not run multiple models on the same task by default or use broad “second opinions” without a disputed claim.

Batch independent cheap reads and use scripts for repetitive parsing, scheduling status, and figure generation.
Poll Slurm at practical intervals or through existing job-completion mechanisms; avoid tight polling loops and repeated agent invocations while jobs are pending.
Once an output passes its stated acceptance checks, close the task and move on.
The final high-effort attention should go to whether the evidence supports the comparisons, not to polishing already adequate prose.

## Authorization and operating scope

The user has explicitly said that cluster jobs can be prepared and sent off for this work.
Prepare and submit the necessary bounded Slurm campaign under the user's existing access and allocation, monitor it, and collect its outputs.
The inspected `CLAUDE.md` and `scripts/slurm/README.md` normally reserve submission for the user; this task-specific user authorization overrides that repository workflow convention for this campaign.
It does not override platform permissions, scheduler policy, account limits, or authorization boundaries.
Do not change permissions or request a new allocation to bypass a limit.

Read applicable `AGENTS.md`/`CLAUDE.md`, `ARCHITECTURE.md`, `benchmarks/PROFILING.md`, `research/FINDINGS.md`, and `research/HARDWARE.md` before modifying measurement code.
Respect the repository's tests, release-build rules, prose conventions, and performance-measurement practices.
Work in an isolated branch/worktree, preserve the user's work, and pin the exact source revision or patch used by every job.
Do not merge, publish, or push changes unless separately authorized.

Discover the accessible cluster configuration and existing account defaults without requesting information already available in the environment.
Record the proposed matrix, estimated node-hours, memory, wall-time caps, and maximum concurrent allocations before launching the main campaign.
Use scheduler allocations for heavy work, not login nodes.
Submit in bounded stages, reuse completed results, and adapt subsequent job sizes from actual measurements.
If access, account choice, or a concrete resource limit blocks progress, finish the scripts and manifest first, then identify the exact missing input.
Do not invent hardware details or pretend submitted jobs completed.

## Mandatory hardware contract: one architecture, physical cores only

All campaign runs must execute on the same selected CPU architecture and node class on the cluster.
Use one concrete processor model/generation with matching socket/NUMA topology and memory configuration for the benchmark allocations; sharing the x86-64 instruction set is not sufficient.
Choose the class once from available resources, freeze it in `contract.md` and `campaign.json`, and apply the corresponding scheduler constraint to every job.
This applies to external-library baselines, historical variants, pilots, traced/untraced comparisons, single-thread runs, thread/rank scaling, hash comparisons, capacity runs, and supporting bandwidth/cycle measurements.
Every node of a multi-node job must satisfy the same contract.
Do not switch architectures because another partition is available or its queue is shorter.
If the selected class cannot supply a required allocation, report the blocked cells and retain the constraint.
A user-approved architecture change requires a separate campaign and fresh baseline measurements, not combining old and new points into one curve.

Use physical cores only: at most one active logical CPU per physical core across all benchmark workers and MPI ranks on a node.
Hyperthreading/SMT siblings must not provide extra workers or inflate the advertised core count.
The machine may have SMT enabled, but benchmark processes must be restricted to one chosen hardware thread per physical core.
Do not change BIOS or system-wide SMT settings.

Use the cluster-supported scheduler controls for no-SMT allocation, then verify the resulting CPU masks instead of trusting the request alone.
Discover the logical-CPU to `(socket, physical-core, NUMA-node)` mapping from the allocated compute node.
Create physical-core-only masks with no repeated `(socket, physical-core)` pair, respecting the job's allowed CPU set.
Confirm that rank masks are disjoint at the physical-core level and that actual worker pools cannot expand onto sibling hardware threads.
Set library/helper-thread limits where applicable so auxiliary runtimes do not introduce unintended nested parallelism.
A full-node point means the selected node class's physical core count, not its logical CPU count.

For in-process NUMA partitioning, provide explicit per-domain physical-core CPU lists through the engine's placement mechanism while preserving access to all allocated domains.
Do not use a single-domain outer affinity/memory restriction that defeats the engine's ownership placement.
For MPI, use physical-core-only rank affinity and verify the pool size derived from that affinity.
Binding a rank to a NUMA domain alone may still expose both SMT siblings; inspect the effective CPU set and correct it before measurement.
Cap workers per rank to the number of distinct physical cores assigned to that rank.

Before every timed run, fail preflight if the CPU model/class differs from the frozen contract, rank masks overlap physical cores, sibling threads are simultaneously available to worker pools, or actual concurrency exceeds assigned physical cores.
Record CPU model, stepping/microcode where available, topology, selected logical CPU IDs and their physical-core mapping, effective process/rank affinity, worker counts, and the preflight result.
Capture these from the compute allocation, not the login host.
Mark mismatching runs `invalid_hardware` and exclude them from all performance summaries; retain their logs and requeue corrected configurations on the selected class.
Postprocessing and report generation may run elsewhere because they do not contribute performance measurements.

## Scientific and presentation decisions already made

- The running problem is the 127-qubit heavy-hex kicked Ising benchmark.
- Select one fully specified published configuration from the PauliPropagation paper or original IBM experiment, using the repository's matching circuit/task machinery where possible.
- Hold depth, angles, initial state, observable, gate sequence, and numerical precision fixed in the main tolerance sweep.
- Truncation acts on each gate's fully accumulated output, not on individual unsummed contributions or only at circuit-period boundaries.
- The main sweep decreases a coefficient cutoff to grow the operator; histogram count control is a separate policy axis where needed, not a covert replacement for the cutoff on the same curve.
- A human-facing update is one gate application, Clifford or rotation.
- Gate timing includes preparation and rebucketing, overhead, communication, gather/sort/merge, truncation, and any synchronization required by that execution path.
- Collect efficiency data along complete simulations with persistent runtime/scratch, not a standalone saved-input microbenchmark campaign.
- Accuracy/convergence is a consistency check against existing libraries or published results; no new error theory or comprehensive accuracy study is required.
- Single-thread, multithread, multiprocess/distributed, memory-capacity, and communication-aware hash evidence are all essential.
- Record outcomes honestly, including regressions, indistinguishable changes, failures, and limitations.
- No speedup, memory-limit crossing, or convergence improvement may be assumed merely because the talk intends to discuss it.

## Required figure and evidence matrix

### The recurring figure

| Panel | x-axis | y-axis | Required annotations |
| --- | --- | --- | --- |
| Efficiency | Strings entering a gate | Input-string updates per second | Gate class, configuration, thread/rank count, representative working-set sizes |
| Calculation cost | Coefficient tolerance decreasing left to right | Complete propagation wall time in seconds | Resident peak string count, measured memory footprint, completion/OOM status |

Use log axes where appropriate.
Keep colors, baseline/library references, and axis limits consistent across staged versions, until an explicit final extension of the tolerance range.
The left panel is higher-is-better; the right panel is lower-is-better.
Memory bandwidth/access latency explains processing cost, while capacity determines which complete calculations fit.
Do not conflate these limits.

### Datasets supporting the slides

| ID | Evidence | Slide use | Required comparison |
| --- | --- | --- | --- |
| E0 | Original baseline and external libraries | 10 and every recurrence | Fixed circuit and aligned cutoff semantics |
| E1 | Actual kernel improvement | 11 | Genuine baseline versus improved revision/configuration, one thread |
| E2 | Actual attempted threading approach | 12 | Its measured benefit or limitation, with consolidation costs included |
| E3 | Memory diagnosis | 13–14 | Traffic model, phases, relevant bandwidth/cache evidence, single-thread cycle annotation |
| E4 | Bucketed single-thread gain | 25 | Comparable internal baseline plus external references |
| E5 | Bucketed multithread gain | 26–27 | Same implementation/workload over a thread-count ladder |
| E6 | Multiprocess/distributed behavior | 28 | Overlapping workloads and increasing rank/node count |
| E7 | Beyond-single-node capacity | 28 | A completed lower-tolerance distributed run that exceeds the documented single-node feasible regime |
| E8 | Communication-aware hash | 28 or 28b | Random versus cut rows at fixed distributed workload and resources |
| E9 | Observable consistency/convergence | 24 | Matched existing-library or published reference |

E0–E9 are required evidence, not an invitation to invent a result.
If a historical implementation is unavailable or an effect cannot be established, document the gap and the valid comparison available instead.
Do not label a modern `auto`/`sorted` switch as “before bucketing”: both may use bucketing.
Do not implement an intentionally weak baseline or manufacture an unsuccessful threading experiment just to match the narrative.

### Historical and external baselines

Inspect Git history and `research/FINDINGS.md` for the actual kernel and threading changes discussed by the user.
Produce `variants.json` mapping each figure label to a revision, patch/configuration, description, and known confounders.
Use isolated immutable worktrees for historical code.
If old APIs differ, adapt benchmark wrappers while preserving the algorithms and record the adaptation.
Separate a historical end-to-end improvement, which may bundle changes, from a controlled ablation.

Use the existing PauliPropagation.jl comparison as the first external reference.
Inspect the existing PauliStrings.jl comparison/contribution material and include it when the same propagation task can be run comparably.
Do not broaden into a survey of unrelated packages or replace propagation benchmarks with container-operation timings.
If an external package lacks equivalent per-gate instrumentation, retain its valid total-time reference and explicitly mark the missing efficiency series.
Record any resulting evidence gap rather than substituting final term count divided by runtime.

## Implementation facts to preserve

The engine uses GF(2)-linear bucket hashing.
For a prepared gate with label-delta set D, `Gf2Span` constructs V = span(h(D)); cosets of V are closed independent groups of buckets.
Rotation pairs are a special case.
A storage bucket and a NUMA/MPI ownership partition are different levels of decomposition.

`fill_coset` swaps source columns into reusable scratch before writing the owned bucket slots.
The identity-delta stream retains sorted order; changed-label contributions, including remote rows, enter a rest stream that is sorted before merge/reduction.
A layer-level choice can select adaptive comparison or radix sorting.
In-place operation does not mean absence of scratch or expansion storage.

`CoefficientThreshold` filters coefficients after summation inside the merge.
`ApproxTopN` uses 2048 exponent bins of squared coefficient magnitude and retains at most the requested count, possibly undershooting.
The partitioned finalization all-reduces the histogram and length before choosing the common edge.
Exact `TopN` is not supported by the partitioned policy interface in the inspected snapshot.
Do not change these semantics for benchmark convenience.

`PartitionRows::cut` uses z-only block-parity rows.
X generators have zero ownership-partition delta; a ZZ generator's delta is the XOR of its endpoint block labels.
Thus X rotations and within-block ZZ rotations avoid cross-partition row movement, while cut edges may communicate.
Compare this existing construction to seeded random rows; designing a new hash family is outside scope.
Record explicit cut blocks, seeds, bucket hash, and any reseeding performed to keep ownership rows independent of bucket rows.
Measure load distribution alongside communication savings.

## Repository entry points

| Purpose | Starting files |
| --- | --- |
| Algebra and local kernel | `crates/paulistrings/src/engine/coset.rs`, `bucketed.rs`, `merge.rs` |
| Gate loops | `crates/paulistrings/src/engine/mod.rs`, `engine/partitioned/driver.rs` |
| Policy behavior | `crates/paulistrings/src/truncation/builtin.rs`, `engine/partitioned/truncation.rs` |
| Ownership hash | `crates/paulistrings/src/bucket/hash.rs` |
| Existing traces | `engine/bucketed.rs` (`TermTrace`), `engine/partitioned/trace.rs`, `engine/stats.rs` |
| Python stats/API | `crates/paulistrings-py/src/sum.rs`, `mpi.rs` |
| Circuit task contract | `benchmarks/python/bench_jl_performance.py`, `python/paulistrings/interop.py`, `examples/common/` |
| Deep kicked Ising | `benchmarks/python/bench_c_deep_trotter.py` |
| Julia runner | `benchmarks/julia/runner.jl`, `Project.toml`, `Manifest.toml` |
| Phase/hash probe | `crates/paulistrings/examples/phase_breakdown.rs` |
| Launch scaffolding | `scripts/slurm/mpi-ranks.sbatch`, `ab-campaign.sbatch` |
| Measurement discipline | `benchmarks/PROFILING.md`, `scripts/ab-compare.sh`, `scripts/jcc-rustflags.sh`, `scripts/bandwidth.sh` |

The inspected Julia manifest pins PauliPropagation.jl 0.8.2; verify the live environment rather than assuming this remains true.
The comparison driver already shares one task JSON between engines, checks gate-count parity, warms both runtimes, and interleaves timing pairs.
Preserve those controls.
The existing MPI Slurm script runs differential tests and a phase probe; it is launch scaffolding, not the required full-simulation tolerance-sweep driver.

## Phase A: freeze the workload and campaign

1. Inspect the live tree, local instructions, existing results, Git history, available toolchains, and Slurm configuration.
2. Identify real baseline variants and map them to E0–E8.
3. Select and cite the exact published benchmark configuration.
4. Generate one canonical task file shared across wrappers, with explicit gate order and angle/sign conventions.
5. Create a campaign manifest and bounded launch plan before expensive runs.

The existing deep benchmark provides useful candidates: n = 127, theta_zz = -pi/2, depths through 20 steps, and interior kick angles including 7pi/32 and 5pi/16.
These are starting points, not authorization to claim an exact paper match without checking the source.
Record the observable's indexing convention, propagation direction, and expectation-value state.
Use one primary angle/depth for the recurring curves; other existing references may support the consistency check.
A shallow reference calculation must be labeled with its actual depth rather than presented as validation of an unavailable exact deep result.

Start with an existing logarithmic cutoff grid, then extend it toward the single-node limit and the distributed regime using measured memory and time.
Pilot runs calibrate job sizes and runtime, not the selection of a flattering scientific workload.
Freeze the primary workload and analysis rules before collecting the main comparisons.
Once fixed, retain all attempted cutoff points in the run manifest.

## Phase B: add complete-gate measurement

### What is missing

In the inspected source, public stats expose gate counts and exchange volumes but no structured per-gate wall-time vector.
`PhaseStats` aggregates phases; it is not the required per-gate elapsed series.
Debug output has rounded milliseconds and logging overhead.
Add a minimal opt-in structured gate trace in the actual full-propagation loops.
If the live code already supplies one, reuse it after checking the boundaries.

### Timer contract

For gate j, record the input term count N_j before applying the gate and the elapsed time t_j over its complete application.
Start before gate preparation/rebucketing and stop after merge, cutoff filtering, global finalization if selected, and naturally required completion coordination.
Use a monotonic wall clock with integer nanosecond output.
Store records in memory and write them outside the timed region.
Retain persistent buckets, scratch, thread pools, MPI ownership, and transport state across gates.
Do not time repeated convenience API calls that recreate scatter/gather or runtime for each gate.

Record gate type, application index, original circuit index, and Trotter-step index.
Heisenberg traversal reverses circuit order; retain both indices to prevent mislabeled traces.
Keep stateful circuit semantics and zero/threshold behavior unchanged.

Add meaningful tests for trace count/index alignment, input/output counts, gate-finalization inclusion by control-flow design, and unchanged numerical behavior.
Do not write brittle tests asserting exact wall-clock values.
Follow the repository's required Rust/Python/MPI gates for modified code.
Keep tracing off by default and outside hot per-term loops.
Compare traced and untraced complete propagation on representative small and large workloads using the existing interleaved protocol.
If overhead is material relative to the claimed effect, reduce it or report it; never silently subtract an estimated tracing tax.
Collect untraced complete-run timing for the main runtime panel if tracing materially perturbs it.

### Throughput

For an individual gate use R_j = N_j / t_j.
For a bin or aggregate use sum(N_j) / sum(t_j), not an unweighted average of rates.
One string processed by ten gates contributes ten input-string updates.
Do not use final or peak string count as a surrogate for total work.

The data comes from evolving simulations, so equal N_j does not imply identical branching, cancellations, or numerical state.
Provide raw points and group by meaningful gate family where necessary.
If plotting size bins, retain bin edges, number of gates, sum of inputs, sum of time, and separate gate-family summaries.
Treat per-gate plots as application-level observations rather than proof of equal-input kernel efficiency.

### Distributed timing

Record rank-local counts, gate durations, and communication counters without adding a diagnostic global barrier or all-reduce at every gate.
Combine traces after the run.
Global input count is the sum of rank-local input counts.
For the efficiency plot, the maximum rank-local gate duration may be used as an explicitly labeled critical-rank proxy when no synchronized global gate timer exists.
Ranks can run ahead on local gates, so this proxy is not exact globally synchronized gate elapsed time and its sum is not total propagation time.
Retain minimum/median/maximum rank durations to expose skew.
Use the maximum rank-local duration of the complete propagation after a common run-start synchronization for the runtime comparison, with the precise boundary documented.
Do not add worker/rank elapsed times together as wall time.
Keep summed worker busy phases distinct from elapsed phases and avoid double-counting contained subphases.

### Cycles and memory evidence

The recurring efficiency axis is updates per second.
For single-thread kernel/memory slides provide measured core cycles per input-string update where counters are accessible.
Record event, timing scope, multiplexing/scaling information, and frequency conditions.
Otherwise label a fixed-frequency conversion as **equivalent cycles/update at f_ref**, using t_wall * f_ref / N_updates.
Never call advertised GHz times elapsed time measured core cycles.
Do not request privileged counter access or change host settings to obtain optional counter fields.

Build a bytes-per-update traffic model including actual keys/coefficient width, temporary streams, reads, and writes.
For 127 qubits verify W and coefficient representation in the built variant rather than guessing payload size.
Use relevant node-local/all-core sustained bandwidth measurements for the roofline comparison.
Collect supporting counters and bandwidth on the same selected cluster architecture.
If counters are unavailable there, use the documented reference-cycle conversion and state the missing counter evidence; do not substitute measurements from a different architecture or workstation.
A profile or throughput knee alone does not establish bandwidth saturation.

## Phase C: execute a staged campaign

### C1. Smoke and consistency

Run the existing numerical/differential gates and a small shared-task cross-library configuration.
Check observable agreement or convergence against a matched published or library reference.
Retain existing strict term-count parity checks for paired library comparisons; do not relax a failed check merely to obtain timing data.
Explain legitimate semantic differences explicitly and keep such results out of equal-work speedup claims.
Do not add a large independent accuracy study.

### C2. Single-thread tolerance and historical curves

Run the fixed primary task over the cutoff grid for the current bucketed implementation, genuine internal variants, and external references.
Use the frozen CPU model/node class and verified physical-core-only placement for every variant and external baseline.
Warm runtime/JIT outside the timed propagation region; separately record setup if it matters operationally.
Use the existing interleaved A/B protocol for comparisons, especially small effects.
A default of five pairs is reasonable for practical comparison cells; adapt expensive frontier runs transparently, recording their smaller repeat counts and uncertainty.
Do not imply repeated-run confidence for a one-off capacity demonstration.

### C3. Threads and NUMA

At a few fixed cutoffs spanning meaningful working-set sizes, run a thread-count ladder from one physical core to the intended full-node allocation.
Include an intermediate and a near-memory-limit workload, rather than only tiny cases.
Compare the same implementation on one thread versus multiple threads.
Document one shared pool versus in-process ownership partitioning as separate configurations where applicable.
Do not wrap in-process partitioned execution in a placement command that restricts all pages/CPUs to one domain; follow the engine's placement contract.
Record actual affinity and pool size, not only requested values.
Interpret every thread-count point as a count of distinct physical cores and reject SMT sibling use.

### C4. Multiprocess, capacity, and hash comparison

Use valid power-of-two rank counts and one ownership partition per MPI rank, with rank placement matched to NUMA domains and the scheduler allocation.
Confirm MPI build support, matching MPI/mpi4py libraries, required thread support, and accessible shared build/output paths.
Do not let every rank start an unbound full-node pool.
The inspected MPI path sizes pools from affinity rather than `RAYON_NUM_THREADS`; verify the live behavior.

First obtain overlapping full-simulation results that also fit on one node.
Then extend the fixed-depth cutoff sweep until at least one lower-tolerance distributed calculation completes beyond the established single-node feasible regime.
Keep the initial observable small and replicated as required by the interface.
Use local distributed results and scalar expectation reductions, avoiding the default final full-operator gather on rank 0.
Include comparable scalar-output work consistently across configurations and document any excluded setup/final reduction time.

At a fixed distributed problem and resource allocation, compare random and cut ownership rows.
Record bytes sent, exported/received rows, number of communicating gates, communication/phase timing, full gate/run timing, and occupancy imbalance.
A smaller byte count is useful evidence even if runtime is unchanged, but must not be labeled a speedup.
Include any hash reseeding and its effect on local bucket structure as a confounder.
The goal is to measure existing hash engineering, not design another heuristic.

### C5. Resource and failure handling

Use exclusive allocations or the existing quiet-node measurement policy where appropriate.
Record resource estimates and incrementally launch frontier jobs so one underestimated cell does not consume an unbounded allocation.
Do not intentionally exhaust shared infrastructure to produce an OOM marker.
An isolated job/cgroup memory failure can establish the limit, but a documented scheduler allocation limit or measured footprint may support a clearly labeled inferred limit instead.
Distinguish physical RAM, allocated memory, application budget, and sampled/RSS peaks.
If the accessible allocation prevents the required capacity demonstration, report the missing evidence and ready-to-run job rather than inventing a successful frontier.

## Data contract

Write an immutable raw output for every attempted run, with enough provenance to reproduce it.
Use JSON/JSONL for raw structured records and CSV or Parquet for normalized plot tables.
Use null plus a reason for unavailable metrics, never zero or NaN disguised as a successful measurement.
Give each task, configuration, variant, and run stable IDs.

### `runs.jsonl`: one record per attempted run

- Schema version, campaign ID, run ID, task/configuration/variant IDs, repetition/pair index.
- Source commit and dirty-patch hash, dependencies, compiler/runtime versions, build features/flags, executable hash if practical.
- Task-file path/hash, published source, physical parameters, direction, observable/indexing, state, precision, policy, cutoff or count budget.
- Bucket/partition seeds, explicit cut blocks, bucket policy and engine selection.
- Slurm job ID, node names/class, sockets/NUMA, physical/logical cores, affinity, ranks, threads, allocated memory and wall-time limit.
- Trace enabled flag, timing boundaries, complete propagation wall time, setup/scatter/final-reduction times where separately available.
- Global initial/final terms, peak resident terms, per-rank peak RSS, memory provenance, observable real/imaginary values.
- Status: completed, invalid_hardware, numerical_mismatch, oom, timeout, scheduler_failure, build_failure, cancelled, or other explicit failure.
- Hardware-contract ID, architecture/topology preflight outcome, physical-core mapping, effective masks, actual pool sizes, and physical-core-only verification.
- Failure reason, exit/scheduler code, raw log and gate-trace paths.

A sum of per-rank high-water marks is not a simultaneous global peak; name it accordingly.
`peak_terms` is a between-gate resident count and excludes temporary expansion/scratch.
For OOM jobs recover scheduler memory/exit evidence where possible, preserving uncertainty if the sampler missed the peak.

### `gates.rank-N.jsonl`: per gate per rank, or one stream for unpartitioned runs

- Run ID, rank ID, application/circuit/Trotter indices, gate family and support/generator identifier.
- Local input/output terms, complete local gate duration in integer ns.
- Bucket bits/count, communication rows/bytes and partner counts where available.
- Existing compatible phase counters if collected, clearly marked cumulative or per-gate and elapsed or worker-busy.
- Any measured cycle counters with scope, or reference-cycle conversion metadata stored separately.

### Normalized tables

| File | Content |
| --- | --- |
| `efficiency_gates.csv` | Global input count, selected duration definition, throughput, gate family, rank-duration summaries |
| `efficiency_binned.csv` | Size/gate-family bins with summed inputs, summed time, counts, derived throughput |
| `runtime_tolerance.csv` | Completed propagation times, repeat summaries, cutoff, counts, memory, all attempted statuses |
| `thread_scaling.csv` | Fixed-task speedup and efficiency versus same implementation on one thread |
| `rank_scaling.csv` | Fixed-task rank scaling and separately labeled capacity/frontier data |
| `hash_communication.csv` | Random/cut comparison, traffic, imbalance, elapsed effect |
| `accuracy.csv` | Observable/reference values, source, matched parameters, discrepancy or convergence diagnostic |
| `memory_model.csv` | Traffic assumptions, bandwidth/cycles evidence, hardware and scope |

Provide one validation script that checks joins, IDs, units, trace length, nonnegative counts, completed-run timing presence, and failure exclusions from speedup calculations.
Check that aggregation never treats missing ranks as zero work or combines different scientific tasks under one comparison label.

## Slide-ready outputs

Produce editable plotting code and SVG/PDF previews of the recurring two-panel figure at these stages:

1. Baseline and external references.
2. Kernel improvement.
3. Historical threading attempt.
4. Annotated memory diagnosis.
5. Bucketed single thread.
6. Bucketed multiple threads.
7. Distributed extension to lower tolerance.

Supply separate compact figures for thread scaling, hash communication, and accuracy/convergence.
Keep past curves visible but muted and highlight the newly introduced variant.
Include negative or neutral results honestly.
Never assign a runtime to a failed run; put OOM/timeout markers outside the completed-runtime series and explain the marker.
Show the overlap where distributed execution may pay a communication penalty before the capacity extension.
Do not smooth through gate families, failures, or gaps to create a cleaner story.

The primary deliverable is data and reproducible analysis, not a finished slide deck.
Preserve raw traces and full run metadata even when plots use filtered/binned summaries.

## Deliverable layout

Place scripts in an appropriate repository benchmark location following local conventions and large raw data in a durable shared results directory.
Do not rely on node-local scratch for the final handoff.
A suitable results layout is:

```text
quera-talk-data/<campaign-id>/
  README.md
  campaign.json
  variants.json
  tasks/
  jobs/
  raw/
  tables/
  figures/
  analysis/
  evidence.md
  reproduce.sh
```

`README.md` must describe the workload, environment, metrics, plotting commands, and where large raw files live.
`campaign.json` records the planned and executed matrix, resource bounds, and cell statuses.
`evidence.md` maps E0–E9 and slide numbers to exact rows/figures, states each supported claim, and identifies any unfulfilled result.
Include a compact table of headline numbers with workload/resource scope; do not return only plots.
`reproduce.sh` should expose explicit preparation, submission, collection, and plotting commands rather than unexpectedly launching the full expensive campaign when inspected.
Archive the exact job scripts, source revision/patch, task files, and dependency lock information.

## Completion criteria

- All required E0–E9 evidence is produced, or each genuinely blocked item has a precise reason and a concrete prepared next step.
- Complete-gate throughput comes from persistent full simulations and includes the required overhead and truncation.
- Total propagation runtime is independently measured with aligned boundaries across engines.
- Historical curves correspond to real code/configurations, with bundled changes disclosed.
- Numeric consistency checks pass for reported comparable results.
- Every timing run and supporting performance measurement uses the same frozen CPU architecture/node class on the cluster, verified within the allocation.
- Every worker/rank uses physical cores only, with no concurrent SMT siblings and no overlapping rank ownership of a physical core.
- Thread/rank counts, hardware, build settings, and truncation semantics are explicit.
- Distributed capacity results do not gather the final operator onto one rank.
- Hash traffic changes and runtime effects are reported separately.
- OOM, timeout, unavailable, and numerical-failure states remain visible and are never treated as completed timing points.
- Raw data, normalized tables, plotting scripts, figure stages, and evidence mapping are durable and reproducible.
- The user receives exact output paths, measured headline findings, code changes, job status, and any remaining limitation.

Do not stop after writing a plan or submitting jobs when the requested data can still be collected and analyzed.
Do not continue optional algorithm work once the required data package is complete.
