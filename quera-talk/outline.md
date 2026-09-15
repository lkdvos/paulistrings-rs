# Pauli propagation at scale: “Wie niet sterk is, moet slim zijn”

Expanded slide-by-slide working outline for Lukas Devos's QuEra technical job talk

## How to use this draft

This is the expanded 37-slide structure: 30 technical slides and 7 slides about your QuEra vision. Each entry separates on-screen material from the spoken explanation. The speaker cues are a first draft to adapt to your voice, not a script to memorize. Figure descriptions specify assets to prepare, rather than figures already produced.

Keep the submitted title and abstract unchanged. The talk's core story is your progression from understanding a kernel to reorganizing the computation through algebraic structure. Use Rust as evidence of learning, and your established maintenance work as evidence of sustained software ownership.

The attached source snapshot has been inspected for the implementation details below. Benchmarks have not been rerun and performance results have not been independently reproduced. Bracketed items such as [measured speedup] need your data. Where the implementation determines the explanation, the entry explicitly says what to fill in. Conditional speaker cues must be revised if the measurements do not support them.

## Confirmed scope and source inspection

The following planning decisions come from Lukas. The attached paulistrings-rs-main.zip snapshot was subsequently inspected to verify the central implementation details. Direct GitHub access failed, so this is a review of the attached snapshot, not the current remote HEAD. No benchmark or cluster job has been run during this review.

- The human-facing and timed operation is one complete gate application, either Clifford or rotation.
- The independently owned work unit is a closed coset of partitions that can map into one another. A worker can read and write those partitions in place. In code, distinguish fine-grained buckets (local coset tasks) from ownership partitions (NUMA domains or MPI ranks). Rotation bucket pairs are the simplest illustrative case, not the complete scheduling abstraction.
- Local accumulation uses sort-merge. Truncation occurs at the end of each gate, through a cutoff or a histogram-based total-count policy.
- Gate timing includes overhead, communication, the gather/sort/merge loop, truncation, and completion synchronization.
- Throughput is measured along complete simulations, rather than through a separate saved-input microbenchmark campaign.
- Use one fully specified published benchmark configuration from the PauliPropagation paper or original IBM experiment. Select the exact source and convention when the benchmark entry points are available. Do not mix conventions between papers or invent parameters here.
- Accuracy/convergence is a consistency check against existing library results or published plots. This talk does not need a separate scientific accuracy study. Match parameters and observables and state the precision of the comparison, especially if reading values from a figure.
- Execute the required benchmark suite on the user's Slurm cluster. Hardware, partition, account, launch method, and available software details will follow. Existing Slurm scripts were found in the snapshot, but no new jobs have been submitted and no remote access has been established.
- Single-thread, multithread, multiprocess/distributed, peak-memory, and communication-reducing hash results are all required parts of the final story. These are presentation evidence to assemble from the existing implementation and necessary runs, not optional research directions.
- QuantumKitHub supplies the evidence for architecture, maintenance, collaboration, and mentoring. The Pauli project deliberately showcases technical ownership and performance research. Usability and maintainability are equally important to the role Lukas wants.

## Rehearsal budget

More slides means finer steps in the explanation, not a longer time slot. These are initial pacing targets. Adjust them after a spoken rehearsal with the actual figures.

| Section | Slides | Prepared time | Cumulative technical time |
| --- | --- | ---: | ---: |
| Introduction | 1–3 | 2:30 | 2:30 |
| Being strong: Pauli propagation | 4–9 | 6:30 | 9:00 |
| Optimizations and the memory bottleneck | 10–14 | 5:30 | 14:30 |
| Being smart: structured partitioning | 15–23 | 11:00 | 25:30 |
| Results and tuning | 24–28 | 6:00 | 31:30 |
| Technical conclusion | 29–30 | 1:30 | 33:00 |
| QuEra vision | 31–37 | 12:00 | Separate segment |

Slide 28 now includes essential hash-engineering evidence as well as distributed capacity. Its original 1:15 allocation is provisional: use a second build or slide 28b and retime the technical section during rehearsal. The original budget leaves approximately seven minutes for interruptions in the 40-minute technical slot, followed by the vision section and ten minutes of final Q&A. Some slides last only 30–45 seconds. Slides 20–23 are deliberately slower.

## Recurring performance figure

Use one evolving two-panel figure to carry the quantitative story. Keep baseline and external-library reference curves visible as each implemented improvement adds a curve. Highlight the newly introduced configuration and mute previous configurations without changing their identities. Keep colors, line styles, timing definitions, and axes consistent across appearances.

| Panel | Horizontal axis | Vertical axis | Purpose |
| --- | --- | --- | --- |
| Processing efficiency | Strings entering an update | Input-string updates per second, higher is better | Show how processing efficiency changes with working-set size |
| Full calculation | Truncation tolerance, decreasing from left to right | Total simulation wall time in seconds, lower is better | Show the cost and eventual feasibility of progressively less-truncated calculations |

Use logarithmic axes where the data spans orders of magnitude. Label selected tolerance points with measured peak string count and peak memory. Do not use a shared string-count axis above the tolerance panel unless the mapping is actually common to the compared runs. Peak counts and runtime need not vary strictly monotonically with tolerance because propagation and truncation interact.

### Unit of work and timing

A string processed by several rotations represents several input-string updates. For update j, record its input count N_j and elapsed wall time t_j. Report

\[
R_j=N_j/t_j,\qquad
R_{\mathrm{aggregate}}=\frac{\sum_j N_j}{\sum_j t_j}.
\]

Do not take an unweighted mean of per-update throughputs. Treat an update as one complete Clifford or rotation gate application. Include overhead, the gather/sort/merge loop, and the end-of-gate truncation in update timing. Include scheduling, synchronization, and communication attributable to an update when comparing threaded or distributed configurations. Use one elapsed time for the whole parallel update, not the sum of worker times. Measure total simulation wall time separately, documenting initialization and other timing boundaries.

For the efficiency panel, collect gate input counts and complete gate wall times along full simulations. This is the selected application-level measurement, not a controlled identical-input kernel benchmark. Label gate type and relevant circuit position, or present separate traces/aggregates where mixing them would obscure interpretation. String count alone does not determine branching, cancellation, or output growth. Use size-bin throughput as total input-string updates divided by total gate time within a bin. Do not sum simultaneous rank times or count only one rank's work in distributed throughput. Instrument external libraries to equivalent gate boundaries where possible, and disclose missing instrumentation rather than substituting incomparable numbers.
For the tolerance panel, hold circuit depth, angles, initial state, and observable fixed. Align precision and truncation semantics where possible and disclose differences. Provide convergence or reference-error evidence on slide 24. A lower tolerance means less aggressive truncation under the chosen rule, not automatically a guaranteed final error bound.

### Clock cycles for the single-threaded explanation

Use cycles per input-string update as an annotation on the single-threaded baseline/kernel comparison and in the memory calculation. Keep updates per second as the recurring efficiency axis through threading and multiple nodes.

If hardware counters are available, report measured core cycles for the stated single-thread timing region divided by its input-string update count. Label the counter and scope. If converting wall time at a fixed frequency instead, explicitly write “equivalent cycles/update at f_ref”:

\[
C_{\mathrm{equiv}}=\frac{t_{\mathrm{wall}}f_{\mathrm{ref}}}{N_{\mathrm{updates}}}
=\frac{f_{\mathrm{ref}}}{R}.
\]

Keep that reference frequency fixed. Do not call wall time multiplied by advertised GHz actual measured core cycles. Actual core frequency can vary with turbo and active-core count. Avoid a cycles axis in the parallel comparison: elapsed equivalent cycles measure performance, whereas summed worker core cycles measure aggregate processor effort. Neither needs to become an additional headline metric here.

### Memory narrative and the final extension

Separate the speed limit from the capacity limit. Annotate measured working-set sizes and relevant memory evidence on the efficiency panel. Use traffic measurements and profiling to distinguish bandwidth pressure from access latency. A bend in throughput alone does not prove a cache or bandwidth bottleneck.

On the tolerance panel, end a curve at its last completed run and mark an actually attempted memory failure at its tolerance with an OOM symbol outside the runtime data series. Do not assign failed runs an invented runtime or connect them as if they completed. Distinguish timeout, untested, and memory failure. Label the actual memory allocation/limit relevant to a failure, rather than assuming the machine's full installed RAM was available.

The multi-node finale extends the tolerance range with completed runs that no longer fit on one node. Show the overlap with single-node runs, including any communication penalty. Label node and thread counts, per-node memory, and aggregate memory. Keep fixed-problem speedup distinct from the increased scientific reach of a larger calculation. Extend the axes visibly at this final reveal rather than silently rescaling earlier figures.

### Figure appearances

| Slide | Figure build | Explanation |
| --- | --- | --- |
| 10 | Baseline and validated external references | Introduce both panels and their metrics |
| 11 | Add the actual kernel improvement | Annotate single-thread cycles/update |
| 12 | Add the attempted threading result | Show the measured limit of that decomposition |
| 14 | Annotate existing curves, without a new algorithm | Connect efficiency to traffic and memory limits |
| 25 | Add bucketed single-thread performance | Reveal locality/data-organization benefit before threading |
| 26 | Add bucketed multithread performance | Isolate the additional parallel benefit |
| 27 | Focused scaling view of the highlighted configuration | Explain scaling and bucket tuning |
| 28 | Add distributed runs, extend tolerance range, then compare hash choices | Demonstrate increased capacity and reduced communication |

Use visual builds or a focused enlargement of one panel when discussing code or profiles. Do not cram a full-size profile, code example, and both plots onto one slide. No curve is a schematic claim of improvement: populate it with measurements, including neutral or negative results.

## 0. Introduction

### 1. Title and idiom — 0:45

**On screen:** The submitted title, your name and affiliation, and a small English gloss: “If you aren't strong, you have to be clever.” Keep the title visually dominant.

**Speaker cue:** “This is a Dutch expression I grew up with. Pauli propagation is an interesting place to test it: we can exploit an enormous number of very cheap operations, but eventually we have to think about how we organize those operations. By the end, I hope to make the case for being both.”

**Audience takeaway:** The talk connects hardware efficiency with algorithm design.

**Transition:** Introduce the experience that led you to this particular problem.

### 2. Who I am — 0:45

**On screen:** Your current role, your tensor-network and symmetry background, and a small selection of software projects representing your contributions. Use short contribution labels rather than a wall of logos.

**Speaker cue:** “Most of my background is in tensor networks and symmetry-aware numerical algorithms, primarily in Julia. The part I particularly enjoy is turning that mathematical structure into software that other researchers can use. This project gave me a chance to apply the same habits to a different representation and a new language.”

**Audience takeaway:** The subject is new, but the underlying engineering experience is substantial.

**Transition:** Explain the original encounter with PauliStrings.jl.

### 3. How I arrived at this problem — 1:00

**On screen:** A short timeline: JuliaCon 2024 encounter; CPU exploration and contributions; unresolved multithreading; current Rust project.

**Speaker cue:** “While reviewing submissions for the JuliaCon quantum minisymposium, I encountered PauliStrings.jl. I was also learning more about caches and pipelines, and this looked like an interesting kernel to experiment with. I contributed several improvements, but never found a threading strategy I was satisfied with. Preparing for this opportunity gave me a concrete reason to return to that question and develop it further.”

**Preparation:** Identify the contributions you will name if asked. Discuss your experience and the public project without revealing submission-review details.

**Transition:** “Let me first show you what one of these updates actually has to do.”

## 1. Being strong: Pauli propagation

### 4. Evolving an observable — 1:00

**On screen:**

\[
\langle O\rangle=\operatorname{Tr}(\rho U^\dagger O U),\qquad O=\sum_p c_pP_p.
\]

A small circuit and an observable propagated backwards through it.

**Speaker cue:** “We expand the observable in Pauli strings and update that expansion through the circuit. The individual operations can be very cheap. The difficulty is that the number of terms can grow rapidly, so practical calculations often need truncation.”

**Audience takeaway:** The central object is a sparse operator, and the scientific output is an expectation value.

**Transition:** Give the abstract operation a concrete running example.

### 5. The kicked Ising benchmark — 1:00

**On screen:** Heavy-hex connectivity and one circuit period. Label [gate ordering], [angles], [initial state], [observable], and [depth range].

**Speaker cue:** “I will use this same circuit throughout the talk. It repeatedly applies [actual gates], and we calculate [actual observable]. I use it as a demanding, reproducible workload for studying the implementation.”

**Preparation:** Fill in the exact circuit convention from your benchmark. Distinguish qubit count from retained term count, since the latter directly controls the data volume. Cite the benchmark source in slide notes.

**Transition:** Zoom in from the circuit to the representation of one term.

### 6. Representing a Pauli string — 1:15

**On screen:** A short Pauli string mapped to two bit vectors \(p=(x,z)\). Show the label product \(p\oplus q\) and commutation test

\[
[p,q]=x_p\cdot z_q+z_p\cdot x_q\pmod 2.
\]

**Speaker cue:** “The label of a product is an XOR. Commutation is another small collection of bit operations. We still have to track phases correctly, but we can separate that bookkeeping from the label routing that matters later.”

**Visual build:** Reveal the encoding first, then XOR, then the commutation test.

**Scope:** State the bit convention used in your code. Do not suggest XOR alone determines the full operator product.

**Transition:** Apply those primitives to gates.

### 7. Clifford gates and Pauli rotations — 1:15

**On screen:** A Clifford's one-output mapping beside a Pauli rotation's commuting and anticommuting cases. Highlight output labels \(p\) and \(p\oplus g\).

**Speaker cue:** “Cliffords preserve individual Pauli strings up to sign. A general Pauli rotation can produce a second contribution when the term anticommutes with its generator. The important label relationship is very simple: the extra branch differs by XOR with the same generator.”

**Preparation:** If displaying the full rotation formula, use the precise conjugation and angle conventions from the implementation. Otherwise label the branches by their coefficients without deriving signs.

**Audience takeaway:** Every input follows a highly structured routing rule.

**Transition:** Explain why many simple updates become a difficult collection operation.

### 8. Managing a growing Pauli sum — 1:00

**On screen:** Two inputs producing a contribution to the same output label. Beside this, a compact comparison of sorted storage and a hash map: lookup, accumulation, and storage/access pattern.

**Speaker cue:** “After generating contributions, we have to combine equal labels. The data structure determines how we find those collisions and how much memory traffic that creates. This is where implementation choices become much more consequential than the arithmetic on one string.”

**Audience takeaway:** Generating a branch and assembling the resulting sparse sum are different costs.

**Transition:** Explain how growth is controlled and what constitutes comparable work.

### 9. End-of-gate truncation — 1:00

**On screen:** A full gate ending in accumulation followed by truncation. Show cutoff-based retention and histogram-based control of total string count. A weight cutoff can remain brief background if useful, rather than implying it is the primary implemented policy.

**Speaker cue:** “We combine the gate's contributions and truncate at the end of the gate. The library supports a cutoff and a histogram-based total-count strategy. All of that cost is included in the gate timing. For the main tolerance sweep I use [published, matched cutoff configuration].”

**Verified implementation:** `CoefficientThreshold` filters fully summed coefficients inside the merge: end-of-gate truncation is a semantic boundary rather than a mandatory separate cutoff pass. `ApproxTopN` uses 2048 exponent bins of squared magnitudes, chooses an octave edge, and retains at most the requested count. It can undershoot because it retains whole bins. Partitioned execution all-reduces the histogram and length before applying the shared edge. Exact `TopN` also exists but is unavailable in the partitioned policy interface. Compare the observable or convergence trend with a matched existing-library or published result.

**Transition:** “With the workload and approximation fixed, we can ask where the time goes.”

## 2. Optimizations and the memory bottleneck

### 10. Baseline performance — 1:00

**On screen:** Introduce the recurring two-panel figure: input-string updates per second against input count, and total wall time against decreasing tolerance. Show your baseline and validated external-library curves. Label representative peak counts and memory footprints. Explain that throughput is higher-is-better, while runtime is lower-is-better.

**Speaker cue:** “This is the starting point on [hardware and configuration]. The workload grows in [measured way], and this is where the implementation begins to struggle. My first question was how much of this cost could be removed inside the kernel.”

**Preparation:** Use the work and timing definitions in the recurring-figure section. Define error bars or repetition method. Keep circuit depth fixed in the tolerance sweep. Use package names neutrally, with version details in notes.

**Transition:** Zoom into the first optimization hypothesis.

### 11. Making the kernel faster — 0:45

**On screen:** Explain one actual kernel improvement, then reveal its curve on the recurring figure. Annotate representative single-thread measurements with core cycles/update, or explicitly labeled equivalent cycles/update at a fixed reference frequency. Use a short before/after fragment only if it remains readable alongside a focused plot view.

**Speaker cue:** “I explored [actual optimization]. It improved [measured component] by [result], but the end-to-end change was [result]. That difference was a clue about where the larger opportunity lay.”

**Scope:** Preserve the real history. If SIMD was only an idea you evaluated conceptually, call it a hypothesis rather than an implemented experiment.

**Transition:** Explain why adding workers did not straightforwardly solve the remaining problem.

### 12. The threading obstacle — 1:00

**On screen:** Show one attempted decomposition, such as worker-local dictionaries followed by consolidation, then add its measured curve to the recurring figure. Highlight the consolidation, allocation, or extra passes identified by measurements. Keep throughput in updates per second when moving to multiple threads.

**Speaker cue:** “It is easy to divide the input terms. It is harder to ensure that contributions reaching the same output label combine efficiently. In this approach, [actual cost] limited the benefit of parallel work.”

**Audience takeaway:** The central challenge is organizing output ownership and accumulation.

**Keep in backup:** Other attempts, including parallel sorting/merging, with their actual measurements.

**Transition:** Inspect the cost directly.

### 13. What profiling revealed — 1:00

**On screen:** A readable, cropped profile with two or three annotations. Add a relevant measured counter or scaling observation if available.

**Speaker cue:** “The profile puts most of the time in [measured paths]. That locates the work. To understand whether the limiting resource is memory bandwidth, access latency, or computation, I also looked at [actual evidence].”

**Scope:** Avoid presenting a flame graph alone as proof of bandwidth saturation. Different representations may suffer from different memory limitations.

**Transition:** Quantify the expected traffic.

### 14. The memory budget — 1:45

**On screen:** Bytes per term, estimated reads/writes per update, and

\[
R_{\mathrm{updates}}\lesssim\frac{B_{\mathrm{sustained}}}{T_{\mathrm{bytes/update}}}.
\]

Return to the recurring figure and compare the estimate with measured throughput on the relevant workload. For the single-thread case, also translate the estimate into equivalent cycles/update at the stated reference frequency. Annotate actual working-set sizes and memory evidence. Point out separately where capacity prevents a complete lower-tolerance run.

**Speaker cue:** “Even if the arithmetic were free, these passes still require approximately [bytes] per update. At [sustained bandwidth], that corresponds to [estimated rate]. The observed [rate/counters] suggest [supported diagnosis].”

**Preparation:** Distinguish payload bytes from actual traffic, including temporaries and writes. Use a bandwidth number representative of the workload, rather than blindly taking the hardware specification. A single thread may not saturate the available memory bandwidth. Keep measured core cycles distinct from reference-frequency conversions. Mark only measured memory failures on the runtime panel.

**Transition:** “To make a larger difference, I needed to change how the data moves through the computation.”

## 3. Being smart: structured partitioning

### 15. Opportunities in memory use — 0:45

**On screen:** Four short possibilities: smaller representation; fewer transfers; more reuse while resident; access to more bandwidth. Add a separate note that capacity limits the largest calculation.

**Speaker cue:** “These are related but distinct opportunities. The approach I will describe reorganizes the work into smaller independent pieces. The measurements will tell us how much that helps through locality and how much through parallel execution.”

**Transition:** Introduce the small ordering problem that suggested the structure.

### 16. Flipping one bit — 0:45

**On screen:**

```text
Sorted input:   [0, 1, 2, 4, 5, 7]
XOR with 2:    [2, 3, 0, 6, 7, 5]
```

Display this as highlighted integer or binary labels in the eventual slide, not a code listing if a clearer visual is available.

**Speaker cue:** “Suppose I have a sorted list and want to flip the same bit in every element. The operation is trivial, but the output is no longer sorted. Do I need to sort everything again?”

**Audience takeaway:** A cheap transformation can create an expensive organization problem.

**Transition:** Reveal the order that survives.

### 17. Two sorted streams — 1:00

**On screen:** The same example partitioned by the bit with value 2:

| Input bit | Input subsequence | After XOR with 2 |
| --- | --- | --- |
| 0 | [0, 1, 4, 5] | [2, 3, 6, 7] |
| 1 | [2, 7] | [0, 5] |

Merged result: \([0,2,3,5,6,7]\).

**Speaker cue:** “Within each subsequence the operation adds or subtracts the same value, so the order survives. We can recover the output with a linear merge.”

**Scope:** This primitive is \(O(N)\). If mentioning independent sorting of balanced buckets elsewhere, its comparison-sort cost is \(O(N\log(N/b))\), plus partitioning. Those are different operations.

**Transition:** Shift attention from preserved ordering to predictable movement.

### 18. Whole-bucket movement — 1:00

**On screen:** The two bit-defined buckets before and after XOR. Show that XOR either preserves each bucket or swaps the buckets, depending on that bit of the XOR mask.

**Speaker cue:** “The property I want to keep is that all labels in one bucket share a destination bucket. We can reason about the movement of a whole collection without inspecting each label's destination separately.”

**Scope:** The ordering property was specific to the example. The generalization will preserve routing, with local ordering handled separately.

**Transition:** “Can we construct many buckets with this same routing property?”

### 19. A predictable destination rule — 1:00

**On screen:** Source label \(p\), bucket \(h(p)\), generator \(g\), and the desired relationship

\[
h(p\oplus g)=h(p)\oplus h(g).
\]

**Speaker cue:** “For a fixed generator, I want one bucket-level rule that works for every label. Ordinary hashing does not generally give this relationship. Here the bucket labels themselves should respect the XOR structure.”

**Audience takeaway:** The desired algebra is motivated by a concrete scheduling requirement.

**Keep in backup:** The precise sense in which this requirement implies an affine or linear map.

**Transition:** Show a hash family that supplies the property directly.

### 20. A GF(2)-linear hash — 1:30

**On screen:**

\[
h(p)=Ap,\quad A\in\mathbb F_2^{k\times 2n},\qquad
A(p\oplus g)=Ap\oplus Ag.
\]

Illustrate each hash bit as a parity of selected label bits. Add “buckets = cosets of \(\ker A\)” after the routing equation is understood.

**Speaker cue:** “A binary matrix gives exactly this behavior. Each output bit is a parity. For a rank-k matrix, there are 2-to-the-k possible bucket IDs, and the buckets are cosets of its kernel. We can now compute the partner offset once for the generator.”

**Scope:** Equal coset sizes in the full vector space do not guarantee equal occupancy in the operator being propagated.

**Transition:** Apply the map to the two branches of a rotation.

### 21. Rotations couple bucket pairs — 1:30

**On screen:** Let \(d=h(g)=101\). Show the four disjoint pairs of three-bit bucket IDs:

| Pair | Bucket IDs |
| --- | --- |
| 1 | 000 and 101 |
| 2 | 001 and 100 |
| 3 | 010 and 111 |
| 4 | 011 and 110 |

A term can stay in its bucket or contribute to its partner. Add the \(d=0\) singleton case as a small final build.

**Speaker cue:** “The original branch stays in bucket b, and the extra branch goes to b XOR d. Applying the same offset twice returns to b. Therefore the bucket graph splits into pairs, and no rotation contribution leaves its pair.”

**Audience takeaway:** Independent work units follow from the update algebra.

**Transition:** Use the rotation pair as the simplest case, then generalize to closed cosets of partitions for a complete gate.

### 22. Closed cosets and in-place gate application — 2:00

**On screen:** A rotation pair, followed by a general closed group of partition IDs. Distinguish a storage partition (a coset of the hash kernel in label space) from a work unit (a closed coset of partition IDs for the gate). Give each work unit exclusive ownership of its partitions.

```text
for each gate, Clifford or rotation:
    begin complete-gate timing
    determine the gate's closed partition cosets
    perform required communication and ownership coordination
    parallel for each independently owned coset:
        gather the required local data
        apply the gate and sort changed-label contributions
        merge/reduce, applying cutoff to fully summed coefficients
        update owned buckets using reusable scratch
    apply histogram-based total-count finalization if selected
    complete required synchronization
    end complete-gate timing
```

**Speaker cue:** “One gate is the operation we ask the library to perform and the operation we time. Internally, the independent unit is a closed coset of partitions. A task owns all the partitions that can exchange contributions within that unit, so it can read and write them in place. The rotation pairs we just saw are a particularly simple example.”

**Verified construction:** `Gf2Span::new` takes the span of prepared bucket deltas even if the delta set is not XOR-closed. For gate label deltas D, let V = span(h(D)). Its cosets partition the bucket-index space into closed work units. `fill_coset` swaps source columns into reusable scratch, gathers into output-member runs, sorts changed-label contributions, and merges/reduces into live bucket slots. Cutoff filtering is fused after summation, while the driver owns global gate finalization. This pseudocode is conceptual, not a literal transcription; communication can overlap local processing.

**Scope:** The gate completes with truncation. Histogram-based total-count selection may need shared/global information beyond each independently updated coset. Include that cost rather than describing the whole timed gate as free of coordination. A Clifford can split a single partition across several destinations while remaining inside a closed work-unit coset; the scheduler does not require every individual partition to move intact.

**Transition:** Show how sort-merge handles the data inside the independently owned set.

### 23. Sort-merge inside the work unit — 1:30

**On screen:** Gathered source partitions, the generated sorted runs, their merge/accumulation, and the owned partitions updated in place. Mark live scratch/output storage when computing the working-set footprint.

**Speaker cue:** “The hash gives us a closed set of partitions to work on. Within that set, accumulation uses sort-merge. We still need to combine equal labels, but we can do so within the owned work unit. The important implementation question is which data remains live and how often we move it.”

**Verified mechanism:** `fill_coset` swaps source bucket columns into scratch before using bucket slots as destinations. The identity-delta stream remains sorted and skips sorting. Other contributions, including received rows, join the rest stream, which is sorted before `merge2_into` combines duplicates and applies `keep_term`. Dense identity plans can borrow original key columns instead of materializing them again. Reusable capacities circulate through scratch and buckets. A layer-level choice selects adaptive comparison or radix sorting. Use measured traffic/cache evidence to explain gains, not code structure alone.

**Audience takeaway:** Algebraic ownership makes local sort-merge and in-place updates possible. The data layout and traffic determine the practical gain.

**Transition:** “We can now measure the complete gate application, including all the coordination and truncation it requires.”

## 4. Results and tuning

### 24. Correctness and comparable accuracy — 0:45

**On screen:** One compact consistency comparison against a matched existing-library result or published observable/convergence plot. State the benchmark source and end-of-gate truncation policy. Use an exact small-system reference if already available, without expanding this into a separate accuracy study.

**Speaker cue:** “Before comparing speed, I checked [actual checks]. For the approximate benchmark, these results show [supported accuracy statement]. The timings that follow use [matching settings or explicitly described differences].”

**Scope:** Show actual tests, not a proposed validation checklist presented as completed work. Truncation order and floating-point reduction order can affect term counts and final results.

**Transition:** Present the main practical result.

### 25. Single-thread performance with bucketing — 1:30

**On screen:** Return to the recurring figure with all previously introduced reference curves. Add the bucketed single-thread implementation to both panels. Highlight its throughput and full-calculation runtime, using the accuracy evidence from slide 24. Reveal the threaded curve on the next slide.

**Speaker cue:** “Before adding threads, this organization changes throughput by [measured result]. Across the tolerance sweep, that translates into [runtime result]. The reference implementations remain on the plot, so we can see both the effect within my implementation and the comparison with existing software.”

**Preparation:** Include versions, hardware, precision, thread counts, timing boundaries, and truncation semantics in notes or backup. Credit relevant packages. Avoid personal commentary or language rankings unsupported by the comparison.

**Transition:** “To understand where the improvement comes from, we can compare versions within this implementation.”

### 26. Separating bucketing from threading — 1:00

**On screen:** Add the bucketed multithread curve to the recurring two-panel figure. Explicitly compare baseline, bucketed one thread, and bucketed multiple threads at selected matched workloads. Label thread count and keep throughput in input-string updates per second. Preserve prior curves and distinguish the bucketing gain from parallel speedup.

**Speaker cue:** “This comparison holds [controlled components] fixed. Bucketing alone changes runtime by [result]. Adding threads gives [result]. That separates the contribution of data organization from the additional parallel resources.”

**Scope:** If the baseline uses different kernels or other changes, disclose them. An ablation isolates an effect only to the extent the configurations control other variables.

**Transition:** Examine how that additional parallel benefit develops.

### 27. Thread scaling and bucket tuning — 1:30

**On screen:** Use a focused scaling view of the configuration just highlighted in the recurring figure. Show a fixed-workload thread-scaling curve with an ideal reference, and a smaller bucket-count sweep if legible. Define speedup relative to the same bucketed algorithm on one thread, not the original baseline. Retain the configuration colors.

**Speaker cue:** “Scaling remains [measured behavior] until [regime]. The bucket count trades off working-set size and available tasks against overhead and occupancy imbalance. On this workload, [measured choice] works well.”

**Preparation:** If both plots need lengthy explanation, make bucket tuning slide 27b rather than shrinking or rushing. Use measured evidence to explain flattening. Do not infer cache fit solely from a performance knee.

**Transition:** Move from how quickly a problem runs to how large a problem fits.

### 28. Multiple nodes and communication-aware hashing — 1:15 plus rehearsal adjustment

**On screen:** Add the distributed configuration to the recurring figure and visibly extend the tolerance range. End single-node curves at their last completed runs and mark measured memory failures separately. Show overlapping workloads, then completed lower-tolerance runs enabled by additional memory. Label peak string counts, per-node and aggregate memory, and node/thread counts. Keep the distributed ownership diagram in backup or a separate visual build.

**Speaker cue:** “At this point the single-node calculation runs out of available memory. Across [nodes], we can complete these additional lower-tolerance calculations, reaching [peak count] strings. In the overlap, the runtime changes by [measurement], including communication. The final extension is about which calculations become possible as well as how quickly they run.”

**Hash engineering build:** At a fixed distributed workload, compare the existing hash choices using communication bytes per gate, communication time, and full gate time. State the rank/thread layout and show occupancy or imbalance alongside the communication result if relevant. This evidence is essential. Use a second build or slide 28b if it cannot fit legibly with the tolerance extension. `PartitionRows::cut` uses z-only block-parity rows: X generators have zero partition delta, while ZZ(i,j) has the XOR of the endpoints' block labels. Thus X rotations and within-block ZZ rotations avoid cross-rank row movement; cut-edge ZZ rotations can communicate. This ownership hash is separate from the fine-grained bucket hash. `phase_breakdown` exposes `--partition-rows random|cut`. Retain occupancy measurements because reduced communication does not by itself establish balanced work.

**Preparation:** Assemble the required distributed and hash results. Retain empty data placeholders until measurements are available; do not downgrade these results to an optional outlook.
**Scope:** Independent arithmetic does not imply communication-free distributed execution. For the capacity demonstration retain local distributed results and reduce the scalar observable; default gathering would materialize the full sum on rank 0. Initial input is replicated in the current interface, which is practical for a small starting Pauli observable. MPI uses one ownership partition per rank with a power-of-two rank count. Keep detailed scheduling in backup.

**Transition:** Summarize what the evidence establishes.

## 5. Technical conclusion

### 29. What the approach establishes — 0:45

**On screen:** The bucket-routing equation, the strongest supported performance result, and the corresponding scope. Return to the idiom visually or verbally.

**Speaker cue:** “The key observation is that respecting the XOR structure gives predictable movement between buckets. That lets us organize independent work and, in this implementation, produces [supported outcomes]. The hardware still matters. The algorithm makes better use of it.”

**Audience takeaway:** The contribution is a specific organization of computation, supported by measurements.

**Transition:** State what is still needed to turn the project into a broadly useful tool.

### 30. Scope and next steps — 0:45

**On screen:** Current demonstrated scope, then the next software/application work: [actual API needs], documentation, validation breadth, and integration. Keep GPU and distributed extensions brief and conditional where untested.

**Speaker cue:** “I chose a performance research topic because it gives a concrete technical story. Interfaces and maintainability matter just as much to me, and my experience there comes primarily from QuantumKitHub. I would like to work in a setting where making software useful to collaborators is a central, sustained part of the work.”

**Scope:** Avoid an unrestricted “kernel- and truncation-independent” claim. Local kernels and policies can vary, but global decisions or other transformations can introduce additional requirements.

**Transition:** Explicitly mark the shift to the requested QuEra vision segment.

## 6. QuEra vision

### 31. Why QuEra — 1:30

**On screen:** Scientific simulation connected to device-relevant questions, with short labels for the advertised themes of dynamics, noise, and reusable software.

**Speaker cue:** “I want to work closer to the scientific questions that determine which tools are most useful. I also want an environment where interface design, maintenance, and helping colleagues use the software have sustained attention. Those are essential to usable scientific software, and they are a large part of what I enjoy doing.”

**Preparation:** Frame this around the role description and what you have learned in interviews. Do not imply knowledge of internal priorities you have not been told about.

**Transition:** Connect your established technical background to those needs.

### 32. What I bring from tensor networks — 1:30

**On screen:** One concrete example from your work, with a short mapping from mathematical structure to numerical method and scientific use. Include numerical linear algebra or differentiation only where it supports the example.

**Speaker cue:** “Tensor-network work has taught me to think carefully about representation, approximation, and the numerical operations that dominate a calculation. For example, in [project], I was responsible for [contribution], which enabled [supported outcome]. Those habits apply when choosing and implementing other simulation methods as well.”

**Preparation:** Choose a contribution you can explain in under a minute. Avoid a second technical lecture.

**Transition:** Add the software responsibilities that extend beyond algorithm development.

### 33. QuantumKitHub: architecture and shared software — 1:45

**On screen:** One concrete QuantumKitHub example linking an architectural/interface decision to downstream use, plus a short contribution label for maintenance, collaboration, or mentoring. Show the original problem, your responsibility, and the effect on other developers or users. Use a second build if needed; avoid a package-name inventory.

**Speaker cue:** “QuantumKitHub is where I can best show the architectural and collaborative side of my work. In [specific example], we needed [user or developer need]. My responsibility was [design and maintenance contribution], which allowed [effect]. I care about this as much as performance: the interface and continuing maintenance determine whether other people can build on the method.”

**Preparation:** Possible topics include interface design, cross-package integration, differentiation support, or mentoring. Select based on the clearest evidence of your own responsibility.

**Transition:** Explain the kind of workflow you would like to help the team support.

### 34. A scientific workflow I would like to support — 2:15

**On screen:** A compact workflow with a concrete question, model assumptions, reference calculation, scalable solver, and comparison of scientifically relevant outputs. Use a driven noisy simulation as a hypothetical example if helpful.

**Speaker cue:** “For example, we might want to understand how a control choice affects a particular observable in a noisy model. I would start by agreeing on the quantity and accuracy we need, establish a reference in a tractable regime, and then choose the method that reaches the relevant scale. The useful output is a calculation colleagues can interpret and repeat.”

**Scope:** Present the example as a possible workflow, not an assertion about QuEra's current roadmap. Mention candidate methods only to explain how the regime determines the choice. Pauli propagation need not be the selected method.

**Transition:** Describe your working approach within such a project.

### 35. How I approach a new problem — 1:45

**On screen:** The questions guiding your decisions: What is the scientific output? What can we validate? What limits the current approach? What change can we integrate and measure?

**Speaker cue:** “The Pauli project is an example of how I work: understand the representation, establish what the code is doing, measure the limiting cost, and change the part that matters. In a team setting, I also want the users involved in choosing the target and assessing whether the result solves their problem.”

**Rust/LLM option:** If you want to address AI assistance proactively, use one factual sentence about the tasks it helped with and your actual review/validation. Fill this in from your real process. Keep a detailed account ready for questions.

**Transition:** Make that approach concrete for joining an existing team.

### 36. How I would start with the team — 1:45

**On screen:** A proposed sequence: learn the current workflow; agree on one bounded contribution; establish reference behavior; implement and integrate; assess the next need. Avoid speculative dated promises.

**Speaker cue:** “I would first learn how the team uses its existing tools and where the recurring difficulties are. Then I would like to agree on one bounded workflow with a clear scientific output, reproduce the baseline, and take responsibility for an improvement through integration and documentation.”

**Audience takeaway:** You want to contribute within the team's context and own delivery.

**Transition:** Explain the longer-term responsibility you want to build toward.

### 37. The responsibility I want to grow into — 1:30

**On screen:** Substantial component ownership, architectural contribution, and support for colleagues, expressed in short concrete phrases. Finish with a discussion prompt about the team's most important simulation needs.

**Speaker cue:** “I would like to own substantial pieces of scientific software and help shape how they fit together. I want to stay close to the science while building on the maintenance and mentoring experience I already have. That combination of technical depth and shared responsibility is what I would like to develop further at QuEra.”

**Closing discussion prompt:** “Where do you see the biggest gap today between the simulations the team needs and what the current tools make practical?”

## Preparation order

Build slides 16–23 first. They contain the core explanation and will determine how much context the audience needs. Next prepare the real results for slides 24–28, then tune the baseline and memory story to the evidence. Complete the QuEra examples before polishing introductory wording.

Prepare the recurring figure from one consistent benchmark dataset and export its staged appearances, preserving reference curves and axis limits until the explicit final extension. Record per-update input counts and timing, complete-run wall time, tolerance, peak string count, peak memory, configuration, and completion/failure status. Collect single-thread core-cycle counters where available, or use a clearly labeled fixed reference-frequency conversion. Use the same circuit, label notation, and bucket colors throughout. Reuse a visual through successive explanatory builds when it helps the audience track the same object. Put one main plot on each result slide where possible. Keep detailed configuration available in notes and backup.

## Rehearsal decisions

Time the spoken explanation, including pauses to read plots and equations. Do not compensate for dense algebra by speaking faster. If the technical section exceeds 33 minutes without questions, first shorten the kernel history and tuning details. Preserve the update rule, bucket-pair argument, local algorithm, and correctness evidence.

If questions consume the buffer, shorten the historical optimization details and the depth of slide 27's tuning explanation. Preserve slide 28's distributed capacity and hash-engineering results, which are essential to the agreed story. Transition to the QuEra section by the end of the technical allocation. The final segment is a requested part of the interview and should retain its own time.

## Backup slides

Prepare these as individually accessible answers, not an additional lecture.

| Topic | Question it answers |
| --- | --- |
| Phase and rotation conventions | Are the update signs and coefficients correct? |
| Exact benchmark definition | What circuit, observable, and truncation settings did you run? |
| Correctness and convergence | How do you know the faster computation is scientifically comparable? |
| Benchmark environment | Which versions, compilers, hardware, threads, and timing boundaries were used? |
| Affine necessity proof | Why this hash family? |
| Multiple generators | What exactly remains independent across a full layer? |
| Clifford handling | How do general Clifford transformations interact with the partition? |
| Local ordering | Does your hash preserve sorting, and how do local updates work? |
| Hash occupancy and tuning | What happens for strongly structured or imbalanced operators? |
| Truncation variants | Which policies remain local, and which require coordination? |
| Memory traffic and cache measurements | Is this bandwidth-bound, latency-bound, or a combination? |
| NUMA and distributed ownership | Where does data live and when does it move? |
| Rust and LLM assistance | What did you learn, delegate, design, and verify? |
| Alternative methods and workloads | Where would this method be a poor choice? |

### Technical details to settle before finalizing the claims

1. **Translation requirement.** If \(h(p\oplus g)=h(p)\oplus\delta(g)\) for all labels, evaluating at zero gives \(\delta(g)=h(g)\oplus h(0)\). Therefore \(h(p)\oplus h(0)\) is linear. State the assumed XOR action when discussing necessity.
2. **Translation-generated closure.** For generators \(g_1,\ldots,g_m\), hash-space orbits lie in cosets of \(D=\operatorname{span}\{Ag_1,\ldots,Ag_m\}\). These contain \(2^{\dim D}\) bucket IDs. The timed operation is a gate. The code constructs span(h(D)) from the prepared gate's realized deltas, including Cliffords. Rotation pairs are a special case.
3. **Cliffords.** For an invertible symplectic label map \(S\), whole-bucket routing under a fixed hash requires \(AS=BA\) for some induced map \(B\), equivalently preservation of \(\ker A\). This condition applies to whole-partition routing, not to the more general closed work-unit cosets described by Lukas. The implementation encloses prepared deltas in span(h(D)), whose cosets remain closed. Rehashing is not required merely because one bucket splits.
4. **Truncation.** Local coefficient and weight rules can fit bucket ownership. A global term budget or error allocation may require global information. Aggregation and truncation order must match the stated scientific semantics.
5. **Ordering.** Local accumulation uses sort-merge. Predictable routing alone does not establish sorted order. The identity stream skips sorting, the rest stream is sorted, and input columns are swapped into scratch before the output merge.

## Evidence and preparation priorities

### Essential for the main talk

- Exact benchmark definition and versions.
- Consistency comparison with an existing library or published observable/convergence plot at matched parameters.
- Recurring two-panel figure with staged baseline, kernel, attempted threading, bucketed single-thread, bucketed multithread, and distributed results.
- Fixed-depth tolerance sweep with total wall time, peak string counts, peak memory, and actual completion/failure status.
- Comparable update-throughput data and single-thread cycles/update annotations with explicit measurement or conversion conventions.
- Main end-to-end result with interpretable accuracy or truncation semantics.
- Internal ablation separating bucketing from threading.
- Fixed-workload threading curve.
- Memory-traffic calculation supported by measurements where possible.
- Concrete bit-flip example and bucket-pair explanation.
- Two examples of your sustained scientific software contributions for the QuEra segment.

### Required results to assemble

- Single-thread and multithread speedups, plus multiprocess/distributed speedups at overlapping workloads.
- Peak-memory measurements and a lower-tolerance calculation enabled by multiple nodes.
- Hash-engineering comparison showing communication volume, communication time, and full gate time at fixed workload.
- Relevant bucket/occupancy information to interpret locality and communication tradeoffs.

These are all essential. No additional exploratory workload or optional research campaign is required for the presentation. The remaining execution work is to inspect existing benchmark entry points, prepare the necessary Slurm jobs once access/configuration is supplied, collect results, and build the figures. Do not present planned runs as completed measurements.

### First cuts if rehearsal runs long

1. Additional unsuccessful optimization approaches.
2. Detailed hash construction and tuning.
3. Detailed distributed mechanics beyond the required capacity, speedup, and hash-communication results.
4. The full necessity proof and Clifford condition, while retaining accurate scope language in the main talk.

Keep the rotation update, bucket-pair argument, correctness evidence, and QuEra vision intact. Rehearse a concise path through slides 16–23 so interruptions do not erase the central contribution.

## Sources and attribution to carry into slide notes

- [QuEra Scientific Software Engineer – AMO Simulations posting](https://job-boards.greenhouse.io/queracomputinginc/jobs/5310810008): role alignment, based on the posting reviewed in the preceding discussion.
- [paulistrings-rs](https://github.com/lkdvos/paulistrings-rs): add the exact commit used for the presented results. The attached snapshot was inspected for the core implementation and benchmark interfaces. Record the actual cluster checkout commit on each benchmark run.
- Select one exact configuration from the PauliPropagation paper or the original IBM paper after inspecting benchmark entry points. Record that source and all compared packages' references and versions when preparing figures.
- Credit PauliStrings.jl for the original inspiration and relevant existing implementations for their contributions. Keep comparisons focused on methods, settings, and evidence.

## Source-backed implementation and benchmark map

Paths are relative to the attached repository root. This is a focused source review, not an execution or performance-validation report.

| Topic | Source | Checked detail |
| --- | --- | --- |
| Work-unit algebra | `crates/paulistrings/src/engine/coset.rs` | `Gf2Span` spans bucket deltas and enumerates closed cosets |
| In-place sort-merge | `crates/paulistrings/src/engine/bucketed.rs` | `fill_coset`, source-column swap, identity/rest split, sort, `merge2_into` |
| Gate boundary | `crates/paulistrings/src/engine/mod.rs` | Rebucketing, preparation, application, finalization, trace |
| Shared partition loop | `crates/paulistrings/src/engine/partitioned/driver.rs` | `run_layers` serves NUMA and MPI transports |
| Histogram count control | `crates/paulistrings/src/truncation/builtin.rs`, `engine/partitioned/truncation.rs` | 2048-bin selection and packed length/histogram all-reduce |
| Communication-aware rows | `crates/paulistrings/src/bucket/hash.rs` | `PartitionRows::cut`, z-only block labels |
| Hash probe | `crates/paulistrings/examples/phase_breakdown.rs` | `--partition-rows random|cut`, row-choice statistics, phase timing |
| Measurement fields | `crates/paulistrings-py/src/sum.rs`, `engine/partitioned/trace.rs` | Gate counts and exchange volumes, but no structured gate wall-time vector |
| Internal timing | `crates/paulistrings/src/engine/stats.rs` | Aggregate wall phases distinguished from summed worker busy time |
| Julia comparison | `benchmarks/python/bench_jl_performance.py`, `benchmarks/julia/runner.jl` | Shared task JSON, warm timing, gate-count parity, interleaved pairs |
| Benchmark family | `benchmarks/python/bench_c_deep_trotter.py` | 127-qubit heavy-hex, theta_zz = -pi/2, depths through 20 steps, interior kick angles |
| Cluster launch | `scripts/slurm/mpi-ranks.sbatch`, `scripts/slurm/ab-campaign.sbatch` | Existing MPI test/probe and A/B launch scaffolding |

### Measurement work still needed

The recurring efficiency plot requires structured per-gate elapsed times along full simulations. `PropagationStats` exposes counts, `PartitionTrace` exposes counts and exchange volumes, and `PhaseStats` accumulates phase totals. They do not provide the paired per-gate count/time series by themselves. Debug logging times whole gates but prints rounded milliseconds, unsuitable for precise short-gate measurements.

Add or adapt an opt-in gate-boundary trace in the existing propagation loop, retaining persistent scratch and partition runtime. Avoid repeated convenience calls that recreate pools, scatter, and gather for every gate. Include preparation/rebucketing, exchange, local updates, and finalization within the timer. Keep output/logging outside the region and check instrumentation overhead on a representative run. This is measurement plumbing, not further algorithm research.

MPI records remain rank-local. Combine traces after the run where possible instead of introducing diagnostic barriers/reductions on every gate. Use global counts with a clearly defined rank timing summary. If ranks are not synchronized at each gate, maximum rank-local gate duration is a labeled critical-rank proxy, not exact synchronized global elapsed time. Do not sum that proxy across gates and call it total runtime. An independent complete-run wall timer remains authoritative for the tolerance panel.

`peak_terms` records resident terms between gates, excluding transient fanout and scratch. Collect peak RSS separately. For the multi-node capacity experiment use `result="local"` and reduce expectation values, avoiding full result gathering on rank 0.

The shared-task Julia environment pins PauliPropagation.jl 0.8.2. Existing Slurm scripts are useful launch scaffolding, but the MPI test/probe script is not a complete full-simulation tolerance-sweep driver. Hardware/account details and the actual cluster revision remain to be supplied. No jobs were submitted in this review.
