# Agent handoff: scaffold the QuEra talk in Typst

## Mission and scope

Create a portable, editable Typst project for Lukas Devos's technical interview talk:

**Pauli propagation at scale: “Wie niet sterk is, moet slim zijn”**

Use the attached Organic design system and the updated `quera-talk-outline.md`.
The immediate deliverable is six visually verified prototype slides plus a compiling scaffold for the complete 37-conceptual-slide outline.
Support progressive builds, stable slide IDs, speaker notes, and replacement of benchmark placeholders with real figures.
Do the implementation, compile it, inspect the rendered output, and deliver the usable project.
Do not stop at a proposal or directory structure.

A separate agent is generating the scientific benchmark data.
This agent owns slide structure, design integration, and the figure-placement contract.
Do not run cluster benchmarks, change the simulation library, invent measurements, or redesign the scientific story.
The final deck is not required at this stage: the six prototypes must be polished, while the remaining slides must be clearly structured and editable.

## Inputs and authority

Required inputs:

- `organic 2.zip`: `organic/BRIEF.md`, `README.md`, `organic.typ`, `extensions.typ`, `talk.typ`, and `talk.pdf`.
- `quera-talk-outline.md`: the current scientific narrative, conceptual slide order, speaker cues, and QuEra motivation.

Helpful input:

- `quera-benchmark-agent-handoff.md`: data semantics, variant identity, figure stages, and the benchmark agent's output contract.
- `paulistrings-rs` checkout/source when an implementation detail needs checking.

Use the user's instructions and current outline for content, and Organic for presentation design.
The archive's existing `talk.typ`/`talk.pdf` is a 57-page style example, not the authoritative version of the talk.
Do not inherit its numerical claims, historical reconstructions, broad simulation-limit claims, or narrative changes without support in the current outline and evidence.
Preserve the submitted title and abstract; the abstract need not appear on a slide.

Locate the inputs in the receiving workspace or user-provided paths.
Do not assume the temporary paths from another agent's environment exist.
If an input is unavailable, finish the work supported by available inputs and identify the specific missing file.
Do not silently substitute another theme.

Read applicable repository instructions before editing.
Work in a new presentation directory or isolated worktree and preserve existing work.
Do not push, publish, or merge without separate authorization.
Do not change files owned by the benchmark agent.

## What is already known about the Organic archive

The archive contains working theme functions and technical extensions, so begin by adapting those rather than introducing a second presentation framework.
`organic.typ` defines explicit 1920pt × 1080pt pages and common layouts.
`extensions.typ` supplies math, code, figure, chart-plus-copy, and numeric-table layouts.

The sample imports figures from `/presentation/figures/organic/`.
Those figures, the associated plotting scripts, and `sizes.json` are not included in the archive.
The README refers to locally installed fonts, which are also absent from the ZIP.
Remove these environmental assumptions from the new project.
Keep the sample source and PDF as read-only reference material outside the compiled deck.

Three theme fixes are already documented in the attached version:

- The bullet helper is `dot-mark`, avoiding collision with the math multiplication symbol.
- Center alignment in `content-slide` is scoped correctly.
- The cover subtitle width was corrected.

Preserve these fixes.
Do not overwrite the supplied theme with an older generic Organic version.
The sample's rendered threading chart has a clipped bottom axis label; explicit figure bounds and rendering checks are required.

## Design contract

Retain the warm parchment/sand ground, terracotta primary accent, sage secondary accent, rounded elements, and spacious composition.
Use one clear alignment axis per slide: centered for the cover and suitable equation/hero slides; flush-left for code, charts, tables, and explanations where that improves reading.
Do not add dashboard styling, decorative badges, arbitrary gradients, or unnecessary panels.
Round diagram containers where appropriate, while preserving precise mathematical geometry and chart axes.

Use the technical type scale already implemented in the archive:

| Role | Typeface | Stage size |
| --- | --- | --- |
| Display | Caprasimo, weight 400 | 100pt |
| Slide title | Caprasimo, weight 400 | 60pt |
| Body/subtitle | Figtree | 40pt |
| Smaller reading text | Figtree | 34pt |
| Metadata/minimum | Figtree | 24pt minimum |
| Mathematics | STIX Two Math | Match surrounding reading size |
| Code | Source Code Pro or documented suitable fallback | Legible at intended placement |

Retain the 54pt baseline rhythm and the existing 144pt horizontal / 108pt vertical margins as starting values.
Do not shrink content below the minimum to force it onto one page.
Simplify, split a build, or adjust the composition instead.
Use darker accent ramp steps for reading text; pale accents and muted labels must remain legible at presentation size.

Include portable fonts where licensing permits, with their license notices, source, and version/hash information.
Prefer a project-local font directory over system installation instructions.
Pin/document a working Typst version and any package dependencies actually introduced.
Avoid optional packages when the provided theme already handles the need.
Verify that the final build uses the intended fonts rather than silent fallbacks.
If a required font cannot be obtained, document the exact fallback and affected slides and deliver the best compiling version possible.
Do not claim exact visual reproduction under substituted fonts.

## Project organization

Use a layout equivalent to the following, adapting names to an existing repository convention if needed:

| Path | Responsibility |
| --- | --- |
| `main.typ` | Global configuration, theme imports, section order, full/prototype selection |
| `theme/organic.typ` | Preserved theme with minimal documented portability/layout fixes |
| `theme/extensions.typ` | Existing technical layout helpers |
| `components.typ` or `components/` | Small talk-specific helpers for labels, buckets, figures, build selection |
| `sections/00-introduction.typ` | Introduction |
| `sections/01-propagation.typ` | Physics and representation |
| `sections/02-memory.typ` | Baselines, attempted optimizations, memory diagnosis |
| `sections/03-partitioning.typ` | Bit-flip example, hashing, cosets, local kernel |
| `sections/04-results.typ` | Data slots and recurring figure stages |
| `sections/05-conclusion.typ` | Technical close |
| `sections/06-quera.typ` | Motivation and QuantumKitHub experience |
| `figures/` | Imported scientific plots and their manifest |
| `fonts/` | Font files and licenses |
| `notes.md` | Speaker cues keyed to stable conceptual slide IDs |
| `slides.json` or equivalent | Slide IDs, order, build count, status, figure dependencies |
| `figure-contract.json` | Agreed export dimensions, typography, colors, stable stage filenames |
| `scripts/` | Build and checks, with no benchmark execution |
| `build/` | Generated full/prototype PDFs and previews |
| `README.md` | Exact setup/build commands and editing guide |

The project must compile from its own documented root without the original `/presentation/...` tree.
Keep theme settings separate from content.
Keep abstractions small: do not build a general slide framework or duplicate every provided layout.
Use the same source definitions for prototype and full-deck modes to prevent them diverging.

## Stable slide identity and builds

Use semantic IDs such as `intro-title`, `pauli-encoding`, `bit-flip`, `coset-closure`, `performance-bucketed`, and `querakit-architecture`.
Keep conceptual IDs stable when physical PDF page numbers change.
A progressive build adds PDF pages, not new conceptual slide identities.
Map the 37 outline entries to those IDs in `slides.json` and `notes.md`.
An extra build or the optional hash-engineering page is allowed without pretending the conceptual count is unchanged if a genuinely new concept is added.

Maintain fixed object positions across progressive builds.
Reserve the space of unrevealed objects where necessary so content does not jump.
Choose a simple explicit build parameter/helper or equivalent native Typst construction; do not add a framework only for reveal effects.
The six prototype concepts below may produce more than six PDF pages.
Document that distinction in the output manifest.

Carry speaker cues, transitions, timing targets, evidence pointers, and unresolved details in `notes.md` rather than on audience-facing pages.
Do not promise embedded PowerPoint-style notes in the PDF.
Do not add internal task IDs, QA statuses, or agent instructions to completed audience-facing slides.

## Six prototype concepts

### P1. Cover

Use the submitted full title, Lukas Devos, and his current affiliation.
Use a deliberate line break between the subject and the idiom if needed.
Retain the Organic cover's spacious composition.
Do not copy the sample's unverified numerical subtitle.
Keep the precise interview date configurable rather than inventing one.

### P2. Pauli encoding

Show a short Pauli string and its x/z encoding using I=(0,0), X=(1,0), Z=(0,1), Y=(1,1).
Explain label multiplication by XOR with phase tracked separately.
Include the binary symplectic commutation test in legible mathematics.
Use genuine modulo-2 semantics, not ordinary real arithmetic with the modulus omitted from the explanation.
Use this prototype to verify display, body, math, monospace, and numeric-table typography together.

### P3. Bit-flip example

Implement progressive builds using this checked example:

- Sorted input: `[0, 1, 2, 4, 5, 7]`.
- XOR with 2: `[2, 3, 0, 6, 7, 5]`.
- Bit-zero input subsequence `[0, 1, 4, 5]` becomes `[2, 3, 6, 7]`.
- Bit-one input subsequence `[2, 7]` becomes `[0, 5]`.
- Final merged output: `[0, 2, 3, 5, 6, 7]`.

Keep the same integer positions/colors where possible through each explanatory step.
Explain preserved order within the subsequences and the linear merge.
Do not imply that arbitrary GF(2)-hash buckets preserve numeric sort order.

### P4. Closed cosets and worker ownership

Begin with the rotation example h(g)=101 on three-bit bucket IDs.
Use the pairs `(000,101)`, `(001,100)`, `(010,111)`, `(011,110)`.
Reveal labels, connections, closed groups, and ownership in successive builds.
Then state the general construction V=span(h(D)) and work units b+V.
Use a final build if needed to make clear that rotation pairs are a special case of the gate's closed cosets.

Distinguish fine-grained buckets from NUMA/MPI ownership partitions.
A task gathers a closed bucket group and updates its owned slots using reusable scratch.
Do not claim zero temporary memory or global coordination-free histogram truncation.
Mathematical diagrams must be authored as precise editable code, not generated bitmap illustrations.

### P5. Recurring two-panel performance figure

Create the actual fixed placement and figure-loading interface with empty, explicitly labeled data slots.
Left: input-string updates per second versus strings entering a gate.
Right: complete propagation wall time versus decreasing coefficient tolerance.
Use a short annotation area for peak resident count, measured memory, and memory-limit markers once real data exists.
Single-thread cycles/update can appear as a scoped annotation; the recurring axis remains updates/second across threads and ranks.

Do not draw plausible fabricated curves, synthetic speedup numbers, or unlabeled dummy points.
An empty layout can say “Benchmark data pending” and show metric/axis labels without invented numeric ticks.
Use this page to settle chart typography, placement bounds, and captions before the data arrives.

### P6. QuantumKitHub architecture and collaboration

Use a concrete, editable layout linking a user/developer need, Lukas's contribution, and its effect.
The current outline establishes QuantumKitHub as the evidence for architecture, maintenance, collaboration, and mentoring.
Use only supplied, supported examples; leave a concise explicit slot if the particular example still needs selection.
Do not invent adoption counts, collaborators' statements, or measured outcomes.

Preserve the motivation: interfaces, usability, and maintenance matter as much as performance, and Lukas wants sustained attention to those responsibilities in a collaborative scientific software team.
Avoid implying that performance is his only interest or that he has already identified QuEra's internal priorities.

## Full-deck scaffold

Create all remaining conceptual slide entries in the updated outline, grouped into the seven sections above.
For each entry provide a title, appropriate layout selection, stable ID, corresponding speaker notes, and the correct figure/equation/content slot.
Populate straightforward content from the outline where it is already settled.
Keep unsettled measurements and claims explicit rather than completing them from the sample PDF.
Do not convert preparation instructions into audience-facing bullet lists.
The scaffold should be usable for immediate manual editing after this task, without a second structural rewrite.

The talk allocation is 40 minutes of technical content including interruptions, 10–15 minutes of QuEra vision, then additional Q&A.
The working outline targets roughly 33 minutes of prepared technical material and 12 minutes of vision, with hash-engineering timing to revisit in rehearsal.
Keep that metadata in notes; do not squeeze the whole talk into the sample's 57-page structure merely because it exists.

## Figure contract with the benchmark agent

Produce a standalone `figure-contract.json` and a short human-readable explanation suitable for handing to the data agent.
Do not require a specific plotting library: Julia/Makie or the existing Python plotting tools are acceptable if the exported assets satisfy the contract.
The slide agent defines placement and typography; the benchmark agent owns data, computation, and scientific plotting.

Specify:

- Stage size and exact bounding box in stage points for each figure class.
- Full recurring two-panel figure width/height, including labels, legends, and annotations.
- Font families, sizes, foreground/background, grid, line widths, and marker choices.
- Fixed semantic variant IDs with consistent colors, line styles, and legend order.
- Stage filenames for baseline, kernel, threading attempt, memory annotation, bucketed single thread, bucketed multithread, and distributed extension.
- A separate filename/layout for hash-communication and accuracy figures.
- Whether text remains embedded text or outlines in each exported format, and how font availability is handled.
- Caption ownership and reserved space so text does not appear twice.

Determine dimensions by rendering the prototype, not guessing how much space a figure “probably” gets.
At the chosen 1920pt stage, preserve font size at actual placement.
Do not export a tightly cropped SVG and scale it into an unrelated width, unintentionally shrinking or enlarging every label.
Use exact authored dimensions with suitable margins and place at that size.
Check actual export units and the resulting Typst placement instead of assuming all exporters interpret points identically.

Keep a figure manifest listing path, intended dimensions, stage ID, and provenance/data status.
Draft mode should compile with an explicit same-size placeholder when data is missing.
Provide a strict figure check/release mode that fails on missing required real assets or unresolved evidence placeholders; a draft PDF is still required now.
Do not present that release-mode failure as a scaffold failure while benchmark data is intentionally pending.

Two accent colors are not enough to identify all benchmark variants by color alone.
Use consistent warm neutral/sage/terracotta assignments plus line styles and markers.
Highlight a newly revealed variant by emphasis while preserving its identity across stages.
Do not recolor a variant simply because it became the current focus.
Check legibility in grayscale and at normal screen-sharing size.

## Cost-aware subagent plan

Use a main orchestrator to own the design/content contract, interfaces, integration, and final visual inspection.
Use moderate effort by default.
Delegate routine file/setup work to cheap models and reserve stronger reasoning for the mathematical diagram review or a concrete Typst/layout failure.
Choose models actually available in the receiving environment rather than assuming names or prices.
Do not use a high-cost model to copy notes, resize previews, or poll compilation.

| Task | Scope and output | Dependency | Tier/effort | Write ownership |
| --- | --- | --- | --- | --- |
| S01 | Audit archive, dependencies/fonts, and existing project integration; concise portability report | None | Cheap, low | Report only |
| S02 | Map outline to stable IDs, sections, note entries, and placeholder/evidence status | None | Cheap, low | Manifest and notes drafts |
| S03 | Implement portable theme/build/font setup and draft/strict behavior | Main contract, S01 | Standard coding, medium | Theme/setup/build files |
| S04 | Build the encoding, bit-flip, and closed-coset prototypes with checked math | Shared helper interface | Standard coding, medium | Assigned prototype/diagram files |
| S05 | Build recurring-figure prototype and export contract | Theme dimensions, S03 | Standard coding, low–medium | Figure layout/contract files |
| S06 | Populate section scaffold, cover, and QuantumKitHub prototype | S02, shared layouts | Cheap coding, low–medium | Section files excluding S04/S05 ownership |
| S07 | Review rendered typography, clipping, alignment, and mathematical builds | Integrated prototype/full PDFs | Strong visual reviewer, medium | Findings only |

S01 and S02 can run in parallel.
Freeze helper signatures and file ownership before S04–S06 run in parallel.
The main agent owns shared `main.typ` and component integration; avoid several workers editing those files simultaneously.
Use isolated worktrees when useful, or explicitly disjoint write sets.
Do not spawn recursively by default or duplicate the same inspection with several models.

Send each worker only a bounded dispatch packet:

```text
Task ID and objective:
Relevant design/content contract:
Input paths and sections to read:
Allowed writes:
Shared helper signatures:
Required outputs:
Acceptance checks:
Escalate if:
Return report path:
```

Workers return at most about 250 words plus file pointers: status, changes, checks, material limitations, and integration needs.
Store long logs in files.
One targeted repair is appropriate before escalating a recurring failure.
Escalate immediately for uncertain scientific content or a shared-interface conflict rather than guessing.
Do not pass the full conversation to every subagent if fresh bounded contexts are supported.
All agents still follow applicable local instructions and permissions.

## Build and verification

Provide documented commands to compile prototype and full-scaffold PDFs, and an optional watch command for editing.
Use an explicit project root and local font path.
Ensure every dependency is declared and avoid absolute references into the original user's home directory or this handoff author's workspace.
Compile from a clean output directory to expose hidden dependencies.

Inspect all rendered prototype/build pages and representative scaffold pages, plus an overview/contact sheet of the full draft.
Use a PDF renderer available in the environment; do not rely on text extraction alone.
Verify:

- Intended fonts and math glyphs render correctly without missing characters.
- No title, chart tick, legend, caption, code line, or folio clips or overlaps.
- Progressive builds retain object positions and the arithmetic/pairing is correct.
- Figure placement preserves minimum text size and the supplied dimensions.
- Section order and conceptual IDs match the outline.
- Physical page count matches the declared build count, with no accidental overflow pages.
- Placeholder/evidence status is explicit in draft assets and strict mode detects missing required data.
- All 37 conceptual entries have corresponding editable source and notes.

Fix the known class of chart-axis clipping seen in the sample by correcting authored bounds and layout space, not by shrinking the plot until text is illegible.
Use compilation and targeted visual checks rather than an elaborate testing framework for reversible layout work.
Record validation privately or in a build report, not on audience slides.

## Final deliverables and completion

Deliver:

1. Portable Typst project, including required fonts/licenses or explicitly documented unresolved font dependencies.
2. Rendered prototype PDF and complete scaffold draft PDF, with accurate conceptual/build counts.
3. Stable slide manifest and editable speaker notes.
4. Figure contract and placeholder/real-asset manifest for the benchmark agent.
5. Build instructions and a concise list of unresolved content/data slots.
6. A short change summary identifying any theme deviations and actual limitations.

If working in the user's Git repository, leave the project in the agreed path with a clean, reviewable change set.
If delivering through a file-based environment, package the project as a ZIP and provide the PDFs separately.
Do not distribute the old sample PDF as though it were the new scaffold.

Success means Lukas can immediately edit the talk, compile it reproducibly, and replace benchmark placeholders without changing the surrounding slide layouts.
The six prototype concepts should establish the visual vocabulary for the entire deck.
Finish this scaffolding work without waiting for the benchmark campaign to complete.
