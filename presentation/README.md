# Presentation: *Wie niet sterk is, moet slim zijn*

A 40-minute talk on `paulistrings-rs`: Pauli propagation, the memory wall, and the GF(2)-linear hash
bucketing that made the engine strong and smart. Everything on a slide traces to a file in this folder or to
a note under `research/notes/`.

| path | what |
|---|---|
| `STORY.md` | the storyline: acts, scenes, claims, evidence |
| `slides/talk.typ`, `slides/lib.typ` | the Typst deck and its hand-rolled theme (no external packages) |
| `slides/talk.pdf` | the compiled deck |
| `notes.md` | speaker notes with minute marks and the source of every number |
| `DECISIONS.md` | log of the choices made while the author was away, for review |
| `bench/` | `presentation-bench`: naive / per-thread-map / mergesort / bucketed variants on the talk's circuit, with agreement tests and the campaign scripts |
| `data/*.jsonl` | measured results with provenance header lines |
| `plots/` | one Python script per figure (`make_all.sh` runs them all) |
| `figures/` | the rendered SVG/PDF figures the deck includes |

Rebuild everything from the repository root:

```bash
cargo test --release --manifest-path presentation/bench/Cargo.toml
presentation/bench/scripts/collect_all.sh          # ~45 min on ccqlin038, quiet box
presentation/bench/scripts/ab_targetcpu.sh
presentation/bench/scripts/perf_bucket_sweep.sh
./.venv/bin/python presentation/plots/collect_term_growth.py
presentation/plots/make_all.sh
typst compile --root . presentation/slides/talk.typ presentation/slides/talk.pdf
```
