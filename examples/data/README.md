# `examples/data/` — checked-in inputs, with provenance

Every file here carries a provenance record, either in its own header (text
formats) or in a `provenance` block (JSON). That is a hard rule of the suite:
every numeric claim is computed by an oracle or loaded from a provenance-tagged
reference file, and no reference value is written down from recollection. If a
fact cannot be traced to a fetched source, it does not get a literal here — the
consuming builder raises instead.

## `heavy_hex_127.edges` — the 127-qubit Eagle coupling map

Generated, never hand-typed, by `generate_heavy_hex.py` in this directory.

| | |
|---|---|
| Source package | `qiskit-ibm-runtime` 0.49.0 |
| Source object | `qiskit_ibm_runtime.fake_provider.FakeSherbrooke().coupling_map` |
| Device | Recorded configuration of IBM's 127-qubit `ibm_sherbrooke`, Eagle r3 |
| Structure | 127 nodes, 144 undirected edges, degrees {1: 2 qubits, 2: 89, 3: 36} |
| Indexing | The device's own qubit numbering, 0..126 |

Regenerate with:

```bash
source .venv/bin/activate
pip install qiskit-ibm-runtime        # dev dep; nothing in examples/common/ imports it
python examples/data/generate_heavy_hex.py            # rewrite
python examples/data/generate_heavy_hex.py --check     # exit 1 if stale
```

`qiskit-ibm-runtime` is a **development** dependency. The generated file is
checked in precisely so that running the showcases needs neither it nor network
access; `examples/common/circuits.py` reads the file and nothing else. The
generator asserts the Eagle r3 structural facts above before writing, so a
package upgrade that changed the topology would fail rather than silently
rewrite the lattice. `--check` compares edge lists, not bytes, so a bumped
package version in the header does not read as staleness.

qiskit 2.x removed its bundled fake backends, so on a current qiskit only
`qiskit_ibm_runtime.fake_provider` supplies this map; the script also tries the
pre-1.0 `qiskit.providers.fake_provider` location.

## `kim2023_observables.json` — published kicked-Ising observables

Supports of the observables measured in the IBM 127-qubit utility experiment.
Loaded by `examples/common/observables.py::kim2023_operator`; the supports are
deliberately **not** literals in that module, so that an unverifiable support
fails at the point of use instead of turning into a citation-shaped constant.

**Primary source.** Y. Kim, A. Eddins, S. Anand, K. X. Wei, E. van den Berg,
S. Rosenblatt, H. Nayfeh, Y. Wu, M. Zaletel, K. Temme, A. Kandala, "Evidence
for the utility of quantum computing before fault tolerance", *Nature* **618**,
500–505 (2023), doi
[10.1038/s41586-023-06096-3](https://doi.org/10.1038/s41586-023-06096-3).

**What was checked, and how.** `nature.com` refuses automated fetch (303 to a
cookie-auth endpoint), so the published-version PDF was read from the UC
eScholarship mirror
(<https://escholarship.org/content/qt6bf3w13h/qt6bf3w13h.pdf>) together with the
Supplementary Information PDF
(<https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-023-06096-3/MediaObjects/41586_2023_6096_MOESM1_ESM.pdf>),
and cross-read against the PubMed Central copy
(<https://pmc.ncbi.nlm.nih.gov/articles/PMC10266970/>). The operator supports do
not appear in prose — they are printed as the **Fig. 3 / Fig. 4 panel titles**,
which survive PDF text extraction. Each observable's `source_detail` field
quotes the panel title it was read from plus the surrounding definition.

Corroborated verbatim by four independent reproduction papers, listed in the
file's `corroborating_sources`; Begušić, Gray & Chan (arXiv:2308.05077), Fig. 3
caption is the single most useful citation because it expands all four panel
labels and so disambiguates the two distinct weight-17 operators.

**Contents.** Four observables, all on the Eagle 0..126 numbering:

| Key | Weight | Figure | Steps | Clifford eigenvalue at θ_h = π/2 |
|---|---|---|---|---|
| `weight_1_z62` | 1 | Fig. 4b | 20 | — |
| `weight_10` | 10 | Fig. 3b | 5 | +1 |
| `weight_17` | 17 | Fig. 3c | 5 | −1 |
| `weight_17_modified` | 17 | Fig. 4a | 5 + final RX layer | −1 |

`weight_17` and `weight_17_modified` share their X support and differ by
swapping the Y and Z sets (they are RX(π/2) conjugates). Mixing them up yields
a silently wrong observable of the same weight — hence two separate keys and a
test that pins them apart.

**Independent verification, not just transcription.** The paper states that the
weight-10 and weight-17 operators are stabilizers of the θ_h = π/2 Clifford
circuit obtained by evolving `Z_13` and `Z_58` for five Trotter steps (SI §VII:
`Z(5, 13)`, `Z(5, 58)`). That makes the supports *derivable*, and
`python/paulistrings/tests/test_examples_circuits.py::test_published_supports_are_reproduced_by_the_stabilizer_relation`
derives them: propagating the seed forward through
`circuits.heavy_hex_kicked_ising(127, steps, theta_h=pi/2)` returns a single
Pauli string whose support and sign match every entry in this file, for all
three stabilizer entries including the Fig. 4a variant.

That single test closes four gaps at once:

1. the transcribed supports are right;
2. the generated heavy-hex edge list is right;
3. the Trotter layer order is right — one step is `X` layer *then* `ZZ` layer
   (SI Eq. 4 writes `U(θ_h) = ∏⟨i,j⟩ R_{Z_iZ_j}(−π/2) · ∏_i R_{X_i}(θ_h)`, whose
   rightmost factor acts first). Under the reverse order the same evolution
   gives weight 6 and 15 instead of 10 and 17, which is why
   `heavy_hex_kicked_ising` defaults to `order="x-then-zz"`;
4. the experiment's `ibm_kyiv` qubit numbering agrees index-for-index with the
   `ibm_sherbrooke` map generated above — no fetched source asserted this, and
   the weight-17 five-step causal cone covers 68 qubits, so an index permutation
   could not survive the check.

**No reference expectation values are shipped here.** Kim et al.'s experimental
values with error bars are in the paper's dataset at
<https://doi.org/10.6084/m9.figshare.22500355>, which refuses automated fetch.
Converged exact benchmarks for all four observables on a θ_h grid of π/32 are
published by Begušić, Gray & Chan at
<https://github.com/tbegusic/arxiv-2308.05077-data> (`exact.csv`; Zenodo
[10.5281/zenodo.10223349](https://doi.org/10.5281/zenodo.10223349)). Neither has
been fetched into this repo. Benchmark C's reference is to be loaded through
`examples/common/oracles.py::load_published_reference` with its own provenance
header, or self-converged with documented convergence evidence — so nothing in
`kim2023_observables.json` can be mistaken for a verified reference number.

Also recorded in the file, from the same sources: θ_J = −π/2 ("such that the ZZ
rotation requires only one CNOT"), the θ_h sweep range, the Trotter depths per
figure, and the SI §VII B causal-cone sizes (≤31 / 37 / 68 qubits for the
weight-1 / weight-10 / weight-17 observables).
