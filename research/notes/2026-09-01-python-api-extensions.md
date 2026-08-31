# Python API extensions for the examples & benchmarks suite

Forward design note for the capability register in
`research/plans/2026-08-31-examples-benchmarks-suite.md` §3 (items A1–A8). Each section fixes the
Python signature, validation, semantics, the layer it touches, the TDD test plan, and what an
implementing agent must **not** do. Signatures were verified implementable against the sources
cited inline. Nothing in this note is implemented yet; implementation is one logical commit per
item (Wave 2).

Shared constraints, all items:

- TDD per CLAUDE.md §Testing & TDD policy: red test first, hand-computed expected values, W ∈ {1, 2}
  parameterization wherever the core is touched.
- The Hermitian-Y convention everywhere: a coefficient multiplies the literal Pauli string,
  `Y ↔ (x=1, z=1)`, no phase factor (CLAUDE.md §Known gaps).
- No change may alter `propagate`'s existing signature or default behavior; new capability = new
  method/function.
- CI's python job installs numpy only: every new CI-visible test `importorskip`s optional deps.

---

## A1 — `gates.pauli_rotation` (multi-qubit Pauli-string rotation)

**Status:** bindings-only. The core `PauliRotation::new(gen, theta)` already accepts any generator
weight (`crates/paulistrings/src/channel/rotation.rs:68`), with the `Prepared::Rotation` override
above weight 2 (`rotation.rs:177-190`). Nothing in the core changes.

**Signatures**

```python
paulistrings.gates.pauli_rotation(pauli: str, qubits: Sequence[int], theta: float) -> Channel
paulistrings.Circuit.pauli_rotation(self, pauli, qubits, theta) -> None   # convenience, same args
```

`U = exp(-i·θ·P/2)` where `P = pauli[0] on qubits[0] ⊗ pauli[1] on qubits[1] ⊗ …`. The compact
form (`"ZZ", [3, 9]`) rather than a full-length `IXYZ` string: suite circuits address 127-qubit
lattices where full-length strings are unreadable and error-prone. Examples:
`pauli_rotation("ZZ", [i, j], -pi/2)` is the kicked-Ising Clifford-point bond
(the handoff's `exp(iπ/4·Z_iZ_j)`); `pauli_rotation("X", [q], theta)` ≡ `rx(theta, q)`.

Argument order diverges deliberately from `rz(theta, qubit)`'s angle-first order: `(what, where,
how much)` reads correctly for a multi-qubit generator, and the two-string-arguments-first shape
makes accidental transposition a `TypeError` rather than a silent angle/qubit swap. `rz/rx/ry`
keep their existing signatures untouched (they remain the single-qubit spellings; internally they
already build `PauliRotation`, `channel_spec.rs:102-113`).

**Validation & errors** (all `ValueError` unless noted)

- `len(pauli) == len(qubits)`, both non-empty.
- `pauli` characters ∈ `{X, Y, Z}` only. `I` is rejected — identity positions are expressed by
  omission, and allowing them would create two spellings of the same channel.
- `qubits` pairwise distinct (`TypeError` for non-int entries via pyo3 extraction).
- Bounds: qubit indices are validated against `Circuit.num_qubits` **at append time** (see the
  bounds-check block below), not in the factory — a factory-made `Channel` is width- and
  circuit-agnostic by design (`channel_spec.rs:1-7`).

**Adjoint semantics:** none needed beyond the core's — `direction="heisenberg"` dispatches
`apply_adjoint`, which is `theta → -theta` (`rotation.rs:97-99`).

**ChannelSpec change:** new variant

```rust
PauliRotationN { theta: f64, paulis: Vec<(u32, Axis)> }   // Axis ∈ {X, Y, Z}
```

`Vec` breaks `#[derive(Copy)]` on `ChannelSpec` (`channel_spec.rs:20`). **Decision: drop `Copy`,
keep `Clone`.** The "small and `Copy`" rationale in the spec's comment predates `Unitary2Q`, which
already made the enum 256+ bytes; nothing requires `Copy` (`push_into` switches from `match *self`
to `match self` with per-field copies/borrows). `push_into` materializes the generator by setting
x/z bits directly per `(qubit, axis)` — sites are distinct, so no `mul_assign` phase can arise.

**Bounds checks (part of this item):** `CircuitImpl::push_spec` → `PyResult<()>`; `ChannelSpec`
gains `fn max_qubit(&self) -> u32` (and the 2q constructors `cnot`/`cz`/`swap` gain a
distinctness check, closing the same class of hole `unitary_2q` already closes at
`gates.rs:127-129`). `Circuit.append` and every convenience method raise `ValueError` when
`max_qubit >= num_qubits`. This turns today's silent misbehaviour/panic into a clean error.

**Test plan (red first)**

- `pauli_rotation("Z", [0], θ)` ≡ `rz(θ, 0)` on X-observable: two-term output `cos θ·X − sin θ·Y`
  hand-computed (matches `rotation.rs` slice tests).
- `pauli_rotation("ZZ", [0, 1], -π/2)` on `XI`: kicked-Ising bond conjugation, hand-computed
  (`XI → YZ` up to sign; pin the sign by hand from `i·sinθ·Q·P` with `i^k` from `mul_assign`).
- Weight-3 generator (`"ZZZ"`, qubits crossing the word boundary at 63/64/65, W=2) — exercises the
  `Prepared::Rotation` path from Python for the first time; Heisenberg round-trip
  (`forward` then `heisenberg` restores the input, matching `test_circuit.py`'s rz round-trip).
- Error cases: length mismatch, bad character (incl. `"I"`), duplicate qubits, out-of-range qubit
  at append (both `append(gates.pauli_rotation(...))` and `Circuit.pauli_rotation`).
- Existing-surface regression: out-of-range `h`/`cnot` now `ValueError` (new behavior, new tests);
  `cnot(q, q)` now `ValueError`.

**Implementer must NOT:** change existing `ChannelSpec` variants' shapes or their `Debug` output
(`Channel(...)` reprs are user-visible); reintroduce a full-length-string constructor; touch
`rotation.rs`.

---

## A2 — propagation term statistics (`propagate_with_stats`)

**Status:** core + bindings. Term counts must become programmatic: the harness's parity check and
every Part A benchmark consume per-layer and peak counts, and today they exist only as DEBUG log
text (`engine/mod.rs:221-231`) or behind the compile-gated `phase-timing` feature.

**Decision — split term counts from phase timings.** Phase *timings* stay exactly where they are
(`phase-timing` feature, `engine/stats.rs`). Term *counts* become an always-available, opt-in
trace: `propagate_with_scratch` already reads `sum.len()` unconditionally before and after every
layer (`engine/mod.rs:170` `terms_before`, `:213` post-finalize) — recording them is two `usize`
pushes per layer on the calling thread, outside the Rayon region and outside the merge/coset hot
path.

**Core surface**

```rust
/// Always-compiled, opt-in per-layer term-count trace. All fields populated by
/// propagate_with_scratch when enabled; never touched inside apply_layer_bucketed.
pub struct TermTrace {
    pub terms_in: Vec<usize>,   // per layer, before the layer
    pub terms_out: Vec<usize>,  // per layer, after finalize_layer
}
```

`LayerScratch` gains a private `term_trace: Option<TermTrace>` with `enable_term_trace(&mut self)`
and `take_term_trace(&mut self) -> Option<TermTrace>`. `propagate_with_scratch` records into it
per layer iff `Some` — one branch per layer when disabled. No public signature changes; `propagate`
(the scratch-less front door) never enables it.

**Semantics of "peak":** peak = max over `terms_in[0]` and all `terms_out[k]`. This is the peak
*resident* term count between layers. The transient in-layer expansion (post-fanout, pre-merge) is
**deliberately not captured** — observing it would require instrumenting the coset loop, which is
exactly what this design forbids. Document this in the docstring; peak RSS stays the harness's
memory metric (`/proc/self/status`, as `phase_breakdown.rs` does).

**Python surface**

```python
evolved, stats = pauli_sum.propagate_with_stats(circuit, policy=None, direction=None)
stats.layers        # int
stats.terms_in      # list[int], len == layers
stats.terms_out     # list[int], len == layers
stats.peak_terms    # int (see semantics above)
stats.final_terms   # int == terms_out[-1] (== len(evolved)); 0-layer circuit: len(input)
```

`PropagationStats` is a plain `#[pyclass]` (getters only, no methods). `propagate_with_stats`
mirrors `propagate`'s argument handling verbatim (same direction/policy/num_qubits validation,
GIL released) but drives `propagate_with_scratch` with a locally-created, trace-enabled scratch.
`propagate` itself is untouched.

**Performance gate (blocking for merge):** the one risk is code-layout perturbation (the
`engine/merge.rs` `#[inline]` set is A/B-verified load-bearing at ±6–34%, CLAUDE.md §Performance
discipline). Gate:

```
scripts/ab-compare.sh --a main --b . --pairs 3 --probe '<the standard trotter probe args from benchmarks/PROFILING.md §Interleaved A/B>'
```

Acceptance per PROFILING.md: **direction consistency across every pair** — pairs disagreeing in
sign = no consistent change = merge OK; a consistent regression blocks the merge and the change is
reworked (e.g. `#[cold]` the trace-recording path) rather than accepted.

**Test plan**

- Rust: trace disabled by default (`take_term_trace()` is `None` after a normal propagate); enabled
  trace on a hand-built 3-layer circuit has `terms_in/out` lengths 3 and hand-computed counts
  (e.g. T-gate on X: 1 → 2 terms; a `coeff` policy that prunes back down); W ∈ {1, 2}.
- Python: `propagate_with_stats` returns the same evolved sum as `propagate` (same
  coefficients via the `_as_dict` helper pattern from `test_general_unitary.py`); stats fields
  consistent (`final_terms == len(evolved)`, `peak_terms >= final_terms`); empty circuit → zero
  layers, `peak_terms == len(input)`.

**Implementer must NOT:** touch `apply_layer_bucketed`, the coset loop, or `engine/merge.rs` in
any way; add any per-term work; make the trace always-on (opt-in only); move `PhaseStats` out from
behind `phase-timing`; skip the ab-compare gate even if criterion looks flat.

---

## A3 — `PauliSum.from_arrays` + `.npz` save/load

**Status:** small binding + pure Python. The missing inverse of
`x_array`/`z_array`/`coefficients_array` (`sum.rs:249-275`); B5 needs evolved-observable
round-trips through files.

**Signatures**

```python
PauliSum.from_arrays(x, z, coefficients, num_qubits: int) -> PauliSum   # classmethod
# x, z: uint64 arrays of shape (n_terms, w); coefficients: complex128 (n_terms,)

paulistrings.io.save(path, pauli_sum) -> None      # .npz, pure Python
paulistrings.io.load(path) -> PauliSum
```

**Validation & errors** (`ValueError`)

- `x.shape == z.shape == (n, w)` with `1 <= w <= band_width(num_qubits)`; `coefficients.shape == (n,)`.
  Arrays narrower than the band width are zero-padded on ingest, so a sum saved at one band loads
  into the same band; the exported width is always the band width, so round-trips are exact.
- Any set bit at qubit index `>= num_qubits` → `ValueError` (mask check per word; this is the same
  class of hole the A1 bounds checks close).
- Dtypes: accept exactly `uint64` for x/z; `complex128` or real-float castable for coefficients.

**Semantics:** rows are symplectic keys in the Hermitian convention (no phases — `Phase::ONE` on
every `add_term`). Ingest routes through `BuildAccumulator` (`accumulator.rs`), so duplicate keys
sum and exact zeros drop — i.e. `from_arrays` canonicalizes exactly like `from_strings`, and
`from_arrays(s.x_array(), s.z_array(), s.coefficients_array(), s.num_qubits)` reproduces `s`
(coefficients bit-equal; ordering canonical).

**`io.py` format** (`np.savez_compressed`): keys `format = "paulistrings-npz-v1"`, `num_qubits`
(int), `x`, `z`, `coefficients`. `load` hard-errors on a missing/unknown `format` key. No pickle
anywhere.

**Test plan:** hand-built 2-term sum from arrays matches `from_strings` twin (W ∈ {1,2}, incl. a
qubit-64 row); duplicate rows sum; zero row dropped; out-of-range bit rejected; shape/dtype
rejections; save→load round-trip equality via `_as_dict`; `io` tests importorskip nothing (numpy
is a hard dep) but live beside the API tests in `python/paulistrings/tests/`.

**Implementer must NOT:** add serde or any Rust-side file I/O; accept phases or non-Hermitian
keys; bypass `BuildAccumulator`.

---

## A4 — non-uniform product states for `expectation`

**Status:** core (moderate) + bindings. Today `ProductState` is a 3-variant uniform enum
(`pauli_sum.rs:71-79`) and the masked scan lives in `bucket/sum.rs::expectation_product_state`.

**Python surface** — extend the existing `state=` string, no new method:

```python
sum.expectation(state="x+")        # existing uniform spellings, unchanged
sum.expectation(state="0100")      # per-qubit labels, len == num_qubits
```

Per-qubit alphabet (qiskit `Statevector.from_label` convention): `0`/`1` (Z±), `+`/`-` (X±),
`r`/`l` (Y±). Character `i` addresses qubit `i` (same order as `from_strings`). Dispatch rule: the
three uniform names are matched first (case-insensitive, as today); otherwise the string must have
length `num_qubits` and draw from the 6-char alphabet, else `ValueError` naming both accepted
forms.

**Core surface**

```rust
/// A product of single-qubit stabilizer states, one per qubit: axis A_q ∈ {X,Y,Z}
/// with sign s_q. Stored as per-word masks in the symplectic layout.
pub struct ProductBasis<const W: usize> {
    pub ax_x: [u64; W],  // x-bit of the axis Pauli per qubit
    pub ax_z: [u64; W],  // z-bit of the axis Pauli per qubit
    pub neg:  [u64; W],  // sign bit: 1 = minus eigenstate
}
impl PauliSum<W> { pub fn expectation_product_basis(&self, basis: &ProductBasis<W>) -> Complex64; }
```

**Semantics** (hand-derivable, encode in the doc comment): for a term `P` with key `(x, z)`,
`⟨P⟩ = Π_q ⟨P_q⟩` where `⟨P_q⟩ = s_q` if `P_q = A_q`, `1` if `P_q = I`, else `0`. Bit-parallel per
word: the term contributes iff `x == (x|z) & ax_x` **and** `z == (x|z) & ax_z` (every non-identity
site's Pauli equals the local axis exactly — note a subset match must *not* pass: an `X` term on a
Y-axis qubit is zero); its sign is `(-1)^popcount((x|z) & neg)`. The existing uniform states are
the special cases `ax_x/ax_z` all-ones-pattern with `neg = 0` (e.g. ZPlus: `ax_x = 0`,
`ax_z = !0` → condition reduces to `x == 0`, matching `bucket/sum.rs` today) — reimplement the
three `ProductState` variants **on top of** `ProductBasis` so there is exactly one scan, and keep
the per-bucket parallel-reduction structure of the existing scan unchanged.

Bits at qubit index `>= num_qubits` in the masks are zero by construction at the binding (the
label string has exactly `num_qubits` characters).

**Test plan** (hand-computed, W ∈ {1,2}):

- `⟨0|Z|0⟩ = 1`, `⟨1|Z|1⟩ = -1`, `⟨0|X|0⟩ = 0`, `⟨+|X|+⟩ = 1`, `⟨-|X|-⟩ = -1`, `⟨r|Y|r⟩ = 1`,
  `⟨l|Y|l⟩ = -1` (single qubit, each axis).
- Mixed multi-qubit: `state="0+r"` against `ZXY → +1`, `ZXX → 0`, `ZIY → +1`, sign product
  `state="1-l"` against `ZXY → (-1)^3 = -1`.
- Subset-match trap: `X` term on a `r` (Y+) qubit → 0 (this is the test that kills the
  `x & ~ax_x == 0` wrong implementation).
- Uniform spellings still exact on the existing `test_expectation.py` matrix (regression: the
  reimplementation on ProductBasis must not move any existing value).
- Word boundary: qubit 64 label differs from qubit 0 (W=2).
- Linearity with complex coefficients (mirrors existing tests).

**Implementer must NOT:** add a state *object* to the Python surface (strings only — a
`StabilizerState` type is A8-ii's future work, and two state vocabularies would collide); expand
anything into 2^n terms; change `overlap` or `identity_coefficient`.

---

## A5 — importers (`paulistrings/interop.py`) and the frozen task-JSON schema

**Status:** pure Python, shipped API (not example code — every Part 0 module and the Julia driver
consume it). Lazy imports; `stim`/`qiskit` are needed only by their respective functions.

**Signatures**

```python
paulistrings.interop.circuit_from_stim(src: stim.Circuit | str | os.PathLike)
    -> tuple[Circuit, PauliSum | None]        # observable from OBSERVABLE_INCLUDE Pauli targets, if present
paulistrings.interop.circuit_from_qiskit(qc: qiskit.QuantumCircuit) -> Circuit
paulistrings.interop.load_task(path) -> Task  # dataclass, below
paulistrings.interop.circuit_from_json(obj: dict) -> Circuit   # the "circuit" object only
```

`Task` is a frozen dataclass: `n_qubits: int`, `circuit: Circuit`, `observable: PauliSum | None`,
`truncation: Truncation | None` (built from the two knobs), `direction: str`,
`threads: int`, `state: str | None`, plus `raw: dict` for provenance echoing.

### Frozen task-JSON schema — version 1

Both `interop.load_task` and `benchmarks/julia/runner.jl` implement this **verbatim**; it is the
only interchange format between the two engines. Unknown top-level keys, unknown gate names, and
missing required keys are hard errors on both sides (no silent tolerance — the schema is
versioned instead).

```jsonc
{
  "version": 1,                          // required, must be 1
  "n_qubits": 127,                       // required
  "circuit": {"gates": [ /* gate objects, in application order */ ]},
       // OR {"stim_file": "relative/path.stim"}  (Python side only; the Julia
       //     runner hard-errors on stim_file — convert to an inline gate list
       //     first. jl cannot parse Stim.)
  "observable": {"III...Z...": 1.0},     // full-length keys, Hermitian-Y; value = number or [re, im].
                                         // Omitted iff the stim file carries OBSERVABLE_INCLUDE.
  "truncation": {"max_weight": 6, "min_abs_coeff": 6.1e-5},   // both keys optional; omitted = no policy
  "run": {"direction": "heisenberg",     // required — no defaulting, ever (README-default trap)
          "threads": 1,                  // optional, default 1
          "state": "z+"}                 // optional; uniform name or per-qubit label string (A4)
}
```

Gate object vocabulary (name → required fields; `qubits` is always a list of ints):

| name | fields |
|---|---|
| `h` `s` `x` `y` `z` | `qubits` (len 1) |
| `cnot` | `qubits` = `[control, target]` |
| `cz` `swap` | `qubits` (len 2) |
| `rz` `rx` `ry` | `qubits` (len 1), `theta` |
| `pauli_rotation` | `pauli` (str over XYZ), `qubits` (len == len(pauli)), `theta` |
| `depolarize` `dephase` | `qubits` (len 1), `p` |
| `amplitude_damping` | `qubits` (len 1), `gamma` |
| `pauli_channel` | `qubits` (len 1), `px`, `py`, `pz` (A6) |
| `depolarize2` | `qubits` (len 2), `p` (A6) |
| `unitary_1q` | `qubits` (len 1), `matrix` (2×2 nested lists of `[re, im]`) |
| `unitary_2q` | `qubits` (len 2), `matrix` (4×4, acts on `|q0 q1⟩`) |

One gate object = one channel push (the one-gate-per-channel parity rule is structural in this
format).

### `circuit_from_stim` mapping

Supported: `H`, `S`, `X`, `Y`, `Z`, `CX`/`CNOT`, `CZ`, `SWAP`, `DEPOLARIZE1(p)` → `depolarize(p)`
(stim's uniform-Pauli p/3 semantics match `Depolarizing`'s `1 − 4p/3` rescale exactly),
`X_ERROR/Y_ERROR/Z_ERROR(p)` → `pauli_channel` with the single matching component (Z_ERROR ≡
`dephase(p)`; keep the pauli_channel spelling for uniformity), `DEPOLARIZE2(p)` → `depolarize2(p)`,
`REPEAT` blocks (expanded), `I` (skipped), annotations `TICK` / `QUBIT_COORDS` / `SHIFT_COORDS`
(ignored, they are not operations). `OBSERVABLE_INCLUDE` with **Pauli targets** builds the returned
observable (product of the targets, coefficient 1.0; multiple indices → multiple terms).

Hard errors (`ValueError` naming the instruction and its line): measurements/reset of any kind
(`M`, `MR`, `R`, `MPP`, …), `DETECTOR`, `OBSERVABLE_INCLUDE` with measurement-record targets,
`CORRELATED_ERROR`/`ELSE_CORRELATOR`, `PAULI_CHANNEL_1/2` (until mapped — note as follow-up),
sweep/combined targets, and any instruction not listed above. **Never skip silently** (adapted
plan D-rule). Implementation detail to verify at implementation time: the installed stim's
`OBSERVABLE_INCLUDE`-with-Pauli-targets support (stim ≥ 1.10); if absent, the checked-in files
carry the observable in a sidecar task JSON via `"stim_file"` — the schema already supports this,
so no format change either way.

### `circuit_from_qiskit` mapping

Named mapping where exact: `h s x y z cx cz swap rz rx ry` plus `rzz/rxx/ryy → pauli_rotation`
and `sdg/t/tdg → unitary_1q` (matrix). Any other 1q/2q gate with `to_matrix()` → checked
`unitary_1q/2q` fallback (the binding's unitarity check is the safety net, `gates.rs:32-53`).
`barrier` ignored; measurements, classical bits/conditions, >2q gates → `ValueError`. Qubit index =
`qc.find_bit(q).index`.

**Test plan** (all importorskip'd): stim round-trip conjugation checks — a stim Clifford circuit
imported and Heisenberg-propagated agrees with `stim.PauliString.after` / tableau conjugation on
hand-picked observables (this doubles as the Hermitian-Y-vs-stim phase-convention test, citing
`research/notes/2026-08-31-python-test-triage.md`); DEPOLARIZE1 factor 1 − 4p/3 pinned by hand at
p=0.3; unsupported-instruction errors name the instruction; REPEAT expansion count; qiskit rzz
equals `pauli_rotation("ZZ", …)` term-for-term; qiskit T-gate fallback matches
`test_general_unitary.py`'s T conjugation; task-JSON: full round-trip build → run → expectation on
a 2-qubit hand-computed case; every hard-error branch.

**Implementer must NOT:** skip or warn-and-drop any unsupported instruction; default `direction`
when absent from a task file; write a Rust-side parser; make `stim`/`qiskit` hard imports at
module scope.

---

## A6 — Pauli channel noise

**Status:** small core + binding. B2's "realistic arbitrary local noise" needs independent
`(px, py, pz)`; hardware models also want 2q depolarizing after two-qubit gates. **Decision: ship
both** `pauli_channel` (1q) and `depolarize2` (2q) — the second is ~30 lines against the same
pattern and B2 uses it immediately.

**Signatures** (param-first, matching `noise.depolarize(p, qubit)`):

```python
paulistrings.noise.pauli_channel(px: float, py: float, pz: float, qubit: int) -> Channel
paulistrings.noise.depolarize2(p: float, q0: int, q1: int) -> Channel
Circuit.pauli_channel(px, py, pz, qubits: list[int])   # broadcast, one channel per qubit
Circuit.depolarize2(p, pairs: list[tuple[int, int]])   # broadcast, one channel per pair
```

**Semantics** (Heisenberg duals, key-preserving, fanout 1, self-adjoint):

- `pauli_channel`: `E(ρ) = (1−px−py−pz)ρ + px XρX + py YρY + pz ZρZ`. Dual scales:
  `I → 1`, `X → 1 − 2(py + pz)`, `Y → 1 − 2(px + pz)`, `Z → 1 − 2(px + py)` (each Pauli
  anticommutes with the other two). Core: `PauliChannel { support: [u32; 1], px, py, pz }` in
  `channel/noise.rs`, a per-index-scale sibling of `rescale_on_support` (`noise.rs:13-28`; the
  packed local index there is `I=0, X=1, Z=2, Y=3` — mind the order). Consistency identities to
  encode as tests: `pauli_channel(p/3, p/3, p/3) ≡ depolarize(p)`;
  `pauli_channel(0, 0, p) ≡ dephase(p)`.
- `depolarize2`: uniform 2q depolarizing, prob `p` spread over the 15 non-identity 2q Paulis.
  Dual: identity-on-the-pair → 1, every other Pauli on the pair → `1 − 16p/15`. Core:
  `Depolarizing2Q { support: [u32; 2], p }`, support weight 2 fits `MAX_LOCAL_SUPPORT`
  (`derive_local` handles preparation; fanout 1).

**Validation:** `px, py, pz >= 0`, `px + py + pz <= 1` (`pauli_channel`); `0 <= p <= 1` and
`q0 != q1` (`depolarize2`); bounds at append per A1.

**Test plan** (hand-computed, W ∈ {1,2}; Rust unit tests beside the code + Python boundary tests):
the four scale factors at `(px, py, pz) = (0.1, 0.2, 0.3)` pinned by hand per input Pauli; the two
consistency identities above at a generic p; `depolarize2` at `p = 15/16` annihilates `XZ` exactly
and leaves `II` alone; qubit-64 word-boundary case; adjoint = same (self-adjoint default,
propagate round-trip under `heisenberg` on a commuting/scaled case).

**Implementer must NOT:** implement via `GeneralUnitary` PTMs (these are diagonal rescalings —
keep them fanout-1 key-preserving like their siblings, or the engine loses the rescale fast path
`engine/bucketed.rs::rescale_in_place`); allow support overlap in `depolarize2`.

---

## A7 — harness truncation aliases + thread pinning

**Status:** pure Python, lives in `examples/common/harness.py` (not the shipped API — the API
keeps one construction style, policy objects; adapted plan D-decision).

**Aliases:**

```python
def make_policy(max_weight: int | None = None, min_abs_coeff: float | None = None) -> Truncation | None
```

`weight(w) & coeff(eps)` when both, the single one when one, `None` when neither. Document the
inclusive-drop boundary (`|c| <= eps` is dropped, `truncation/builtin.rs:22`) at this choke point;
the jl-boundary probe (adapted plan §5) lives next to it. `topn` is deliberately absent (banned in
comparative runs).

**Thread pinning — empirical facts (measured on ccqlin004-class host, 2026-08-31):** the Rayon
global pool spawns at **`import paulistrings` time, not lazily at first propagate** — with
`RAYON_NUM_THREADS` unset the process shows 33 threads (32 workers + main) immediately after
import; with `RAYON_NUM_THREADS=1` it shows exactly 2 (main + 1 worker), and a 16.7M-term
propagate stays at 2. Consequences:

- The env var **must be exported before the interpreter imports `paulistrings`**. Setting it in
  Python works only if it happens before the first `import paulistrings` anywhere in the process.
- Verification is trivial and robust: read `Threads:` from `/proc/self/status`.

```python
def assert_single_threaded() -> None
    # raises RuntimeError unless os.environ["RAYON_NUM_THREADS"] == "1" AND
    # /proc/self/status Threads <= SINGLE_THREAD_MAX (empirically 2; keep a
    # small named constant with a comment, not a magic number)
```

Every timed entry point in the harness calls this first. **No `set_num_threads` binding is
needed** — the env route is verified working; a binding could not help anyway since the pool
already exists by the time user code runs.

**Test plan:** `make_policy` truth table (4 cases) against hand-built policy behavior on a probe
propagate (the `z(0)` probe-layer idiom from `test_truncation.py`); `assert_single_threaded`
passes under `RAYON_NUM_THREADS=1` and raises without it (subprocess test, since the pool state is
process-global). These live with the examples' own test collection, not CI.

**Implementer must NOT:** add truncation kwargs to `propagate` itself; spawn or resize Rayon pools
from Rust; rely on setting the env var after import.

---

## A8 — deferred designs (docs only, no code)

### A8-i — symbolic / surrogate coefficients (blocks F, B3, B4)

Why blocked today: `Complex64` is load-bearing in the `Channel::apply` and
`TruncationPolicy::keep_term` trait signatures, `OutputBuffer`, the SoA coefficient column, the
merge kernels' exact-zero drop, `TopN`'s magnitude ties, `Phase::apply`, and the
`Pod`/`bytemuck`/GPU-readiness pillar (ARCHITECTURE.md §Data-Model, §GPU-Readiness). A generic
coefficient type parameter would ripple through every trait and double the monomorphization
surface.

Sketch that preserves the Pod story (the direction to explore, not a commitment): coefficients in
surrogate mode are **`u32 handles into an append-only evaluation tape**, not expression objects.
The channel path only ever does three things to a coefficient — scale by a per-layer constant
(`cos θ`, noise factors), scale by `i^k·sin θ`, and sum two coefficients in the merge — so the
tape needs three node kinds, and the handle stays `Pod`. Costs to quantify before committing:
tape growth is proportional to *path count* (not term count — every merge allocates a SUM node);
coefficient-threshold truncation has no meaning on unevaluated handles (jl's surrogate mode
truncates by weight/frequency for exactly this reason — the same restriction would apply, or a
reference-point evaluation pass per layer); gradients = reverse-mode sweep over the tape.
Realistic scope: a parallel `propagate_surrogate` entry point generic over a small `CoeffOps`
trait implemented by `Complex64` and `TapeHandle`, leaving the shipping path untouched. This is a
core redesign with its own plan/measurement cycle — out of scope for this branch.

### A8-ii — stabilizer-state contraction (blocks B7)

Scope: an *expectation* feature — `⟨ψ|P|ψ⟩` for a stabilizer state `|ψ⟩` given by `n` signed
generators — not stabilizer simulation (no tableau updates under gates), so it does not conflict
with the `lib.rs` non-goal.

Math: `⟨ψ|P|ψ⟩ ∈ {0, ±1}`: nonzero iff `P` (up to sign) is in the stabilizer group `S`; the value
is the sign carried by the group element. Algorithm: preprocess the `n` generators into RREF over
GF(2) symplectic rows once (`O(n³/64)`); per term, decompose `P`'s key against the RREF
(`O(n²/64)` word ops) — if it reduces to zero, `P ∈ ±S` and the sign is accumulated from the
generators used (Aaronson–Gottesman phase bookkeeping, including the `i` powers from
`mul_assign`); else the term contributes 0. Whole-sum cost `O(m·n²/64)` — exactly the handoff's
requirement, never a 2^n expansion.

API sketch (phase 2): `StabilizerState.from_generators(["+ZZ", "-XI", ...])` and
`from_stim_tableau(t)` (interop-level), accepted by `expectation(state=...)` alongside the A4
strings. Core: a `StabilizerBasis<W>` beside `ProductBasis<W>` with the RREF held in the same
`[u64; W]` row layout. Validation: exactly `num_qubits` independent, pairwise-commuting
generators (independence and commutation checked at construction; `ValueError` otherwise).
Product states become the special case of diagonal generators — A4 stays the fast path.

---

## Cross-item sequencing note

A1 lands first (everything else's tests want `pauli_rotation` fixtures cheap); A5's stim mapping
of `X_ERROR`/`PAULI_CHANNEL` variants depends on A6 — land A6 before or with A5, or gate those two
instruction mappings on it. A3/A4 are independent of both. A7 is independent of everything
(pure Python) but its `make_policy` docstring cites the jl-boundary probe, which only exists once
the Julia driver (J1) runs.
