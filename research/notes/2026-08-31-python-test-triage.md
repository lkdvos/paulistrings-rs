# Python test suite: first execution (Stage-1 Track C)

Context: `python/paulistrings/tests/` (57 tests at the time the task was
written; 71 collected once `pytest` actually ran, because several are
`@pytest.mark.parametrize`d) had never been executed — no `maturin`/venv was
available on the dev host through v0.2/v0.3 (see `CLAUDE.md`'s known-gaps
list). This is the first run.

## Environment bring-up

`/usr/bin/python3.11` (the script's default) is not present on this host.
Flatiron's Lmod module system provides one instead:

```
module load modules/2.4-20250724 python/3.11.11
```

Ran setup with that interpreter:

```
PYTHON=/mnt/sw/nix/store/a8ppasiv1n969bz16j9109jm05a13ncy-python-3.11.11-view/bin/python3.11 ./scripts/setup.sh
```

This succeeded end to end: venv created, dev tooling installed, optional
cross-library benchmark deps (`qiskit`, `openfermion`, `stim`) installed
without falling back to the "best-effort" warning path, and
`maturin develop --release -m crates/paulistrings-py/Cargo.toml` built and
installed the extension (release profile, ~35s wall for the `paulistrings` +
`paulistrings-py` link step under `lto = "fat"`, `codegen-units = 1`).

No bring-up failures to record on the venv/maturin side.

## Result

```
./.venv/bin/pytest python/paulistrings/tests -q
```

**69 passed, 2 failed** (71 collected). Both failures are in
`test_expectation.py`:

- `test_single_pauli_expectations[YI-0.0-1.0-0.0]`
  ```
  assert s.expectation("y+").real == pytest.approx(y_plus)
  E   assert 0.0 == 1.0 ± 1.0e-06
  E     Obtained: 0.0
  E     Expected: 1.0 ± 1.0e-06
  ```
- `test_expectation_of_multi_qubit_products`
  ```
  assert s.expectation("y+").real == pytest.approx(100.0)
  E   assert -100.0 == 100.0 ± 1.0e-04
  E     Obtained: -100.0
  E     Expected: 100.0 ± 1.0e-04
  ```

## Not caused by the Track C refactor

`crates/paulistrings-py/src/sum.rs`'s `parse_terms`, `extract_complex`, and the
`PauliSum::expectation` / `PauliSumImpl::expectation` bodies are byte-for-byte
unchanged by the width-dispatch macro work — the refactor only replaced
identical per-arm `match` bodies with `for_each_width!` calls that expand back
to the same code (verified by reading the diff; also confirmed independently
by reasoning through the call graph rather than needing a `git stash`
rebuild, since none of the touched functions are on the path these two tests
exercise). So this is pre-existing.

## Diagnosis

This is a **Y-phase convention conflict** between the Python string parser and
the core's `expectation_product_state`, not a typo, and it traces back through
documentation in at least two other places — so it isn't a "renamed kwarg"
class of trivial fix and hasn't been touched.

**The parser's convention.** `PauliSumImpl::from_strings_dict` →
`parse_terms` (`crates/paulistrings-py/src/sum.rs`) encodes a `'Y'` character
as `x |= bit; z |= bit; phase += Phase::I` — i.e. it stores `i · c` at the
`(x=1, z=1)` key for a user-supplied coefficient `c`. The module doc comment
states this explicitly: *"`Y` is `Y_canonical`, i.e. `i · (x=1, z=1)`, with the
`i` factor folded into the coefficient."* This convention is itself pinned by
a currently-**passing** Python test,
`test_pauli_sum.py::test_from_strings_with_complex_and_y_phase`:
`PauliSum.from_strings({"Y": 1.0}, num_qubits=1).coefficients() == [0 + 1j]`.

**The core's convention.** `PauliSum::expectation_product_state`
(`crates/paulistrings/src/bucket/sum.rs`) assumes the *opposite*: that a term
at key `(x, z)` with coefficient `c` already denotes `c` times the literal
Hermitian Pauli string (so an unmodified real `c` at the Y-slot is expected to
contribute `c` — real — to the Y+ expectation, exactly like the X and Z
slots). This is pinned by two currently-**passing** core tests in
`crates/paulistrings/src/pauli_sum.rs`:
`expectation_of_single_paulis_in_each_product_state` and
`expectation_of_multi_qubit_products`, both of which build sums via
`PauliString::y(qubit)` (which sets `x = z = bit` with **no** phase
adjustment) plus a plain real `Complex64` coefficient, and assert a real,
unscaled Y+ expectation.

**These two conventions cannot both hold for the same stored value.** Feed a
term built the parser's way (`i · c` at the Y-slot) into the core's
expectation code (which assumes `c` at the Y-slot) and the extra `i` factor
either zeroes the real part (single Y: pure-imaginary contribution, `.real ==
0.0` — the first failure) or flips its sign (`YY`: `i · i = -1`, so `100.0`
becomes `-100.0` — the second failure). The arithmetic matches exactly.

**Why this isn't a one-line fix.** The parser's `i`-folding isn't an isolated
mistake in `sum.rs` — the identical framing ("Y as `i · (X·Z)`") appears in
`crates/paulistrings/src/accumulator.rs`'s `BuildAccumulator::add_term` doc
comment and doctest (`// Add a Y-like term written as i · (X · Z) on qubit
0`, asserting the stored coefficient comes out as `0 + 1i` for input `1.0`).
So the same convention is asserted, and doc-tested, in three places: this
task's target file, a core-crate doc comment (out of scope — "do NOT touch the
core crate"), and a currently-green Python test that would need its expected
value changed if the parser changed. Reconciling them means picking one
convention and updating the pinned tests/docs on the other side to match —
a real design decision, not a trivial patch, and one that reaches outside
this Stage-1 Track C's scope (bindings + macro refactor only). Recorded here
rather than fixed.

**Practical note for whoever resolves it:** `PauliSum.overlap`,
`identity_coefficient`, and `Circuit` propagation are unaffected in this run
— none of the passing tests exercise a `'Y'` character through
`from_strings` *and* an operation whose correctness depends on the Hermitian
Y-coefficient convention (the `Circuit`/gate tests use Clifford/rotation
channels operating on X/Z inputs, and `test_general_unitary.py`'s Y-adjacent
assertions only check `abs(coeff)`, which is convention-independent). The
blast radius is specifically "a `'Y'` character parsed via `from_strings`,
then read through `expectation`."

## Proposed CI job sketch (not implemented)

`.github/workflows/ci.yml` currently runs Rust only (fmt/clippy/test/doc), per
CLAUDE.md's known-gaps note. A `python` job could look like:

```yaml
  python:
    name: python
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions-rust-lang/setup-rust-toolchain@v1
      - uses: Swatinem/rust-cache@v2
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install 'maturin>=1.5,<2.0' 'pytest>=7' 'numpy>=1.24'
      - run: maturin develop --release -m crates/paulistrings-py/Cargo.toml
      - run: pytest python/paulistrings/tests -q
```

Notes for whoever wires this up:

- Skip the optional cross-library benchmark deps (`qiskit`, `openfermion`,
  `stim`) — they're for `benchmarks/python/`, not `python/paulistrings/tests/`,
  and add several minutes plus a large dependency surface for no coverage
  gain in this job.
- `maturin develop --release` pays the full `lto = "fat"`, `codegen-units = 1`
  link cost (~35s on this host, on top of the normal crate build) every run.
  `Swatinem/rust-cache@v2` should absorb most of the incremental cost;
  consider a debug-profile `maturin develop` (no `--release`) instead if CI
  time matters more than exercising the release codegen path, since the
  tests only check correctness, not performance.
- This job should be allowed to fail *loudly* on the Y-phase issue above
  once it's turned on, rather than starting in `continue-on-error` — the two
  failures are legitimate and the point of adding the job is to keep them
  from silently regressing further while unresolved.
- Gate it the same as the Rust jobs (`on: push: branches: [main]` /
  `pull_request`), not on a separate schedule — the whole point of Stage-1
  Track C was catching drift between the Rust core and the Python surface,
  which only happens if it runs on every PR.

## Resolution (2026-08-31, same day)

**Decision: the core's Hermitian convention wins.** A coefficient multiplies
the literal Hermitian Pauli string; `Y` maps to the symplectic key
`(x=1, z=1)` with no phase factor. This is what `PauliString::y`,
`expectation_product_state`, and the algebra tests (`X·Z = −iY` at bits
`(1,1)` with phase `i³`) already pinned, and it keeps Hermitian observables
on real coefficients — the parser's `i`-folding was the outlier.

Changes:
- `parse_terms` (`crates/paulistrings-py/src/sum.rs`) and the test helper
  `PauliSum::from_strings` (`crates/paulistrings/src/pauli_sum.rs`) no longer
  fold a phase for `'Y'`; their doc comments state the Hermitian convention.
- `BuildAccumulator::add_term`'s doctest reframed: it demonstrates folding a
  *product* phase (`Z·X = +iY`), which is the mechanism's actual purpose —
  the mechanics are unchanged.
- Tests that pinned the folding updated: Python
  `test_from_strings_with_complex_and_y_phase` → `test_from_strings_y_is_hermitian`;
  Rust `from_strings_y*_phase_*` → `from_strings_y_is_hermitian` +
  `from_strings_real_coeffs_stay_real_for_any_y_count`; the `YI` coefficient
  in `from_strings_sorts_lex_keys`.
- Both `test_expectation.py` failures now pass: the suite is 71/71.

The CI `python` job (added the same day per the sketch above) keeps the two
surfaces from drifting again.
