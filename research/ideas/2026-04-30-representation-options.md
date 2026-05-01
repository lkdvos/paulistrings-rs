# Representation options for Pauli strings

Seed file. Things to think through before committing to a core representation.

## Candidate encodings

- **Symplectic bit-pair (x, z)**: two bits per qubit, packed into `u64` words.
  Multiplication is XOR on (x, z); phase from popcount of `x_a & z_b`. Standard
  in stabilizer simulators (stim, qiskit).
- **Dense `u8` per site (0,1,2,3 → I,X,Y,Z)**: simpler, slower, easier to read.
  Useful as a reference impl for fuzzing.
- **Sparse {site → P}**: good for low-weight strings, bad for dense ones. May
  need both and pick at runtime by weight threshold.
- **Pauli sum (linear combination of strings)**: hashmap or sorted vec of
  (string, coeff). Hot question: hashmap (O(1) lookup, bad cache) vs. sorted
  vec (O(log n) lookup, great cache, batch-mergeable).

## Open questions

- Should the qubit count be a const generic, runtime, or both via a trait?
- 64-qubit fast path vs. arbitrary-length via `SmallVec<[u64; N]>`?
- Phase: `{+1, +i, -1, -i}` as `u8` mod 4 vs. complex coeff folded into the
  Pauli sum coefficient? Pure strings probably want the former.
- SIMD: AVX2 / AVX-512 / NEON for batch multiplication of many strings against
  one. Worth measuring before designing the API around it.

## Comparison targets to study

- `PauliStrings.jl` storage layout — what tradeoffs did Loïc pick and why?
- `stim`'s `PauliString` — heavily optimized, Clifford-focused.
- `qiskit`'s `SparsePauliOp` — pragmatic, Python-first.
