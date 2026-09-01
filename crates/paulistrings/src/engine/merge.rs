//! Per-run sort and fused two-stream merge — the bucketed engine's inner
//! kernels.
//!
//! [`sort_rows_with_scratch`] and [`sort_rows_radix_with_scratch`] canonicalize
//! one gather run's *rest* stream — the first by comparison, the second by a
//! radix pass over a monotone surrogate; `bucketed.rs` picks between them per
//! layer from the plan's rest-delta count, see [`RADIX_MIN_REST_STREAMS`].
//! [`merge2_into`] then fuses the id/rest two-stream merge with the segmented
//! reduction that restores the `PauliSum` invariant (strictly ascending, no
//! duplicates) inside a destination bucket. All are called per gather run by
//! `engine::bucketed`; [`SortScratch`] is the worker-persistent scratch the
//! sorts reuse so a steady-state layer allocates nothing.
//!
//! Both sorts satisfy one contract and nothing more, pinned by
//! `tests::assert_sort_contract`: the output is **ascending** in lex `(x, z)`
//! with duplicates allowed, and is a permutation of the input `(x, z, c)`
//! triples. Equal-key order is unspecified (ARCHITECTURE.md §Determinism), so
//! the two kernels are interchangeable to floating-point tolerance and not
//! bitwise.

use num_complex::Complex64;

use crate::truncation::TruncationPolicy;

/// Worker-persistent scratch for the per-run sorts.
///
/// Held across coset tasks (one instance per `CosetScratch`, in turn one per
/// Rayon worker, per `bucketed.rs`'s `LayerScratch`): every buffer retains its
/// high-water capacity across calls, so a run at or below a previously-seen
/// size sorts without allocating. `perm` serves the comparison kernel;
/// `packed`/`aux` serve the radix kernel; the `tmp_*` triple is the output
/// staging both share.
#[derive(Clone, Debug, Default)]
pub(crate) struct SortScratch<const W: usize> {
    perm: Vec<u32>,
    /// `(surrogate << 32) | row index` records, in sorted order once the
    /// radix kernel has run. Empty unless that kernel is selected.
    packed: Vec<u64>,
    /// The radix kernel's double buffer for `packed`.
    aux: Vec<u64>,
    tmp_x: Vec<[u64; W]>,
    tmp_z: Vec<[u64; W]>,
    tmp_c: Vec<Complex64>,
}

impl<const W: usize> SortScratch<W> {
    /// Total heap capacity held across this scratch's buffers — a private
    /// implementation detail exposed only for
    /// `bucketed::tests::capacity_stabilizes_across_repeated_layers`, which
    /// needs it to confirm the sort scratch's footprint stops growing too.
    #[cfg(test)]
    pub(crate) fn total_capacity(&self) -> usize {
        self.perm.capacity()
            + self.packed.capacity()
            + self.aux.capacity()
            + self.tmp_x.capacity()
            + self.tmp_z.capacity()
            + self.tmp_c.capacity()
    }
}

/// Sort `(x, z, c)` columns in place by the key `(x, z)` alone, using `s` as
/// reusable scratch.
///
/// Equal-key summation order is not required to be bucket-count- or
/// hash-seed-independent (floating-point associativity variation across
/// those axes is accepted, ARCHITECTURE.md §Determinism), so this sort
/// compares the key alone — cheaper, and with one fewer column to carry
/// through the gather.
///
/// The sort is the **stable** `sort_by`, but not for stability (nothing
/// depends on equal-key order any more — an unstable sort would be
/// semantically fine): it is for *adaptivity*. A gather run is a
/// concatenation of per-delta streams, each drawn from one sorted source
/// bucket — the identity stream arrives fully sorted, and an XOR-by-constant
/// stream is piecewise sorted (order survives wherever the mask's high bits
/// don't flip) — and Rust's stable driftsort detects and merges those natural
/// ascending runs while the unstable pdqsort does not. Measured: switching
/// this line to `sort_unstable_by` cost +77% on a 10⁶ `rotation_zz` layer
/// and +43% on CNOT.
///
/// What must still hold — and does, structurally: cosets are write-disjoint,
/// work within one is sequential, and the sort is a deterministic function of
/// its input, so **thread-count determinism and repeat-run determinism at
/// fixed configuration** are unaffected. A later merge sums whatever order
/// equal keys land in; that sum agrees with any other order to floating-point
/// tolerance (real addition is associative; `f64` addition is not, only up to
/// rounding), never bit-for-bit across a different order.
///
/// Scratch-swap capacity circulation: `s.perm` is filled with the identity
/// permutation `0..len` and reordered by the sort; the caller's columns are
/// then read out through the permutation directly into `s.tmp_*` (one pass,
/// not two), and finally
/// each `tmp_*` is `mem::swap`ped with the caller's `Vec`. The caller ends up
/// holding the sorted columns; `s` ends up holding the caller's pre-sort
/// columns' storage (cleared next call) as its own scratch capacity — so
/// capacity circulates between the live columns and the scratch instead of
/// either side ever growing past its high-water mark.
// `#[inline]` is load-bearing: without it, moving this function between
// modules measured ~6% slower single-threaded on the rotation family
// (interleaved A/B, 3/3 pairs) — an LTO code-layout effect, not logic.
// Hint the sort ONLY: adding `#[inline]` to `merge2_into` as well measured
// +20-34% on criterion's apply_layer_bucketed/rotation_zz.
#[inline]
pub(crate) fn sort_rows_with_scratch<const W: usize>(
    x: &mut Vec<[u64; W]>,
    z: &mut Vec<[u64; W]>,
    c: &mut Vec<Complex64>,
    s: &mut SortScratch<W>,
) {
    let len = x.len();
    debug_assert_eq!(len, z.len());
    debug_assert_eq!(len, c.len());
    debug_assert!(len <= u32::MAX as usize);
    if len < 2 {
        return;
    }
    s.perm.clear();
    s.perm.extend(0..len as u32);
    s.perm.sort_by(|&a, &b| {
        x[a as usize]
            .cmp(&x[b as usize])
            .then_with(|| z[a as usize].cmp(&z[b as usize]))
    });
    s.tmp_x.clear();
    s.tmp_x.extend(s.perm.iter().map(|&i| x[i as usize]));
    s.tmp_z.clear();
    s.tmp_z.extend(s.perm.iter().map(|&i| z[i as usize]));
    s.tmp_c.clear();
    s.tmp_c.extend(s.perm.iter().map(|&i| c[i as usize]));
    std::mem::swap(x, &mut s.tmp_x);
    std::mem::swap(z, &mut s.tmp_z);
    std::mem::swap(c, &mut s.tmp_c);
}

/// Minimum number of *rest* delta streams in a gather run before
/// `bucketed.rs` switches that layer to [`sort_rows_radix_with_scratch`].
///
/// The two kernels win in opposite regimes, and the crossover is steep, so
/// this is deliberately conservative — only a genuinely dense two-qubit PTM
/// (a general SU(4) realizes all 16 bucket deltas, so 15 rest streams) clears
/// it. Measured on synthetic runs faithful to `fill_coset`'s output-major
/// gather (non-authoritative microbench, `research/notes/2026-09-01-sort-kernel.md`):
///
/// | run shape | rest streams | duplicate density | radix vs comparison |
/// |---|---|---|---|
/// | dense PTM (`su4`) | 15 | ~15× | **−36…−51 % at `W = 1`, −7…−16 % at `W = 2`** |
/// | dense PTM, one bucket | 15 | ~15× | **−31…−38 % at `W = 2`** |
/// | sparse PTM (`rotation_zz`) | 1 | ~1× | **+130…+165 %** |
///
/// The sparse-PTM row is why this is a gate and not a replacement: with one
/// nearly-sorted stream the comparison sort costs about one comparison per row
/// and the radix's fixed passes are pure overhead. `2..8` rest streams —
/// `sqrt(SWAP)`'s regime, whose sort is 33 % of its layer — is **unmeasured**
/// and deliberately left on the comparison kernel.
pub(crate) const RADIX_MIN_REST_STREAMS: usize = 8;

/// Surrogate width the radix kernel sorts on, in [`RADIX_DIGIT_BITS`] digits.
///
/// 16 bits is two passes. It is not chosen to resolve every key — it is chosen
/// to resolve *groups*: a dense-PTM run of `n` rows holds about `n / 15`
/// distinct keys, so 65536 surrogate values leave the residual tie groups at
/// the duplicate groups themselves, which the fixup pass then orders with
/// roughly one full-key comparison per row. Wider surrogates buy fewer ties at
/// the cost of another whole pass and measured worse everywhere (24 bits
/// −2…−38 %, 32 bits +11…−32 % against 16 bits' −7…−51 %); a single 11-bit
/// pass beat it below ~8 k rows and lost above.
const RADIX_SURROGATE_BITS: u32 = 16;
/// Digit width per radix pass. 256 counters is 1 KiB of stack histogram.
const RADIX_DIGIT_BITS: u32 = 8;
const RADIX_BUCKETS: usize = 1 << RADIX_DIGIT_BITS;
/// Below this many *discriminating* bits in the surrogate window the radix
/// pass cannot separate the run into useful groups, so the comparison kernel
/// runs instead. Reached by low-weight sums (a `WeightCutoff`-truncated sum
/// whose `x` words are nearly constant) and by narrow key spaces.
const RADIX_MIN_WINDOW_BITS: u32 = 8;

/// Word `k` of the lex key `(x, z)`: `k < W` selects `x[k]`, `k >= W` selects
/// `z[k - W]`. Word 0 is the **most** significant, matching the derived `Ord`
/// on `[u64; W]` and hence `PauliString`'s (ARCHITECTURE.md §Data-Model).
#[inline(always)]
fn key_word<const W: usize>(x: &[[u64; W]], z: &[[u64; W]], k: usize, i: usize) -> u64 {
    if k < W {
        x[i][k]
    } else {
        z[i][k - W]
    }
}

/// The most significant key word the rows disagree on, its index, and the
/// disagreeing bits within it. `None` ⟺ every row carries the same key.
///
/// One `OR`/`AND` reduction per word, stopping at the first word that
/// disagrees — for the dense-PTM runs this kernel serves that is word 0, so
/// the scan touches only the `x` column.
fn discriminating_window<const W: usize>(
    x: &[[u64; W]],
    z: &[[u64; W]],
) -> Option<(usize, u32, u64)> {
    let n = x.len();
    for k in 0..2 * W {
        let mut any = 0u64;
        let mut all = !0u64;
        for i in 0..n {
            let v = key_word(x, z, k, i);
            any |= v;
            all &= v;
        }
        let diff = any & !all;
        if diff != 0 {
            // Rows agree on every word before `k` and, within word `k`, on
            // every bit above the highest set bit of `diff`.
            return Some((k, 63 - diff.leading_zeros(), diff));
        }
    }
    None
}

/// Sort `(x, z, c)` columns in place by the key `(x, z)` alone — radix
/// variant, for gather runs assembled from many delta streams.
///
/// Same contract as [`sort_rows_with_scratch`] and freely interchangeable with
/// it (equal-key order differs; see the module doc). Chosen per layer by
/// `bucketed.rs` when the plan has at least [`RADIX_MIN_REST_STREAMS`] rest
/// streams, and it falls back to the comparison kernel itself whenever its
/// surrogate cannot discriminate.
///
/// # Why a surrogate, and why it is order-faithful
///
/// The full key is `16·W` bytes — 32 at `W = 2` — so an LSD radix over all of
/// it would be 32 passes. Instead one pass finds
/// [`discriminating_window`]: the most significant key word `k` the rows
/// actually disagree on, and the highest disagreeing bit `hb` inside it. Every
/// row then shares the same value on words before `k` and on bits above `hb`,
/// so writing `word_k = H·2^(hb+1) + L` with `H` **constant across the run**,
///
/// ```text
/// surrogate = (word_k >> shift) & (2^NBITS − 1) = L >> shift,
///     shift = (hb + 1) − NBITS
/// ```
///
/// — the mask erases exactly the constant `H`, and `L >> shift` is monotone in
/// `L`, which is monotone in the key. So `key₁ < key₂ ⟹ surrogate₁ ≤
/// surrogate₂`: sorting by the surrogate never puts two rows in the wrong
/// order, it only leaves ties, which the fixup pass resolves on the full key.
/// A run whose keys are *all equal* needs no work at all and returns early.
///
/// # Why this beats the comparison sort where it is selected
///
/// Not by doing less work — a dense-PTM run arrives as ~15 ascending blocks
/// and driftsort already merges them at about `log₂ 15 + 1 ≈ 4.9` comparisons
/// per row, the information-theoretic floor. It wins on the cost of that work:
/// each of those comparisons is a *dependent indexed load* into a 100–400 KiB
/// key column (the permutation sort's whole cost, ~10–13 cycles), whereas a
/// radix pass streams 8-byte records sequentially at ~2 cycles each. Two
/// passes plus the fixup replace 4.9 such comparisons with ~1.
// No `#[inline]` hint, deliberately: the one on `sort_rows_with_scratch` is
// A/B-verified worth ~6% and the one tried on `merge2_into` cost +20-34%
// (both recorded on those items), so the attribute is load-bearing in both
// directions here and this function has no measurement either way yet. Leave
// it at the default and A/B the hint as its own change.
pub(crate) fn sort_rows_radix_with_scratch<const W: usize>(
    x: &mut Vec<[u64; W]>,
    z: &mut Vec<[u64; W]>,
    c: &mut Vec<Complex64>,
    s: &mut SortScratch<W>,
) {
    let len = x.len();
    debug_assert_eq!(len, z.len());
    debug_assert_eq!(len, c.len());
    if len < 2 {
        return;
    }
    // The record packs the row index into the low 32 bits.
    if len > u32::MAX as usize {
        sort_rows_with_scratch(x, z, c, s);
        return;
    }
    let Some((k, hb, diff)) = discriminating_window(x, z) else {
        // Every row carries the same key, so any order is already ascending.
        return;
    };
    let shift = (hb + 1).saturating_sub(RADIX_SURROGATE_BITS);
    let mask = (1u64 << RADIX_SURROGATE_BITS) - 1;
    if ((diff >> shift) & mask).count_ones() < RADIX_MIN_WINDOW_BITS {
        sort_rows_with_scratch(x, z, c, s);
        return;
    }

    let SortScratch {
        packed,
        aux,
        tmp_x,
        tmp_z,
        tmp_c,
        ..
    } = s;
    packed.clear();
    packed.extend((0..len).map(|i| (((key_word(x, z, k, i) >> shift) & mask) << 32) | i as u64));
    // Exactly `len`, so the pass-to-pass `swap` keeps both buffers that long;
    // `resize` on the retained capacity writes only when the run grew.
    aux.resize(len, 0);

    let mut digit = 0u32;
    while digit * RADIX_DIGIT_BITS < RADIX_SURROGATE_BITS {
        let sh = 32 + digit * RADIX_DIGIT_BITS;
        let mut count = [0u32; RADIX_BUCKETS + 1];
        for &v in packed.iter() {
            count[(((v >> sh) as usize) & (RADIX_BUCKETS - 1)) + 1] += 1;
        }
        // A constant digit contributes no ordering: skip its scatter. Common
        // on the high digit, whose bits the window shift often leaves fixed.
        if count[1..].iter().filter(|&&n| n != 0).count() > 1 {
            for t in 1..=RADIX_BUCKETS {
                count[t] += count[t - 1];
            }
            for &v in packed.iter() {
                let b = ((v >> sh) as usize) & (RADIX_BUCKETS - 1);
                aux[count[b] as usize] = v;
                count[b] += 1;
            }
            std::mem::swap(packed, aux);
        }
        digit += 1;
    }

    // Fixup: rows sharing a surrogate are ordered on the full key. The radix
    // is stable and the index sits in the low bits, so a group arrives in
    // gather order; on the dense-PTM runs this kernel serves, a group is one
    // duplicate key's ~15 rows and this costs ~1 comparison per row.
    let mut i = 0usize;
    while i < len {
        let surrogate = packed[i] >> 32;
        let mut j = i + 1;
        while j < len && packed[j] >> 32 == surrogate {
            j += 1;
        }
        if j - i > 1 {
            packed[i..j].sort_by(|&a, &b| {
                let (ia, ib) = (a as u32 as usize, b as u32 as usize);
                x[ia].cmp(&x[ib]).then_with(|| z[ia].cmp(&z[ib]))
            });
        }
        i = j;
    }

    // Read the columns out through the record order into the staging triple,
    // then swap — the same capacity circulation `sort_rows_with_scratch` does.
    tmp_x.clear();
    tmp_x.extend(packed.iter().map(|&v| x[v as u32 as usize]));
    tmp_z.clear();
    tmp_z.extend(packed.iter().map(|&v| z[v as u32 as usize]));
    tmp_c.clear();
    tmp_c.extend(packed.iter().map(|&v| c[v as u32 as usize]));
    std::mem::swap(x, tmp_x);
    std::mem::swap(z, tmp_z);
    std::mem::swap(c, tmp_c);
}

/// Fused two-stream merge + segmented reduction.
///
/// `a` is a gather run's identity-delta stream: its keys are untouched source
/// keys, so it inherits the bucket invariant — strictly ascending, no
/// duplicates — and is **never sorted**. (Under a dense identity plan the
/// key slices are the *source bucket's own columns*, borrowed in place, with
/// only the coefficients gathered; this function cannot tell and
/// need not care.) `b` is the run's remaining rows, canonicalized by
/// [`sort_rows_with_scratch`] (ascending, duplicates allowed).
/// The two-pointer walk consumes rows in global key order, seeding a key tie
/// from the `a` row first and then adding the equal-key `b` rows in their
/// sorted order; that order is deterministic for a fixed input but not
/// specified across partitions (ARCHITECTURE.md §Determinism). Zero-drop and
/// `keep_term` see the fully summed coefficient. When `a` is empty this
/// degenerates to the plain single-stream segmented reduction, which is the
/// whole story for a channel with no identity delta: everything is gathered
/// into `b`.
///
/// Exact-zero rows are consumed like any other (a `θ = π/2` rotation emits
/// `cos·coeff = ±0.0` rows): dropping them *before* the reduction could flip
/// the sign of a zero sum, so the only zero test is on the final accumulator.
///
/// Do not restructure this walk into gallop + bulk segment copies: measured
/// +20–35% merge busy on every real cell except 1t trotter, because
/// the workloads' id/rest densities make the average id segment one or two
/// rows (gu2q: mostly empty) — per-segment overhead swamps the per-row
/// compare it saves. Full data in `research/notes/2026-08-31-v0.6-results.md`.
#[allow(clippy::too_many_arguments)]
// Deliberately NOT `#[inline]`: hinting this function measured +20-34% on
// criterion's apply_layer_bucketed/rotation_zz (layout/icache), while the
// `sort_rows_with_scratch` hint alone already recovers the probe path.
pub(crate) fn merge2_into<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    a_x: &[[u64; W]],
    a_z: &[[u64; W]],
    a_c: &[Complex64],
    b_x: &[[u64; W]],
    b_z: &[[u64; W]],
    b_c: &[Complex64],
    dst_x: &mut Vec<[u64; W]>,
    dst_z: &mut Vec<[u64; W]>,
    dst_coeff: &mut Vec<Complex64>,
    policy: &T,
) {
    let zero = Complex64::new(0.0, 0.0);
    let (an, bn) = (a_c.len(), b_c.len());
    debug_assert_eq!(an, a_x.len());
    debug_assert_eq!(an, a_z.len());
    debug_assert_eq!(bn, b_x.len());
    debug_assert_eq!(bn, b_z.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < an || j < bn {
        // Take the smaller next key; on a tie the `a` row seeds the sum. After
        // an `a` seed there is no second `a` row for the key (`a` is unique),
        // and after a `b` seed every equal-key `a` row would have compared
        // `<=`, so only `b` rows can extend the segment either way.
        let take_a = j >= bn || (i < an && (a_x[i], a_z[i]) <= (b_x[j], b_z[j]));
        let (key_x, key_z, mut acc) = if take_a {
            debug_assert!(
                i == 0 || (a_x[i - 1], a_z[i - 1]) < (a_x[i], a_z[i]),
                "merge2_into: identity stream must be strictly ascending at {i}",
            );
            let t = (a_x[i], a_z[i], a_c[i]);
            i += 1;
            t
        } else {
            debug_assert!(
                j == 0 || (b_x[j - 1], b_z[j - 1]) <= (b_x[j], b_z[j]),
                "merge2_into: rest stream must be sorted at {j}",
            );
            let t = (b_x[j], b_z[j], b_c[j]);
            j += 1;
            t
        };
        while j < bn && b_x[j] == key_x && b_z[j] == key_z {
            acc += b_c[j];
            j += 1;
        }
        if acc != zero && policy.keep_term(&key_x, &key_z, acc) {
            dst_x.push(key_x);
            dst_z.push(key_z);
            dst_coeff.push(acc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::approx_eq;
    use crate::truncation::CoefficientThreshold;
    use proptest::prelude::*;

    const TOL: f64 = 1e-12;

    // ---- the per-run sort kernel's contract ----
    //
    // Every kernel `bucketed.rs` may pick for a gather run's rest stream must
    // satisfy exactly this, and nothing more: the output is **ascending** in
    // lex `(x, z)` (duplicates allowed — `merge2_into` reduces them) and is a
    // permutation of the input `(x, z, c)` triples, so a coefficient still
    // travels with its own key. Equal-key order is explicitly *not* pinned
    // (ARCHITECTURE.md §Determinism), which is why the check is a multiset
    // comparison rather than an element-wise one.

    type SortKernel<const W: usize> =
        fn(&mut Vec<[u64; W]>, &mut Vec<[u64; W]>, &mut Vec<Complex64>, &mut SortScratch<W>);

    /// Assert the contract above for `kernel` on one run.
    fn assert_sort_contract<const W: usize>(
        kernel: SortKernel<W>,
        x: &[[u64; W]],
        z: &[[u64; W]],
        c: &[Complex64],
        what: &str,
    ) {
        let (mut gx, mut gz, mut gc) = (x.to_vec(), z.to_vec(), c.to_vec());
        let mut scratch = SortScratch::<W>::default();
        kernel(&mut gx, &mut gz, &mut gc, &mut scratch);

        assert_eq!(gx.len(), x.len(), "{what}: row count changed");
        assert_eq!(gz.len(), z.len(), "{what}: row count changed");
        assert_eq!(gc.len(), c.len(), "{what}: row count changed");
        for i in 1..gx.len() {
            assert!(
                (gx[i - 1], gz[i - 1]) <= (gx[i], gz[i]),
                "{what}: not ascending at row {i}",
            );
        }
        // Multiset of triples, with the coefficient bits as the tiebreak so
        // the comparison is exact and order-insensitive.
        let key =
            |(a, b, v): &([u64; W], [u64; W], Complex64)| (*a, *b, v.re.to_bits(), v.im.to_bits());
        let mut want: Vec<([u64; W], [u64; W], Complex64)> = x
            .iter()
            .zip(z)
            .zip(c)
            .map(|((&a, &b), &v)| (a, b, v))
            .collect();
        let mut got: Vec<([u64; W], [u64; W], Complex64)> = gx
            .iter()
            .zip(&gz)
            .zip(&gc)
            .map(|((&a, &b), &v)| (a, b, v))
            .collect();
        want.sort_by_key(key);
        got.sort_by_key(key);
        assert_eq!(
            got, want,
            "{what}: output is not a permutation of the input"
        );
    }

    /// Xorshift64 — local so the fixtures below need no dev-dependency draw
    /// order shared with `test_support`.
    fn xs64(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// The shapes a real gather run takes, plus the degenerate ones a kernel
    /// that looks at the key *bits* (rather than only comparing keys) can trip
    /// over. `(label, x, z, c)`.
    #[allow(clippy::type_complexity)]
    fn sort_fixtures<const W: usize>(
        num_qubits: usize,
    ) -> Vec<(String, Vec<[u64; W]>, Vec<[u64; W]>, Vec<Complex64>)> {
        let mut out = Vec::new();
        let mask = |w: usize| crate::test_support::word_mask(num_qubits, w);
        let mut push = |label: &str, keys: Vec<([u64; W], [u64; W])>, seed: u64| {
            let mut st = seed | 1;
            let c: Vec<Complex64> = keys
                .iter()
                .map(|_| {
                    Complex64::new(
                        (xs64(&mut st) % 17) as f64 - 8.0,
                        (xs64(&mut st) % 13) as f64 - 6.0,
                    )
                })
                .collect();
            out.push((
                format!("{label} W={W} n={}", keys.len()),
                keys.iter().map(|k| k.0).collect(),
                keys.iter().map(|k| k.1).collect(),
                c,
            ));
        };

        // Degenerate lengths.
        for n in [0usize, 1, 2] {
            let keys: Vec<([u64; W], [u64; W])> = (0..n as u64)
                .map(|i| {
                    let mut kx = [0u64; W];
                    kx[0] = (2 - i) & mask(0);
                    ([kx[0]; W].map(|v| v & mask(0)), [0u64; W])
                })
                .collect();
            push(&format!("len{n}"), keys, 0x11);
        }

        // Dense random keys, no duplicates: the sparse-PTM shape.
        let mut st = 0xC0FF_EE00_1234_5678u64;
        let mut keys: Vec<([u64; W], [u64; W])> = (0..400)
            .map(|_| {
                let mut kx = [0u64; W];
                let mut kz = [0u64; W];
                for w in 0..W {
                    kx[w] = xs64(&mut st) & mask(w);
                    kz[w] = xs64(&mut st) & mask(w);
                }
                (kx, kz)
            })
            .collect();
        push("dense_random", keys.clone(), 0x22);
        keys.sort_unstable();
        push("already_sorted", keys.clone(), 0x23);
        keys.reverse();
        push("reverse_sorted", keys, 0x24);

        // Heavy duplicates: the dense-PTM shape. 40 distinct keys, each
        // repeated 15 times, the repeats interleaved as 15 sorted streams.
        let mut st = 0x5EED_0000_0000_0001u64;
        let mut distinct: Vec<([u64; W], [u64; W])> = (0..40)
            .map(|_| {
                let mut kx = [0u64; W];
                let mut kz = [0u64; W];
                for w in 0..W {
                    kx[w] = xs64(&mut st) & mask(w);
                    kz[w] = xs64(&mut st) & mask(w);
                }
                (kx, kz)
            })
            .collect();
        distinct.sort_unstable();
        distinct.dedup();
        let mut dup = Vec::new();
        for _ in 0..15 {
            dup.extend(distinct.iter().copied());
        }
        push("dup15_streams", dup, 0x25);

        // Every row the same key.
        push("all_equal", vec![distinct[0]; 64], 0x26);

        // `x` identically zero: all discrimination lives in `z`, so a kernel
        // that keys off the most significant word must walk past `x`.
        let zero_x: Vec<([u64; W], [u64; W])> = distinct.iter().map(|k| ([0u64; W], k.1)).collect();
        push("x_all_zero", zero_x, 0x27);

        // Two-bit key space: only bits 0 and 1 of `x[0]` vary, so any
        // fixed-width surrogate window has almost no discriminating power.
        let thin: Vec<([u64; W], [u64; W])> = (0..200u64)
            .map(|i| {
                let mut kx = [0u64; W];
                kx[0] = (i % 4) & mask(0);
                (kx, [0u64; W])
            })
            .collect();
        push("thin_window", thin, 0x28);

        // A constant *nonzero* high part above the varying bits — the case
        // where masking a shifted window must not reorder rows.
        let hi = if num_qubits >= 64 {
            1u64 << 63
        } else {
            1 << (num_qubits - 1)
        };
        let biased: Vec<([u64; W], [u64; W])> = (0..300u64)
            .map(|i| {
                let mut kx = [0u64; W];
                kx[0] = (hi | (i * 7)) & mask(0);
                (kx, [0u64; W])
            })
            .collect();
        push("constant_high_bit", biased, 0x29);

        out
    }

    /// The shipping comparison kernel satisfies the contract on every shape.
    #[test]
    fn sort_rows_with_scratch_honors_the_kernel_contract() {
        for (label, x, z, c) in sort_fixtures::<1>(64) {
            assert_sort_contract(sort_rows_with_scratch::<1>, &x, &z, &c, &label);
        }
        for (label, x, z, c) in sort_fixtures::<2>(128) {
            assert_sort_contract(sort_rows_with_scratch::<2>, &x, &z, &c, &label);
        }
        for (label, x, z, c) in sort_fixtures::<2>(65) {
            assert_sort_contract(sort_rows_with_scratch::<2>, &x, &z, &c, &label);
        }
    }

    /// The radix kernel satisfies the *same* contract on every shape — the
    /// point of the harness. Includes the two shapes that must reach its
    /// fallbacks: `thin_window` (fewer than `RADIX_MIN_WINDOW_BITS`
    /// discriminating bits) and `all_equal` (no discriminating word at all).
    #[test]
    fn sort_rows_radix_with_scratch_honors_the_kernel_contract() {
        for (label, x, z, c) in sort_fixtures::<1>(64) {
            assert_sort_contract(sort_rows_radix_with_scratch::<1>, &x, &z, &c, &label);
        }
        for (label, x, z, c) in sort_fixtures::<2>(128) {
            assert_sort_contract(sort_rows_radix_with_scratch::<2>, &x, &z, &c, &label);
        }
        for (label, x, z, c) in sort_fixtures::<2>(65) {
            assert_sort_contract(sort_rows_radix_with_scratch::<2>, &x, &z, &c, &label);
        }
        // Narrow qubit counts put every discriminating bit low in word 0, so
        // the window shift saturates at 0 and the mask covers the whole word.
        for q in [3usize, 8, 17, 33] {
            for (label, x, z, c) in sort_fixtures::<1>(q) {
                assert_sort_contract(sort_rows_radix_with_scratch::<1>, &x, &z, &c, &label);
            }
        }
    }

    /// Both kernels agree on the *reduced* content of every fixture: same keys
    /// in the same order, and equal-key coefficient sums that agree exactly
    /// (the fixtures' coefficients are small integers, so any summation order
    /// is exact). This is the interchangeability claim `bucketed.rs` relies on
    /// when it picks a kernel per layer.
    #[test]
    fn the_two_sort_kernels_reduce_to_the_same_sum() {
        #[allow(clippy::type_complexity)]
        fn reduced<const W: usize>(
            kernel: SortKernel<W>,
            x: &[[u64; W]],
            z: &[[u64; W]],
            c: &[Complex64],
        ) -> Vec<([u64; W], [u64; W], Complex64)> {
            let (mut gx, mut gz, mut gc) = (x.to_vec(), z.to_vec(), c.to_vec());
            let mut scratch = SortScratch::<W>::default();
            kernel(&mut gx, &mut gz, &mut gc, &mut scratch);
            let (mut ox, mut oz, mut oc) = (Vec::new(), Vec::new(), Vec::new());
            merge2_into(
                &[],
                &[],
                &[],
                &gx,
                &gz,
                &gc,
                &mut ox,
                &mut oz,
                &mut oc,
                &AlwaysKeep,
            );
            ox.into_iter()
                .zip(oz)
                .zip(oc)
                .map(|((a, b), v)| (a, b, v))
                .collect()
        }
        for (label, x, z, c) in sort_fixtures::<1>(64) {
            assert_eq!(
                reduced(sort_rows_radix_with_scratch::<1>, &x, &z, &c),
                reduced(sort_rows_with_scratch::<1>, &x, &z, &c),
                "{label}",
            );
        }
        for (label, x, z, c) in sort_fixtures::<2>(128) {
            assert_eq!(
                reduced(sort_rows_radix_with_scratch::<2>, &x, &z, &c),
                reduced(sort_rows_with_scratch::<2>, &x, &z, &c),
                "{label}",
            );
        }
    }

    /// A steady-state layer must not allocate: the radix kernel's own buffers
    /// have to stop growing once the largest run has been seen.
    #[test]
    fn radix_sort_scratch_capacity_stabilizes() {
        let mut scratch = SortScratch::<2>::default();
        let fixtures = sort_fixtures::<2>(128);
        let run = |s: &mut SortScratch<2>| {
            for (_, x, z, c) in fixtures.iter() {
                let (mut gx, mut gz, mut gc) = (x.clone(), z.clone(), c.clone());
                sort_rows_radix_with_scratch(&mut gx, &mut gz, &mut gc, s);
            }
        };
        run(&mut scratch);
        run(&mut scratch);
        let after_two = scratch.total_capacity();
        run(&mut scratch);
        run(&mut scratch);
        assert_eq!(
            scratch.total_capacity(),
            after_two,
            "radix scratch capacity kept growing after the high-water run",
        );
    }

    proptest! {
        /// Randomized shapes, including short runs, narrow key spaces and
        /// heavy duplication (the `% modulus` draw makes collisions common).
        #[test]
        fn sort_rows_radix_contract_proptest(
            rows in prop::collection::vec((any::<u64>(), any::<u64>()), 0..300usize),
            modulus in 1u64..64,
            spread in 0u32..60,
        ) {
            // `spread` slides the varying bits up and down word 0, exercising
            // every window shift including the saturating one.
            let x: Vec<[u64; 1]> = rows.iter().map(|r| [(r.0 % modulus) << spread]).collect();
            let z: Vec<[u64; 1]> = rows.iter().map(|r| [(r.1 % modulus) << spread]).collect();
            let c: Vec<Complex64> = rows
                .iter()
                .map(|r| Complex64::new((r.0 % 11) as f64 - 5.0, (r.1 % 7) as f64 - 3.0))
                .collect();
            assert_sort_contract(sort_rows_radix_with_scratch::<1>, &x, &z, &c, "radix proptest w1");

            let x2: Vec<[u64; 2]> = rows
                .iter()
                .map(|r| [(r.0 % modulus) << spread, r.1 % modulus])
                .collect();
            let z2: Vec<[u64; 2]> = rows
                .iter()
                .map(|r| [(r.1 % modulus) << spread, r.0 % modulus])
                .collect();
            assert_sort_contract(sort_rows_radix_with_scratch::<2>, &x2, &z2, &c, "radix proptest w2");
        }

        /// Randomized shapes, including short runs, narrow key spaces and
        /// heavy duplication (the `% modulus` draw makes collisions common).
        #[test]
        fn sort_rows_with_scratch_contract_proptest(
            rows in prop::collection::vec((any::<u64>(), any::<u64>()), 0..300usize),
            modulus in 1u64..64,
        ) {
            let x: Vec<[u64; 1]> = rows.iter().map(|r| [r.0 % modulus]).collect();
            let z: Vec<[u64; 1]> = rows.iter().map(|r| [r.1 % modulus]).collect();
            let c: Vec<Complex64> = rows
                .iter()
                .map(|r| Complex64::new((r.0 % 11) as f64 - 5.0, (r.1 % 7) as f64 - 3.0))
                .collect();
            assert_sort_contract(sort_rows_with_scratch::<1>, &x, &z, &c, "proptest w1");

            let x2: Vec<[u64; 2]> = rows.iter().map(|r| [r.0 % modulus, r.1 % modulus]).collect();
            let z2: Vec<[u64; 2]> = rows.iter().map(|r| [r.1 % modulus, r.0 % modulus]).collect();
            assert_sort_contract(sort_rows_with_scratch::<2>, &x2, &z2, &c, "proptest w2");
        }
    }

    /// Truncation policy that always keeps terms — exercises the trait bound
    /// without filtering anything out.
    struct AlwaysKeep;
    impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

    // ---- `sort_rows_with_scratch` ----

    /// Sortedness on distinct keys. Lex on `(x, z)`: `I < Z < X` per word,
    /// since `x[0]` dominates.
    #[test]
    fn sort_rows_with_scratch_orders_by_lex_key() {
        let mut x: Vec<[u64; 1]> = vec![[1], [0], [0]];
        let mut z: Vec<[u64; 1]> = vec![[0], [0], [1]];
        let mut c: Vec<Complex64> = vec![
            Complex64::new(7.0, 0.0), // X
            Complex64::new(8.0, 0.0), // I
            Complex64::new(9.0, 0.0), // Z
        ];
        let mut scratch = SortScratch::<1>::default();
        sort_rows_with_scratch(&mut x, &mut z, &mut c, &mut scratch);
        assert_eq!(x, vec![[0u64], [0u64], [1u64]]);
        assert_eq!(z, vec![[0u64], [1u64], [0u64]]);
        assert_eq!(
            c,
            vec![
                Complex64::new(8.0, 0.0),
                Complex64::new(9.0, 0.0),
                Complex64::new(7.0, 0.0),
            ]
        );
    }

    /// Coefficient-permutation consistency across the word boundary: `x[0]`
    /// decides before `x[1]`, and a coefficient must follow its key through
    /// the permutation, not just land in the right count.
    #[test]
    fn sort_rows_with_scratch_keeps_coefficients_with_their_keys() {
        let mut x: Vec<[u64; 2]> = vec![[1, 0], [0, 99]];
        let mut z: Vec<[u64; 2]> = vec![[0, 0], [0, 0]];
        let mut c: Vec<Complex64> = vec![Complex64::new(11.0, 0.0), Complex64::new(22.0, 0.0)];
        let mut scratch = SortScratch::<2>::default();
        sort_rows_with_scratch(&mut x, &mut z, &mut c, &mut scratch);
        assert_eq!(x[0], [0, 99]);
        assert_eq!(c[0], Complex64::new(22.0, 0.0));
        assert_eq!(x[1], [1, 0]);
        assert_eq!(c[1], Complex64::new(11.0, 0.0));
    }

    /// Empty/single-row: `len < 2` is a no-op short-circuit.
    #[test]
    fn sort_rows_with_scratch_len_lt_2_is_noop() {
        let mut x: Vec<[u64; 1]> = vec![[5]];
        let mut z: Vec<[u64; 1]> = vec![[7]];
        let mut c: Vec<Complex64> = vec![Complex64::new(1.0, 2.0)];
        let mut scratch = SortScratch::<1>::default();
        sort_rows_with_scratch(&mut x, &mut z, &mut c, &mut scratch);
        assert_eq!(x[0], [5]);
        assert_eq!(z[0], [7]);
        assert_eq!(c[0], Complex64::new(1.0, 2.0));

        let mut empty_x: Vec<[u64; 1]> = vec![];
        let mut empty_z: Vec<[u64; 1]> = vec![];
        let mut empty_c: Vec<Complex64> = vec![];
        sort_rows_with_scratch(&mut empty_x, &mut empty_z, &mut empty_c, &mut scratch);
        assert!(empty_x.is_empty());
    }

    // ---- merge2_into: fused id/rest merge + reduction ----

    /// Plain single-stream segmented reduction over sorted columns: adjacent
    /// equal keys are summed, exact-zero sums are dropped, and `keep_term`
    /// sees the summed coefficient. Used only to build `merge2_reference`.
    fn reduce_sorted<const W: usize, T: TruncationPolicy<W> + ?Sized>(
        sorted_x: &[[u64; W]],
        sorted_z: &[[u64; W]],
        sorted_c: &[Complex64],
        policy: &T,
    ) -> (Vec<[u64; W]>, Vec<[u64; W]>, Vec<Complex64>) {
        let zero = Complex64::new(0.0, 0.0);
        let (mut ox, mut oz, mut oc) = (Vec::new(), Vec::new(), Vec::new());
        let end = sorted_c.len();
        let mut i = 0usize;
        while i < end {
            let (key_x, key_z) = (sorted_x[i], sorted_z[i]);
            let mut acc = sorted_c[i];
            let mut j = i + 1;
            while j < end && sorted_x[j] == key_x && sorted_z[j] == key_z {
                acc += sorted_c[j];
                j += 1;
            }
            if acc != zero && policy.keep_term(&key_x, &key_z, acc) {
                ox.push(key_x);
                oz.push(key_z);
                oc.push(acc);
            }
            i = j;
        }
        (ox, oz, oc)
    }

    /// Reference for `merge2_into`: concatenate both streams, sort by key,
    /// reduce. Coefficients in these tests are small integers, so `f64`
    /// addition is exact in any order and the comparison can be `==` even
    /// where the two pipelines sum in different orders.
    #[allow(clippy::type_complexity)]
    fn merge2_reference<const W: usize, T: TruncationPolicy<W> + ?Sized>(
        a: (&[[u64; W]], &[[u64; W]], &[Complex64]),
        b: (&[[u64; W]], &[[u64; W]], &[Complex64]),
        policy: &T,
    ) -> (Vec<[u64; W]>, Vec<[u64; W]>, Vec<Complex64>) {
        let mut rows: Vec<([u64; W], [u64; W], Complex64)> =
            a.0.iter()
                .zip(a.1)
                .zip(a.2)
                .map(|((&x, &z), &c)| (x, z, c))
                .chain(b.0.iter().zip(b.1).zip(b.2).map(|((&x, &z), &c)| (x, z, c)))
                .collect();
        rows.sort_by_key(|&(x, z, _)| (x, z));
        let (sx, sz, sc): (Vec<_>, Vec<_>, Vec<_>) =
            rows.into_iter()
                .fold((vec![], vec![], vec![]), |(mut x, mut z, mut c), r| {
                    x.push(r.0);
                    z.push(r.1);
                    c.push(r.2);
                    (x, z, c)
                });
        reduce_sorted(&sx, &sz, &sc, policy)
    }

    fn run_merge2<const W: usize, T: TruncationPolicy<W> + ?Sized>(
        a: (&[[u64; W]], &[[u64; W]], &[Complex64]),
        b: (&[[u64; W]], &[[u64; W]], &[Complex64]),
        policy: &T,
    ) -> (Vec<[u64; W]>, Vec<[u64; W]>, Vec<Complex64>) {
        let mut ox = vec![];
        let mut oz = vec![];
        let mut oc = vec![];
        merge2_into(
            a.0, a.1, a.2, b.0, b.1, b.2, &mut ox, &mut oz, &mut oc, policy,
        );
        (ox, oz, oc)
    }

    /// Randomized differential against the concat-sort-reduce reference:
    /// unique sorted id keys, rest with duplicates and cross-stream
    /// collisions, integer coefficients so any summation order is exact.
    #[test]
    fn merge2_matches_concat_sort_reduce() {
        // Tiny xorshift so the cases are deterministic without new deps.
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for case in 0..50 {
            // id: strictly ascending unique keys (sorted subset of 0..24).
            let mut id_keys: Vec<u64> = (0..24).filter(|_| next() % 2 == 0).collect();
            id_keys.dedup();
            let a_x: Vec<[u64; 1]> = id_keys.iter().map(|&k| [k]).collect();
            let a_z: Vec<[u64; 1]> = id_keys.iter().map(|&k| [k >> 1]).collect();
            let a_c: Vec<Complex64> = id_keys
                .iter()
                .map(|_| Complex64::new((next() % 7) as f64 - 3.0, (next() % 5) as f64 - 2.0))
                .collect();
            // rest: sorted, duplicates allowed, keys overlapping id's range.
            let mut rest_keys: Vec<u64> = (0..(next() % 40)).map(|_| next() % 24).collect();
            rest_keys.sort_unstable();
            let b_x: Vec<[u64; 1]> = rest_keys.iter().map(|&k| [k]).collect();
            let b_z: Vec<[u64; 1]> = rest_keys.iter().map(|&k| [k >> 1]).collect();
            let b_c: Vec<Complex64> = rest_keys
                .iter()
                .map(|_| Complex64::new((next() % 9) as f64 - 4.0, 0.0))
                .collect();

            let got = run_merge2((&a_x, &a_z, &a_c), (&b_x, &b_z, &b_c), &AlwaysKeep);
            let want = merge2_reference((&a_x, &a_z, &a_c), (&b_x, &b_z, &b_c), &AlwaysKeep);
            assert_eq!(got, want, "case {case} diverged from the reference");
        }
    }

    /// Both degenerate stream shapes: empty id (a channel with no identity
    /// delta) reduces to plain single-stream behavior; empty rest (a fully
    /// commuting coset) passes the unique id stream through the zero-drop
    /// and policy filters untouched.
    #[test]
    fn merge2_handles_empty_streams() {
        let x: Vec<[u64; 1]> = vec![[1], [2], [3]];
        let z: Vec<[u64; 1]> = vec![[0], [0], [1]];
        let c: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];
        let empty: (Vec<[u64; 1]>, Vec<[u64; 1]>, Vec<Complex64>) = (vec![], vec![], vec![]);

        let id_only = run_merge2((&x, &z, &c), (&empty.0, &empty.1, &empty.2), &AlwaysKeep);
        assert_eq!(id_only, (x.clone(), z.clone(), c.clone()));

        let rest_only = run_merge2((&empty.0, &empty.1, &empty.2), (&x, &z, &c), &AlwaysKeep);
        assert_eq!(rest_only, (x, z, c));
    }

    /// A cross-stream cancellation must drop the key entirely, and an
    /// exact-zero id coefficient (a `θ = π/2` rotation's `cos`-scaled row)
    /// must still participate: `-0.0 + 0.0 = +0.0` — pre-filtering zero rows
    /// would flip the sign of a zero sum against the single-stream pipeline.
    #[test]
    fn merge2_cancellation_and_signed_zero() {
        let a_x: Vec<[u64; 1]> = vec![[1], [2]];
        let a_z: Vec<[u64; 1]> = vec![[0], [0]];
        let a_c: Vec<Complex64> = vec![Complex64::new(-0.0, 0.0), Complex64::new(5.0, 0.0)];
        let b_x: Vec<[u64; 1]> = vec![[1], [2]];
        let b_z: Vec<[u64; 1]> = vec![[0], [0]];
        let b_c: Vec<Complex64> = vec![Complex64::new(0.0, 0.0), Complex64::new(-5.0, 0.0)];
        let (ox, _, oc) = run_merge2((&a_x, &a_z, &a_c), (&b_x, &b_z, &b_c), &AlwaysKeep);
        // Key [2]: exact cancellation, dropped. Key [1]: sums to +0.0 exactly
        // (the sign a zero-row prefilter would get wrong), which the zero-drop
        // then removes — matching the single-stream reduction on the
        // concatenated streams.
        assert!(ox.is_empty(), "got keys {ox:?} with coeffs {oc:?}");
    }

    /// `keep_term` sees the fully summed coefficient.
    #[test]
    fn merge2_policy_sees_summed_coefficient() {
        let a_x: Vec<[u64; 1]> = vec![[3]];
        let a_z: Vec<[u64; 1]> = vec![[0]];
        let a_c: Vec<Complex64> = vec![Complex64::new(0.04, 0.0)];
        let b_x: Vec<[u64; 1]> = vec![[3], [3]];
        let b_z: Vec<[u64; 1]> = vec![[0], [0]];
        let b_c: Vec<Complex64> = vec![Complex64::new(0.04, 0.0), Complex64::new(0.04, 0.0)];
        // Each row is below the 0.1 threshold; the sum (0.12) is above it.
        let policy = CoefficientThreshold(0.1);
        let (ox, _, oc) = run_merge2((&a_x, &a_z, &a_c), (&b_x, &b_z, &b_c), &policy);
        assert_eq!(ox, vec![[3u64]]);
        assert!(approx_eq(oc[0], Complex64::new(0.12, 0.0), TOL));
    }
}
