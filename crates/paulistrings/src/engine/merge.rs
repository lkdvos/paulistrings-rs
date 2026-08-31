//! Per-run sort and fused two-stream merge — the bucketed engine's inner
//! kernels.
//!
//! [`sort_rows_with_scratch`] canonicalizes one gather run's *rest* stream and
//! [`merge2_into`] fuses the id/rest two-stream merge with the segmented
//! reduction that restores the `PauliSum` invariant (strictly ascending, no
//! duplicates) inside a destination bucket. Both are called per gather run by
//! `engine::bucketed`; [`SortScratch`] is the worker-persistent scratch the
//! sort reuses so a steady-state layer allocates nothing.

use num_complex::Complex64;

use crate::truncation::TruncationPolicy;

/// Worker-persistent scratch for [`sort_rows_with_scratch`].
///
/// Held across coset tasks (one instance per `CosetScratch`, in turn one per
/// Rayon worker, per `bucketed.rs`'s `LayerScratch`): `perm` and the `tmp_*`
/// triple retain their high-water capacity across calls, so a run at or below
/// a previously-seen size sorts without allocating.
#[derive(Clone, Debug, Default)]
pub(crate) struct SortScratch<const W: usize> {
    perm: Vec<u32>,
    tmp_x: Vec<[u64; W]>,
    tmp_z: Vec<[u64; W]>,
    tmp_c: Vec<Complex64>,
}

impl<const W: usize> SortScratch<W> {
    /// Total heap capacity held across this scratch's buffers — a private
    /// implementation detail exposed only for
    /// `bucketed::tests::capacity_stabilizes_across_repeated_layers`, which
    /// needs it to confirm the sort scratch's footprint stops growing too.
    pub(crate) fn total_capacity(&self) -> usize {
        self.perm.capacity() + self.tmp_x.capacity() + self.tmp_z.capacity() + self.tmp_c.capacity()
    }
}

/// Sort `(x, z, c)` columns in place by the key `(x, z)` alone, using `s` as
/// reusable scratch.
///
/// v0.5 S1 policy: equal-key summation order is no longer required to be
/// bucket-count- or hash-seed-independent (floating-point associativity
/// variation across those axes is accepted), so the `u8` delta tag that used
/// to break ties in `local_delta` order (the deleted `sort_phase_tagged`) is
/// gone, and this sort compares the key alone — cheaper, and with one fewer
/// column to carry through the gather.
///
/// The sort is the **stable** `sort_by`, but not for stability (nothing
/// depends on equal-key order any more — an unstable sort would be
/// semantically fine): it is for *adaptivity*. A gather run is a
/// concatenation of per-delta streams, each drawn from one sorted source
/// bucket — the identity stream arrives fully sorted, and an XOR-by-constant
/// stream is piecewise sorted (order survives wherever the mask's high bits
/// don't flip) — and Rust's stable driftsort detects and merges those natural
/// ascending runs while the unstable pdqsort does not. Measured (v0.5 S1
/// fix): switching this line to `sort_unstable_by` cost +77% on a 10⁶
/// `rotation_zz` layer and +43% on CNOT.
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
/// not two — the v0.3 `sort_phase_tagged` built a `Vec<usize>` perm and then a
/// separate `collect` + `copy_from_slice` round trip per column), and finally
/// each `tmp_*` is `mem::swap`ped with the caller's `Vec`. The caller ends up
/// holding the sorted columns; `s` ends up holding the caller's pre-sort
/// columns' storage (cleared next call) as its own scratch capacity — so
/// capacity circulates between the live columns and the scratch instead of
/// either side ever growing past its high-water mark.
// `#[inline]` is load-bearing: without it, moving this function between
// modules measured ~6% slower single-threaded on the rotation family
// (interleaved A/B, 3/3 pairs) — an LTO code-layout effect, not logic.
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

/// Fused two-stream merge + segmented reduction (v0.5 S2).
///
/// `a` is a gather run's identity-delta stream: its keys are untouched source
/// keys, so it inherits the bucket invariant — strictly ascending, no
/// duplicates — and is **never sorted**. (Under a dense identity plan the
/// key slices are the *source bucket's own columns*, borrowed in place, with
/// only the coefficients gathered — v0.6 G1d; this function cannot tell and
/// need not care.) `b` is the run's remaining rows, canonicalized by
/// [`sort_rows_with_scratch`] (ascending, duplicates allowed).
/// The two-pointer walk consumes rows in global key order, seeding a key tie
/// from the `a` row first and then adding the equal-key `b` rows in their
/// sorted order; that order is deterministic for a fixed input but, per the
/// v0.5 S1 policy, not specified across partitions. Zero-drop and `keep_term`
/// see the fully summed coefficient. When `a` is empty this degenerates to the
/// plain single-stream segmented reduction, which is the whole story for a
/// channel with no identity delta: everything is gathered into `b`.
///
/// Exact-zero rows are consumed like any other (a `θ = π/2` rotation emits
/// `cos·coeff = ±0.0` rows): dropping them *before* the reduction could flip
/// the sign of a zero sum, so the only zero test is on the final accumulator.
///
/// Do not restructure this walk into gallop + bulk segment copies (v0.6 M1):
/// measured +20–35% merge busy on every real cell except 1t trotter, because
/// the workloads' id/rest densities make the average id segment one or two
/// rows (gu2q: mostly empty) — per-segment overhead swamps the per-row
/// compare it saves. Full data in the 2026-08-31 v0.6 results note.
#[allow(clippy::too_many_arguments)]
// `#[inline]` is load-bearing — same A/B-verified layout effect as on
// `sort_rows_with_scratch` above.
#[inline]
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
    use crate::truncation::CoefficientThreshold;

    const TOL: f64 = 1e-12;

    fn approx_eq(a: Complex64, b: Complex64, tol: f64) -> bool {
        (a - b).norm() <= tol
    }

    /// Truncation policy that always keeps terms — exercises the trait bound
    /// without filtering anything out.
    struct AlwaysKeep;
    impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

    // ---- `sort_rows_with_scratch` (v0.5 S1) ----

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

    // ---- merge2_into (v0.5 S2): fused id/rest merge + reduction ----

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
