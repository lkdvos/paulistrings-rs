use super::*;
use crate::test_support::approx_eq;
use crate::truncation::CoefficientThreshold;
use proptest::prelude::*;

const TOL: f64 = 1e-12;

// ---- the per-run sort kernel's contract ----
//
// Every kernel `bucketed.rs` may pick for a gather run's rest stream must satisfy exactly this, and nothing more: the output is **ascending** in lex `(x, z)` (duplicates allowed — `merge2_into` reduces them) and is a permutation of the input `(x, z, c)` triples, so a coefficient still travels with its own key.
// Equal-key order is explicitly *not* pinned (ARCHITECTURE.md §Determinism), which is why the check is a multiset comparison rather than an element-wise one.

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
    // Multiset of triples, with the coefficient bits as the tiebreak so the comparison is exact and order-insensitive.
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

/// Xorshift64 — local so the fixtures below need no dev-dependency draw order shared with `test_support`.
fn xs64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// The shapes a real gather run takes, plus the degenerate ones a kernel that looks at the key *bits* (rather than only comparing keys) can trip over.
/// `(label, x, z, c)`.
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

    // Heavy duplicates: the dense-PTM shape.
    // 40 distinct keys, each repeated 15 times, the repeats interleaved as 15 sorted streams.
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

    // `x` identically zero: all discrimination lives in `z`, so a kernel that keys off the most significant word must walk past `x`.
    let zero_x: Vec<([u64; W], [u64; W])> = distinct.iter().map(|k| ([0u64; W], k.1)).collect();
    push("x_all_zero", zero_x, 0x27);

    // Two-bit key space: only bits 0 and 1 of `x[0]` vary, so any fixed-width surrogate window has almost no discriminating power.
    let thin: Vec<([u64; W], [u64; W])> = (0..200u64)
        .map(|i| {
            let mut kx = [0u64; W];
            kx[0] = (i % 4) & mask(0);
            (kx, [0u64; W])
        })
        .collect();
    push("thin_window", thin, 0x28);

    // A constant *nonzero* high part above the varying bits — the case where masking a shifted window must not reorder rows.
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

/// The radix kernel satisfies the *same* contract on every shape — the point of the harness.
/// Includes the two shapes that must reach its fallbacks: `thin_window` (fewer than `RADIX_MIN_WINDOW_BITS` discriminating bits) and `all_equal` (no discriminating word at all).
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
    // Narrow qubit counts put every discriminating bit low in word 0, so the window shift saturates at 0 and the mask covers the whole word.
    for q in [3usize, 8, 17, 33] {
        for (label, x, z, c) in sort_fixtures::<1>(q) {
            assert_sort_contract(sort_rows_radix_with_scratch::<1>, &x, &z, &c, &label);
        }
    }
}

/// Both kernels agree on the *reduced* content of every fixture: same keys in the same order, and equal-key coefficient sums that agree exactly (the fixtures' coefficients are small integers, so any summation order is exact).
/// This is the interchangeability claim `bucketed.rs` relies on when it picks a kernel per layer.
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

/// A steady-state layer must not allocate: the radix kernel's own buffers have to stop growing once the largest run has been seen.
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
    /// Randomized shapes, including short runs, narrow key spaces and heavy duplication (the `% modulus` draw makes collisions common).
    #[test]
    fn sort_rows_radix_contract_proptest(
        rows in prop::collection::vec((any::<u64>(), any::<u64>()), 0..300usize),
        modulus in 1u64..64,
        spread in 0u32..60,
    ) {
        // `spread` slides the varying bits up and down word 0, exercising every window shift including the saturating one.
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

    /// Randomized shapes, including short runs, narrow key spaces and heavy duplication (the `% modulus` draw makes collisions common).
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

/// Truncation policy that always keeps terms — exercises the trait bound without filtering anything out.
struct AlwaysKeep;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

// ---- `sort_rows_with_scratch` ----

/// Sortedness on distinct keys. Lex on `(x, z)`: `I < Z < X` per word, since `x[0]` dominates.
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

/// Coefficient-permutation consistency across the word boundary: `x[0]` decides before `x[1]`, and a coefficient must follow its key through the permutation, not just land in the right count.
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

/// Plain single-stream segmented reduction over sorted columns: adjacent equal keys are summed, exact-zero sums are dropped, and `keep_term` sees the summed coefficient.
/// Used only to build `merge2_reference`.
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

/// Reference for `merge2_into`: concatenate both streams, sort by key, reduce.
/// Coefficients in these tests are small integers, so `f64` addition is exact in any order and the comparison can be `==` even where the two pipelines sum in different orders.
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

/// Randomized differential against the concat-sort-reduce reference: unique sorted id keys, rest with duplicates and cross-stream collisions, integer coefficients so any summation order is exact.
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

/// Both degenerate stream shapes: empty id (a channel with no identity delta) reduces to plain single-stream behavior; empty rest (a fully commuting coset) passes the unique id stream through the zero-drop and policy filters untouched.
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

/// A cross-stream cancellation must drop the key entirely, and an exact-zero id coefficient (a `θ = π/2` rotation's `cos`-scaled row) must still participate: `-0.0 + 0.0 = +0.0` — pre-filtering zero rows would flip the sign of a zero sum against the single-stream pipeline.
#[test]
fn merge2_cancellation_and_signed_zero() {
    let a_x: Vec<[u64; 1]> = vec![[1], [2]];
    let a_z: Vec<[u64; 1]> = vec![[0], [0]];
    let a_c: Vec<Complex64> = vec![Complex64::new(-0.0, 0.0), Complex64::new(5.0, 0.0)];
    let b_x: Vec<[u64; 1]> = vec![[1], [2]];
    let b_z: Vec<[u64; 1]> = vec![[0], [0]];
    let b_c: Vec<Complex64> = vec![Complex64::new(0.0, 0.0), Complex64::new(-5.0, 0.0)];
    let (ox, _, oc) = run_merge2((&a_x, &a_z, &a_c), (&b_x, &b_z, &b_c), &AlwaysKeep);
    // Key [2]: exact cancellation, dropped.
    // Key [1]: sums to +0.0 exactly (the sign a zero-row prefilter would get wrong), which the zero-drop then removes — matching the single-stream reduction on the concatenated streams.
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

impl<const W: usize> SortScratch<W> {
    /// Total heap capacity held across this scratch's buffers.
    /// Exposed only for `bucketed::tests::capacity_stabilizes_across_repeated_layers`.
    pub(crate) fn total_capacity(&self) -> usize {
        self.perm.capacity()
            + self.packed.capacity()
            + self.aux.capacity()
            + self.tmp_x.capacity()
            + self.tmp_z.capacity()
            + self.tmp_c.capacity()
    }
}
