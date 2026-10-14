use super::*;

// ---- the per-run sort kernel's contract ----
//
// Output ascending in lex `(x, z)` with duplicates allowed, as a permutation of the input triples; equal-key order is unpinned, so the check is a multiset comparison.

type SortFn<const W: usize> = fn(&mut BucketColumns<W>, &mut SortScratch<W>);

/// Owned columns from slices.
fn columns<const W: usize>(x: &[[u64; W]], z: &[[u64; W]], c: &[Complex64]) -> BucketColumns<W> {
    BucketColumns {
        x: x.to_vec(),
        z: z.to_vec(),
        coeff: c.to_vec(),
    }
}

/// Assert the contract above for `kernel` on one run.
fn assert_sort_contract<const W: usize>(
    kernel: SortFn<W>,
    x: &[[u64; W]],
    z: &[[u64; W]],
    c: &[Complex64],
    what: &str,
) {
    let mut sorted = columns(x, z, c);
    let mut scratch = SortScratch::<W>::default();
    kernel(&mut sorted, &mut scratch);
    let BucketColumns {
        x: gx,
        z: gz,
        coeff: gc,
    } = sorted;

    assert_eq!(gx.len(), x.len(), "{what}: row count changed");
    assert_eq!(gz.len(), z.len(), "{what}: row count changed");
    assert_eq!(gc.len(), c.len(), "{what}: row count changed");
    for i in 1..gx.len() {
        assert!(
            (gx[i - 1], gz[i - 1]) <= (gx[i], gz[i]),
            "{what}: not ascending at row {i}",
        );
    }
    // Coefficient bits break ties, so the multiset comparison is exact.
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

/// Xorshift64, local so the fixtures share no draw order with `test_support`.
fn xs64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Real gather-run shapes plus degenerate ones for a kernel that reads key bits, as `(label, x, z, c)`.
#[allow(clippy::type_complexity)]
fn sort_fixtures<const W: usize>(
    num_qubits: usize,
) -> Vec<(String, Vec<[u64; W]>, Vec<[u64; W]>, Vec<Complex64>)> {
    let mut out = Vec::new();
    let mask = |w: usize| crate::test_support::word_mask(num_qubits, w);
    let mut push = |label: &str, keys: Vec<([u64; W], [u64; W])>, seed: u64| {
        let mut state = seed | 1;
        let c: Vec<Complex64> = keys
            .iter()
            .map(|_| {
                Complex64::new(
                    (xs64(&mut state) % 17) as f64 - 8.0,
                    (xs64(&mut state) % 13) as f64 - 6.0,
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
    let mut state = 0xC0FF_EE00_1234_5678u64;
    let mut keys: Vec<([u64; W], [u64; W])> = (0..400)
        .map(|_| {
            let mut kx = [0u64; W];
            let mut kz = [0u64; W];
            for w in 0..W {
                kx[w] = xs64(&mut state) & mask(w);
                kz[w] = xs64(&mut state) & mask(w);
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
    // 40 distinct keys, each repeated 15 times as 15 sorted streams.
    let mut state = 0x5EED_0000_0000_0001u64;
    let mut distinct: Vec<([u64; W], [u64; W])> = (0..40)
        .map(|_| {
            let mut kx = [0u64; W];
            let mut kz = [0u64; W];
            for w in 0..W {
                kx[w] = xs64(&mut state) & mask(w);
                kz[w] = xs64(&mut state) & mask(w);
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

    // A constant nonzero high part above the varying bits, which masking a shifted window must not reorder.
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

/// The radix kernel satisfies the same contract, including the `thin_window` and `all_equal` fallbacks.
#[test]
fn sort_by_radix_honors_the_kernel_contract() {
    for (label, x, z, c) in sort_fixtures::<1>(64) {
        assert_sort_contract(sort_by_radix::<1>, &x, &z, &c, &label);
    }
    for (label, x, z, c) in sort_fixtures::<2>(128) {
        assert_sort_contract(sort_by_radix::<2>, &x, &z, &c, &label);
    }
    for (label, x, z, c) in sort_fixtures::<2>(65) {
        assert_sort_contract(sort_by_radix::<2>, &x, &z, &c, &label);
    }
    // Narrow qubit counts saturate the window shift at 0.
    for q in [3usize, 8, 17, 33] {
        for (label, x, z, c) in sort_fixtures::<1>(q) {
            assert_sort_contract(sort_by_radix::<1>, &x, &z, &c, &label);
        }
    }
}

/// Both kernels agree exactly on the reduced content of every fixture, whose integer coefficients sum exactly in any order.
#[test]
fn the_two_sort_kernels_reduce_to_the_same_sum() {
    #[allow(clippy::type_complexity)]
    fn reduced<const W: usize>(
        kernel: SortFn<W>,
        x: &[[u64; W]],
        z: &[[u64; W]],
        c: &[Complex64],
    ) -> Vec<([u64; W], [u64; W], Complex64)> {
        let mut sorted = columns(x, z, c);
        let mut scratch = SortScratch::<W>::default();
        kernel(&mut sorted, &mut scratch);
        let empty = ColumnsRef {
            x: &[],
            z: &[],
            coeff: &[],
        };
        let mut out = BucketColumns::default();
        merge_identity_and_rest(empty, sorted.view(), &mut out, &AlwaysKeep);
        out.x
            .into_iter()
            .zip(out.z)
            .zip(out.coeff)
            .map(|((a, b), v)| (a, b, v))
            .collect()
    }
    for (label, x, z, c) in sort_fixtures::<1>(64) {
        assert_eq!(
            reduced(sort_by_radix::<1>, &x, &z, &c),
            reduced(sort_by_comparison::<1>, &x, &z, &c),
            "{label}",
        );
    }
    for (label, x, z, c) in sort_fixtures::<2>(128) {
        assert_eq!(
            reduced(sort_by_radix::<2>, &x, &z, &c),
            reduced(sort_by_comparison::<2>, &x, &z, &c),
            "{label}",
        );
    }
}

/// The radix kernel's buffers stop growing once the largest run has been seen.
#[test]
fn radix_sort_scratch_capacity_stabilizes() {
    let mut scratch = SortScratch::<2>::default();
    let fixtures = sort_fixtures::<2>(128);
    let run = |s: &mut SortScratch<2>| {
        for (_, x, z, c) in fixtures.iter() {
            sort_by_radix(&mut columns(x, z, c), s);
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

/// Keeps every term.
struct AlwaysKeep;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

// ---- merge_identity_and_rest: fused id/rest merge + reduction ----

/// Single-stream segmented reduction over sorted columns, for `merge_reference`.
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
        let mut accumulator = sorted_c[i];
        let mut j = i + 1;
        while j < end && sorted_x[j] == key_x && sorted_z[j] == key_z {
            accumulator += sorted_c[j];
            j += 1;
        }
        if accumulator != zero && policy.keep_term(&key_x, &key_z, accumulator) {
            ox.push(key_x);
            oz.push(key_z);
            oc.push(accumulator);
        }
        i = j;
    }
    (ox, oz, oc)
}

/// Reference for `merge_identity_and_rest`: concatenate, sort, reduce; integer coefficients make `==` valid.
#[allow(clippy::type_complexity)]
fn merge_reference<const W: usize, T: TruncationPolicy<W> + ?Sized>(
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
    let (sx, sz, scratch): (Vec<_>, Vec<_>, Vec<_>) =
        rows.into_iter()
            .fold((vec![], vec![], vec![]), |(mut x, mut z, mut c), r| {
                x.push(r.0);
                z.push(r.1);
                c.push(r.2);
                (x, z, c)
            });
    reduce_sorted(&sx, &sz, &scratch, policy)
}

fn run_merge<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    a: (&[[u64; W]], &[[u64; W]], &[Complex64]),
    b: (&[[u64; W]], &[[u64; W]], &[Complex64]),
    policy: &T,
) -> (Vec<[u64; W]>, Vec<[u64; W]>, Vec<Complex64>) {
    let identity = ColumnsRef {
        x: a.0,
        z: a.1,
        coeff: a.2,
    };
    let rest = ColumnsRef {
        x: b.0,
        z: b.1,
        coeff: b.2,
    };
    let mut out = BucketColumns::default();
    merge_identity_and_rest(identity, rest, &mut out, policy);
    (out.x, out.z, out.coeff)
}

/// Randomized differential against the concat-sort-reduce reference, with cross-stream collisions.
#[test]
fn merge_matches_concat_sort_reduce() {
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

        let got = run_merge((&a_x, &a_z, &a_c), (&b_x, &b_z, &b_c), &AlwaysKeep);
        let want = merge_reference((&a_x, &a_z, &a_c), (&b_x, &b_z, &b_c), &AlwaysKeep);
        assert_eq!(got, want, "case {case} diverged from the reference");
    }
}

/// A cross-stream cancellation drops the key, and an exact-zero id row still participates: `-0.0 + 0.0 = +0.0`.
#[test]
fn merge_cancellation_and_signed_zero() {
    let a_x: Vec<[u64; 1]> = vec![[1], [2]];
    let a_z: Vec<[u64; 1]> = vec![[0], [0]];
    let a_c: Vec<Complex64> = vec![Complex64::new(-0.0, 0.0), Complex64::new(5.0, 0.0)];
    let b_x: Vec<[u64; 1]> = vec![[1], [2]];
    let b_z: Vec<[u64; 1]> = vec![[0], [0]];
    let b_c: Vec<Complex64> = vec![Complex64::new(0.0, 0.0), Complex64::new(-5.0, 0.0)];
    let (ox, _, oc) = run_merge((&a_x, &a_z, &a_c), (&b_x, &b_z, &b_c), &AlwaysKeep);
    // Key [2]: exact cancellation, dropped.
    // Key [1]: sums to +0.0 exactly, which the zero-drop then removes.
    assert!(ox.is_empty(), "got keys {ox:?} with coeffs {oc:?}");
}

impl<const W: usize> SortScratch<W> {
    /// Total heap capacity held across this scratch's buffers.
    pub(crate) fn total_capacity(&self) -> usize {
        self.permutation.capacity()
            + self.packed.capacity()
            + self.spare.capacity()
            + self.staging_x.capacity()
            + self.staging_z.capacity()
            + self.staging_coefficient.capacity()
    }
}
