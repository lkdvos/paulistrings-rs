use super::*;
use crate::channel::prepared::Prepare;
use crate::channel::rotation::PauliRotation;
use crate::channel::Channel;
use crate::engine::partitioned::in_process::InProcessTransport;
use crate::engine::plan::PartitionPlan;
use crate::pauli_sum::hash::{Gf2Hash, PartitionRows};
use crate::pauli_sum::PauliSum;
use crate::test_support::{differential_channels_w1, differential_channels_w2, tie_heavy_sum};
use crate::truncation::octave_window::{decide_from_window, BoundaryCut};
use num_complex::Complex64;

/// Bucket `b`'s terms that `keep` keeps and `test` emits for, counted directly.
fn counted<const W: usize>(
    sum: &PauliSum<W>,
    b: usize,
    test: &EmitTest<'_, W>,
    mut keep: impl FnMut(&[u64; W], &[u64; W], Complex64) -> bool,
) -> u32 {
    let (x, z, coeff) = sum.bucket(b);
    (0..coeff.len())
        .map(|row| u32::from(keep(&x[row], &z[row], coeff[row]) & test.emits(&x[row], &z[row])))
        .sum()
}

/// `octaves` as a coset loop leaves them after writing `sum` under `window`, with counts ahead for `test`.
fn counted_ahead<const W: usize>(
    sum: &PauliSum<W>,
    window: OctaveWindow,
    test: &EmitTest<'_, W>,
) -> LayerOctaves {
    let mut octaves = LayerOctaves::default();
    for b in 0..sum.num_buckets() {
        let (x, z, coeff) = sum.bucket(b);
        let columns = BucketColumns { x: x.to_vec(), z: z.to_vec(), coeff: coeff.to_vec() };
        let mut counts = [0; WINDOW_OCTAVES + 2];
        let mut ahead = BucketAhead::default();
        window.count_emitting(&columns, test, Some(&mut counts), &mut ahead);
        let mut plain = [0; WINDOW_OCTAVES + 2];
        window.count(&columns.coeff, &mut plain);
        assert_eq!(counts, plain, "bucket {b}: the fused count bins as `count` does");
        let emitting: Vec<usize> =
            (0..x.len()).filter(|&row| test.emits(&x[row], &z[row])).collect();
        assert_eq!(ahead.marked_rows().collect::<Vec<_>>(), emitting, "bucket {b}: marks");
        octaves.per_bucket.push(counts);
        octaves.ahead.push(ahead);
    }
    octaves.ahead_window = Some(window);
    octaves
}

/// Under no cut and under a cut at every `n`, the counts ahead give each bucket's exported rows among the survivors, or `None` at one bucket at most, the one the boundary quota splits.
fn assert_rows_ahead_count_the_survivors<const W: usize>(
    sum: &PauliSum<W>,
    channels: &[(&'static str, Box<dyn Channel<W>>)],
) {
    let rows = PartitionRows::<W>::from_seed(sum.num_qubits(), 1, 0x1234);
    let solo = InProcessTransport::group(1).pop().expect("one partition");
    let window = OctaveWindow::around(crate::truncation::builtin::octave(0.01));
    let mut cuts = 0;
    for (name, channel) in channels {
        let prepared = channel.prepare(sum.hash(), false).expect("prepare");
        let plan = PartitionPlan::new(&prepared, &rows, 0);
        for remote in &plan.remote {
            let test = EmitTest::of(&prepared, remote);
            let mut octaves = counted_ahead(sum, window, &test);
            for b in 0..sum.num_buckets() {
                assert_eq!(
                    octaves.rows_ahead(b),
                    Some(counted(sum, b, &test, |_, _, _| true)),
                    "{name} b={b}"
                );
            }
            for n in [1usize, sum.len() / 3, sum.len() / 2, sum.len() - 1] {
                let Some((_, _, Some(cut))) =
                    decide_from_window(sum.len(), &solo, n, window, &octaves.per_bucket)
                else {
                    continue;
                };
                cuts += 1;
                octaves.defer(cut.clone());
                let mut unanswered = 0;
                for b in 0..sum.num_buckets() {
                    let want = counted(sum, b, &test, cut.filter(b));
                    match octaves.rows_ahead(b) {
                        Some(rows) => assert_eq!(rows, want, "{name} n={n} b={b}"),
                        None => unanswered += 1,
                    }
                }
                assert!(unanswered <= 1, "{name} n={n}: {unanswered} buckets unanswered");
                octaves.pending = None;
            }
        }
    }
    assert!(cuts > 0, "no `n` cut inside the window");
}

#[test]
fn rows_ahead_count_the_survivors_w1() {
    let sum = tie_heavy_sum::<1>(3000, 8, 0xA4E1).with_hash(Gf2Hash::<1>::new(8, 5, 0xAB));
    assert_rows_ahead_count_the_survivors(&sum, &differential_channels_w1());
}

#[test]
fn rows_ahead_count_the_survivors_w2() {
    let sum = tie_heavy_sum::<2>(3000, 128, 0xA4E2).with_hash(Gf2Hash::<2>::new(128, 5, 0xAB));
    assert_rows_ahead_count_the_survivors(&sum, &differential_channels_w2());
}

/// A NaN coefficient is exported with no cut pending and dropped under one, as `BoundaryCut::filter` does.
#[test]
fn a_nan_term_is_counted_only_without_a_cut() {
    let generator = PauliString::<1>::x(0);
    let rotation = PauliRotation::new(generator, 0.3);
    let coefficients = [f64::NAN, 1.0, 0.5, 0.25];
    let sum = PauliSum::<1>::from_sorted_columns(
        vec![[0u64]; coefficients.len()],
        (0u64..coefficients.len() as u64).map(|i| [1 | (i << 1)]).collect(),
        coefficients.iter().map(|&c| Complex64::new(c, 0.0)).collect(),
        4,
    );
    let prepared = Prepare::<1>::prepare(&rotation, sum.hash(), false).expect("prepare");
    let rows = PartitionRows::<1>::from_rows(4, vec![[1u64]], vec![[0u64]]);
    let plan = PartitionPlan::new(&prepared, &rows, 0);
    let [remote] = plan.remote.as_slice() else { panic!("the generator pass must be remote") };
    let test = EmitTest::of(&prepared, remote);
    let window = OctaveWindow::around(crate::truncation::builtin::octave(0.25));
    let mut octaves = counted_ahead(&sum, window, &test);
    assert_eq!(octaves.rows_ahead(0), Some(4));
    let bin = crate::truncation::builtin::octave(0.25);
    octaves.defer(BoundaryCut { bin, quotas: vec![1], kept: 3 });
    assert_eq!(octaves.rows_ahead(0), Some(2));
}

/// A bucket of three terms, the first and third emitting, both in slot 3.
#[cfg(debug_assertions)]
fn two_of_three_marked() -> BucketAhead {
    let mut ahead = BucketAhead::default();
    ahead.emitting[3] = 2;
    ahead.marks = vec![0b101];
    ahead
}

#[cfg(debug_assertions)]
#[test]
fn marks_one_per_counted_term_pass_the_debug_check() {
    two_of_three_marked().debug_assert_marks_match(3, 0);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "bucket 7: 1 rows marked where the counts ahead count 2")]
fn a_missing_mark_fails_the_debug_check() {
    let mut ahead = two_of_three_marked();
    ahead.marks = vec![0b001];
    ahead.debug_assert_marks_match(3, 7);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "bucket 7: 1 mark words for a bucket of 65 terms")]
fn marks_for_another_bucket_length_fail_the_debug_check() {
    two_of_three_marked().debug_assert_marks_match(65, 7);
}
