use super::*;
use crate::channel::prepared::Prepare;
use crate::channel::rotation::PauliRotation;
use crate::engine::plan::PartitionPlan;
use crate::pauli_sum::hash::PartitionRows;
use crate::pauli_sum::PauliSum;
use crate::truncation::octave_window::BoundaryCut;
use num_complex::Complex64;

/// A NaN coefficient is exported with no cut pending and dropped under one, as `BoundaryCut::filter` does.
#[test]
fn a_nan_term_is_counted_only_without_a_cut() {
    let rotation = PauliRotation::new(PauliString::<1>::x(0), 0.3);
    let coefficients = [f64::NAN, 1.0, 0.5, 0.25];
    let sum = PauliSum::<1>::from_sorted_columns(
        vec![[0u64]; coefficients.len()],
        (0u64..coefficients.len() as u64).map(|i| [1 | (i << 1)]).collect(),
        coefficients.iter().map(|&c| Complex64::new(c, 0.0)).collect(),
        4,
    );
    assert_eq!(sum.num_buckets(), 1);
    let prepared = Prepare::<1>::prepare(&rotation, sum.hash(), false).expect("prepare");
    let rows = PartitionRows::<1>::from_rows(4, vec![[1u64]], vec![[0u64]]);
    let plan = PartitionPlan::new(&prepared, &rows, 0);
    let [remote] = plan.remote.as_slice() else { panic!("the generator pass must be remote") };
    let bin = octave(0.25);
    let window = OctaveWindow::around(bin);
    let (x, z, coeff) = sum.bucket(0);
    let columns = BucketColumns { x: x.to_vec(), z: z.to_vec(), coeff: coeff.to_vec() };
    let mut counts = [0; WINDOW_OCTAVES + 2];
    let mut ahead = BucketAhead::default();
    window.count_emitting(
        &columns,
        &RowEmitter::of(&prepared, remote),
        Some(&mut counts),
        &mut ahead,
    );
    let mut octaves = LayerOctaves::default();
    octaves.per_bucket.push(counts);
    octaves.ahead.push(ahead);
    octaves.ahead_window = Some(window);
    assert_eq!(octaves.rows_ahead(0), Some(4));
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
