use super::*;

#[test]
fn the_fingerprint_separates_every_field_it_covers() {
    let base = run_fingerprint(3, Direction::Forward, PropagateOptions::default(), 8, 1);
    assert_ne!(
        base,
        run_fingerprint(4, Direction::Forward, PropagateOptions::default(), 8, 1),
    );
    assert_ne!(
        base,
        run_fingerprint(3, Direction::Heisenberg, PropagateOptions::default(), 8, 1),
    );
    assert_ne!(
        base,
        run_fingerprint(3, Direction::Forward, PropagateOptions::default(), 9, 1),
    );
    assert_ne!(
        base,
        run_fingerprint(3, Direction::Forward, PropagateOptions::default(), 8, 2),
    );
    let mut options = PropagateOptions::default();
    options.target_bucket_len += 1;
    assert_ne!(base, run_fingerprint(3, Direction::Forward, options, 8, 1),);
    let mut options = PropagateOptions::default();
    options.min_buckets += 1;
    assert_ne!(base, run_fingerprint(3, Direction::Forward, options, 8, 1),);
    // And it is a function of its inputs, not of the call.
    assert_eq!(
        base,
        run_fingerprint(3, Direction::Forward, PropagateOptions::default(), 8, 1),
    );
}
