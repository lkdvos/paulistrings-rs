use super::*;

#[test]
fn the_fingerprint_separates_every_field_it_covers() {
    let dense = Gf2Hash::<1>::new(8, 3, 7);
    let rows = PartitionRows::<1>::from_seed(8, 1, 7);
    let fingerprint = |channels, direction, options, num_qubits, w, hash: &Gf2Hash<1>, rows| {
        run_fingerprint(channels, direction, options, num_qubits, w, hash, rows)
    };
    let default = PropagateOptions::default();
    let base = fingerprint(3, Direction::Forward, default, 8, 1, &dense, &rows);
    assert_ne!(base, fingerprint(4, Direction::Forward, default, 8, 1, &dense, &rows));
    assert_ne!(base, fingerprint(3, Direction::Heisenberg, default, 8, 1, &dense, &rows));
    assert_ne!(base, fingerprint(3, Direction::Forward, default, 9, 1, &dense, &rows));
    assert_ne!(base, fingerprint(3, Direction::Forward, default, 8, 2, &dense, &rows));
    let mut options = PropagateOptions::default();
    options.target_bucket_len += 1;
    assert_ne!(base, fingerprint(3, Direction::Forward, options, 8, 1, &dense, &rows));
    let mut options = PropagateOptions::default();
    options.min_buckets += 1;
    assert_ne!(base, fingerprint(3, Direction::Forward, options, 8, 1, &dense, &rows));
    let other_hash = Gf2Hash::<1>::new(8, 3, 8);
    assert_ne!(base, fingerprint(3, Direction::Forward, default, 8, 1, &other_hash, &rows));
    let other_rows = PartitionRows::<1>::from_rows(8, vec![[0b11]], vec![[0]]);
    assert_ne!(base, fingerprint(3, Direction::Forward, default, 8, 1, &dense, &other_rows));
    let same_rows =
        PartitionRows::<1>::from_rows(8, rows.rows().0.to_vec(), rows.rows().1.to_vec());
    assert_eq!(
        base,
        fingerprint(3, Direction::Forward, default, 8, 1, &Gf2Hash::<1>::new(8, 5, 7), &same_rows),
        "the bucket count is not part of it, and rows decide sameness"
    );
}
