use super::*;

#[test]
fn the_fingerprint_separates_every_field_it_covers() {
    let dense = Gf2Hash::<1>::new(8, 3, 7);
    let fingerprint = |channels, direction, options, num_qubits, w, hash: &Gf2Hash<1>| {
        run_fingerprint(channels, direction, options, num_qubits, w, hash)
    };
    let default = PropagateOptions::default();
    let base = fingerprint(3, Direction::Forward, default, 8, 1, &dense);
    assert_ne!(base, fingerprint(4, Direction::Forward, default, 8, 1, &dense));
    assert_ne!(base, fingerprint(3, Direction::Heisenberg, default, 8, 1, &dense));
    assert_ne!(base, fingerprint(3, Direction::Forward, default, 9, 1, &dense));
    assert_ne!(base, fingerprint(3, Direction::Forward, default, 8, 2, &dense));
    let mut options = PropagateOptions::default();
    options.target_bucket_len += 1;
    assert_ne!(base, fingerprint(3, Direction::Forward, options, 8, 1, &dense));
    let mut options = PropagateOptions::default();
    options.min_buckets += 1;
    assert_ne!(base, fingerprint(3, Direction::Forward, options, 8, 1, &dense));
    for other in [
        Gf2Hash::<1>::new(8, 3, 8),
        Gf2Hash::with_family(8, 3, 7, HashFamily::Projection),
        Gf2Hash::with_family(8, 3, 7, HashFamily::Sparse { weight: 3 }),
        Gf2Hash::with_family(8, 3, 7, HashFamily::Sparse { weight: 5 }),
    ] {
        assert_ne!(
            base,
            fingerprint(3, Direction::Forward, default, 8, 1, &other),
            "{:?}",
            other.origin()
        );
    }
    assert_eq!(
        base,
        fingerprint(3, Direction::Forward, default, 8, 1, &Gf2Hash::<1>::new(8, 5, 7)),
        "the bucket count is not part of it"
    );
}
