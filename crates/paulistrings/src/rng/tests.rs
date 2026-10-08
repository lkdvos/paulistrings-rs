use super::*;

/// The first output of splitmix64 from state 0 is the published reference value.
#[test]
fn splitmix64_matches_the_reference_value() {
    let mut state = 0u64;
    assert_eq!(splitmix64(&mut state), 0xE220_A839_7B1D_CDAF);
}

/// From state `[1, 2, 3, 4]`, worked by hand: `rotl(1 + 4, 23) + 1 = 41943041`, and after one update the state is `[7, 0, 262146, 6·2^45]`, giving `rotl(7 + 6·2^45, 23) + 7 = 58720359`.
#[test]
fn xoshiro256pp_matches_hand_computed_outputs() {
    let mut rng = Rng { s: [1, 2, 3, 4] };
    assert_eq!(rng.next_u64(), 41_943_041);
    assert_eq!(rng.next_u64(), 58_720_359);
}

#[test]
fn a_stream_is_a_function_of_its_key() {
    let draw = |key: &[u64]| {
        let mut rng = Rng::from_key(key);
        [rng.next_u64(), rng.next_u64()]
    };
    assert_eq!(draw(&[7, 1, 2]), draw(&[7, 1, 2]));
    assert_ne!(draw(&[7, 1, 2]), draw(&[7, 1, 3]));
    assert_ne!(draw(&[7, 1, 2]), draw(&[7, 2, 1]));
    assert_ne!(draw(&[7, 1]), draw(&[7, 1, 0]));
}

/// Every draw in `[0, 1)`, with mean `1/2` within 5σ over 100k draws.
#[test]
fn uniform_is_in_the_unit_interval_with_mean_one_half() {
    let n = 100_000;
    let mut rng = Rng::from_key(&[0xABCD]);
    let draws: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
    assert!(draws.iter().all(|u| (0.0..1.0).contains(u)));
    let mean = draws.iter().sum::<f64>() / n as f64;
    assert!(
        (mean - 0.5).abs() < 5.0 * (1.0 / 12.0 / n as f64).sqrt(),
        "mean {mean}"
    );
}
