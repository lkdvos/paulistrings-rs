use super::*;

const SEVEN: Lengths = Lengths { len: 7, longest: Some(3) };

#[test]
fn a_join_doubles_a_cut_caps_and_a_regroup_forgets_the_longest() {
    assert_eq!(SEVEN.after(Growth::Join), Lengths { len: 14, longest: Some(6) });
    assert_eq!(SEVEN.after(Growth::Shrink { cap: 5 }), Lengths { len: 5, longest: Some(3) });
    assert_eq!(SEVEN.after(Growth::Shrink { cap: 9 }), SEVEN);
    assert_eq!(SEVEN.after(Growth::Regroup), Lengths { len: 7, longest: None });
    assert_eq!(SEVEN.estimate_after(Growth::Join), SEVEN, "an estimate keeps a join's length");
}

fn tracker(lag: usize) -> (Arc<CudaStream>, LengthTracker) {
    let context = CudaContext::new(0).expect("context");
    let stream = context.new_stream().expect("stream");
    (stream, LengthTracker::new(context, SEVEN, lag))
}

/// A read-back anchors the bound only once `lag` later ones are in flight, and steps after it fold on top of it.
#[test]
fn a_read_back_anchors_the_bound_after_lag_more_steps() {
    crate::require_cuda!();
    let (stream, mut length) = tracker(1);
    let after_join = stream.clone_htod(&[10u32, 4]).unwrap();
    let after_cut = stream.clone_htod(&[6u32, 4]).unwrap();
    length.step(Growth::Join, Some((&stream, &after_join))).unwrap();
    assert_eq!(length.bound(), Lengths { len: 14, longest: Some(6) }, "still in flight");
    assert_eq!(length.estimate(), SEVEN);
    length.step(Growth::Shrink { cap: 8 }, Some((&stream, &after_cut))).unwrap();
    assert_eq!(
        length.bound(),
        Lengths { len: 8, longest: Some(4) },
        "the join's read-back, capped"
    );
    assert_eq!(length.estimate(), Lengths { len: 8, longest: Some(4) });
    assert!(!length.is_exact());
    length.consume(0).unwrap();
    assert_eq!(length.bound(), Lengths { len: 6, longest: Some(4) });
    assert!(length.is_exact());
}

/// A group member's tracker (lag 0) consumes every read-back at its step, so its bound is exact after every published step.
#[test]
fn lag_zero_is_exact_after_every_published_step() {
    crate::require_cuda!();
    let (stream, mut length) = tracker(0);
    let device = stream.clone_htod(&[11u32, 5]).unwrap();
    length.step(Growth::Join, Some((&stream, &device))).unwrap();
    assert!(length.is_exact());
    assert_eq!(length.bound(), Lengths { len: 11, longest: Some(5) });
    length.step(Growth::Regroup, None).unwrap();
    assert_eq!(length.bound(), Lengths { len: 11, longest: None });
    length.set_exact(SEVEN);
    assert_eq!(length.bound(), SEVEN);
}

/// An exact length after a shortening step keeps the longest bucket as a bound, which leaves the tracker exact but unsettled until a read anchors it.
#[test]
fn an_exact_length_alone_leaves_the_longest_unsettled() {
    crate::require_cuda!();
    let (stream, mut length) = tracker(0);
    assert!(length.is_settled());
    length.set_exact_len(4);
    assert_eq!(length.bound(), Lengths { len: 4, longest: Some(3) });
    assert!(length.is_exact() && !length.is_settled());
    let device = stream.clone_htod(&[4u32, 2]).unwrap();
    length.step(Growth::Shrink { cap: 4 }, Some((&stream, &device))).unwrap();
    assert!(length.is_settled());
    assert_eq!(length.bound(), Lengths { len: 4, longest: Some(2) });
}
