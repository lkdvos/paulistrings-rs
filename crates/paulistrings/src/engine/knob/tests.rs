use super::*;

static COUNT: Knob<usize> = Knob::new(
    "PAULISTRINGS_TEST_KNOB_NEVER_SET",
    "paulistrings::propagate",
    parse_positive,
    || 8,
);

#[test]
fn unset_empty_and_unparsable_values_fall_back_to_the_default() {
    for raw in [
        None,
        Some(""),
        Some("  "),
        Some("0"),
        Some("-2"),
        Some("many"),
    ] {
        assert_eq!(COUNT.resolve(raw), 8, "{raw:?}");
    }
}

#[test]
fn a_valid_value_is_parsed_after_trimming() {
    assert_eq!(COUNT.resolve(Some("64")), 64);
    assert_eq!(COUNT.resolve(Some(" 3 ")), 3);
}

#[test]
fn get_reads_once_and_caches() {
    let first: *const usize = COUNT.get();
    assert_eq!(*COUNT.get(), 8);
    assert!(std::ptr::eq(first, COUNT.get()));
}
