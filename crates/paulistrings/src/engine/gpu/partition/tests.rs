use super::*;
use crate::test_support::{and, or, rand_sum_real, LoggingTransport};
use crate::truncation::BuiltinTruncation as T;
use crate::TruncationPolicy;

/// A one-rank group whose `allreduce_sum_u64` calls are counted.
fn one_rank() -> LoggingTransport {
    LoggingTransport::group(1).pop().expect("one rank")
}

fn reductions(t: &LoggingTransport) -> usize {
    t.log.count("allreduce_sum_u64")
}

fn part(input: &crate::PauliSum<1>, tree: &T) -> DevicePartition<1> {
    let mut part = DevicePartition::new(
        GpuSum::from_host(input, 0).expect("upload"),
        GpuLayerOptions::default(),
    )
    .expect("partition");
    part.keep = KeepProgram::lower(tree).expect("lower");
    part
}

/// `got` keeps the terms `want` does up to `ApproxTopN`'s free choice within a boundary octave: as many, the same octave populations, and each an input term with its coefficient.
fn assert_same_cut(
    got: &crate::PauliSum<1>,
    want: &crate::PauliSum<1>,
    input: &crate::PauliSum<1>,
    what: &str,
) {
    use crate::truncation::builtin::octave_histogram;
    assert_eq!(got.len(), want.len(), "{what}: len");
    assert_eq!(octave_histogram(got), octave_histogram(want), "{what}: octave populations");
    let terms: std::collections::HashMap<_, _> =
        input.iter().map(|(x, z, c)| ((*x, *z), c)).collect();
    for (x, z, c) in got.iter() {
        assert_eq!(terms.get(&(*x, *z)), Some(&c), "{what}: a kept term is an input term");
    }
}

/// A lone device's layer pass decides every `ApproxTopN` on the device and issues no reduction, and keeps the host's terms up to the boundary octave's free choice.
#[test]
fn a_layer_pass_issues_the_hosts_collectives() {
    crate::require_cuda!();
    let input = rand_sum_real::<1>(4000, 32, 0xC011);
    let cases = [
        (T::ApproxTopN(1000), 0),
        (and(T::Coefficient(1e-3), T::ApproxTopN(1000)), 0),
        (and(T::ApproxTopN(2000), T::ApproxTopN(500)), 0),
        (or(T::Weight(10), T::Coefficient(1e-3)), 0),
    ];
    for (tree, want) in cases {
        let device = one_rank();
        let mut part = part(&input, &tree);
        PartitionBackend::<1, Complex64, T>::finalize_layer(&mut part, &tree, &device);
        part.take_error().expect("device layer pass");
        let mut want_sum = input.clone();
        <T as TruncationPolicy<1>>::finalize_layer(&tree, &mut want_sum);
        assert_eq!(reductions(&device), want, "{tree:?}: device");
        assert_eq!(part.len(), want_sum.len(), "{tree:?}: len");
        let got = part.sum().to_host().unwrap();
        assert_same_cut(&got, &want_sum, &input, &format!("{tree:?}"));
    }
}

/// Exact `TopN` on a lone device partition (`group_size == 1`) issues no collective at all, unlike `ApproxTopN`, and keeps the host's terms exactly — alone and composed with `And`/`Or`.
#[test]
fn a_lone_partition_runs_exact_topn_with_no_collective() {
    crate::require_cuda!();
    use crate::truncation::TopN;
    let input = rand_sum_real::<1>(4000, 32, 0xC012);
    let cases = [
        T::TopN(1000),
        and(T::Coefficient(1e-3), T::TopN(1000)),
        and(T::TopN(2000), T::Weight(20)),
        and(T::TopN(1000), or(T::Weight(10), T::Coefficient(1e-3))),
    ];
    for tree in cases {
        let device = one_rank();
        let mut part = part(&input, &tree);
        PartitionBackend::<1, Complex64, T>::finalize_layer(&mut part, &tree, &device);
        part.take_error().expect("device layer pass");
        assert_eq!(reductions(&device), 0, "{tree:?}: exact TopN has no collective form");
        let mut want_sum = input.clone();
        <T as TruncationPolicy<1>>::finalize_layer(&tree, &mut want_sum);
        assert_eq!(part.len(), want_sum.len(), "{tree:?}: len");
        assert_eq!(
            part.sum().to_host().unwrap().to_arrays(),
            want_sum.to_arrays(),
            "{tree:?}: terms"
        );
    }
    // Sanity: `TopN` alone actually truncates against this input.
    let mut sanity = input.clone();
    TopN(1000).finalize_layer(&mut sanity);
    assert_eq!(sanity.len(), 1000);
}

/// A group member (`group_size > 1`) reports `Unsupported` on an exact `TopN` rather than run a wrong local selection.
#[test]
fn a_group_member_rejects_exact_topn() {
    crate::require_cuda!();
    let input = rand_sum_real::<1>(500, 32, 0xC013);
    let tree = T::TopN(100);
    let mut part = part(&input, &tree);
    part.group_size = 2;
    PartitionBackend::<1, Complex64, T>::finalize_layer(&mut part, &tree, &one_rank());
    assert!(matches!(part.take_error(), Err(GpuError::Unsupported(_))));
}

/// A partition that already failed still enters every reduction, so a group cannot fall out of step on one device's error.
#[test]
fn a_failed_partition_still_enters_the_reduction() {
    crate::require_cuda!();
    let input = rand_sum_real::<1>(500, 32, 0xFA11);
    let tree = and(T::ApproxTopN(100), T::ApproxTopN(50));
    let mut part = part(&input, &tree);
    part.error = Some(GpuError::Unsupported("injected"));
    let group = one_rank();
    PartitionBackend::<1, Complex64, T>::finalize_layer(&mut part, &tree, &group);
    assert_eq!(reductions(&group), 2);
    assert!(matches!(part.take_error(), Err(GpuError::Unsupported("injected"))));
    assert_eq!(part.len(), input.len(), "a failed partition keeps its terms");
}

/// A lone partition already within `n` keeps every term, and one term more cuts to `n`, with no reduction either way.
#[test]
fn a_lone_partition_within_n_skips_the_approx_pass() {
    crate::require_cuda!();
    let input = rand_sum_real::<1>(500, 32, 0x1E55);
    for (n, want) in [(500, 0), (499, 0)] {
        let tree = T::ApproxTopN(n);
        let mut part = part(&input, &tree);
        let group = one_rank();
        PartitionBackend::<1, Complex64, T>::finalize_layer(&mut part, &tree, &group);
        part.take_error().expect("device layer pass");
        assert_eq!(reductions(&group), want, "n={n}");
        assert_eq!(part.len(), n, "n={n}");
    }
}
