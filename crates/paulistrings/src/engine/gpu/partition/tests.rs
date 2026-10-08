use super::*;
use crate::engine::partitioned::truncation::PartitionedTruncation;
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

/// A device layer pass issues as many reductions as the host's `PartitionedTruncation` does for the same tree, and keeps the same terms.
#[test]
fn a_layer_pass_issues_the_hosts_collectives() {
    crate::require_cuda!();
    let input = rand_sum_real::<1>(4000, 32, 0xC011);
    let cases = [
        (T::ApproxTopN(1000), 1),
        (and(T::Coeff(1e-3), T::ApproxTopN(1000)), 1),
        (and(T::ApproxTopN(2000), T::ApproxTopN(500)), 2),
        (or(T::ApproxTopN(10), T::Coeff(1e-3)), 0),
    ];
    for (tree, want) in cases {
        let device = one_rank();
        let mut part = part(&input, &tree);
        PartitionBackend::<1, T>::finalize_layer(&mut part, &tree, &device);
        part.take_error().expect("device layer pass");
        let host = one_rank();
        let mut want_sum = input.clone();
        <T as PartitionedTruncation<1>>::finalize_layer_partitioned(&tree, &mut want_sum, &host);
        assert_eq!(reductions(&device), want, "{tree:?}: device");
        assert_eq!(reductions(&host), want, "{tree:?}: host");
        assert_eq!(part.len(), want_sum.len(), "{tree:?}: len");
        assert_eq!(
            part.sum().to_host().unwrap().to_arrays(),
            want_sum.to_arrays(),
            "{tree:?}: terms"
        );
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
        and(T::Coeff(1e-3), T::TopN(1000)),
        and(T::TopN(2000), T::Weight(20)),
        or(T::TopN(10), T::Coeff(1e-3)),
    ];
    for tree in cases {
        let device = one_rank();
        let mut part = part(&input, &tree);
        PartitionBackend::<1, T>::finalize_layer(&mut part, &tree, &device);
        part.take_error().expect("device layer pass");
        assert_eq!(
            reductions(&device),
            0,
            "{tree:?}: exact TopN has no collective form"
        );
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
    PartitionBackend::<1, T>::finalize_layer(&mut part, &tree, &one_rank());
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
    PartitionBackend::<1, T>::finalize_layer(&mut part, &tree, &group);
    assert_eq!(reductions(&group), 2);
    assert!(matches!(
        part.take_error(),
        Err(GpuError::Unsupported("injected"))
    ));
    assert_eq!(
        part.len(),
        input.len(),
        "a failed partition keeps its terms"
    );
}
