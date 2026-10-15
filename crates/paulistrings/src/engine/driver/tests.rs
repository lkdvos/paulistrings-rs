// The check under test exists only in a debug build.
#![cfg(debug_assertions)]

use super::*;
use crate::engine::partitioned::in_process::InProcessTransport;

/// Run `f` on every rank of an in-process group of `size`, returning each rank's panic message, if any.
fn on_every_rank_caught(
    size: u32,
    f: impl Fn(&InProcessTransport) + Send + Sync,
) -> Vec<Option<String>> {
    let group = InProcessTransport::group(size);
    let f = &f;
    std::thread::scope(|scope| {
        let handles: Vec<_> = group
            .into_iter()
            .map(|transport| {
                scope.spawn(move || {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&transport)))
                        .err()
                        .map(|payload| match payload.downcast::<String>() {
                            Ok(message) => *message,
                            Err(payload) => payload
                                .downcast::<&'static str>()
                                .map(|message| message.to_string())
                                .unwrap_or_default(),
                        })
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

#[test]
fn agreeing_bucket_bits_pass_the_debug_check() {
    let outcomes = on_every_rank_caught(4, |transport| {
        assert_bucket_bits_agree(transport, 7, 3);
    });
    assert_eq!(outcomes, vec![None; 4]);
}

/// One partition a refinement ahead of the others fails the check on every partition, not only its own.
#[test]
fn a_partition_holding_other_bucket_bits_fails_the_debug_check_everywhere() {
    let outcomes = on_every_rank_caught(4, |transport| {
        let bits = if transport.rank() == 2 { 8 } else { 7 };
        assert_bucket_bits_agree(transport, bits, 5);
    });
    for (rank, outcome) in outcomes.iter().enumerate() {
        let message = outcome
            .as_deref()
            .unwrap_or_else(|| panic!("rank {rank} passed"));
        assert!(
            message.contains(&format!(
                "partition {rank}, layer 5: the partitions hold between 7 and 8 bucket bits"
            )),
            "rank {rank}: {message}",
        );
    }
}
