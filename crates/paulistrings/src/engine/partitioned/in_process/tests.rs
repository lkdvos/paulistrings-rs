use super::*;
use crate::engine::partitioned::exchange_block::check_stride;
use std::time::{Duration, Instant};

/// A one-column [`Payload`].
impl Payload for Vec<u64> {
    fn byte_parts(&self) -> Vec<&[u8]> {
        vec![bytemuck::cast_slice(&self[..])]
    }

    fn receive_into(&mut self, lens: &[usize]) -> Vec<&mut [u8]> {
        assert_eq!(lens.len(), 1, "test payload: expected one part");
        check_stride(lens[0], size_of::<u64>(), "test payload");
        self.clear();
        self.resize(lens[0] / size_of::<u64>(), 0);
        vec![bytemuck::cast_slice_mut(&mut self[..])]
    }

    fn finish_receive(&mut self) {}
}

/// A column unique to the ordered pair `(from, to)`.
fn message(from: u32, to: u32) -> Vec<u64> {
    vec![(u64::from(from) << 32) | u64::from(to); from as usize + 1]
}

fn sends_nothing(from: u32, to: u32) -> bool {
    from == 0 && to == 1
}

fn exchange_round(size: u32) {
    let group = InProcessTransport::group(size);
    assert_eq!(group.len(), size as usize);

    std::thread::scope(|scope| {
        let handles: Vec<_> = group
            .into_iter()
            .map(|transport| {
                scope.spawn(move || {
                    let rank = transport.rank();
                    assert_eq!(transport.size(), size);
                    let send: Vec<Option<Vec<u64>>> = (0..size)
                        .map(|q| (q != rank && !sends_nothing(rank, q)).then(|| message(rank, q)))
                        .collect();
                    (rank, transport.exchange(send, &mut Vec::new()))
                })
            })
            .collect();

        for handle in handles {
            let (rank, recv) = handle.join().expect("rank thread panicked");
            assert_eq!(recv.len(), size as usize, "rank {rank}");
            for q in 0..size {
                let expected = (q != rank && !sends_nothing(q, rank)).then(|| message(q, rank));
                assert_eq!(recv[q as usize], expected, "rank {rank} slot {q}");
            }
        }
    });
}

#[test]
fn exchange_delivers_each_payload_to_its_partner() {
    for size in [2u32, 4] {
        exchange_round(size);
    }
}

#[test]
fn reductions_and_barrier_agree_on_every_rank() {
    let size = 4u32;
    let group = InProcessTransport::group(size);

    let mut results: Vec<(u32, u8, Vec<u64>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = group
            .into_iter()
            .map(|transport| {
                scope.spawn(move || {
                    let rank = transport.rank();
                    transport.barrier();
                    // Contributions 0, 7, 14, 21 → max 21.
                    let max = transport.allreduce_max_u8((rank * 7) as u8);
                    // Columns (r+1, 10·(r+1)) → sums (10, 100).
                    let mut buf = vec![u64::from(rank) + 1, 10 * (u64::from(rank) + 1)];
                    transport.allreduce_sum_u64(&mut buf);
                    transport.barrier();
                    (rank, max, buf)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread panicked"))
            .collect()
    });

    results.sort_by_key(|(rank, _, _)| *rank);
    assert_eq!(results.len(), size as usize);
    for (rank, max, sum) in results {
        assert_eq!(max, 21, "rank {rank}");
        assert_eq!(sum, vec![10, 100], "rank {rank}");
    }
}

#[test]
fn a_group_of_one_is_a_no_op() {
    let group = InProcessTransport::group(1);
    assert_eq!(group.len(), 1);
    let transport = &group[0];
    assert_eq!(transport.rank(), 0);
    assert_eq!(transport.size(), 1);

    let recv: Vec<Option<Vec<u64>>> = transport.exchange(vec![None], &mut Vec::new());
    assert_eq!(recv.len(), 1);
    assert!(recv[0].is_none());

    assert_eq!(transport.allreduce_max_u8(9), 9);
    let mut buf = vec![3, 4];
    transport.allreduce_sum_u64(&mut buf);
    assert_eq!(buf, vec![3, 4]);
    let mut buf = vec![-0.5, 1e300];
    transport.allreduce_sum_f64(&mut buf);
    assert_eq!(buf, vec![-0.5, 1e300]);
    transport.barrier();
}

/// `1e16 + 1 + 1 - 1e16` folds to `0` in rank order on every rank; a one-rank slot comes back exactly.
#[test]
fn f64_sums_are_bitwise_identical_on_every_rank() {
    let inputs = [
        [1e16, 0.25, 0.1, 0.0, 0.0],
        [1.0, 1.25, 0.0, 0.2, 0.0],
        [1.0, 2.25, 0.0, 0.0, 0.3],
        [-1e16, 3.25, 0.0, 0.0, 0.0],
    ];
    let results = on_every_rank(4, |transport| {
        (0..50)
            .map(|_| {
                let mut buf = inputs[transport.rank() as usize];
                transport.allreduce_sum_f64(&mut buf);
                buf
            })
            .collect::<Vec<_>>()
    });
    for (rank, rounds) in results.iter().enumerate() {
        for round in rounds {
            assert_eq!(round, &[0.0, 7.0, 0.1, 0.2, 0.3], "rank {rank}");
        }
    }
}

#[test]
#[should_panic(expected = "allreduce_sum_f64")]
fn mixing_integer_and_float_sums_is_detected() {
    let group = InProcessTransport::group(2);
    std::thread::scope(|scope| {
        let handles: Vec<_> = group
            .into_iter()
            .map(|transport| {
                scope.spawn(move || {
                    if transport.rank() == 0 {
                        transport.allreduce_sum_u64(&mut [1]);
                    } else {
                        transport.allreduce_sum_f64(&mut [1.0]);
                    }
                })
            })
            .collect();
        let joined: Vec<_> = handles.into_iter().map(|h| h.join()).collect();
        for result in joined {
            if let Err(payload) = result {
                std::panic::resume_unwind(payload);
            }
        }
    });
}

#[test]
#[should_panic(expected = "must be None")]
fn exchange_rejects_a_payload_addressed_to_this_partition() {
    let group = InProcessTransport::group(1);
    let _: Vec<Option<Vec<u64>>> = group[0].exchange(vec![Some(vec![1u64])], &mut Vec::new());
}

#[test]
#[should_panic(expected = "one entry per partition")]
fn exchange_rejects_a_wrongly_sized_send_vector() {
    let group = InProcessTransport::group(2);
    let _: Vec<Option<Vec<u64>>> = group[0].exchange(vec![None], &mut Vec::new());
}

/// The in-step rank reports the mismatch; the desynchronized one dies of its partner's departure.
#[test]
fn a_desynchronized_partition_is_caught_by_the_generation_stamp() {
    let mut group = InProcessTransport::group(2);
    let one = group.pop().expect("rank 1");
    let zero = group.pop().expect("rank 0");

    let (desynced, in_step) = std::thread::scope(|scope| {
        let desynced = scope.spawn(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                zero.skip_sequence_for_test();
                zero.barrier();
            }))
        });
        let in_step = scope.spawn(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || one.barrier()))
        });
        (
            desynced.join().expect("rank 0 thread"),
            in_step.join().expect("rank 1 thread"),
        )
    });

    let in_step = panic_message(
        in_step
            .expect_err("the in-step rank must reject the mismatched generation")
            .as_ref(),
    );
    assert!(in_step.contains("collective order mismatch"), "{in_step}");
    let desynced = panic_message(
        desynced
            .expect_err("the desynchronized rank waits for a call nobody makes")
            .as_ref(),
    );
    assert!(desynced.contains("terminated"), "{desynced}");
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<panic payload is not a string>".to_string()
    }
}

#[test]
fn a_partner_that_dies_between_collectives_fails_the_others_promptly() {
    let mut group = InProcessTransport::group(2);
    let one = group.pop().expect("rank 1");
    let zero = group.pop().expect("rank 0");

    let (elapsed, message) = std::thread::scope(|scope| {
        scope.spawn(move || {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                one.barrier();
                panic!("rank 1 dies after its barrier");
            }));
        });
        zero.barrier();
        let started = Instant::now();
        let payload =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| zero.allreduce_max_u8(3)))
                .expect_err("the surviving rank cannot complete a collective alone");
        (started.elapsed(), panic_message(payload.as_ref()))
    });

    assert!(message.contains("terminated"), "{message}");
    assert!(message.contains("partition 1"), "{message}");
    assert!(
        elapsed < Duration::from_secs(2),
        "the survivor waited {elapsed:?}, so it fell back on the {DEFAULT_WAIT_TIMEOUT:?} timeout \
         instead of noticing the departure",
    );
}

#[test]
fn a_thousand_back_to_back_sums_never_mix_generations() {
    const ROUNDS: u64 = 1_000;
    let size = 4u32;
    let group = InProcessTransport::group(size);

    std::thread::scope(|scope| {
        let handles: Vec<_> = group
            .into_iter()
            .map(|transport| {
                scope.spawn(move || {
                    let rank = u64::from(transport.rank());
                    for round in 1..=ROUNDS {
                        let mut buf = vec![rank * round, round, rank];
                        transport.allreduce_sum_u64(&mut buf);
                        // Σ rank = 0+1+2+3 = 6, Σ round = 4·round.
                        assert_eq!(
                            buf,
                            vec![6 * round, 4 * round, 6],
                            "rank {rank}, round {round}",
                        );
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("rank thread panicked");
        }
    });
}

/// The first `max`, the reduced buffer, and the flag `max` of one round.
type MixedRound = (u8, Vec<u64>, u8);
type MixedScript = (u32, Vec<MixedRound>);

#[test]
fn mixed_collective_sequences_agree_on_every_rank() {
    const ROUNDS: u8 = 25;
    let size = 4u32;
    let group = InProcessTransport::group(size);

    let mut results: Vec<MixedScript> = std::thread::scope(|scope| {
        let handles: Vec<_> = group
            .into_iter()
            .map(|transport| {
                scope.spawn(move || {
                    let rank = transport.rank();
                    let mut rounds = Vec::new();
                    for round in 0..ROUNDS {
                        let hi = transport.allreduce_max_u8(rank as u8 * 3 + round);
                        transport.barrier();
                        let mut buf =
                            vec![u64::from(rank) + 1, u64::from(round), u64::from(rank) << 8];
                        transport.allreduce_sum_u64(&mut buf);
                        transport.barrier();
                        let flag = transport.allreduce_max_u8(if rank == 2 { 255 } else { 0 });
                        rounds.push((hi, buf, flag));
                    }
                    (rank, rounds)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread panicked"))
            .collect()
    });

    results.sort_by_key(|(rank, _)| *rank);
    for (rank, rounds) in &results {
        assert_eq!(rounds.len(), ROUNDS as usize, "rank {rank}");
        for (round, (hi, sum, flag)) in rounds.iter().enumerate() {
            let round = round as u8;
            // max over 3·rank + round is 9 + round; Σ(rank+1) = 10,
            // Σ round = 4·round, Σ(rank << 8) = 6·256.
            assert_eq!(*hi, 9 + round, "rank {rank}, round {round}");
            assert_eq!(
                *sum,
                vec![10, 4 * u64::from(round), 6 << 8],
                "rank {rank}, round {round}",
            );
            assert_eq!(*flag, 255, "rank {rank}, round {round}");
        }
    }
    for (rank, rounds) in &results[1..] {
        assert_eq!(rounds, &results[0].1, "rank {rank} disagrees with rank 0");
    }
}

/// Completing is the assertion.
#[test]
fn sixteen_ranks_complete_when_oversubscribed() {
    const ROUNDS: u64 = 50;
    let size = 16u32;
    let group = InProcessTransport::group(size);

    std::thread::scope(|scope| {
        let handles: Vec<_> = group
            .into_iter()
            .map(|transport| {
                scope.spawn(move || {
                    let rank = transport.rank();
                    for round in 1..=ROUNDS {
                        assert_eq!(transport.allreduce_max_u8(rank as u8 + 1), 16);
                        let mut buf = vec![u64::from(rank), round];
                        transport.allreduce_sum_u64(&mut buf);
                        // Σ_{r<16} r = 120.
                        assert_eq!(buf, vec![120, 16 * round], "rank {rank}, round {round}");
                        transport.barrier();
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("rank thread panicked");
        }
    });
}

fn on_every_rank<O: Send>(size: u32, f: impl Fn(&InProcessTransport) -> O + Send + Sync) -> Vec<O> {
    let group = InProcessTransport::group(size);
    let f = &f;
    std::thread::scope(|scope| {
        let handles: Vec<_> = group
            .into_iter()
            .map(|transport| scope.spawn(move || f(&transport)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread panicked"))
            .collect()
    })
}

#[test]
fn check_consistency_passes_when_every_rank_agrees() {
    for size in [1u32, 2, 4] {
        on_every_rank(size, |transport| {
            transport.check_consistency(0xdead_beef_0000_0001);
        });
    }
}

#[test]
#[should_panic(expected = "disagree about the run")]
fn check_consistency_names_a_rank_that_disagrees() {
    let mut group = InProcessTransport::group(2);
    let one = group.pop().expect("rank 1");
    let zero = group.pop().expect("rank 0");

    std::thread::scope(|scope| {
        scope.spawn(move || {
            let _ =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| one.check_consistency(5)));
        });
        zero.check_consistency(4);
    });
}

#[test]
fn gather_to_root_collects_every_ranks_parts_in_rank_order() {
    for size in [1u32, 2, 4] {
        let got = on_every_rank(size, |transport| {
            let rank = transport.rank();
            // Rank `r` sends `r + 1` parts, part `j` being `r + 1` copies of `10 · r + j`.
            let owned: Vec<Vec<u8>> = (0..=rank)
                .map(|j| vec![(10 * rank + j) as u8; rank as usize + 1])
                .collect();
            let parts: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
            transport.gather_to_root(parts)
        });

        for (rank, out) in got.iter().enumerate() {
            if rank != 0 {
                assert!(out.is_none(), "rank {rank} must not gather");
                continue;
            }
            let all = out.as_ref().expect("rank 0 gathers");
            assert_eq!(all.len(), size as usize);
            for (r, parts) in all.iter().enumerate() {
                let r = r as u32;
                assert_eq!(parts.len(), r as usize + 1, "rank {r} part count");
                for (j, part) in parts.iter().enumerate() {
                    assert_eq!(part, &vec![(10 * r + j as u32) as u8; r as usize + 1]);
                }
            }
        }
    }
}

#[test]
fn gather_to_root_of_no_parts_is_an_empty_vector_per_rank() {
    let got = on_every_rank(2, |transport| transport.gather_to_root(Vec::new()));
    assert_eq!(got[0], Some(vec![Vec::<Vec<u8>>::new(); 2]));
    assert_eq!(got[1], None);
}

#[test]
#[should_panic(expected = "terminated")]
fn a_partner_that_panicked_is_reported_rather_than_hanging() {
    let mut group = InProcessTransport::group(2);
    let one = group.pop().expect("rank 1");
    let zero = group.pop().expect("rank 0");

    std::thread::scope(|scope| {
        scope.spawn(move || {
            let _ = std::panic::catch_unwind(|| panic!("rank 1 dies before its exchange"));
            drop(one);
        });
        let _: Vec<Option<Vec<u64>>> = zero.exchange(vec![None, Some(vec![7u64])], &mut Vec::new());
    });
}

/// Rank 32 is past a `u32` departure mask's width.
#[test]
#[should_panic(expected = "partition 32 terminated")]
fn a_partner_at_rank_32_that_panicked_is_reported_by_its_own_rank() {
    let mut group = InProcessTransport::group(33);
    let far = group.pop().expect("rank 32");
    let zero = group.remove(0);
    let middle = group; // ranks 1..=31

    std::thread::scope(|scope| {
        scope.spawn(move || {
            let _ = std::panic::catch_unwind(|| panic!("rank 32 dies before its barrier"));
            drop(far);
        });
        for transport in middle {
            scope.spawn(move || transport.barrier());
        }
        zero.barrier();
    });
}

/// Rank 0 is already waiting when rank 1 leaves without arriving.
#[test]
fn a_partner_that_departs_during_the_wait_fails_the_waiter_promptly() {
    let mut group = InProcessTransport::group(2);
    let one = group.pop().expect("rank 1");
    let zero = group.pop().expect("rank 0");

    let (elapsed, message) = std::thread::scope(|scope| {
        scope.spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            drop(one);
        });
        let started = Instant::now();
        let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| zero.barrier()))
            .expect_err("the waiter cannot complete a barrier alone");
        (started.elapsed(), panic_message(payload.as_ref()))
    });

    assert!(message.contains("partition 1 terminated"), "{message}");
    assert!(
        elapsed < Duration::from_secs(2),
        "the waiter took {elapsed:?}"
    );
}

/// A partner arriving long after the waiter's spin is waited for on the condvar.
#[test]
fn a_partner_arriving_after_the_spin_completes_the_sum() {
    let results = on_every_rank(2, |transport| {
        if transport.rank() == 1 {
            std::thread::sleep(Duration::from_millis(100));
        }
        let mut buffer = [u64::from(transport.rank()) + 1];
        transport.allreduce_sum_u64(&mut buffer);
        transport.barrier();
        buffer[0]
    });
    assert_eq!(results, vec![3, 3]);
}

/// A partner that neither arrives nor leaves trips the backstop.
#[test]
fn a_silent_partner_trips_the_timeout() {
    let mut group = InProcessTransport::group_with_timeout(2, Duration::from_millis(200));
    let one = group.pop().expect("rank 1");
    let zero = group.pop().expect("rank 0");

    let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        zero.allreduce_sum_u64(&mut [1])
    }))
    .expect_err("nobody else arrives");
    let message = panic_message(payload.as_ref());
    assert!(message.contains("no response in 200ms"), "{message}");
    assert!(message.contains("that one at 0"), "{message}");
    drop(one);
}

#[test]
#[should_panic(expected = "contributed 2 values")]
fn sums_of_different_lengths_are_rejected() {
    let group = InProcessTransport::group(2);
    std::thread::scope(|scope| {
        let handles: Vec<_> = group
            .into_iter()
            .map(|transport| {
                scope.spawn(move || {
                    if transport.rank() == 0 {
                        transport.allreduce_sum_u64(&mut [1]);
                    } else {
                        transport.allreduce_sum_u64(&mut [1, 2]);
                    }
                })
            })
            .collect();
        let joined: Vec<_> = handles.into_iter().map(|h| h.join()).collect();
        for result in joined {
            if let Err(payload) = result {
                std::panic::resume_unwind(payload);
            }
        }
    });
}

/// An exchange against a collective at the same call panics on both sides, and one of them names both calls.
#[test]
fn an_exchange_against_a_collective_is_caught() {
    let messages = on_every_rank_caught(2, |transport| {
        if transport.rank() == 0 {
            let _: Vec<Option<Vec<u64>>> = transport.exchange(vec![None, None], &mut Vec::new());
        } else {
            transport.barrier();
        }
    });
    assert!(
        messages.iter().all(Option::is_some),
        "both ranks must panic: {messages:?}"
    );
    assert!(
        messages
            .iter()
            .flatten()
            .any(|m| m.contains("collective order mismatch")
                && m.contains("exchange")
                && m.contains("barrier")),
        "{messages:?}",
    );
}

/// [`on_every_rank`], with each rank's panic message (or `None`) in rank order.
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
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || f(&transport)))
                        .err()
                        .map(|payload| panic_message(payload.as_ref()))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread"))
            .collect()
    })
}
