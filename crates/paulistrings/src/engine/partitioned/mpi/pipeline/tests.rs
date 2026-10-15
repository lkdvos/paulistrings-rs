use super::*;
use ::mpi::point_to_point::{Destination, Source};
use ::mpi::topology::Communicator;
use ::mpi::Threading;
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::time::Duration;

const CHUNKS: usize = 64;
const PIECES: usize = 3;
const WAITERS: usize = 8;

/// Abort the process unless the returned sender reports in within a minute, so a deadlock fails the run instead of hanging it.
fn watchdog(what: &'static str) -> std::sync::mpsc::Sender<()> {
    let (done, report) = channel::<()>();
    std::thread::spawn(move || {
        if let Err(RecvTimeoutError::Timeout) = report.recv_timeout(Duration::from_secs(60)) {
            eprintln!("{what}: no progress in 60 s, aborting");
            std::process::abort();
        }
    });
    done
}

/// Many threads waiting on a self-exchange's chunks in clashing orders all return, each only after its chunk completed, and every byte arrives.
#[test]
fn concurrent_waiters_each_see_their_chunk_complete() {
    // The only test in this binary that initialises MPI, which a process may do once.
    let Some((universe, threading)) = ::mpi::initialize_with_threading(Threading::Serialized)
    else {
        return;
    };
    if threading < Threading::Serialized {
        return;
    }
    let world = universe.world();
    let me = world.process_at_rank(world.rank());
    let done = watchdog("concurrent_waiters_each_see_their_chunk_complete");

    for round in 0..20usize {
        let len = 256 << (round % 11);
        let send: Vec<Vec<u8>> =
            (0..CHUNKS * PIECES).map(|i| vec![(i + round) as u8; len]).collect();
        let mut recv: Vec<Vec<u8>> = vec![vec![0; len]; CHUNKS * PIECES];
        ::mpi::request::multiple_scope(2 * CHUNKS * PIECES, |scope, requests| {
            let mut slot_of = Vec::new();
            for (i, buffer) in send.iter().enumerate() {
                let r = requests.add(me.immediate_send_with_tag(scope, &buffer[..], i as i32));
                note_slot(&mut slot_of, r, SLOT_SEND);
            }
            for (i, buffer) in recv.iter_mut().enumerate() {
                let r = requests.add(me.immediate_receive_into_with_tag(
                    scope,
                    &mut buffer[..],
                    i as i32,
                ));
                note_slot(&mut slot_of, r, i / PIECES);
            }
            let mut pipeline = ChunkPipeline::new(requests, slot_of, CHUNKS);
            std::thread::scope(|threads| {
                for waiter in 0..WAITERS {
                    let pipeline = &pipeline;
                    threads.spawn(move || {
                        for j in 0..CHUNKS {
                            let k = (j * (2 * waiter + 1) + waiter * 7) % CHUNKS;
                            pipeline.wait_chunk(k);
                            assert!(pipeline.complete(k), "chunk {k} returned incomplete");
                        }
                    });
                }
            });
            pipeline.finish();
        });
        for (i, buffer) in recv.iter().enumerate() {
            assert!(buffer.iter().all(|&b| b == (i + round) as u8), "piece {i}");
        }
    }
    done.send(()).unwrap();
}
