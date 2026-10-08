use super::*;
use crate::engine::partitioned::InProcessTransport;

#[test]
fn a_probe_that_never_readies_times_out() {
    let start = Instant::now();
    let mut calls = 0;
    let ready = poll_until(Duration::from_millis(20), || {
        calls += 1;
        Ok(false)
    })
    .expect("the probe never fails");
    assert!(!ready);
    assert!(calls > 1);
    assert!(start.elapsed() < Duration::from_secs(5));
}

#[test]
fn a_probe_error_ends_the_poll() {
    let err = poll_until(Duration::from_secs(60), || {
        Err(GpuError::Unsupported("probe"))
    })
    .expect_err("the error propagates");
    assert!(matches!(err, GpuError::Unsupported("probe")));
}

#[test]
fn the_unique_id_survives_the_word_packing() {
    let mut id = sys::ncclUniqueId { internal: [0; 128] };
    for (i, c) in id.internal.iter_mut().enumerate() {
        *c = (i as u8).wrapping_mul(37).wrapping_add(11) as std::ffi::c_char;
    }
    let mut words = [0u64; ID_WORDS];
    pack_id(&id, &mut words);
    assert_eq!(unpack_id(&words), id);
}

#[test]
fn the_id_broadcast_reaches_every_rank_over_an_all_reduce() {
    let mut id = sys::ncclUniqueId { internal: [0; 128] };
    for (i, c) in id.internal.iter_mut().enumerate() {
        *c = (255 - i as u8) as std::ffi::c_char;
    }
    let got: Vec<sys::ncclUniqueId> = std::thread::scope(|s| {
        let handles: Vec<_> = InProcessTransport::group(4)
            .into_iter()
            .map(|t| {
                s.spawn(move || {
                    let mut buf = [0u64; ID_WORDS + 1];
                    if t.rank() == 0 {
                        pack_id(&id, &mut buf[..ID_WORDS]);
                    }
                    t.allreduce_sum_u64(&mut buf);
                    assert_eq!(buf[ID_WORDS], 0);
                    unpack_id(&buf[..ID_WORDS])
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert!(got.iter().all(|g| *g == id));
}

/// Every readiness and device pattern on two ranks, and a sample on four: one outcome on every rank, an error unless every rank can and no device repeats.
#[test]
fn the_start_is_agreed_for_every_readiness_and_device() {
    let run = |ranks: &[(bool, [u64; 2])]| -> Vec<Result<(), String>> {
        let size = ranks.len() as u32;
        std::thread::scope(|s| {
            let hs: Vec<_> = InProcessTransport::group(size)
                .into_iter()
                .map(|t| {
                    let (can, id) = ranks[t.rank() as usize];
                    s.spawn(move || agree_start(&t, can, id).map_err(|e| e.to_string()))
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        })
    };
    let mut cases: Vec<Vec<(bool, [u64; 2])>> = Vec::new();
    for c0 in [false, true] {
        for c1 in [false, true] {
            for id1 in [[1, 0], [2, 0]] {
                cases.push(vec![(c0, [1, 0]), (c1, id1)]);
            }
        }
    }
    for mask in 0..32u32 {
        cases.push(
            (0..4)
                .map(|r| {
                    let id = if mask & 16 != 0 && r == 3 {
                        1
                    } else {
                        r as u64 + 1
                    };
                    ((mask >> r) & 1 == 0, [id, 9])
                })
                .collect(),
        );
    }
    for ranks in &cases {
        let got = run(ranks);
        assert!(
            got.iter().all(|g| g == &got[0]),
            "{ranks:?}: ranks disagree: {got:?}"
        );
        let ids: Vec<_> = ranks.iter().map(|r| r.1).collect();
        let shared = (0..ids.len()).any(|i| ids[i + 1..].contains(&ids[i]));
        let can = ranks.iter().all(|r| r.0);
        match &got[0] {
            Ok(()) => assert!(can && !shared, "{ranks:?}"),
            Err(msg) if !can => assert!(msg.contains("libnccl"), "{msg}"),
            Err(msg) => assert!(shared && msg.contains("share a device"), "{msg}"),
        }
    }
}

#[test]
fn the_config_matches_the_initializer_layout() {
    let c = default_config();
    assert_eq!(c.size, 48);
    assert_eq!(c.magic, 0xcafe_beef);
    assert_eq!(c.blocking, i32::MIN);
    assert!(c.netName.is_null());
}

/// Held by every test that starts a real communicator, so no two initialize or abort at once in the test binary.
static REAL_NCCL: Mutex<()> = Mutex::new(());

type OneRank = (MutexGuard<'static, ()>, NcclComm, Arc<CudaStream>);

/// A one-rank NCCL communicator on device 0 and a stream on it, holding [`REAL_NCCL`], or `None` where NCCL is absent.
fn one_rank() -> Option<OneRank> {
    if !crate::engine::gpu::nccl_available() {
        return None;
    }
    let guard = REAL_NCCL.lock().unwrap_or_else(PoisonError::into_inner);
    let ctx = super::super::device::context(0).expect("a visible device");
    let t = InProcessTransport::group(1).pop().expect("one rank");
    let comm = NcclComm::init(&t, &ctx)
        .unwrap_or_else(|e| panic!("a one-rank communicator fails to initialize: {e}"));
    comm.set_timeout(Duration::from_secs(60));
    let stream = ctx.new_stream().expect("a stream");
    Some((guard, comm, stream))
}

#[test]
fn a_one_rank_communicator_warms_up_and_shuts_down_cleanly() {
    let Some((_nccl, comm, stream)) = one_rank() else {
        return;
    };
    assert_eq!((comm.rank, comm.size), (0, 1));
    comm.warm_up(&stream).expect("the warm-up completes");
    assert!(comm.is_healthy());
    drop(comm);
    assert!(
        ABORTS.lock().unwrap().is_empty(),
        "a healthy communicator shuts down without an abort"
    );
}

#[test]
fn an_aborted_communicator_refuses_every_later_call() {
    let Some((_nccl, comm, stream)) = one_rank() else {
        return;
    };
    comm.abort();
    comm.abort();
    assert!(!comm.is_healthy());
    assert!(matches!(comm.warm_up(&stream), Err(GpuError::Nccl { .. })));
    drop(comm);
    assert!(
        ABORTS.lock().unwrap().is_empty(),
        "a test build joins every abort thread at drop"
    );
}

/// Message sizes with distinct contents per message, so any receive matched to the wrong send fails on length or content; one is empty.
const SIZES: [usize; 7] = [1, 1000, 3, 65_536 + 5, 0, 17, 257 * 1024];

fn messages() -> Vec<Vec<u64>> {
    SIZES
        .iter()
        .enumerate()
        .map(|(m, &n)| {
            (0..n as u64)
                .map(|i| ((m as u64 + 1) << 48) ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
                .collect()
        })
        .collect()
}

fn upload(stream: &Arc<CudaStream>, msgs: &[Vec<u64>]) -> Vec<cudarc::driver::CudaSlice<u64>> {
    msgs.iter()
        .map(|m| {
            let mut s = stream.alloc_zeros::<u64>(m.len().max(1)).expect("alloc");
            if !m.is_empty() {
                stream.memcpy_htod(m, &mut s).expect("upload");
            }
            s
        })
        .collect()
}

/// Self-sends interleaved with their receives, one receive buffer per message: the rank is its own peer and each receive gets the send posted in its position.
#[test]
fn interleaved_self_sends_match_in_posting_order() {
    let Some((_nccl, comm, stream)) = one_rank() else {
        return;
    };
    let wire = NcclWire::new(Arc::new(comm));
    assert_eq!((wire.rank(), wire.size()), (0, 1));
    let msgs = messages();
    let sends = upload(&stream, &msgs);
    let mut recvs: Vec<_> = SIZES
        .iter()
        .map(|&n| stream.alloc_zeros::<u64>(n + 1).expect("alloc"))
        .collect();
    let mut group = WireGroup::new();
    for ((send, recv), &n) in sends.iter().zip(recvs.iter_mut()).zip(&SIZES) {
        group.send(send.slice(0..n), 0, &stream);
        group.recv(recv.slice_mut(0..n), 0, &stream);
    }
    assert_eq!(group.ops().len(), 2 * SIZES.len());
    group.post(&wire).expect("the group posts");
    wire.wait(&stream).expect("the group completes");
    for (m, (msg, recv)) in msgs.iter().zip(&recvs).enumerate() {
        let got = stream.clone_dtoh(recv).expect("download");
        assert_eq!(
            &got[..msg.len()],
            &msg[..],
            "message {m} of {} u64",
            msg.len()
        );
        assert_eq!(got[msg.len()], 0, "message {m} overran its receive");
    }
    assert!(wire.comm().is_healthy());
}

/// Every send first, then every receive carved from one concatenated column, as the exchange's `recv_*` layout is: matching is per peer in posting order across the whole group.
#[test]
fn self_sends_land_in_one_column_in_posting_order() {
    let Some((_nccl, comm, stream)) = one_rank() else {
        return;
    };
    let wire = NcclWire::new(Arc::new(comm));
    let msgs = messages();
    let sends = upload(&stream, &msgs);
    let total: usize = SIZES.iter().sum();
    let mut recv = stream.alloc_zeros::<u64>(total + 1).expect("alloc");
    let mut group = WireGroup::new();
    for (send, &n) in sends.iter().zip(&SIZES) {
        group.send(send.slice(0..n), 0, &stream);
    }
    let parts: Vec<(usize, u32)> = SIZES.iter().map(|&n| (n, 0)).collect();
    group.recv_parts(recv.as_view_mut(), &parts, &stream);
    group.post(&wire).expect("the group posts");
    wire.wait(&stream).expect("the group completes");
    let got = stream.clone_dtoh(&recv).expect("download");
    let mut at = 0;
    for (m, msg) in msgs.iter().enumerate() {
        assert_eq!(
            &got[at..at + msg.len()],
            &msg[..],
            "message {m} of {} u64",
            msg.len()
        );
        at += msg.len();
    }
    assert_eq!(got[total], 0, "nothing lands past the last message");
    assert!(wire.comm().is_healthy());
}

/// NCCL itself refuses a self-receive with no matching send at group end; the error aborts the communicator.
#[test]
fn an_unmatched_self_receive_fails_the_group_and_aborts() {
    let Some((_nccl, comm, stream)) = one_rank() else {
        return;
    };
    let wire = NcclWire::new(Arc::new(comm));
    let mut dst = stream.alloc_zeros::<u64>(4).expect("alloc");
    let mut group = WireGroup::new();
    group.recv(dst.as_view_mut(), 0, &stream);
    let posted = group.post(&wire);
    assert!(
        matches!(posted, Err(GpuError::Nccl { code: 5, .. })),
        "{posted:?}"
    );
    assert!(!wire.comm().is_healthy());
    assert!(matches!(wire.wait(&stream), Err(GpuError::Nccl { .. })));
}

/// The forced timeout: a real self send/recv group whose wait is told its work never completes returns `Timeout` within the bound, aborts the communicator, and nothing hangs.
#[test]
fn a_forced_timeout_returns_within_the_bound_and_aborts() {
    let Some((_nccl, comm, stream)) = one_rank() else {
        return;
    };
    let bound = Duration::from_millis(300);
    comm.set_timeout(bound);
    comm.force_timeout();
    let wire = NcclWire::new(Arc::new(comm));
    let src = stream.alloc_zeros::<u64>(64).expect("alloc");
    let mut dst = stream.alloc_zeros::<u64>(64).expect("alloc");
    let mut group = WireGroup::new();
    group.send(src.as_view(), 0, &stream);
    group.recv(dst.as_view_mut(), 0, &stream);
    group.post(&wire).expect("the group posts");
    let start = Instant::now();
    let waited = wire.wait(&stream);
    let elapsed = start.elapsed();
    assert!(
        matches!(waited, Err(GpuError::Timeout { .. })),
        "{waited:?}"
    );
    assert!(elapsed >= bound, "{elapsed:?}");
    assert!(elapsed < Duration::from_secs(30), "{elapsed:?}");
    assert!(!wire.comm().is_healthy());
    assert!(matches!(wire.wait(&stream), Err(GpuError::Nccl { .. })));
    stream.synchronize().expect("the stream drains");
    drop(wire);
}

#[test]
fn an_op_naming_a_peer_outside_the_group_is_refused_before_nccl() {
    let Some((_nccl, comm, stream)) = one_rank() else {
        return;
    };
    let wire = NcclWire::new(Arc::new(comm));
    let src = stream.alloc_zeros::<u8>(8).expect("alloc");
    let mut group = WireGroup::new();
    group.send(src.as_view(), 1, &stream);
    assert!(matches!(group.post(&wire), Err(GpuError::Nccl { .. })));
    assert!(wire.comm().is_healthy());
}

impl NcclComm {
    /// Replace the wait bound, so a test can force a timeout without waiting out the default.
    pub(crate) fn set_timeout(&self, timeout: Duration) {
        self.lock().timeout = timeout;
    }

    /// Make the next [`DeviceWire::wait`] treat its work as never completing, so it times out and aborts whatever the device does.
    pub(crate) fn force_timeout(&self) {
        self.lock().force_timeout = true;
    }
}
