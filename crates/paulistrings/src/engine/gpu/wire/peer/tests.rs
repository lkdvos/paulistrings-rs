use super::super::WireGroup;
use super::*;

fn on_ranks<R: Send>(size: u32, f: impl Fn(PeerWire) -> R + Sync) -> Vec<std::thread::Result<R>> {
    let wires = PeerWire::group(size);
    std::thread::scope(|s| {
        let f = &f;
        let hs: Vec<_> = wires.into_iter().map(|w| s.spawn(move || f(w))).collect();
        hs.into_iter().map(|h| h.join()).collect()
    })
}

/// The panic message every rank of `out` ended on.
fn panics(out: Vec<std::thread::Result<()>>) -> Vec<String> {
    out.into_iter()
        .map(|r| {
            let e = r.expect_err("every rank panics");
            e.downcast_ref::<String>().cloned().unwrap_or_default()
        })
        .collect()
}

/// Each rank sends every other rank two messages of distinct sizes and contents and receives both of each peer's into one concatenated column.
#[test]
fn every_pair_exchanges_in_posting_order() {
    crate::require_cuda!();
    let size = 3u32;
    let msg = |from: u32, to: u32, m: u64| -> Vec<u64> {
        let n = 1 + from as usize * 7 + to as usize * 3 + m as usize * 11;
        (0..n as u64)
            .map(|i| (u64::from(from) << 40) ^ (u64::from(to) << 32) ^ (m << 24) ^ i)
            .collect()
    };
    let out = on_ranks(size, |wire| {
        let me = wire.rank();
        let context = crate::engine::gpu::device::context(0).expect("device 0");
        let stream = context.new_stream().expect("stream");
        let peers: Vec<u32> = (0..size).filter(|&q| q != me).collect();
        let sends: Vec<_> = peers
            .iter()
            .flat_map(|&q| (0..2).map(move |m| (q, m)))
            .map(|(q, m)| (q, stream.clone_htod(&msg(me, q, m)).expect("upload")))
            .collect();
        let parts: Vec<(usize, u32)> = peers
            .iter()
            .flat_map(|&q| (0..2).map(move |m| (msg(q, me, m).len(), q)))
            .collect();
        let total: usize = parts.iter().map(|p| p.0).sum();
        let mut dst = stream.alloc_zeros::<u64>(total).expect("alloc");
        let mut group = WireGroup::new();
        for (q, s) in &sends {
            group.send(s.as_view(), *q, &stream);
        }
        group.recv_parts(dst.as_view_mut(), &parts, &stream);
        group.post(&wire).expect("post");
        wire.wait(&stream).expect("wait");
        let got = stream.clone_dtoh(&dst).expect("download");
        let want: Vec<u64> = peers
            .iter()
            .flat_map(|&q| (0..2).flat_map(move |m| msg(q, me, m)))
            .collect();
        assert_eq!(got, want, "rank {me}");
    });
    for r in out {
        r.expect("every rank completes");
    }
}

/// A receive with no matching send fails every rank, and the failing rank's message names both ends.
#[test]
fn an_unmatched_receive_panics_naming_both_ranks() {
    crate::require_cuda!();
    let out = on_ranks(2, |wire| {
        let context = crate::engine::gpu::device::context(0).expect("device 0");
        let stream = context.new_stream().expect("stream");
        let mut dst = stream.alloc_zeros::<u64>(4).expect("alloc");
        let mut group = WireGroup::new();
        if wire.rank() == 0 {
            group.recv(dst.as_view_mut(), 1, &stream);
        }
        group.post(&wire).expect("post");
        wire.wait(&stream).expect("wait");
    });
    let msgs = panics(out);
    assert!(
        msgs.iter()
            .all(|m| m.contains("rank 0 posts 1 receives from rank 1, which posts 0 sends")),
        "{msgs:?}"
    );
}

#[test]
fn a_size_mismatch_panics_naming_both_ranks() {
    crate::require_cuda!();
    let out = on_ranks(2, |wire| {
        let context = crate::engine::gpu::device::context(0).expect("device 0");
        let stream = context.new_stream().expect("stream");
        let src = stream.alloc_zeros::<u64>(8).expect("alloc");
        let mut dst = stream.alloc_zeros::<u64>(8).expect("alloc");
        let mut group = WireGroup::new();
        let peer = 1 - wire.rank();
        let n = if wire.rank() == 0 { 8 } else { 4 };
        group.send(src.slice(0..n), peer, &stream);
        group.recv(dst.slice_mut(0..8), peer, &stream);
        group.post(&wire).expect("post");
        wire.wait(&stream).expect("wait");
    });
    let msgs = panics(out);
    assert!(
        msgs.iter().all(|m| m.contains(
            "receive 0 of rank 0 from rank 1 is 64 bytes, rank 1's send 0 to rank 0 is 32"
        )),
        "{msgs:?}"
    );
}

#[test]
fn peer_access_is_reported_for_every_pair() {
    crate::require_cuda!();
    let n = crate::engine::gpu::device_count();
    for dst in 0..n {
        for src in 0..n {
            let context =
                |o: usize| crate::engine::gpu::device::context(o as u32).expect("visible");
            let access = enable_peer_access(&context(dst), &context(src));
            if dst == src {
                assert_eq!(access, PeerAccess::SameDevice);
            } else {
                assert!(
                    !matches!(access, PeerAccess::Failed(_)),
                    "{dst} <- {src}: {access:?}"
                );
            }
        }
    }
}
