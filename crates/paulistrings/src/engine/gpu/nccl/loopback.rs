//! [`LoopbackWire`], an in-process [`DeviceWire`] for testing the NCCL exchange with every rank in one process.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use cudarc::driver::{CudaEvent, CudaStream};

use super::{DeviceWire, WireOp, WireOpKind};
use crate::engine::gpu::error::GpuError;

/// Groups a [`LoopbackWire`] group's ranks have posted, summed over the ranks.
#[derive(Clone)]
pub struct LoopbackTally(Arc<Shared>);

impl LoopbackTally {
    /// Groups posted so far, one per rank per exchange.
    pub fn groups(&self) -> u64 {
        self.0.posted.load(Ordering::Relaxed)
    }
}

/// One posted op, with an event marking where its stream stood when it was posted.
struct Posted {
    kind: WireOpKind,
    peer: u32,
    ptr: u64,
    bytes: usize,
    stream: usize,
    ready: CudaEvent,
}

/// One generation of groups, one per rank.
struct Round {
    posted: Vec<Option<Vec<Posted>>>,
    done: u32,
    left: u32,
}

struct State {
    rounds: BTreeMap<u64, Round>,
    /// The first strictness failure, which every waiter re-raises so no rank is left waiting on a panicked peer.
    poisoned: Option<String>,
}

struct Shared {
    size: u32,
    state: Mutex<State>,
    cv: Condvar,
    timeout: Duration,
    /// Groups posted by every rank so far.
    posted: AtomicU64,
    /// The rank that fails and how; with one set, a timeout is an error, as NCCL's bounded wait is, instead of a panic.
    fault: Option<(u32, LoopbackFault)>,
}

/// Where a [`LoopbackWire`] group's failing rank fails, returning an error as a failed NCCL call would.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopbackFault {
    /// Its first post fails before any op is visible to a peer.
    Post,
    /// Its first post succeeds and its wait fails once its peers have copied from it, with no copies of its own, so its peers' waits time out.
    Wait,
}

/// A [`DeviceWire`] over a group of in-process ranks: each rank's `n`-th group is matched against every other rank's `n`-th, per peer in posting order, and every receive is a device-to-device copy from its send.
///
/// Stricter than NCCL, which would hang: a rank posting a different number of receives from a peer than the peer posts sends to it, or a receive whose size differs from its send, panics naming both ranks, and so does every other rank of the group.
/// Every rank must post a group whenever one does, as [`DeviceWire::post`] requires of any peer an op names.
/// A test hands one to each rank's split through `GpuDistributedSum::use_loopback_wire` (feature `test-utils`).
pub struct LoopbackWire {
    rank: u32,
    shared: Arc<Shared>,
    /// This rank's next generation and the one it posted but has not waited on.
    gen: Mutex<(u64, Option<u64>)>,
    /// Set once this wire failed or timed out; every later call fails.
    dead: AtomicBool,
}

impl LoopbackWire {
    /// One wire per rank of a group of `size`, rank `r`'s at index `r`.
    pub fn group(size: u32) -> Vec<LoopbackWire> {
        Self::group_with_timeout(size, Duration::from_secs(120))
    }

    /// As [`group`](Self::group), panicking after `timeout` if a peer never posts or never finishes.
    pub fn group_with_timeout(size: u32, timeout: Duration) -> Vec<LoopbackWire> {
        Self::build(size, timeout, None)
    }

    /// A group whose rank `rank` fails as `fault` says, every other rank's wait then timing out after `timeout` with [`GpuError::Timeout`].
    pub fn group_with_fault(
        size: u32,
        rank: u32,
        fault: LoopbackFault,
        timeout: Duration,
    ) -> Vec<LoopbackWire> {
        Self::build(size, timeout, Some((rank, fault)))
    }

    fn build(
        size: u32,
        timeout: Duration,
        fault: Option<(u32, LoopbackFault)>,
    ) -> Vec<LoopbackWire> {
        let shared = Arc::new(Shared {
            size,
            state: Mutex::new(State {
                rounds: BTreeMap::new(),
                poisoned: None,
            }),
            cv: Condvar::new(),
            timeout,
            posted: AtomicU64::new(0),
            fault,
        });
        (0..size)
            .map(|rank| LoopbackWire {
                rank,
                shared: shared.clone(),
                gen: Mutex::new((0, None)),
                dead: AtomicBool::new(false),
            })
            .collect()
    }

    /// A handle that counts this wire's group's posted groups after the wires are handed out.
    pub fn tally(&self) -> LoopbackTally {
        LoopbackTally(self.shared.clone())
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.shared
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Record `msg` for the group and panic with it.
    fn fail(&self, mut st: MutexGuard<'_, State>, msg: String) -> ! {
        st.poisoned.get_or_insert_with(|| msg.clone());
        drop(st);
        self.shared.cv.notify_all();
        panic!("{msg}");
    }

    /// Block until `ready` holds for generation `g`, re-raising a peer's panic and panicking at the timeout.
    fn wait_for<'s>(
        &'s self,
        mut st: MutexGuard<'s, State>,
        g: u64,
        what: &str,
        ready: impl Fn(&Round) -> bool,
    ) -> Result<MutexGuard<'s, State>, GpuError> {
        let start = Instant::now();
        loop {
            if let Some(msg) = &st.poisoned {
                let msg = format!("loopback wire, rank {}: a peer failed: {msg}", self.rank);
                drop(st);
                panic!("{msg}");
            }
            let round = st
                .rounds
                .get(&g)
                .expect("a posted round stays until every rank left");
            if ready(round) {
                return Ok(st);
            }
            let elapsed = start.elapsed();
            if elapsed >= self.shared.timeout {
                if self.shared.fault.is_some() {
                    self.dead.store(true, Ordering::Relaxed);
                    return Err(GpuError::Timeout {
                        what: "a loopback wire group",
                    });
                }
                let missing: Vec<usize> = round
                    .posted
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.is_none())
                    .map(|(r, _)| r)
                    .collect();
                let msg = format!(
                    "loopback wire, rank {}: group {g} timed out waiting for {what} (ranks not posted: {missing:?})",
                    self.rank
                );
                self.fail(st, msg);
            }
            st = self
                .shared
                .cv
                .wait_timeout(
                    st,
                    (self.shared.timeout - elapsed).min(Duration::from_millis(50)),
                )
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }
}

impl LoopbackWire {
    fn alive(&self) -> Result<(), GpuError> {
        if self.dead.load(Ordering::Relaxed) {
            return Err(GpuError::Nccl {
                code: sys_invalid_argument(),
                what: "a call on a failed loopback wire".to_string(),
            });
        }
        Ok(())
    }

    fn die(&self, what: &str) -> GpuError {
        self.dead.store(true, Ordering::Relaxed);
        GpuError::Nccl {
            code: cudarc::nccl::sys::ncclResult_t::ncclSystemError as i32,
            what: what.to_string(),
        }
    }
}

impl DeviceWire for LoopbackWire {
    fn rank(&self) -> u32 {
        self.rank
    }

    fn size(&self) -> u32 {
        self.shared.size
    }

    fn abort(&self) {
        self.dead.store(true, Ordering::Relaxed);
    }

    fn is_healthy(&self) -> bool {
        !self.dead.load(Ordering::Relaxed)
    }

    fn post(&self, ops: &[WireOp<'_>]) -> Result<(), GpuError> {
        let size = self.shared.size;
        self.alive()?;
        if self.shared.fault == Some((self.rank, LoopbackFault::Post)) {
            return Err(self.die("an injected loopback post failure"));
        }
        if let Some(op) = ops.iter().find(|op| op.peer() >= size) {
            return Err(GpuError::Nccl {
                code: sys_invalid_argument(),
                what: format!("a loopback op to peer {} of {size}", op.peer()),
            });
        }
        let mut posted = Vec::with_capacity(ops.len());
        for op in ops {
            posted.push(Posted {
                kind: op.kind(),
                peer: op.peer(),
                ptr: op.ptr(),
                bytes: op.bytes(),
                stream: op.stream().cu_stream() as usize,
                ready: op.stream().record_event(None)?,
            });
        }
        let g = {
            let mut gen = self.gen.lock().unwrap_or_else(PoisonError::into_inner);
            assert!(
                gen.1.is_none(),
                "loopback wire, rank {}: a second group posted before the first was waited on",
                self.rank
            );
            let g = gen.0;
            *gen = (g + 1, Some(g));
            g
        };
        let mut st = self.lock();
        let round = st.rounds.entry(g).or_insert_with(|| Round {
            posted: (0..size).map(|_| None).collect(),
            done: 0,
            left: 0,
        });
        round.posted[self.rank as usize] = Some(posted);
        drop(st);
        self.shared.posted.fetch_add(1, Ordering::Relaxed);
        self.shared.cv.notify_all();
        Ok(())
    }

    fn wait(&self, stream: &CudaStream) -> Result<(), GpuError> {
        self.alive()?;
        let pending = self
            .gen
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .1
            .take();
        let Some(g) = pending else {
            stream.synchronize()?;
            return Ok(());
        };
        let (me, size) = (self.rank as usize, self.shared.size as usize);
        if self.shared.fault == Some((self.rank, LoopbackFault::Wait)) {
            // The peers' copies read this rank's buffers, which its caller recycles as soon as this returns.
            let st = self.lock();
            let st = self.wait_for(st, g, "every rank's group", |r| {
                r.posted.iter().all(Option::is_some)
            })?;
            drop(self.wait_for(st, g, "the peers' copies", |r| r.done + 1 >= size as u32)?);
            return Err(self.die("an injected loopback wait failure"));
        }
        let st = self.lock();
        let st = self.wait_for(st, g, "every rank's group", |r| {
            r.posted.iter().all(Option::is_some)
        })?;
        let copies = match check_round(&st.rounds[&g], me, size, stream.cu_stream() as usize) {
            Ok(copies) => copies,
            Err(msg) => self.fail(st, msg),
        };
        let enqueued: Result<(), GpuError> = (|| {
            stream.context().bind_to_thread()?;
            let round = &st.rounds[&g];
            for (q, i, j) in copies {
                let recv = &round.posted[me].as_ref().expect("checked")[i];
                let send = &round.posted[q].as_ref().expect("checked")[j];
                stream.wait(&recv.ready)?;
                stream.wait(&send.ready)?;
                if recv.bytes > 0 {
                    // SAFETY: both addresses are live device allocations of at least `bytes` bytes, the receiver's borrowed by its caller until this wait returns and the sender's until its own wait returns, which is after every receiver's copy has completed.
                    unsafe {
                        cudarc::driver::result::memcpy_dtod_async(
                            recv.ptr,
                            send.ptr,
                            recv.bytes,
                            stream.cu_stream(),
                        )?;
                    }
                }
            }
            Ok(())
        })();
        drop(st);
        let synced = enqueued.and_then(|()| Ok(stream.synchronize()?));
        let mut st = self.lock();
        st.rounds.get_mut(&g).expect("the round is live").done += 1;
        self.shared.cv.notify_all();
        let mut st = self.wait_for(st, g, "every rank's copies", |r| r.done == size as u32)?;
        let round = st.rounds.get_mut(&g).expect("the round is live");
        round.left += 1;
        if round.left == size as u32 {
            st.rounds.remove(&g);
        }
        synced
    }
}

/// Rank `me`'s copies of round `round` as `(peer, receive index, send index)`, or the strictness failure naming both ranks.
fn check_round(
    round: &Round,
    me: usize,
    size: usize,
    stream: usize,
) -> Result<Vec<(usize, usize, usize)>, String> {
    let mine = round.posted[me].as_ref().expect("every rank posted");
    if let Some(r) = mine
        .iter()
        .find(|p| p.kind == WireOpKind::Recv && p.stream != stream)
    {
        return Err(format!(
            "loopback wire: rank {me} waits on another stream than its receive from rank {} was posted on",
            r.peer
        ));
    }
    let mut copies = Vec::new();
    for q in 0..size {
        let theirs = round.posted[q].as_ref().expect("every rank posted");
        let recvs: Vec<usize> = (0..mine.len())
            .filter(|&i| mine[i].kind == WireOpKind::Recv && mine[i].peer as usize == q)
            .collect();
        let sends: Vec<usize> = (0..theirs.len())
            .filter(|&j| theirs[j].kind == WireOpKind::Send && theirs[j].peer as usize == me)
            .collect();
        if recvs.len() != sends.len() {
            return Err(format!(
                "loopback wire: rank {me} posts {} receives from rank {q}, which posts {} sends to rank {me}",
                recvs.len(),
                sends.len()
            ));
        }
        for (n, (&i, &j)) in recvs.iter().zip(&sends).enumerate() {
            if mine[i].bytes != theirs[j].bytes {
                return Err(format!(
                    "loopback wire: receive {n} of rank {me} from rank {q} is {} bytes, rank {q}'s send {n} to rank {me} is {}",
                    mine[i].bytes, theirs[j].bytes
                ));
            }
            copies.push((q, i, j));
        }
    }
    Ok(copies)
}

fn sys_invalid_argument() -> i32 {
    cudarc::nccl::sys::ncclResult_t::ncclInvalidArgument as i32
}

#[cfg(test)]
mod tests {
    use super::super::WireGroup;
    use super::*;

    fn on_ranks<R: Send>(
        size: u32,
        f: impl Fn(LoopbackWire) -> R + Sync,
    ) -> Vec<std::thread::Result<R>> {
        let wires = LoopbackWire::group_with_timeout(size, Duration::from_secs(30));
        std::thread::scope(|s| {
            let f = &f;
            let hs: Vec<_> = wires.into_iter().map(|w| s.spawn(move || f(w))).collect();
            hs.into_iter().map(|h| h.join()).collect()
        })
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
            let ctx = crate::engine::gpu::device::context(0).expect("device 0");
            let stream = ctx.new_stream().expect("stream");
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
            let ctx = crate::engine::gpu::device::context(0).expect("device 0");
            let stream = ctx.new_stream().expect("stream");
            let mut dst = stream.alloc_zeros::<u64>(4).expect("alloc");
            let mut group = WireGroup::new();
            if wire.rank() == 0 {
                group.recv(dst.as_view_mut(), 1, &stream);
            }
            group.post(&wire).expect("post");
            wire.wait(&stream).expect("wait");
        });
        let msgs: Vec<String> = out
            .into_iter()
            .map(|r| {
                let e = r.expect_err("every rank panics");
                e.downcast_ref::<String>().cloned().unwrap_or_default()
            })
            .collect();
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
            let ctx = crate::engine::gpu::device::context(0).expect("device 0");
            let stream = ctx.new_stream().expect("stream");
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
        let msgs: Vec<String> = out
            .into_iter()
            .map(|r| {
                let e = r.expect_err("every rank panics");
                e.downcast_ref::<String>().cloned().unwrap_or_default()
            })
            .collect();
        assert!(
            msgs.iter().all(|m| m.contains(
                "receive 0 of rank 0 from rank 1 is 64 bytes, rank 1's send 0 to rank 0 is 32"
            )),
            "{msgs:?}"
        );
    }
}
