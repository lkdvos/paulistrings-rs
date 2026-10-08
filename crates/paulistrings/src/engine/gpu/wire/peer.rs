//! [`PeerWire`], the [`DeviceWire`] of an in-process device group, and the peer access its cross-device copies use.

use std::collections::BTreeMap;
#[cfg(test)]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use cudarc::driver::{sys, CudaContext, CudaEvent, CudaStream};

use super::{wire_timeout, DeviceWire, WireOp, WireOpKind};
use crate::engine::gpu::error::GpuError;

/// Groups a [`PeerWire`] group's ranks have posted, summed over the ranks.
#[cfg(test)]
#[derive(Clone)]
pub(crate) struct PeerTally(Arc<Shared>);

#[cfg(test)]
impl PeerTally {
    /// Groups posted so far, one per rank per exchange.
    pub(crate) fn groups(&self) -> u64 {
        self.0.posted.load(Ordering::Relaxed)
    }
}

/// One posted op; a send carries an event marking where its stream stood when it was posted, shared by the post's sends on that stream.
struct Posted {
    kind: WireOpKind,
    peer: u32,
    ptr: u64,
    bytes: usize,
    stream: usize,
    ctx: Arc<CudaContext>,
    ready: Option<Arc<CudaEvent>>,
}

/// One generation of groups, one per rank.
struct Round {
    posted: Vec<Option<Vec<Posted>>>,
    done: u32,
    left: u32,
}

struct State {
    rounds: BTreeMap<u64, Round>,
    /// The first strictness failure or peer panic, which every waiter re-raises so no rank is left waiting on it.
    poisoned: Option<String>,
}

struct Shared {
    size: u32,
    state: Mutex<State>,
    condvar: Condvar,
    timeout: Duration,
    /// Groups posted by every rank so far.
    #[cfg(test)]
    posted: AtomicU64,
    /// The rank that fails and how (test hook).
    #[cfg(test)]
    fault: Option<(u32, PeerFault)>,
}

/// Where a [`PeerWire`] group's failing rank fails, returning an error as a failed NCCL call would.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PeerFault {
    /// Its first post fails before any op is visible to a peer.
    Post,
    /// Its first post succeeds and its wait fails once its peers have copied from it, with no copies of its own, so its peers' waits time out.
    Wait,
}

/// A [`DeviceWire`] over a group of in-process ranks: each rank's `n`-th group is matched against every other rank's `n`-th, per peer in posting order, and every receive is a device-to-device copy from its send on the receiver's stream, a peer copy across devices.
///
/// Stricter than NCCL, which would hang: a rank posting a different number of receives from a peer than the peer posts sends to it, or a receive whose size differs from its send, panics naming both ranks, and so does every other rank of the group.
pub(crate) struct PeerWire {
    rank: u32,
    shared: Arc<Shared>,
    /// This rank's next generation and the one it posted but has not waited on.
    generation: Mutex<(u64, Option<u64>)>,
    /// Set once this wire failed or timed out; every later call fails.
    dead: AtomicBool,
}

impl PeerWire {
    /// One wire per rank of a group of `size`, rank `r`'s at index `r`.
    pub(crate) fn group(size: u32) -> Vec<PeerWire> {
        Self::build(
            size,
            wire_timeout(),
            #[cfg(test)]
            None,
        )
    }

    /// A group whose rank `rank` fails as `fault` says, every other rank's wait then timing out after `timeout`.
    #[cfg(test)]
    pub(crate) fn group_with_fault(
        size: u32,
        rank: u32,
        fault: PeerFault,
        timeout: Duration,
    ) -> Vec<PeerWire> {
        Self::build(size, timeout, Some((rank, fault)))
    }

    fn build(
        size: u32,
        timeout: Duration,
        #[cfg(test)] fault: Option<(u32, PeerFault)>,
    ) -> Vec<PeerWire> {
        let shared = Arc::new(Shared {
            size,
            state: Mutex::new(State {
                rounds: BTreeMap::new(),
                poisoned: None,
            }),
            condvar: Condvar::new(),
            timeout,
            #[cfg(test)]
            posted: AtomicU64::new(0),
            #[cfg(test)]
            fault,
        });
        (0..size)
            .map(|rank| PeerWire {
                rank,
                shared: shared.clone(),
                generation: Mutex::new((0, None)),
                dead: AtomicBool::new(false),
            })
            .collect()
    }

    /// A handle that counts this wire's group's posted groups after the wires are handed out.
    #[cfg(test)]
    pub(crate) fn tally(&self) -> PeerTally {
        PeerTally(self.shared.clone())
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.shared
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    #[cfg(test)]
    fn faulty(&self, fault: PeerFault) -> bool {
        self.shared.fault == Some((self.rank, fault))
    }

    /// Record `msg` for the group and panic with it.
    fn fail(&self, mut state: MutexGuard<'_, State>, msg: String) -> ! {
        state.poisoned.get_or_insert_with(|| msg.clone());
        drop(state);
        self.shared.condvar.notify_all();
        panic!("{msg}");
    }

    /// Block until `ready` holds for `generation`, re-raising a peer's failure; a timeout kills the wire.
    fn wait_for<'s>(
        &'s self,
        mut state: MutexGuard<'s, State>,
        generation: u64,
        what: &str,
        ready: impl Fn(&Round) -> bool,
    ) -> Result<MutexGuard<'s, State>, GpuError> {
        let start = Instant::now();
        loop {
            if let Some(msg) = &state.poisoned {
                let msg = format!("peer wire, rank {}: a peer failed: {msg}", self.rank);
                drop(state);
                panic!("{msg}");
            }
            let round = state
                .rounds
                .get(&generation)
                .expect("a posted round stays until every rank left");
            if ready(round) {
                return Ok(state);
            }
            let elapsed = start.elapsed();
            if elapsed >= self.shared.timeout {
                let missing: Vec<usize> = (0..round.posted.len())
                    .filter(|&r| round.posted[r].is_none())
                    .collect();
                log::warn!(
                    "gpu: peer wire rank {}: group {generation} timed out waiting for {what} (ranks not posted: {missing:?})",
                    self.rank
                );
                self.dead.store(true, Ordering::Relaxed);
                return Err(GpuError::Timeout {
                    what: "a peer wire group",
                });
            }
            state = self
                .shared
                .condvar
                .wait_timeout(
                    state,
                    (self.shared.timeout - elapsed).min(Duration::from_millis(50)),
                )
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    fn alive(&self) -> Result<(), GpuError> {
        if self.dead.load(Ordering::Relaxed) {
            return Err(GpuError::Wire("a call on a failed peer wire"));
        }
        Ok(())
    }

    #[cfg(test)]
    fn die(&self, what: &'static str) -> GpuError {
        self.dead.store(true, Ordering::Relaxed);
        GpuError::Wire(what)
    }
}

/// A rank unwinding out of its layer poisons the group, so its peers fail at once instead of at the timeout.
impl Drop for PeerWire {
    fn drop(&mut self) {
        if std::thread::panicking() {
            let mut state = self.lock();
            let rank = self.rank;
            state
                .poisoned
                .get_or_insert_with(|| format!("rank {rank} panicked"));
            drop(state);
            self.shared.condvar.notify_all();
        }
    }
}

impl DeviceWire for PeerWire {
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
        #[cfg(test)]
        if self.faulty(PeerFault::Post) {
            return Err(self.die("an injected peer wire post failure"));
        }
        if ops.iter().any(|op| op.peer() >= size) {
            return Err(GpuError::Wire("a peer wire op to a peer outside the group"));
        }
        let mut posted = Vec::with_capacity(ops.len());
        let mut events: Vec<(usize, Arc<CudaEvent>)> = Vec::new();
        for op in ops {
            let stream = op.stream().cu_stream() as usize;
            let ready = match op.kind() {
                WireOpKind::Recv => None,
                WireOpKind::Send => Some(match events.iter().find(|(s, _)| *s == stream) {
                    Some((_, e)) => e.clone(),
                    None => {
                        let e = Arc::new(op.stream().record_event(None)?);
                        events.push((stream, e.clone()));
                        e
                    }
                }),
            };
            posted.push(Posted {
                kind: op.kind(),
                peer: op.peer(),
                ptr: op.ptr(),
                bytes: op.bytes(),
                stream,
                ctx: op.stream().context().clone(),
                ready,
            });
        }
        let generation = {
            let mut counter = self
                .generation
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            assert!(
                counter.1.is_none(),
                "peer wire, rank {}: a second group posted before the first was waited on",
                self.rank
            );
            let generation = counter.0;
            *counter = (generation + 1, Some(generation));
            generation
        };
        let mut state = self.lock();
        let round = state.rounds.entry(generation).or_insert_with(|| Round {
            posted: (0..size).map(|_| None).collect(),
            done: 0,
            left: 0,
        });
        round.posted[self.rank as usize] = Some(posted);
        drop(state);
        #[cfg(test)]
        self.shared.posted.fetch_add(1, Ordering::Relaxed);
        self.shared.condvar.notify_all();
        Ok(())
    }

    fn wait(&self, stream: &CudaStream) -> Result<(), GpuError> {
        self.alive()?;
        let pending = self
            .generation
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .1
            .take();
        let Some(generation) = pending else {
            stream.synchronize()?;
            return Ok(());
        };
        let (me, size) = (self.rank as usize, self.shared.size);
        let state = self.lock();
        let state = self.wait_for(state, generation, "every rank's group", |r| {
            r.posted.iter().all(Option::is_some)
        })?;
        #[cfg(test)]
        if self.faulty(PeerFault::Wait) {
            // The peers' copies read this rank's buffers, which its caller reuses as soon as this returns.
            drop(self.wait_for(state, generation, "the peers' copies", |r| {
                r.done + 1 >= size
            })?);
            return Err(self.die("an injected peer wire wait failure"));
        }
        let copies = match check_round(
            &state.rounds[&generation],
            me,
            size as usize,
            stream.cu_stream() as usize,
        ) {
            Ok(copies) => copies,
            Err(msg) => self.fail(state, msg),
        };
        drop(state);
        let enqueued: Result<(), GpuError> = (|| {
            let ctx = stream.context();
            ctx.bind_to_thread()?;
            let mut waited: Option<&Arc<CudaEvent>> = None;
            for c in &copies {
                if c.src_ctx.ordinal() != ctx.ordinal() {
                    enable_peer_access(ctx, &c.src_ctx);
                }
                if !waited.is_some_and(|w| Arc::ptr_eq(w, &c.ready)) {
                    stream.wait(&c.ready)?;
                    waited = Some(&c.ready);
                }
                if c.bytes > 0 {
                    // SAFETY: both addresses are live device allocations of at least `bytes` bytes, the receiver's borrowed by its caller until this wait returns and the sender's until its own wait returns, which is after every receiver's copy has completed.
                    unsafe {
                        cudarc::driver::result::memcpy_dtod_async(
                            c.dst,
                            c.src,
                            c.bytes,
                            stream.cu_stream(),
                        )?;
                    }
                }
            }
            Ok(())
        })();
        let synced = enqueued.and_then(|()| Ok(stream.synchronize()?));
        let mut state = self.lock();
        state
            .rounds
            .get_mut(&generation)
            .expect("the round is live")
            .done += 1;
        self.shared.condvar.notify_all();
        let mut state =
            self.wait_for(state, generation, "every rank's copies", |r| r.done == size)?;
        let round = state
            .rounds
            .get_mut(&generation)
            .expect("the round is live");
        round.left += 1;
        if round.left == size {
            state.rounds.remove(&generation);
        }
        synced
    }
}

/// One receive's copy from its matching send.
struct PeerCopy {
    dst: u64,
    src: u64,
    bytes: usize,
    src_ctx: Arc<CudaContext>,
    ready: Arc<CudaEvent>,
}

/// Rank `me`'s copies of round `round`, or the strictness failure naming both ranks.
fn check_round(
    round: &Round,
    me: usize,
    size: usize,
    stream: usize,
) -> Result<Vec<PeerCopy>, String> {
    let mine = round.posted[me].as_ref().expect("every rank posted");
    if let Some(r) = mine
        .iter()
        .find(|p| p.kind == WireOpKind::Recv && p.stream != stream)
    {
        return Err(format!(
            "peer wire: rank {me} waits on another stream than its receive from rank {} was posted on",
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
                "peer wire: rank {me} posts {} receives from rank {q}, which posts {} sends to rank {me}",
                recvs.len(),
                sends.len()
            ));
        }
        for (n, (&i, &j)) in recvs.iter().zip(&sends).enumerate() {
            if mine[i].bytes != theirs[j].bytes {
                return Err(format!(
                    "peer wire: receive {n} of rank {me} from rank {q} is {} bytes, rank {q}'s send {n} to rank {me} is {}",
                    mine[i].bytes, theirs[j].bytes
                ));
            }
            let (recv, send) = (&mine[i], &theirs[j]);
            copies.push(PeerCopy {
                dst: recv.ptr,
                src: send.ptr,
                bytes: recv.bytes,
                src_ctx: send.ctx.clone(),
                ready: send.ready.clone().expect("a send records its event"),
            });
        }
    }
    Ok(copies)
}

/// Whether a device-to-device copy between two devices goes direct or through the host.
#[derive(Clone, Debug, PartialEq, Eq)]
enum PeerAccess {
    /// Both ends are the same device.
    SameDevice,
    /// The destination's context and the source's memory pool both map the source's memory into the destination, so copies go over NVLink or PCIe peer-to-peer.
    Enabled,
    /// The driver reports the pair cannot access each other; copies stage through the host.
    Unsupported,
    /// The driver allows the pair but enabling failed with this error; copies stage through the host.
    Failed(String),
}

/// Enable direct access from `dst`'s context to `src`'s memory, once per ordered pair; a pair without it stays correct through host staging, so the outcome is logged, not an error.
fn enable_peer_access(dst: &Arc<CudaContext>, src: &Arc<CudaContext>) -> PeerAccess {
    static DONE: Mutex<Vec<((usize, usize), PeerAccess)>> = Mutex::new(Vec::new());
    let pair = (dst.ordinal(), src.ordinal());
    if pair.0 == pair.1 {
        return PeerAccess::SameDevice;
    }
    let mut done = DONE.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, access)) = done.iter().find(|(p, _)| *p == pair) {
        return access.clone();
    }
    let access = try_enable_peer_access(dst, src, pair);
    match &access {
        PeerAccess::Enabled => log::info!("gpu: peer access {} -> {} enabled", pair.1, pair.0),
        other => log::warn!(
            "gpu: peer access {} -> {} is {other:?}; copies between them stage through the host",
            pair.1,
            pair.0
        ),
    }
    done.push((pair, access.clone()));
    access
}

fn try_enable_peer_access(
    dst: &Arc<CudaContext>,
    src: &Arc<CudaContext>,
    pair: (usize, usize),
) -> PeerAccess {
    let failed = |e: sys::CUresult| PeerAccess::Failed(format!("{e:?}"));
    let mut can = 0i32;
    // SAFETY: a driver query on two live ordinals.
    if let Err(e) =
        unsafe { sys::cuDeviceCanAccessPeer(&mut can, pair.0 as i32, pair.1 as i32) }.result()
    {
        return failed(e.0);
    }
    if can == 0 {
        return PeerAccess::Unsupported;
    }
    if let Err(e) = dst.bind_to_thread() {
        return failed(e.0);
    }
    // SAFETY: `dst` is the bound context and `src`'s handle stays valid while `src` is alive.
    match unsafe { sys::cuCtxEnablePeerAccess(src.cu_ctx(), 0) } {
        sys::CUresult::CUDA_SUCCESS | sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => {}
        e => return failed(e),
    }
    // cudarc allocates through `cuMemAllocAsync`, and pool memory is mapped to a peer only by the pool's own access list: without this, peer copies stage through the host and peer loads fault.
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
    // SAFETY: a driver query on a live ordinal.
    if let Err(e) = unsafe { sys::cuDeviceGetMemPool(&mut pool, pair.1 as i32) }.result() {
        return failed(e.0);
    }
    let desc = sys::CUmemAccessDesc {
        location: sys::CUmemLocation {
            type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: pair.0 as i32,
        },
        flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };
    // SAFETY: `pool` is `src`'s current pool and `desc` one valid entry.
    match unsafe { sys::cuMemPoolSetAccess(pool, &desc, 1) }.result() {
        Ok(()) => PeerAccess::Enabled,
        Err(e) => failed(e.0),
    }
}

#[cfg(test)]
mod tests;
