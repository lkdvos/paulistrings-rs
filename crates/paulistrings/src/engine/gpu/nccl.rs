//! [`NcclComm`], the NCCL communicator of a device group over MPI, the [`DeviceWire`] seam the device exchange posts its bulk columns through with [`NcclWire`] its NCCL form, and the exchange's host half: [`BlockSkeletons`], [`nccl_schedule`] and the group's mode agreement.
//! See ARCHITECTURE.md §Partitioning.

#[cfg(any(test, feature = "test-utils"))]
mod loopback;

use std::ffi::CStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaStream, CudaView, CudaViewMut, SyncOnDrop};
use cudarc::nccl::result::{self as nccl, NcclError, NcclStatus};
use cudarc::nccl::sys;

use super::error::GpuError;
use super::payload::DeviceBlock;
use crate::engine::partitioned::transport::{
    chunk_rows_of, BlockHeader, ChunkMap, Collectives, Payload,
};

#[cfg(any(test, feature = "test-utils"))]
pub use loopback::{LoopbackFault, LoopbackTally, LoopbackWire};

/// The oldest runtime `libnccl` the `nccl-02022` bindings are sound against, as `ncclGetVersion` codes it.
pub(crate) const MIN_NCCL_VERSION: i32 = 22200;

/// The bound on every wait when `PAULISTRINGS_NCCL_TIMEOUT_S` is unset.
pub(crate) const DEFAULT_NCCL_TIMEOUT: Duration = Duration::from_secs(300);

/// `sizeof(ncclUniqueId)` in `u64` words.
const ID_WORDS: usize = 16;

/// `PAULISTRINGS_NCCL_TIMEOUT_S` as a positive number of seconds, else [`DEFAULT_NCCL_TIMEOUT`].
pub(crate) fn nccl_timeout() -> Duration {
    parse_timeout(std::env::var("PAULISTRINGS_NCCL_TIMEOUT_S").ok().as_deref())
}

fn parse_timeout(raw: Option<&str>) -> Duration {
    raw.and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s > 0.0)
        .map_or(DEFAULT_NCCL_TIMEOUT, Duration::from_secs_f64)
}

/// Poll `probe` until it reports ready (`true`), fails, or `timeout` elapses (`Ok(false)`).
fn poll_until(
    timeout: Duration,
    mut probe: impl FnMut() -> Result<bool, GpuError>,
) -> Result<bool, GpuError> {
    let start = Instant::now();
    let mut spins = 0u32;
    loop {
        if probe()? {
            return Ok(true);
        }
        if start.elapsed() >= timeout {
            return Ok(false);
        }
        if spins < 1024 {
            spins += 1;
            std::thread::yield_now();
        } else {
            std::thread::sleep(Duration::from_micros(50));
        }
    }
}

fn nccl_error(e: NcclError, what: impl Into<String>) -> GpuError {
    GpuError::Nccl {
        code: e.0 as i32,
        what: what.into(),
    }
}

/// The communicator handle and whether it may still be used.
struct Raw {
    comm: sys::ncclComm_t,
    /// Set once the communicator has been aborted, after which `comm` is dangling.
    aborted: bool,
    timeout: Duration,
    /// Test hook: the next completion wait sees its work as never finishing.
    #[cfg(any(test, feature = "test-utils"))]
    force_timeout: bool,
}

// SAFETY: an NCCL communicator may be driven from any thread provided no two calls on it overlap, which the `Mutex` around every `Raw` enforces.
unsafe impl Send for Raw {}

/// One rank's non-blocking NCCL communicator, every wait bounded; a failed or timed-out call aborts it and later calls fail without touching NCCL.
/// Drop finalizes and destroys a healthy one (bounded) and aborts any other, never panicking.
pub(crate) struct NcclComm {
    raw: Mutex<Raw>,
    ctx: Arc<CudaContext>,
    rank: u32,
    size: u32,
}

/// The warm-up's per-peer byte count.
const WARM_UP_BYTES: usize = 8;

impl NcclComm {
    /// [`init_with_timeout`](Self::init_with_timeout) with [`nccl_timeout`].
    pub(crate) fn init(coll: &dyn Collectives, ctx: &Arc<CudaContext>) -> Result<Self, GpuError> {
        Self::init_with_timeout(coll, ctx, nccl_timeout())
    }

    /// This rank's communicator over `coll`'s group on `ctx`'s device. **Collective**: one `allreduce_sum_u64` whatever the outcome, carrying rank 0's id and every rank's readiness.
    /// A setup that fails after it aborts and fails on this rank alone, so the caller agrees the outcome before [`warm_up`](Self::warm_up).
    pub(crate) fn init_with_timeout(
        coll: &dyn Collectives,
        ctx: &Arc<CudaContext>,
        timeout: Duration,
    ) -> Result<Self, GpuError> {
        let (rank, size) = (coll.rank(), coll.size());
        let local = local_readiness();
        let id = match (&local, rank) {
            (Ok(()), 0) => nccl::get_uniqueid().map_err(|e| nccl_error(e, "ncclGetUniqueId")),
            _ => Ok(sys::ncclUniqueId { internal: [0; 128] }),
        };
        let mut buf = [0u64; ID_WORDS + 1];
        match &id {
            Ok(id) if local.is_ok() => pack_id(id, &mut buf[..ID_WORDS]),
            _ => buf[ID_WORDS] = 1,
        }
        coll.allreduce_sum_u64(&mut buf);
        local?;
        id?;
        if buf[ID_WORDS] != 0 {
            return Err(GpuError::Unsupported(
                "a peer rank cannot start NCCL (libnccl missing, older than 2.22, or no unique id)",
            ));
        }
        let id = unpack_id(&buf[..ID_WORDS]);
        ctx.bind_to_thread()?;
        let mut config = default_config();
        config.blocking = 0;
        let mut comm: sys::ncclComm_t = std::ptr::null_mut();
        // SAFETY: `comm` and `config` are live locals, and `config` is initialized as `NCCL_CONFIG_INITIALIZER` does.
        let started = unsafe {
            nccl::comm_init_rank_config(&mut comm, size as i32, id, rank as i32, &mut config)
        };
        LIVE.fetch_add(1, Ordering::Relaxed);
        let this = Self {
            raw: Mutex::new(Raw {
                comm,
                aborted: comm.is_null(),
                timeout,
                #[cfg(any(test, feature = "test-utils"))]
                force_timeout: false,
            }),
            ctx: ctx.clone(),
            rank,
            size,
        };
        if let Err(e) = started {
            this.abort();
            return Err(nccl_error(e, "ncclCommInitRankConfig"));
        }
        this.settle("ncclCommInitRankConfig", "NCCL communicator init")?;
        log::info!("gpu: NCCL communicator up, rank {rank} of {size}");
        Ok(this)
    }

    /// This rank's index in the communicator.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn rank(&self) -> u32 {
        self.rank
    }

    /// Ranks in the communicator.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn size(&self) -> u32 {
        self.size
    }

    /// The bound on every wait on this communicator.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn timeout(&self) -> Duration {
        self.lock().timeout
    }

    /// Replace the wait bound, so a test can force a timeout without waiting out the default.
    #[cfg(any(test, feature = "test-utils"))]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn set_timeout(&self, timeout: Duration) {
        self.lock().timeout = timeout;
    }

    /// Make the next [`DeviceWire::wait`] treat its work as never completing, so it times out and aborts whatever the device does.
    #[cfg(any(test, feature = "test-utils"))]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn force_timeout(&self) {
        self.lock().force_timeout = true;
    }

    /// Whether the communicator has not been aborted.
    pub(crate) fn is_healthy(&self) -> bool {
        !self.lock().aborted
    }

    /// Pay NCCL's lazy connection setup now: one byte-sized send/recv with every other rank (with itself in a one-rank world), completed on `stream`.
    /// **Collective over the communicator**: every rank calls it, after the group has agreed that every [`init`](Self::init) succeeded.
    pub(crate) fn warm_up(&self, stream: &Arc<CudaStream>) -> Result<(), GpuError> {
        let peers: Vec<u32> = if self.size == 1 {
            vec![0]
        } else {
            (0..self.size).filter(|&q| q != self.rank).collect()
        };
        let n = peers.len() * WARM_UP_BYTES;
        let out = stream.alloc_zeros::<u8>(n)?;
        let mut back = stream.alloc_zeros::<u8>(n)?;
        let mut group = WireGroup::new();
        for (i, &q) in peers.iter().enumerate() {
            group.send(
                out.slice(i * WARM_UP_BYTES..(i + 1) * WARM_UP_BYTES),
                q,
                stream,
            );
        }
        let parts: Vec<(usize, u32)> = peers.iter().map(|&q| (WARM_UP_BYTES, q)).collect();
        group.recv_parts(back.as_view_mut(), &parts, stream);
        group.post_with(|ops| self.post(ops))?;
        self.wait(stream)
    }

    /// NCCL's asynchronous state: `Ok(true)` while an operation is still in progress, `Ok(false)` when idle, and on an error the communicator aborted and the error returned.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn check_async(&self) -> Result<bool, GpuError> {
        let mut raw = self.lock();
        self.check_async_locked(&mut raw, "ncclCommGetAsyncError")
    }

    /// Abort the communicator if it is still live; idempotent, and returns at once (see `abort_locked`).
    pub(crate) fn abort(&self) {
        let mut raw = self.lock();
        Self::abort_locked(&mut raw, self.rank, &self.ctx);
    }

    fn lock(&self) -> MutexGuard<'_, Raw> {
        self.raw.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Mark the communicator dead and abort it on a detached thread.
    /// `ncclCommAbort` blocks until every stream on the device drains, so a stall that is not NCCL's own would otherwise hold the caller past its timeout.
    fn abort_locked(raw: &mut Raw, rank: u32, ctx: &Arc<CudaContext>) {
        if raw.aborted {
            return;
        }
        raw.aborted = true;
        let comm = CommPtr(std::mem::replace(&mut raw.comm, std::ptr::null_mut()));
        let owned = ctx.clone();
        let spawned = std::thread::Builder::new()
            .name(format!("nccl-abort-{rank}"))
            .spawn(move || abort_now(comm, rank, &owned));
        match spawned {
            Ok(handle) => ABORTS
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(handle),
            Err(e) => {
                log::warn!(
                    "gpu: no thread for the NCCL abort on rank {rank} ({e}); aborting inline"
                );
                abort_now(comm, rank, ctx);
            }
        }
    }

    fn usable(raw: &Raw, what: &str) -> Result<(), GpuError> {
        if raw.aborted {
            return Err(GpuError::Nccl {
                code: sys::ncclResult_t::ncclInvalidUsage as i32,
                what: format!("{what} on an aborted NCCL communicator"),
            });
        }
        Ok(())
    }

    /// `ncclCommGetAsyncError`, aborting on an error.
    fn check_async_locked(&self, raw: &mut Raw, what: &str) -> Result<bool, GpuError> {
        Self::usable(raw, what)?;
        let mut state = sys::ncclResult_t::ncclSuccess;
        // SAFETY: `comm` is live (not aborted) and `state` a live local.
        let polled = unsafe { sys::ncclCommGetAsyncError(raw.comm, &mut state) }
            .result()
            .and_then(|_| state.result());
        match polled {
            Ok(NcclStatus::InProgress) => Ok(true),
            Ok(_) => Ok(false),
            Err(e) => {
                let detail = self.last_error(raw);
                Self::abort_locked(raw, self.rank, &self.ctx);
                Err(nccl_error(e, format!("{what}{detail}")))
            }
        }
    }

    /// `": <message>"` from `ncclGetLastError`, or empty.
    fn last_error(&self, raw: &Raw) -> String {
        // SAFETY: `comm` is live, and NCCL returns a NUL-terminated string it owns, or null.
        let msg = unsafe { sys::ncclGetLastError(raw.comm) };
        if msg.is_null() {
            return String::new();
        }
        // SAFETY: non-null, NUL-terminated, valid until the next NCCL call on this thread; copied out at once.
        let text = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
        if text.is_empty() {
            String::new()
        } else {
            format!(": {text}")
        }
    }

    /// Poll the asynchronous state until the last call has finished, bounded; on a timeout the communicator is aborted.
    fn settle(&self, what: &str, waiting_on: &'static str) -> Result<(), GpuError> {
        let mut raw = self.lock();
        self.settle_locked(&mut raw, what, waiting_on)
    }

    fn settle_locked(
        &self,
        raw: &mut Raw,
        what: &str,
        waiting_on: &'static str,
    ) -> Result<(), GpuError> {
        let timeout = raw.timeout;
        if poll_until(timeout, || {
            self.check_async_locked(raw, what).map(|busy| !busy)
        })? {
            return Ok(());
        }
        Self::abort_locked(raw, self.rank, &self.ctx);
        Err(GpuError::Timeout { what: waiting_on })
    }

    fn post(&self, ops: &[WireOp<'_>]) -> Result<(), GpuError> {
        let mut raw = self.lock();
        Self::usable(&raw, "ncclGroupStart")?;
        if let Some(op) = ops
            .iter()
            .find(|op| op.peer >= self.size || op.stream.context() != &self.ctx)
        {
            return Err(GpuError::Nccl {
                code: sys::ncclResult_t::ncclInvalidArgument as i32,
                what: format!(
                    "a wire op to peer {} of {} on device {}, for a communicator on device {}",
                    op.peer,
                    self.size,
                    op.stream.context().ordinal(),
                    self.ctx.ordinal()
                ),
            });
        }
        self.ctx.bind_to_thread()?;
        nccl::group_start().map_err(|e| nccl_error(e, "ncclGroupStart"))?;
        let mut first: Option<GpuError> = None;
        for op in ops {
            let stream = op.stream.cu_stream() as sys::cudaStream_t;
            let peer = op.peer as i32;
            let dtype = sys::ncclDataType_t::ncclUint8;
            // SAFETY: `op.ptr` addresses `op.bytes` bytes of device memory on this communicator's device, borrowed by the `WireGroup` until the group is enqueued, and `comm` is live.
            let posted = unsafe {
                match op.kind {
                    WireOpKind::Send => {
                        nccl::send(op.ptr as _, op.bytes, dtype, peer, raw.comm, stream)
                    }
                    WireOpKind::Recv => {
                        nccl::recv(op.ptr as _, op.bytes, dtype, peer, raw.comm, stream)
                    }
                }
            };
            if let Err(e) = posted {
                first = Some(nccl_error(e, format!("ncclSend/ncclRecv to peer {peer}")));
                break;
            }
        }
        // A group must be closed even after a failed post, or this thread's next NCCL call joins it.
        let ended = nccl::group_end();
        if let Some(e) = first {
            Self::abort_locked(&mut raw, self.rank, &self.ctx);
            return Err(e);
        }
        if let Err(e) = ended {
            let detail = self.last_error(&raw);
            Self::abort_locked(&mut raw, self.rank, &self.ctx);
            return Err(nccl_error(e, format!("ncclGroupEnd{detail}")));
        }
        // A non-blocking communicator may still be enqueueing the group's kernels; until it is done, later work on the streams would run ahead of them.
        self.settle_locked(&mut raw, "ncclGroupEnd", "an NCCL group to be enqueued")
    }

    fn wait(&self, stream: &CudaStream) -> Result<(), GpuError> {
        Self::usable(&self.lock(), "a wait")?;
        let done = self.ctx.new_event(None)?;
        done.record(stream)?;
        let mut raw = self.lock();
        let timeout = raw.timeout;
        #[cfg(any(test, feature = "test-utils"))]
        let forced = std::mem::take(&mut raw.force_timeout);
        #[cfg(not(any(test, feature = "test-utils")))]
        let forced = false;
        let ready = poll_until(timeout, || {
            self.check_async_locked(&mut raw, "an NCCL group")?;
            Ok(done.try_is_complete()? && !forced)
        })?;
        if ready {
            return Ok(());
        }
        Self::abort_locked(&mut raw, self.rank, &self.ctx);
        Err(GpuError::Timeout {
            what: "an NCCL group to complete",
        })
    }
}

impl Drop for NcclComm {
    /// Shut down, then, as the last communicator of the process (or in any test), join the abort threads within the wait bound.
    fn drop(&mut self) {
        let bound = self
            .raw
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .timeout;
        self.shut_down();
        let last = LIVE.fetch_sub(1, Ordering::Relaxed) == 1;
        if last || cfg!(test) {
            reap_aborts(bound);
        }
    }
}

impl NcclComm {
    fn shut_down(&mut self) {
        let rank = self.rank;
        let bound = self.ctx.bind_to_thread().is_ok();
        let raw = self.raw.get_mut().unwrap_or_else(PoisonError::into_inner);
        if raw.aborted {
            return;
        }
        if std::thread::panicking() || !bound {
            Self::abort_locked(raw, rank, &self.ctx);
            return;
        }
        let comm = raw.comm;
        // SAFETY: `comm` is live; after a failed finalize it is aborted, never reused.
        let settled = unsafe { nccl::comm_finalize(comm) }.is_ok()
            && poll_until(raw.timeout, || {
                let mut state = sys::ncclResult_t::ncclSuccess;
                // SAFETY: `comm` is live until destroyed or aborted below.
                let polled = unsafe { sys::ncclCommGetAsyncError(comm, &mut state) }
                    .result()
                    .and_then(|_| state.result());
                match polled {
                    Ok(NcclStatus::InProgress) => Ok(false),
                    Ok(_) => Ok(true),
                    Err(e) => Err(nccl_error(e, "ncclCommFinalize")),
                }
            })
            .unwrap_or(false);
        if !settled {
            log::warn!("gpu: NCCL finalize failed or timed out on rank {rank}; aborting");
            Self::abort_locked(raw, rank, &self.ctx);
            return;
        }
        raw.aborted = true;
        raw.comm = std::ptr::null_mut();
        // SAFETY: finalized, and never touched again.
        if let Err(e) = unsafe { nccl::comm_destroy(comm) } {
            log::warn!("gpu: ncclCommDestroy on rank {rank} returned {:?}", e.0);
        }
    }
}

/// Communicators alive in this process.
static LIVE: AtomicUsize = AtomicUsize::new(0);

/// Abort threads not yet joined.
static ABORTS: Mutex<Vec<std::thread::JoinHandle<()>>> = Mutex::new(Vec::new());

/// Join every finished abort thread, waiting up to `bound` for the rest; one still running past it stays listed for the next reap.
fn reap_aborts(bound: Duration) {
    let start = Instant::now();
    let mut pending = std::mem::take(&mut *ABORTS.lock().unwrap_or_else(PoisonError::into_inner));
    loop {
        let (done, rest): (Vec<_>, Vec<_>) = pending.into_iter().partition(|h| h.is_finished());
        for h in done {
            if h.join().is_err() {
                log::warn!("gpu: an NCCL abort thread panicked");
            }
        }
        pending = rest;
        if pending.is_empty() || start.elapsed() >= bound {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    if !pending.is_empty() {
        log::warn!(
            "gpu: {} NCCL abort(s) still running after {bound:?}",
            pending.len()
        );
        ABORTS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend(pending);
    }
}

/// A communicator handle on its way to the thread that aborts it.
#[derive(Clone, Copy)]
struct CommPtr(sys::ncclComm_t);

// SAFETY: the handle moves to the one thread that aborts it, after `Raw::aborted` has taken every other path off it.
unsafe impl Send for CommPtr {}

fn abort_now(comm: CommPtr, rank: u32, ctx: &Arc<CudaContext>) {
    let start = Instant::now();
    if let Err(e) = ctx.bind_to_thread() {
        log::warn!(
            "gpu: binding device {} for the NCCL abort on rank {rank}: {e:?}",
            ctx.ordinal()
        );
    }
    // SAFETY: `comm` is a live communicator no other thread touches again.
    match unsafe { nccl::comm_abort(comm.0) } {
        Ok(_) => log::info!(
            "gpu: NCCL communicator on rank {rank} aborted in {:?}",
            start.elapsed()
        ),
        Err(e) => log::warn!("gpu: ncclCommAbort on rank {rank} returned {:?}", e.0),
    }
}

/// Whether this process can start an NCCL communicator: the library loads and is at least [`MIN_NCCL_VERSION`].
fn local_readiness() -> Result<(), GpuError> {
    if !super::device::nccl_available() {
        return Err(GpuError::LibraryMissing("libnccl"));
    }
    let version = nccl::get_nccl_version().map_err(|e| nccl_error(e, "ncclGetVersion"))?;
    log::info!("gpu: libnccl version code {version}");
    if version < MIN_NCCL_VERSION {
        return Err(GpuError::Unsupported("libnccl older than 2.22"));
    }
    Ok(())
}

/// `NCCL_CONFIG_INITIALIZER` for the bound `ncclConfig_v21700`, versioned as the 2.22 bindings.
fn default_config() -> sys::ncclConfig_t {
    const UNDEF: i32 = i32::MIN;
    sys::ncclConfig_t {
        size: std::mem::size_of::<sys::ncclConfig_t>(),
        magic: 0xcafe_beef,
        version: MIN_NCCL_VERSION as u32,
        blocking: UNDEF,
        cgaClusterSize: UNDEF,
        minCTAs: UNDEF,
        maxCTAs: UNDEF,
        netName: std::ptr::null(),
        splitShare: UNDEF,
    }
}

fn pack_id(id: &sys::ncclUniqueId, words: &mut [u64]) {
    for (w, chunk) in words.iter_mut().zip(id.internal.chunks_exact(8)) {
        *w = u64::from_ne_bytes(std::array::from_fn(|i| chunk[i] as u8));
    }
}

fn unpack_id(words: &[u64]) -> sys::ncclUniqueId {
    let mut id = sys::ncclUniqueId { internal: [0; 128] };
    for (chunk, w) in id.internal.chunks_exact_mut(8).zip(words) {
        for (c, b) in chunk.iter_mut().zip(w.to_ne_bytes()) {
            *c = b as std::ffi::c_char;
        }
    }
    id
}

/// Which way a [`WireOp`] moves bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WireOpKind {
    /// This rank's bytes to `peer`.
    Send,
    /// `peer`'s bytes into this rank's buffer.
    Recv,
}

/// One point-to-point transfer of a [`WireGroup`]: `bytes` bytes at device address `ptr`, to or from `peer`, ordered on `stream`.
/// Only a [`WireGroup`] builds one, which is what lets [`DeviceWire::post`] trust `ptr` without being `unsafe`.
#[derive(Clone, Copy)]
pub(crate) struct WireOp<'a> {
    kind: WireOpKind,
    peer: u32,
    ptr: u64,
    bytes: usize,
    stream: &'a CudaStream,
}

/// Read by an implementation of [`DeviceWire`] other than NCCL's, which reads the fields.
#[cfg_attr(not(any(test, feature = "test-utils")), allow(dead_code))]
impl<'a> WireOp<'a> {
    pub(crate) fn kind(&self) -> WireOpKind {
        self.kind
    }
    pub(crate) fn peer(&self) -> u32 {
        self.peer
    }
    /// The device address, valid for [`bytes`](Self::bytes) bytes while the op's [`WireGroup`] lives.
    pub(crate) fn ptr(&self) -> u64 {
        self.ptr
    }
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
    pub(crate) fn stream(&self) -> &'a CudaStream {
        self.stream
    }
}

/// The ops of one wire group, holding every buffer borrowed until [`post`](Self::post) has enqueued them, which is what lets posting be safe: cudarc's per-buffer events are recorded only after the ops are on their streams.
#[derive(Default)]
pub(crate) struct WireGroup<'a> {
    ops: Vec<WireOp<'a>>,
    guards: Vec<SyncOnDrop<'a>>,
}

impl<'a> WireGroup<'a> {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Send `src`'s bytes to `peer`, ordered on `stream`.
    pub(crate) fn send<T>(&mut self, src: CudaView<'a, T>, peer: u32, stream: &'a CudaStream) {
        let bytes = src.len() * std::mem::size_of::<T>();
        let (ptr, guard) = src.view_ptr(stream);
        self.push(WireOpKind::Send, peer, ptr, bytes, stream, guard);
    }

    /// Receive `dst.len()` elements' bytes from `peer` into `dst`, ordered on `stream`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn recv<T>(&mut self, dst: CudaViewMut<'a, T>, peer: u32, stream: &'a CudaStream) {
        let bytes = dst.len() * std::mem::size_of::<T>();
        let (ptr, guard) = dst.view_ptr(stream);
        self.push(WireOpKind::Recv, peer, ptr, bytes, stream, guard);
    }

    /// Receive consecutive ranges of `dst` in order, `parts[i] = (len, peer)` being `len` elements from `peer`, on `stream`; one view carved here, since a borrowed split of a cudarc view cannot outlive its parent.
    /// Panics if the parts are longer than `dst`.
    pub(crate) fn recv_parts<T>(
        &mut self,
        dst: CudaViewMut<'a, T>,
        parts: &[(usize, u32)],
        stream: &'a CudaStream,
    ) {
        let total: usize = parts.iter().map(|&(len, _)| len).sum();
        assert!(
            total <= dst.len(),
            "recv_parts: {total} elements into a view of {}",
            dst.len()
        );
        let elem = std::mem::size_of::<T>();
        let (base, guard) = dst.view_ptr(stream);
        let mut at = 0usize;
        for &(len, peer) in parts {
            self.ops.push(WireOp {
                kind: WireOpKind::Recv,
                peer,
                ptr: base + (at * elem) as u64,
                bytes: len * elem,
                stream,
            });
            at += len;
        }
        self.guards.push(guard);
    }

    fn push(
        &mut self,
        kind: WireOpKind,
        peer: u32,
        ptr: u64,
        bytes: usize,
        stream: &'a CudaStream,
        guard: SyncOnDrop<'a>,
    ) {
        self.ops.push(WireOp {
            kind,
            peer,
            ptr,
            bytes,
            stream,
        });
        self.guards.push(guard);
    }

    /// The ops in posting order.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn ops(&self) -> &[WireOp<'a>] {
        &self.ops
    }

    /// Post every op through `wire` as one group, then release the buffers' borrows.
    pub(crate) fn post(self, wire: &dyn DeviceWire) -> Result<(), GpuError> {
        self.post_with(|ops| wire.post(ops))
    }

    fn post_with(
        self,
        post: impl FnOnce(&[WireOp<'a>]) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        let posted = post(&self.ops);
        drop(self.guards);
        posted
    }
}

/// Point-to-point transfers of device memory within a group of ranks, one group of transfers at a time.
///
/// The matching contract an implementation must honour: a rank's sends to `q` match `q`'s receives from that rank one to one, in posting order, with equal byte counts, and a rank may be its own peer.
/// Every wait is bounded; a timed-out or failed wire is dead, and every later call on it returns an error.
pub(crate) trait DeviceWire: Send + Sync {
    /// This rank's index in the group.
    #[cfg_attr(not(any(test, feature = "test-utils")), allow(dead_code))]
    fn rank(&self) -> u32;
    /// Ranks in the group.
    #[cfg_attr(not(any(test, feature = "test-utils")), allow(dead_code))]
    fn size(&self) -> u32;
    /// Post `ops` as one group: when it returns `Ok`, every op is enqueued on its stream, so later work on that stream runs after it.
    /// Every rank a posted op names must post its matching group, or the transfers never complete and [`wait`](Self::wait) times out.
    fn post(&self, ops: &[WireOp<'_>]) -> Result<(), GpuError>;
    /// Block, bounded by the wire's timeout, until all work enqueued on `stream` so far has completed.
    fn wait(&self, stream: &CudaStream) -> Result<(), GpuError>;
    /// Give the wire up for good, without waiting on anything; every later call fails.
    fn abort(&self) {}
    /// Whether the wire can still carry a group: `false` once it failed, timed out or was aborted.
    fn is_healthy(&self) -> bool {
        true
    }
}

/// The [`DeviceWire`] over a real NCCL communicator.
#[derive(Clone)]
pub(crate) struct NcclWire {
    comm: Arc<NcclComm>,
}

impl NcclWire {
    pub(crate) fn new(comm: Arc<NcclComm>) -> Self {
        Self { comm }
    }

    /// The communicator, for the health checks and abort of the failure paths.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn comm(&self) -> &Arc<NcclComm> {
        &self.comm
    }
}

impl DeviceWire for NcclWire {
    fn rank(&self) -> u32 {
        self.comm.rank
    }
    fn size(&self) -> u32 {
        self.comm.size
    }
    fn post(&self, ops: &[WireOp<'_>]) -> Result<(), GpuError> {
        self.comm.post(ops)
    }
    fn wait(&self, stream: &CudaStream) -> Result<(), GpuError> {
        self.comm.wait(stream)
    }
    fn abort(&self) {
        self.comm.abort();
    }
    fn is_healthy(&self) -> bool {
        self.comm.is_healthy()
    }
}

/// One exchange block without its columns: what a receiver sizes its receives and the fused layer's segments from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Skeleton {
    pub(crate) header: BlockHeader,
    pub(crate) offsets: Vec<u32>,
}

/// One partner's block skeletons in ascending remote-delta index, the host half of an NCCL exchange; not a column-less `PartnerPayload`, whose `finish_recv` holds a header to the columns that arrived with it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockSkeletons<const W: usize> {
    pub(crate) blocks: Vec<Skeleton>,
}

impl<const W: usize> BlockSkeletons<W> {
    /// The headers and offsets of `blocks`, reusing this payload's allocations.
    pub(crate) fn fill_from(&mut self, blocks: &[DeviceBlock<W>]) {
        self.blocks.truncate(blocks.len());
        self.blocks.resize_with(blocks.len(), Skeleton::default);
        for (s, b) in self.blocks.iter_mut().zip(blocks) {
            s.header = b.header;
            s.offsets.clear();
            s.offsets.extend_from_slice(&b.offsets);
        }
    }

    /// Append an empty block for remote-delta `entry` over `b` positions.
    pub(crate) fn push_empty(&mut self, entry: u32, b: usize) {
        self.blocks.push(Skeleton {
            header: BlockHeader {
                num_buckets: b as u32,
                rows: 0,
                w: W as u32,
                entry,
            },
            offsets: vec![0; b + 1],
        });
    }
}

impl<const W: usize> Payload for BlockSkeletons<W> {
    fn byte_parts(&self) -> Vec<&[u8]> {
        let mut parts = Vec::with_capacity(2 * self.blocks.len());
        for s in &self.blocks {
            parts.push(bytemuck::bytes_of(&s.header));
            parts.push(bytemuck::cast_slice(&s.offsets));
        }
        parts
    }

    fn recv_into(&mut self, lens: &[usize]) -> Vec<&mut [u8]> {
        assert_eq!(
            lens.len() % 2,
            0,
            "block skeletons: {} parts is not a whole number of two-part blocks",
            lens.len()
        );
        let n = lens.len() / 2;
        self.blocks.truncate(n);
        self.blocks.resize_with(n, Skeleton::default);
        let mut parts = Vec::with_capacity(lens.len());
        for (s, lens) in self.blocks.iter_mut().zip(lens.chunks_exact(2)) {
            assert_eq!(
                lens[0],
                std::mem::size_of::<BlockHeader>(),
                "block skeletons: a header of {} bytes",
                lens[0]
            );
            assert_eq!(
                lens[1] % 4,
                0,
                "block skeletons: {} offset bytes is not a whole number of u32",
                lens[1]
            );
            s.offsets.clear();
            s.offsets.resize(lens[1] / 4, 0);
            let Skeleton { header, offsets } = s;
            parts.push(bytemuck::bytes_of_mut(header));
            parts.push(bytemuck::cast_slice_mut(&mut offsets[..]));
        }
        parts
    }

    /// The receive sizes itself from these offsets, so each must be a CSR index that ends at the header's row count.
    fn finish_recv(&mut self) {
        for s in &self.blocks {
            let h = &s.header;
            assert_eq!(
                h.w as usize, W,
                "block skeletons: a block encoded at W={} decoded at W={W}",
                h.w
            );
            assert_eq!(
                s.offsets.len(),
                h.num_buckets as usize + 1,
                "block skeletons: a block of {} buckets arrived with {} offsets",
                h.num_buckets,
                s.offsets.len()
            );
            assert!(
                s.offsets[0] == 0 && s.offsets.windows(2).all(|w| w[0] <= w[1]),
                "block skeletons: offsets that are not a CSR index"
            );
            assert_eq!(
                s.offsets[h.num_buckets as usize], h.rows,
                "block skeletons: offsets end at {}, the header names {} rows",
                s.offsets[h.num_buckets as usize], h.rows
            );
        }
    }
}

/// A column of an exchange block as the wire moves it; the fingerprint column never travels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WireColumn {
    X,
    Z,
    Coeff,
}

impl WireColumn {
    pub(crate) const ALL: [WireColumn; 3] = [WireColumn::X, WireColumn::Z, WireColumn::Coeff];

    /// Device elements per row at width `W`: `u64` words for a key column, `f64` halves for the coefficient.
    pub(crate) fn elems_per_row<const W: usize>(self) -> usize {
        match self {
            WireColumn::X | WireColumn::Z => W,
            WireColumn::Coeff => 2,
        }
    }
}

/// One transfer of an NCCL exchange: rows `rows.0..rows.1` of `column` of the block remote delta `k` carries, in chunk `chunk`, to or from `peer`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScheduledOp {
    pub(crate) kind: WireOpKind,
    pub(crate) peer: u32,
    pub(crate) chunk: usize,
    pub(crate) column: WireColumn,
    pub(crate) k: usize,
    pub(crate) rows: (usize, usize),
}

/// The transfers of one NCCL exchange: `partners[k]`, `own[k]` and `recv[k]` are remote delta `k`'s partner and the offsets of the blocks sent and received for it, `k` naming one block pair on both ends.
/// Per peer, sends and receives both run chunk, column, delta, the order a wire matches in; receives are column-major across peers so each column's tile its `recv_*` column, and an empty piece is posted by neither end.
pub(crate) fn nccl_schedule(
    partners: &[u32],
    own: &[&[u32]],
    recv: &[&[u32]],
    map: &ChunkMap,
) -> Vec<ScheduledOp> {
    assert_eq!(partners.len(), own.len());
    assert_eq!(partners.len(), recv.len());
    let own_rows: Vec<Vec<usize>> = own.iter().map(|o| chunk_rows_of(o, map)).collect();
    let recv_rows: Vec<Vec<usize>> = recv.iter().map(|o| chunk_rows_of(o, map)).collect();
    let mut peers: Vec<u32> = partners.to_vec();
    peers.sort_unstable();
    peers.dedup();
    let mut ops = Vec::new();
    for chunk in 0..map.chunks() {
        for &q in &peers {
            for column in WireColumn::ALL {
                for (k, _) in partners.iter().enumerate().filter(|&(_, &p)| p == q) {
                    let rows = (own_rows[k][chunk], own_rows[k][chunk + 1]);
                    if rows.1 > rows.0 {
                        ops.push(ScheduledOp {
                            kind: WireOpKind::Send,
                            peer: q,
                            chunk,
                            column,
                            k,
                            rows,
                        });
                    }
                }
            }
        }
    }
    for chunk in 0..map.chunks() {
        for column in WireColumn::ALL {
            for (k, &q) in partners.iter().enumerate() {
                let rows = (recv_rows[k][chunk], recv_rows[k][chunk + 1]);
                if rows.1 > rows.0 {
                    ops.push(ScheduledOp {
                        kind: WireOpKind::Recv,
                        peer: q,
                        chunk,
                        column,
                        k,
                        rows,
                    });
                }
            }
        }
    }
    ops
}

/// `PAULISTRINGS_GPU_EXCHANGE` as a device group over a byte transport reads it: `host`, `nccl`, or anything else (`device`, unset) for NCCL where the whole group can run it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExchangeKnob {
    Host,
    Auto,
    Nccl,
}

pub(crate) fn parse_exchange_knob(raw: Option<&str>) -> ExchangeKnob {
    match raw.map(str::trim) {
        Some("host") => ExchangeKnob::Host,
        Some("nccl") => ExchangeKnob::Nccl,
        _ => ExchangeKnob::Auto,
    }
}

/// The group's exchange decision: whether to start NCCL, and whether any rank's knob makes a failure to start it an error rather than a host fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExchangePlan {
    pub(crate) nccl: bool,
    pub(crate) strict: bool,
}

/// The group's [`ExchangePlan`], the same on every rank. **Collective**: one `allreduce_sum_u64` of `3 + 2 × size` words.
/// NCCL starts iff the group has more than one rank, every rank wants it and `can`, and no two `device` UUIDs coincide (NCCL refuses a shared device); a strict group that cannot start it errs on every rank.
pub(crate) fn agree_exchange(
    coll: &dyn Collectives,
    knob: ExchangeKnob,
    can: bool,
    device: [u64; 2],
) -> Result<ExchangePlan, GpuError> {
    let (rank, size) = (coll.rank() as usize, coll.size() as usize);
    let group = size > 1;
    let wants = group && knob != ExchangeKnob::Host;
    let mut buf = vec![0u64; 3 + 2 * size];
    buf[0] = u64::from(wants && can);
    buf[1] = u64::from(wants);
    buf[2] = u64::from(group && knob == ExchangeKnob::Nccl);
    buf[3 + 2 * rank..5 + 2 * rank].copy_from_slice(&device);
    coll.allreduce_sum_u64(&mut buf);
    let ids: Vec<[u64; 2]> = buf[3..].chunks_exact(2).map(|w| [w[0], w[1]]).collect();
    let distinct = ids
        .iter()
        .enumerate()
        .all(|(i, a)| ids[i + 1..].iter().all(|b| b != a));
    let nccl = group && buf[0] == size as u64 && distinct;
    let strict = buf[2] > 0;
    if strict && !nccl {
        return Err(GpuError::Unsupported(
            "PAULISTRINGS_GPU_EXCHANGE=nccl, but a rank cannot start NCCL, a rank asks for the host exchange, or two ranks share a device",
        ));
    }
    if wants && !nccl {
        log::info!(
            "gpu: rank {rank} of {size} exchanges through the host ({} of {size} ranks can start NCCL, devices distinct: {distinct})",
            buf[0]
        );
    }
    Ok(ExchangePlan { nccl, strict })
}

/// A device's UUID as the two words [`agree_exchange`] compares.
pub(crate) fn device_uuid(ctx: &CudaContext) -> Result<[u64; 2], GpuError> {
    let id = ctx.uuid()?;
    let b: [u8; 16] = std::array::from_fn(|i| id.bytes[i] as u8);
    Ok([
        u64::from_ne_bytes(b[..8].try_into().expect("eight bytes")),
        u64::from_ne_bytes(b[8..].try_into().expect("eight bytes")),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::partitioned::InProcessTransport;

    #[test]
    fn the_timeout_knob_parses_positive_seconds() {
        assert_eq!(parse_timeout(None), DEFAULT_NCCL_TIMEOUT);
        assert_eq!(parse_timeout(Some("12")), Duration::from_secs(12));
        assert_eq!(parse_timeout(Some(" 0.5 ")), Duration::from_millis(500));
        for bad in ["0", "-3", "nan", "inf", "soon", ""] {
            assert_eq!(parse_timeout(Some(bad)), DEFAULT_NCCL_TIMEOUT, "{bad:?}");
        }
    }

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

    fn skeletons_of<const W: usize>(seed: u64, blocks: usize, b: usize) -> BlockSkeletons<W> {
        let mut out = BlockSkeletons::<W>::default();
        for j in 0..blocks {
            let mut offsets = vec![0u32];
            for p in 0..b as u64 {
                let n = (seed ^ (j as u64 * 31) ^ (p * 7)).wrapping_mul(0x9E37_79B9) >> 60;
                offsets.push(offsets.last().unwrap() + n as u32);
            }
            let rows = offsets[b];
            out.blocks.push(Skeleton {
                header: BlockHeader {
                    num_buckets: b as u32,
                    rows,
                    w: W as u32,
                    entry: 3 * j as u32 + 1,
                },
                offsets,
            });
        }
        out
    }

    /// The byte form a byte transport moves, decoded into a pooled payload of another shape.
    fn byte_round_trip<const W: usize>(
        sent: &BlockSkeletons<W>,
        pooled: BlockSkeletons<W>,
    ) -> BlockSkeletons<W> {
        let parts: Vec<Vec<u8>> = sent.byte_parts().iter().map(|p| p.to_vec()).collect();
        let lens: Vec<usize> = parts.iter().map(Vec::len).collect();
        let mut got = pooled;
        for (dst, src) in got.recv_into(&lens).into_iter().zip(&parts) {
            dst.copy_from_slice(src);
        }
        got.finish_recv();
        got
    }

    #[test]
    fn skeletons_round_trip_through_their_byte_form() {
        let sent = skeletons_of::<2>(0xA1, 3, 16);
        assert_eq!(sent.byte_parts().len(), 6);
        assert_eq!(byte_round_trip(&sent, BlockSkeletons::default()), sent);
        assert_eq!(byte_round_trip(&sent, skeletons_of::<2>(0xB2, 5, 64)), sent);
        let empty = BlockSkeletons::<1>::default();
        assert_eq!(byte_round_trip(&empty, skeletons_of::<1>(1, 2, 4)), empty);
    }

    #[test]
    fn skeletons_travel_over_the_in_process_transport() {
        use crate::engine::partitioned::transport::Transport;
        let got: Vec<Vec<Option<BlockSkeletons<1>>>> = std::thread::scope(|s| {
            let hs: Vec<_> = InProcessTransport::group(4)
                .into_iter()
                .map(|t| {
                    s.spawn(move || {
                        let me = t.rank();
                        let send = (0..4)
                            .map(|q| {
                                (q != me).then(|| skeletons_of::<1>(u64::from(me * 8 + q), 2, 8))
                            })
                            .collect();
                        t.exchange(send, &mut Vec::new())
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (me, recv) in got.iter().enumerate() {
            for (q, r) in recv.iter().enumerate() {
                if q == me {
                    assert!(r.is_none());
                } else {
                    let want = skeletons_of::<1>((q * 8 + me) as u64, 2, 8);
                    assert_eq!(r.as_ref(), Some(&want), "{q} -> {me}");
                }
            }
        }
    }

    #[test]
    fn a_skeleton_whose_offsets_miss_its_header_is_refused() {
        let mut bad = skeletons_of::<1>(0xC3, 1, 8);
        bad.blocks[0].header.rows += 1;
        let r = std::panic::catch_unwind(|| byte_round_trip(&bad, BlockSkeletons::default()));
        assert!(r.is_err());
        let mut wide = skeletons_of::<1>(0xC4, 1, 8);
        wide.blocks[0].header.w = 2;
        assert!(
            std::panic::catch_unwind(|| byte_round_trip(&wide, BlockSkeletons::default())).is_err()
        );
        let mut descending = skeletons_of::<1>(0xC5, 1, 2);
        descending.blocks[0].offsets = vec![0, 5, 3];
        descending.blocks[0].header.rows = 3;
        assert!(std::panic::catch_unwind(|| byte_round_trip(
            &descending,
            BlockSkeletons::default()
        ))
        .is_err());
    }

    /// The pieces rank `me` sends to `to` as `(chunk, column, k, rows)`, and those `to` receives from `me`.
    fn sends_to(ops: &[ScheduledOp], to: u32) -> Vec<(usize, WireColumn, usize, (usize, usize))> {
        ops.iter()
            .filter(|op| op.kind == WireOpKind::Send && op.peer == to)
            .map(|op| (op.chunk, op.column, op.k, op.rows))
            .collect()
    }

    fn recvs_from(
        ops: &[ScheduledOp],
        from: u32,
    ) -> Vec<(usize, WireColumn, usize, (usize, usize))> {
        ops.iter()
            .filter(|op| op.kind == WireOpKind::Recv && op.peer == from)
            .map(|op| (op.chunk, op.column, op.k, op.rows))
            .collect()
    }

    /// Per rank of a group of `size`, the schedule for remote deltas of partition deltas `pds`, where `counts[sender][k][p]` is the rows the sender's block for delta `k` puts at position `p`.
    fn group_schedules(
        size: u32,
        pds: &[u32],
        counts: &[Vec<Vec<u32>>],
        map: &ChunkMap,
    ) -> Vec<Vec<ScheduledOp>> {
        let csr = |c: &Vec<u32>| -> Vec<u32> {
            let mut o = vec![0u32];
            for &n in c {
                o.push(o.last().unwrap() + n);
            }
            o
        };
        let offsets: Vec<Vec<Vec<u32>>> = counts
            .iter()
            .map(|per_k| per_k.iter().map(csr).collect())
            .collect();
        (0..size)
            .map(|r| {
                let partners: Vec<u32> = pds.iter().map(|&pd| r ^ pd).collect();
                let own: Vec<&[u32]> = offsets[r as usize].iter().map(Vec::as_slice).collect();
                let recv: Vec<&[u32]> = partners
                    .iter()
                    .enumerate()
                    .map(|(k, &q)| offsets[q as usize][k].as_slice())
                    .collect();
                nccl_schedule(&partners, &own, &recv, map)
            })
            .collect()
    }

    proptest::proptest! {
        /// Rank `a`'s sends to `b` are `b`'s receives from `a`, in order and in rows, chunk-major; within a chunk each column's receives tile the chunk's buffer in plan order.
        #[test]
        fn the_schedule_is_symmetric(
            size_bits in 1u32..=3,
            pd_seeds in proptest::collection::vec(0u32..1000, 1..=10),
            bits in 0u8..=4,
            chunks in proptest::sample::select(vec![1usize, 3, 8]),
            empty_every in 1usize..5,
            seed in 0u64..u64::MAX,
        ) {
            let size = 1u32 << size_bits;
            let pds: Vec<u32> = pd_seeds.iter().map(|s| 1 + s % (size - 1)).collect();
            let b = 1usize << bits;
            let mut state = seed | 1;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let counts: Vec<Vec<Vec<u32>>> = (0..size)
                .map(|r| {
                    (0..pds.len())
                        .map(|k| {
                            (0..b)
                                .map(|_| {
                                    if (r as usize + k).is_multiple_of(empty_every) { 0 } else { (next() % 5) as u32 }
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();
            let mut map = ChunkMap::default();
            map.rebuild(&crate::engine::coset::Gf2Span::new(&[], bits), b, chunks);
            let ops = group_schedules(size, &pds, &counts, &map);
            for a in 0..size {
                for bb in 0..size {
                    let s = sends_to(&ops[a as usize], bb);
                    let r = recvs_from(&ops[bb as usize], a);
                    proptest::prop_assert_eq!(&s, &r, "{} -> {}", a, bb);
                    proptest::prop_assert!(s.iter().all(|x| x.3 .1 > x.3 .0), "an empty piece posted");
                }
                let pds = &pds;
                let rows: u64 = counts.iter().enumerate().flat_map(|(r, per_k)| {
                    per_k.iter().enumerate().filter(move |&(k, _)| r as u32 ^ pds[k] == a)
                        .map(|(_, c)| c.iter().map(|&n| u64::from(n)).sum::<u64>())
                }).sum();
                let got: u64 = ops[a as usize].iter().filter(|op| op.kind == WireOpKind::Recv && op.column == WireColumn::X)
                    .map(|op| (op.rows.1 - op.rows.0) as u64).sum();
                proptest::prop_assert_eq!(got, rows, "rank {} receives every row once", a);
                for chunk in 0..map.chunks() {
                    for column in WireColumn::ALL {
                        let ks: Vec<(usize, (usize, usize))> = ops[a as usize].iter()
                            .filter(|op| op.kind == WireOpKind::Recv && op.column == column && op.chunk == chunk)
                            .map(|op| (op.k, op.rows)).collect();
                        proptest::prop_assert!(ks.windows(2).all(|w| w[0].0 < w[1].0));
                        if chunk == 0 {
                            proptest::prop_assert!(ks.iter().all(|&(_, (lo, _))| lo == 0));
                        }
                    }
                }
                for kind in [WireOpKind::Send, WireOpKind::Recv] {
                    let chunk_of: Vec<usize> = ops[a as usize].iter().filter(|op| op.kind == kind).map(|op| op.chunk).collect();
                    proptest::prop_assert!(chunk_of.windows(2).all(|w| w[0] <= w[1]), "{:?}s are chunk-major", kind);
                }
            }
        }
    }

    /// Every combination of knob and readiness on two ranks, and a sample on four: one outcome on every rank, the documented one.
    #[test]
    fn the_exchange_mode_is_agreed_for_every_knob_and_readiness() {
        use ExchangeKnob::{Auto, Host, Nccl};
        let run = |ranks: &[(ExchangeKnob, bool, [u64; 2])]| -> Vec<Result<bool, String>> {
            let size = ranks.len() as u32;
            std::thread::scope(|s| {
                let hs: Vec<_> = InProcessTransport::group(size)
                    .into_iter()
                    .map(|t| {
                        let (knob, can, id) = ranks[t.rank() as usize];
                        s.spawn(move || {
                            agree_exchange(&t, knob, can, id)
                                .map(|p| p.nccl)
                                .map_err(|e| e.to_string())
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            })
        };
        let want = |ranks: &[(ExchangeKnob, bool, [u64; 2])]| -> Result<bool, ()> {
            let group = ranks.len() > 1;
            let ids: Vec<_> = ranks.iter().map(|r| r.2).collect();
            let distinct = (0..ids.len()).all(|i| (i + 1..ids.len()).all(|j| ids[i] != ids[j]));
            let nccl = group && distinct && ranks.iter().all(|&(k, c, _)| k != Host && c);
            if group && !nccl && ranks.iter().any(|r| r.0 == Nccl) {
                Err(())
            } else {
                Ok(nccl)
            }
        };
        let knobs = [Host, Auto, Nccl];
        let mut cases: Vec<Vec<(ExchangeKnob, bool, [u64; 2])>> = Vec::new();
        for &k in &knobs {
            for can in [false, true] {
                cases.push(vec![(k, can, [1, 0])]);
            }
        }
        for &k0 in &knobs {
            for c0 in [false, true] {
                for &k1 in &knobs {
                    for c1 in [false, true] {
                        cases.push(vec![(k0, c0, [1, 0]), (k1, c1, [2, 0])]);
                    }
                }
            }
        }
        for shared in [[7, 7], [7, 8]] {
            for &k in &knobs {
                cases.push(vec![(k, true, [7, 7]), (k, true, shared)]);
            }
        }
        for mask in 0..64u32 {
            let ranks: Vec<_> = (0..4)
                .map(|r| {
                    let k = knobs[((mask >> r) as usize + r as usize) % 3];
                    (k, (mask >> (r + 2)) & 1 == 0 || r == 0, [r as u64 + 1, 9])
                })
                .collect();
            cases.push(ranks);
        }
        for ranks in &cases {
            let got = run(ranks);
            let first = &got[0];
            assert!(
                got.iter().all(|g| g == first),
                "{ranks:?}: ranks disagree: {got:?}"
            );
            match (want(ranks), first) {
                (Ok(w), Ok(g)) => assert_eq!(*g, w, "{ranks:?}"),
                (Err(()), Err(msg)) => {
                    assert!(msg.contains("PAULISTRINGS_GPU_EXCHANGE=nccl"), "{msg}")
                }
                (w, g) => panic!("{ranks:?}: want {w:?}, got {g:?}"),
            }
        }
    }

    #[test]
    fn the_exchange_knob_parses() {
        assert_eq!(parse_exchange_knob(None), ExchangeKnob::Auto);
        assert_eq!(parse_exchange_knob(Some("device")), ExchangeKnob::Auto);
        assert_eq!(parse_exchange_knob(Some("host")), ExchangeKnob::Host);
        assert_eq!(parse_exchange_knob(Some(" nccl ")), ExchangeKnob::Nccl);
        assert_eq!(parse_exchange_knob(Some("garbage")), ExchangeKnob::Auto);
    }

    #[test]
    fn the_config_matches_the_initializer_layout() {
        let c = default_config();
        assert_eq!(c.size, 48);
        assert_eq!(c.magic, 0xcafe_beef);
        assert_eq!(c.blocking, i32::MIN);
        assert!(c.netName.is_null());
    }

    #[test]
    fn a_group_records_its_ops_in_posting_order_with_carved_ranges() {
        crate::require_cuda!();
        let ctx = super::super::device::context(0).expect("a visible device");
        let stream = ctx.new_stream().expect("a stream");
        let src = stream.alloc_zeros::<f64>(6).expect("alloc");
        let mut dst = stream.alloc_zeros::<u64>(10).expect("alloc");
        let mut group = WireGroup::new();
        group.send(src.slice(2..6), 3, &stream);
        group.recv_parts(dst.as_view_mut(), &[(4, 1), (0, 2), (5, 1)], &stream);
        let ops = group.ops();
        let shape: Vec<_> = ops
            .iter()
            .map(|op| (op.kind(), op.peer(), op.bytes()))
            .collect();
        assert_eq!(
            shape,
            [
                (WireOpKind::Send, 3, 32),
                (WireOpKind::Recv, 1, 32),
                (WireOpKind::Recv, 2, 0),
                (WireOpKind::Recv, 1, 40),
            ]
        );
        assert_eq!(ops[2].ptr() - ops[1].ptr(), 32);
        assert_eq!(ops[3].ptr() - ops[1].ptr(), 32);
        assert!(ops.iter().all(|op| std::ptr::eq(op.stream(), &*stream)));
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
        assert_eq!(comm.timeout(), nccl_timeout());
        comm.set_timeout(Duration::from_secs(60));
        let stream = ctx.new_stream().expect("a stream");
        Some((guard, comm, stream))
    }

    #[test]
    fn a_one_rank_communicator_warms_up_and_shuts_down_cleanly() {
        let Some((_nccl, comm, stream)) = one_rank() else {
            return;
        };
        assert_eq!((comm.rank(), comm.size()), (0, 1));
        comm.warm_up(&stream).expect("the warm-up completes");
        assert!(comm.is_healthy());
        assert!(!comm.check_async().expect("no async error"));
        drop(comm);
    }

    #[test]
    fn an_aborted_communicator_refuses_every_later_call() {
        let Some((_nccl, comm, stream)) = one_rank() else {
            return;
        };
        comm.abort();
        comm.abort();
        assert!(!comm.is_healthy());
        assert!(matches!(comm.check_async(), Err(GpuError::Nccl { .. })));
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
}
