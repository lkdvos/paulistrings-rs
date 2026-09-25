//! [`NcclComm`], the NCCL communicator of a device group over MPI, and the [`DeviceWire`] seam the device exchange posts its bulk columns through, with [`NcclWire`] its NCCL form.
//! See ARCHITECTURE.md §Partitioning.

use std::ffi::CStr;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaStream, CudaView, CudaViewMut, SyncOnDrop};
use cudarc::nccl::result::{self as nccl, NcclError, NcclStatus};
use cudarc::nccl::sys;

use super::error::GpuError;
use crate::engine::partitioned::transport::Collectives;

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

/// One rank's NCCL communicator over a device group, non-blocking, with every wait on it bounded by a timeout.
///
/// Built collectively over the group's [`Collectives`] and owned by the split for its lifetime (ARCHITECTURE.md §Partitioning).
/// A failed or timed-out operation aborts the communicator, and every later operation on it returns an error without touching NCCL.
/// Drop finalizes (bounded) and destroys a healthy communicator and aborts any other; it never panics.
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

    /// This rank's communicator over `coll`'s group on `ctx`'s device.
    ///
    /// **Collective**: exactly one `allreduce_sum_u64` on every rank whatever the outcome, carrying rank 0's unique id and every rank's readiness, so a rank without a usable `libnccl` fails the call everywhere before any rank starts NCCL's bootstrap.
    /// A rank whose communicator setup then fails or times out aborts it and returns the error alone; the caller agrees the outcome over the group before [`warm_up`](Self::warm_up).
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
    pub(crate) fn rank(&self) -> u32 {
        self.rank
    }

    /// Ranks in the communicator.
    pub(crate) fn size(&self) -> u32 {
        self.size
    }

    /// The bound on every wait on this communicator.
    pub(crate) fn timeout(&self) -> Duration {
        self.lock().timeout
    }

    /// Replace the wait bound, so a test can force a timeout without waiting out the default.
    #[cfg(any(test, feature = "test-utils"))]
    pub(crate) fn set_timeout(&self, timeout: Duration) {
        self.lock().timeout = timeout;
    }

    /// Make the next [`DeviceWire::wait`] treat its work as never completing, so it times out and aborts whatever the device does.
    #[cfg(any(test, feature = "test-utils"))]
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
        if let Err(e) = spawned {
            log::warn!("gpu: no thread for the NCCL abort on rank {rank} ({e}); aborting inline");
            abort_now(comm, rank, ctx);
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
    fn drop(&mut self) {
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

/// The ops of one wire group, holding every buffer borrowed until [`post`](Self::post) has enqueued them.
///
/// The borrow is what makes posting safe: a buffer cannot be freed or reused before its op is on its stream, and cudarc's per-buffer events are recorded only after that, so they cover the transfer.
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
    pub(crate) fn recv<T>(&mut self, dst: CudaViewMut<'a, T>, peer: u32, stream: &'a CudaStream) {
        let bytes = dst.len() * std::mem::size_of::<T>();
        let (ptr, guard) = dst.view_ptr(stream);
        self.push(WireOpKind::Recv, peer, ptr, bytes, stream, guard);
    }

    /// Receive consecutive ranges of `dst` in order, `parts[i] = (len, peer)` being `len` elements from `peer`, all ordered on `stream`.
    /// One view carved in the group, since a borrowed split of a cudarc view cannot outlive its parent.
    ///
    /// # Panics
    ///
    /// If the parts are longer than `dst`.
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
    fn rank(&self) -> u32;
    /// Ranks in the group.
    fn size(&self) -> u32;
    /// Post `ops` as one group: when it returns `Ok`, every op is enqueued on its stream, so later work on that stream runs after it.
    /// Every rank a posted op names must post its matching group, or the transfers never complete and [`wait`](Self::wait) times out.
    fn post(&self, ops: &[WireOp<'_>]) -> Result<(), GpuError>;
    /// Block, bounded by the wire's timeout, until all work enqueued on `stream` so far has completed.
    fn wait(&self, stream: &CudaStream) -> Result<(), GpuError>;
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

    /// A one-rank NCCL communicator on device 0 and a stream on it, or `None` where NCCL is absent.
    fn one_rank() -> Option<(NcclComm, Arc<CudaStream>)> {
        if !crate::engine::gpu::nccl_available() {
            return None;
        }
        let ctx = super::super::device::context(0).expect("a visible device");
        let t = InProcessTransport::group(1).pop().expect("one rank");
        let comm = NcclComm::init(&t, &ctx).expect("a one-rank communicator initializes");
        assert_eq!(comm.timeout(), nccl_timeout());
        comm.set_timeout(Duration::from_secs(60));
        let stream = ctx.new_stream().expect("a stream");
        Some((comm, stream))
    }

    #[test]
    fn a_one_rank_communicator_warms_up_and_shuts_down_cleanly() {
        let Some((comm, stream)) = one_rank() else {
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
        let Some((comm, stream)) = one_rank() else {
            return;
        };
        comm.abort();
        comm.abort();
        assert!(!comm.is_healthy());
        assert!(matches!(comm.check_async(), Err(GpuError::Nccl { .. })));
        assert!(matches!(comm.warm_up(&stream), Err(GpuError::Nccl { .. })));
        drop(comm);
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
        let Some((comm, stream)) = one_rank() else {
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
        let Some((comm, stream)) = one_rank() else {
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
        let Some((comm, stream)) = one_rank() else {
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
        let Some((comm, stream)) = one_rank() else {
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
        let Some((comm, stream)) = one_rank() else {
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
