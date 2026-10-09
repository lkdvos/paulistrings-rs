//! [`NcclComm`], the NCCL communicator of a device group over MPI, its [`DeviceWire`] form [`NcclWire`], and the group's start agreement.
//! See ARCHITECTURE.md §Partitioning.

use std::ffi::CStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaStream};
use cudarc::nccl::result::{self as nccl, NcclError, NcclStatus};
use cudarc::nccl::sys;

use super::error::GpuError;
use super::wire::{wire_timeout, DeviceWire, WireGroup, WireOp, WireOpKind};
use crate::collectives::Collectives;

/// The oldest runtime `libnccl` the `nccl-02022` bindings are sound against, as `ncclGetVersion` codes it.
const MIN_NCCL_VERSION: i32 = 22200;

/// `sizeof(ncclUniqueId)` in `u64` words.
const ID_WORDS: usize = 16;

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
    #[cfg(test)]
    force_timeout: bool,
}

// SAFETY: an NCCL communicator may be driven from any thread provided no two calls on it overlap, which the `Mutex` around every `Raw` enforces.
unsafe impl Send for Raw {}

/// One rank's non-blocking NCCL communicator, every wait bounded; a failed or timed-out call aborts it and later calls fail without touching NCCL.
pub(crate) struct NcclComm {
    raw: Mutex<Raw>,
    context: Arc<CudaContext>,
    rank: u32,
    size: u32,
}

/// The warm-up's per-peer byte count.
const WARM_UP_BYTES: usize = 8;

impl NcclComm {
    /// This rank's non-blocking communicator over `collectives`'s group on `context`'s device, every wait bounded by [`wire_timeout`]. **Collective**: one `allreduce_sum_u64` whatever the outcome, carrying rank 0's id and every rank's readiness.
    /// A setup that fails after it aborts and fails on this rank alone, so the caller agrees the outcome before [`warm_up`](Self::warm_up).
    pub(crate) fn init(
        collectives: &dyn Collectives,
        context: &Arc<CudaContext>,
    ) -> Result<Self, GpuError> {
        let timeout = wire_timeout();
        let (rank, size) = (collectives.rank(), collectives.size());
        let local = local_readiness();
        let id = match (&local, rank) {
            (Ok(()), 0) => nccl::get_uniqueid().map_err(|e| nccl_error(e, "ncclGetUniqueId")),
            _ => Ok(sys::ncclUniqueId { internal: [0; 128] }),
        };
        let mut buffer = [0u64; ID_WORDS + 1];
        match &id {
            Ok(id) if local.is_ok() => pack_id(id, &mut buffer[..ID_WORDS]),
            _ => buffer[ID_WORDS] = 1,
        }
        collectives.allreduce_sum_u64(&mut buffer);
        local?;
        id?;
        if buffer[ID_WORDS] != 0 {
            return Err(GpuError::Unsupported(
                "a peer rank cannot start NCCL (libnccl missing, older than 2.22, or no unique id)",
            ));
        }
        let id = unpack_id(&buffer[..ID_WORDS]);
        context.bind_to_thread()?;
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
                #[cfg(test)]
                force_timeout: false,
            }),
            context: context.clone(),
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

    /// Whether the communicator has not been aborted.
    pub(crate) fn is_healthy(&self) -> bool {
        !self.lock().aborted
    }

    /// Pay NCCL's lazy connection setup now: one small send/recv with every other rank (with itself in a one-rank world), completed on `stream`.
    /// **Collective over the communicator**, after the group has agreed that every [`init`](Self::init) succeeded.
    pub(crate) fn warm_up(&self, stream: &Arc<CudaStream>) -> Result<(), GpuError> {
        let (me, size) = (self.rank, self.size);
        let peers: Vec<u32> = if size == 1 {
            vec![0]
        } else {
            (0..size).filter(|&peer| peer != me).collect()
        };
        let out = stream.alloc_zeros::<u8>(peers.len() * WARM_UP_BYTES)?;
        let mut back = stream.alloc_zeros::<u8>(peers.len() * WARM_UP_BYTES)?;
        let mut group = WireGroup::new();
        for (i, &peer) in peers.iter().enumerate() {
            group.send(
                out.slice(i * WARM_UP_BYTES..(i + 1) * WARM_UP_BYTES),
                peer,
                stream,
            );
        }
        let parts: Vec<(usize, u32)> = peers.iter().map(|&peer| (WARM_UP_BYTES, peer)).collect();
        group.recv_parts(back.as_view_mut(), &parts, stream);
        group.post_with(|ops| self.post(ops))?;
        self.wait(stream)
    }

    /// Abort the communicator if it is still live; idempotent, and returns at once.
    pub(crate) fn abort(&self) {
        let mut raw = self.lock();
        Self::abort_locked(&mut raw, self.rank, &self.context);
    }

    fn lock(&self) -> MutexGuard<'_, Raw> {
        self.raw.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Mark the communicator dead and abort it on a detached thread.
    fn abort_locked(raw: &mut Raw, rank: u32, context: &Arc<CudaContext>) {
        if raw.aborted {
            return;
        }
        // Detached: `ncclCommAbort` blocks until every stream on the device drains, so a stall that is not NCCL's own would hold the caller past its timeout.
        raw.aborted = true;
        let comm = CommPtr(std::mem::replace(&mut raw.comm, std::ptr::null_mut()));
        let owned = context.clone();
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
                abort_now(comm, rank, context);
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
                Self::abort_locked(raw, self.rank, &self.context);
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
        Self::abort_locked(raw, self.rank, &self.context);
        Err(GpuError::Timeout { what: waiting_on })
    }

    fn post(&self, ops: &[WireOp<'_>]) -> Result<(), GpuError> {
        let mut raw = self.lock();
        Self::usable(&raw, "ncclGroupStart")?;
        if let Some(op) = ops
            .iter()
            .find(|op| op.peer() >= self.size || op.stream().context() != &self.context)
        {
            return Err(GpuError::Nccl {
                code: sys::ncclResult_t::ncclInvalidArgument as i32,
                what: format!(
                    "a wire op to peer {} of {} on device {}, for a communicator on device {}",
                    op.peer(),
                    self.size,
                    op.stream().context().ordinal(),
                    self.context.ordinal()
                ),
            });
        }
        self.context.bind_to_thread()?;
        nccl::group_start().map_err(|e| nccl_error(e, "ncclGroupStart"))?;
        let mut first: Option<GpuError> = None;
        for op in ops {
            let stream = op.stream().cu_stream() as sys::cudaStream_t;
            let peer = op.peer() as i32;
            let dtype = sys::ncclDataType_t::ncclUint8;
            let (ptr, bytes) = (op.ptr(), op.bytes());
            // SAFETY: `ptr` addresses `bytes` bytes of device memory on this communicator's device, borrowed by the `WireGroup` until the group is enqueued, and `comm` is live.
            let posted = unsafe {
                match op.kind() {
                    WireOpKind::Send => nccl::send(ptr as _, bytes, dtype, peer, raw.comm, stream),
                    WireOpKind::Recv => nccl::recv(ptr as _, bytes, dtype, peer, raw.comm, stream),
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
            Self::abort_locked(&mut raw, self.rank, &self.context);
            return Err(e);
        }
        if let Err(e) = ended {
            let detail = self.last_error(&raw);
            Self::abort_locked(&mut raw, self.rank, &self.context);
            return Err(nccl_error(e, format!("ncclGroupEnd{detail}")));
        }
        // A non-blocking communicator may still be enqueueing the group's kernels; until it is done, later work on the streams would run ahead of them.
        self.settle_locked(&mut raw, "ncclGroupEnd", "an NCCL group to be enqueued")
    }

    fn wait(&self, stream: &CudaStream) -> Result<(), GpuError> {
        Self::usable(&self.lock(), "a wait")?;
        let done = self.context.new_event(None)?;
        done.record(stream)?;
        let mut raw = self.lock();
        let timeout = raw.timeout;
        #[cfg(test)]
        let forced = std::mem::take(&mut raw.force_timeout);
        #[cfg(not(test))]
        let forced = false;
        let ready = poll_until(timeout, || {
            self.check_async_locked(&mut raw, "an NCCL group")?;
            Ok(done.try_is_complete()? && !forced)
        })?;
        if ready {
            return Ok(());
        }
        Self::abort_locked(&mut raw, self.rank, &self.context);
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
        let bound = self.context.bind_to_thread().is_ok();
        let raw = self.raw.get_mut().unwrap_or_else(PoisonError::into_inner);
        if raw.aborted {
            return;
        }
        if std::thread::panicking() || !bound {
            Self::abort_locked(raw, rank, &self.context);
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
            Self::abort_locked(raw, rank, &self.context);
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

fn abort_now(comm: CommPtr, rank: u32, context: &Arc<CudaContext>) {
    let start = Instant::now();
    if let Err(e) = context.bind_to_thread() {
        log::warn!(
            "gpu: binding device {} for the NCCL abort on rank {rank}: {e:?}",
            context.ordinal()
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
    for (word, chunk) in words.iter_mut().zip(id.internal.chunks_exact(8)) {
        *word = u64::from_ne_bytes(std::array::from_fn(|i| chunk[i] as u8));
    }
}

fn unpack_id(words: &[u64]) -> sys::ncclUniqueId {
    let mut id = sys::ncclUniqueId { internal: [0; 128] };
    for (chunk, word) in id.internal.chunks_exact_mut(8).zip(words) {
        for (target, byte) in chunk.iter_mut().zip(word.to_ne_bytes()) {
            *target = byte as std::ffi::c_char;
        }
    }
    id
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
    #[cfg(test)]
    pub(crate) fn comm(&self) -> &Arc<NcclComm> {
        &self.comm
    }
}

impl DeviceWire for NcclWire {
    #[cfg(test)]
    fn rank(&self) -> u32 {
        self.comm.rank
    }
    #[cfg(test)]
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

/// Whether the group can start NCCL, the same verdict on every rank: every rank `can` (libnccl 2.22+ loads) and no two `device` UUIDs coincide, which NCCL refuses (**collective**: one `allreduce_sum_u64` of `1 + 2 × size` words).
pub(crate) fn agree_start(
    collectives: &dyn Collectives,
    can: bool,
    device: [u64; 2],
) -> Result<(), GpuError> {
    let (rank, size) = (collectives.rank() as usize, collectives.size() as usize);
    let mut buffer = vec![0u64; 1 + 2 * size];
    buffer[0] = u64::from(!can);
    buffer[1 + 2 * rank..3 + 2 * rank].copy_from_slice(&device);
    collectives.allreduce_sum_u64(&mut buffer);
    if buffer[0] > 0 {
        return Err(GpuError::Unsupported(
            "an MPI device group exchanges over NCCL, and a rank cannot load libnccl 2.22 or newer",
        ));
    }
    let ids: Vec<[u64; 2]> = buffer[1..].chunks_exact(2).map(|w| [w[0], w[1]]).collect();
    if (0..size).any(|i| ids[i + 1..].contains(&ids[i])) {
        return Err(GpuError::Unsupported(
            "two ranks of an MPI device group share a device, which NCCL refuses; give every rank its own GPU",
        ));
    }
    Ok(())
}

/// A device's UUID as the two words [`agree_start`] compares.
pub(crate) fn device_uuid(context: &CudaContext) -> Result<[u64; 2], GpuError> {
    let id = context.uuid()?;
    let bytes: [u8; 16] = std::array::from_fn(|i| id.bytes[i] as u8);
    Ok([
        u64::from_ne_bytes(bytes[..8].try_into().expect("eight bytes")),
        u64::from_ne_bytes(bytes[8..].try_into().expect("eight bytes")),
    ])
}

#[cfg(test)]
mod tests;
