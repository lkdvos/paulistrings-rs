//! The MPI transport and the distributed driver over it, one partition per rank.
//!
//! The library never calls `MPI_Init` or `MPI_Finalize`: the application owns the [`rsmpi`] `Universe`, initialized with at least `Threading::Serialized`, since the layer loop calls MPI from any worker of the pinned pool, one at a time.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

use ::mpi::collective::{CommunicatorCollectives, Root, SystemOperation};
use ::mpi::point_to_point::{Destination, Source};
use ::mpi::raw::FromRaw;
use ::mpi::request::{RequestCollection, Scope};
use ::mpi::topology::{Communicator, Rank, SimpleCommunicator};

mod pipeline;

use pipeline::{note_slot, ChunkPipeline, SLOT_EARLY, SLOT_SEND};

use super::distributed::DistributedSum;
use super::topology::{PartitionConfig, Placement};
use super::transport::{AlreadyHere, ChunkMap, ChunkWait, Collectives, Payload, Transport, ROOT};
use super::truncation::PartitionedTruncation;
use crate::circuit::Circuit;
use crate::engine::{Direction, PropagateOptions};
use crate::pauli_sum::PauliSum;

/// The `rsmpi` crate this transport is built against; create the `Universe` from it, since two copies of `mpi` in one binary would not share their statics.
pub use ::mpi as rsmpi;

/// A [`DistributedSum`] over the MPI transport, the persistent form of a [`propagate_mpi`] run.
pub type MpiSum<const W: usize> = DistributedSum<W, MpiTransport>;

/// `log` target shared with the partition topology and runtime.
const LOG_TARGET: &str = "paulistrings::partitioned";

/// Version of the header framing, bumped when it changes.
const WIRE_VERSION: u32 = 1;

/// Tags pack `epoch:11 | kind:4`, inside the guaranteed `MPI_TAG_UB` of 32767.
const KIND_BITS: i32 = 4;
const EPOCHS: u32 = 1 << 11;

const KIND_EXCHANGE_HEADER: i32 = 0;
const KIND_EXCHANGE_PART: i32 = 1;
const KIND_GATHER_HEADER: i32 = 2;
const KIND_GATHER_PART: i32 = 3;
const KIND_EXCHANGE_EARLY: i32 = 4;

/// One exchange's three tags: framing headers, early parts, bulk chunks.
#[derive(Clone, Copy)]
struct Tags {
    header: Rank,
    early: Rank,
    part: Rank,
}

impl Tags {
    fn exchange(epoch: u32) -> Self {
        Self {
            header: MpiTransport::tag(KIND_EXCHANGE_HEADER, epoch),
            early: MpiTransport::tag(KIND_EXCHANGE_EARLY, epoch),
            part: MpiTransport::tag(KIND_EXCHANGE_PART, epoch),
        }
    }
}

/// Largest single message in bytes, below the `i32` count ceiling.
const DEFAULT_CHUNK_BYTES: usize = 1 << 30;

/// What can go wrong building an [`MpiTransport`]; everything after construction is a panic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MpiError {
    /// `MPI_Init` has not been called, or `MPI_Finalize` already has.
    NotInitialized,
    /// `MPI_Comm_dup` failed, with the error code it returned.
    DuplicateFailed(i32),
    /// The handle passed to [`MpiTransport::from_raw_handle`] was `MPI_COMM_NULL`.
    NullCommunicator,
    /// The communicator's size is not a power of two.
    SizeNotPowerOfTwo(u32),
}

impl std::fmt::Display for MpiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotInitialized => write!(
                f,
                "MPI is not initialized: paulistrings never calls MPI_Init, so the application \
                 must create the universe (paulistrings::mpi::rsmpi::initialize_with_threading) \
                 and keep it alive for the transport's lifetime",
            ),
            Self::DuplicateFailed(code) => {
                write!(f, "MPI_Comm_dup failed with error code {code}")
            }
            Self::NullCommunicator => write!(f, "the communicator handle is MPI_COMM_NULL"),
            Self::SizeNotPowerOfTwo(size) => write!(
                f,
                "an MPI group of {size} ranks cannot be a partitioning: a partition is named by \
                 log2(P) GF(2) rows, so the rank count must be a power of two",
            ),
        }
    }
}

impl std::error::Error for MpiError {}

/// One MPI rank's [`Transport`] endpoint.
///
/// Holds its own duplicate of the communicator, so its tags never collide with the application's traffic; it must not outlive the `Universe`.
pub struct MpiTransport {
    comm: SimpleCommunicator,
    rank: u32,
    size: u32,
    /// Incremented per exchange and per gather.
    epoch: AtomicU32,
    /// Input copy for `allreduce_sum_u64`, since rsmpi's safe `all_reduce_into` has no `MPI_IN_PLACE`.
    scratch: Mutex<Vec<u64>>,
    /// Largest message in bytes.
    chunk: usize,
}

// SAFETY: `MPI_Comm` is a raw pointer on Open MPI, so `SimpleCommunicator` is neither `Send` nor `Sync`.
// The handle is a plain value, and no two threads ever call MPI through it at once, which the `Threading::Serialized` level the application requested makes safe.
unsafe impl Send for MpiTransport {}
unsafe impl Sync for MpiTransport {}

impl MpiTransport {
    /// Duplicate `comm` and wrap it. **Collective** over `comm`.
    ///
    /// # Panics
    ///
    /// If the communicator's size is not a power of two; [`from_raw_handle`](Self::from_raw_handle) is the fallible entry point.
    pub fn from_communicator(comm: &impl Communicator) -> Self {
        Self::adopt(comm.duplicate()).unwrap_or_else(|err| panic!("{err}"))
    }

    /// Duplicate a raw `MPI_Comm` handle, such as a Python host's, and wrap it. **Collective**; the caller keeps ownership of `handle`.
    ///
    /// # Safety
    ///
    /// `handle` must be a live intra-communicator of the process's MPI library, bit-for-bit what C would pass.
    ///
    /// # Errors
    ///
    /// [`MpiError::NotInitialized`] if MPI is not up, [`MpiError::NullCommunicator`] for a null handle, [`MpiError::DuplicateFailed`] if `MPI_Comm_dup` reports an error, and [`MpiError::SizeNotPowerOfTwo`] for a group that cannot be a partitioning.
    pub unsafe fn from_raw_handle(handle: usize) -> Result<Self, MpiError> {
        if !::mpi::environment::is_initialized() {
            return Err(MpiError::NotInitialized);
        }
        let raw = comm_from_usize(handle);
        if raw == ::mpi::ffi::RSMPI_COMM_NULL {
            return Err(MpiError::NullCommunicator);
        }
        let mut duplicate: ::mpi::ffi::MPI_Comm = ::mpi::ffi::RSMPI_COMM_NULL;
        // MPI_SUCCESS is 0 by the standard.
        let code = ::mpi::ffi::MPI_Comm_dup(raw, &mut duplicate);
        if code != 0 {
            return Err(MpiError::DuplicateFailed(code));
        }
        Self::adopt(SimpleCommunicator::from_raw(duplicate))
    }

    /// Take ownership of an already duplicated communicator.
    fn adopt(comm: SimpleCommunicator) -> Result<Self, MpiError> {
        let rank = comm.rank() as u32;
        let size = comm.size() as u32;
        if !size.is_power_of_two() {
            return Err(MpiError::SizeNotPowerOfTwo(size));
        }

        let built = env!("PAULISTRINGS_MPI_BUILD_VERSION");
        let running = ::mpi::environment::library_version().unwrap_or_default();
        let running = running.trim();
        if rank == 0 {
            log::info!(
                target: LOG_TARGET,
                "MPI transport: rank {rank}/{size}, thread support {:?}, library {running:?} \
                 (built against {built:?})",
                ::mpi::environment::threading_support(),
            );
        }
        if !running.is_empty() && !version_matches(built, running) {
            // Not fatal: the version strings are free-form, so this may be a false alarm.
            log::warn!(
                target: LOG_TARGET,
                "MPI library version {running:?} differs from the one this build probed \
                 ({built:?}); Open MPI 4.1 and 5.0 share a soname, so a stale module load links \
                 but may misbehave",
            );
        }

        Ok(Self {
            comm,
            rank,
            size,
            epoch: AtomicU32::new(0),
            scratch: Mutex::new(Vec::new()),
            chunk: DEFAULT_CHUNK_BYTES,
        })
    }

    /// The communicator this transport owns, for a collective the caller runs over the same group.
    pub fn communicator(&self) -> &SimpleCommunicator {
        &self.comm
    }

    /// Override the largest message the transport sends, in bytes (default 1 GiB), so tests can reach the chunked path.
    ///
    /// # Panics
    ///
    /// If `bytes` is zero.
    pub fn with_chunk_bytes(mut self, bytes: usize) -> Self {
        assert!(bytes > 0, "the chunk size must be positive");
        self.chunk = bytes;
        self
    }

    /// The MPI library version string this build probed through `mpicc`.
    pub fn build_library_version() -> &'static str {
        env!("PAULISTRINGS_MPI_BUILD_VERSION")
    }

    fn next_epoch(&self) -> u32 {
        self.epoch.fetch_add(1, Ordering::Relaxed) % EPOCHS
    }

    fn tag(kind: i32, epoch: u32) -> Rank {
        ((epoch % EPOCHS) as i32) << KIND_BITS | kind
    }

    /// Send this rank's gather contribution to the root and wait it out.
    fn send_gather(&self, parts: &[&[u8]], header_tag: Rank, part_tag: Rank) {
        let header = encode_header(self.rank, parts);
        let header: &[u8] = bytemuck::cast_slice(&header);
        let sends = 1 + parts
            .iter()
            .map(|p| chunk_count(p.len(), self.chunk))
            .sum::<usize>();
        let peer = self.comm.process_at_rank(ROOT as Rank);
        ::mpi::request::multiple_scope(sends, |scope, requests| {
            requests.add(peer.immediate_send_with_tag(scope, header, header_tag));
            for part in parts {
                for chunk in part.chunks(self.chunk) {
                    requests.add(peer.immediate_send_with_tag(scope, chunk, part_tag));
                }
            }
            let mut done = Vec::with_capacity(sends);
            requests.wait_all(&mut done);
        });
    }

    /// Post one partner's header, early parts, then bulk parts chunk-major, so non-overtaking delivers chunk `k` before chunk `k + 1`.
    #[allow(clippy::too_many_arguments)]
    fn post_layer_send<'a, Sc>(
        &self,
        dst: usize,
        header: &'a [u8],
        early: &[&'a [u8]],
        bulk: &[Vec<&'a [u8]>],
        chunks: usize,
        tags: Tags,
        scope: Sc,
        requests: &mut RequestCollection<'a, [u8]>,
        slot_of: &mut Vec<usize>,
    ) where
        Sc: Scope<'a> + Copy,
    {
        let peer = self.comm.process_at_rank(dst as Rank);
        let mut post = |req, slot_of: &mut Vec<usize>| {
            let i = requests.add(req);
            note_slot(slot_of, i, SLOT_SEND);
        };
        post(
            peer.immediate_send_with_tag(scope, header, tags.header),
            slot_of,
        );
        for part in early {
            for piece in part.chunks(self.chunk) {
                post(
                    peer.immediate_send_with_tag(scope, piece, tags.early),
                    slot_of,
                );
            }
        }
        for k in 0..chunks {
            for part in bulk {
                for piece in part[k].chunks(self.chunk) {
                    post(
                        peer.immediate_send_with_tag(scope, piece, tags.part),
                        slot_of,
                    );
                }
            }
        }
    }

    /// Blocking receive of one framing header from `src`, returning the declared part lengths.
    fn recv_header(&self, src: usize, header_tag: Rank) -> Vec<usize> {
        let peer = self.comm.process_at_rank(src as Rank);
        let (bytes, _status) = peer.receive_vec_with_tag::<u8>(header_tag);
        decode_header(&bytes, src as u32, self.rank)
    }

    /// The root's half of the gather into `(source rank, sized parts)`, in `send_gather`'s message order.
    fn recv_gather(&self, buffers: &mut [(usize, Vec<&mut [u8]>)], part_tag: Rank) {
        let mut chunks: Vec<(usize, &mut [u8])> = Vec::new();
        for (src, parts) in buffers.iter_mut() {
            let src = *src;
            for part in parts.iter_mut() {
                for chunk in part.chunks_mut(self.chunk) {
                    chunks.push((src, chunk));
                }
            }
        }

        let n = chunks.len();
        ::mpi::request::multiple_scope(n, |scope, requests| {
            for (src, chunk) in chunks {
                requests.add(
                    self.comm
                        .process_at_rank(src as Rank)
                        .immediate_receive_into_with_tag(scope, chunk, part_tag),
                );
            }
            let mut done = Vec::with_capacity(n);
            requests.wait_all(&mut done);
        });
    }
}

/// Rebuild an `MPI_Comm` (a pointer on Open MPI, an `int` on MPICH) from a `usize` by width-checked reinterpretation.
///
/// # Safety
///
/// `handle` must be what C would pass as an `MPI_Comm` on this platform.
unsafe fn comm_from_usize(handle: usize) -> ::mpi::ffi::MPI_Comm {
    match size_of::<::mpi::ffi::MPI_Comm>() {
        n if n == size_of::<usize>() => std::mem::transmute_copy(&handle),
        4 => std::mem::transmute_copy(&(handle as u32)),
        n => panic!("unsupported MPI_Comm width: {n} bytes"),
    }
}

/// Whether the running library's banner mentions the first dotted number of the build's version string.
fn version_matches(built: &str, running: &str) -> bool {
    let number = built
        .split_whitespace()
        .find(|tok| tok.contains('.') && tok.starts_with(|c: char| c.is_ascii_digit()));
    match number {
        Some(number) => running.contains(number),
        None => true,
    }
}

/// The framing header for `parts`: `(source << 32) | version`, the part count, one byte length per part.
fn encode_header(src: u32, parts: &[&[u8]]) -> Vec<u64> {
    let mut words = Vec::with_capacity(parts.len() + 2);
    words.push((u64::from(src) << 32) | u64::from(WIRE_VERSION));
    words.push(parts.len() as u64);
    words.extend(parts.iter().map(|p| p.len() as u64));
    words
}

/// The part lengths [`encode_header`] declared; panics on a malformed header, another wire version, or a source other than `expected_src`.
fn decode_header(bytes: &[u8], expected_src: u32, this_rank: u32) -> Vec<usize> {
    assert!(
        bytes.len() >= 2 * size_of::<u64>() && bytes.len().is_multiple_of(size_of::<u64>()),
        "rank {this_rank}: framing header from rank {expected_src} is {} bytes, expected a whole number \
         of u64 words, at least two",
        bytes.len(),
    );
    let words: Vec<u64> = bytes
        .chunks_exact(size_of::<u64>())
        .map(bytemuck::pod_read_unaligned)
        .collect();

    let version = (words[0] & 0xffff_ffff) as u32;
    assert_eq!(
        version, WIRE_VERSION,
        "rank {this_rank}: rank {expected_src} speaks exchange wire version {version}, this rank speaks \
         {WIRE_VERSION} — the ranks are not running the same build",
    );
    let src = (words[0] >> 32) as u32;
    assert_eq!(
        src, expected_src,
        "rank {this_rank}: a framing header tagged for this layer came from rank {src}, not the expected \
         rank {expected_src}",
    );
    let n_parts = words[1] as usize;
    assert_eq!(
        words.len(),
        n_parts + 2,
        "rank {this_rank}: rank {expected_src} declared {n_parts} parts in a {}-word header",
        words.len(),
    );
    words[2..].iter().map(|&len| len as usize).collect()
}

/// Messages a part of `len` bytes is sent in, exactly `slice::chunks(chunk).count()`.
fn chunk_count(len: usize, chunk: usize) -> usize {
    len.div_ceil(chunk)
}

impl super::transport::sealed::Sealed for MpiTransport {}

impl Collectives for MpiTransport {
    fn rank(&self) -> u32 {
        self.rank
    }

    fn size(&self) -> u32 {
        self.size
    }

    fn allreduce_max_u8(&self, v: u8) -> u8 {
        if self.size == 1 {
            return v;
        }
        let send = [v];
        let mut recv = [0u8];
        self.comm
            .all_reduce_into(&send[..], &mut recv[..], SystemOperation::max());
        recv[0]
    }

    fn allreduce_sum_u64(&self, buffer: &mut [u64]) {
        if self.size == 1 {
            return;
        }
        let mut scratch = self
            .scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        scratch.clear();
        scratch.extend_from_slice(buffer);
        self.comm
            .all_reduce_into(&scratch[..], buffer, SystemOperation::sum());
    }

    /// Not `MPI_Allreduce`, which does not promise every rank the same bits: reduce to the root and broadcast.
    fn allreduce_sum_f64(&self, buffer: &mut [f64]) {
        if self.size == 1 {
            return;
        }
        let root = self.comm.process_at_rank(ROOT as Rank);
        if self.rank as usize == ROOT {
            let send = buffer.to_vec();
            root.reduce_into_root(&send[..], buffer, SystemOperation::sum());
        } else {
            root.reduce_into(&buffer[..], SystemOperation::sum());
        }
        root.broadcast_into(buffer);
    }

    fn barrier(&self) {
        if self.size == 1 {
            return;
        }
        self.comm.barrier();
    }
}

impl Transport for MpiTransport {
    /// Wire framing and deadlock freedom: ARCHITECTURE.md §Transport composition.
    fn exchange_layer<P, F, R>(
        &self,
        send: Vec<Option<P>>,
        spare: &mut Vec<P>,
        map: &ChunkMap,
        body: F,
    ) -> (Vec<Option<P>>, R)
    where
        P: Payload,
        F: FnOnce(&[Option<P>], &dyn ChunkWait) -> R,
    {
        let n = self.size as usize;
        assert_eq!(
            send.len(),
            n,
            "exchange: send has {} entries, expected one entry per rank ({n})",
            send.len(),
        );
        assert!(
            send[self.rank as usize].is_none(),
            "exchange: send[{}] is this rank's own slot and must be None",
            self.rank,
        );
        let recv: Vec<Option<P>> = (0..n).map(|_| None).collect();
        if n == 1 {
            let out = body(&recv, &AlreadyHere);
            return (recv, out);
        }

        let tags = Tags::exchange(self.next_epoch());
        let partners: Vec<usize> = (0..n).filter(|&q| send[q].is_some()).collect();
        if partners.is_empty() {
            let out = body(&recv, &AlreadyHere);
            return (recv, out);
        }

        let all_parts: Vec<Vec<&[u8]>> = partners
            .iter()
            .map(|&q| send[q].as_ref().expect("a partner's payload").byte_parts())
            .collect();
        let headers: Vec<Vec<u64>> = all_parts
            .iter()
            .map(|p| encode_header(self.rank, p))
            .collect();
        let early: Vec<Vec<&[u8]>> = partners
            .iter()
            .map(|&q| send[q].as_ref().expect("a partner's payload").early_parts())
            .collect();
        let bulk: Vec<Vec<Vec<&[u8]>>> = partners
            .iter()
            .map(|&q| {
                send[q]
                    .as_ref()
                    .expect("a partner's payload")
                    .bulk_parts(map)
            })
            .collect();
        let chunks = map.chunks();

        // SAFETY: while `body` runs, the `&mut` views MPI fills and the `&[Option<P>]` the body reads alias.
        // They never touch the same bytes at once: the early parts complete before `body` starts, and a chunk's columns are read only after `wait_chunk` saw its receives complete.
        let recv_cell = std::cell::UnsafeCell::new(recv);

        let out =
            ::mpi::request::multiple_scope(partners.len() * (2 + 3 * chunks), |scope, requests| {
                let mut slot_of: Vec<usize> = Vec::new();
                for (i, &dst) in partners.iter().enumerate() {
                    self.post_layer_send(
                        dst,
                        bytemuck::cast_slice(&headers[i]),
                        &early[i],
                        &bulk[i],
                        chunks,
                        tags,
                        scope,
                        requests,
                        &mut slot_of,
                    );
                }

                // Every send is posted before the first blocking receive.
                let mut lens: Vec<Vec<usize>> = Vec::with_capacity(partners.len());
                for &src in &partners {
                    lens.push(self.recv_header(src, tags.header));
                }

                // SAFETY: no other alias exists yet; the collection holds only this rank's sends.
                let recv_mut = unsafe { &mut *recv_cell.get() };
                for &src in &partners {
                    recv_mut[src] = Some(spare.pop().unwrap_or_default());
                }
                let mut early_left = 0usize;
                let mut lens = lens.into_iter();
                for (q, slot) in recv_mut.iter_mut().enumerate() {
                    let Some(payload) = slot.as_mut() else {
                        continue;
                    };
                    let lens = lens.next().expect("one header per partner");
                    let peer = self.comm.process_at_rank(q as Rank);
                    for view in payload.early_receive_into(&lens) {
                        for piece in view.chunks_mut(self.chunk) {
                            let i = requests.add(
                                peer.immediate_receive_into_with_tag(scope, piece, tags.early),
                            );
                            note_slot(&mut slot_of, i, SLOT_EARLY);
                            early_left += 1;
                        }
                    }
                }

                let mut done = Vec::new();
                while early_left > 0 {
                    requests.wait_some(&mut done);
                    for &(i, _, _) in &done {
                        if slot_of[i] == SLOT_EARLY {
                            early_left -= 1;
                        }
                    }
                }

                // SAFETY: as above; the collection holds only completed requests and this rank's sends.
                let recv_mut = unsafe { &mut *recv_cell.get() };
                for payload in recv_mut.iter_mut().flatten() {
                    payload.finish_receive();
                }

                for (q, slot) in recv_mut.iter_mut().enumerate() {
                    let Some(payload) = slot.as_mut() else {
                        continue;
                    };
                    let peer = self.comm.process_at_rank(q as Rank);
                    let mut parts: Vec<Vec<Option<&mut [u8]>>> = payload
                        .bulk_receive_into(map)
                        .into_iter()
                        .map(|pieces| pieces.into_iter().map(Some).collect())
                        .collect();
                    for k in 0..chunks {
                        for part in parts.iter_mut() {
                            let piece = part[k].take().expect("one view per chunk");
                            for piece in piece.chunks_mut(self.chunk) {
                                let i = requests.add(
                                    peer.immediate_receive_into_with_tag(scope, piece, tags.part),
                                );
                                note_slot(&mut slot_of, i, k);
                            }
                        }
                    }
                }

                let mut pipeline = ChunkPipeline::new(requests, slot_of, chunks.max(1));
                // SAFETY: see `recv_cell`.
                let out = body(unsafe { &*recv_cell.get() }, &pipeline);
                pipeline.finish();
                out
            });

        spare.extend(send.into_iter().flatten());
        (recv_cell.into_inner(), out)
    }

    /// Overridden because this transport infers its partner set from the `Some` positions.
    fn gather_to_root(&self, parts: Vec<&[u8]>) -> Option<Vec<Vec<Vec<u8>>>> {
        let n = self.size as usize;
        let this_rank = self.rank as usize;
        if n == 1 {
            return Some(vec![parts.iter().map(|p| p.to_vec()).collect()]);
        }

        let epoch = self.next_epoch();
        let (header_tag, part_tag) = (
            Self::tag(KIND_GATHER_HEADER, epoch),
            Self::tag(KIND_GATHER_PART, epoch),
        );

        if this_rank != ROOT {
            self.send_gather(&parts, header_tag, part_tag);
            return None;
        }

        let mut buffers: Vec<Vec<Vec<u8>>> = (1..n)
            .map(|src| {
                self.recv_header(src, header_tag)
                    .into_iter()
                    .map(|len| vec![0u8; len])
                    .collect()
            })
            .collect();
        let mut views: Vec<(usize, Vec<&mut [u8]>)> = buffers
            .iter_mut()
            .enumerate()
            .map(|(i, parts)| (i + 1, parts.iter_mut().map(Vec::as_mut_slice).collect()))
            .collect();
        self.recv_gather(&mut views, part_tag);
        drop(views);

        let mut out = Vec::with_capacity(n);
        out.push(parts.iter().map(|p| p.to_vec()).collect());
        out.extend(buffers);
        Some(out)
    }
}

/// One partition over the CPUs the launcher left in the affinity mask, memory bound to its node.
///
/// Under `mpirun --map-by ppr:1:numa --bind-to numa` (or `srun --cpu-bind=ldoms`) that is one NUMA domain; unbound, it is the whole node.
pub fn default_config() -> PartitionConfig {
    PartitionConfig {
        placement: Placement::Auto {
            max_partitions: Some(1),
        },
        bind_memory: true,
        partition_row_seed: None,
    }
}

/// Propagate the replicated `sum` through `circuit` with one partition per rank of `comm`, returning the gathered result on rank 0 and `None` elsewhere.
///
/// **Collective**: every rank passes the same sum, circuit, direction and options.
/// A driver stepping through many circuits holds an [`MpiSum`] instead, which also takes a placement other than [`default_config`].
///
/// # Panics
///
/// If the group size is not a power of two, if the placement cannot be resolved, or as [`DistributedSum::propagate_with`].
pub fn propagate_mpi<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: PauliSum<W>,
    policy: &T,
    direction: Direction,
    options: PropagateOptions,
    comm: &impl Communicator,
) -> Option<PauliSum<W>>
where
    T: PartitionedTruncation<W> + ?Sized,
{
    let transport = MpiTransport::from_communicator(comm);
    let mut split = MpiSum::<W>::scatter(sum, transport, &default_config())
        .unwrap_or_else(|err| panic!("could not place the MPI rank's partition: {err}"));
    split.propagate_with(circuit, policy, direction, options);
    split.gather()
}

#[cfg(test)]
mod tests;
