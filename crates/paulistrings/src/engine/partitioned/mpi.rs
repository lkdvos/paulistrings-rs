//! The MPI transport: one partition per rank.
//!
//! [`MpiTransport`] is the distributed implementation of the two traits in
//! `engine::partitioned::transport` — nothing else in the partitioned engine
//! changes, because the exchange's wire unit, its receive rule and the
//! collective-order invariant were designed against those traits and not
//! against the in-process channel matrix (ARCHITECTURE.md §Partitioning,
//! *Transport composition*).
//!
//! # The library never initializes MPI
//!
//! There is no `Universe` in here and no `MPI_Init`. The application owns
//! initialization and finalization: it creates the universe, hands the
//! communicator to [`MpiTransport::from_communicator`] (or a raw handle to
//! [`MpiTransport::from_raw_handle`], for a Python or C caller whose MPI is
//! already up), and drops the universe last. That is the only arrangement that
//! composes — a library that called `MPI_Init` would fight `mpi4py`, a host
//! application, or a second Rust crate for the same one-shot global.
//!
//! Two consequences worth spelling out:
//!
//! - **[`rsmpi`] is re-exported.** `Universe`, `Threading` and the
//!   `Communicator` trait are types of the `mpi` crate, so a caller must build
//!   its universe with the *same* version this crate links. Use
//!   `paulistrings::mpi::rsmpi` rather than a separate `mpi` dependency.
//! - **Request at least `rsmpi::Threading::Serialized`.** The partition's
//!   layer loop runs inside a pinned Rayon pool (`ThreadPool::install`), so
//!   the thread making MPI calls
//!   is a pool worker rather than the process's main thread, and it need not be
//!   the same worker on every layer. Only ever *one* thread calls MPI at a time,
//!   which is exactly `MPI_THREAD_SERIALIZED`; `Funneled` would be a false
//!   claim.
//!
//! # What crosses the wire
//!
//! [`Transport::exchange`] is **point-to-point, not an all-to-all-v**. The
//! partner set is symmetric by construction: a remote delta moves rank `R`'s
//! rows to `R ⊕ pd`, and the same delta on rank `R ⊕ pd` moves its rows back
//! to `R`, so "who sends to me" is exactly "who I send to" — the `Some`
//! positions of the caller's `send` vector. No exchange of counts is needed to
//! learn the partner set, and no rank has to participate in a collective sized
//! by the whole group.
//!
//! Per partner the framing is one **header** message followed by the payload's
//! [`Payload::byte_parts`], in order:
//!
//! ```text
//! header  : u64[2 + n_parts]  = [ (src << 32) | WIRE_VERSION, n_parts, len[0], …, len[n_parts-1] ]
//! parts   : the bytes of part 0, then part 1, … each split into ≤ CHUNK-byte messages
//! ```
//!
//! The header is the only message whose size the receiver cannot predict, so it
//! is received with a matched probe (`receive_vec_with_tag`); everything after
//! it is a posted [`immediate_receive_into`](mpi::point_to_point::Source::immediate_receive_into)
//! of a known length. Parts share one tag and MPI's non-overtaking guarantee
//! for a `(source, tag, communicator)` triple matches them up in posting order,
//! which is why both sides walk partners, parts and chunks in the same
//! ascending order.
//!
//! **The parts are received into the receiving payload's own columns.** Those
//! declared lengths are all the receiver needs to size a
//! [`PartnerPayload`](super::transport::PartnerPayload) — five parts per block,
//! and each block's shape follows from the lengths — so
//! [`Payload::recv_into`] hands back mutable byte views of the very `Vec<[u64;
//! W]>` and `Vec<Complex64>` the coset loop will read, and MPI writes into
//! them. There is no staging buffer and no decode pass; what is left afterwards
//! is [`Payload::finish_recv`], which checks the header against the shape the
//! lengths implied. The payloads themselves come from the caller's pool and go
//! back into it (`spare`), so a steady-state layer's receive allocates nothing.
//!
//! **Chunking.** MPI counts are `i32`, so one message carries under 2 GiB; a
//! coefficient column at large `m` can exceed that. Every part is therefore
//! split into chunks of at most [`DEFAULT_CHUNK_BYTES`] (1 GiB) under the same
//! tag. An empty part is zero chunks — `slice::chunks` and `slice::chunks_mut`
//! agree on that, which is what keeps the two sides' message counts equal
//! without a second rule.
//!
//! *Deviation from the phase-3 brief:* the brief proposed viewing the parts as
//! `u64` (raising the per-message ceiling to 16 GiB). That is not sound —
//! `BlockHeader` is four `u32`s and the CSR `offsets` column is `Vec<u32>`, both
//! 4-aligned, so `bytemuck::cast_slice::<u8, u64>` on their bytes would panic.
//! Bytes plus chunking has the same effect and no alignment precondition.
//!
//! **Endianness and padding.** The parts are raw host bytes (that is what
//! `byte_parts` is), sent as `MPI_UINT8_T`, so a run must be homogeneous — the
//! same architecture and the same `W` on every rank. The header is a
//! `Vec<u64>` viewed as bytes for the same reason: one datatype for the whole
//! framing keeps sends and receives in a single `RequestCollection`.
//!
//! # Tags
//!
//! `tag = epoch:11 | kind:4`, so at most 32767 — inside the `MPI_TAG_UB ≥
//! 32767` the standard guarantees. `kind` separates the exchange's header and
//! part streams from the gather's; `epoch` increments on every
//! [`Transport::exchange`] and `Transport::gather_to_root` and wraps at 2048,
//! so a straggling message from one layer cannot be mistaken for the next
//! layer's. It is a tripwire, not a protocol: the layer loop is lock-step, so
//! consecutive epochs are never in flight at once.
//!
//! # Deadlock freedom
//!
//! Every send for a call is posted (non-blocking) before the call's first
//! receive. The blocking header receive can therefore always complete: the
//! matching send is already in flight, and `MPI_Mrecv` drives the rendezvous
//! itself if the message is large. The part receives are all posted at once and
//! waited on together, so a partner's rendezvous has somewhere to land.
//!
//! What *is* a hang: an asymmetric partner set. If rank `q` sends here and this
//! rank does not send to `q`, the message is never received. That cannot happen
//! for a layer's exchange (the partner set is a function of the delta mask, and
//! every rank computes it identically), and [`Collectives::check_consistency`]
//! catches the wider version of the same mistake — ranks driven through
//! different circuits — once per propagation.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

use ::mpi::collective::{CommunicatorCollectives, SystemOperation};
use ::mpi::point_to_point::{Destination, Source};
use ::mpi::raw::FromRaw;
use ::mpi::request::{RequestCollection, Scope};
use ::mpi::topology::{Communicator, Rank, SimpleCommunicator};

use super::distributed::DistributedSum;
use super::topology::{PartitionConfig, Placement};
#[cfg(feature = "phase-timing")]
use super::transport::ExchangeTimings;
use super::transport::{AlreadyHere, ChunkMap, ChunkWait, Collectives, Payload, Transport, ROOT};
use super::truncation::PartitionedTruncation;
use crate::circuit::Circuit;
use crate::engine::{Direction, PropagateOptions};
use crate::pauli_sum::PauliSum;

/// The `rsmpi` crate this transport is built against.
///
/// Re-exported so a caller creates its `Universe` from the same version:
/// `paulistrings::mpi::rsmpi::initialize_with_threading(...)`. Two copies of
/// `mpi` in one binary would each have their own datatype and operation
/// statics.
pub use ::mpi as rsmpi;

/// The distributed sum a [`propagate_mpi`] run drives: one partition per rank,
/// over the MPI transport. The persistent form a Trotter driver holds across
/// steps.
pub type MpiSum<const W: usize> = DistributedSum<W, MpiTransport>;

/// `log` target for the transport's construction diagnostics, shared with
/// [`topology`](super::topology) and [`runtime`](super::runtime).
const LOG_TARGET: &str = "paulistrings::partitioned";

/// Version of the header framing, in the low 32 bits of the header's first
/// word. Bumped when the framing changes; a mismatch is an error naming both
/// versions rather than a garbage length vector.
const WIRE_VERSION: u32 = 1;

/// Bits of the tag reserved for the message kind.
const KIND_BITS: i32 = 4;
/// Epochs before the counter wraps: 11 bits, so `tag < 2^15`.
const EPOCHS: u32 = 1 << 11;

/// Tag kinds. The exchange and the gather have separate streams so a bug in one
/// cannot consume the other's messages.
const KIND_EXCHANGE_HEADER: i32 = 0;
const KIND_EXCHANGE_PART: i32 = 1;
const KIND_GATHER_HEADER: i32 = 2;
const KIND_GATHER_PART: i32 = 3;
/// The two-phase exchange's *early* stream: block headers and CSR offsets.
/// A stream of its own so the bulk parts, posted later and waited on chunk by
/// chunk, cannot be matched against it however the two are interleaved.
const KIND_EXCHANGE_EARLY: i32 = 4;

/// The three tags one exchange's framing uses: the framing headers, the early
/// parts, and the bulk chunks. Carried together so the phases of one call
/// cannot be mixed up, and so a call cannot reach for the gather's stream.
#[derive(Clone, Copy)]
struct Tags {
    header: Rank,
    early: Rank,
    part: Rank,
}

impl Tags {
    /// The exchange's tags for `epoch`.
    fn exchange(epoch: u32) -> Self {
        Self {
            header: MpiTransport::tag(KIND_EXCHANGE_HEADER, epoch),
            early: MpiTransport::tag(KIND_EXCHANGE_EARLY, epoch),
            part: MpiTransport::tag(KIND_EXCHANGE_PART, epoch),
        }
    }
}

/// Largest single message the transport sends, in bytes.
///
/// MPI counts are `i32`, so the hard ceiling is just under 2 GiB; 1 GiB leaves
/// room and is large enough that a chunked part is the exception. See the
/// module docs on chunking; [`MpiTransport::with_chunk_bytes`] overrides it
/// for the tests that need the chunked path to fire.
const DEFAULT_CHUNK_BYTES: usize = 1 << 30;

/// What can go wrong building an [`MpiTransport`]. Everything after
/// construction is a panic, as everywhere else in the engine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MpiError {
    /// `MPI_Init` has not been called (or `MPI_Finalize` already has). The
    /// library never initializes MPI itself — see the module docs.
    NotInitialized,
    /// `MPI_Comm_dup` failed, with the error code it returned.
    DuplicateFailed(i32),
    /// The handle passed to [`MpiTransport::from_raw_handle`] was
    /// `MPI_COMM_NULL`.
    NullCommunicator,
    /// The communicator's size is not a power of two. The partition rows name
    /// a partition with `log2(P)` GF(2) rows, so `P` must be a power of two
    /// (ARCHITECTURE.md §Partitioning).
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

/// One MPI rank's endpoint: the [`Transport`] the partitioned engine drives a
/// distributed run through.
///
/// Holds a **duplicate** of the communicator it was built from, freed when the
/// transport drops, so the engine's tags can never collide with the
/// application's traffic on the original. That duplicate must not outlive the
/// `Universe`; hold the transport (and the `DistributedSum` built on it)
/// inside the universe's scope.
///
/// # Examples
///
/// ```no_run
/// use paulistrings::mpi::{rsmpi, MpiTransport};
/// use paulistrings::engine::partitioned::Collectives;
///
/// let (universe, _threading) =
///     rsmpi::initialize_with_threading(rsmpi::Threading::Serialized).expect("MPI initializes");
/// let world = universe.world();
/// let transport = MpiTransport::from_communicator(&world);
/// println!("rank {} of {}", transport.rank(), transport.size());
/// // `universe` drops last, calling MPI_Finalize.
/// ```
pub struct MpiTransport {
    /// The duplicated communicator this transport owns.
    comm: SimpleCommunicator,
    /// This rank, cached: `Collectives::rank` is called per layer.
    rank: u32,
    /// The group size, cached.
    size: u32,
    /// Tag epoch, incremented per exchange and per gather.
    epoch: AtomicU32,
    /// Send-side scratch for [`Collectives::allreduce_sum_u64`] — rsmpi's safe
    /// `all_reduce_into` has no `MPI_IN_PLACE`, so the input needs a copy, and
    /// the buffer is the same length every layer.
    scratch: Mutex<Vec<u64>>,
    /// Largest message the transport sends, in bytes. A test knob (see
    /// [`with_chunk_bytes`](Self::with_chunk_bytes)); production uses
    /// [`DEFAULT_CHUNK_BYTES`].
    chunk: usize,
    /// Sub-phase laps of [`Transport::exchange`], drained by
    /// [`Transport::drain_timings`]. Measurement only.
    #[cfg(feature = "phase-timing")]
    timings: super::transport::ExchangeTimings,
}

// SAFETY: `SimpleCommunicator` is not `Send`/`Sync` because `MPI_Comm` is a raw
// pointer on Open MPI. `Collectives: Send + Sync` needs both, because the
// driving thread hands `&self` down into code that a Rayon pool has borrowed —
// but no MPI call is ever made from a pool worker other than the one driving,
// and never from two threads at once (see the `Threading::Serialized`
// requirement in the module docs). The handle itself is a plain value; MPI's
// own thread-safety, at the level the application requested, is what makes the
// calls safe.
unsafe impl Send for MpiTransport {}
unsafe impl Sync for MpiTransport {}

impl MpiTransport {
    /// Duplicate `comm` and wrap it.
    ///
    /// **Collective** over `comm`: every rank must call it, `MPI_Comm_dup`
    /// being collective. Pass the world communicator, or any sub-communicator
    /// whose size is a power of two.
    ///
    /// # Panics
    ///
    /// If the communicator's size is not a power of two (see
    /// [`MpiError::SizeNotPowerOfTwo`]). A caller that would rather handle
    /// that goes through [`from_raw_handle`](Self::from_raw_handle), the
    /// fallible entry point the Python bindings use.
    pub fn from_communicator(comm: &impl Communicator) -> Self {
        Self::adopt(comm.duplicate()).unwrap_or_else(|err| panic!("{err}"))
    }

    /// Duplicate a raw `MPI_Comm` handle and wrap it.
    ///
    /// The entry point for a caller whose MPI is already up and whose
    /// communicator does not come from this crate's `rsmpi` — a Python host
    /// passing `MPI._addressof(comm)`, say. The handle is duplicated, so
    /// passing `MPI_COMM_WORLD` is fine here (unlike
    /// `SimpleCommunicator::from_raw`, which takes ownership and would free
    /// it); ownership of `handle` itself stays with the caller.
    ///
    /// **Collective** over the communicator `handle` names.
    ///
    /// # Safety
    ///
    /// `handle` must be a live `MPI_Comm` of the process's MPI library —
    /// bit-for-bit what C would pass, an `int` on MPICH-derived libraries and a
    /// pointer on Open MPI — and must be an intra-communicator. It stays valid
    /// for the caller after the call.
    ///
    /// # Errors
    ///
    /// [`MpiError::NotInitialized`] if MPI is not up,
    /// [`MpiError::NullCommunicator`] for a null handle,
    /// [`MpiError::DuplicateFailed`] if `MPI_Comm_dup` reports an error, and
    /// [`MpiError::SizeNotPowerOfTwo`] for a group that cannot be a
    /// partitioning.
    pub unsafe fn from_raw_handle(handle: usize) -> Result<Self, MpiError> {
        if !::mpi::environment::is_initialized() {
            return Err(MpiError::NotInitialized);
        }
        let raw = comm_from_usize(handle);
        if raw == ::mpi::ffi::RSMPI_COMM_NULL {
            return Err(MpiError::NullCommunicator);
        }
        let mut dup: ::mpi::ffi::MPI_Comm = ::mpi::ffi::RSMPI_COMM_NULL;
        // MPI_SUCCESS is 0 in every implementation (the standard fixes it), so
        // this needs no generated constant.
        let code = ::mpi::ffi::MPI_Comm_dup(raw, &mut dup);
        if code != 0 {
            return Err(MpiError::DuplicateFailed(code));
        }
        Self::adopt(SimpleCommunicator::from_raw(dup))
    }

    /// Take ownership of an already-duplicated communicator.
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
            // Open MPI 4.1 and 5.0 share the soname `libmpi.so.40`, so a
            // module swap between build and run links cleanly and then
            // misbehaves. Not fatal: the strings are free-form and a false
            // alarm must not stop a run.
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
            #[cfg(feature = "phase-timing")]
            timings: super::transport::ExchangeTimings::default(),
        })
    }

    /// Override the largest message the transport sends, in bytes.
    ///
    /// The chunking path is otherwise unreachable in a test (a part would have
    /// to exceed 1 GiB), so the multi-chunk net sets this to a few kilobytes.
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

    /// The next tag epoch, wrapping at [`EPOCHS`].
    fn next_epoch(&self) -> u32 {
        self.epoch.fetch_add(1, Ordering::Relaxed) % EPOCHS
    }

    /// One rank's tag for message `kind` in `epoch`.
    fn tag(kind: i32, epoch: u32) -> Rank {
        ((epoch % EPOCHS) as i32) << KIND_BITS | kind
    }

    /// Send this rank's gather contribution to the root: a framing header, then
    /// the parts in chunked messages, waited out before returning.
    ///
    /// The gather is single-phase — one message stream, nothing to overlap —
    /// so it opens and closes its own request scope rather than sharing the
    /// exchange's.
    fn send_gather(&self, parts: &[&[u8]], hdr_tag: Rank, part_tag: Rank) {
        let header = encode_header(self.rank, parts);
        let header: &[u8] = bytemuck::cast_slice(&header);
        let sends = 1 + parts
            .iter()
            .map(|p| chunk_count(p.len(), self.chunk))
            .sum::<usize>();
        let peer = self.comm.process_at_rank(ROOT as Rank);
        ::mpi::request::multiple_scope(sends, |scope, coll| {
            coll.add(peer.immediate_send_with_tag(scope, header, hdr_tag));
            for part in parts {
                for chunk in part.chunks(self.chunk) {
                    coll.add(peer.immediate_send_with_tag(scope, chunk, part_tag));
                }
            }
            let mut done = Vec::with_capacity(sends);
            coll.wait_all(&mut done);
        });
    }

    /// Post one partner's two-phase framing: the header, then the early parts,
    /// then the bulk parts **chunk-major**.
    ///
    /// Chunk-major is the whole point of the layout: MPI's non-overtaking
    /// guarantee for a `(source, tag, communicator)` triple means the receiver
    /// sees chunk `k`'s messages before chunk `k + 1`'s, so a receiver that
    /// waits chunk by chunk is waiting on a prefix of the transfer rather than
    /// on all of it.
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
        coll: &mut RequestCollection<'a, [u8]>,
        slot_of: &mut Vec<usize>,
    ) where
        Sc: Scope<'a> + Copy,
    {
        let peer = self.comm.process_at_rank(dst as Rank);
        let mut add = |req, slot_of: &mut Vec<usize>| {
            let i = coll.add(req);
            if slot_of.len() <= i {
                slot_of.resize(i + 1, SLOT_SEND);
            }
        };
        add(
            peer.immediate_send_with_tag(scope, header, tags.header),
            slot_of,
        );
        for part in early {
            for piece in part.chunks(self.chunk) {
                add(
                    peer.immediate_send_with_tag(scope, piece, tags.early),
                    slot_of,
                );
            }
        }
        for k in 0..chunks {
            for part in bulk {
                for piece in part[k].chunks(self.chunk) {
                    add(
                        peer.immediate_send_with_tag(scope, piece, tags.part),
                        slot_of,
                    );
                }
            }
        }
    }

    /// Blocking receive of one framing header from `src`, returning the part
    /// lengths it declares.
    ///
    /// # Panics
    ///
    /// If the header's wire version or source rank is not what this rank
    /// expects — a crossed message or a version-skewed peer, either of which
    /// would otherwise produce a garbage length vector and a wild allocation.
    fn recv_header(&self, src: usize, hdr_tag: Rank) -> Vec<usize> {
        let peer = self.comm.process_at_rank(src as Rank);
        let (bytes, _status) = peer.receive_vec_with_tag::<u8>(hdr_tag);
        decode_header(&bytes, src as u32, self.rank)
    }

    /// The root's half of the gather: receive every part of every rank in one
    /// `wait_all`.
    ///
    /// `buffers[i]` is `(source rank, the parts expected from it)`, each part
    /// already sized. The chunk order — rank, then part, then chunk, all
    /// ascending — mirrors [`send_gather`](Self::send_gather), which is what
    /// pairs the messages up under a shared tag.
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
        ::mpi::request::multiple_scope(n, |scope, coll| {
            for (src, chunk) in chunks {
                coll.add(
                    self.comm
                        .process_at_rank(src as Rank)
                        .immediate_receive_into_with_tag(scope, chunk, part_tag),
                );
            }
            let mut done = Vec::with_capacity(n);
            coll.wait_all(&mut done);
        });
    }
}

/// Rebuild an `MPI_Comm` from the `usize` a foreign caller passed.
///
/// `MPI_Comm` is not one type across implementations — Open MPI's bindings make
/// it a pointer newtype, MPICH's a plain `int` — so `as` cannot name its
/// primitive form. Both are scalars the C ABI passes by value and `usize` is at
/// least as wide as either, so the conversion is a width-checked
/// reinterpretation of the value instead.
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

/// Whether the run-time library version string is the one the build probed.
///
/// Both are free-form (`"Open MPI 5.0.6 (Language: C)"` from `mpicc
/// --showme:version`, a multi-line banner from `MPI_Get_library_version`), so
/// the test is "does the running banner mention the build's version number" —
/// the first dotted number in the build string.
fn version_matches(built: &str, running: &str) -> bool {
    let number = built
        .split_whitespace()
        .find(|tok| tok.contains('.') && tok.starts_with(|c: char| c.is_ascii_digit()));
    match number {
        Some(number) => running.contains(number),
        // Nothing to compare against (the build script could not probe): stay
        // quiet rather than warn on every run.
        None => true,
    }
}

/// The framing header for `parts`, as `u64` words: version and source, the part
/// count, then one byte length per part.
fn encode_header(src: u32, parts: &[&[u8]]) -> Vec<u64> {
    let mut words = Vec::with_capacity(parts.len() + 2);
    words.push((u64::from(src) << 32) | u64::from(WIRE_VERSION));
    words.push(parts.len() as u64);
    words.extend(parts.iter().map(|p| p.len() as u64));
    words
}

/// The inverse of [`encode_header`]: the declared part lengths.
///
/// # Panics
///
/// If the header is malformed, carries another wire version, or names a source
/// other than `expected_src`. All three mean the group is not running the same
/// code, and the alternative is a garbage length vector.
fn decode_header(bytes: &[u8], expected_src: u32, me: u32) -> Vec<usize> {
    assert!(
        bytes.len() >= 2 * size_of::<u64>() && bytes.len().is_multiple_of(size_of::<u64>()),
        "rank {me}: framing header from rank {expected_src} is {} bytes, expected a whole number \
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
        "rank {me}: rank {expected_src} speaks exchange wire version {version}, this rank speaks \
         {WIRE_VERSION} — the ranks are not running the same build",
    );
    let src = (words[0] >> 32) as u32;
    assert_eq!(
        src, expected_src,
        "rank {me}: a framing header tagged for this layer came from rank {src}, not the expected \
         rank {expected_src}",
    );
    let n_parts = words[1] as usize;
    assert_eq!(
        words.len(),
        n_parts + 2,
        "rank {me}: rank {expected_src} declared {n_parts} parts in a {}-word header",
        words.len(),
    );
    words[2..].iter().map(|&len| len as usize).collect()
}

/// Messages a part of `len` bytes is sent in at chunk size `chunk`.
///
/// Exactly `slice::chunks(chunk).count()` — the same rule both sides use — and
/// zero for an empty part.
fn chunk_count(len: usize, chunk: usize) -> usize {
    len.div_ceil(chunk)
}

/// Slot number a send request carries: not a chunk, and never waited on by
/// name — the closing [`ChunkPipeline::finish`] waits it out with the rest.
const SLOT_SEND: usize = usize::MAX;

/// Slot number an early receive carries: waited out by the driving thread
/// before the body starts, so the pipeline never counts it.
const SLOT_EARLY: usize = usize::MAX - 1;

/// Spin iterations a waiting Rayon worker burns before it starts yielding.
///
/// Only one thread may be inside MPI at a time
/// (`MPI_THREAD_SERIALIZED`), so the worker that wins the pipeline's mutex
/// drives the transfer for everybody and the rest wait on its published
/// counters. A chunk is milliseconds of copying, so this tier exists only for
/// the case where the chunk landed while this worker was on its way in.
const PIPELINE_SPINS: u32 = 256;

/// `yield_now` calls after the spin tier before a waiter starts sleeping.
const PIPELINE_YIELDS: u32 = 64;

/// How long a waiter sleeps per iteration once yielding has not helped.
///
/// A yield loop is not free to the rest of the machine: it takes its share of
/// the very CPUs the driving worker's copy is running on. Past this point the
/// chunk is not close, so paying a step of latency to stay off those cores is
/// the right trade — the same argument as the in-process collectives' three
/// wait tiers in [`transport`](super::transport).
const PIPELINE_SLEEP: std::time::Duration = std::time::Duration::from_micros(20);

/// The in-flight half of a two-phase exchange: the posted bulk receives, this
/// rank's sends, and which chunk each receive belongs to.
///
/// Handed to the coset loop as a [`ChunkWait`]. A task that reaches
/// `append_into` for a bucket in chunk `k` calls `wait_chunk(k)`; whichever
/// worker takes the mutex drives MPI until something completes, credits it to
/// its chunk and releases, and the waiters see the counter fall.
///
/// # One thread at a time is enough
///
/// The application requested `MPI_THREAD_SERIALIZED`, so exactly one thread may
/// be inside the library at once — which the mutex enforces. Nothing is lost by
/// serializing: MPI's progress engine moves every outstanding request of this
/// rank, receives and sends alike, so one thread inside `MPI_Waitsome` is
/// driving the whole layer's transfer.
///
/// # It cannot deadlock
///
/// Every send of the call is posted before the call's first receive, on every
/// rank, so a partner's rendezvous always has a matching posted receive to land
/// in. A rank blocked in `wait_chunk` is inside `MPI_Waitsome` over *all* of its
/// requests, so it services its partner's transfer as well as its own. A rank
/// whose coset loop never asks for a chunk still reaches
/// [`finish`](Self::finish), which waits everything out. And a coset task waits
/// for exactly one chunk (`ChunkMap` puts a whole coset in one), so no task
/// holds a wait on a chunk behind a wait on another.
struct ChunkPipeline<'a, 'c> {
    /// The request collection and its bookkeeping. `try_lock`ed, never blocked
    /// on: a worker that cannot get in must not queue behind the driver, which
    /// is itself blocked inside MPI.
    inner: Mutex<PipelineInner<'a, 'c>>,
    /// Outstanding receives per chunk. Sends are not counted.
    remaining: Vec<std::sync::atomic::AtomicUsize>,
}

struct PipelineInner<'a, 'c> {
    coll: &'c mut RequestCollection<'a, [u8]>,
    /// Chunk per request index, [`SLOT_SEND`] for a send or an early part.
    slot_of: Vec<usize>,
    /// Reused `wait_some` result buffer.
    done: Vec<(usize, ::mpi::point_to_point::Status, &'a [u8])>,
}

// SAFETY: `RequestCollection` is neither `Send` nor `Sync`, because `MPI_Request`
// is a raw handle. It is reachable here from any worker of the partition's Rayon
// pool, but only through `inner`'s mutex, so at most one thread is ever inside
// MPI — exactly what `MPI_THREAD_SERIALIZED` allows and what this module's docs
// require of the application. Same argument as the `unsafe impl`s on
// `MpiTransport`.
unsafe impl Send for ChunkPipeline<'_, '_> {}
unsafe impl Sync for ChunkPipeline<'_, '_> {}

impl<'a, 'c> ChunkPipeline<'a, 'c> {
    /// Wrap `coll`, whose request `i` belongs to chunk `slot_of[i]`, with
    /// `chunks` chunk counters.
    fn new(coll: &'c mut RequestCollection<'a, [u8]>, slot_of: Vec<usize>, chunks: usize) -> Self {
        let remaining: Vec<_> = (0..chunks)
            .map(|_| std::sync::atomic::AtomicUsize::new(0))
            .collect();
        for &slot in &slot_of {
            if slot < chunks {
                remaining[slot].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        Self {
            inner: Mutex::new(PipelineInner {
                coll,
                slot_of,
                done: Vec::new(),
            }),
            remaining,
        }
    }

    /// Drive MPI once from this thread if nobody else is inside it.
    ///
    /// `None` when another thread holds the pipeline, `Some(false)` when
    /// nothing is outstanding at all.
    fn try_drive(&self) -> Option<bool> {
        let mut inner = match self.inner.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return None,
        };
        let PipelineInner {
            coll,
            slot_of,
            done,
        } = &mut *inner;
        if coll.incomplete() == 0 {
            return Some(false);
        }
        coll.wait_some(done);
        for &(index, _, _) in done.iter() {
            let slot = slot_of[index];
            if slot < self.remaining.len() {
                self.remaining[slot].fetch_sub(1, std::sync::atomic::Ordering::Release);
            }
        }
        Some(true)
    }

    /// Block until every receive of chunk `k` has completed.
    fn wait_slot(&self, k: usize) {
        let mut spins: u32 = 0;
        loop {
            if self.remaining[k].load(std::sync::atomic::Ordering::Acquire) == 0 {
                return;
            }
            match self.try_drive() {
                // Nothing outstanding: this chunk arrived (or never existed).
                Some(false) => return,
                Some(true) => spins = 0,
                None => {
                    spins = spins.saturating_add(1);
                    if spins <= PIPELINE_SPINS {
                        std::hint::spin_loop();
                    } else if spins <= PIPELINE_SPINS + PIPELINE_YIELDS {
                        std::thread::yield_now();
                    } else {
                        std::thread::sleep(PIPELINE_SLEEP);
                    }
                }
            }
        }
    }

    /// Wait every outstanding request out: the receives the coset loop never
    /// reached, and this rank's own sends.
    fn finish(&mut self) {
        let inner = self.inner.get_mut().unwrap_or_else(|e| e.into_inner());
        inner.coll.wait_all(&mut inner.done);
        for slot in &self.remaining {
            slot.store(0, std::sync::atomic::Ordering::Release);
        }
    }
}

impl ChunkWait for ChunkPipeline<'_, '_> {
    fn wait_chunk(&self, k: usize) {
        self.wait_slot(k);
    }
}

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

    fn allreduce_sum_u64(&self, buf: &mut [u64]) {
        if self.size == 1 {
            return;
        }
        let mut scratch = self
            .scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        scratch.clear();
        scratch.extend_from_slice(buf);
        self.comm
            .all_reduce_into(&scratch[..], buf, SystemOperation::sum());
    }

    fn barrier(&self) {
        if self.size == 1 {
            return;
        }
        self.comm.barrier();
    }
}

impl Transport for MpiTransport {
    /// The two-phase exchange (see `ChunkPipeline`): the framing header, the
    /// block headers and the CSR offsets arrive before `body` starts, the
    /// columns arrive under it, chunk by chunk.
    ///
    /// ```text
    /// header : u64[2 + n_parts]         the length of every part, as before
    /// early  : block headers + CSR offsets, tag `early`   (waited here)
    /// bulk   : the columns, chunk-major, tag `part`       (waited in `body`)
    /// ```
    ///
    /// Nothing about the deadlock argument changes: every send is posted before
    /// the first receive, and a rank waiting on a chunk is inside
    /// `MPI_Waitsome` over all of its requests, so it drives its partner's
    /// rendezvous too.
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

        #[cfg(feature = "phase-timing")]
        let mut lap = std::time::Instant::now();

        // Everything the sends borrow must outlive the request scope.
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

        // SAFETY (the `UnsafeCell` below). Two aliases exist while `body` runs:
        // the `&mut` byte views MPI is filling, held by the request collection,
        // and the `&[Option<P>]` the body reads through. They never touch the
        // same bytes at the same time — the early parts are complete before the
        // body starts, and a chunk's columns are read only after
        // `wait_chunk` has seen its receives complete. The cell is what lets
        // the two coexist without the borrow checker having to see the
        // argument; it is the same situation as any posted `MPI_Irecv`, made
        // explicit.
        let recv_cell = std::cell::UnsafeCell::new(recv);

        let out =
            ::mpi::request::multiple_scope(partners.len() * (2 + 3 * chunks), |scope, coll| {
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
                        coll,
                        &mut slot_of,
                    );
                }
                #[cfg(feature = "phase-timing")]
                ExchangeTimings::lap(&self.timings.send_post_ns, &mut lap);

                // Every send is in flight, so no blocking receive below can
                // wait on a partner that has not spoken yet.
                let mut lens: Vec<Vec<usize>> = Vec::with_capacity(partners.len());
                for &src in &partners {
                    lens.push(self.recv_header(src, tags.header));
                }
                #[cfg(feature = "phase-timing")]
                ExchangeTimings::lap(&self.timings.hdr_wait_ns, &mut lap);

                // Phase 1: the payloads come from the caller's pool, are sized
                // from the declared part lengths, and hand out byte views of
                // their own columns — the receive *is* the decode.
                // SAFETY: no other alias exists yet; the request collection is
                // holding only this rank's sends.
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
                    for view in payload.early_recv_into(&lens) {
                        for piece in view.chunks_mut(self.chunk) {
                            let i = coll.add(
                                peer.immediate_receive_into_with_tag(scope, piece, tags.early),
                            );
                            if slot_of.len() <= i {
                                slot_of.resize(i + 1, SLOT_SEND);
                            }
                            slot_of[i] = SLOT_EARLY;
                            early_left += 1;
                        }
                    }
                }
                #[cfg(feature = "phase-timing")]
                ExchangeTimings::lap(&self.timings.recv_alloc_ns, &mut lap);

                // The early parts are kilobytes; wait them out here so the CSR
                // offsets are in place before anything reads a block.
                let mut done = Vec::new();
                while early_left > 0 {
                    coll.wait_some(&mut done);
                    // `wait_some` completes this rank's sends too; only the
                    // early receives are what this phase is waiting for.
                    for &(i, _, _) in &done {
                        if slot_of[i] == SLOT_EARLY {
                            early_left -= 1;
                        }
                    }
                }
                #[cfg(feature = "phase-timing")]
                ExchangeTimings::lap(&self.timings.hdr_wait_ns, &mut lap);

                // Nothing borrows the payloads now, so the arrived headers and
                // offsets can be checked against the shape the lengths implied.
                // SAFETY: as above; the collection holds only completed
                // requests and this rank's sends.
                let recv_mut = unsafe { &mut *recv_cell.get() };
                for payload in recv_mut.iter_mut().flatten() {
                    payload.finish_recv();
                }

                // Phase 2: the columns, cut at the same chunk boundaries the
                // sender used, posted chunk-major so they complete in order.
                for (q, slot) in recv_mut.iter_mut().enumerate() {
                    let Some(payload) = slot.as_mut() else {
                        continue;
                    };
                    let peer = self.comm.process_at_rank(q as Rank);
                    let mut parts: Vec<Vec<Option<&mut [u8]>>> = payload
                        .bulk_recv_into(map)
                        .into_iter()
                        .map(|pieces| pieces.into_iter().map(Some).collect())
                        .collect();
                    for k in 0..chunks {
                        for part in parts.iter_mut() {
                            let piece = part[k].take().expect("one view per chunk");
                            for piece in piece.chunks_mut(self.chunk) {
                                let i = coll.add(
                                    peer.immediate_receive_into_with_tag(scope, piece, tags.part),
                                );
                                if slot_of.len() <= i {
                                    slot_of.resize(i + 1, SLOT_SEND);
                                }
                                slot_of[i] = k;
                            }
                        }
                    }
                }
                #[cfg(feature = "phase-timing")]
                ExchangeTimings::lap(&self.timings.recv_alloc_ns, &mut lap);

                let mut pipeline = ChunkPipeline::new(coll, slot_of, chunks.max(1));
                // SAFETY: the body reads a chunk's rows only after
                // `wait_chunk` reports its receives complete (see the cell's
                // comment above).
                let out = body(unsafe { &*recv_cell.get() }, &pipeline);
                // The body is the layer's own work, not the exchange's; only
                // the transfer left over after it counts as a data wait.
                #[cfg(feature = "phase-timing")]
                {
                    lap = std::time::Instant::now();
                }
                pipeline.finish();
                #[cfg(feature = "phase-timing")]
                ExchangeTimings::lap(&self.timings.data_wait_ns, &mut lap);
                out
            });

        // This rank's own blocks are off the wire now; back into the pool they
        // go, so the next layer's export reuses them.
        spare.extend(send.into_iter().flatten());
        (recv_cell.into_inner(), out)
    }

    #[cfg(feature = "phase-timing")]
    fn drain_timings(&self, stats: &mut crate::engine::stats::PhaseStats) {
        self.timings.drain_into(stats);
    }

    /// Root-centric, not an exchange: the default body in
    /// [`Transport`](super::transport::Transport) infers its partner set from
    /// the `Some` positions, which a gather does not populate symmetrically.
    fn gather_to_root(&self, parts: Vec<&[u8]>) -> Option<Vec<Vec<Vec<u8>>>> {
        let n = self.size as usize;
        let me = self.rank as usize;
        if n == 1 {
            return Some(vec![parts.iter().map(|p| p.to_vec()).collect()]);
        }

        let epoch = self.next_epoch();
        let (hdr_tag, part_tag) = (
            Self::tag(KIND_GATHER_HEADER, epoch),
            Self::tag(KIND_GATHER_PART, epoch),
        );

        if me != ROOT {
            self.send_gather(&parts, hdr_tag, part_tag);
            return None;
        }

        let mut buffers: Vec<Vec<Vec<u8>>> = (1..n)
            .map(|src| {
                self.recv_header(src, hdr_tag)
                    .into_iter()
                    .map(|len| vec![0u8; len])
                    .collect()
            })
            .collect();
        // One gather per run, so the copy-free receive the exchange uses would
        // buy nothing here: these are plain byte parts either way.
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

/// The placement a distributed rank uses when the caller does not say
/// otherwise: **one** partition over whatever CPUs the launcher left in the
/// process's affinity mask, with memory bound to its node.
///
/// Under `mpirun --map-by ppr:1:numa --bind-to numa` (or `srun
/// --cpu-bind=ldoms`) that mask *is* one NUMA domain, so `Auto` resolves to a
/// single slot covering exactly it — the process placement does the work the
/// in-process engine would have done with explicit CPU lists. Under no binding
/// at all it is one slot over the whole node, which is correct but unplaced.
pub fn default_config() -> PartitionConfig {
    PartitionConfig {
        placement: Placement::Auto {
            max_partitions: Some(1),
        },
        bind_memory: true,
        partition_row_seed: None,
    }
}

/// Propagate `sum` through `circuit` with one partition per rank of `comm`, and
/// gather the result on rank 0.
///
/// **Collective**: every rank of `comm` must call this, with the *same*
/// replicated `sum`, the same circuit, direction and options. Rank 0 gets
/// `Some(gathered)`; every other rank gets `None`.
///
/// One-shot convenience — it duplicates the communicator, builds a pinned pool,
/// scatters, runs and gathers. A driver stepping an observable through many
/// circuits should hold an [`MpiSum`] instead, so the communicator, the pool,
/// the split and the scratch survive between steps. For a placement other than
/// [`default_config`]'s, build the [`MpiSum`] directly:
///
/// ```no_run
/// # use paulistrings::mpi::{default_config, MpiSum, MpiTransport, rsmpi};
/// # use paulistrings::{Circuit, Direction, PartitionedTruncation, TruncationPolicy, PauliSum};
/// # struct KeepAll;
/// # impl<const W: usize> TruncationPolicy<W> for KeepAll { fn finalizes_layer(&self) -> bool { false } }
/// # impl<const W: usize> PartitionedTruncation<W> for KeepAll {}
/// # fn go(circuit: &Circuit<1>, sum: PauliSum<1>) {
/// let (universe, _) =
///     rsmpi::initialize_with_threading(rsmpi::Threading::Serialized).expect("MPI initializes");
/// let transport = MpiTransport::from_communicator(&universe.world());
/// let mut split: MpiSum<1> =
///     MpiSum::scatter(sum, transport, &default_config()).expect("topology resolves");
/// for _ in 0..10 {
///     split.propagate(circuit, &KeepAll, Direction::Heisenberg);
/// }
/// if let Some(out) = split.gather() {
///     println!("{} terms", out.len());
/// }
/// # }
/// ```
///
/// # Panics
///
/// If the group size is not a power of two, if the placement cannot be
/// resolved, or for any of the reasons
/// [`DistributedSum::propagate_with_options`] panics — including the ranks
/// disagreeing about the run.
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
    split.propagate_with_options(circuit, policy, direction, options);
    split.gather()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tag layout: kind in the low four bits, epoch above, and never past
    /// the `MPI_TAG_UB` the standard guarantees.
    #[test]
    fn tags_pack_kind_and_epoch_below_the_guaranteed_tag_bound() {
        assert_eq!(MpiTransport::tag(KIND_EXCHANGE_HEADER, 0), 0);
        assert_eq!(MpiTransport::tag(KIND_EXCHANGE_PART, 0), 1);
        assert_eq!(MpiTransport::tag(KIND_GATHER_PART, 0), 3);
        assert_eq!(MpiTransport::tag(KIND_EXCHANGE_HEADER, 1), 16);
        assert_eq!(MpiTransport::tag(KIND_EXCHANGE_PART, 1), 17);
        // The last epoch before the wrap sits exactly on 32767.
        assert_eq!(MpiTransport::tag(KIND_GATHER_PART + 12, EPOCHS - 1), 32767);
        for epoch in 0..EPOCHS {
            for kind in [
                KIND_EXCHANGE_HEADER,
                KIND_EXCHANGE_PART,
                KIND_GATHER_HEADER,
                KIND_GATHER_PART,
            ] {
                let tag = MpiTransport::tag(kind, epoch);
                assert!((0..=32767).contains(&tag), "epoch {epoch} kind {kind}");
                assert_eq!(tag & 0xf, kind);
            }
        }
        // Epochs wrap rather than overflowing the tag space.
        assert_eq!(
            MpiTransport::tag(KIND_EXCHANGE_PART, EPOCHS),
            MpiTransport::tag(KIND_EXCHANGE_PART, 0),
        );
    }

    #[test]
    fn header_round_trips_through_its_byte_form() {
        let a = [1u8, 2, 3];
        let b: [u8; 0] = [];
        let c = [7u8; 40];
        let parts: Vec<&[u8]> = vec![&a, &b, &c];

        let words = encode_header(3, &parts);
        assert_eq!(words[0] & 0xffff_ffff, u64::from(WIRE_VERSION));
        assert_eq!(words[0] >> 32, 3);
        assert_eq!(words[1], 3);
        assert_eq!(&words[2..], &[3, 0, 40]);

        let bytes: &[u8] = bytemuck::cast_slice(&words);
        assert_eq!(decode_header(bytes, 3, 0), vec![3, 0, 40]);
    }

    #[test]
    fn an_empty_payload_encodes_as_a_two_word_header() {
        let words = encode_header(0, &[]);
        assert_eq!(words.len(), 2);
        let bytes: &[u8] = bytemuck::cast_slice(&words);
        assert_eq!(decode_header(bytes, 0, 0), Vec::<usize>::new());
    }

    #[test]
    #[should_panic(expected = "wire version")]
    fn a_header_from_another_wire_version_is_rejected() {
        let mut words = encode_header(1, &[]);
        words[0] = (words[0] & !0xffff_ffff) | 99;
        let bytes: &[u8] = bytemuck::cast_slice(&words);
        let _ = decode_header(bytes, 1, 0);
    }

    #[test]
    #[should_panic(expected = "not the expected rank")]
    fn a_header_from_the_wrong_rank_is_rejected() {
        let words = encode_header(2, &[]);
        let bytes: &[u8] = bytemuck::cast_slice(&words);
        let _ = decode_header(bytes, 5, 0);
    }

    #[test]
    #[should_panic(expected = "declared")]
    fn a_header_whose_part_count_disagrees_with_its_length_is_rejected() {
        let a = [1u8, 2];
        let mut words = encode_header(0, &[&a]);
        words[1] = 4;
        let bytes: &[u8] = bytemuck::cast_slice(&words);
        let _ = decode_header(bytes, 0, 0);
    }

    #[test]
    #[should_panic(expected = "at least two")]
    fn a_truncated_header_is_rejected() {
        let _ = decode_header(&[0u8; 8], 0, 0);
    }

    /// The chunk count must agree with `slice::chunks` at every boundary — that
    /// agreement is what makes the sender's and receiver's message counts equal
    /// with no negotiation.
    #[test]
    fn chunk_counts_agree_with_slice_chunks_at_the_boundaries() {
        let buf = vec![0u8; 4096];
        for chunk in [1usize, 2, 3, 1024, 4095, 4096, 4097] {
            for len in [
                0,
                1,
                chunk.saturating_sub(1),
                chunk,
                chunk + 1,
                2 * chunk,
                2 * chunk + 1,
            ] {
                if len > buf.len() {
                    continue;
                }
                assert_eq!(
                    chunk_count(len, chunk),
                    buf[..len].chunks(chunk).count(),
                    "len {len} chunk {chunk}",
                );
            }
        }
        assert_eq!(chunk_count(0, 1 << 30), 0);
        assert_eq!(chunk_count(1 << 30, 1 << 30), 1);
        assert_eq!(chunk_count((1 << 30) + 1, 1 << 30), 2);
    }

    #[test]
    fn version_matching_finds_the_build_version_in_the_runtime_banner() {
        assert!(version_matches(
            "Open MPI 5.0.6 (Language: C)",
            "Open MPI v5.0.6, package: Open MPI build, ident: 5.0.6",
        ));
        assert!(!version_matches(
            "Open MPI 5.0.6 (Language: C)",
            "Open MPI v4.1.6, package: Open MPI build, ident: 4.1.6",
        ));
        // No number to compare: stay quiet.
        assert!(version_matches("unknown", "Open MPI v5.0.6"));
    }
}
