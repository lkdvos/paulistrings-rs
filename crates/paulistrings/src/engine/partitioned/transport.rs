//! The [`Collectives`] and [`Transport`] traits, the [`Payload`] wire seam, the destination-coset [`ChunkMap`], and the in-process transport (ARCHITECTURE.md §Partitioning).

use crate::engine::coset::Gf2Span;

mod exchange_block;
mod in_process;

#[cfg(feature = "cuda")]
pub(crate) use exchange_block::{chunk_rows_of, BlockHeader};
pub(crate) use exchange_block::{ExchangeBlock, PartnerPayload};
pub use in_process::InProcessTransport;

/// Seals [`Collectives`] and, through it, [`Transport`].
pub(crate) mod sealed {
    pub trait Sealed {}
}

/// The destination-coset order a layer's exchange blocks are laid out in, and the chunks the bulk transfer is cut into (ARCHITECTURE.md §Partitioning).
///
/// Chunk edges always fall on coset boundaries, so a coset task waits on exactly one chunk.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChunkMap {
    /// `perm[β]` is bucket `β`'s destination position; empty when the permutation is the identity (`r == 0`).
    perm: Vec<u32>,
    /// `inv[p]` is the bucket at position `p`; empty with `perm`.
    inv: Vec<u32>,
    positions: u32,
    /// `log2` of the coset size: coset `c` owns positions `c << r .. (c + 1) << r`.
    r: u32,
    cosets: u32,
    /// Clamped to `1..=cosets`.
    chunks: u32,
}

impl ChunkMap {
    /// Re-aim the map at `span` over `num_buckets` buckets in at most `chunks` pieces, keeping the allocations; panics unless `num_buckets` is a power of two and `chunks > 0`.
    pub(crate) fn rebuild(&mut self, span: &Gf2Span, num_buckets: usize, chunks: usize) {
        assert!(
            num_buckets.is_power_of_two(),
            "ChunkMap: {num_buckets} buckets is not a power of two",
        );
        assert!(chunks > 0, "ChunkMap: a layer needs at least one chunk");
        let positions = num_buckets as u32;
        let r = span.r() as u32;
        debug_assert!(
            (1u32 << r) <= positions,
            "ChunkMap: a coset of {} buckets does not fit in {positions}",
            1u32 << r,
        );
        self.positions = positions;
        self.r = r;
        self.cosets = positions >> r;
        self.chunks = (chunks as u32).min(self.cosets).max(1);
        self.perm.clear();
        self.inv.clear();
        if r == 0 {
            return;
        }
        self.perm.reserve(num_buckets);
        self.inv.resize(num_buckets, 0);
        for beta in 0..positions {
            let p = span.perm_index(beta);
            self.perm.push(p);
            self.inv[p as usize] = beta;
        }
    }

    /// Bucket `beta`'s destination position.
    #[inline]
    pub(crate) fn position_of(&self, beta: u32) -> u32 {
        if self.perm.is_empty() {
            beta
        } else {
            self.perm[beta as usize]
        }
    }

    /// The bucket at destination position `p`, the inverse of `position_of`.
    #[inline]
    pub(crate) fn bucket_at(&self, p: u32) -> u32 {
        if self.inv.is_empty() {
            p
        } else {
            self.inv[p as usize]
        }
    }

    /// Positions, i.e. buckets.
    pub(crate) fn positions(&self) -> usize {
        self.positions as usize
    }

    pub(crate) fn chunks(&self) -> usize {
        self.chunks as usize
    }

    /// The first position of chunk `k`, a multiple of the coset size; `bound(chunks())` is the position count.
    pub(crate) fn bound(&self, k: usize) -> u32 {
        assert!(k <= self.chunks(), "ChunkMap: chunk {k} is out of range");
        let coset = (k as u64 * u64::from(self.cosets)).div_ceil(u64::from(self.chunks)) as u32;
        coset << self.r
    }

    /// The `k` with `bound(k) <= p < bound(k + 1)`.
    #[inline]
    pub(crate) fn chunk_of_position(&self, p: u32) -> usize {
        let coset = u64::from(p >> self.r);
        ((coset * u64::from(self.chunks)) / u64::from(self.cosets)) as usize
    }
}

/// Something a [`Transport`] can move between partitions as zero-copy byte views of its own columns.
///
/// The in-process transport moves the typed value and calls none of these methods.
pub trait Payload: Default + Send + 'static {
    /// Borrowed byte views of this payload's columns, in wire order.
    fn byte_parts(&self) -> Vec<&[u8]>;

    /// Reshape for an incoming message of part lengths `lens` (growing, never shrinking, the storage) and hand out one mutable byte view per part, in `byte_parts` order.
    ///
    /// Panics if `lens` is not a shape this payload can take.
    fn recv_into(&mut self, lens: &[usize]) -> Vec<&mut [u8]>;

    /// Check, once the bytes have arrived and before the engine reads them, what `recv_into` could not; panics on an inconsistent payload.
    fn finish_recv(&mut self);

    /// The parts a two-phase transport must deliver before the body runs, in `byte_parts` order; every part by default.
    fn early_parts(&self) -> Vec<&[u8]> {
        self.byte_parts()
    }

    /// The parts after `early_parts`, each cut at `map`'s chunks: `bulk_parts()[i][k]` is part `i`'s slice for chunk `k`; none by default.
    ///
    /// Must cut exactly as `bulk_recv_into` does on the receiver, or the two sides post different message sizes.
    fn bulk_parts(&self, map: &ChunkMap) -> Vec<Vec<&[u8]>> {
        let _ = map;
        Vec::new()
    }

    /// `recv_into` sizing every part from `lens` but handing out only the `early_parts` views.
    fn early_recv_into(&mut self, lens: &[usize]) -> Vec<&mut [u8]> {
        self.recv_into(lens)
    }

    /// The receive-side mirror of `bulk_parts`, called once the early parts have arrived.
    fn bulk_recv_into(&mut self, map: &ChunkMap) -> Vec<Vec<&mut [u8]>> {
        let _ = map;
        Vec::new()
    }
}

/// What a coset task waits on before it reads a received row of its chunk.
///
/// Called from every Rayon worker at once, so an MPI implementation must serialize itself (`MPI_THREAD_SERIALIZED`).
pub trait ChunkWait: Sync {
    /// Block until every row of chunk `k` has arrived; safe to call concurrently, repeatedly and out of order.
    fn wait_chunk(&self, k: usize);
}

/// The [`ChunkWait`] of a transport whose exchange already completed.
pub(super) struct AlreadyHere;

impl ChunkWait for AlreadyHere {
    fn wait_chunk(&self, _k: usize) {}
}

/// The group operations a partition needs besides the exchange: its rank, the reductions, and a barrier.
///
/// Every partition must issue the identical sequence of collective and transport calls, and every reduction must return the identical value on every partition.
/// Sealed: implemented only by the in-process transport and `mpi::MpiTransport`.
pub trait Collectives: sealed::Sealed + Send + Sync {
    /// This partition's index in the group, `0 <= rank < size`.
    fn rank(&self) -> u32;
    /// Number of partitions in the group.
    fn size(&self) -> u32;
    /// Maximum of `v` over the group.
    fn allreduce_max_u8(&self, v: u8) -> u8;
    /// Element-wise wrapping sum of `buffer` over the group, in place; every partition passes the same length.
    fn allreduce_sum_u64(&self, buffer: &mut [u64]);
    /// Element-wise sum of `buffer` over the group, in place, bitwise identical on every partition.
    ///
    /// The combination order is the implementation's, so the result may differ by rounding from a serial sum; a slot only one partition fills is exact.
    fn allreduce_sum_f64(&self, buffer: &mut [f64]);
    /// Block until every partition has arrived.
    fn barrier(&self);

    /// Panic unless every partition passed the same `fingerprint`; one collective.
    fn check_consistency(&self, fingerprint: u64) {
        // Per bit, how many partitions set it: an agreeing group answers 0 or `size` for every bit.
        let mut counts = [0u64; 64];
        for (i, count) in counts.iter_mut().enumerate() {
            *count = (fingerprint >> i) & 1;
        }
        self.allreduce_sum_u64(&mut counts);
        let size = u64::from(self.size());
        for (bit, &count) in counts.iter().enumerate() {
            assert!(
                count == 0 || count == size,
                "the partitions disagree about the run: partition {} offered fingerprint \
                 {fingerprint:#018x}, and {count} of {size} partitions set bit {bit} of theirs. \
                 Every partition must be driven through the same circuit, in the same direction, \
                 under the same policy and options — a group that is not stays in step only by \
                 luck.",
                self.rank(),
            );
        }
    }
}

/// The per-layer all-to-all exchange between the partitions of a group.
///
/// Sealed through [`Collectives`]; not object-safe, so the layer loop monomorphizes over it.
pub trait Transport: Collectives {
    /// Send `send[q]` to partition `q`, run `body` on what the partners sent here, and return the received payloads with `body`'s result.
    ///
    /// Two-phase: the `early_parts` have arrived before `body` starts, and `body` calls `ChunkWait::wait_chunk` before reading a chunk's bulk rows.
    /// `send.len()` must be the group size with `None` in the own slot and for any partner with nothing to send; every partition passes the same `map`.
    /// `spare` is a payload pool in unspecified state, possibly empty: the transport takes receive payloads from it and returns finished send payloads to it.
    fn exchange_layer<P, F, R>(
        &self,
        send: Vec<Option<P>>,
        spare: &mut Vec<P>,
        map: &ChunkMap,
        body: F,
    ) -> (Vec<Option<P>>, R)
    where
        P: Payload,
        F: FnOnce(&[Option<P>], &dyn ChunkWait) -> R;

    /// The blocking [`exchange_layer`](Self::exchange_layer) under an empty `ChunkMap`, for payloads whose every part is early.
    fn exchange<P: Payload>(&self, send: Vec<Option<P>>, spare: &mut Vec<P>) -> Vec<Option<P>> {
        self.exchange_layer(send, spare, &ChunkMap::default(), |_, _| ())
            .0
    }

    /// Collect every partition's `parts` on rank 0: `Some(v)` there with `v[q]` partition `q`'s parts, `None` elsewhere; one collective.
    ///
    /// A transport that infers its partner set from the `Some` positions, as the MPI one does, must override this asymmetric call.
    fn gather_to_root(&self, parts: Vec<&[u8]>) -> Option<Vec<Vec<Vec<u8>>>> {
        let n = self.size() as usize;
        let this_rank = self.rank() as usize;
        let mine = ByteParts(parts.iter().map(|p| p.to_vec()).collect());

        if this_rank != ROOT {
            let mut send: Vec<Option<ByteParts>> = (0..n).map(|_| None).collect();
            send[ROOT] = Some(mine);
            self.exchange(send, &mut Vec::new());
            return None;
        }

        let mut recv = self.exchange(
            (0..n).map(|_| None).collect::<Vec<Option<ByteParts>>>(),
            &mut Vec::new(),
        );
        let mut mine = Some(mine);
        Some(
            (0..n)
                .map(|q| {
                    let got = if q == ROOT {
                        mine.take()
                    } else {
                        recv[q].take()
                    };
                    got.unwrap_or_else(|| {
                        panic!(
                            "gather_to_root: partition {q} sent nothing (it must send its \
                                parts, empty or not)"
                        )
                    })
                    .0
                })
                .collect(),
        )
    }
}

/// The partition every gather lands on.
pub(super) const ROOT: usize = 0;

/// A [`Payload`] that is already bytes, the default gather's wire form.
#[derive(Default)]
struct ByteParts(Vec<Vec<u8>>);

impl Payload for ByteParts {
    fn byte_parts(&self) -> Vec<&[u8]> {
        self.0.iter().map(Vec::as_slice).collect()
    }

    fn recv_into(&mut self, lens: &[usize]) -> Vec<&mut [u8]> {
        self.0.resize_with(lens.len(), Vec::new);
        for (part, &len) in self.0.iter_mut().zip(lens) {
            part.clear();
            part.resize(len, 0);
        }
        self.0.iter_mut().map(Vec::as_mut_slice).collect()
    }

    fn finish_recv(&mut self) {}
}

#[cfg(test)]
mod tests;
