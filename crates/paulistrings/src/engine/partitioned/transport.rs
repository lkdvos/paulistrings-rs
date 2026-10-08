//! Transport traits and the exchange wire format.
//!
//! A partition talks to the rest of the group only through [`Collectives`] (rank/size, the reductions, the barrier) and [`Transport`] (the per-layer all-to-all [`Transport::exchange_layer`]); the sum is split across partitions by designated partition rows of the GF(2) hash (ARCHITECTURE.md §Bucketing gives the hash, §Partitioning the split), and the wire unit is one [`ExchangeBlock`] per remote delta, laid out in the receiver's destination-coset order (ARCHITECTURE.md §Engine) so a coset's rows are contiguous and can be sent in chunks while the rest is still in flight.
//!
//! **Every partition issues the identical sequence of transport calls, in the same order, on every layer** — a call's `n`-th message pairs with the partner's `n`-th positionally, so empty is sent as `None`, never silence, and every rank stamps its calls with a generation counter so a violation of this invariant panics naming both partitions instead of hanging.
//!
//! Payloads are pooled on both sides ([`Payload::byte_parts`]/[`Payload::recv_into`] move each column as a zero-copy byte view), so a steady-state layer neither allocates nor re-zeroes its megabytes.

use crate::engine::coset::Gf2Span;

mod exchange_block;
mod in_process;

#[cfg(feature = "cuda")]
pub(crate) use exchange_block::{chunk_rows_of, BlockHeader};
pub(crate) use exchange_block::{ExchangeBlock, PartnerPayload};
pub use in_process::InProcessTransport;

/// The private supertrait that seals [`Collectives`] and, through it, [`Transport`].
pub(crate) mod sealed {
    /// Implemented only inside this crate.
    pub trait Sealed {}
}

/// The order a layer's exchange blocks are laid out in, and the chunks the
/// bulk transfer is cut into.
///
/// # Destination-coset order
///
/// The receiver's unit of work is a **coset** of `span(h(D_local))`, and `Gf2Span::perm_index` renumbers the bucket index so that a coset occupies a contiguous run of *positions* (ARCHITECTURE.md §Engine).
/// Both sides of an exchange can compute that renumbering — the local bucket deltas are a function of the channel and the hash, not of the rank, and the bucket count is agreed by collective before the layer — so the **sender** lays its CSR block out in the receiver's position order instead of its own source-bucket order:
///
/// ```text
/// segment p  holds the rows generated from source bucket  bucket_at(p) ^ bd
/// ```
///
/// and the receiver filling output bucket `β′` reads `segment(position_of(β′))` with no arithmetic of its own.
/// Reordering costs the sender nothing: it is a permutation of the count and offset arrays, applied before the fill pass walks them.
///
/// # Chunks
///
/// Because a coset is contiguous in position space, a contiguous *range* of positions is a whole number of cosets, so the block splits into `chunks` pieces that the receiver can consume one at a time, in order, as they arrive.
/// Chunk `k` covers positions `bound(k)..bound(k + 1)`, always on a coset boundary.
///
/// A chunk count above the coset count would produce empty chunks, so it is clamped; `chunks == 1` is the un-pipelined layout.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChunkMap {
    /// `perm[β]` is bucket `β`'s destination position.
    /// Empty when the permutation is the identity (`r == 0`), which is what a layer whose local deltas are all zero — a rotation with a remote generator, the common case — produces.
    perm: Vec<u32>,
    /// `inv[p]` is the bucket at position `p`. Empty with [`Self::perm`].
    inv: Vec<u32>,
    /// Positions, i.e. buckets.
    positions: u32,
    /// `log2` of the coset size: coset `c` owns positions `c << r .. (c + 1) << r`.
    r: u32,
    /// Cosets, `positions >> r`.
    cosets: u32,
    /// Chunks the bulk transfer is cut into, `1..=cosets`.
    chunks: u32,
}

impl ChunkMap {
    /// Re-aim the map at `span` over `num_buckets` buckets, cut into at most `chunks` pieces, **keeping the permutation buffers' allocations**.
    ///
    /// # Panics
    ///
    /// If `num_buckets` is zero or not a power of two, or if `chunks` is zero.
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
            // `perm_index` compresses over every bit, so it is the identity; the empty vectors say so and save both the build and the indirection.
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
    pub fn position_of(&self, beta: u32) -> u32 {
        if self.perm.is_empty() {
            beta
        } else {
            self.perm[beta as usize]
        }
    }

    /// The bucket at destination position `p` — the inverse of [`position_of`](Self::position_of).
    #[inline]
    pub fn bucket_at(&self, p: u32) -> u32 {
        if self.inv.is_empty() {
            p
        } else {
            self.inv[p as usize]
        }
    }

    /// Positions, i.e. buckets.
    pub fn positions(&self) -> usize {
        self.positions as usize
    }

    /// Chunks the bulk transfer is cut into.
    pub fn chunks(&self) -> usize {
        self.chunks as usize
    }

    /// The first position of chunk `k`; `bound(chunks())` is the position count, so chunk `k` is `bound(k)..bound(k + 1)`.
    ///
    /// Always a multiple of the coset size, so no coset straddles two chunks.
    ///
    /// # Panics
    ///
    /// If `k > chunks()`.
    pub fn bound(&self, k: usize) -> u32 {
        assert!(k <= self.chunks(), "ChunkMap: chunk {k} is out of range");
        let c = (k as u64 * u64::from(self.cosets)).div_ceil(u64::from(self.chunks)) as u32;
        c << self.r
    }

    /// The chunk position `p` belongs to.
    ///
    /// The inverse of [`bound`](Self::bound): `bound(k) <= p < bound(k + 1)`.
    #[inline]
    pub(crate) fn chunk_of_position(&self, p: u32) -> usize {
        let c = u64::from(p >> self.r);
        ((c * u64::from(self.chunks)) / u64::from(self.cosets)) as usize
    }
}

/// Something a [`Transport`] can move between partitions as bytes.
///
/// [`byte_parts`](Self::byte_parts) is zero-copy — borrowed views of the payload's own columns, one part per column — so a sending transport never packs or allocates.
/// [`recv_into`](Self::recv_into) + [`finish_recv`](Self::finish_recv) is its mirror: the payload sizes its own typed columns from the part lengths the wire declared and hands out mutable byte views *of those columns*, so a transport receives straight into the storage the engine will read, with no decode pass and no second buffer.
/// That is why a payload is `Default` — a transport pools them across layers, so the steady state allocates nothing.
///
/// The in-process transport moves the typed value and calls none of them.
pub trait Payload: Default + Send + 'static {
    /// Borrowed byte views of this payload's columns, in wire order.
    fn byte_parts(&self) -> Vec<&[u8]>;

    /// Reshape this payload for an incoming message whose parts have byte lengths `lens`, and hand out one mutable byte view per part to receive into — the same parts, in the same order, [`byte_parts`](Self::byte_parts) would produce for it.
    ///
    /// Reuses whatever storage the payload already holds and only ever grows it.
    /// The views are of the payload's own typed columns, so filling them is the decode.
    ///
    /// # Panics
    ///
    /// If `lens` is not a shape this payload can take (a partial block, a part length that is not a whole number of elements).
    fn recv_into(&mut self, lens: &[usize]) -> Vec<&mut [u8]>;

    /// Check what [`recv_into`](Self::recv_into) could not: the invariants that only hold once the bytes have arrived.
    ///
    /// Called by the transport after the receives complete, before the payload is handed to the engine.
    ///
    /// # Panics
    ///
    /// If the received payload is inconsistent — a block encoded at another width, offsets that disagree with the row count.
    fn finish_recv(&mut self);

    /// The parts the engine reads **before** it reads a single row, so a two-phase transport must have them in hand before it hands the payload over: for the layer's exchange blocks that is the block headers and the CSR offsets, a few tens of kilobytes against a layer's hundreds of megabytes, and all the engine needs to size a gather run (`ExtraRows::count`).
    ///
    /// A prefix of [`byte_parts`](Self::byte_parts) in the same order — the framing still declares every part's length, so the receiver can size the bulk columns from the header alone.
    /// The default is *every* part, which is what a payload with no interesting internal structure wants: a transport then behaves exactly as a blocking one.
    fn early_parts(&self) -> Vec<&[u8]> {
        self.byte_parts()
    }

    /// The rest of [`byte_parts`](Self::byte_parts), each split into `map`'s chunks: `bulk_parts()[i][k]` is part `i`'s slice for chunk `k`.
    ///
    /// A chunk is a contiguous range of the receiver's destination positions ([`ChunkMap`]), so a column's chunk is the byte range of the rows in those positions, which the CSR offsets give exactly.
    /// Both sides compute it from the same offsets, so the sender's pieces and the receiver's posted receives line up with no further exchange.
    ///
    /// The default is no bulk parts at all, the counterpart of [`early_parts`](Self::early_parts)'s default.
    fn bulk_parts(&self, map: &ChunkMap) -> Vec<Vec<&[u8]>> {
        let _ = map;
        Vec::new()
    }

    /// [`recv_into`](Self::recv_into), returning only the [`early_parts`](Self::early_parts) views.
    ///
    /// It still sizes *every* column from `lens` — that is what lets [`bulk_recv_into`](Self::bulk_recv_into) slice them afterwards — but the bulk views are not handed out yet, so the payload can be inspected ([`finish_recv`](Self::finish_recv)) between the two phases with no outstanding borrow.
    fn early_recv_into(&mut self, lens: &[usize]) -> Vec<&mut [u8]> {
        self.recv_into(lens)
    }

    /// The receive-side mirror of [`bulk_parts`](Self::bulk_parts): the same parts, in the same order, split into the same chunks.
    ///
    /// Called after the early parts have arrived, so the CSR offsets the split reads are the ones the sender used.
    fn bulk_recv_into(&mut self, map: &ChunkMap) -> Vec<Vec<&mut [u8]>> {
        let _ = map;
        Vec::new()
    }
}

/// What a coset task waits on before it reads a received row.
///
/// The partitioned layer hands one of these to its `RecvRows`, which calls [`wait_chunk`](Self::wait_chunk) at the top of `append_into` with the chunk its output bucket belongs to.
/// A blocking transport's implementation is empty: everything arrived before the body ever ran.
///
/// `Sync` because the coset loop calls it from every Rayon worker at once.
/// An implementation that talks to MPI must therefore serialize itself — one thread in the library at a time is exactly `MPI_THREAD_SERIALIZED`.
pub trait ChunkWait: Sync {
    /// Block until every row of chunk `k` has arrived.
    ///
    /// Must be safe to call concurrently, repeatedly, and out of order — a coset task knows only its own chunk, and Rayon decides who runs when.
    fn wait_chunk(&self, k: usize);
}

/// The [`ChunkWait`] of a transport whose exchange already completed.
pub(crate) struct AlreadyHere;

impl ChunkWait for AlreadyHere {
    fn wait_chunk(&self, _k: usize) {}
}

/// The collective operations a partition needs outside the exchange itself: its identity in the group, the reductions a layer's truncation and bookkeeping need, and a barrier.
///
/// Every reduction must return **the identical value on every partition** — callers use them to agree on a global decision (a truncation threshold, a term-count total), and a partition that computed a different answer would diverge silently.
/// The integer reductions are exact and order-independent (a maximum, and a wrapping integer sum), so an implementation is free to combine in arrival order; [`allreduce_sum_f64`](Self::allreduce_sum_f64) is not, and must fix one combination order for the whole group.
///
/// Every method obeys the collective-order invariant in the module docs: all partitions call them in the same order, the same number of times.
pub trait Collectives: sealed::Sealed + Send + Sync {
    /// This partition's index in the group, `0 <= rank < size`.
    fn rank(&self) -> u32;
    /// Number of partitions in the group.
    fn size(&self) -> u32;
    /// Maximum of `v` over the group. Same value on every partition.
    fn allreduce_max_u8(&self, v: u8) -> u8;
    /// Element-wise sum of `buf` over the group, in place. Every partition passes the same length and gets the same values back.
    /// Sums wrap rather than panic on overflow, so debug and release agree.
    fn allreduce_sum_u64(&self, buf: &mut [u64]);
    /// Element-wise sum of `buf` over the group, in place. Every partition passes the same length and gets **bitwise the same** values back.
    /// The combination order is the implementation's, so the result may differ by rounding from a serial sum or from another group size; a slot only one partition fills is reduced exactly.
    fn allreduce_sum_f64(&self, buf: &mut [f64]);
    /// Block until every partition has arrived.
    fn barrier(&self);

    /// Panic unless every partition passed the same `fingerprint`.
    ///
    /// The intended fingerprint is whatever a run's partitions *must* agree on before they can be driven in lock-step — the channel count, the direction, the qubit count, the truncation policy's identity.
    /// Disagree on any of those and the group deadlocks on the first layer whose collectives no longer pair up; this turns that hang into a message.
    /// The driver calls it exactly once per propagation, before the first layer.
    ///
    /// One collective, and it obeys the collective-order invariant like the rest: every partition calls it, with its own fingerprint.
    ///
    /// # Panics
    ///
    /// If the fingerprints differ, naming a bit the group disagrees on and how many partitions set it.
    fn check_consistency(&self, fingerprint: u64) {
        // 64 counters, one per bit of the fingerprint: "how many partitions have this bit set".
        // A group that agrees answers 0 or `size` for every bit; a group that disagrees cannot, because the counts pin every partition's value bit by bit.
        let mut counts = [0u64; 64];
        for (i, c) in counts.iter_mut().enumerate() {
            *c = (fingerprint >> i) & 1;
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

/// The per-layer all-to-all: each partition hands over what it exports and gets back what its partners exported to it.
///
/// Not object-safe ([`exchange_layer`](Self::exchange_layer) is generic over the payload) — deliberate, since the layer code is generic over the transport, so a call monomorphizes into the partition's driving thread with no virtual dispatch on a per-layer path.
pub trait Transport: Collectives {
    /// **The** exchange: send `send[q]` to partition `q`, run `body` on what the partners sent here, and give both back.
    ///
    /// It is **two-phase**. Everything the coset loop needs to size its gather runs — the block headers and the CSR offsets (`Payload::early_parts`) — has arrived before `body` starts; the rows themselves (`Payload::bulk_parts`) may still be in flight while it runs, and `body` blocks on the `ChunkWait` it is handed before it reads a chunk's rows.
    /// A transport with nothing to overlap completes the transfer first and hands over a no-op `ChunkWait`, which makes it a blocking exchange with extra steps — that is what `InProcessTransport` is, its "transfer" being a moved pointer.
    ///
    /// `send.len()` must be [`size`](Collectives::size) and `send[self.rank()]` must be `None`; the slots handed to `body` have the same length and `None` in the same self slot.
    /// A partner with nothing to send still participates, with `None` — silence would desynchronize the group (module docs).
    ///
    /// `map` is the destination-coset order both sides laid the blocks out in and the chunks the bulk transfer is cut into.
    /// Every partition passes the same one, for the same reason both sides agree on the bucket count.
    ///
    /// `spare` is the caller's **payload pool**, and it is what keeps a steady-state layer from allocating: a transport that has to materialize the received payloads takes them from here rather than building them fresh, and returns any payload from `send` it is finished with.
    /// The caller returns the results to the pool once the layer has consumed them.
    /// Pooled payloads are in an unspecified state — a taker reshapes one before it reads anything back — and an empty pool is always legal, so a transport must fall back to [`Default`].
    /// `InProcessTransport` uses neither direction: it *moves* the sender's payload to the receiver, so the pool circulates through the partners instead.
    ///
    /// `body` returns whatever the caller needs out of the layer; the received payloads come back with it, for the caller to return to `spare`.
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

    /// [`exchange_layer`](Self::exchange_layer) with nothing to overlap: the blocking all-to-all, for a caller that wants the payloads and no more.
    ///
    /// The empty `ChunkMap` cuts no chunks, so every part travels as an early one — which is what a payload with no interesting internal structure wants, and the gather's is the only one.
    /// A payload whose `bulk_parts` needs a real map goes through `exchange_layer`.
    fn exchange<P: Payload>(&self, send: Vec<Option<P>>, spare: &mut Vec<P>) -> Vec<Option<P>> {
        self.exchange_layer(send, spare, &ChunkMap::default(), |_, _| ())
            .0
    }

    /// Collect every partition's `parts` on partition 0.
    ///
    /// `Some(v)` on rank 0, where `v[q]` is partition `q`'s parts in the order it passed them (`v[0]` being the caller's own); `None` everywhere else.
    /// This is the gather that ends a distributed run: each partition hands over its share of the sum as bytes and rank 0 reassembles.
    ///
    /// One collective. Unlike the exchange it is deliberately *not* symmetric — every partition talks to rank 0 and to nobody else — so a transport that infers its partner set from the `Some` positions (as the MPI one does) overrides this rather than inheriting it.
    /// The default body is the honest all-to-all: rank 0 sends nothing and receives from everyone.
    fn gather_to_root(&self, parts: Vec<&[u8]>) -> Option<Vec<Vec<Vec<u8>>>> {
        let n = self.size() as usize;
        let me = self.rank() as usize;
        let mine = ByteParts(parts.iter().map(|p| p.to_vec()).collect());

        if me != ROOT {
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
pub(crate) const ROOT: usize = 0;

/// A [`Payload`] that is already bytes: the gather's wire form.
///
/// [`Transport::gather_to_root`]'s default body moves one of these per partition through [`Transport::exchange`], so a transport gets a gather for free from the exchange it already implements.
#[derive(Default)]
pub(crate) struct ByteParts(pub(crate) Vec<Vec<u8>>);

impl Payload for ByteParts {
    fn byte_parts(&self) -> Vec<&[u8]> {
        self.0.iter().map(Vec::as_slice).collect()
    }

    /// Already bytes, so "receiving into the typed columns" is receiving into
    /// the parts themselves.
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
