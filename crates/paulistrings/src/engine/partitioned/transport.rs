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
    pub fn chunk_of_position(&self, p: u32) -> usize {
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
pub trait Collectives: Send + Sync {
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
    /// It is **two-phase**. Everything the coset loop needs to size its gather runs — the block headers and the CSR offsets ([`Payload::early_parts`]) — has arrived before `body` starts; the rows themselves ([`Payload::bulk_parts`]) may still be in flight while it runs, and `body` blocks on the [`ChunkWait`] it is handed before it reads a chunk's rows.
    /// A transport with nothing to overlap completes the transfer first and hands over a no-op [`ChunkWait`], which makes it a blocking exchange with extra steps — that is what [`InProcessTransport`] is, its "transfer" being a moved pointer.
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
    /// [`InProcessTransport`] uses neither direction: it *moves* the sender's payload to the receiver, so the pool circulates through the partners instead.
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
    /// The empty [`ChunkMap`] cuts no chunks, so every part travels as an early one — which is what a payload with no interesting internal structure wants, and the gather's is the only one.
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
mod tests {
    use super::exchange_block::{check_stride, BlockHeader, PARTS_PER_BLOCK};
    use super::in_process::WAIT_TIMEOUT;
    use super::*;
    use num_complex::Complex64;
    use std::time::{Duration, Instant};

    use proptest::prelude::*;

    // ---- the destination-coset order and its chunks -----------------------

    /// The map for `deltas` over `2^bits` buckets, cut into `chunks`.
    fn map_of(bits: u8, deltas: &[u32], chunks: usize) -> ChunkMap {
        let mut map = ChunkMap::default();
        map.rebuild(&Gf2Span::new(deltas, bits), 1usize << bits, chunks);
        map
    }

    /// A payload of `blocks` blocks over `2^bits` positions, with a few rows
    /// per position, filled deterministically.
    fn payload_of<const W: usize>(bits: u8, blocks: usize, seed: u64) -> PartnerPayload<W> {
        let n = 1usize << bits;
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut payload = PartnerPayload::<W>::default();
        for e in 0..blocks {
            let counts: Vec<u32> = (0..n).map(|_| (next() % 4) as u32).collect();
            let mut block = ExchangeBlock::<W>::with_counts(e as u32, &counts);
            let rows = block.rows();
            block.x = (0..rows).map(|_| std::array::from_fn(|_| next())).collect();
            block.z = (0..rows).map(|_| std::array::from_fn(|_| next())).collect();
            block.coeff = (0..rows)
                .map(|_| Complex64::new(next() as f64 * 1e-18, next() as f64 * 1e-18))
                .collect();
            payload.blocks.push(block);
        }
        payload
    }

    /// The two-phase framing is a partition of the one-phase one: the early parts are the block headers and offsets, and each bulk part's chunks concatenate to exactly the column that part carries.
    ///
    /// That is the whole contract between the sender's `bulk_parts` and the receiver's `bulk_recv_into` — get it wrong and the two sides post different message sizes, which MPI reports as a truncated message rather than as wrong rows.
    #[test]
    fn the_early_and_bulk_parts_partition_the_wire() {
        for bits in [0u8, 1, 4, 5] {
            for blocks in [1usize, 3] {
                let mut payload = payload_of::<2>(bits, blocks, 0xB1A5 + u64::from(bits));
                let all: Vec<Vec<u8>> = payload
                    .byte_parts()
                    .into_iter()
                    .map(<[u8]>::to_vec)
                    .collect();
                for chunks in [1usize, 2, 3, 8, 64] {
                    let map = map_of(bits, &[0], chunks);
                    let early: Vec<Vec<u8>> = payload
                        .early_parts()
                        .into_iter()
                        .map(<[u8]>::to_vec)
                        .collect();
                    assert_eq!(early.len(), 2 * blocks);
                    for b in 0..blocks {
                        assert_eq!(early[2 * b], all[PARTS_PER_BLOCK * b], "block {b} header");
                        assert_eq!(
                            early[2 * b + 1],
                            all[PARTS_PER_BLOCK * b + 1],
                            "block {b} offsets",
                        );
                    }

                    let bulk = payload.bulk_parts(&map);
                    assert_eq!(bulk.len(), 3 * blocks);
                    for (i, pieces) in bulk.iter().enumerate() {
                        assert_eq!(pieces.len(), map.chunks(), "part {i} chunk count");
                        let joined: Vec<u8> =
                            pieces.iter().flat_map(|p| p.iter().copied()).collect();
                        let want = &all[PARTS_PER_BLOCK * (i / 3) + 2 + i % 3];
                        assert_eq!(&joined, want, "part {i} at {chunks} chunks");
                    }
                    // The receive side cuts the same column the same way.
                    let want_lens: Vec<Vec<usize>> = bulk
                        .iter()
                        .map(|p| p.iter().map(|c| c.len()).collect())
                        .collect();
                    let got_lens: Vec<Vec<usize>> = payload
                        .bulk_recv_into(&map)
                        .iter()
                        .map(|p| p.iter().map(|c| c.len()).collect())
                        .collect();
                    assert_eq!(got_lens, want_lens, "receive side at {chunks} chunks");
                }
            }
        }
    }

    /// Both directions of the layout are one permutation: every bucket has one position, every position one bucket, and nothing is dropped or duplicated.
    /// This is the property the sender's permuted CSR and the receiver's table lookup both rest on.
    #[test]
    fn the_destination_order_is_a_bijection() {
        for (bits, deltas) in [
            (0u8, vec![0u32]),
            (1, vec![0]),
            (4, vec![0]),
            (4, vec![0, 1]),
            (4, vec![0, 3, 5]),
            (6, vec![0, 1, 2, 3]),
            (6, vec![0, 9, 18, 27]),
        ] {
            let map = map_of(bits, &deltas, 1);
            let n = 1u32 << bits;
            assert_eq!(map.positions(), n as usize);
            let mut seen = vec![false; n as usize];
            for beta in 0..n {
                let p = map.position_of(beta);
                assert!(p < n, "position {p} outside {n} for bits={bits}");
                assert!(!seen[p as usize], "position {p} claimed twice");
                seen[p as usize] = true;
                assert_eq!(map.bucket_at(p), beta, "bucket_at is not the inverse");
            }
            assert!(
                seen.iter().all(|&s| s),
                "bits={bits} left a position unfilled"
            );
        }
    }

    /// The chunks tile the position range: ascending bounds from 0 to the position count, and `chunk_of_position` is their inverse.
    #[test]
    fn the_chunks_tile_the_position_range() {
        for (bits, deltas) in [(4u8, vec![0u32]), (6, vec![0, 3]), (6, vec![0, 1, 2, 3])] {
            for chunks in [1usize, 2, 3, 5, 8, 16] {
                let map = map_of(bits, &deltas, chunks);
                let k = map.chunks();
                assert!(k >= 1 && k <= chunks);
                assert_eq!(map.bound(0), 0);
                assert_eq!(map.bound(k) as usize, map.positions());
                for i in 0..k {
                    assert!(
                        map.bound(i) < map.bound(i + 1),
                        "chunk {i} of {k} is empty at bits={bits}",
                    );
                }
                for p in 0..map.positions() as u32 {
                    let c = map.chunk_of_position(p);
                    assert!(
                        map.bound(c) <= p && p < map.bound(c + 1),
                        "position {p} says chunk {c}, whose range is                          {}..{}",
                        map.bound(c),
                        map.bound(c + 1),
                    );
                }
            }
        }
    }

    /// A chunk boundary always falls between cosets, so a coset task never has to wait for two chunks — which is what makes the receiver's per-chunk wait sound.
    #[test]
    fn a_chunk_never_splits_a_coset() {
        for (bits, deltas) in [(6u8, vec![0u32, 3]), (6, vec![0, 1, 2, 3]), (5, vec![0, 7])] {
            let span = Gf2Span::new(&deltas, bits);
            for chunks in [1usize, 2, 4, 8, 64] {
                let mut map = ChunkMap::default();
                map.rebuild(&span, 1usize << bits, chunks);
                let coset = span.coset_size() as u32;
                for k in 0..=map.chunks() {
                    assert_eq!(
                        map.bound(k) % coset,
                        0,
                        "chunk bound {} is not on a coset boundary of {coset}",
                        map.bound(k),
                    );
                }
                for beta in 0..(1u32 << bits) {
                    let want = map.chunk_of_position(map.position_of(span.rep_of(beta)));
                    assert_eq!(
                        map.chunk_of_position(map.position_of(beta)),
                        want,
                        "bucket {beta} is in another chunk than its coset representative",
                    );
                }
            }
        }
    }

    /// Asking for more chunks than there are cosets would make empty ones; the map clamps instead, so the pipeline degrades to one batch per coset.
    #[test]
    fn the_chunk_count_is_clamped_to_the_coset_count() {
        // 2^3 buckets, a span of dimension 2, so two cosets.
        let map = map_of(3, &[0, 1, 2, 3], 16);
        assert_eq!(map.chunks(), 2);
        assert_eq!(map.bound(0), 0);
        assert_eq!(map.bound(1), 4);
        assert_eq!(map.bound(2), 8);
    }

    proptest! {
        /// The two properties above over arbitrary spans and chunk counts.
        #[test]
        fn the_layout_is_a_bijection_and_its_chunks_tile_it(
            bits in 0u8..=7,
            deltas in prop::collection::vec(0u32..128, 0..5),
            chunks in 1usize..=20,
        ) {
            let hi = 1u32 << bits;
            let mut deltas: Vec<u32> = deltas.into_iter().map(|d| d % hi).chain([0]).collect();
            deltas.sort_unstable();
            deltas.dedup();
            let map = map_of(bits, &deltas, chunks);
            let mut seen = vec![false; hi as usize];
            for beta in 0..hi {
                let p = map.position_of(beta);
                prop_assert!(p < hi);
                prop_assert!(!seen[p as usize]);
                seen[p as usize] = true;
                prop_assert_eq!(map.bucket_at(p), beta);
                let c = map.chunk_of_position(p);
                prop_assert!(map.bound(c) <= p && p < map.bound(c + 1));
            }
            prop_assert_eq!(map.bound(map.chunks()) as usize, map.positions());
        }
    }

    /// Deterministic pseudo-random row filler: xorshift64, so the tests carry no RNG dependency and a failing case is reproducible from its seed.
    fn fill<const W: usize>(block: &mut ExchangeBlock<W>, seed: u64) {
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..block.rows() {
            block.x.push(std::array::from_fn(|_| next()));
            block.z.push(std::array::from_fn(|_| next()));
            block
                .coeff
                .push(Complex64::new(next() as f64 * 1e-18, next() as f64 * 1e-18));
        }
    }

    #[test]
    fn with_counts_builds_offsets_and_reserves_the_columns() {
        let block = ExchangeBlock::<2>::with_counts(3, &[2, 0, 5, 1]);

        assert_eq!(block.offsets, vec![0, 2, 2, 7, 8]);
        assert_eq!(
            block.header,
            BlockHeader {
                num_buckets: 4,
                rows: 8,
                w: 2,
                entry: 3,
            }
        );
        assert_eq!(block.rows(), 8);
        assert_eq!(block.num_buckets(), 4);
        // Sized, not filled: the export pass writes the rows by index.
        assert_eq!(block.x.len(), 8);
        assert_eq!(block.z.len(), 8);
        assert_eq!(block.coeff.len(), 8);
    }

    /// A block re-aimed at a new layer keeps its storage and reports the new shape: the columns grow to what the widest layer needed and stay there, so a narrower layer neither shrinks nor re-zeroes them — and `rows()`, not `x.len()`, is what says how much of a column is live.
    #[test]
    fn set_counts_reuses_the_columns_and_grows_only() {
        let mut block = ExchangeBlock::<1>::with_counts(0, &[4, 4]);
        let wide = block.x.as_ptr();
        assert_eq!(block.rows(), 8);

        block.set_counts(1, &[1, 2]);
        assert_eq!(block.rows(), 3);
        assert_eq!(block.offsets, vec![0, 1, 3]);
        assert_eq!(block.header.entry, 1);
        assert_eq!(block.cols().2.len(), 3, "three live rows");
        assert_eq!(block.x.len(), 8, "the storage of the wider layer is kept");
        assert_eq!(block.x.as_ptr(), wide, "and it is the same allocation");
        assert_eq!(block.bytes(), 16 + 3 * 4 + 3 * 8 + 3 * 8 + 3 * 16);

        block.set_counts(1, &[9, 9]);
        assert_eq!(block.rows(), 18);
        assert!(block.coeff.len() >= 18, "grown for the wider layer");
    }

    #[test]
    fn with_counts_of_no_buckets_is_an_empty_block() {
        let block = ExchangeBlock::<1>::with_counts(0, &[]);
        assert_eq!(block.offsets, vec![0]);
        assert_eq!(block.rows(), 0);
        assert_eq!(block.num_buckets(), 0);
    }

    #[test]
    fn segment_slices_the_columns_by_source_bucket() {
        let mut block = ExchangeBlock::<1>::with_counts(0, &[2, 0, 1]);
        fill(&mut block, 0xa5a5);

        let (x0, z0, c0) = block.segment(0);
        assert_eq!(x0.len(), 2);
        assert_eq!(x0, &block.x[0..2]);
        assert_eq!(z0, &block.z[0..2]);
        assert_eq!(c0, &block.coeff[0..2]);

        // An empty source bucket yields three empty slices, not a panic.
        let (x1, z1, c1) = block.segment(1);
        assert!(x1.is_empty() && z1.is_empty() && c1.is_empty());

        let (x2, z2, c2) = block.segment(2);
        assert_eq!(x2, &block.x[2..3]);
        assert_eq!(z2, &block.z[2..3]);
        assert_eq!(c2, &block.coeff[2..3]);
    }

    #[test]
    fn bytes_counts_the_wire_footprint() {
        let mut block = ExchangeBlock::<1>::with_counts(0, &[2, 1]);
        fill(&mut block, 7);
        // header 16 + offsets 3*4 + x 3*8 + z 3*8 + coeff 3*16
        assert_eq!(block.bytes(), 16 + 12 + 24 + 24 + 48);

        let mut wide = ExchangeBlock::<2>::with_counts(0, &[1]);
        fill(&mut wide, 7);
        // header 16 + offsets 2*4 + x 1*16 + z 1*16 + coeff 1*16
        assert_eq!(wide.bytes(), 16 + 8 + 16 + 16 + 16);
    }

    #[test]
    fn byte_parts_are_five_borrowed_views_per_block() {
        let mut payload = PartnerPayload::<1>::default();
        let mut block = ExchangeBlock::<1>::with_counts(2, &[1, 1]);
        fill(&mut block, 11);
        payload.blocks.push(block);

        let parts = payload.byte_parts();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0].len(), 16); // header
        assert_eq!(parts[1].len(), 12); // offsets: 3 × u32
        assert_eq!(parts[2].len(), 16); // x: 2 rows × 1 word
        assert_eq!(parts[3].len(), 16); // z
        assert_eq!(parts[4].len(), 32); // coeff: 2 × 16 B
        assert_eq!(parts.iter().map(|p| p.len()).sum::<usize>(), 92);
    }

    /// Move `payload` over the wire and back: [`Payload::byte_parts`] on the sender, [`Payload::recv_into`] + [`Payload::finish_recv`] on a fresh receiver, with the bytes copied across the way a transport moves them.
    ///
    /// That pair is the only encode/decode path the engine has, so it is what the wire-format tests exercise.
    fn wire_round_trip<const W: usize>(payload: &PartnerPayload<W>) -> PartnerPayload<W> {
        let sent: Vec<Vec<u8>> = payload
            .byte_parts()
            .into_iter()
            .map(<[u8]>::to_vec)
            .collect();
        let lens: Vec<usize> = sent.iter().map(Vec::len).collect();
        let mut back = PartnerPayload::<W>::default();
        for (view, bytes) in back.recv_into(&lens).into_iter().zip(&sent) {
            view.copy_from_slice(bytes);
        }
        back.finish_recv();
        back
    }

    #[test]
    fn an_empty_payload_round_trips_as_zero_parts() {
        let payload = PartnerPayload::<2>::default();
        assert!(payload.byte_parts().is_empty());
        assert_eq!(wire_round_trip(&payload), payload);
    }

    #[test]
    fn payload_round_trips_through_byte_parts() {
        let mut payload = PartnerPayload::<2>::default();
        for (entry, counts) in [(0u32, &[2u32, 0, 1][..]), (1, &[0, 3, 0][..])] {
            let mut block = ExchangeBlock::<2>::with_counts(entry, counts);
            fill(&mut block, entry as u64 + 3);
            payload.blocks.push(block);
        }

        let back = wire_round_trip(&payload);
        assert_eq!(back, payload);
        // And the CSR indexing survives byte-for-byte.
        assert_eq!(back.blocks[1].segment(1).0, payload.blocks[1].segment(1).0);
    }

    /// A block whose header says it was built at another width is rejected rather than reinterpreted.
    /// The declared part lengths cannot catch it — they are a whole number of rows either way — so [`Payload::finish_recv`] is what does.
    #[test]
    #[should_panic(expected = "width")]
    fn a_block_encoded_at_another_width_is_rejected() {
        let mut payload = payload_of::<2>(2, 1, 0x5);
        payload.blocks[0].header.w = 1;
        let _ = wire_round_trip(&payload);
    }

    /// Small random payloads: 0–2 blocks, 1–4 source buckets, 0–3 rows each.
    fn arb_payload<const W: usize>() -> impl Strategy<Value = PartnerPayload<W>> {
        let block = (
            0u32..8,
            proptest::collection::vec(0u32..4, 1..5),
            any::<u64>(),
        )
            .prop_map(|(entry, counts, seed)| {
                let mut block = ExchangeBlock::<W>::with_counts(entry, &counts);
                fill(&mut block, seed);
                block
            });
        proptest::collection::vec(block, 0..3).prop_map(|blocks| PartnerPayload { blocks })
    }

    proptest! {
        #[test]
        fn arbitrary_payloads_round_trip_at_w1(payload in arb_payload::<1>()) {
            prop_assert_eq!(wire_round_trip(&payload), payload);
        }

        #[test]
        fn arbitrary_payloads_round_trip_at_w2(payload in arb_payload::<2>()) {
            prop_assert_eq!(wire_round_trip(&payload), payload);
        }
    }

    // ---- transport ------------------------------------------------------

    /// A minimal [`Payload`] for the transport tests: one column of `u64`.
    impl Payload for Vec<u64> {
        fn byte_parts(&self) -> Vec<&[u8]> {
            vec![bytemuck::cast_slice(&self[..])]
        }

        fn recv_into(&mut self, lens: &[usize]) -> Vec<&mut [u8]> {
            assert_eq!(lens.len(), 1, "test payload: expected one part");
            check_stride(lens[0], size_of::<u64>(), "test payload");
            self.clear();
            self.resize(lens[0] / size_of::<u64>(), 0);
            vec![bytemuck::cast_slice_mut(&mut self[..])]
        }

        fn finish_recv(&mut self) {}
    }

    /// What rank `from` sends to rank `to`: a `from`-long column of a code unique to the ordered pair, so a crossed delivery cannot pass.
    fn message(from: u32, to: u32) -> Vec<u64> {
        vec![(u64::from(from) << 32) | u64::from(to); from as usize + 1]
    }

    /// Rank 0 sends nothing to rank 1, so a `None` slot is exercised too.
    fn sends_nothing(from: u32, to: u32) -> bool {
        from == 0 && to == 1
    }

    fn exchange_round(size: u32) {
        let group = InProcessTransport::group(size);
        assert_eq!(group.len(), size as usize);

        std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .into_iter()
                .map(|transport| {
                    scope.spawn(move || {
                        let rank = transport.rank();
                        assert_eq!(transport.size(), size);
                        let send: Vec<Option<Vec<u64>>> = (0..size)
                            .map(|q| {
                                (q != rank && !sends_nothing(rank, q)).then(|| message(rank, q))
                            })
                            .collect();
                        (rank, transport.exchange(send, &mut Vec::new()))
                    })
                })
                .collect();

            for handle in handles {
                let (rank, recv) = handle.join().expect("rank thread panicked");
                assert_eq!(recv.len(), size as usize, "rank {rank}");
                for q in 0..size {
                    let expected = (q != rank && !sends_nothing(q, rank)).then(|| message(q, rank));
                    assert_eq!(recv[q as usize], expected, "rank {rank} slot {q}");
                }
            }
        });
    }

    #[test]
    fn exchange_delivers_each_payload_to_its_partner() {
        for size in [2u32, 4] {
            exchange_round(size);
        }
    }

    #[test]
    fn reductions_and_barrier_agree_on_every_rank() {
        let size = 4u32;
        let group = InProcessTransport::group(size);

        let mut results: Vec<(u32, u8, Vec<u64>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .into_iter()
                .map(|transport| {
                    scope.spawn(move || {
                        let rank = transport.rank();
                        transport.barrier();
                        // Contributions 0, 7, 14, 21 → max 21.
                        let max = transport.allreduce_max_u8((rank * 7) as u8);
                        // Columns (r+1, 10·(r+1)) → sums (10, 100).
                        let mut buf = vec![u64::from(rank) + 1, 10 * (u64::from(rank) + 1)];
                        transport.allreduce_sum_u64(&mut buf);
                        transport.barrier();
                        (rank, max, buf)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank thread panicked"))
                .collect()
        });

        results.sort_by_key(|(rank, _, _)| *rank);
        assert_eq!(results.len(), size as usize);
        for (rank, max, sum) in results {
            assert_eq!(max, 21, "rank {rank}");
            assert_eq!(sum, vec![10, 100], "rank {rank}");
        }
    }

    #[test]
    fn a_group_of_one_is_a_no_op() {
        let group = InProcessTransport::group(1);
        assert_eq!(group.len(), 1);
        let transport = &group[0];
        assert_eq!(transport.rank(), 0);
        assert_eq!(transport.size(), 1);

        let recv: Vec<Option<Vec<u64>>> = transport.exchange(vec![None], &mut Vec::new());
        assert_eq!(recv.len(), 1);
        assert!(recv[0].is_none());

        assert_eq!(transport.allreduce_max_u8(9), 9);
        let mut buf = vec![3, 4];
        transport.allreduce_sum_u64(&mut buf);
        assert_eq!(buf, vec![3, 4]);
        let mut buf = vec![-0.5, 1e300];
        transport.allreduce_sum_f64(&mut buf);
        assert_eq!(buf, vec![-0.5, 1e300]);
        transport.barrier();
    }

    /// Contributions chosen so that the sum depends on the order: `1e16 + 1 + 1 - 1e16` is `0` or `2` by association.
    /// Every rank must return the same bits, namely the rank-order fold `((0 + 1e16) + 1) + 1) - 1e16 = 0`, and a one-rank slot comes back exactly.
    #[test]
    fn f64_sums_are_bitwise_identical_on_every_rank() {
        let inputs = [
            [1e16, 0.25, 0.1, 0.0, 0.0],
            [1.0, 1.25, 0.0, 0.2, 0.0],
            [1.0, 2.25, 0.0, 0.0, 0.3],
            [-1e16, 3.25, 0.0, 0.0, 0.0],
        ];
        let results = on_every_rank(4, |transport| {
            (0..50)
                .map(|_| {
                    let mut buf = inputs[transport.rank() as usize];
                    transport.allreduce_sum_f64(&mut buf);
                    buf
                })
                .collect::<Vec<_>>()
        });
        for (rank, rounds) in results.iter().enumerate() {
            for round in rounds {
                assert_eq!(round, &[0.0, 7.0, 0.1, 0.2, 0.3], "rank {rank}");
            }
        }
    }

    /// An integer sum on one rank against a float sum on the others is a collective-order violation, reported by name.
    #[test]
    #[should_panic(expected = "allreduce_sum_f64")]
    fn mixing_integer_and_float_sums_is_detected() {
        let group = InProcessTransport::group(2);
        std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .into_iter()
                .map(|transport| {
                    scope.spawn(move || {
                        if transport.rank() == 0 {
                            transport.allreduce_sum_u64(&mut [1]);
                        } else {
                            transport.allreduce_sum_f64(&mut [1.0]);
                        }
                    })
                })
                .collect();
            let joined: Vec<_> = handles.into_iter().map(|h| h.join()).collect();
            for result in joined {
                if let Err(payload) = result {
                    std::panic::resume_unwind(payload);
                }
            }
        });
    }

    #[test]
    #[should_panic(expected = "must be None")]
    fn exchange_rejects_a_payload_addressed_to_this_partition() {
        let group = InProcessTransport::group(1);
        let _: Vec<Option<Vec<u64>>> = group[0].exchange(vec![Some(vec![1u64])], &mut Vec::new());
    }

    #[test]
    #[should_panic(expected = "one entry per partition")]
    fn exchange_rejects_a_wrongly_sized_send_vector() {
        let group = InProcessTransport::group(2);
        let _: Vec<Option<Vec<u64>>> = group[0].exchange(vec![None], &mut Vec::new());
    }

    /// A partition that issues one collective more than its partners is caught by the generation stamp rather than crossing payloads.
    ///
    /// The *in-step* rank is the one that names it: the desynchronized rank has published a generation its partner never reached, so the partner finds a call it did not issue in that generation's slot.
    /// The desynchronized rank is left waiting for a generation that will never come, and dies of its partner's departure instead — both panics are checked here, and both ranks own their endpoint inside their own thread so neither drop waits on the other.
    #[test]
    fn a_desynchronized_partition_is_caught_by_the_generation_stamp() {
        let mut group = InProcessTransport::group(2);
        let one = group.pop().expect("rank 1");
        let zero = group.pop().expect("rank 0");

        let (desynced, in_step) = std::thread::scope(|scope| {
            // Rank 0 behaves as if it had issued one extra collective.
            let desynced = scope.spawn(move || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                    zero.skip_sequence_for_test();
                    zero.barrier();
                }))
            });
            let in_step = scope.spawn(move || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || one.barrier()))
            });
            (
                desynced.join().expect("rank 0 thread"),
                in_step.join().expect("rank 1 thread"),
            )
        });

        let in_step = panic_message(
            in_step
                .expect_err("the in-step rank must reject the mismatched generation")
                .as_ref(),
        );
        assert!(in_step.contains("collective order mismatch"), "{in_step}");
        let desynced = panic_message(
            desynced
                .expect_err("the desynchronized rank waits for a call nobody makes")
                .as_ref(),
        );
        assert!(desynced.contains("terminated"), "{desynced}");
    }

    /// The panic message behind a `catch_unwind` payload.
    fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
        if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "<panic payload is not a string>".to_string()
        }
    }

    /// A rank that dies *between* two collectives must fail its partners fast — through the departure mask, not the 10 s backstop — and name itself.
    #[test]
    fn a_partner_that_dies_between_collectives_fails_the_others_promptly() {
        let mut group = InProcessTransport::group(2);
        let one = group.pop().expect("rank 1");
        let zero = group.pop().expect("rank 0");

        let (elapsed, message) = std::thread::scope(|scope| {
            scope.spawn(move || {
                // The transport is dropped *while unwinding*, which is what a
                // partitioned run does when a partition body panics.
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                    one.barrier();
                    panic!("rank 1 dies after its barrier");
                }));
            });
            zero.barrier();
            let started = Instant::now();
            let payload =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| zero.allreduce_max_u8(3)))
                    .expect_err("the surviving rank cannot complete a collective alone");
            (started.elapsed(), panic_message(payload.as_ref()))
        });

        assert!(message.contains("terminated"), "{message}");
        assert!(message.contains("partition 1"), "{message}");
        assert!(
            elapsed < Duration::from_secs(2),
            "the survivor waited {elapsed:?}, so it fell back on the {WAIT_TIMEOUT:?} timeout \
             instead of noticing the departure",
        );
    }

    /// A thousand back-to-back reductions on four ranks with rank- *and*
    /// round-dependent contributions: a generation that mixed with its
    /// neighbour shows up as a wrong sum, not as a hang.
    #[test]
    fn a_thousand_back_to_back_sums_never_mix_generations() {
        const ROUNDS: u64 = 1_000;
        let size = 4u32;
        let group = InProcessTransport::group(size);

        std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .into_iter()
                .map(|transport| {
                    scope.spawn(move || {
                        let rank = u64::from(transport.rank());
                        for round in 1..=ROUNDS {
                            let mut buf = vec![rank * round, round, rank];
                            transport.allreduce_sum_u64(&mut buf);
                            // Σ rank = 0+1+2+3 = 6, Σ round = 4·round.
                            assert_eq!(
                                buf,
                                vec![6 * round, 4 * round, 6],
                                "rank {rank}, round {round}",
                            );
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().expect("rank thread panicked");
            }
        });
    }

    /// One round of the mixed script: the first `max`, the reduced buffer, and the flag `max`.
    type MixedRound = (u8, Vec<u64>, u8);
    /// What one rank came out of the mixed script with.
    type MixedScript = (u32, Vec<MixedRound>);

    /// Reductions and barriers interleaved: every rank runs the same mixed script and every rank must come out with the same hand-computed answers.
    #[test]
    fn mixed_collective_sequences_agree_on_every_rank() {
        const ROUNDS: u8 = 25;
        let size = 4u32;
        let group = InProcessTransport::group(size);

        let mut results: Vec<MixedScript> = std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .into_iter()
                .map(|transport| {
                    scope.spawn(move || {
                        let rank = transport.rank();
                        let mut rounds = Vec::new();
                        for round in 0..ROUNDS {
                            let hi = transport.allreduce_max_u8(rank as u8 * 3 + round);
                            transport.barrier();
                            let mut buf =
                                vec![u64::from(rank) + 1, u64::from(round), u64::from(rank) << 8];
                            transport.allreduce_sum_u64(&mut buf);
                            transport.barrier();
                            let flag = transport.allreduce_max_u8(if rank == 2 { 255 } else { 0 });
                            rounds.push((hi, buf, flag));
                        }
                        (rank, rounds)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank thread panicked"))
                .collect()
        });

        results.sort_by_key(|(rank, _)| *rank);
        for (rank, rounds) in &results {
            assert_eq!(rounds.len(), ROUNDS as usize, "rank {rank}");
            for (round, (hi, sum, flag)) in rounds.iter().enumerate() {
                let round = round as u8;
                // max over 3·rank + round is 9 + round; Σ(rank+1) = 10,
                // Σ round = 4·round, Σ(rank << 8) = 6·256.
                assert_eq!(*hi, 9 + round, "rank {rank}, round {round}");
                assert_eq!(
                    *sum,
                    vec![10, 4 * u64::from(round), 6 << 8],
                    "rank {rank}, round {round}",
                );
                assert_eq!(*flag, 255, "rank {rank}, round {round}");
            }
        }
        // And identical across ranks, not merely correct on each.
        for (rank, rounds) in &results[1..] {
            assert_eq!(rounds, &results[0].1, "rank {rank} disagrees with rank 0");
        }
    }

    /// Sixteen ranks — more than a CI box has cores — must finish rather than livelock: past [`SPINS_BEFORE_YIELD`] a waiting rank hands the core to the partner it is waiting for.
    /// Completing *is* the assertion.
    #[test]
    fn sixteen_ranks_complete_when_oversubscribed() {
        const ROUNDS: u64 = 50;
        let size = 16u32;
        let group = InProcessTransport::group(size);

        std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .into_iter()
                .map(|transport| {
                    scope.spawn(move || {
                        let rank = transport.rank();
                        for round in 1..=ROUNDS {
                            assert_eq!(transport.allreduce_max_u8(rank as u8 + 1), 16);
                            let mut buf = vec![u64::from(rank), round];
                            transport.allreduce_sum_u64(&mut buf);
                            // Σ_{r<16} r = 120.
                            assert_eq!(buf, vec![120, 16 * round], "rank {rank}, round {round}");
                            transport.barrier();
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().expect("rank thread panicked");
            }
        });
    }

    // ---- the two collectives every transport inherits ------------------

    /// Run `f` on every rank of a fresh group of `size`, joining in rank order.
    fn on_every_rank<O: Send>(
        size: u32,
        f: impl Fn(&InProcessTransport) -> O + Send + Sync,
    ) -> Vec<O> {
        let group = InProcessTransport::group(size);
        let f = &f;
        std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .into_iter()
                .map(|transport| scope.spawn(move || f(&transport)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank thread panicked"))
                .collect()
        })
    }

    #[test]
    fn check_consistency_passes_when_every_rank_agrees() {
        for size in [1u32, 2, 4] {
            on_every_rank(size, |transport| {
                transport.check_consistency(0xdead_beef_0000_0001);
            });
        }
    }

    /// One rank out of step is named, rather than left to deadlock two layers later.
    /// Every rank sees the mismatch (the reduction is symmetric), so rank 1 swallows its own panic and only rank 0's reaches the harness.
    #[test]
    #[should_panic(expected = "disagree about the run")]
    fn check_consistency_names_a_rank_that_disagrees() {
        let mut group = InProcessTransport::group(2);
        let one = group.pop().expect("rank 1");
        let zero = group.pop().expect("rank 0");

        std::thread::scope(|scope| {
            scope.spawn(move || {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    one.check_consistency(5)
                }));
            });
            zero.check_consistency(4);
        });
    }

    #[test]
    fn gather_to_root_collects_every_ranks_parts_in_rank_order() {
        for size in [1u32, 2, 4] {
            let got = on_every_rank(size, |transport| {
                let rank = transport.rank();
                // Rank `r` contributes `r + 1` parts, part `j` being `r + 1` copies of the byte `10 · r + j`, so a crossed or reordered delivery cannot pass.
                let owned: Vec<Vec<u8>> = (0..=rank)
                    .map(|j| vec![(10 * rank + j) as u8; rank as usize + 1])
                    .collect();
                let parts: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
                transport.gather_to_root(parts)
            });

            for (rank, out) in got.iter().enumerate() {
                if rank != 0 {
                    assert!(out.is_none(), "rank {rank} must not gather");
                    continue;
                }
                let all = out.as_ref().expect("rank 0 gathers");
                assert_eq!(all.len(), size as usize);
                for (r, parts) in all.iter().enumerate() {
                    let r = r as u32;
                    assert_eq!(parts.len(), r as usize + 1, "rank {r} part count");
                    for (j, part) in parts.iter().enumerate() {
                        assert_eq!(part, &vec![(10 * r + j as u32) as u8; r as usize + 1]);
                    }
                }
            }
        }
    }

    #[test]
    fn gather_to_root_of_no_parts_is_an_empty_vector_per_rank() {
        let got = on_every_rank(2, |transport| transport.gather_to_root(Vec::new()));
        assert_eq!(got[0], Some(vec![Vec::<Vec<u8>>::new(); 2]));
        assert_eq!(got[1], None);
    }

    /// A partner that died mid-layer must be reported, not waited on forever.
    #[test]
    #[should_panic(expected = "terminated")]
    fn a_partner_that_panicked_is_reported_rather_than_hanging() {
        let mut group = InProcessTransport::group(2);
        let one = group.pop().expect("rank 1");
        let zero = group.pop().expect("rank 0");

        std::thread::scope(|scope| {
            scope.spawn(move || {
                let _ = std::panic::catch_unwind(|| panic!("rank 1 dies before its exchange"));
                drop(one);
            });
            let _: Vec<Option<Vec<u64>>> =
                zero.exchange(vec![None, Some(vec![7u64])], &mut Vec::new());
        });
    }

    /// `GroupState::departed` marks a rank's bit with `1 << rank`: at rank 32 (the first index
    /// beyond a `u32` mask's width), a `u32` mask either panics on the shift (debug) or wraps the
    /// exponent and silently marks rank 0 instead (release) — both wrong, and `P_MAX_BITS` is
    /// meant to allow a group this large. Rank 32 dying must be reported by its own number, not
    /// rank 0's, and must not panic on the shift itself. Ranks 1..=31 must genuinely participate
    /// (not just sit idle) so none of them is itself mistaken for the dead partner.
    #[test]
    #[should_panic(expected = "partition 32 terminated")]
    fn a_partner_at_rank_32_that_panicked_is_reported_by_its_own_rank() {
        let mut group = InProcessTransport::group(33);
        let far = group.pop().expect("rank 32");
        let zero = group.remove(0);
        let middle = group; // ranks 1..=31

        std::thread::scope(|scope| {
            scope.spawn(move || {
                let _ = std::panic::catch_unwind(|| panic!("rank 32 dies before its barrier"));
                drop(far);
            });
            for transport in middle {
                scope.spawn(move || transport.barrier());
            }
            zero.barrier();
        });
    }
}
