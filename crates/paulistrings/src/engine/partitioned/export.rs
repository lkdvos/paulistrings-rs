//! The export pass: the rows a layer generates that belong to another
//! partition, gathered into one [`ExchangeBlock`] per remote delta.
//!
//! A remote delta `e` (see [`PartitionPlan`]) moves *every* row it produces to
//! the one partner `rank ^ pd[e]`, from local bucket `β` to that partner's
//! bucket `β ^ bd[e]`. So the export is a pure per-delta gather, with no
//! per-term routing decision.
//!
//! The block is CSR **in the receiver's destination-coset order**
//! ([`ChunkMap`](super::transport::ChunkMap)): segment `p` holds the rows for
//! the receiver's position `p`, generated from source bucket
//! `map.bucket_at(p) ^ bd[e]`. The permutation costs one pass over the count
//! array — the fill walks positions instead of buckets and is otherwise
//! unchanged — and it is what makes a receiving coset's rows contiguous, so
//! the transfer can be cut into chunks the receiver consumes as they land.
//!
//! Two passes over the local buckets, both parallel because a position's rows
//! occupy one contiguous CSR segment of each block and segments are disjoint:
//!
//! 1. **Count** rows per (remote delta, source bucket). An entry whose
//!    amplitude is nonzero on every active support pattern emits one row per
//!    term, so its count is the bucket length with no scan at all; a sparse
//!    entry is counted with one support-bit lookup per term, shared across
//!    every sparse entry of the layer.
//! 2. **Fill** each block, position by position, writing straight into
//!    `offsets[p]..offsets[p + 1]`.
//!
//! The row arithmetic is not reimplemented here: it is
//! [`DeltaEntry::emit`](crate::channel::prepared::DeltaEntry::emit) and
//! [`RotationPrep::emit_gen`](crate::channel::prepared::RotationPrep::emit_gen),
//! the row-level forms of the engine's own gather, so an exported row is
//! bitwise the row a local gather would have produced.

use num_complex::Complex64;
use rayon::prelude::*;

use super::plan::PartitionPlan;
use super::transport::{ChunkMap, ExchangeBlock, PartnerPayload};
use crate::bucket::sum::PauliSum;
use crate::channel::prepared::{DeltaEntry, LocalPtm, Prepared, RotationPrep};
use crate::pauli_string::PauliString;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Rows below which a block's fill stays on one thread.
///
/// The fill splits a block's position range in two and hands the halves
/// to `rayon::join`; the split is exact (the CSR offsets say where the halves
/// meet), so this threshold is purely about not paying task overhead for a
/// handful of rows.
const FILL_PARALLEL_MIN_ROWS: usize = 4096;

/// Reusable scratch for [`export_layer`], held by the caller across layers so
/// a layer allocates nothing after the first.
///
/// Both buffers are capacity-retaining: [`Self::counts`] is the pass-1 count
/// table in **bucket-major** order (`counts[β * K + k]` is remote delta `k`'s
/// row count from source bucket `β`), which is the layout that lets pass 1 run
/// as one disjoint `par_chunks_mut(K)` over buckets, and
/// [`Self::block_counts`] is the per-block column that layout is transposed
/// into for [`ExchangeBlock::set_counts`].
#[derive(Debug)]
pub(crate) struct ExportScratch<const W: usize> {
    /// Pass-1 counts, bucket-major: `counts[β * K + k]`.
    counts: Vec<u32>,
    /// One block's counts by destination position, gathered out of
    /// [`Self::counts`].
    block_counts: Vec<u32>,
    /// One block's source bucket per destination position:
    /// `src_of[p] = map.bucket_at(p) ^ bd`. Built once per block and read by
    /// both the count transpose and the fill.
    src_of: Vec<u32>,
    /// Payloads not currently in flight, with their block columns intact.
    ///
    /// The layer's own pool ([`Transport::exchange`](super::transport::Transport::exchange)):
    /// the export takes this layer's outgoing payloads from here, a transport
    /// that materializes the incoming ones takes those from here too, and both
    /// come back when the layer is done with them. It is what makes a
    /// steady-state remote layer allocate nothing at all — the alternative is
    /// faulting in and zeroing the layer's whole export volume, twice, every
    /// layer.
    pub(crate) pool: Vec<PartnerPayload<W>>,
    /// Sub-phase laps of the two passes, drained by the layer into
    /// [`PhaseStats`](crate::engine::stats::PhaseStats). Measurement only.
    #[cfg(feature = "phase-timing")]
    pub(crate) count_ns: u64,
    /// Pass 2's lap. Measurement only.
    #[cfg(feature = "phase-timing")]
    pub(crate) fill_ns: u64,
}

// Hand-written because `#[derive(Default)]` would demand `W: Default`.
impl<const W: usize> Default for ExportScratch<W> {
    fn default() -> Self {
        Self {
            counts: Vec::new(),
            block_counts: Vec::new(),
            src_of: Vec::new(),
            pool: Vec::new(),
            #[cfg(feature = "phase-timing")]
            count_ns: 0,
            #[cfg(feature = "phase-timing")]
            fill_ns: 0,
        }
    }
}

/// What one layer's export costs, by partner rank (length is the group size).
///
/// Rows and bytes are what this partition *sent*; a partner it sent nothing to
/// has zeros. The bytes are the wire footprint
/// ([`ExchangeBlock::bytes`]) of the blocks as they stand, so an
/// in-process transport reports the same number an MPI one would move.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExportCounts {
    /// Rows sent to each partner.
    pub rows_to: Vec<u64>,
    /// Wire bytes sent to each partner.
    pub bytes_to: Vec<u64>,
}

/// One remote delta's row-level emitter: the tabulated entry, or the wide
/// rotation's generator pass.
///
/// Exists so the two prepared forms share the count and fill loops. Both arms
/// delegate to the emitters in `channel::prepared`, which are copies of the
/// engine's gather arithmetic.
enum RowEmitter<'p, const W: usize> {
    /// A tabulated delta of a [`Prepared::Local`] layer.
    Tabulated {
        ptm: &'p LocalPtm<W>,
        entry: &'p DeltaEntry<W>,
    },
    /// The generator pass of a [`Prepared::Rotation`] layer. The identity pass
    /// is never remote (`part(0) = 0`), so it has no arm here.
    Generator { prep: &'p RotationPrep<W> },
}

impl<const W: usize> RowEmitter<'_, W> {
    /// Whether every term emits a row, so a bucket's count is its length.
    ///
    /// Dense over the **active** support patterns only: `amp` is sized
    /// `LOCAL_DIM` but the channel populates `4^k` entries, exactly as
    /// `engine::bucketed::DeltaPlan::new` reads it. A rotation's generator
    /// pass is never dense — a commuting term emits nothing — and its count
    /// path does not consult this at all, counting anticommuting terms
    /// directly.
    fn is_dense(&self) -> bool {
        match self {
            RowEmitter::Tabulated { ptm, entry } => {
                let dim = 1usize << (2 * ptm.k());
                entry.amp[..dim].iter().all(|a| *a != ZERO)
            }
            RowEmitter::Generator { .. } => false,
        }
    }

    /// The row this delta emits for one term, or `None` when it emits none.
    #[inline]
    fn emit(
        &self,
        x: &[u64; W],
        z: &[u64; W],
        c: Complex64,
    ) -> Option<([u64; W], [u64; W], Complex64)> {
        match self {
            RowEmitter::Tabulated { ptm, entry } => entry.emit(ptm.support_bits(x, z), x, z, c),
            RowEmitter::Generator { prep } => prep.emit_gen(x, z, c),
        }
    }
}

/// Build this partition's outgoing payloads for one layer.
///
/// Returns one slot per partner rank (`size` of them): `Some(payload)` for
/// every partner the plan names, `None` for the rest — including this
/// partition's own slot, which no remote delta can name because a remote
/// delta's partition delta is nonzero by construction.
///
/// A partner's [`PartnerPayload::blocks`] is one block per remote delta
/// destined for it, in ascending remote-delta index, **including blocks with
/// no rows**: the receiver indexes blocks positionally, and both sides derive
/// the same delta list from the same channel and hash, so a block is never
/// dropped for being empty.
pub(crate) fn export_layer<const W: usize>(
    local: &PauliSum<W>,
    prep: &Prepared<W>,
    plan: &PartitionPlan,
    size: u32,
    map: &ChunkMap,
    scratch: &mut ExportScratch<W>,
) -> (Vec<Option<PartnerPayload<W>>>, ExportCounts) {
    let k = plan.remote.len();
    let nb = local.num_buckets();
    let mut send: Vec<Option<PartnerPayload<W>>> = (0..size).map(|_| None).collect();
    let mut counts = ExportCounts {
        rows_to: vec![0; size as usize],
        bytes_to: vec![0; size as usize],
    };
    if k == 0 {
        return (send, counts);
    }

    let emitters: Vec<RowEmitter<'_, W>> = plan
        .remote
        .iter()
        .map(|r| match prep {
            Prepared::Local(ptm) => RowEmitter::Tabulated {
                ptm,
                entry: &ptm.deltas()[r.entry],
            },
            Prepared::Rotation(p) => {
                debug_assert_eq!(
                    r.entry, 1,
                    "a rotation's only remote entry is the generator pass",
                );
                RowEmitter::Generator { prep: p }
            }
        })
        .collect();
    let dense: Vec<bool> = emitters.iter().map(RowEmitter::is_dense).collect();

    #[cfg(feature = "phase-timing")]
    let mut st = crate::engine::stats::Stamp::now();

    // Pass 1. Bucket-major counts, so each bucket owns one contiguous chunk
    // and the pass needs no synchronization.
    scratch.counts.clear();
    scratch.counts.resize(nb * k, 0);
    scratch
        .counts
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(b, slot)| count_bucket(local, prep, plan, &dense, b, slot));
    #[cfg(feature = "phase-timing")]
    st.lap(&mut scratch.count_ns);

    // Pass 2, one block per remote delta, written into payloads taken from the
    // pool: their blocks already have the columns this layer needs, so the fill
    // writes by index into storage that is neither allocated nor zeroed here.
    debug_assert_eq!(
        map.positions(),
        nb,
        "the chunk map is built for a different bucket count than the sum has",
    );
    scratch.block_counts.clear();
    scratch.block_counts.resize(nb, 0);
    scratch.src_of.clear();
    scratch.src_of.resize(nb, 0);
    let mut blocks_used = vec![0usize; size as usize];
    for (i, r) in plan.remote.iter().enumerate() {
        // Destination-coset order: position `p` of the receiver is filled from
        // this partition's bucket `bucket_at(p) ^ bd`.
        for p in 0..nb {
            let src = map.bucket_at(p as u32) ^ r.bucket_delta;
            scratch.src_of[p] = src;
            scratch.block_counts[p] = scratch.counts[src as usize * k + i];
        }
        let q = r.partner as usize;
        let payload = send[q].get_or_insert_with(|| scratch.pool.pop().unwrap_or_default());
        let j = blocks_used[q];
        blocks_used[q] += 1;
        if payload.blocks.len() <= j {
            payload
                .blocks
                .resize_with(j + 1, ExchangeBlock::<W>::default);
        }
        let block = &mut payload.blocks[j];
        block.set_counts(r.entry as u32, &scratch.block_counts);
        let rows = block.rows();
        {
            let ExchangeBlock {
                ref offsets,
                ref mut x,
                ref mut z,
                ref mut coeff,
                ..
            } = *block;
            let cols = BlockCols {
                x: &mut x[..rows],
                z: &mut z[..rows],
                c: &mut coeff[..rows],
            };
            fill_range(local, &emitters[i], offsets, &scratch.src_of, 0, nb, cols);
        }
        counts.rows_to[q] += rows as u64;
        counts.bytes_to[q] += block.bytes() as u64;
    }
    // A payload out of the pool may have carried more blocks than this layer
    // has remote deltas for it; the receiver indexes blocks positionally, so
    // the extras must not travel.
    for (q, payload) in send.iter_mut().enumerate() {
        if let Some(payload) = payload {
            payload.blocks.truncate(blocks_used[q]);
        }
    }
    #[cfg(feature = "phase-timing")]
    st.lap(&mut scratch.fill_ns);

    (send, counts)
}

/// Count one source bucket's exported rows, one slot per remote delta.
///
/// The support pattern is computed once per term and reused across every
/// sparse entry, matching the engine's input-major gather; dense entries are
/// filled from the bucket length without touching a term at all.
fn count_bucket<const W: usize>(
    local: &PauliSum<W>,
    prep: &Prepared<W>,
    plan: &PartitionPlan,
    dense: &[bool],
    b: usize,
    out: &mut [u32],
) {
    let (bx, bz, bc) = local.bucket(b);
    let n = bc.len();
    match prep {
        Prepared::Local(ptm) => {
            let mut all_dense = true;
            for (k, &d) in dense.iter().enumerate() {
                if d {
                    out[k] = n as u32;
                } else {
                    out[k] = 0;
                    all_dense = false;
                }
            }
            if all_dense {
                return;
            }
            for t in 0..n {
                let s = ptm.support_bits(&bx[t], &bz[t]);
                for (k, r) in plan.remote.iter().enumerate() {
                    if !dense[k] && ptm.deltas()[r.entry].amp[s] != ZERO {
                        out[k] += 1;
                    }
                }
            }
        }
        Prepared::Rotation(p) => {
            debug_assert_eq!(out.len(), 1, "a rotation has at most one remote delta");
            let mut anti = 0u32;
            for t in 0..n {
                let v = PauliString::<W> { x: bx[t], z: bz[t] };
                if !v.commutes_with(&p.gen) {
                    anti += 1;
                }
            }
            out[0] = anti;
        }
    }
}

/// Fill destination positions `lo..hi` of one block.
///
/// `src_of[p]` is the source bucket position `p` draws from. The column slices
/// are exactly that range's rows (`offsets[lo]..offsets[hi]`), so splitting the
/// range at `mid` splits the columns at `offsets[mid] - offsets[lo]` and the
/// two halves are disjoint by construction — no atomics, no locks, and the same
/// rows in the same slots however the split falls, which is what keeps a
/// block's contents independent of the thread count.
fn fill_range<const W: usize>(
    local: &PauliSum<W>,
    emitter: &RowEmitter<'_, W>,
    offsets: &[u32],
    src_of: &[u32],
    lo: usize,
    hi: usize,
    cols: BlockCols<'_, W>,
) {
    debug_assert_eq!(cols.len(), (offsets[hi] - offsets[lo]) as usize);
    if hi - lo > 1 && cols.len() > FILL_PARALLEL_MIN_ROWS {
        let mid = lo + (hi - lo) / 2;
        let (head, tail) = cols.split_at((offsets[mid] - offsets[lo]) as usize);
        rayon::join(
            || fill_range(local, emitter, offsets, src_of, lo, mid, head),
            || fill_range(local, emitter, offsets, src_of, mid, hi, tail),
        );
        return;
    }
    let BlockCols { x, z, c } = cols;
    let mut w = 0usize;
    for &src in &src_of[lo..hi] {
        let (bx, bz, bc) = local.bucket(src as usize);
        for t in 0..bc.len() {
            if let Some((kx, kz, kc)) = emitter.emit(&bx[t], &bz[t], bc[t]) {
                x[w] = kx;
                z[w] = kz;
                c[w] = kc;
                w += 1;
            }
        }
    }
    debug_assert_eq!(
        w,
        x.len(),
        "export: pass 2 filled {w} rows where pass 1 counted {}",
        x.len(),
    );
}

/// One block's three columns, restricted to a source-bucket range.
///
/// A bundle, not an abstraction: it keeps [`fill_range`]'s recursive signature
/// inside clippy's argument budget and makes the three-way split one call
/// instead of three.
struct BlockCols<'a, const W: usize> {
    x: &'a mut [[u64; W]],
    z: &'a mut [[u64; W]],
    c: &'a mut [Complex64],
}

impl<'a, const W: usize> BlockCols<'a, W> {
    /// Rows in the range.
    fn len(&self) -> usize {
        self.c.len()
    }

    /// Split every column at the same row, consuming the borrow.
    fn split_at(self, at: usize) -> (BlockCols<'a, W>, BlockCols<'a, W>) {
        let (x0, x1) = self.x.split_at_mut(at);
        let (z0, z1) = self.z.split_at_mut(at);
        let (c0, c1) = self.c.split_at_mut(at);
        (
            BlockCols {
                x: x0,
                z: z0,
                c: c0,
            },
            BlockCols {
                x: x1,
                z: z1,
                c: c1,
            },
        )
    }
}

/// Every exported row belongs to the partner it is addressed to.
///
/// Lives here rather than inside [`export_layer`] because the export pass does
/// not need the partition rows for anything else — it routes by the plan
/// alone. Called from the partitioned layer, which has them; debug builds
/// only, and `O(exported rows)`.
///
/// A failure means either the plan misclassified a delta or the caller's
/// `local` sum was not the pure partition it claims to be.
#[cfg(debug_assertions)]
pub(super) fn debug_assert_exported_partitions<const W: usize>(
    send: &[Option<PartnerPayload<W>>],
    rows: &crate::bucket::hash::PartitionRows<W>,
) {
    for (q, payload) in send.iter().enumerate() {
        let Some(payload) = payload else { continue };
        for block in &payload.blocks {
            for i in 0..block.rows() {
                let part = rows.partition_of(&block.x[i], &block.z[i]);
                debug_assert_eq!(
                    part, q as u32,
                    "export: row {i} of the block for entry {} is addressed to partition {q} but \
                     belongs to partition {part}",
                    block.header.entry,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulator::BuildAccumulator;
    use crate::bucket::hash::{Gf2Hash, PartitionRows};
    use crate::channel::clifford::Clifford1Q;
    use crate::channel::rotation::PauliRotation;
    use crate::channel::{Channel, OutputBuffer};
    use crate::phase::Phase;
    use crate::test_support::{
        differential_channels_w1, differential_channels_w2, rand_sum, rand_sum_real,
    };
    use std::collections::HashMap;

    const TOL: f64 = 1e-12;

    /// The destination-coset order for `plan` over `num_buckets` buckets, in
    /// one chunk — the layout every export test compares against.
    fn map_for(plan: &PartitionPlan, num_buckets: usize) -> ChunkMap {
        let mut map = ChunkMap::default();
        map.rebuild(
            &crate::engine::coset::Gf2Span::new(
                &plan.local_bucket_deltas,
                num_buckets.trailing_zeros() as u8,
            ),
            num_buckets,
            1,
        );
        map
    }

    /// One partition's export of one layer.
    struct Exported<const W: usize> {
        rank: u32,
        send: Vec<Option<PartnerPayload<W>>>,
        counts: ExportCounts,
    }

    /// Split `input` across `rows`'s partitions and export one layer from each.
    fn export_all<const W: usize>(
        input: &PauliSum<W>,
        ch: &dyn Channel<W>,
        adjoint: bool,
        bits: u8,
        seed: u64,
        rows: &PartitionRows<W>,
    ) -> Vec<Exported<W>> {
        let hash = Gf2Hash::<W>::new(input.num_qubits(), bits, seed);
        let whole = input.clone().with_hash(hash);
        let prep = ch
            .prepare(whole.hash(), adjoint)
            .expect("channel could not be prepared");
        let size = rows.num_partitions() as u32;
        (0..size)
            .map(|rank| {
                let local = whole.filter_partition(rows, rank);
                let plan = PartitionPlan::new(&prep, rows, rank);
                let mut scratch = ExportScratch::default();
                let map = map_for(&plan, local.num_buckets());
                let (send, counts) = export_layer(&local, &prep, &plan, size, &map, &mut scratch);
                assert!(
                    send[rank as usize].is_none(),
                    "a partition exports to itself"
                );
                assert_eq!(send.len(), size as usize);
                Exported { rank, send, counts }
            })
            .collect()
    }

    /// Every exported row of every partition, as `(x, z, coeff)`.
    fn all_rows<const W: usize>(exports: &[Exported<W>]) -> Vec<([u64; W], [u64; W], Complex64)> {
        let mut out = Vec::new();
        for e in exports {
            for payload in e.send.iter().flatten() {
                for block in &payload.blocks {
                    for i in 0..block.rows() {
                        out.push((block.x[i], block.z[i], block.coeff[i]));
                    }
                }
            }
        }
        out
    }

    /// Group rows by key into `(count, sum)`.
    fn by_key<const W: usize>(
        rows: impl IntoIterator<Item = ([u64; W], [u64; W], Complex64)>,
    ) -> HashMap<([u64; W], [u64; W]), (usize, Complex64)> {
        let mut map: HashMap<([u64; W], [u64; W]), (usize, Complex64)> = HashMap::new();
        for (x, z, c) in rows {
            let slot = map.entry((x, z)).or_insert((0, ZERO));
            slot.0 += 1;
            slot.1 += c;
        }
        map
    }

    /// The **row-level** oracle: for every input term, the rows
    /// [`Channel::apply`] emits whose output key lands in a different
    /// partition than the term itself.
    ///
    /// Deliberately not `test_support::naive_apply_layer`: that sums a key's
    /// contributions from *every* source, and the export carries only the
    /// partition-crossing ones. Rows are summed per (input term, output key)
    /// first, which is exactly what the prepared PTM tabulates, and
    /// exactly-zero results are dropped the way a zero amplitude emits
    /// nothing.
    fn crossing_rows<const W: usize>(
        input: &PauliSum<W>,
        ch: &dyn Channel<W>,
        adjoint: bool,
        rows: &PartitionRows<W>,
    ) -> Vec<([u64; W], [u64; W], Complex64)> {
        let mf = ch.max_fanout().max(1);
        let mut buf_x = vec![[0u64; W]; mf];
        let mut buf_z = vec![[0u64; W]; mf];
        let mut buf_c = vec![ZERO; mf];
        let mut out = Vec::new();
        for (x, z, c) in input.iter() {
            let src = rows.partition_of(x, z);
            let mut len = 0usize;
            {
                let mut buf = OutputBuffer::<W> {
                    x: &mut buf_x,
                    z: &mut buf_z,
                    coeff: &mut buf_c,
                    len: &mut len,
                };
                if adjoint {
                    ch.apply_adjoint(x, z, c, &mut buf);
                } else {
                    ch.apply(x, z, c, &mut buf);
                }
            }
            let mut per_term: HashMap<([u64; W], [u64; W]), Complex64> = HashMap::new();
            for i in 0..len {
                *per_term.entry((buf_x[i], buf_z[i])).or_insert(ZERO) += buf_c[i];
            }
            for ((kx, kz), kc) in per_term {
                if kc != ZERO && rows.partition_of(&kx, &kz) != src {
                    out.push((kx, kz, kc));
                }
            }
        }
        out
    }

    /// Two terms, one bucket, one remote delta: the export is hand-checkable
    /// row for row.
    ///
    /// `H` on qubit 0 sends `X₀ → Z₀` and `Z₀ → X₀`, so its non-identity delta
    /// is the mask `x₀ z₀`. Under a partition row that reads the `x` bit of
    /// qubit 0, that mask has `part = 1`: the delta is remote, and — because
    /// `part` is linear — `X₀` and `Z₀` necessarily sit in *different*
    /// partitions. Each therefore exports its single image to the other.
    #[test]
    fn a_single_bucket_h_layer_exports_the_swapped_keys() {
        let mut acc = BuildAccumulator::<1>::with_capacity(8, 2);
        acc.add_term(PauliString::<1>::x(0), Phase::ONE, Complex64::new(1.0, 0.0));
        acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(2.0, 0.0));
        let input = acc.finalize();
        let rows = PartitionRows::<1>::from_rows(8, vec![[1u64]], vec![[0u64]]);
        assert_eq!(rows.partition_of(&[1], &[0]), 1, "X₀ is in partition 1");
        assert_eq!(rows.partition_of(&[0], &[1]), 0, "Z₀ is in partition 0");

        // `bits = 0`: one bucket, so every block is a single segment.
        let exports = export_all(&input, &Clifford1Q::h(0), false, 0, 0x51, &rows);
        assert_eq!(exports.len(), 2);

        // Partition 0 holds Z₀ (coeff 2) and ships X₀ to partition 1.
        // Partition 1 holds X₀ (coeff 1) and ships Z₀ to partition 0.
        let expected: [(u32, [u64; 1], [u64; 1], f64); 2] =
            [(1, [1], [0], 2.0), (0, [0], [1], 1.0)];
        for (e, (partner, kx, kz, kc)) in exports.iter().zip(expected) {
            assert_eq!(e.counts.rows_to[partner as usize], 1);
            assert_eq!(e.counts.rows_to.iter().sum::<u64>(), 1, "one row only");
            assert!(e.counts.bytes_to[partner as usize] > 0);
            let payload = e.send[partner as usize]
                .as_ref()
                .unwrap_or_else(|| panic!("rank {} sent nothing to {partner}", e.rank));
            assert_eq!(payload.blocks.len(), 1, "one remote delta");
            let block = &payload.blocks[0];
            // `deltas()` is ascending by `local_delta`, so entry 0 is the
            // identity and the X↔Z swap (local delta `0b11`) is entry 1.
            assert_eq!(block.header.entry, 1);
            assert_eq!(block.num_buckets(), 1);
            assert_eq!(block.rows(), 1);
            let (sx, sz, sc) = block.segment(0);
            assert_eq!((sx, sz), (&[kx][..], &[kz][..]));
            assert!((sc[0] - Complex64::new(kc, 0.0)).norm() < TOL);
        }
    }

    /// A rotation every term commutes with produces no rows — but still
    /// produces its block, because the receiver indexes blocks positionally.
    #[test]
    fn an_all_commuting_rotation_exports_empty_blocks() {
        // Generator Z₀X₂X₄X₆, weight 4 > MAX_LOCAL_SUPPORT (the Rotation arm).
        let gen = {
            let mut g = PauliString::<1>::z(0);
            for q in [2u32, 4, 6] {
                g.mul_assign(&PauliString::<1>::x(q));
            }
            g
        };
        let mut acc = BuildAccumulator::<1>::with_capacity(8, 3);
        for p in [gen, PauliString::<1>::z(1), PauliString::<1>::x(3)] {
            assert!(p.commutes_with(&gen), "fixture term must commute");
            acc.add_term(p, Phase::ONE, Complex64::new(1.5, 0.0));
        }
        let input = acc.finalize();
        // A row on the `x` bit of qubit 2 makes `part(gen) = 1`: the generator
        // pass is remote.
        let rows = PartitionRows::<1>::from_rows(8, vec![[1u64 << 2]], vec![[0u64]]);
        assert_eq!(rows.partition_of(&gen.x, &gen.z), 1);

        let rot = PauliRotation::new(gen, 0.41);
        let exports = export_all(&input, &rot, false, 2, 0x52, &rows);
        assert!(
            all_rows(&exports).is_empty(),
            "commuting terms emit nothing"
        );
        for e in &exports {
            assert!(e.counts.rows_to.iter().all(|&r| r == 0));
            let blocks: Vec<_> = e.send.iter().flatten().flat_map(|p| &p.blocks).collect();
            assert_eq!(blocks.len(), 1, "the empty block is still exported");
            assert_eq!(blocks[0].rows(), 0);
        }
    }

    /// The CSR is in the *receiver's* order: every row of segment `p` lands in
    /// the receiver's bucket `map.bucket_at(p)`.
    ///
    /// This is the contract `RecvRows` reads the block through
    /// (`segment(position_of(β′))`), and the only thing that makes a coset's
    /// rows contiguous — so it is checked directly against the hash rather than
    /// only end to end through the differential nets. The matrix covers a
    /// non-trivial permutation (`bits > 0` with a local delta, so the span has
    /// `r > 0`) and a non-zero bucket delta on the remote entry.
    #[test]
    fn every_segment_holds_the_rows_of_its_destination_bucket() {
        let input = rand_sum::<1>(700, 8, 0x9C7);
        for (_name, ch) in &differential_channels_w1() {
            for &adjoint in &[false, true] {
                for &bits in &[1u8, 3, 4] {
                    for &pbits in &[1u8, 2] {
                        let rows = PartitionRows::<1>::from_seed(8, pbits, 0x1234);
                        let hash = Gf2Hash::<1>::new(8, bits, 0xAB);
                        let whole = input.clone().with_hash(hash);
                        let prep = ch.prepare(whole.hash(), adjoint).expect("prepare");
                        let size = rows.num_partitions() as u32;
                        for rank in 0..size {
                            let local = whole.filter_partition(&rows, rank);
                            let plan = PartitionPlan::new(&prep, &rows, rank);
                            let map = map_for(&plan, local.num_buckets());
                            let mut scratch = ExportScratch::default();
                            let (send, _) =
                                export_layer(&local, &prep, &plan, size, &map, &mut scratch);
                            for payload in send.iter().flatten() {
                                for block in &payload.blocks {
                                    for p in 0..block.num_buckets() {
                                        let want = map.bucket_at(p);
                                        let (sx, sz, _) = block.segment(p);
                                        for (x, z) in sx.iter().zip(sz) {
                                            assert_eq!(
                                                whole.hash().bucket_of(x, z),
                                                want,
                                                "segment {p} of entry {} carries a row for                                                  another bucket",
                                                block.header.entry,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Pass 1 sized every block exactly: the counted row total is what pass 2
    /// wrote, in every block, for every channel class.
    ///
    /// The two passes agreeing is `fill_range`'s own `debug_assert` (this suite
    /// runs in debug), which fires unless pass 2 emitted exactly the rows pass 1
    /// counted for its range. What is checked here is the shape around it: the
    /// CSR offsets end at `header.rows`, the grow-only columns hold at least
    /// that many rows — `segment()` would panic on a block that did not — and
    /// the reported per-partner totals are the blocks' own.
    #[test]
    fn the_counted_rows_are_the_filled_rows() {
        let input = rand_sum::<1>(700, 8, 0x9C0);
        for (name, ch) in &differential_channels_w1() {
            for &adjoint in &[false, true] {
                for &bits in &[0u8, 3] {
                    for &pbits in &[1u8, 2] {
                        let rows = PartitionRows::<1>::from_seed(8, pbits, 0x1234);
                        let exports = export_all(&input, ch.as_ref(), adjoint, bits, 0xAB, &rows);
                        for e in &exports {
                            let mut rows_to = vec![0u64; rows.num_partitions()];
                            for (q, payload) in e.send.iter().enumerate() {
                                let Some(payload) = payload else { continue };
                                for block in &payload.blocks {
                                    let what = format!(
                                        "{name} adjoint={adjoint} bits={bits} p={pbits} \
                                         rank={} entry={}",
                                        e.rank, block.header.entry,
                                    );
                                    // Grow-only columns: at least the rows the
                                    // counts sized, and the CSR offsets end
                                    // exactly there.
                                    assert!(block.coeff.len() >= block.rows(), "{what}: rows");
                                    assert!(block.x.len() >= block.rows(), "{what}: x");
                                    assert!(block.z.len() >= block.rows(), "{what}: z");
                                    assert_eq!(
                                        block.offsets.last().copied(),
                                        Some(block.header.rows),
                                        "{what}: offsets",
                                    );
                                    rows_to[q] += block.rows() as u64;
                                }
                            }
                            assert_eq!(e.counts.rows_to, rows_to, "{name}: reported rows");
                        }
                    }
                }
            }
        }
    }

    /// The union of every partition's export is exactly the layer's
    /// partition-crossing rows, key by key, count and sum.
    #[test]
    fn the_export_is_the_partition_crossing_rows_w1() {
        let input = rand_sum::<1>(700, 8, 0x9C1);
        for (name, ch) in &differential_channels_w1() {
            for &adjoint in &[false, true] {
                for &pbits in &[1u8, 2] {
                    for &seed in &[0x2222u64, 0x7777] {
                        let rows = PartitionRows::<1>::from_seed(8, pbits, seed);
                        let want = by_key(crossing_rows(&input, ch.as_ref(), adjoint, &rows));
                        for &bits in &[0u8, 4] {
                            let exports =
                                export_all(&input, ch.as_ref(), adjoint, bits, 0xAB, &rows);
                            let got = by_key(all_rows(&exports));
                            let what = format!(
                                "{name} adjoint={adjoint} bits={bits} p={pbits} seed={seed:x}"
                            );
                            assert_eq!(got.len(), want.len(), "{what}: distinct keys");
                            for (key, (n, sum)) in &want {
                                let (gn, gsum) = got
                                    .get(key)
                                    .unwrap_or_else(|| panic!("{what}: missing key {key:?}"));
                                assert_eq!(gn, n, "{what}: row count for {key:?}");
                                assert!(
                                    (gsum - sum).norm() < TOL,
                                    "{what}: coeff for {key:?}: {gsum} vs {sum}",
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The same, at `W = 2`: wide keys, word-boundary supports.
    #[test]
    fn the_export_is_the_partition_crossing_rows_w2() {
        let input = rand_sum_real::<2>(800, 128, 0x9C2);
        for (name, ch) in &differential_channels_w2() {
            for &adjoint in &[false, true] {
                let rows = PartitionRows::<2>::from_seed(128, 2, 0x3333);
                let want = by_key(crossing_rows(&input, ch.as_ref(), adjoint, &rows));
                for &bits in &[2u8, 5] {
                    let exports = export_all(&input, ch.as_ref(), adjoint, bits, 0xCD, &rows);
                    let got = by_key(all_rows(&exports));
                    let what = format!("{name} adjoint={adjoint} bits={bits}");
                    assert_eq!(got.len(), want.len(), "{what}: distinct keys");
                    for (key, (n, sum)) in &want {
                        let (gn, gsum) = got
                            .get(key)
                            .unwrap_or_else(|| panic!("{what}: missing key {key:?}"));
                        assert_eq!(gn, n, "{what}: row count");
                        assert!((gsum - sum).norm() < TOL, "{what}: coeff {gsum} vs {sum}");
                    }
                }
            }
        }
    }

    /// Every exported row is addressed to the partition it belongs to — the
    /// property [`debug_assert_exported_partitions`] pins in debug builds,
    /// asserted here unconditionally.
    #[test]
    fn exported_rows_are_addressed_to_their_own_partition() {
        let input = rand_sum::<1>(400, 8, 0x9C3);
        for (name, ch) in &differential_channels_w1() {
            let rows = PartitionRows::<1>::from_seed(8, 2, 0x4444);
            for exported in export_all(&input, ch.as_ref(), false, 3, 0xAB, &rows) {
                for (q, payload) in exported.send.iter().enumerate() {
                    let Some(payload) = payload else { continue };
                    for block in &payload.blocks {
                        for i in 0..block.rows() {
                            assert_eq!(
                                rows.partition_of(&block.x[i], &block.z[i]),
                                q as u32,
                                "{name}: row {i} of entry {} is misaddressed",
                                block.header.entry,
                            );
                        }
                    }
                }
            }
        }
    }

    /// A partitioning under which no delta crosses exports nothing at all,
    /// with no blocks and no payload allocated.
    #[test]
    fn a_layer_with_no_remote_delta_exports_nothing() {
        let input = rand_sum::<1>(200, 8, 0x9C4);
        // `h(3)`'s only non-identity delta is the mask `x₃ z₃`; a partition
        // row reading qubit 0 alone cannot see it.
        let rows = PartitionRows::<1>::from_rows(8, vec![[1u64]], vec![[0u64]]);
        let exports = export_all(&input, &Clifford1Q::h(3), false, 3, 0x55, &rows);
        for e in &exports {
            assert!(e.send.iter().all(Option::is_none));
            assert_eq!(e.counts.rows_to, vec![0, 0]);
            assert_eq!(e.counts.bytes_to, vec![0, 0]);
        }
    }

    /// The scratch is reusable: a layer through a scratch a wider layer has
    /// already grown gives the same payload as one through a fresh scratch.
    #[test]
    fn a_reused_scratch_gives_the_same_export() {
        let input = rand_sum::<1>(500, 8, 0x9C5);
        let rows = PartitionRows::<1>::from_seed(8, 2, 0x5555);
        let hash = Gf2Hash::<1>::new(8, 4, 0xAB);
        let whole = input.with_hash(hash);
        let mut scratch = ExportScratch::default();

        // A wide-fanout layer first, so the scratch is left large.
        let channels = differential_channels_w1();
        let (_, wide) = channels
            .iter()
            .find(|(n, _)| *n == "haar_su4")
            .expect("the dense SU(4) cell");
        let prep = wide.prepare(whole.hash(), false).unwrap();
        let plan = PartitionPlan::new(&prep, &rows, 0);
        let wide_local = whole.filter_partition(&rows, 0);
        let map = map_for(&plan, wide_local.num_buckets());
        let _ = export_layer(&wide_local, &prep, &plan, 4, &map, &mut scratch);

        let h = Clifford1Q::h(3);
        let prep = Channel::<1>::prepare(&h, whole.hash(), false).unwrap();
        let plan = PartitionPlan::new(&prep, &rows, 0);
        let local = whole.filter_partition(&rows, 0);
        let map = map_for(&plan, local.num_buckets());
        let (reused, counts_reused) = export_layer(&local, &prep, &plan, 4, &map, &mut scratch);
        let (fresh, counts_fresh) =
            export_layer(&local, &prep, &plan, 4, &map, &mut ExportScratch::default());
        assert_eq!(counts_reused, counts_fresh);
        assert_eq!(reused.len(), fresh.len());
        for (a, b) in reused.iter().zip(&fresh) {
            assert_eq!(a, b, "a reused scratch changed the payload");
        }
    }
}
