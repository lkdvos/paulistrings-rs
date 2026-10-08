use super::premerge::SCAN_LIMIT;
use super::receive::{chunk_layout, chunk_max, recv_chunks, rows_between};
use super::*;
use crate::channel::clifford::Clifford2Q;
use crate::channel::{Channel, GeneralUnitary2Q};
use crate::engine::coset::Gf2Span;
use crate::engine::gpu::layer::GpuLayerOptions;
use crate::engine::gpu::payload::DeviceBlock;
use crate::engine::gpu::DEFAULT_ARENA_BYTES;
use crate::engine::partitioned::export::{export_layer, ExportScratch};
use crate::engine::partitioned::transport::{ExchangeBlock, PartnerPayload};
use crate::pauli_sum::hash::{Gf2Hash, PartitionRows, B_MAX_BITS};
use crate::pauli_sum::PauliSum;
use crate::test_support::{haar_su4_matrix, rand_sum, zz_rotation};
use num_complex::Complex64;

/// The three channels of the export fixtures: a dense SU(4), a Clifford and a rotation.
fn fixture_channels<const W: usize>() -> Vec<(&'static str, Box<dyn Channel<W>>)> {
    vec![
        (
            "su4",
            Box::new(GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix())),
        ),
        ("cnot", Box::new(Clifford2Q::cnot(1, 3))),
        ("zz", Box::new(zz_rotation::<W>(0, 2, 0.3))),
    ]
}

fn su4<const W: usize>() -> GeneralUnitary2Q {
    GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix())
}

/// Every support pattern of qubits 0 and 1 under each of `base`'s keys, so an SU(4) on `(0, 1)` makes one partner's remote rows collide.
fn dense_on_01<const W: usize>(base: &PauliSum<W>) -> PauliSum<W> {
    let mut accumulator =
        crate::pauli_sum::accumulator::BuildAccumulator::<W>::new(base.num_qubits());
    for (x, z, c) in base.iter() {
        for s in 0..16u64 {
            let (mut x, mut z) = (*x, *z);
            x[0] = (x[0] & !0b11) | (s & 0b11);
            z[0] = (z[0] & !0b11) | (s >> 2);
            accumulator.add_term(
                crate::pauli_string::PauliString::<W> { x, z },
                crate::phase::Phase::ONE,
                c * (1.0 + s as f64),
            );
        }
    }
    accumulator.finalize()
}

/// One rank's layer: the host's `export_layer` and the device's `export_blocks_device` of the same share.
struct Exported<const W: usize> {
    plan: PartitionPlan,
    nb: usize,
    want: Vec<Option<PartnerPayload<W>>>,
    want_rows: Vec<u64>,
    dev: GpuSum<W>,
    scratch: LayerScratch<W>,
    got: Vec<Option<DevicePayload<W>>>,
    got_counts: LayerExchangeCounts,
}

/// Rank `rank`'s share of `input` under `rows` through `ch`, exported on the host and on the device (sender-side merge `premerge`, arena `arena_bytes`, NVRTC options `nvrtc`); `None` when the rank has no remote delta.
fn export_rank<const W: usize>(
    input: &PauliSum<W>,
    rows: &PartitionRows<W>,
    channel: &dyn Channel<W>,
    rank: u32,
    (premerge, arena_bytes, nvrtc): (bool, usize, &[String]),
) -> Option<Exported<W>> {
    let size = rows.num_partitions() as u32;
    let local = input.filter_partition(rows, rank);
    let prepared = channel.prepare(local.hash(), false).expect("prepared");
    let plan = PartitionPlan::new(&prepared, rows, rank);
    if !plan.has_remote() {
        return None;
    }
    let nb = local.num_buckets();
    let mut map = ChunkMap::default();
    map.rebuild(
        &Gf2Span::new(&plan.local_bucket_deltas, local.hash().bits()),
        nb,
        1,
    );
    let (want, want_counts) = export_layer(
        &local,
        &prepared,
        &plan,
        size,
        &map,
        &mut ExportScratch::default(),
    );
    let dev = GpuSum::from_host_with_options(&local, 0, nvrtc).expect("upload");
    let options = GpuLayerOptions {
        premerge,
        arena_bytes,
        ..GpuLayerOptions::default()
    };
    let mut scratch = LayerScratch::new(&dev, options).expect("scratch");
    let table = DevicePrepared::new(&prepared, dev.hash(), &dev.fingerprints, &plan.remote);
    scratch.upload_table(&dev, &table).expect("table");
    scratch.count_local(&dev, &table).expect("count");
    let (got, got_counts) =
        export_blocks_device(&dev, &table, &plan, &mut scratch, size).expect("export");
    Some(Exported {
        plan,
        nb,
        want,
        want_rows: want_counts.rows_to,
        dev,
        scratch,
        got,
        got_counts,
    })
}

/// A device block's columns on the host.
fn download<const W: usize>(
    s: &Arc<CudaStream>,
    block: &DeviceBlock<W>,
) -> (Vec<u64>, Vec<u64>, Vec<f64>) {
    let n = block.rows();
    let x = s.clone_dtoh(&block.x.slice(0..n * W)).unwrap();
    let z = s.clone_dtoh(&block.z.slice(0..n * W)).unwrap();
    let c = s.clone_dtoh(&block.c.slice(0..2 * n)).unwrap();
    s.synchronize().unwrap();
    (x, z, c)
}

/// The device payloads equal the host payloads bitwise: presence, headers, offsets and the three columns.
fn assert_payloads_eq<const W: usize>(e: &Exported<W>, what: &str) {
    assert_eq!(e.got_counts.rows_sent, e.want_rows, "{what}: rows");
    assert_eq!(e.got.len(), e.want.len());
    for (q, (g, w)) in e.got.iter().zip(&e.want).enumerate() {
        match (g, w) {
            (None, None) => {}
            (Some(g), Some(w)) => {
                assert_eq!(g.blocks.len(), w.blocks.len(), "{what}: blocks to {q}");
                for (j, (gb, wb)) in g.blocks.iter().zip(&w.blocks).enumerate() {
                    assert_eq!(gb.header, wb.header, "{what}: header {j} to {q}");
                    assert_eq!(gb.offsets, wb.offsets, "{what}: offsets {j} to {q}");
                    let (x, z, c) = download(&e.dev.stream, gb);
                    let (wx, wz, wc) = wb.cols();
                    assert_eq!(x, wx.as_flattened(), "{what}: x {j} to {q}");
                    assert_eq!(z, wz.as_flattened(), "{what}: z {j} to {q}");
                    assert_eq!(
                        c,
                        bytemuck::cast_slice::<Complex64, f64>(wc),
                        "{what}: coeff {j} to {q}"
                    );
                }
            }
            _ => panic!("{what}: payload presence to {q} differs"),
        }
    }
}

/// Unmerged, the device payloads equal `export_layer`'s blocks bitwise, and the SU(4) ships several blocks to one partner.
fn device_export_matches_host<const W: usize>(num_qubits: usize, seed: u64, pbits: u8) {
    let input = rand_sum::<W>(6000, num_qubits, seed);
    let rows = PartitionRows::<W>::from_seed(num_qubits, pbits, seed ^ 0x77);
    let (mut exported, mut multi_block_payloads) = (0usize, 0);
    for (name, channel) in &fixture_channels::<W>() {
        for rank in 0..rows.num_partitions() as u32 {
            let off = (false, DEFAULT_ARENA_BYTES, &[][..]);
            let Some(e) = export_rank(&input, &rows, channel.as_ref(), rank, off) else {
                continue;
            };
            assert_payloads_eq(&e, &format!("W={W} {name} rank {rank}"));
            let blocks = e.got.iter().flatten().map(|p| p.blocks.len());
            multi_block_payloads += blocks.clone().filter(|&n| n >= 2).count();
            exported += blocks.sum::<usize>();
        }
    }
    assert!(exported > 0, "the fixture must export something");
    assert!(
        multi_block_payloads > 0,
        "the SU(4) layer must ship several remote deltas to one partner"
    );
}

#[test]
fn device_payloads_match_export_layer_bitwise() {
    crate::require_cuda!();
    device_export_matches_host::<1>(12, 0xE7, 1);
    device_export_matches_host::<1>(12, 0xE8, 2);
    device_export_matches_host::<2>(100, 0xE9, 1);
    device_export_matches_host::<2>(100, 0xEA, 2);
}

/// Two blocks over eight positions receiving `[1, 0, 2, 4, 0, 3, 1, 1]` rows: each cap takes the fewest power-of-two chunks that fit it, and one position alone may exceed it.
#[test]
fn the_receive_takes_the_fewest_power_of_two_chunks_under_its_cap() {
    let off: Vec<u32> = [[0u32, 1, 1, 3, 3, 3, 6, 7, 8], [0, 0, 0, 0, 4, 4, 4, 4, 4]].concat();
    let b = 8;
    assert_eq!(rows_between(&off, b, 0, 8), 12);
    assert_eq!(rows_between(&off, b, 3, 4), 4);
    assert_eq!(
        (0..=3).map(|j| chunk_max(&off, b, j)).collect::<Vec<_>>(),
        vec![12, 7, 6, 4],
        "a finer power-of-two cut never grows the largest chunk"
    );
    assert_eq!(recv_chunks(&off, b, usize::MAX), (0, 12));
    assert_eq!(recv_chunks(&off, b, 12), (0, 12));
    assert_eq!(recv_chunks(&off, b, 7), (1, 7));
    assert_eq!(recv_chunks(&off, b, 6), (2, 6));
    assert_eq!(recv_chunks(&off, b, 3), (3, 4));
    assert_eq!(
        recv_chunks(&[0, 5], 1, 1),
        (0, 5),
        "one position is one chunk"
    );
    let mut base = Vec::new();
    assert_eq!(chunk_layout(&off, b, 2, 4, &mut base), 6);
    assert_eq!(base[0].wrapping_add(off[2]), 0);
    assert_eq!(base[1].wrapping_add(off[9 + 3]), 2);
    assert_eq!(base[1].wrapping_add(off[9 + 4]), 6);
}

/// Every rank learns one verdict and the largest chunk count any ready rank asked for; one rank not ready is a no everywhere.
#[test]
fn the_vote_agrees_the_largest_chunk_count() {
    use crate::engine::partitioned::transport::{Collectives, InProcessTransport};
    let run = |asks: [Option<u8>; 4]| -> Vec<Option<u8>> {
        let group = InProcessTransport::group(4);
        std::thread::scope(|s| {
            let hs: Vec<_> = group
                .into_iter()
                .map(|t| s.spawn(move || vote(&t, asks[t.rank() as usize])))
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        })
    };
    assert_eq!(run([Some(1), Some(3), Some(0), Some(2)]), vec![Some(3); 4]);
    assert_eq!(run([Some(0); 4]), vec![Some(0); 4]);
    assert_eq!(run([Some(2), None, Some(0), Some(5)]), vec![None; 4]);
}

/// A partition with no terms still ships one empty block per remote delta, the host's headers and offsets.
#[test]
fn an_empty_partition_ships_empty_blocks() {
    crate::require_cuda!();
    let seed = 0xE55u64;
    let input = PauliSum::<1>::empty_with_hash(12, Gf2Hash::<1>::new(12, 3, seed));
    let rows = PartitionRows::<1>::from_seed(12, 1, seed ^ 0x77);
    let off = (false, DEFAULT_ARENA_BYTES, &[][..]);
    let e = export_rank(&input, &rows, &su4::<1>(), 0, off).expect("the SU(4) must cross");
    assert_payloads_eq(&e, "empty");
    let payload = e.got[1].as_ref().expect("a device payload for the partner");
    assert_eq!(payload.blocks.len(), e.plan.remote.len());
    assert!(payload
        .blocks
        .iter()
        .all(|b| b.rows() == 0 && b.offsets.len() == 9));
}

type Key<const W: usize> = ([u64; W], [u64; W]);

/// Per destination position, one partner's rows keyed by `(x, z)` with their coefficients summed across the partner's blocks.
type PositionSums<const W: usize> = Vec<std::collections::BTreeMap<Key<W>, Complex64>>;

/// One block's `(offsets, x, z, coeff)`, live rows only.
type BlockCols<const W: usize> = (Vec<u32>, Vec<[u64; W]>, Vec<[u64; W]>, Vec<Complex64>);

/// The per-position key sums of `blocks`, and the largest number of times one key occurs at one position.
fn position_sums<const W: usize>(blocks: &[BlockCols<W>], b: usize) -> (PositionSums<W>, usize) {
    let mut out: PositionSums<W> = vec![Default::default(); b];
    let mut seen: Vec<std::collections::BTreeMap<Key<W>, usize>> = vec![Default::default(); b];
    let mut most = 0;
    for (off, x, z, c) in blocks {
        for p in 0..b {
            for i in off[p] as usize..off[p + 1] as usize {
                *out[p]
                    .entry((x[i], z[i]))
                    .or_insert(Complex64::new(0.0, 0.0)) += c[i];
                let n = seen[p].entry((x[i], z[i])).or_insert(0);
                *n += 1;
                most = most.max(*n);
            }
        }
    }
    (out, most)
}

/// Two per-position key sums agree to `tol`, a key absent on one side counting as zero.
fn assert_position_sums_close<const W: usize>(
    got: &PositionSums<W>,
    want: &PositionSums<W>,
    tol: f64,
    what: &str,
) {
    let zero = Complex64::new(0.0, 0.0);
    for (p, (g, w)) in got.iter().zip(want).enumerate() {
        for (k, gc) in g {
            let wc = w.get(k).copied().unwrap_or(zero);
            assert!(
                (gc - wc).norm() <= tol,
                "{what}: position {p} key {k:?}: {gc} vs {wc}"
            );
        }
        for (k, wc) in w {
            if !g.contains_key(k) {
                assert!(
                    wc.norm() <= tol,
                    "{what}: position {p} lost key {k:?} ({wc})"
                );
            }
        }
    }
}

fn host_block_cols<const W: usize>(b: &ExchangeBlock<W>) -> BlockCols<W> {
    let (x, z, c) = b.cols();
    (b.offsets.clone(), x.to_vec(), z.to_vec(), c.to_vec())
}

fn device_block_cols<const W: usize>(s: &Arc<CudaStream>, b: &DeviceBlock<W>) -> BlockCols<W> {
    let (x, z, c) = download(s, b);
    let x: Vec<[u64; W]> = x.chunks(W).map(|w| w.try_into().unwrap()).collect();
    let z: Vec<[u64; W]> = z.chunks(W).map(|w| w.try_into().unwrap()).collect();
    let c: Vec<Complex64> = c.chunks(2).map(|p| Complex64::new(p[0], p[1])).collect();
    (b.offsets.clone(), x, z, c)
}

/// With the sender-side merge on, the device payloads carry per position what `export_layer` carries summed by key, no key twice across one partner's blocks, and no segment longer than its unmerged one.
/// Returns `(rows sent, unmerged rows, fallback blocks)` per channel name, summed over ranks.
fn premerge_matches_host_by_key<const W: usize>(
    input: &PauliSum<W>,
    rows: &PartitionRows<W>,
    arena_bytes: usize,
    nvrtc: &[String],
) -> Vec<(&'static str, u64, u64, (u32, u32))> {
    let mut totals = Vec::new();
    for (name, channel) in &fixture_channels::<W>() {
        let (mut sent, mut unmerged, mut fallbacks) = (0u64, 0u64, (0u32, 0u32));
        for rank in 0..rows.num_partitions() as u32 {
            let on = (true, arena_bytes, nvrtc);
            let Some(e) = export_rank(input, rows, channel.as_ref(), rank, on) else {
                continue;
            };
            let (nb, what) = (
                e.nb,
                format!("W={W} {name} rank {rank} arena={arena_bytes}"),
            );
            for (q, w) in e.want.iter().enumerate() {
                let (Some(w), Some(g)) = (w, &e.got[q]) else {
                    assert!(w.is_none() && e.got[q].is_none(), "{what}: presence to {q}");
                    continue;
                };
                assert_eq!(g.blocks.len(), w.blocks.len(), "{what}: blocks to {q}");
                let g: Vec<BlockCols<W>> = g
                    .blocks
                    .iter()
                    .map(|b| {
                        assert_eq!(b.header.num_buckets as usize, nb);
                        assert_eq!(b.header.rows, b.offsets[nb]);
                        device_block_cols(&e.dev.stream, b)
                    })
                    .collect();
                for (j, (gb, wb)) in g.iter().zip(&w.blocks).enumerate() {
                    assert_eq!(gb.0.len(), nb + 1, "{what}: offsets {j} to {q}");
                    for p in 0..nb {
                        assert!(
                            gb.0[p + 1] - gb.0[p] <= wb.offsets[p + 1] - wb.offsets[p],
                            "{what}: block {j} to {q} grew at position {p}"
                        );
                    }
                }
                let wcols: Vec<_> = w.blocks.iter().map(host_block_cols).collect();
                let (want_sums, _) = position_sums(&wcols, nb);
                let (got_sums, most) = position_sums(&g, nb);
                assert_position_sums_close(&got_sums, &want_sums, 1e-12, &what);
                if w.blocks.len() >= 2 && *name == "su4" {
                    assert_eq!(most, 1, "{what}: a key twice across the blocks to {q}");
                }
            }
            let shipped: usize = e
                .got
                .iter()
                .flatten()
                .flat_map(|p| &p.blocks)
                .map(DeviceBlock::rows)
                .sum();
            assert_eq!(
                e.got_counts.rows_sent.iter().sum::<u64>(),
                shipped as u64,
                "{what}: counted rows"
            );
            sent += shipped as u64;
            unmerged += e.want_rows.iter().sum::<u64>();
            fallbacks.0 += e.scratch.counters.fallback_hi;
            fallbacks.1 += e.scratch.counters.fallback_key;
        }
        totals.push((*name, sent, unmerged, fallbacks));
    }
    totals
}

fn premerge_fixture<const W: usize>(num_qubits: usize, seed: u64, pbits: u8) {
    let input = dense_on_01(&rand_sum::<W>(400, num_qubits, seed));
    let rows = PartitionRows::<W>::from_seed(num_qubits, pbits, seed ^ 0x77);
    // The SU(4) ships under half its unmerged rows; the Clifford and the rotation have nothing to merge.
    let check_shrink = |name: &str, sent: u64, unmerged: u64| {
        if name == "su4" {
            assert!(
                sent * 2 < unmerged,
                "P={}: su4 sent {sent} of {unmerged} rows",
                1 << pbits
            );
        } else {
            assert_eq!(sent, unmerged, "{name}: nothing to merge");
        }
    };
    for arena in [DEFAULT_ARENA_BYTES, 1] {
        for (name, sent, unmerged, fallbacks) in
            premerge_matches_host_by_key(&input, &rows, arena, &[])
        {
            check_shrink(name, sent, unmerged);
            assert_eq!(
                fallbacks,
                (0, 0),
                "{name}: a 64-bit fingerprint needs no fallback"
            );
        }
    }
    // Every merge block takes a fallback: the `g_hi32` passes with the low word cleared, the full-key sort with no fingerprint at all.
    for (opt, full_key) in [("-DFP_ZERO_LO", false), ("-DFP_BITS=0", true)] {
        let nvrtc = [opt.to_string()];
        for (name, sent, unmerged, fallbacks) in
            premerge_matches_host_by_key(&input, &rows, DEFAULT_ARENA_BYTES, &nvrtc)
        {
            check_shrink(name, sent, unmerged);
            if name == "su4" {
                let taken = if full_key { fallbacks.1 } else { fallbacks.0 };
                assert!(taken > 0, "{opt}: su4 merged without the expected fallback");
            }
        }
    }
}

#[test]
fn premerged_blocks_match_export_layer_by_key_and_shrink() {
    crate::require_cuda!();
    premerge_fixture::<1>(12, 0xF7, 1);
    premerge_fixture::<1>(12, 0xF8, 2);
    premerge_fixture::<2>(100, 0xF9, 1);
    premerge_fixture::<2>(100, 0xFA, 2);
}

/// Two terms whose remote rows to one key cancel exactly: the merged export drops the key, the unmerged one ships both rows.
/// Unit coefficients against equal-magnitude amplitudes keep every product exact whatever the kernel's FMA contraction.
#[test]
fn exactly_cancelling_remote_rows_are_not_shipped() {
    crate::require_cuda!();
    use crate::channel::prepared::Prepared;
    use crate::test_support::sqrt_swap_matrix;
    let nq = 8;
    let hash = crate::pauli_sum::hash::Gf2Hash::<1>::new(nq, 2, 0xCA);
    // The row reads x on qubit 0, so a delta flipping qubit 0's x-bit is remote.
    let rows = PartitionRows::<1>::from_rows(nq, vec![[0b1u64]], vec![[0u64]]);
    let channel = GeneralUnitary2Q::from_matrix(0, 1, sqrt_swap_matrix());
    let prepared = channel.prepare(&hash, false).expect("prepared");
    let Prepared::Local(ptm) = &prepared else {
        unreachable!()
    };
    let plan = PartitionPlan::new(&prepared, &rows, 0);
    let d = ptm.deltas();
    let zero = Complex64::new(0.0, 0.0);
    let mut pick = None;
    'outer: for (i, a) in plan.remote.iter().enumerate() {
        for b in &plan.remote[i + 1..] {
            let (da, db) = (&d[a.entry], &d[b.entry]);
            for sa in (0..16usize).step_by(2) {
                let sb = sa ^ (da.local_delta ^ db.local_delta) as usize;
                let (aa, ab) = (da.amp[sa], db.amp[sb]);
                if sb & 1 == 0 && aa != zero && (aa == ab || aa == -ab) {
                    pick = Some((a.entry, sa, b.entry, sb));
                    break 'outer;
                }
            }
        }
    }
    let (ea, sa, eb, sb) =
        pick.expect("sqrt(SWAP) has two colliding remote entries of equal magnitude");
    let key = |s: usize| {
        let x = (s & 1) as u64 | (((s >> 2) & 1) as u64) << 1 | 1 << 5;
        let z = ((s >> 1) & 1) as u64 | (((s >> 3) & 1) as u64) << 1;
        crate::pauli_string::PauliString::<1> { x: [x], z: [z] }
    };
    let (aa, ab) = (d[ea].amp[sa], d[eb].amp[sb]);
    let cb = if aa == ab { -1.0 } else { 1.0 };
    let mut accumulator = crate::pauli_sum::accumulator::BuildAccumulator::<1>::new(nq);
    accumulator.add_term(key(sa), crate::phase::Phase::ONE, Complex64::new(1.0, 0.0));
    accumulator.add_term(key(sb), crate::phase::Phase::ONE, Complex64::new(cb, 0.0));
    let local = accumulator.finalize().with_hash(hash.clone());
    let (ka, kb) = (key(sa), key(sb));
    assert_eq!(rows.partition_of(&ka.x, &ka.z), 0);
    assert_eq!(rows.partition_of(&kb.x, &kb.z), 0);
    let (tx, tz) = (ka.x[0] ^ d[ea].mask_x[0], ka.z[0] ^ d[ea].mask_z[0]);
    assert_eq!(
        (tx, tz),
        (kb.x[0] ^ d[eb].mask_x[0], kb.z[0] ^ d[eb].mask_z[0])
    );
    let dev = GpuSum::from_host(&local, 0).expect("upload");
    let table = DevicePrepared::new(&prepared, &hash, &dev.fingerprints, &plan.remote);
    let shipped = |premerge: bool| {
        let options = GpuLayerOptions {
            premerge,
            ..GpuLayerOptions::default()
        };
        let mut scratch = LayerScratch::new(&dev, options).expect("scratch");
        scratch.upload_table(&dev, &table).expect("table");
        scratch.count_local(&dev, &table).expect("count");
        let (got, counts) =
            export_blocks_device(&dev, &table, &plan, &mut scratch, 2).expect("export");
        let keys: Vec<(u64, u64)> = got[1]
            .as_ref()
            .unwrap()
            .blocks
            .iter()
            .flat_map(|b| {
                let (x, z, _) = download(&dev.stream, b);
                x.into_iter().zip(z).collect::<Vec<_>>()
            })
            .collect();
        (keys, counts.rows_sent[1])
    };
    let (plain, plain_rows) = shipped(false);
    assert_eq!(plain.iter().filter(|k| **k == (tx, tz)).count(), 2);
    let (merged, merged_rows) = shipped(true);
    assert!(!merged.contains(&(tx, tz)), "the cancelled key travelled");
    assert!(
        merged_rows + 2 <= plain_rows,
        "{merged_rows} vs {plain_rows}"
    );
}

/// With the merge on, every rank of `input` under `rows` through an SU(4) falls back to the unmerged export, bitwise `export_layer`'s.
fn assert_unmerged_fallback(input: &PauliSum<1>, rows: &PartitionRows<1>, nvrtc: &[String]) {
    let mut remote_layers = 0;
    for rank in 0..rows.num_partitions() as u32 {
        let on = (true, DEFAULT_ARENA_BYTES, nvrtc);
        let Some(e) = export_rank(input, rows, &su4::<1>(), rank, on) else {
            continue;
        };
        remote_layers += 1;
        assert_eq!(
            e.scratch.counters.rows_premerged, 0,
            "rank {rank}: no rows were premerged"
        );
        assert_payloads_eq(&e, &format!("rank {rank}"));
    }
    assert!(remote_layers > 0, "the fixture must export something");
}

/// A source bucket longer than [`MAX_BUCKET_LEN`] clears every merge group, the tag-overflow guard; one bucket of 12000 rows stays over the limit after the split halves it.
#[test]
fn oversize_source_bucket_falls_back_to_unmerged_export() {
    crate::require_cuda!();
    let input = rand_sum::<1>(12_000, 12, 0xB16).with_hash(Gf2Hash::<1>::new(12, 0, 0xB16));
    let rows = PartitionRows::<1>::from_seed(12, 1, 0xB16 ^ 0x77);
    assert!(input.filter_partition(&rows, 0).bucket(0).2.len() > MAX_BUCKET_LEN);
    assert_unmerged_fallback(&input, &rows, &[]);
}

/// A merge position whose records exceed the fused kernel's record cap: `premerge_partner` writes nothing and returns `None`.
/// `-DTEST_SHARED_LIMIT` shrinks the cap to the smallest variant so a bucket under [`MAX_BUCKET_LEN`] exceeds it.
#[test]
fn oversize_merge_position_falls_back_to_unmerged_export() {
    crate::require_cuda!();
    let input = rand_sum::<1>(6000, 12, 0xCA9).with_hash(Gf2Hash::<1>::new(12, 0, 0xCA9));
    let rows = PartitionRows::<1>::from_seed(12, 1, 0xCA9 ^ 0x77);
    assert!(input.filter_partition(&rows, 0).bucket(0).2.len() <= MAX_BUCKET_LEN);
    assert_unmerged_fallback(&input, &rows, &["-DTEST_SHARED_LIMIT=17000".to_string()]);
}

/// A partner group whose `entries × buckets` exceeds [`SCAN_LIMIT`] is cleared before K3: `B_MAX_BITS` buckets times the SU(4)'s eight-entry group already does.
#[test]
fn oversize_partner_group_falls_back_to_unmerged_export() {
    crate::require_cuda!();
    let base = rand_sum::<1>(400, 32, 0xB5CA);
    let input = dense_on_01(&base).with_hash(Gf2Hash::<1>::new(32, B_MAX_BITS, 0xB5CA));
    assert!(input.num_buckets() * 8 > SCAN_LIMIT);
    assert_unmerged_fallback(
        &input,
        &PartitionRows::<1>::from_seed(32, 1, 0xB5CA ^ 0x77),
        &[],
    );
}
