use super::*;
use crate::channel::clifford::Clifford1Q;
use crate::channel::rotation::PauliRotation;
use crate::channel::{Channel, OutputBuffer};
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::pauli_sum::hash::{Gf2Hash, PartitionRows};
use crate::phase::Phase;
use crate::test_support::{
    differential_channels_w1, differential_channels_w2, rand_sum, rand_sum_real,
};
use std::collections::HashMap;

const TOL: f64 = 1e-12;

/// The destination-coset order for `plan` over `num_buckets` buckets, in one chunk: the layout every export test compares against.
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

/// The **row-level** oracle: for every input term, the rows [`Channel::apply`] emits whose output key lands in a different partition than the term itself.
///
/// Deliberately not `test_support::naive_apply_layer`, which sums a key's contributions from *every* source; the export carries only the partition-crossing ones.
/// Rows are summed per (input term, output key) first, and exactly-zero results are dropped the way a zero amplitude emits nothing.
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

/// Two terms, one bucket, one remote delta: the export is hand-checkable row for row.
///
/// `H` on qubit 0 sends `X₀ → Z₀` and `Z₀ → X₀`, so its non-identity delta is the mask `x₀ z₀`.
/// Under a partition row that reads the `x` bit of qubit 0, that mask has `part = 1`, so the delta is remote and `X₀`/`Z₀` sit in different partitions; each exports its single image to the other.
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
    let expected: [(u32, [u64; 1], [u64; 1], f64); 2] = [(1, [1], [0], 2.0), (0, [0], [1], 1.0)];
    for (e, (partner, kx, kz, kc)) in exports.iter().zip(expected) {
        assert_eq!(e.counts.rows_to[partner as usize], 1);
        assert_eq!(e.counts.rows_to.iter().sum::<u64>(), 1, "one row only");
        assert!(e.counts.bytes_to[partner as usize] > 0);
        let payload = e.send[partner as usize]
            .as_ref()
            .unwrap_or_else(|| panic!("rank {} sent nothing to {partner}", e.rank));
        assert_eq!(payload.blocks.len(), 1, "one remote delta");
        let block = &payload.blocks[0];
        // `deltas()` is ascending by `local_delta`, so entry 0 is the identity and the X↔Z swap (local delta `0b11`) is entry 1.
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
    // A row on the `x` bit of qubit 2 makes `part(gen) = 1`: the generator pass is remote.
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

/// The CSR is in the *receiver's* order: every row of segment `p` lands in the receiver's bucket `map.bucket_at(p)`.
///
/// This is the contract `RecvRows` reads the block through, so it is checked directly against the hash rather than only end to end through the differential nets.
/// The matrix covers a non-trivial permutation and a non-zero bucket delta on the remote entry.
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

/// Pass 1 sized every block exactly: the counted row total is what pass 2 wrote, in every block, for every channel class.
///
/// The two passes agreeing is `fill_range`'s own `debug_assert` (this suite runs in debug).
/// What is checked here is the shape around it: the CSR offsets end at `header.rows`, the grow-only columns hold at least that many rows, and the reported per-partner totals are the blocks' own.
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
                                // Grow-only columns: at least the rows the counts sized, and the CSR offsets end exactly there.
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
                        let exports = export_all(&input, ch.as_ref(), adjoint, bits, 0xAB, &rows);
                        let got = by_key(all_rows(&exports));
                        let what =
                            format!("{name} adjoint={adjoint} bits={bits} p={pbits} seed={seed:x}");
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

/// Every exported row is addressed to the partition it belongs to — the property [`debug_assert_exported_partitions`] pins in debug builds, asserted here unconditionally.
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

/// A partitioning under which no delta crosses exports nothing at all, with no blocks and no payload allocated.
#[test]
fn a_layer_with_no_remote_delta_exports_nothing() {
    let input = rand_sum::<1>(200, 8, 0x9C4);
    // `h(3)`'s only non-identity delta is the mask `x₃ z₃`; a partition row reading qubit 0 alone cannot see it.
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
