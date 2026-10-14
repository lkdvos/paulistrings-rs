//! Partition-row choice: the generator scan, the `cut` rows, and the hash re-seed.

use paulistrings::test_support::{
    circuit_generators, count_remote_deltas, with_hash, GeneratorWeight, B_MAX_BITS,
};
use paulistrings::{Circuit, Gf2Hash, PartitionRows, PauliSum};

use crate::cli::{Config, PartitionRowSpec};
use crate::workloads::{build_circuit, haar_su4_block, zz_rotation, LayerKind};

/// The `(q0, q1)` a `rotation_local` / `rotation_remote` cell rotates about: the smallest `q1 > 0`
/// whose one-channel `ZZ(0, q1)` layer has no remote delta (`local`), respectively at least one
/// (`remote`), under `rows`.
///
/// # Panics
///
/// If no qubit in `1..qubits` gives the requested class (possible for `remote` on a pathological
/// row draw); the message names `--partition-seed` as the knob.
pub(crate) fn choose_generator<const W: usize>(
    layer: LayerKind,
    qubits: usize,
    base: &PauliSum<W>,
    rows: &PartitionRows<W>,
) -> (usize, usize) {
    if !layer.picks_generator() {
        return (0, 1);
    }
    let want_remote = layer == LayerKind::RotationRemote;
    // A dense SU(4) needs all 15 deltas local, so it scans every pair, not just those touching 0.
    let pairs: Vec<(usize, usize)> = if layer == LayerKind::Su4Local {
        (0..qubits)
            .flat_map(|q0| ((q0 + 1)..qubits).map(move |q1| (q0, q1)))
            .collect()
    } else {
        (1..qubits).map(|q| (0, q)).collect()
    };
    for (q0, q1) in pairs {
        let mut probe = Circuit::<W>::new(qubits);
        if layer == LayerKind::Su4Local {
            probe.push(haar_su4_block(q0, q1));
        } else {
            probe.push(zz_rotation::<W>(q0, q1, 0.1));
        }
        let (_, remote) = count_remote_deltas(&probe, base.hash(), rows, false)[0];
        if (remote > 0) == want_remote {
            return (q0, q1);
        }
    }
    panic!(
        "phase_breakdown: no {}(q0, q1) layer on {qubits} qubits is {} under these partition \
         rows — try another --partition-seed or more --qubits",
        if layer == LayerKind::Su4Local {
            "SU4"
        } else {
            "ZZ"
        },
        if want_remote { "remote" } else { "local" },
    );
}

/// `P` contiguous qubit blocks cutting as few of `edges` as possible; returns the blocks' end
/// positions.
///
/// Exact by dynamic program, minimizing cut edges first and then size imbalance, with block
/// sizes additionally held within ±25% of `num_qubits / partitions` (without that bound, the
/// heavy-hex lattice's minimum is a 4/123 split whose smaller half holds almost nothing).
pub(crate) fn cut_blocks(
    num_qubits: usize,
    partitions: usize,
    edges: &[(usize, usize)],
) -> Vec<usize> {
    let n = num_qubits;
    assert!(partitions >= 1 && partitions <= n);

    // internal[l * (n + 1) + r] = edges with both endpoints inside the block [l, r).
    let mut internal = vec![0i64; (n + 1) * (n + 1)];
    let mut row = vec![0i64; n + 1];
    for l in (0..n).rev() {
        row.iter_mut().for_each(|v| *v = 0);
        for &(a, b) in edges {
            if a == l && b < n {
                row[b + 1] += 1;
            }
        }
        let mut running = 0i64;
        for r in 0..=n {
            running += row[r];
            internal[l * (n + 1) + r] = internal[(l + 1) * (n + 1) + r] + running;
        }
    }

    // Fewest cut edges dominates; evenness breaks the ties.
    let weight = 4 * (n as i64) * (partitions as i64) + 1;
    let target = n as f64 / partitions as f64;
    let lo = ((target * 0.75).floor() as usize).max(1);
    let hi = ((target * 1.25).ceil() as usize).max(lo);

    const UNSET: i64 = i64::MAX;
    let mut cost = vec![UNSET; (partitions + 1) * (n + 1)];
    let mut from = vec![usize::MAX; (partitions + 1) * (n + 1)];
    cost[0] = 0;
    for k in 1..=partitions {
        for pos in 1..=n {
            let first = pos.saturating_sub(hi);
            let last = pos.saturating_sub(lo);
            for prev in first..=last {
                let before = cost[(k - 1) * (n + 1) + prev];
                if before == UNSET {
                    continue;
                }
                let size = (pos - prev) as i64;
                let deviation = (size * partitions as i64 - n as i64).abs();
                let c = before - internal[prev * (n + 1) + pos] * weight + deviation;
                let slot = k * (n + 1) + pos;
                if cost[slot] == UNSET || c < cost[slot] {
                    cost[slot] = c;
                    from[slot] = prev;
                }
            }
        }
    }
    assert!(
        cost[partitions * (n + 1) + n] != UNSET,
        "cut_blocks: no {partitions}-block split of {n} qubits fits the ±25% size window",
    );

    let mut ends = vec![0usize; partitions];
    let mut pos = n;
    for k in (1..=partitions).rev() {
        ends[k - 1] = pos;
        pos = from[k * (n + 1) + pos];
    }
    ends
}

/// The `log2(P)` z-only "cut" rows labelling the blocks [`cut_blocks`] found, plus the number of
/// `edges` the cut crosses. Every single-qubit rotation is then local, and a `ZZ(i, j)` rotation
/// is remote exactly when the edge crosses the cut.
pub(crate) fn cut_rows<const W: usize>(
    num_qubits: usize,
    partitions: usize,
    edges: &[(usize, usize)],
) -> (PartitionRows<W>, Vec<usize>, usize) {
    let ends = cut_blocks(num_qubits, partitions, edges);

    // label[q] = the partition label of q's block.
    let mut label = vec![0u32; num_qubits];
    let mut blocks: Vec<Vec<usize>> = Vec::with_capacity(partitions);
    let mut start = 0usize;
    for (block, &end) in ends.iter().enumerate() {
        label[start..end].iter_mut().for_each(|l| *l = block as u32);
        blocks.push((start..end).collect());
        start = end;
    }

    let crossed = edges
        .iter()
        .filter(|&&(a, b)| a < num_qubits && b < num_qubits && label[a] != label[b])
        .count();
    (PartitionRows::cut(num_qubits, &blocks), ends, crossed)
}

/// What the chosen rows cost the circuit, for the sidecar.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RowChoiceStats {
    /// Distinct key-delta masks the rows leave remote.
    pub(crate) remote_gens: usize,
    /// Their total weight: the export-and-exchange count a run of this circuit will pay.
    pub(crate) remote_weight: f64,
}

/// How many bucket bits [`choose_partition_rows`] checks row independence at (ten bits is 1024
/// buckets, roughly where the default target puts a million-term sum).
pub(crate) const INDEPENDENCE_PROBE_BITS: u8 = 10;

/// Fresh hash seeds tried when the chosen rows are dependent on `H`'s.
pub(crate) const INDEPENDENCE_RETRIES: usize = 16;

/// The partition rows one cell runs under, the sum they will be scattered from, and what they
/// cost the circuit.
///
/// A non-random row set that is dependent on `H`'s active rows costs load balance, so it gets the
/// hash re-seeded until [`PartitionRows::is_independent_of`] at [`INDEPENDENCE_PROBE_BITS`]
/// passes; this changes the coset dimension too, so the cell reports the seed it ended up with.
pub(crate) fn choose_partition_rows<const W: usize>(
    layer: LayerKind,
    cfg: &Config,
    base: PauliSum<W>,
    partitions: usize,
) -> (PauliSum<W>, PartitionRows<W>, RowChoiceStats) {
    let num_qubits = base.num_qubits();
    let bits = partitions.trailing_zeros() as u8;
    // The generator scan needs the circuit. A layer whose generator qubits
    // come from the rows themselves (`rotation_local` and friends) is
    // circular, so the scan sees that layer's `(0, 1)` variant — harmless,
    // since those layers exist to probe a *given* row set.
    let circuit = build_circuit::<W>(layer, cfg.qubits, cfg.reps, (0, 1));
    let gens = circuit_generators(&circuit, base.hash(), false);

    let rows = match cfg.partition_rows {
        PartitionRowSpec::Random => PartitionRows::<W>::from_seed(
            num_qubits,
            bits,
            cfg.partition_seed.unwrap_or_else(|| base.hash().seed()),
        ),
        PartitionRowSpec::Cut => {
            let edges = layer.cut_edges(num_qubits);
            let (rows, ends, crossed) = cut_rows::<W>(num_qubits, partitions, &edges);
            if partitions > 1 {
                eprintln!(
                    "phase_breakdown: note: --partition-rows cut on {} at P={partitions}: blocks \
                     ending at {ends:?}, crossing {crossed} of {} two-qubit generators (every \
                     single-qubit rotation is local by construction).",
                    layer.name(),
                    edges.len(),
                );
            }
            rows
        }
    };

    // `random` rows are effectively never dependent, so they skip the re-seed check.
    let base = if cfg.partition_rows == PartitionRowSpec::Random || bits == 0 {
        base
    } else {
        reseed_hash_until_independent(cfg, base, &rows, layer, partitions)
    };

    let remote: Vec<&GeneratorWeight<W>> = gens
        .iter()
        .filter(|g| rows.partition_of(&g.mask_x, &g.mask_z) != 0)
        .collect();
    let stats = RowChoiceStats {
        remote_gens: remote.len(),
        remote_weight: remote.iter().map(|g| g.weight).sum(),
    };
    (base, rows, stats)
}

/// Re-seed the sum's hash until `rows` is independent of the rows `H` will have at
/// [`INDEPENDENCE_PROBE_BITS`], or until the tries run out.
pub(crate) fn reseed_hash_until_independent<const W: usize>(
    cfg: &Config,
    base: PauliSum<W>,
    rows: &PartitionRows<W>,
    layer: LayerKind,
    partitions: usize,
) -> PauliSum<W> {
    let num_qubits = base.num_qubits();
    // Rows can only be independent while they fit in the `2n` key columns.
    let headroom = (2 * num_qubits).saturating_sub(rows.bits() as usize);
    let probe_bits = INDEPENDENCE_PROBE_BITS
        .min(B_MAX_BITS)
        .min(headroom.min(u8::MAX as usize) as u8);
    let independent_at =
        |seed: u64| rows.is_independent_of(&Gf2Hash::<W>::new(num_qubits, probe_bits, seed));
    let seed = base.hash().seed();
    if independent_at(seed) {
        return base;
    }

    // Splitmix64's increment: a deterministic full-period walk, identical on every rank.
    let mut candidate = seed;
    for attempt in 1..=INDEPENDENCE_RETRIES {
        candidate = candidate.wrapping_add(0x9E37_79B9_7F4A_7C15);
        if !independent_at(candidate) {
            continue;
        }
        eprintln!(
            "phase_breakdown: note: the {} rows for {} at P={partitions} are NOT independent of \
             H's first {probe_bits} rows under hash seed {seed:#x} — the joint bucket would carry \
             fewer bits than it claims and the split would be unbalanced. Re-seeded the hash to \
             {candidate:#x} (attempt {attempt}). This changes the coset dimension too, so do not \
             compare this cell's phase timings against a differently seeded one \
             (research/FINDINGS.md).",
            cfg.partition_rows.label(),
            layer.name(),
        );
        return with_hash(base, Gf2Hash::<W>::new(num_qubits, 0, candidate));
    }
    eprintln!(
        "phase_breakdown: warning: the {} rows for {} at P={partitions} are dependent on H's \
         first {probe_bits} rows, and {INDEPENDENCE_RETRIES} re-seeds did not fix it. Running \
         anyway: dependence costs load balance, not correctness.",
        cfg.partition_rows.label(),
        layer.name(),
    );
    base
}
