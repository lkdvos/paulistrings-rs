//! Operator products of a replicated generator with a partitioned operand, accumulated into a destination share as one collective over a [`Transport`] ([`pair_share_into`]).
//!
//! Partition rows are GF(2)-linear, so a generator term `h` takes an operand term from partition `p` to `p ⊕ R·h`, and rank `r` computes `Σ_δ H_δ · O_{r ⊕ δ}` with `H_δ` the generator terms of partition delta `δ`.
//! Partner `r ⊕ δ` sends rank `r` the operand terms `H_δ` can reach, only the anticommuting ones for a Liouvillian, so operands cross the wire and products never do.

use std::collections::BTreeMap;
use std::time::Instant;

use rayon::prelude::*;

use super::distributed::{decode_rank, group_bits, identity_fingerprint};
use crate::coefficient::Coefficient;
use crate::engine::trace::{LayerClock, LayerPhases, PartitionLayerRow};
use crate::engine::transport::{ByteParts, Transport};
use crate::pauli_string::PauliString;
use crate::pauli_sum::hash::PartitionRows;
use crate::pauli_sum::product::support::{SupportIndex, SCAN_ABOVE_WEIGHT};
use crate::pauli_sum::product::{accumulate_into, Entry, Job, Pairing, Weights};
use crate::pauli_sum::storage::{DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN};
use crate::pauli_sum::PauliSum;
use crate::truncation::{assert_supports_partitions, TruncationPolicy};

/// The generator's terms keyed by partition delta, ascending.
type ByDelta<const W: usize, C> = BTreeMap<u32, Vec<Entry<W, C>>>;

/// Every rank must hold the same generator; folded over its keys and coefficient bytes in canonical order.
fn generator_fingerprint<const W: usize, C: Coefficient>(generator: &PauliSum<W, C>) -> u64 {
    let mut folded: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |word: u64| folded = (folded ^ word).wrapping_mul(0x0000_0100_0000_01b3);
    mix(generator.len() as u64);
    for (x, z, c) in generator.iter() {
        x.iter().chain(z).for_each(|&w| mix(w));
        bytemuck::bytes_of(&c).chunks(8).for_each(|chunk| {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            mix(u64::from_le_bytes(word));
        });
    }
    folded
}

/// The terms of bucket `b` that some term of `generator` anticommutes with, or every term without an `index`, in order.
fn reachable_in_bucket<const W: usize, C: Coefficient>(
    share: &PauliSum<W, C>,
    b: usize,
    index: Option<&SupportIndex>,
    generator: &[Entry<W, C>],
    candidates: &mut Vec<u64>,
) -> (Vec<[u64; W]>, Vec<[u64; W]>, Vec<C>) {
    let (bucket_x, bucket_z, bucket_coeff) = share.bucket(b);
    let Some(index) = index else {
        return (bucket_x.to_vec(), bucket_z.to_vec(), bucket_coeff.to_vec());
    };
    let mut reached = vec![0u64; bucket_x.len().div_ceil(64)];
    for h in generator {
        let key = PauliString { x: h.x, z: h.z };
        let support: [u64; W] = std::array::from_fn(|w| h.x[w] | h.z[w]);
        let weight: u32 = support.iter().map(|w| w.count_ones()).sum();
        if weight <= SCAN_ABOVE_WEIGHT {
            index.candidates(b, &support, candidates);
        } else {
            candidates.clear();
            candidates.resize(reached.len(), u64::MAX);
        }
        for (w, &word) in candidates.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let i = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if i < bucket_x.len() {
                    let term = PauliString { x: bucket_x[i], z: bucket_z[i] };
                    reached[w] |= (!key.commutes_with(&term) as u64) << (i % 64);
                }
            }
        }
    }
    let kept = (0..bucket_x.len()).filter(|&i| reached[i / 64] >> (i % 64) & 1 == 1);
    let mut out = (Vec::new(), Vec::new(), Vec::new());
    for i in kept {
        out.0.push(bucket_x[i]);
        out.1.push(bucket_z[i]);
        out.2.push(bucket_coeff[i]);
    }
    out
}

/// The terms of `share` that some term of `generator` anticommutes with, or every term without an `index`, in the gather's wire form, keeping every bucket's order.
fn encode_reachable<const W: usize, C: Coefficient>(
    share: &PauliSum<W, C>,
    index: Option<&SupportIndex>,
    generator: &[Entry<W, C>],
) -> ByteParts {
    let buckets: Vec<_> = (0..share.num_buckets())
        .into_par_iter()
        .map_init(Vec::new, |candidates, b| {
            reachable_in_bucket(share, b, index, generator, candidates)
        })
        .collect();
    let lens: Vec<u64> = buckets.iter().map(|(_, _, c)| c.len() as u64).collect();
    let terms: usize = buckets.iter().map(|(_, _, c)| c.len()).sum();
    let (mut x, mut z, mut coeff) =
        (Vec::with_capacity(terms), Vec::with_capacity(terms), Vec::with_capacity(terms));
    for (bx, bz, bc) in buckets {
        x.extend(bx);
        z.extend(bz);
        coeff.extend(bc);
    }
    ByteParts(vec![
        bytemuck::cast_slice(&lens).to_vec(),
        bytemuck::cast_slice(x.as_flattened()).to_vec(),
        bytemuck::cast_slice(z.as_flattened()).to_vec(),
        bytemuck::cast_slice(&coeff).to_vec(),
    ])
}

/// `dest ← alpha · pairing(generator, operand) + beta · dest` on this rank, where `share` is this rank's share of the operand under `rows`, `dest` holds only this rank's keys and `generator` is the same on every rank, returning the call as one trace row. **Collective.**
///
/// Every rank passes the same `generator`, `rows` and `policy`; partner `r ⊕ δ` receives only the terms of `share` that some generator term of partition delta `δ` can reach, the anticommuting ones for a Liouvillian, so a generator whose terms mostly keep their partition exchanges a halo rather than the share.
/// Every rank's share must carry the same bucket hash rows; bucket counts may differ and are raised to the group's largest.
///
/// # Panics
///
/// If the ranks disagree about `generator`, `rows` or the bucket hash rows, if `rows` does not name one partition per rank, if the qubit counts differ, if `policy` has a layer pass with no partitioned form, or for a product of a real `C`.
pub(crate) fn pair_share_into<const W: usize, C, X, P>(
    dest: &mut PauliSum<W, C>,
    generator: &PauliSum<W, C>,
    share: &PauliSum<W, C>,
    rows: &PartitionRows<W>,
    transport: &X,
    (pairing, weights): (Pairing, Weights<C>),
    policy: &P,
) -> PartitionLayerRow
where
    C: Coefficient,
    X: Transport,
    P: TruncationPolicy<W, C> + ?Sized,
{
    let started = Instant::now();
    let mut clock = LayerClock::new(Some(started));
    let mut phases = LayerPhases::default();
    if pairing == Pairing::Product {
        assert!(
            !C::IS_REAL,
            "product_share: a product is not Hermitian, so a real coefficient type ({}) cannot hold it; cast to a complex type first",
            C::NAME,
        );
    }
    let num_qubits = share.num_qubits();
    assert!(
        generator.num_qubits() == num_qubits && dest.num_qubits() == num_qubits,
        "{pairing:?} of a partitioned operand: num_qubits mismatch (generator {}, share {num_qubits}, destination {})",
        generator.num_qubits(),
        dest.num_qubits(),
    );
    let size = transport.size() as usize;
    rows.assert_splits(num_qubits, 1usize << group_bits(transport.size()));
    assert_supports_partitions(policy, size);
    transport.check_consistency(
        identity_fingerprint(rows, share.hash()) ^ generator_fingerprint(generator).rotate_left(7),
    );

    let bits = transport.allreduce_max_u8(share.hash().bits());
    clock.lap(&mut phases.collective_ns);
    let refined;
    let share = if share.hash().bits() < bits {
        let mut copy = share.clone();
        while copy.hash().bits() < bits {
            copy.refine();
        }
        refined = copy;
        &refined
    } else {
        share
    };
    clock.lap(&mut phases.rebucket_ns);

    let mut by_delta: ByDelta<W, C> = BTreeMap::new();
    for (x, z, c) in generator.iter() {
        let c = c * weights.alpha;
        by_delta.entry(rows.partition_of(x, z)).or_default().push(Entry { x: *x, z: *z, c });
    }
    let remote: Vec<u32> = by_delta.keys().copied().filter(|&delta| delta != 0).collect();

    let rank = transport.rank();
    // Built once: the sends filter by it and the local job reads by it.
    let mut index = (pairing == Pairing::Liouvillian).then(|| SupportIndex::new(share));
    clock.lap(&mut phases.prepare_ns);
    // Every rank holds every delta, so each sends to exactly the partners it receives from, an empty payload included.
    let mut send: Vec<Option<ByteParts>> = (0..size).map(|_| None).collect();
    for &delta in &remote {
        send[(rank ^ delta) as usize] =
            Some(encode_reachable(share, index.as_ref(), &by_delta[&delta]));
    }
    let (mut rows_sent, mut bytes_sent) = (vec![0u64; size], vec![0u64; size]);
    for (to, payload) in send.iter().enumerate() {
        if let Some(ByteParts(parts)) = payload {
            rows_sent[to] = bytemuck::cast_slice::<u8, u64>(&parts[0]).iter().sum();
            bytes_sent[to] = parts.iter().map(|part| part.len() as u64).sum();
        }
    }
    clock.lap(&mut phases.export_ns);
    let mut received = transport.exchange(send, &mut Vec::new());
    let slices: Vec<(u32, PauliSum<W, C>)> = remote
        .iter()
        .map(|&delta| {
            let from = (rank ^ delta) as usize;
            let payload = received[from]
                .take()
                .unwrap_or_else(|| panic!("{pairing:?}: rank {from} sent nothing"));
            (delta, decode_rank(&payload.0, share.hash().clone(), num_qubits, from))
        })
        .collect();
    clock.lap(&mut phases.exchange_ns);
    let rows_received = slices.iter().map(|(_, slice)| slice.len() as u64).sum();

    let mut jobs = Vec::with_capacity(slices.len() + 1);
    if let Some(local) = by_delta.get(&0) {
        jobs.push(Job::with_index(local.iter().copied(), share, index.take(), true));
    }
    for (delta, slice) in &slices {
        jobs.push(Job::new(by_delta[delta].iter().copied(), slice, pairing, true));
    }
    dest.align_destination(share.hash(), weights.beta);
    accumulate_into(dest, weights.beta, &jobs, pairing, &|x, z, c| policy.keep_term(x, z, c));
    drop(jobs);
    drop(slices);
    clock.lap(&mut phases.coset_loop_ns);
    if policy.finalizes_layer() {
        policy.finalize_layer_partitioned(dest, transport);
    }
    clock.lap(&mut phases.finalize_ns);
    dest.rebucket(DEFAULT_TARGET_BUCKET_LEN, DEFAULT_MIN_BUCKETS);
    clock.lap(&mut phases.rebucket_ns);
    PartitionLayerRow {
        bits,
        remote_deltas: remote.len() as u32,
        collectives: 2 + u32::from(policy.finalizes_layer()),
        gate_name: pairing.name(),
        terms_in: share.len(),
        terms_out: dest.len(),
        rows_sent,
        bytes_sent,
        rows_received,
        nanos: started.elapsed().as_nanos() as u64,
        phases,
        ..PartitionLayerRow::default()
    }
}

#[cfg(test)]
mod tests;
