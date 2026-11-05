//! The rotation layer: a two-delta table `{0, g}` applied to each bucket's own columns by a scan, a linear partner order and a merge-join.
//!
//! Contracts: the result agrees with the general coset fill to floating-point tolerance, every output passes `keep_term` exactly as there, and on the Clifford arm the identity residues `|a₀| ≤ ε·|a₁|` are dropped where the general path keeps them.

use crate::coefficient::Coefficient;

use super::coset_fill::{CosetScratch, GatherRun};
use super::partner_order::PartnerOrder;
use super::{coset_loop, DisjointSlots, ExtraRows, LayerOctaveIo, LayerScratch};
use crate::channel::prepared::{LocalPtm, Prepared, PreparedRotation, LOCAL_DIM};
use crate::engine::coset::Gf2Span;
use crate::engine::merge::SortKernel;
use crate::pauli_string::PauliString;
use crate::pauli_sum::storage::{BucketColumns, PauliSum};
use crate::truncation::TruncationPolicy;
use merge::merge_partners;

use crate::engine::timing::{lap, stamp, tally};

/// A bucket's `x` and `z` columns, each ORed over its strings, or a superset of them.
pub(super) type Support<const W: usize> = [[u64; W]; 2];

/// Per bucket, a [`Support`], carried from one rotation layer to the next.
///
/// Contract: known only while the sum is exactly as the last layer of the current call left it, give or take removed strings, and that layer was a local rotation layer or run under the call's policy; then every string has passed that policy's `keep_term`, which is a pure function of the term, so a string no layer changes need not pass it again.
/// Anything else that writes the sum or renumbers its buckets clears it ([`forget`](Self::forget)), and so do the start and end of a call.
#[derive(Debug)]
pub(crate) struct BucketSupports<const W: usize> {
    masks: Vec<Support<W>>,
    known: bool,
}

impl<const W: usize> Default for BucketSupports<W> {
    fn default() -> Self {
        Self { masks: Vec::new(), known: false }
    }
}

impl<const W: usize> BucketSupports<W> {
    /// Drop the supports, so the next rotation layer scans and merges every bucket.
    pub(crate) fn forget(&mut self) {
        self.known = false;
    }
}

/// A coset loop over `span` whose `fill` also gets its members' supports, and whether they are known; every support is known afterwards.
pub(super) fn rotation_coset_loop<const W: usize, C: Coefficient>(
    sum: &mut PauliSum<W, C>,
    span: &Gf2Span,
    scratch: &mut LayerScratch<W, C>,
    octaves: LayerOctaveIo<'_, '_, '_, W, C>,
    fill: impl Fn(&mut [BucketColumns<W, C>], &mut CosetScratch<W, C>, &mut [Support<W>], bool) + Sync,
) -> usize {
    let buckets = sum.num_buckets();
    let supports = &mut scratch.supports;
    let known = std::mem::take(&mut supports.known) && supports.masks.len() == buckets;
    let mut masks = std::mem::take(&mut supports.masks);
    masks.resize(buckets, [[0; W]; 2]);
    let slots = DisjointSlots::new(&mut masks);
    let len = coset_loop(sum, span, scratch, octaves, |members, betas, worker_scratch| {
        let mut supports = std::mem::take(&mut worker_scratch.member_supports);
        supports.clear();
        // SAFETY (both loops): the cosets partition the buckets, so no other task reaches bucket `beta`.
        supports.extend(betas.iter().map(|&beta| *unsafe { slots.get(beta as usize) }));
        fill(members, worker_scratch, &mut supports, known);
        for (support, &beta) in supports.iter().zip(betas) {
            *unsafe { slots.get(beta as usize) } = *support;
        }
        worker_scratch.member_supports = supports;
    });
    scratch.supports = BucketSupports { masks, known: true };
    len
}

/// A table shaped like a Pauli rotation about `g`: an identity entry and one generator entry.
///
/// On a pattern the generator entry skips, the identity amplitude is exactly one; the patterns it reaches are closed under `⊕ g`, so a partner never lands on a string that keeps its coefficient.
#[derive(Clone, Debug)]
pub(super) struct RotationLayer<'p, const W: usize, C: Coefficient = num_complex::Complex64> {
    ptm: &'p LocalPtm<W, C>,
    /// `(mask_x, mask_z)` of `g`.
    mask: ([u64; W], [u64; W]),
    /// `a₀[s]`, the identity entry's amplitude.
    identity: [C; LOCAL_DIM],
    /// `a₁[s]`, the amplitude taking pattern `s` to `s ⊕ g`.
    partner: [C; LOCAL_DIM],
    /// Whether every anticommuting pattern has `|a₀| ≤ ε·|a₁|`, so strings move instead of splitting.
    clifford: bool,
    order: PartnerOrder<W>,
    /// [`LocalPtm::support_probe`].
    probe: [(usize, u32); 2],
    /// Bit `p` set iff the generator entry reaches probe pattern `p`, whose support pattern is `p & support_mask`.
    active: u16,
    support_mask: usize,
    /// `h(g)`.
    bucket_delta: u32,
}

impl<'p, const W: usize, C: Coefficient> RotationLayer<'p, W, C> {
    /// The rotation shape of `ptm`, or `None` if it has another.
    pub(super) fn of(ptm: &'p LocalPtm<W, C>) -> Option<Self> {
        let [identity, generator] = ptm.deltas() else { return None };
        if identity.local_delta != 0 {
            return None;
        }
        let dim = 1usize << (2 * ptm.k());
        let delta = generator.local_delta as usize;
        let mut clifford = true;
        for s in 0..dim {
            let (a0, a1) = (identity.amplitude[s], generator.amplitude[s]);
            if a1 == C::ZERO {
                if a0 != C::ONE {
                    return None;
                }
            } else {
                if generator.amplitude[s ^ delta] == C::ZERO {
                    return None;
                }
                // `cos θ` evaluated at a quarter turn is a rounding residue of `sin θ`, not an amplitude.
                clifford &= a0.norm_sqr() <= f64::EPSILON * f64::EPSILON * a1.norm_sqr();
            }
        }
        let mask = generator.mask();
        let support_mask = dim - 1;
        let active = (0..16)
            .filter(|&p| generator.amplitude[p & support_mask] != C::ZERO)
            .fold(0u16, |bits, p| bits | 1 << p);
        Some(Self {
            ptm,
            mask,
            identity: identity.amplitude,
            partner: generator.amplitude,
            clifford,
            order: PartnerOrder::new(&mask.0, &mask.1),
            probe: ptm.support_probe(),
            active,
            support_mask,
            bucket_delta: generator.bucket_delta,
        })
    }

    /// The probe pattern of `(x, z)`, packed like [`LocalPtm::support_bits`] on two probe slots.
    #[inline(always)]
    fn pattern(&self, x: &[u64; W], z: &[u64; W]) -> usize {
        let [(w0, b0), (w1, b1)] = self.probe;
        let bit = |word: u64, b: u32| ((word >> b) & 1) as usize;
        bit(x[w0], b0) | bit(z[w0], b0) << 1 | bit(x[w1], b1) << 2 | bit(z[w1], b1) << 3
    }

    /// Whether strings move instead of splitting, so each input string has at most one output.
    pub(super) fn is_clifford(&self) -> bool {
        self.clifford
    }

    /// Append each anticommuting string's partner `(k ⊕ g, c·a₁)` to `partners`, then scale those strings by `a₀` in place, or on the Clifford arm zero them for the merge to drop; `hits` is scratch.
    ///
    /// A first pass reads only the key columns and lists the anticommuting rows, so the coefficients are read and written for the listed rows alone, without a data-dependent branch.
    /// Unless `known`, `support` becomes the OR of every row; a known support is a superset already, so the pass reads only the column words the anticommute test needs.
    fn scan(
        &self,
        bucket: &mut BucketColumns<W, C>,
        partners: &mut BucketColumns<W, C>,
        hits: &mut Vec<u32>,
        support: &mut Support<W>,
        known: bool,
    ) {
        let BucketColumns { x, z, coeff } = bucket;
        let (mask_x, mask_z) = &self.mask;
        let [(w0, b0), (w1, b1)] = self.probe;
        let bit = |word: u64, b: u32| ((word >> b) & 1) as usize;
        // An `x` bit anticommutes only under `g`'s `z`, and a `z` bit only under its `x`.
        let reads_x = mask_z.iter().any(|&w| w != 0);
        let reads_z = mask_x.iter().any(|&w| w != 0);
        if !known {
            *support = [[0; W]; 2];
        }
        match (known, reads_x, reads_z) {
            (true, true, false) => {
                self.list_hits(x.len(), hits, |row| bit(x[row][w0], b0) | bit(x[row][w1], b1) << 2)
            }
            (true, false, true) => self.list_hits(x.len(), hits, |row| {
                bit(z[row][w0], b0) << 1 | bit(z[row][w1], b1) << 3
            }),
            (true, _, _) => self.list_hits(x.len(), hits, |row| self.pattern(&x[row], &z[row])),
            (false, _, _) => self.list_hits(x.len(), hits, |row| {
                for w in 0..W {
                    support[0][w] |= x[row][w];
                    support[1][w] |= z[row][w];
                }
                self.pattern(&x[row], &z[row])
            }),
        }
        let anticommuting = hits.len();
        if anticommuting == 0 {
            return;
        }
        let start = partners.len();
        partners.x.reserve(anticommuting);
        partners.z.reserve(anticommuting);
        partners.coeff.reserve(anticommuting);
        // SAFETY: `start` is each column's length, within its allocation.
        let (out_x, out_z, out_c) = unsafe {
            (
                partners.x.as_mut_ptr().add(start),
                partners.z.as_mut_ptr().add(start),
                partners.coeff.as_mut_ptr().add(start),
            )
        };
        for (slot, hit) in hits.iter_mut().enumerate() {
            let row = (*hit >> 4) as usize;
            let pattern = self.pattern(&x[row], &z[row]);
            *hit |= pattern as u32;
            let support = pattern & self.support_mask;
            let c = coeff[row];
            // SAFETY: `slot < anticommuting`, reserved past `start` above.
            unsafe {
                out_x.add(slot).write(core::array::from_fn(|w| x[row][w] ^ mask_x[w]));
                out_z.add(slot).write(core::array::from_fn(|w| z[row][w] ^ mask_z[w]));
                out_c.add(slot).write(c * self.partner[support]);
            }
            coeff[row] = if self.clifford { C::ZERO } else { c * self.identity[support] };
        }
        // SAFETY: written above.
        unsafe {
            partners.x.set_len(start + anticommuting);
            partners.z.set_len(start + anticommuting);
            partners.coeff.set_len(start + anticommuting);
        }
    }

    /// List into `hits`, as `row << 4`, the rows below `len` whose probe `pattern` the generator reaches; a pattern may leave out the bits that cannot change that.
    #[inline(always)]
    fn list_hits(&self, len: usize, hits: &mut Vec<u32>, mut pattern: impl FnMut(usize) -> usize) {
        assert!(len < 1 << 28, "a hit packs its row into 28 bits");
        hits.clear();
        // One slot past the exact bound, for the discarded write.
        hits.reserve(len + 1);
        let listed = hits.as_mut_ptr();
        let mut anticommuting = 0usize;
        for row in 0..len {
            let pattern = pattern(row);
            // SAFETY: `anticommuting ≤ row < len`, within the reserved `len + 1`.
            unsafe { listed.add(anticommuting).write((row as u32) << 4) };
            anticommuting += usize::from((self.active >> pattern) & 1 != 0);
        }
        // SAFETY: the first `anticommuting` slots were written above.
        unsafe { hits.set_len(anticommuting) };
    }

    /// Whether a bucket within `support` may hold a string that anticommutes with `g`, which needs an `x` bit under `g`'s `z` or a `z` bit under its `x`.
    fn may_touch(&self, support: &Support<W>) -> bool {
        let (mask_x, mask_z) = &self.mask;
        (0..W).any(|w| support[0][w] & mask_z[w] | support[1][w] & mask_x[w] != 0)
    }
}

/// A rotation whose generator crosses to a partner, as a receiving bucket applies it: its own strings take the identity half, and the partners arrive from the partner's export.
// Built once per layer on the stack, so the table variant's size costs nothing.
#[allow(clippy::large_enum_variant)]
pub(super) enum ReceivedRotation<'p, const W: usize, C: Coefficient> {
    Table(RotationLayer<'p, W, C>),
    /// `order` is `None` when the generator has too many set bits for [`PartnerOrder`], and the partners are sorted instead.
    Wide {
        rotation: &'p PreparedRotation<W, C::Real>,
        order: Option<PartnerOrder<W>>,
        clifford: bool,
    },
}

impl<'p, const W: usize, C: Coefficient> ReceivedRotation<'p, W, C> {
    pub(super) fn wide(rotation: &'p PreparedRotation<W, C::Real>) -> Self {
        let (mask_x, mask_z) = rotation.generator_mask();
        Self::Wide {
            rotation,
            order: PartnerOrder::try_new(&mask_x, &mask_z),
            // As for a table: `cos θ` at a quarter turn is a rounding residue.
            clifford: (C::ONE * rotation.cos).norm_sqr()
                <= f64::EPSILON * f64::EPSILON * (C::ONE * rotation.sin).norm_sqr(),
        }
    }

    /// The identity half of [`RotationLayer::scan`]: scale each anticommuting string by `a₀`, or on the Clifford arm drop it.
    fn scale(&self, bucket: &mut BucketColumns<W, C>) {
        match self {
            Self::Table(layer) => scale_identity(bucket, layer.clifford, |x, z| {
                let pattern = layer.ptm.support_bits(x, z);
                (layer.partner[pattern].is_nonzero(), layer.identity[pattern])
            }),
            Self::Wide { rotation, clifford, .. } => {
                let cos = C::ONE * rotation.cos;
                scale_identity(bucket, *clifford, |x, z| {
                    let commutes =
                        PauliString::<W> { x: *x, z: *z }.commutes_with(&rotation.generator);
                    (!commutes, if commutes { C::ONE } else { cos })
                })
            }
        }
    }

    fn partner_order(&self) -> Option<&PartnerOrder<W>> {
        match self {
            Self::Table(layer) => Some(&layer.order),
            Self::Wide { order, .. } => order.as_ref(),
        }
    }
}

/// Scale `bucket` in place by `identity(x, z) = (anticommutes, a₀)`, or on the Clifford arm drop the anticommuting strings.
fn scale_identity<const W: usize, C: Coefficient>(
    bucket: &mut BucketColumns<W, C>,
    clifford: bool,
    identity: impl Fn(&[u64; W], &[u64; W]) -> (bool, C),
) {
    let BucketColumns { x, z, coeff } = bucket;
    if clifford {
        let mut kept = 0usize;
        for row in 0..coeff.len() {
            let (key_x, key_z, c) = (x[row], z[row], coeff[row]);
            x[kept] = key_x;
            z[kept] = key_z;
            coeff[kept] = c;
            kept += usize::from(!identity(&key_x, &key_z).0);
        }
        x.truncate(kept);
        z.truncate(kept);
        coeff.truncate(kept);
    } else {
        for row in 0..coeff.len() {
            coeff[row] *= identity(&x[row], &z[row]).1;
        }
    }
}

/// Apply `layer` to bucket `β`: scale it in place and merge in the partners `extra` received for it, which come from the single source bucket `β ⊕ h(g)` in its key order, as [`PartnerOrder`] needs.
pub(super) fn fill_received_rotation_bucket<const W: usize, C: Coefficient, T, X>(
    members: &mut [BucketColumns<W, C>],
    betas: &[u32],
    layer: &ReceivedRotation<'_, W, C>,
    policy: &T,
    scratch: &mut CosetScratch<W, C>,
    extra: &X,
) where
    T: TruncationPolicy<W, C> + ?Sized,
    X: ExtraRows<W, C>,
{
    debug_assert_eq!(members.len(), 1, "a remote generator leaves only the identity delta local");
    let bucket = &mut members[0];
    if scratch.runs.is_empty() {
        scratch.runs.push(GatherRun::default());
    }
    if scratch.old.is_empty() {
        scratch.old.push(BucketColumns::default());
    }
    stamp!(stamp);
    tally!(scratch.stats.rows_gathered, bucket.len());
    tally!(scratch.stats.rows_id, bucket.len());
    layer.scale(bucket);
    let CosetScratch { runs, old, sort, .. } = scratch;
    let run = &mut runs[0];
    run.rest.clear();
    extra.append_into(betas[0], &mut run.rest);
    lap!(stamp, scratch.stats.gather_ns);
    tally!(scratch.stats.rows_gathered, run.rest.len());
    tally!(scratch.stats.rows_sorted, run.rest.len());
    let partners = match layer.partner_order() {
        Some(order) => {
            order.order(&run.rest, &mut run.identity, sort.indices());
            run.identity.view()
        }
        None => {
            SortKernel::Comparison.sort(&mut run.rest, sort);
            run.rest.view()
        }
    };
    lap!(stamp, scratch.stats.sort_ns);
    merge_partners(bucket, partners, policy, &mut old[0], None);
    lap!(stamp, scratch.stats.merge_ns);
    tally!(scratch.stats.cosets, 1);
    tally!(scratch.stats.runs, 1);
}

/// Apply `layer` to one coset, one bucket or the pair `{β, β ⊕ h(g)}`, each partner run merging into the other member.
///
/// `supports[j]` is a superset of member `j`'s support coming in when `known`, and is one coming out either way.
/// A known support certifies, by [`BucketSupports`]'s contract, that every string already passes `keep_term`: a member it shows commuting with `g` is not read, and a member neither changed nor receiving partners is not merged.
pub(super) fn fill_rotation_coset<const W: usize, C: Coefficient, T>(
    members: &mut [BucketColumns<W, C>],
    layer: &RotationLayer<'_, W, C>,
    policy: &T,
    scratch: &mut CosetScratch<W, C>,
    supports: &mut [Support<W>],
    known: bool,
) where
    T: TruncationPolicy<W, C> + ?Sized,
{
    let size = members.len();
    debug_assert!(size == 1 || size == 2);
    debug_assert_eq!(supports.len(), size);
    // The general path's buffers, idle between its cosets: each member's run holds its partners in `rest` and in key order in `identity`, and `old[0]` is the merge fallback.
    scratch.runs.resize_with(size.max(scratch.runs.len()), GatherRun::default);
    if scratch.old.is_empty() {
        scratch.old.push(BucketColumns::default());
    }
    stamp!(stamp);
    let mut changed = [false; 2];
    // Every scan precedes every merge, so each partner carries its source's unscaled coefficient.
    for (j, bucket) in members.iter_mut().enumerate() {
        let partners = &mut scratch.runs[j ^ (size - 1)].rest;
        partners.clear();
        scratch.hits[j].clear();
        if known && !layer.may_touch(&supports[j]) {
            #[cfg(test)]
            {
                scratch.rotation_skips.0 += 1;
            }
            continue;
        }
        // Counted as the general path's identity rows, whose keys stay put.
        tally!(scratch.stats.rows_gathered, bucket.len());
        tally!(scratch.stats.rows_id, bucket.len());
        layer.scan(bucket, partners, &mut scratch.hits[j], &mut supports[j], known);
        changed[j] = !scratch.hits[j].is_empty();
    }
    lap!(stamp, scratch.stats.gather_ns);
    let sources = [supports[0], supports[size - 1]];
    for (j, bucket) in members.iter_mut().enumerate() {
        let run = &mut scratch.runs[j];
        if known && !changed[j] && run.rest.len() == 0 {
            #[cfg(test)]
            {
                scratch.rotation_skips.1 += 1;
            }
            continue;
        }
        if run.rest.len() != 0 {
            // A partner `k ⊕ g` lies within its source's support and `g`'s.
            let source = &sources[size - 1 - j];
            let (mask_x, mask_z) = &layer.mask;
            supports[j] = [
                core::array::from_fn(|w| supports[j][0][w] | source[0][w] | mask_x[w]),
                core::array::from_fn(|w| supports[j][1][w] | source[1][w] | mask_z[w]),
            ];
        }
        tally!(scratch.stats.rows_gathered, run.rest.len());
        tally!(scratch.stats.rows_sorted, run.rest.len());
        layer.order.order(&run.rest, &mut run.identity, scratch.sort.indices());
        lap!(stamp, scratch.stats.sort_ns);
        let dirty = known.then_some(scratch.hits[j].as_slice());
        #[cfg_attr(not(test), allow(unused_variables))]
        let in_place =
            merge_partners(bucket, run.identity.view(), policy, &mut scratch.old[0], dirty);
        #[cfg(test)]
        {
            scratch.rotation_merges.0 += usize::from(in_place);
            scratch.rotation_merges.1 += usize::from(!in_place);
        }
        lap!(stamp, scratch.stats.merge_ns);
    }
    tally!(scratch.stats.cosets, 1);
    tally!(scratch.stats.runs, size);
}

/// Apply `layers` in order to one coset of their joint span, each to the member pairs its generator's coordinate in `coordinates` joins, while the coset is in cache; `supports` and `known` as for [`fill_rotation_coset`].
#[allow(clippy::too_many_arguments)]
fn fill_rotation_run<const W: usize, C: Coefficient, T>(
    members: &mut [BucketColumns<W, C>],
    layers: &[RotationLayer<'_, W, C>],
    coordinates: &[u32],
    policy: &T,
    scratch: &mut CosetScratch<W, C>,
    supports: &mut [Support<W>],
    mut known: bool,
) where
    T: TruncationPolicy<W, C> + ?Sized,
{
    let mut pair: [BucketColumns<W, C>; 2] = Default::default();
    for (layer, &coordinate) in layers.iter().zip(coordinates) {
        for j in 0..members.len() {
            let partner = j ^ coordinate as usize;
            if partner == j {
                let (bucket, support) = (&mut members[j], &mut supports[j..j + 1]);
                fill_rotation_coset(
                    core::slice::from_mut(bucket),
                    layer,
                    policy,
                    scratch,
                    support,
                    known,
                );
            } else if j < partner {
                std::mem::swap(&mut pair[0], &mut members[j]);
                std::mem::swap(&mut pair[1], &mut members[partner]);
                let mut pair_supports = [supports[j], supports[partner]];
                fill_rotation_coset(&mut pair, layer, policy, scratch, &mut pair_supports, known);
                [supports[j], supports[partner]] = pair_supports;
                std::mem::swap(&mut pair[0], &mut members[j]);
                std::mem::swap(&mut pair[1], &mut members[partner]);
            }
        }
        // After the first layer every support is exact, and every string it scanned passed `keep_term`.
        known = true;
    }
}

/// Most layers a rotation run joins.
pub(crate) const MAX_ROTATION_RUN_LAYERS: usize = 8;

/// Largest span dimension of a rotation run's bucket deltas, so a coset of `2^r` buckets at the bucket target stays within a core's L2 (research/FINDINGS.md §Consecutive rotation tables run as one coset loop).
const MAX_ROTATION_RUN_DIMENSION: usize = 3;

/// The bucket deltas of a rotation run being joined, at one bucket count.
pub(crate) struct RotationRunSpan {
    deltas: Vec<u32>,
    bits: u8,
}

impl RotationRunSpan {
    /// An empty run at `bits` bucket bits.
    pub(crate) fn new(bits: u8) -> Self {
        Self { deltas: Vec::new(), bits }
    }

    /// Join `layer` if it is a rotation table that keeps the run's span within [`MAX_ROTATION_RUN_DIMENSION`]; returns whether it joined.
    pub(crate) fn join<const W: usize, C: Coefficient>(&mut self, layer: &Prepared<W, C>) -> bool {
        let Prepared::Local(ptm) = layer else { return false };
        if RotationLayer::of(ptm).is_none() {
            return false;
        }
        let joined = self.deltas.len();
        self.deltas.extend(ptm.bucket_deltas());
        if Gf2Span::new(&self.deltas, self.bits).dimension() > MAX_ROTATION_RUN_DIMENSION {
            self.deltas.truncate(joined);
            return false;
        }
        true
    }
}

/// Apply `run`, consecutive rotation tables a [`RotationRunSpan`] joined under the current hash, as one coset loop over their joint span, each coset taking every layer in order while its buckets are in cache.
///
/// Each layer moves a string only within its own coset of the joint span, so the cosets stay independent and every output of every layer passes `keep_term` as when the layers run one by one; no layer pass runs between them, so the policy must have none.
pub(crate) fn apply_rotation_run<const W: usize, C: Coefficient, T>(
    sum: &mut PauliSum<W, C>,
    run: &[Prepared<W, C>],
    policy: &T,
    scratch: &mut LayerScratch<W, C>,
) where
    T: TruncationPolicy<W, C> + ?Sized,
{
    stamp!(stamp);
    debug_assert!(!policy.finalizes_layer(), "a layer pass would run between the run's layers");
    if let Some(cut) = scratch.octaves.pending.take() {
        cut.apply(sum);
    }
    scratch.octaves.counted = false;
    scratch.octaves.drop_ahead();
    let layers: Vec<RotationLayer<'_, W, C>> = run
        .iter()
        .map(|layer| match layer {
            Prepared::Local(ptm) => RotationLayer::of(ptm),
            Prepared::Rotation(_) => None,
        })
        .map(|layer| layer.expect("a rotation run holds rotation tables only"))
        .collect();
    let deltas: Vec<u32> = layers.iter().map(|layer| layer.bucket_delta).collect();
    let span = Gf2Span::new(&deltas, sum.hash().bits());
    let coordinates: Vec<u32> = deltas.iter().map(|&delta| span.coordinate_of(delta)).collect();
    lap!(stamp, scratch.stats.span_plan_ns);
    #[cfg(test)]
    {
        scratch.rotation_run_layers += run.len();
    }
    let octaves = LayerOctaveIo { cut: None, count: None, ahead: None };
    let len = rotation_coset_loop(
        sum,
        &span,
        scratch,
        octaves,
        |members, worker_scratch, supports, known| {
            fill_rotation_run(
                members,
                &layers,
                &coordinates,
                policy,
                worker_scratch,
                supports,
                known,
            )
        },
    );
    lap!(stamp, scratch.stats.coset_loop_ns);
    sum.set_len_after_layer(len);
    lap!(stamp, scratch.stats.recount_ns);

    #[cfg(debug_assertions)]
    sum.assert_invariants();
}

mod merge;

#[cfg(test)]
mod tests;
