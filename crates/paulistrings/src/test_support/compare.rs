//! Comparisons between sums and runs, the storage hooks tests reach through, and the truncation fixtures they compare.

use num_complex::Complex64;

use super::*;
use crate::channel::PtmChannel;
use crate::pauli_string::PauliString;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;

/// Complex numbers equal to within `tol`.
pub fn approx_eq(a: Complex64, b: Complex64, tol: f64) -> bool {
    (a - b).norm() <= tol
}

/// `(x, z, coeff)` triples sorted by the `(x, z)` key.
/// Keys are globally unique (the `PauliSum` invariant forbids duplicates), so this is a canonical, storage-order-independent view.
pub fn canonical_triples<const W: usize>(s: &PauliSum<W>) -> Vec<([u64; W], [u64; W], Complex64)> {
    let mut v: Vec<([u64; W], [u64; W], Complex64)> =
        s.iter().map(|(x, z, c)| (*x, *z, c)).collect();
    v.sort_unstable_by_key(|&(x, z, _)| (x, z));
    v
}

/// `(x, z, coeff)` triples in storage order: bucket by bucket, each bucket in its own order.
pub fn storage_triples<const W: usize>(s: &PauliSum<W>) -> Vec<([u64; W], [u64; W], Complex64)> {
    s.iter().map(|(x, z, c)| (*x, *z, c)).collect()
}

/// The terms `TopN`/`ApproxTopN` keep out of `terms` (storage order), transcribed with a stable sort independently of production: descending by `rank`, the first `n`, back in input order.
/// [`magnitude_rank`] gives `TopN(n)`, [`octave_rank`] gives `ApproxTopN(n)`.
pub fn first_n_by_rank<const W: usize>(
    terms: &[([u64; W], [u64; W], Complex64)],
    n: usize,
    rank: impl Fn(Complex64) -> u64,
) -> Vec<([u64; W], [u64; W], Complex64)> {
    let mut order: Vec<usize> = (0..terms.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(rank(terms[i].2)));
    let mut kept: Vec<usize> = order.into_iter().take(n).collect();
    kept.sort_unstable();
    kept.into_iter().map(|i| terms[i]).collect()
}

/// Checks a partitioned `ApproxTopN(n)` cut, `inputs[r]` cut to `outputs[r]` on rank `r`: every term above the boundary octave survives, `min(n, len)` survive in all, and rank `r` keeps the first `k_r` of its boundary-octave terms in storage order, `k_r` within one of the quota times its fraction of the boundary terms.
pub fn assert_proportional_boundary_cut<const W: usize>(
    inputs: &[PauliSum<W>],
    outputs: &[PauliSum<W>],
    n: usize,
    what: &str,
) {
    assert_eq!(inputs.len(), outputs.len(), "{what}: one output per rank");
    let mut octaves: Vec<u64> =
        inputs.iter().flat_map(storage_triples).map(|t| octave_rank(t.2)).collect();
    let len = octaves.len();
    let kept: usize = outputs.iter().map(PauliSum::len).sum();
    assert_eq!(kept, n.min(len), "{what}: kept count");
    if n == 0 || n >= len {
        for (r, (input, output)) in inputs.iter().zip(outputs).enumerate() {
            assert_eq!(output.len(), if n == 0 { 0 } else { input.len() }, "{what}: rank {r}");
        }
        return;
    }
    octaves.sort_unstable_by(|a, b| b.cmp(a));
    let bin = octaves[n - 1];
    let total = octaves.iter().filter(|&&o| o == bin).count();
    let quota = n - octaves.iter().filter(|&&o| o > bin).count();
    for (r, (input, output)) in inputs.iter().zip(outputs).enumerate() {
        let terms = storage_triples(input);
        let count = terms.iter().filter(|t| octave_rank(t.2) == bin).count();
        let share = output.iter().filter(|&(_, _, c)| octave_rank(c) == bin).count();
        let deviation = (share * total) as i128 - (count * quota) as i128;
        assert!(
            deviation.unsigned_abs() < total as u128,
            "{what}: rank {r} keeps {share} of its {count} boundary terms, the quota is {quota} of {total}",
        );
        let mut taken = 0;
        let mut want: Vec<_> = terms
            .into_iter()
            .filter(|t| match octave_rank(t.2).cmp(&bin) {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Equal => {
                    taken += 1;
                    taken <= share
                }
                std::cmp::Ordering::Less => false,
            })
            .collect();
        want.sort_unstable_by_key(|&(x, z, _)| (x, z));
        assert_eq!(canonical_triples(output), want, "{what}: rank {r}'s kept set");
    }
}

/// `|c|²` as its bits, which order a non-negative `f64` as its value.
pub fn magnitude_rank(c: Complex64) -> u64 {
    c.norm_sqr().to_bits()
}

/// `ApproxTopN`'s bin of `|c|²`.
pub fn octave_rank(c: Complex64) -> u64 {
    crate::truncation::builtin::approx_bin(c.norm_sqr()) as u64
}

/// Same keys, same coefficients bitwise (`Complex64` `==`) — order-agnostic.
/// Only for results that do no arithmetic on coefficients (filters, moves, round trips); anything computed wants [`assert_terms_close`] instead.
pub fn assert_same_terms<const W: usize>(got: &PauliSum<W>, want: &PauliSum<W>, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: term count");
    let got = canonical_triples(got);
    let want = canonical_triples(want);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!((g.0, g.1), (w.0, w.1), "{what}: term {i} key mismatch");
        assert_eq!(
            g.2, w.2,
            "{what}: term {i} key {:?}/{:?} coeff {} vs {} (not bitwise equal)",
            g.0, g.1, g.2, w.2,
        );
    }
}

/// [`PauliSum`] repartitioned under `hash`, every term kept.
pub fn with_hash<const W: usize>(sum: PauliSum<W>, hash: crate::Gf2Hash<W>) -> PauliSum<W> {
    sum.with_hash(hash)
}

/// Bucket `b`'s columns as `(x, z, coeff)`.
pub fn bucket<const W: usize>(
    sum: &PauliSum<W>,
    b: usize,
) -> (&[[u64; W]], &[[u64; W]], &[Complex64]) {
    sum.bucket(b)
}

/// Number of terms in bucket `b`.
pub fn bucket_len<const W: usize>(sum: &PauliSum<W>, b: usize) -> usize {
    sum.bucket_len(b)
}

/// Double the bucket count.
pub fn refine<const W: usize>(sum: &mut PauliSum<W>) {
    sum.refine();
}

/// Halve the bucket count.
pub fn coarsen<const W: usize>(sum: &mut PauliSum<W>) {
    sum.coarsen();
}

/// Panics unless `sum` holds [`PauliSum`]'s structural invariant: every term in its hash bucket, each bucket strictly ascending in `(x, z)`, every key within `num_qubits`.
pub fn assert_invariants<const W: usize>(sum: &PauliSum<W>) {
    sum.assert_invariants();
}

/// Same keys; coefficients within `tol`, since two implementations can sum duplicate keys in different orders and floating-point addition is not associative — the correctness bar per the crate's determinism policy.
pub fn assert_terms_close<const W: usize>(
    got: &PauliSum<W>,
    want: &PauliSum<W>,
    tol: f64,
    what: &str,
) {
    assert_eq!(got.len(), want.len(), "{what}: term count");
    let got = canonical_triples(got);
    let want = canonical_triples(want);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!((g.0, g.1), (w.0, w.1), "{what}: term {i} key mismatch");
        let d = (g.2 - w.2).norm();
        assert!(
            d < tol,
            "{what}: term {i} key {:?}/{:?} coeff {} vs {} (delta {d:e})",
            g.0,
            g.1,
            g.2,
            w.2,
        );
    }
}

/// [`assert_terms_close`], except that a key on one side only must carry a coefficient within `tol` of zero: two orders of one product can leave a rounding residue where the other cancels exactly.
pub fn assert_terms_close_up_to_residues<const W: usize>(
    got: &PauliSum<W>,
    want: &PauliSum<W>,
    tol: f64,
    what: &str,
) {
    let mut deltas = std::collections::BTreeMap::new();
    for (x, z, c) in got.iter() {
        *deltas.entry((*x, *z)).or_insert(ZERO) += c;
    }
    for (x, z, c) in want.iter() {
        *deltas.entry((*x, *z)).or_insert(ZERO) -= c;
    }
    for (key, delta) in deltas {
        assert!(delta.norm() < tol, "{what}: key {key:?} differs by {:e}", delta.norm());
    }
}

/// `policy` that also sums, over layers, the 2-norm of the terms it drops for their coefficient: the `Δ` of [`assert_runs_agree`].
pub struct DiscardMeter<'p, P: ?Sized> {
    policy: &'p P,
    /// `Σ|c|²` the per-term filter dropped in the current layer, as `f64` bits.
    pending: std::sync::atomic::AtomicU64,
    /// `Σ_layers sqrt(dropped Σ|c|²)`, as `f64` bits.
    total: std::sync::atomic::AtomicU64,
}

fn add_f64(cell: &std::sync::atomic::AtomicU64, v: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let _ = cell.fetch_update(Relaxed, Relaxed, |b| Some((f64::from_bits(b) + v).to_bits()));
}

impl<'p, P: ?Sized> DiscardMeter<'p, P> {
    pub fn new(policy: &'p P) -> Self {
        Self { policy, pending: 0f64.to_bits().into(), total: 0f64.to_bits().into() }
    }

    /// The summed per-layer discarded norm so far.
    pub fn discarded(&self) -> f64 {
        f64::from_bits(self.total.load(std::sync::atomic::Ordering::Relaxed))
    }
}

impl<const W: usize, P: TruncationPolicy<W> + ?Sized> TruncationPolicy<W> for DiscardMeter<'_, P> {
    fn keep_term(&self, x: &[u64; W], z: &[u64; W], c: Complex64) -> bool {
        let keep = self.policy.keep_term(x, z, c);
        // A key-only rejection is an orthogonal projection, part of the channel rather than a cut.
        if !keep && self.policy.keep_term(x, z, Complex64::new(1e300, 0.0)) {
            add_f64(&self.pending, c.norm_sqr());
        }
        keep
    }

    fn finalize_layer(&self, sum: &mut PauliSum<W>) {
        let mut dropped = 0.0;
        if self.policy.finalizes_layer() {
            let before = sum.clone();
            self.policy.finalize_layer(sum);
            if sum.len() != before.len() {
                // Kept terms cancel exactly, leaving the dropped ones.
                dropped = norm_sqr(&before.add(&scaled(sum, -1.0)));
            }
        }
        let pending = f64::from_bits(self.pending.swap(0, std::sync::atomic::Ordering::Relaxed));
        add_f64(&self.total, (pending + dropped).sqrt());
    }
}

/// [`ApproxTopN`](crate::ApproxTopN)`(n)`'s layer passes, hidden from the engine's fused form: the oracle for the fused pass's kept set.
#[derive(Clone, Copy, Debug)]
pub struct PlainApproxTopN(pub usize);

impl<const W: usize> TruncationPolicy<W> for PlainApproxTopN {
    fn finalize_layer(&self, sum: &mut PauliSum<W>) {
        crate::ApproxTopN(self.0).finalize_layer(sum);
    }

    fn finalize_layer_partitioned(
        &self,
        local: &mut PauliSum<W>,
        collectives: &dyn crate::Collectives,
    ) {
        crate::ApproxTopN(self.0).finalize_layer_partitioned(local, collectives);
    }

    fn supports_partitioned(&self) -> bool {
        true
    }
}

/// `sum` with the terms partition `r` holds under [`unpinned_scatter`]`(partitions, _, row_seed)` scaled by `2^(-12 r)` and thinned to every `(r + 1)`-th, so the partitions of that split differ in length and in the octave they cut at.
pub fn skewed_by_partition<const W: usize>(
    sum: &PauliSum<W>,
    partitions: usize,
    row_seed: u64,
) -> PauliSum<W> {
    let split =
        crate::PartitionedSum::scatter_with(sum.clone(), unpinned_scatter(partitions, 1, row_seed));
    let mut accumulator = BuildAccumulator::<W>::with_capacity(sum.num_qubits(), sum.len());
    for r in 0..partitions {
        let scale = 2f64.powi(-12 * r as i32);
        for (x, z, c) in split.partition(r).iter().step_by(r + 1) {
            accumulator.add_term(PauliString::<W> { x: *x, z: *z }, c * scale);
        }
    }
    accumulator.finalize()
}

/// `sum` split into four partitions on cut rows over qubit blocks of `num_qubits / 4`, partition `r` thinned to every `(r + 1)`-th term.
pub fn skewed_cut_split<const W: usize>(sum: &PauliSum<W>) -> Vec<PauliSum<W>> {
    let block = sum.num_qubits() / 4;
    let blocks: Vec<Vec<usize>> = (0..4).map(|b| (b * block..(b + 1) * block).collect()).collect();
    let rows = crate::PartitionRows::<W>::cut(sum.num_qubits(), &blocks);
    (0..4u32)
        .map(|r| {
            let mut local = sum.filter_partition(&rows, r);
            local.retain_per_bucket(|_| {
                let mut index = 0;
                move |_, _, _| {
                    index += 1;
                    (index - 1) % (r as usize + 1) == 0
                }
            });
            local
        })
        .collect()
}

fn norm_sqr<const W: usize>(sum: &PauliSum<W>) -> f64 {
    sum.overlap(sum).re
}

fn scaled<const W: usize>(sum: &PauliSum<W>, factor: f64) -> PauliSum<W> {
    let mut out = sum.clone();
    out.scale(Complex64::new(factor, 0.0));
    out
}

/// [`propagate`](crate::propagate) under `policy`, and the summed per-layer norm of what `policy` dropped ([`DiscardMeter`]).
pub fn propagate_discarding<const W: usize, P: TruncationPolicy<W> + ?Sized>(
    circuit: &crate::Circuit<W>,
    sum: PauliSum<W>,
    policy: &P,
    direction: crate::Direction,
) -> (PauliSum<W>, f64) {
    let meter = DiscardMeter::new(policy);
    let out = crate::propagate(circuit, sum, &meter, direction);
    (out, meter.discarded())
}

/// Two runs of `policy` from identical inputs, where the reference dropped a summed per-layer norm `discarded` ([`propagate_discarding`]).
///
/// Equal term counts always; term for term to `tol` without a layer pass or when the reference dropped nothing.
/// Otherwise the runs may keep different boundary terms (`TopN`/`ApproxTopN` fill in bucket order), and each lies within `Δ = discarded + √len·tol` of the untruncated operator, so `1 − Re⟨got, want⟩ / (‖got‖ ‖want‖) ≤ 2Δ² / (‖got‖ ‖want‖)`.
/// Panics if that bound could not see a sign flip on the largest term: such a cut is too heavy to compare.
pub fn assert_runs_agree<const W: usize>(
    policy: impl Into<crate::truncation::BuiltinTruncation>,
    discarded: f64,
    got: &PauliSum<W>,
    want: &PauliSum<W>,
    tol: f64,
    what: &str,
) {
    let cuts = <crate::truncation::BuiltinTruncation as TruncationPolicy<W>>::finalizes_layer(
        &policy.into(),
    );
    assert_eq!(got.len(), want.len(), "{what}: term count");
    if !cuts || discarded == 0.0 || want.is_empty() {
        assert_terms_close(got, want, tol, what);
        return;
    }
    let overlap = Overlap::of(discarded, got, want, tol);
    assert!(
        overlap.sees_a_sign_flip,
        "{what}: Δ = {discarded:e} of ‖want‖ = {:e} is too heavy a cut to compare",
        overlap.want_norm,
    );
    assert!(
        overlap.distance <= overlap.bound,
        "{what}: 1 − cos = {:e} exceeds 2Δ²/(‖got‖‖want‖) = {:e} (Δ = {:e})",
        overlap.distance,
        overlap.bound,
        overlap.slack,
    );
}

/// [`assert_runs_agree`]'s operator check: `1 − cos` between the runs, its bound from `Δ`, and whether that bound could see a sign flip on the reference's largest term.
struct Overlap {
    distance: f64,
    bound: f64,
    slack: f64,
    want_norm: f64,
    sees_a_sign_flip: bool,
}

impl Overlap {
    fn of<const W: usize>(discarded: f64, got: &PauliSum<W>, want: &PauliSum<W>, tol: f64) -> Self {
        let (got_norm, want_norm) = (norm_sqr(got).sqrt(), norm_sqr(want).sqrt());
        // `1 − cos = ‖got/‖got‖ − want/‖want‖‖² / 2`, free of cancellation.
        let difference = scaled(got, 1.0 / got_norm).add(&scaled(want, -1.0 / want_norm));
        let slack = discarded + (want.len() as f64).sqrt() * tol;
        let bound = 2.0 * slack * slack / (got_norm * want_norm);
        let largest = want.iter().map(|t| t.2.norm_sqr()).fold(0.0, f64::max);
        Self {
            distance: norm_sqr(&difference) / 2.0,
            bound,
            slack,
            want_norm,
            sees_a_sign_flip: bound < 2.0 * largest / (want_norm * want_norm),
        }
    }
}

/// Two runs of `policy`, a bare `TopN(n)` or `ApproxTopN(n)`, through a cut too heavy for [`assert_runs_agree`] to compare as operators: both keep exactly `n` terms, all finite.
/// Panics if the cut is light enough for [`assert_runs_agree`], which is then the check to use.
pub fn assert_runs_agree_count_only<const W: usize>(
    policy: impl Into<crate::truncation::BuiltinTruncation>,
    discarded: f64,
    got: &PauliSum<W>,
    want: &PauliSum<W>,
    what: &str,
) {
    use crate::truncation::BuiltinTruncation as T;
    let n = match policy.into() {
        T::TopN(n) | T::ApproxTopN(n) => n,
        other => {
            panic!("{what}: a count-only comparison needs a bare TopN or ApproxTopN, got {other:?}")
        }
    };
    assert_eq!(got.len(), want.len(), "{what}: term count");
    assert_eq!(want.len(), n, "{what}: the last layer must cut");
    for (run, sum) in [("got", got), ("want", want)] {
        assert!(
            sum.iter().all(|t| t.2.re.is_finite() && t.2.im.is_finite()),
            "{what}: {run} has a non-finite coefficient"
        );
    }
    assert!(
        !Overlap::of(discarded, got, want, 0.0).sees_a_sign_flip,
        "{what}: Δ = {discarded:e} is light enough for assert_runs_agree"
    );
}

/// `trotter_steps` at angle `0.4` over [`tie_heavy_sum`] of `terms`: under a `TopN`/`ApproxTopN` well below `terms`, every layer cuts through a tie group or an octave, too heavily for [`assert_runs_agree`].
pub fn heavy_cut<const W: usize>(
    num_qubits: usize,
    steps: usize,
    terms: usize,
    seed: u64,
) -> (crate::Circuit<W>, PauliSum<W>) {
    (trotter_steps(num_qubits, 0.4, steps), tie_heavy_sum(terms, num_qubits, seed))
}

/// One-layer circuits whose single cut splits a boundary group and drops little weight, each with its input and the `n` that cuts it: a Haar SU(4) on `(0, 1)` over [`spread_magnitudes`] of [`rand_sum`], and a Hadamard on qubit 0 over magnitudes `1`, `2⁻²⁰`, `2⁻⁴⁰`, where `n` falls inside the `2⁻²⁰` ties.
pub fn single_cuts<const W: usize>(
    num_qubits: usize,
    seed: u64,
) -> Vec<(&'static str, crate::Circuit<W>, PauliSum<W>, usize)> {
    let mut dense = crate::Circuit::<W>::new(num_qubits);
    dense.push(PtmChannel::unitary_2q(0, 1, haar_su4_matrix()));
    let mut hadamard = crate::Circuit::<W>::new(num_qubits);
    hadamard.push(PtmChannel::hadamard(0));
    let mut tied = BuildAccumulator::<W>::new(num_qubits);
    for (x, z, c) in rand_sum_real::<W>(2_000, num_qubits, seed ^ 1).iter() {
        let class = (c.re.abs() * 3.0).floor().min(2.0) as i32;
        let p = PauliString::<W> { x: *x, z: *z };
        let magnitude = 2f64.powi(20 * class - 40);
        tied.add_term(p, Complex64::new(magnitude.copysign(c.re), 0.0));
    }
    let random = rand_sum::<W>(1_500, num_qubits, seed);
    vec![
        ("dense cut", dense, spread_magnitudes(&random, 40, seed), 2_000),
        ("tied cut", hadamard, tied.finalize(), 1_300),
    ]
}

/// The keys of [`weighted_four_term_sum`], in order: `X0`, `Z3 X5`, `Y2 Y7`, `X1 Z6`.
pub fn four_term_keys<const W: usize>() -> [PauliString<W>; 4] {
    let key = |x: u64, z: u64| {
        let mut p = PauliString::<W>::identity();
        p.x[0] = x;
        p.z[0] = z;
        p
    };
    [
        key(1, 0),
        key(1 << 5, 1 << 3),
        key((1 << 2) | (1 << 7), (1 << 2) | (1 << 7)),
        key(1 << 1, 1 << 6),
    ]
}

/// Squared magnitudes of [`weighted_four_term_sum`]'s terms, which sum to one.
pub const FOUR_TERM_WEIGHTS: [f64; 4] = [0.1, 0.2, 0.3, 0.4];

/// [`four_term_keys`] on `num_qubits >= 8` qubits with `|c|² =` [`FOUR_TERM_WEIGHTS`] and four different phases.
pub fn weighted_four_term_sum<const W: usize>(num_qubits: usize) -> PauliSum<W> {
    assert!(num_qubits >= 8, "the fixture's keys reach qubit 7");
    let phases = [0.0f64, 1.0, 2.5, -2.0];
    let mut acc = BuildAccumulator::<W>::new(num_qubits);
    for ((p, w), phi) in four_term_keys::<W>().into_iter().zip(FOUR_TERM_WEIGHTS).zip(phases) {
        acc.add_term(p, Complex64::from_polar(w.sqrt(), phi));
    }
    acc.finalize()
}

/// `X`-strings with `x[0] = 1, 2, 3, …` carrying `coefficients` in that (iteration) order; a zero coefficient is kept as a term.
pub fn ordered_sum<const W: usize, C: crate::Coefficient>(coefficients: &[C]) -> PauliSum<W, C> {
    let key = |i: usize| {
        let mut p = PauliString::<W>::identity();
        p.x[0] = i as u64 + 1;
        p
    };
    let mut acc = BuildAccumulator::<W, C>::new(8);
    for i in 0..coefficients.len() {
        acc.add_term(key(i), C::ONE);
    }
    let mut sum = acc.finalize();
    sum.update_per_bucket(|_| {
        |x: &[u64; W], _: &[u64; W], _: C| Some(coefficients[x[0] as usize - 1])
    });
    let order: Vec<u64> = sum.iter().map(|(x, _, _)| x[0]).collect();
    assert_eq!(order, (1..=coefficients.len() as u64).collect::<Vec<_>>(), "iteration order");
    sum
}

/// Each term's multiplicity under the systematic draw [`crate::CollapseSample`] documents over `magnitudes` laid end to end, by a search per point: the reference its bucketed walk is held to.
pub fn systematic_multiplicities(magnitudes: &[f64], u: f64, samples: usize) -> Vec<usize> {
    let ends: Vec<f64> = magnitudes
        .iter()
        .scan(0.0f64, |end, &magnitude| {
            *end += magnitude;
            Some(*end)
        })
        .collect();
    let total = ends.last().copied().unwrap_or(0.0);
    let step = total / samples as f64;
    let last_nonzero = ends.partition_point(|&end| end < total);
    let mut counts = vec![0usize; magnitudes.len()];
    for k in 0..samples {
        let t = (u + k as f64) * step;
        let owner = if t >= total { last_nonzero } else { ends.partition_point(|&end| end <= t) };
        counts[owner] += 1;
    }
    counts
}

/// Assert `got` is the systematic resample of `terms`, listed in draw order, at offset `u` with `samples` points: exactly the drawn keys, each with coefficient `m·h·c/|c|` to `tolerance·h`, and every `m` the floor or ceiling of `samples·|c|/Σ|c|`.
pub fn assert_systematic_resample<const W: usize, C: crate::Coefficient>(
    got: &PauliSum<W, C>,
    terms: &[([u64; W], [u64; W], C)],
    u: f64,
    samples: usize,
    tolerance: f64,
    what: &str,
) {
    let magnitudes: Vec<f64> = terms.iter().map(|term| term.2.magnitude()).collect();
    let counts = systematic_multiplicities(&magnitudes, u, samples);
    let total: f64 = magnitudes.iter().sum();
    let step = total / samples as f64;
    let mut kept = 0;
    for ((x, z, c), m) in terms.iter().zip(counts) {
        let share = c.magnitude() / step;
        assert!(
            share.floor() - 1e-9 <= m as f64 && m as f64 <= share.ceil() + 1e-9,
            "{what}: drawn {m} times for a share of {share} points",
        );
        let found = got.get(x, z);
        if m == 0 {
            assert!(found.is_none(), "{what}: an undrawn term survived");
            continue;
        }
        kept += 1;
        let found = found
            .unwrap_or_else(|| panic!("{what}: a term drawn {m} times is missing"))
            .to_complex64();
        let want = c.to_complex64() * (m as f64 * step / c.magnitude());
        assert!(
            (found - want).norm() <= tolerance * step,
            "{what}: drawn {m} times, {found} vs {want}"
        );
    }
    assert_eq!(got.len(), kept, "{what}: kept count");
}

/// Assert `got` is `samples` draws of `l1 / samples` each: every `|c|` a positive whole number of draws, the draws summing to `samples`.
pub fn assert_draws_tile_l1<const W: usize>(
    got: &PauliSum<W>,
    l1: f64,
    samples: usize,
    what: &str,
) {
    let mut drawn = 0.0;
    for (_, _, c) in got.iter() {
        let draws = c.norm() * samples as f64 / l1;
        assert!((draws - draws.round()).abs() < 1e-9 && draws.round() >= 1.0, "{what}: {draws}");
        drawn += draws.round();
    }
    assert_eq!(drawn as usize, samples, "{what}: draws over the group");
}

/// The buckets whose export read the counts ahead under a pending cut, summed over `sum`'s partitions and every layer they drove.
pub fn buckets_exported_ahead_under_cut<const W: usize>(sum: &crate::PartitionedSum<W>) -> u64 {
    sum.buckets_exported_ahead_under_cut()
}

/// [`buckets_exported_ahead_under_cut`] of this rank's partition of `sum`.
pub fn buckets_exported_ahead_under_cut_on_rank<const W: usize, X: crate::Transport>(
    sum: &crate::DistributedSum<W, X>,
) -> u64 {
    sum.buckets_exported_ahead_under_cut()
}

/// `sum` without the terms below `1e-13`, where two summation orders disagree on whether a key cancels exactly.
pub fn drop_tiny<const W: usize>(sum: &PauliSum<W>) -> PauliSum<W> {
    let mut out = sum.clone();
    out.retain(|_, _, c| c.norm() > 1e-13);
    out
}

/// `i[generator, operand]` truncated by `policy` as one layer, into a fresh sum: the one-process reference for a partitioned or recursive Liouvillian.
pub fn truncated_liouvillian<const W: usize, C, P>(
    generator: &PauliSum<W, C>,
    operand: &PauliSum<W, C>,
    policy: &P,
) -> PauliSum<W, C>
where
    C: crate::Coefficient,
    P: crate::TruncationPolicy<W, C> + ?Sized,
{
    let mut out = PauliSum::empty(operand.num_qubits());
    out.add_liouvillian(generator, operand, C::ONE, C::ZERO, policy);
    out
}
