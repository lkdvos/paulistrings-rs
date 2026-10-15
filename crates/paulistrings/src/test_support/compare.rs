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

/// The terms `TopN`/`ApproxTopN` keep out of `terms` (storage order, partitions concatenated in rank order), transcribed with a stable sort independently of production: descending by `rank`, the first `n`, back in input order.
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

/// `|c|²` as its bits, which order a non-negative `f64` as its value.
pub fn magnitude_rank(c: Complex64) -> u64 {
    c.norm_sqr().to_bits()
}

/// The octave of `|c|²`, `ApproxTopN`'s bin.
pub fn octave_rank(c: Complex64) -> u64 {
    c.norm_sqr().to_bits() >> 52
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

/// Squared magnitudes of [`weighted_four_term_sum`]'s terms, which sum to one: each term's probability under a draw by weight.
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

/// Which of `keys` a collapsed sum holds, asserting it is one string with coefficient exactly `1`.
pub fn collapsed_index<const W: usize>(sum: &PauliSum<W>, keys: &[PauliString<W>]) -> usize {
    assert_eq!(sum.len(), 1, "a collapse leaves exactly one string");
    let (x, z, c) = sum.iter().next().unwrap();
    assert_eq!(c, Complex64::new(1.0, 0.0), "the survivor's coefficient is 1");
    keys.iter()
        .position(|p| &p.x == x && &p.z == z)
        .expect("the survivor is one of the input strings")
}

/// Assert each empirical frequency `counts[i] / trials` lies within 4σ of `probs[i]`, `σ² = p(1 − p) / trials`.
pub fn assert_frequencies(counts: &[usize], probs: &[f64], what: &str) {
    let trials: usize = counts.iter().sum();
    for (i, (&k, &p)) in counts.iter().zip(probs).enumerate() {
        let f = k as f64 / trials as f64;
        let sigma = (p * (1.0 - p) / trials as f64).sqrt();
        assert!(
            (f - p).abs() < 4.0 * sigma,
            "{what}: slot {i} drawn {k}/{trials} = {f:.4}, expected {p} ± {sigma:.4}",
        );
    }
}
