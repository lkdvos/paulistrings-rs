//! The differential oracle and the channels it is run against.
//!
//! [`naive_apply_layer`] is the oracle the bucketed engine is tested against — one `Channel::apply` call per term, a hashmap accumulation, and a sort — and shares no code with the thing it checks.

use hashbrown::HashMap;
use num_complex::Complex64;
use rustc_hash::FxBuildHasher;

use super::*;
use crate::channel::prepared::Prepared;
use crate::channel::{Channel, OutputBuffer, Prepare, PtmChannel};
use crate::pauli_string::PauliString;
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;

/// Apply one channel layer the obvious way, as a differential oracle.
///
/// For every input term: [`Channel::apply`] (or `apply_adjoint` when `adjoint`) into an [`OutputBuffer`], accumulate into a hashmap keyed by `(x, z)`, filter the summed coefficients through [`TruncationPolicy::keep_term`], drop exact zeros, sort by key, rebuild a [`PauliSum`].
/// Deliberately naive: no bucketing, no coset structure, no parallelism, no code shared with `engine::bucketed` beyond the `Channel` trait itself.
/// Hashmap summation order is unspecified, so compare with [`assert_terms_close`], never bitwise.
pub fn naive_apply_layer<const W: usize>(
    input: &PauliSum<W>,
    ch: &dyn Channel<W>,
    policy: &dyn TruncationPolicy<W>,
    adjoint: bool,
) -> PauliSum<W> {
    let mut out = OutputBuffer::<W>::new();
    let mut acc: HashMap<([u64; W], [u64; W]), Complex64, FxBuildHasher> =
        HashMap::with_capacity_and_hasher(input.len(), FxBuildHasher);

    for (x, z, c) in input.iter() {
        out.clear();
        if adjoint {
            ch.apply_adjoint(x, z, c, &mut out);
        } else {
            ch.apply(x, z, c, &mut out);
        }
        for (out_x, out_z, out_c) in out.terms() {
            *acc.entry((out_x, out_z)).or_insert(ZERO) += out_c;
        }
    }

    let mut kept: Vec<([u64; W], [u64; W], Complex64)> = acc
        .into_iter()
        .filter(|((x, z), v)| *v != ZERO && policy.keep_term(x, z, *v))
        .map(|((x, z), v)| (x, z, v))
        .collect();
    kept.sort_unstable_by_key(|&(x, z, _)| (x, z));

    let xs: Vec<[u64; W]> = kept.iter().map(|t| t.0).collect();
    let zs: Vec<[u64; W]> = kept.iter().map(|t| t.1).collect();
    let cs: Vec<Complex64> = kept.iter().map(|t| t.2).collect();
    PauliSum::from_key_sorted(&xs, &zs, &cs, input.hash().clone(), input.num_qubits())
}

/// The engine form of `channel` for one layer, for driving [`apply_layer_bucketed`] directly; `None` where `propagate` would panic.
pub fn prepare<const W: usize, C: Channel<W> + ?Sized>(
    channel: &C,
    hash: &crate::Gf2Hash<W>,
    adjoint: bool,
) -> Option<Prepared<W>> {
    channel.prepare(hash, adjoint)
}

/// Whether `channel` prepares to an in-place coefficient rescale under `hash`.
pub fn is_key_preserving<const W: usize>(
    channel: &dyn Channel<W>,
    hash: &crate::Gf2Hash<W>,
) -> bool {
    matches!(channel.prepare(hash, false), Some(Prepared::Local(ptm)) if ptm.is_key_preserving())
}

/// The XOR masks of `channel`'s key deltas under `hash`: every table entry's, or a rotation's generator.
pub fn key_delta_masks<const W: usize>(
    channel: &dyn Channel<W>,
    hash: &crate::Gf2Hash<W>,
) -> Vec<([u64; W], [u64; W])> {
    channel.prepare(hash, false).expect("a built-in always prepares").key_delta_masks()
}

/// One channel application, normalized the way the merge phase would leave it: exact-ish zeros dropped, rows sorted by key.
/// Use this to compare two channels' mathematical action; when emission order or a zero row is itself under test, use [`raw_outputs`] instead.
pub fn outputs<const W: usize, C: Channel<W> + ?Sized>(
    ch: &C,
    adjoint: bool,
    p: PauliString<W>,
    coeff: Complex64,
) -> Vec<([u64; W], [u64; W], Complex64)> {
    let mut v = raw_outputs(ch, adjoint, p, coeff)
        .into_iter()
        .map(|(q, c)| (q.x, q.z, c))
        .collect::<Vec<_>>();
    v.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    v.retain(|t| t.2.norm() > 1e-15);
    v
}

/// One channel application, exactly as emitted: buffer order, zeros included.
pub fn raw_outputs<const W: usize, C: Channel<W> + ?Sized>(
    ch: &C,
    adjoint: bool,
    p: PauliString<W>,
    coeff: Complex64,
) -> Vec<(PauliString<W>, Complex64)> {
    let mut out = OutputBuffer::<W>::new();
    if adjoint {
        ch.apply_adjoint(&p.x, &p.z, coeff, &mut out);
    } else {
        ch.apply(&p.x, &p.z, coeff, &mut out);
    }
    out.terms().map(|(x, z, c)| (PauliString::<W> { x, z }, c)).collect()
}

/// The engine's differential channel net at `W = 1`: every built-in channel class, on an 8-qubit key space.
///
/// Shared by `engine::bucketed`'s differential net and the partitioned engine's, so both exercise exactly the same channels.
/// Ends with three structurally distinct gather shapes: a fanout-2 non-Clifford, a sparse-but-wide 2Q PTM, a dense one, and a rotation wider than `MAX_LOCAL_SUPPORT` (the `Prepared::Rotation` arm).
pub fn differential_channels_w1() -> Vec<(&'static str, Box<dyn Channel<1>>)> {
    use crate::channel::rotation::PauliRotation;

    vec![
        ("identity", Box::new(IdentityChannel::new())),
        ("h", Box::new(PtmChannel::hadamard(3))),
        ("s", Box::new(PtmChannel::phase_gate(3))),
        ("x", Box::new(PtmChannel::pauli_x(3))),
        ("y", Box::new(PtmChannel::pauli_y(3))),
        ("z", Box::new(PtmChannel::pauli_z(3))),
        ("cnot", Box::new(PtmChannel::cnot(1, 5))),
        ("cz", Box::new(PtmChannel::cz(1, 5))),
        ("swap", Box::new(PtmChannel::swap(1, 5))),
        ("depolarizing", Box::new(PtmChannel::depolarizing(2, 0.07))),
        ("dephasing", Box::new(PtmChannel::dephasing(2, 0.07))),
        ("amp_damping", Box::new(PtmChannel::amplitude_damping(2, 0.3))),
        ("rot_z", Box::new(PauliRotation::new(PauliString::<1>::z(2), 0.41))),
        (
            "rot_zz",
            Box::new(PauliRotation::new(
                {
                    let mut g = PauliString::<1>::z(1);
                    g.mul_assign(&PauliString::<1>::z(6));
                    g
                },
                0.41,
            )),
        ),
        (
            // A non-Clifford T gate (fanout 2), as a local PTM.
            "t_gate",
            Box::new(PtmChannel::unitary_1q(
                2,
                [
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                    [
                        Complex64::new(0.0, 0.0),
                        Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
                    ],
                ],
            )),
        ),
        (
            // sqrt(SWAP): a wide delta set with a sparse PTM.
            "general_2q",
            Box::new(PtmChannel::unitary_2q(1, 5, sqrt_swap_matrix())),
        ),
        (
            // A dense SU(4): every PTM entry nonzero (fanout ~15) — the shape the per-run sort kernel is selected on (`merge::SortKernel::Radix`).
            "haar_su4",
            Box::new(PtmChannel::unitary_2q(1, 5, haar_su4_matrix())),
        ),
        (
            // Weight 4 > MAX_LOCAL_SUPPORT: exercises the Rotation variant.
            "rot_wide",
            Box::new(PauliRotation::new(
                {
                    let mut g = PauliString::<1>::z(0);
                    for q in [2, 4, 6] {
                        g.mul_assign(&PauliString::<1>::x(q));
                    }
                    g
                },
                0.41,
            )),
        ),
    ]
}

/// The engine's differential channel net at `W = 2`: the other occupancy regime — 128 qubits, wide keys, supports straddling the 64-bit word boundary.
/// Shared by `engine::bucketed`'s differential net and the partitioned engine's, as [`differential_channels_w1`] is.
pub fn differential_channels_w2() -> Vec<(&'static str, Box<dyn Channel<2>>)> {
    use crate::channel::rotation::PauliRotation;

    vec![
        ("h@70", Box::new(PtmChannel::hadamard(70))),
        ("s@64", Box::new(PtmChannel::phase_gate(64))),
        ("cnot@60,70", Box::new(PtmChannel::cnot(60, 70))),
        ("swap@0,127", Box::new(PtmChannel::swap(0, 127))),
        ("amp_damping@70", Box::new(PtmChannel::amplitude_damping(70, 0.25))),
        ("rot_y@70", Box::new(PauliRotation::new(PauliString::<2>::y(70), 0.33))),
        (
            "rot_zz_cross_word",
            Box::new(PauliRotation::new(
                {
                    let mut g = PauliString::<2>::z(9);
                    g.mul_assign(&PauliString::<2>::z(70));
                    g
                },
                0.33,
            )),
        ),
        // Dense SU(4), support straddling the word boundary — the dense-PTM run shape at `W = 2`.
        ("haar_su4_cross_word", Box::new(PtmChannel::unitary_2q(60, 70, haar_su4_matrix()))),
    ]
}

/// The identity channel, with empty support.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityChannel;

impl IdentityChannel {
    pub const fn new() -> Self {
        Self
    }
}

impl<const W: usize> Channel<W> for IdentityChannel {
    fn support(&self) -> [u64; W] {
        [0; W]
    }

    fn apply(&self, x: &[u64; W], z: &[u64; W], c: Complex64, out: &mut OutputBuffer<W>) {
        out.push(*x, *z, c);
    }

    fn apply_adjoint(&self, x: &[u64; W], z: &[u64; W], c: Complex64, out: &mut OutputBuffer<W>) {
        self.apply(x, z, c, out);
    }
}

/// A channel with no identity delta: every term moves to `v ⊕ x₀` with its coefficient unchanged.
/// Not physical; it exercises a prepared table whose entry 0 is not the identity.
pub struct ShiftX;

impl<const W: usize> Channel<W> for ShiftX {
    fn apply_adjoint(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut crate::channel::OutputBuffer<W>,
    ) {
        self.apply(input_x, input_z, coeff, out);
    }

    fn support(&self) -> [u64; W] {
        crate::channel::support_mask(&[0])
    }

    fn apply(&self, x: &[u64; W], z: &[u64; W], c: Complex64, out: &mut OutputBuffer<W>) {
        let mut kx = *x;
        kx[0] ^= 1;
        out.push(kx, *z, c);
    }
}

/// The channel [`cancellation_sum`] is built for.
pub fn cancellation_channel() -> PtmChannel {
    PtmChannel::amplitude_damping(0, 0.5)
}
