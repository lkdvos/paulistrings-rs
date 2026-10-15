//! Gate matrices, seeded circuits, and GF(2) rank helpers.

use num_complex::Complex64;

use super::*;
use crate::channel::PtmChannel;
use crate::pauli_string::PauliString;

/// The `sqrt(SWAP)` 4×4 unitary on `(a, b)`, as a matrix for [`PtmChannel::unitary_2q`](crate::PtmChannel::unitary_2q).
/// The canonical sparse-but-wide two-qubit fixture: its delta set spans more than one bucket bit, yet its PTM is far from dense (steady-state fanout 3.65 vs. a dense PTM's 14.94), exercising multi-delta behaviour without [`haar_su4_matrix`]'s all-sixteen-entries cost.
pub fn sqrt_swap_matrix() -> [[Complex64; 4]; 4] {
    let h = Complex64::new(0.5, 0.5);
    let hc = Complex64::new(0.5, -0.5);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    [[one, zero, zero, zero], [zero, h, hc, zero], [zero, hc, h, zero], [zero, zero, zero, one]]
}

/// One draw of Haar-random SU(4), as a 4×4 unitary in the computational basis.
///
/// Transcribed from `examples/common/circuits.py::haar_su4` under `numpy.random.default_rng(0xC0FFEE)` — the same distribution `bench_jl_performance.py::su4_gates` and benchmark E's `random_su4_staircase` draw from. Unitary to 2.5e-16; `GeneralUnitary2Q::from_matrix` does not check.
///
/// The canonical dense-PTM fixture: a generic SU(4) gives all sixteen local delta entries a nonzero amplitude, which is what makes the sort dominate the layer (research/FINDINGS.md). `sqrt(SWAP)` and the Cliffords are not substitutes — their PTMs are sparse.
pub fn haar_su4_matrix() -> [[Complex64; 4]; 4] {
    [
        [
            Complex64::new(0.44535882417102446, 0.1243885298575445),
            Complex64::new(-0.09402453034947537, -0.14670085591185988),
            Complex64::new(0.7459177705812382, -0.3801992439705379),
            Complex64::new(0.052557524520682804, 0.22828530169893588),
        ],
        [
            Complex64::new(-0.04863200501298571, -0.40347772310563557),
            Complex64::new(0.7069517563162028, 0.008408200837924597),
            Complex64::new(0.26012555224671347, -0.12053357328338017),
            Complex64::new(-0.3528728311960538, -0.3581567969892209),
        ],
        [
            Complex64::new(-0.35880447773297086, 0.11743595956162649),
            Complex64::new(-0.3097428484619983, 0.594366207605036),
            Complex64::new(0.1610278707748687, -0.25258937630123157),
            Complex64::new(-0.5597163461470217, 0.07240555858329784),
        ],
        [
            Complex64::new(-0.578592686173378, -0.3791072567045837),
            Complex64::new(0.05738813758483608, 0.13145928539206422),
            Complex64::new(0.3453330441780492, 0.08874282848443517),
            Complex64::new(0.5033919610984813, 0.34698642893070086),
        ],
    ]
}

/// GF(2) rank of a set of bucket indices, by Gaussian elimination.
///
/// One pivot slot per bit position, so each vector is either absorbed by an existing pivot or becomes a new one — the same reduced-basis subtlety `engine::coset::Gf2Span::new` handles by back-substituting.
/// `Gf2Span::dimension()` is exactly this rank applied to a prepared channel's `bucket_deltas()`, the coset dimension the engine gets for a layer (research/FINDINGS.md).
fn gf2_rank(vs: &[u32]) -> usize {
    let mut pivot = [0u32; 32];
    let mut r = 0usize;
    for &v in vs {
        let mut v = v;
        while v != 0 {
            let hb = (31 - v.leading_zeros()) as usize;
            if pivot[hb] == 0 {
                pivot[hb] = v;
                r += 1;
                break;
            }
            v ^= pivot[hb];
        }
    }
    r
}

/// Rank of `h` restricted to the key-delta space of a support.
///
/// A channel on `qubits` can only change those qubits' `x`/`z` bits, so its key-delta set lies in `span{X_q, Z_q : q ∈ qubits}` (dimension `2·|qubits|`); this returns the dimension of that space's image under `h`.
/// Full rank means `h` separates every local delta; anything less means two distinct local deltas share one bucket delta.
pub fn support_delta_rank<const W: usize>(
    h: &crate::pauli_sum::Gf2Hash<W>,
    qubits: &[usize],
) -> usize {
    let mut imgs: Vec<u32> = Vec::with_capacity(2 * qubits.len());
    for &q in qubits {
        let (x, z) = (PauliString::<W>::x(q), PauliString::<W>::z(q));
        imgs.push(h.bucket_of(&x.x, &x.z));
        imgs.push(h.bucket_of(&z.x, &z.z));
    }
    gf2_rank(&imgs)
}

fn set_x<const W: usize>(p: &mut PauliString<W>, q: usize) {
    p.x[q / 64] |= 1u64 << (q % 64);
}

fn set_z<const W: usize>(p: &mut PauliString<W>, q: usize) {
    p.z[q / 64] |= 1u64 << (q % 64);
}

/// A seeded circuit drawing from every built-in channel class.
///
/// `dense` adds the wide-fanout classes (a dense 1Q PTM, sqrt-SWAP, a Haar SU(4) block); without it every layer has fanout at most 2, which is what keeps an untruncated run bounded.
/// Kind 8 is a weight-4 rotation, so `prepare` takes the `Prepared::Rotation` arm.
pub fn random_circuit<const W: usize>(
    num_qubits: usize,
    layers: usize,
    seed: u64,
    dense: bool,
) -> crate::Circuit<W> {
    use crate::channel::rotation::PauliRotation;

    let mut rng = Xs64::new(seed);
    let mut circuit = crate::Circuit::<W>::new(num_qubits);
    let kinds: u64 = if dense { 17 } else { 14 };
    let n = num_qubits as u64;
    for _ in 0..layers {
        let q0 = (rng.next_u64() % n) as usize;
        let q1 = ((q0 as u64 + 1 + rng.next_u64() % (n - 1)) % n) as usize;
        let wrap = |q: usize, d: usize| (q + d) % num_qubits;
        match rng.next_u64() % kinds {
            0 => circuit.push(PtmChannel::hadamard(q0)),
            1 => circuit.push(PtmChannel::phase_gate(q0)),
            2 => circuit.push(PtmChannel::pauli_y(q0)),
            3 => circuit.push(PtmChannel::cnot(q0, q1)),
            4 => circuit.push(PtmChannel::cz(q0, q1)),
            5 => circuit.push(PtmChannel::swap(q0, q1)),
            6 => circuit.push(PauliRotation::new(PauliString::<W>::z(q0), 0.37)),
            7 => circuit.push(zz_rotation::<W>(q0, q1, 0.21)),
            8 => {
                let mut gen = PauliString::<W> { x: [0u64; W], z: [0u64; W] };
                set_x(&mut gen, q0);
                set_z(&mut gen, wrap(q0, 1));
                set_x(&mut gen, wrap(q0, 2));
                set_z(&mut gen, wrap(q0, 3));
                circuit.push(PauliRotation::new(gen, 0.29));
            }
            9 => circuit.push(PtmChannel::depolarizing(q0, 0.05)),
            10 => circuit.push(PtmChannel::dephasing(q0, 0.11)),
            11 => circuit.push(PtmChannel::pauli_channel(q0, 0.03, 0.04, 0.05)),
            12 => circuit.push(PtmChannel::depolarizing_2q(q0, q1, 0.07)),
            13 => circuit.push(PtmChannel::amplitude_damping(q0, 0.09)),
            14 => circuit.push(PtmChannel::unitary_1q(
                q0,
                [
                    [Complex64::new(0.6, 0.0), Complex64::new(0.0, -0.8)],
                    [Complex64::new(0.0, -0.8), Complex64::new(0.6, 0.0)],
                ],
            )),
            15 => circuit.push(PtmChannel::unitary_2q(q0, q1, sqrt_swap_matrix())),
            16 => circuit.push(PtmChannel::unitary_2q(q0, q1, haar_su4_matrix())),
            _ => unreachable!(),
        }
    }
    circuit
}

/// A seeded circuit of Cliffords alone: `H`, `S`, `Y`, `CNOT`, `CZ` and `SWAP` on random qubits, so every layer is a permutation of keys (`Y` a key-preserving one).
pub fn random_clifford_circuit<const W: usize>(
    num_qubits: usize,
    layers: usize,
    seed: u64,
) -> crate::Circuit<W> {
    let mut rng = Xs64::new(seed);
    let mut circuit = crate::Circuit::<W>::new(num_qubits);
    let n = num_qubits as u64;
    for _ in 0..layers {
        let q0 = (rng.next_u64() % n) as usize;
        let q1 = ((q0 as u64 + 1 + rng.next_u64() % (n - 1)) % n) as usize;
        match rng.next_u64() % 6 {
            0 => circuit.push(PtmChannel::hadamard(q0)),
            1 => circuit.push(PtmChannel::phase_gate(q0)),
            2 => circuit.push(PtmChannel::pauli_y(q0)),
            3 => circuit.push(PtmChannel::cnot(q0, q1)),
            4 => circuit.push(PtmChannel::cz(q0, q1)),
            _ => circuit.push(PtmChannel::swap(q0, q1)),
        }
    }
    circuit
}

/// Haar SU(4) layers on `(0, 1)`, `(1, 2)` and `(0, 1)` of `num_qubits` qubits: dense and overlapping, so one partner's remote rows collide at every partition count.
pub fn su4_chain<const W: usize>(num_qubits: usize) -> crate::Circuit<W> {
    let mut c = crate::Circuit::<W>::new(num_qubits);
    for (a, b) in [(0, 1), (1, 2), (0, 1)] {
        c.push(PtmChannel::unitary_2q(a, b, haar_su4_matrix()));
    }
    c
}

/// A weight-2 `ZZ` rotation — the TFIM bond term, the smallest layer whose generator can cross a partition boundary.
pub fn zz_rotation<const W: usize>(
    q0: usize,
    q1: usize,
    theta: f64,
) -> crate::channel::rotation::PauliRotation<W> {
    let mut gen = PauliString::<W> { x: [0u64; W], z: [0u64; W] };
    for q in [q0, q1] {
        gen.z[q / 64] |= 1u64 << (q % 64);
    }
    crate::channel::rotation::PauliRotation::new(gen, theta)
}

/// One TFIM Trotter step: `num_qubits` periodic `ZZ` bond rotations, then that many transverse-field `X` rotations, all at angle `2 · theta`.
/// `2 · num_qubits` layers, enough that the term count grows across the run.
pub fn trotter_circuit<const W: usize>(num_qubits: usize, theta: f64) -> crate::Circuit<W> {
    trotter_steps(num_qubits, theta, 1)
}

/// `steps` repetitions of [`trotter_circuit`].
pub(super) fn trotter_steps<const W: usize>(
    num_qubits: usize,
    theta: f64,
    steps: usize,
) -> crate::Circuit<W> {
    let mut circuit = crate::Circuit::<W>::new(num_qubits);
    for _ in 0..steps {
        for q in 0..num_qubits {
            let q1 = (q + 1) % num_qubits;
            circuit.push(zz_rotation::<W>(q, q1, 2.0 * theta));
        }
        for q in 0..num_qubits {
            circuit.push(crate::channel::rotation::PauliRotation::new(
                PauliString::<W>::x(q),
                2.0 * theta,
            ));
        }
    }
    circuit
}

/// Three `trotter_steps` on eight qubits at a large angle, enough to grow [`z0_sum`] past a cache of 6 several times.
pub fn collapsing_circuit() -> crate::Circuit<1> {
    trotter_steps(8, 0.3, 3)
}

/// The 127-qubit heavy-hex coupling map (IBM Eagle r3), 144 undirected edges as `(lo, hi)` pairs in sorted order.
///
/// A verbatim copy of `examples/data/heavy_hex_127.edges`, kept here as a constant so Rust probes need no file I/O; `heavy_hex_127_edges_match_the_source_lattice` pins the transcription.
/// Degree histogram: 2 qubits of degree 1, 89 of degree 2, 36 of degree 3.
#[rustfmt::skip]
pub const HEAVY_HEX_127_EDGES: [(usize, usize); 144] = [
    (0, 1), (0, 14), (1, 2), (2, 3), (3, 4), (4, 5),
    (4, 15), (5, 6), (6, 7), (7, 8), (8, 9), (8, 16),
    (9, 10), (10, 11), (11, 12), (12, 13), (12, 17), (14, 18),
    (15, 22), (16, 26), (17, 30), (18, 19), (19, 20), (20, 21),
    (20, 33), (21, 22), (22, 23), (23, 24), (24, 25), (24, 34),
    (25, 26), (26, 27), (27, 28), (28, 29), (28, 35), (29, 30),
    (30, 31), (31, 32), (32, 36), (33, 39), (34, 43), (35, 47),
    (36, 51), (37, 38), (37, 52), (38, 39), (39, 40), (40, 41),
    (41, 42), (41, 53), (42, 43), (43, 44), (44, 45), (45, 46),
    (45, 54), (46, 47), (47, 48), (48, 49), (49, 50), (49, 55),
    (50, 51), (52, 56), (53, 60), (54, 64), (55, 68), (56, 57),
    (57, 58), (58, 59), (58, 71), (59, 60), (60, 61), (61, 62),
    (62, 63), (62, 72), (63, 64), (64, 65), (65, 66), (66, 67),
    (66, 73), (67, 68), (68, 69), (69, 70), (70, 74), (71, 77),
    (72, 81), (73, 85), (74, 89), (75, 76), (75, 90), (76, 77),
    (77, 78), (78, 79), (79, 80), (79, 91), (80, 81), (81, 82),
    (82, 83), (83, 84), (83, 92), (84, 85), (85, 86), (86, 87),
    (87, 88), (87, 93), (88, 89), (90, 94), (91, 98), (92, 102),
    (93, 106), (94, 95), (95, 96), (96, 97), (96, 109), (97, 98),
    (98, 99), (99, 100), (100, 101), (100, 110), (101, 102), (102, 103),
    (103, 104), (104, 105), (104, 111), (105, 106), (106, 107), (107, 108),
    (108, 112), (109, 114), (110, 118), (111, 122), (112, 126), (113, 114),
    (114, 115), (115, 116), (116, 117), (117, 118), (118, 119), (119, 120),
    (120, 121), (121, 122), (122, 123), (123, 124), (124, 125), (125, 126),
];

/// [`HEAVY_HEX_127_EDGES`] as a `Vec`, for callers that want to own it.
pub fn heavy_hex_127_edges() -> Vec<(usize, usize)> {
    HEAVY_HEX_127_EDGES.to_vec()
}
