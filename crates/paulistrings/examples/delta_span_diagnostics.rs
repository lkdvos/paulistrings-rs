//! Deterministic diagnostic for the dense-PTM sort: how presorted is a gather
//! run's *rest* stream, and how many comparisons does the per-run sort make?
//!
//! This is the instrument behind `research/notes/2026-09-01-bucket-cliff.md`.
//! It exists because the quantity that governs the dense-PTM sort's cost —
//! `merge::sort_rows_with_scratch`'s comparison count — is a **deterministic**
//! function of the configuration, while the wall-clock time it produces is not:
//! single-shot timings on the reference host move ±5–8% single-threaded
//! (`benchmarks/PROFILING.md`), which is far too coarse to tell the
//! configurations apart at the small end. Comparison counts have no noise at
//! all, and measured `ns / comparison` is flat to ±15% across the whole 3.3×
//! configuration spread, so this is the better instrument.
//!
//! Run with:
//! ```bash
//! cargo run --release -p paulistrings --example delta_span_diagnostics
//! ```
//!
//! # What it reports, per configuration
//!
//! - `bits` / `B` — the partition the engine's own `desired_bits` policy picked
//!   (or `--bucket-bits` forced, mirrored here by `force_bits`).
//! - `rank` — `rank(h(D))` for the layer's key-delta space, via
//!   `test_support::support_delta_rank`.
//! - `m` — the coset width `2^r`, `r = min(rank, bits)`; the engine's
//!   `GATHER_OUTPUT_MAJOR_MIN_R = 3` switches gather order at `m = 8`.
//! - `runs` — maximal ascending runs in one member's rest stream. Rust's stable
//!   `sort_by` is driftsort, which *detects and merges* natural ascending runs;
//!   `sort_rows_with_scratch`'s doc comment records that switching to
//!   `sort_unstable_by` cost +77%, i.e. that adaptivity is the whole design.
//! - `cmp/row` — comparisons the sort performs per row. Its floor for a
//!   `k`-way merge of sorted runs is `log2(k) + 1`; a dense two-qubit PTM has
//!   15 non-identity delta streams, so the floor is ≈ 4.9.
//!
//! # Caveat
//!
//! The two gather orders are *re-implemented* here from
//! `engine::bucketed::gather_local_{input,output}_major`, because those are
//! private and the whole point is to compare orders the engine would not
//! choose. The row *multiset* is the same as the engine's; only per-member
//! amplitude filtering is reproduced, not the engine's scratch reuse. If the
//! gather in `engine/bucketed.rs` changes shape, this example has to follow —
//! the numbers in the note above were taken against the code as of the commit
//! that added it.

use num_complex::Complex64;
use paulistrings::bucket::sum::DEFAULT_HASH_SEED;
use paulistrings::channel::prepared::{LocalPtm, Prepared};
use paulistrings::channel::{Channel, GeneralUnitary2Q};
use paulistrings::test_support::{haar_su4_matrix, rand_sum, support_delta_rank};
use paulistrings::truncation::TruncationPolicy;
use paulistrings::{Circuit, Direction, Gf2Hash, PauliSum};

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// The support the probe's `su4` layer uses (`phase_breakdown::build_circuit`).
const SUPPORT: [u32; 2] = [0, 1];

struct Keep;
impl<const W: usize> TruncationPolicy<W> for Keep {}

// ---------------------------------------------------------------------
// GF(2) span, mirroring `engine::coset::Gf2Span::new`
// ---------------------------------------------------------------------

fn highest_bit(v: u32) -> u32 {
    31 - v.leading_zeros()
}

/// Software `pext`, as `engine::coset::pext`.
fn pext(value: u32, mask: u32) -> u32 {
    let (mut out, mut k, mut m) = (0u32, 0u32, mask);
    while m != 0 {
        let bit = m & m.wrapping_neg();
        if value & bit != 0 {
            out |= 1 << k;
        }
        k += 1;
        m ^= bit;
    }
    out
}

/// Reduced echelon basis (ascending pivot) plus pivot mask.
fn span(deltas: &[u32]) -> (Vec<u32>, u32) {
    let mut basis: Vec<u32> = Vec::new();
    let mut pivot_mask = 0u32;
    for &d in deltas {
        let mut v = d;
        for &b in &basis {
            if v & (1 << highest_bit(b)) != 0 {
                v ^= b;
            }
        }
        if v == 0 {
            continue;
        }
        let p = highest_bit(v);
        for b in basis.iter_mut() {
            if *b & (1 << p) != 0 {
                *b ^= v;
            }
        }
        let idx = basis.partition_point(|&b| highest_bit(b) < p);
        basis.insert(idx, v);
        pivot_mask |= 1 << p;
    }
    (basis, pivot_mask)
}

// ---------------------------------------------------------------------
// Run structure and comparison count
// ---------------------------------------------------------------------

type Key<const W: usize> = ([u64; W], [u64; W]);

/// Maximal non-decreasing runs — what driftsort's run detection finds.
fn ascending_runs<const W: usize>(keys: &[Key<W>]) -> usize {
    if keys.is_empty() {
        return 0;
    }
    1 + keys.windows(2).filter(|w| w[1] < w[0]).count()
}

/// Comparisons `sort_rows_with_scratch`'s `perm.sort_by` performs, counted by
/// running the identical sort through a counting comparator.
fn sort_comparisons<const W: usize>(keys: &[Key<W>]) -> u64 {
    let mut perm: Vec<u32> = (0..keys.len() as u32).collect();
    let mut n = 0u64;
    perm.sort_by(|&a, &b| {
        n += 1;
        keys[a as usize]
            .0
            .cmp(&keys[b as usize].0)
            .then_with(|| keys[a as usize].1.cmp(&keys[b as usize].1))
    });
    n
}

/// One member's rest stream in the engine's two gather orders.
///
/// `member[i]` is the bucket index of coset-0 member `i`; `coords[e]` is delta
/// `e`'s coset coordinate. Entry 0 is the identity delta, which the engine
/// routes into the pre-sorted `id` columns and never sorts, so it is skipped.
fn rest_stream<const W: usize>(
    sum: &PauliSum<W>,
    ptm: &LocalPtm<W>,
    member: &[usize],
    coords: &[usize],
    j: usize,
    output_major: bool,
) -> Vec<Key<W>> {
    let mut keys: Vec<Key<W>> = Vec::new();
    let emit = |keys: &mut Vec<Key<W>>, src: usize, e: usize| {
        let d = &ptm.deltas()[e];
        let (bx, bz, _) = sum.bucket(src);
        for t in 0..bx.len() {
            if d.amp[ptm.support_bits(&bx[t], &bz[t])] == ZERO {
                continue;
            }
            let mut kx = bx[t];
            let mut kz = bz[t];
            for w in 0..W {
                kx[w] ^= d.mask_x[w];
                kz[w] ^= d.mask_z[w];
            }
            keys.push((kx, kz));
        }
    };
    if output_major {
        // (delta, input position): one contiguous block per delta.
        for e in 1..ptm.deltas().len() {
            emit(&mut keys, member[j ^ coords[e]], e);
        }
    } else {
        // (input member, input position, delta): the deltas sharing a coset
        // coordinate interleave row by row.
        for (i, &src) in member.iter().enumerate() {
            let mine: Vec<usize> = (1..ptm.deltas().len())
                .filter(|&e| i ^ coords[e] == j)
                .collect();
            let (bx, bz, _) = sum.bucket(src);
            for t in 0..bx.len() {
                let s = ptm.support_bits(&bx[t], &bz[t]);
                for &e in &mine {
                    let d = &ptm.deltas()[e];
                    if d.amp[s] == ZERO {
                        continue;
                    }
                    let mut kx = bx[t];
                    let mut kz = bz[t];
                    for w in 0..W {
                        kx[w] ^= d.mask_x[w];
                        kz[w] ^= d.mask_z[w];
                    }
                    keys.push((kx, kz));
                }
            }
        }
    }
    keys
}

struct Cell {
    terms: usize,
    bits: u8,
    rank: usize,
    m: usize,
    ndelta: usize,
    /// `(rows, runs, comparisons)` for input-major then output-major.
    orders: [(usize, usize, u64); 2],
}

fn measure<const W: usize>(
    qubits: usize,
    seed_terms: usize,
    hash_seed: u64,
    force_bits: u8,
) -> Cell {
    // The probe's own input, driven to the layer's closed key set by two
    // untruncated passes — the fixed point every steady-state number is taken
    // at (`phase_breakdown::run_cell`'s warm-up does the same thing).
    let mut base = rand_sum::<W>(seed_terms, qubits, 0xC0FFEE);
    if hash_seed != DEFAULT_HASH_SEED {
        let nq = base.num_qubits();
        base = base.with_hash(Gf2Hash::<W>::new(nq, 0, hash_seed));
    }
    while base.hash().bits() < force_bits {
        base.refine();
    }
    let ch = GeneralUnitary2Q::from_matrix(SUPPORT[0], SUPPORT[1], haar_su4_matrix());
    let mut circuit = Circuit::<W>::new(qubits);
    circuit.push(ch);
    let mut sum = base;
    for _ in 0..2 {
        sum = paulistrings::propagate(&circuit, sum, &Keep, Direction::Forward);
    }

    let bits = sum.hash().bits();
    let ch = GeneralUnitary2Q::from_matrix(SUPPORT[0], SUPPORT[1], haar_su4_matrix());
    let prep = Channel::<W>::prepare(&ch, sum.hash(), false).expect("prepare");
    let ptm = match &prep {
        Prepared::Local(p) => p,
        _ => panic!("a GeneralUnitary2Q must prepare to a Local plan"),
    };
    let (basis, pivot_mask) = span(&prep.bucket_deltas());
    let m = 1usize << basis.len();
    // Coset 0: member `i` is the basis combination indexed by `i`.
    let member: Vec<usize> = (0..m)
        .map(|i| {
            let mut b = 0u32;
            for (j, &bv) in basis.iter().enumerate() {
                if i >> j & 1 == 1 {
                    b ^= bv;
                }
            }
            b as usize
        })
        .collect();
    let coords: Vec<usize> = ptm
        .deltas()
        .iter()
        .map(|d| pext(d.bucket_delta, pivot_mask) as usize)
        .collect();

    let mut orders = [(0usize, 0usize, 0u64); 2];
    for (i, output_major) in [false, true].into_iter().enumerate() {
        let keys = rest_stream(&sum, ptm, &member, &coords, 0, output_major);
        orders[i] = (keys.len(), ascending_runs(&keys), sort_comparisons(&keys));
    }

    Cell {
        terms: sum.len(),
        bits,
        rank: support_delta_rank(sum.hash(), &SUPPORT),
        m,
        ndelta: ptm.deltas().len(),
        orders,
    }
}

fn header() {
    println!(
        "{:<12} {:>4} {:>2} {:>8} {:>4} {:>4} {:>3} {:>3} | {:>7} {:>7} {:>7} | {:>7} {:>7}",
        "case",
        "q",
        "W",
        "terms",
        "bits",
        "rank",
        "m",
        "|D|",
        "rows",
        "runs_i",
        "cmp/row_i",
        "runs_o",
        "cmp/row_o",
    );
}

fn row(case: &str, qubits: usize, seed_terms: usize, hash_seed: u64, force_bits: u8) {
    let c = if qubits <= 64 {
        measure::<1>(qubits, seed_terms, hash_seed, force_bits)
    } else {
        measure::<2>(qubits, seed_terms, hash_seed, force_bits)
    };
    let w = if qubits <= 64 { 1 } else { 2 };
    let (rows, runs_i, cmp_i) = c.orders[0];
    let (_, runs_o, cmp_o) = c.orders[1];
    let per = |n: u64| n as f64 / rows.max(1) as f64;
    println!(
        "{case:<12} {qubits:>4} {w:>2} {:>8} {:>4} {:>4} {:>3} {:>3} | {rows:>7} {runs_i:>7} \
         {:>9.2} | {runs_o:>7} {:>9.2}",
        c.terms,
        c.bits,
        c.rank,
        c.m,
        c.ndelta,
        per(cmp_i),
        per(cmp_o),
    );
}

fn main() {
    let seed1 = DEFAULT_HASH_SEED ^ 0x1234_5678_9ABC_DEF1;

    println!("== dense-PTM (Haar SU(4) on qubits 0,1) rest stream, coset-0 member 0");
    println!("   cmp/row floor for a 15-way merge of sorted runs is log2(15)+1 = 4.9\n");

    println!("-- the engine's own policy, W = 2 (q = 128), across the fact sheet's cliff grid");
    header();
    for n in [70usize, 140, 280, 560, 700, 1120, 4480] {
        row("policy", 128, n, DEFAULT_HASH_SEED, 0);
    }

    println!("\n-- W = 1 (q = 64), same policy: rank 3 at every bucket count it reaches");
    header();
    for n in [70usize, 280, 560, 1120, 4480] {
        row("policy", 64, n, DEFAULT_HASH_SEED, 0);
    }

    println!("\n-- W = 1 (q = 64) with a full-rank hash seed: the rank/width confound, broken");
    header();
    for n in [1120usize, 4480] {
        row("seed-r3", 64, n, DEFAULT_HASH_SEED, 0);
        row("seed-r4", 64, n, seed1, 0);
    }

    println!("\n-- forced bucket-bits sweep at 980 terms, W = 2 (what --bucket-bits measures)");
    header();
    for b in 0..=9u8 {
        row("bits", 128, 70, DEFAULT_HASH_SEED, b);
    }
}
