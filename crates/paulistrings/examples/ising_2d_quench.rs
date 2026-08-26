//! 2D transverse-field Ising quench from `|+...+⟩` on 4×4 and 6×6 grids.
//!
//! Heisenberg-evolves the average X magnetization
//! `O = (1/N) · Σ_i X_i` under a first-order Trotterization of
//! `H = -J · Σ_⟨i,j⟩ Z_i Z_j - h · Σ_i X_i` (periodic boundary conditions)
//! and prints the expectation `⟨+...+| O(t) |+...+⟩` over a time grid.
//! Writes CSVs alongside this example for the companion plot script to
//! consume.
//!
//! Run with:
//! ```bash
//! cargo run --example ising_2d_quench --release
//! ```
//!
//! See `crates/paulistrings/docs/examples/ising_2d_quench.md` for the
//! full walkthrough and embedded plot.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use num_complex::Complex64;
use paulistrings::bucket::{
    desired_bits, BucketedSum, Gf2Hash, DEFAULT_HASH_SEED, DEFAULT_TARGET_BUCKET_LEN,
};
use paulistrings::channel::PauliRotation;
use paulistrings::engine::bucketed::LayerScratch;
use paulistrings::truncation::{And, CoefficientThreshold, TopN};
use paulistrings::{
    default_min_buckets, propagate_bucketed, BuildAccumulator, Circuit, Direction, PauliString,
    PauliSum, Phase, ProductState,
};

/// Ising couplings: H = -J · ΣZZ - h · ΣX.
const J: f64 = 1.0;
const H: f64 = 1.0;
/// Trotter step and total evolution time, in units of `1/J`.
const DT: f64 = 0.05;
const T_MAX: f64 = 2.0;
/// Coefficient-magnitude threshold below which terms are dropped.
const EPS: f64 = 1e-10;
/// Hard ceiling on terms kept per layer (largest-magnitude survive).
const TOPN_4X4: usize = 50_000;
const TOPN_6X6: usize = 200_000;

fn qubit_index(x: usize, y: usize, lx: usize) -> u32 {
    (y * lx + x) as u32
}

/// One Trotter step `U(δt) = exp(-i·δt·h·ΣX) · exp(-i·δt·J·ΣZZ)`
/// expressed as a `Circuit` of [`PauliRotation`]s.
///
/// `PauliRotation` is parameterised as `exp(-i·θ·P/2)`, so the angle is
/// `θ = 2 · J · dt` for ZZ bonds and `θ = 2 · h · dt` for X sites.
fn trotter_step(lx: usize, ly: usize, dt: f64) -> Circuit<1> {
    let n = lx * ly;
    assert!(n <= 64, "PauliString<1> covers up to 64 qubits");
    let mut circuit = Circuit::<1>::new(n);

    // ZZ rotations on every nearest-neighbour bond (PBC).
    for y in 0..ly {
        for x in 0..lx {
            let i = qubit_index(x, y, lx);
            let right = qubit_index((x + 1) % lx, y, lx);
            let down = qubit_index(x, (y + 1) % ly, lx);
            for partner in [right, down] {
                let mut gen = PauliString::<1>::z(i);
                gen.mul_assign(&PauliString::<1>::z(partner));
                circuit.push(PauliRotation::new(gen, 2.0 * J * dt));
            }
        }
    }

    // Transverse-field X rotations on every site.
    for site in 0..n as u32 {
        let gen = PauliString::<1>::x(site);
        circuit.push(PauliRotation::new(gen, 2.0 * H * dt));
    }

    circuit
}

/// Initial observable `O = (1/N) · Σ_i X_i` as a `PauliSum`.
fn x_magnetization(lx: usize, ly: usize) -> PauliSum<1> {
    let n = lx * ly;
    let inv_n = Complex64::new(1.0 / n as f64, 0.0);
    let mut acc = BuildAccumulator::<1>::new(n);
    for site in 0..n as u32 {
        acc.add_term(PauliString::<1>::x(site), Phase::ONE, inv_n);
    }
    acc.finalize()
}

/// `⟨+...+| O |+...+⟩` for a Pauli sum `O`.
///
/// `|+⟩^⊗N` is a `+1` eigenstate of every single-qubit `X`, so
/// `⟨+...+| P |+...+⟩` is `1` when every single-qubit factor of `P` is
/// either `I` or `X` and `0` otherwise. That is exactly
/// [`ProductState::XPlus`]; the observable is Hermitian here, so the
/// imaginary part is zero and we take `.re`.
///
/// This used to be hand-rolled against the raw SoA columns, because the crate
/// had no expectation-value API. It does as of v0.2 B.10.
fn expectation_plus_state(sum: &PauliSum<1>) -> f64 {
    sum.expectation_product_state(ProductState::XPlus).re
}

/// Drive a single quench: returns the `(t, m_x)` time series.
fn run_quench(lx: usize, ly: usize, topn: usize) -> (Vec<f64>, Vec<f64>) {
    let steps = (T_MAX / DT).round() as usize;
    let step_circuit = trotter_step(lx, ly, DT);
    let policy = And(CoefficientThreshold(EPS), TopN(topn));

    let initial = x_magnetization(lx, ly);
    let mut t_series = Vec::with_capacity(steps + 1);
    let mut m_series = Vec::with_capacity(steps + 1);
    t_series.push(0.0);
    m_series.push(expectation_plus_state(&initial));

    // Stay bucketed across all `steps` calls rather than using `propagate`,
    // which converts in and out each time. This is the shape the conversion cost
    // actually matters for: a short circuit (one Trotter step) applied many times
    // to a large sum. See `research/notes/2026-08-26-v0.2-results.md` §5.
    let num_qubits = lx * ly;
    let hash = Gf2Hash::<1>::new(
        num_qubits,
        desired_bits(
            initial.len(),
            DEFAULT_TARGET_BUCKET_LEN,
            default_min_buckets(),
        ),
        DEFAULT_HASH_SEED,
    );
    let mut observable = BucketedSum::from_sum(&initial, hash);
    let mut scratch = LayerScratch::<1>::new();

    for k in 1..=steps {
        propagate_bucketed(
            &step_circuit,
            &mut observable,
            &policy,
            Direction::Heisenberg,
            &mut scratch,
        );
        t_series.push(k as f64 * DT);
        m_series.push(observable.expectation_product_state(ProductState::XPlus).re);
        eprintln!(
            "  step {:>3}/{}  t = {:>5.2}  m_x = {:+.6}  terms = {}",
            k,
            steps,
            k as f64 * DT,
            m_series.last().unwrap(),
            observable.len(),
        );
    }

    (t_series, m_series)
}

fn output_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("output")
}

fn write_csv(path: &Path, t: &[f64], m: &[f64]) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "t,m_x")?;
    for (ti, mi) in t.iter().zip(m.iter()) {
        writeln!(w, "{:.6},{:.10}", ti, mi)?;
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    let out = output_dir();
    std::fs::create_dir_all(&out)?;

    eprintln!("== 4×4 Ising quench (J={}, h={}) ==", J, H);
    let t0 = Instant::now();
    let (t4, m4) = run_quench(4, 4, TOPN_4X4);
    eprintln!("  done in {:.1?}", t0.elapsed());
    let path_4x4 = out.join("ising_4x4.csv");
    write_csv(&path_4x4, &t4, &m4)?;
    eprintln!("  wrote {}", path_4x4.display());

    eprintln!("== 6×6 Ising quench (J={}, h={}) ==", J, H);
    let t0 = Instant::now();
    let (t6, m6) = run_quench(6, 6, TOPN_6X6);
    eprintln!("  done in {:.1?}", t0.elapsed());
    let path_6x6 = out.join("ising_6x6.csv");
    write_csv(&path_6x6, &t6, &m6)?;
    eprintln!("  wrote {}", path_6x6.display());

    eprintln!();
    eprintln!("Initial m_x:   4×4 = {:+.6}   6×6 = {:+.6}", m4[0], m6[0]);
    eprintln!(
        "Final   m_x:   4×4 = {:+.6}   6×6 = {:+.6}",
        m4.last().unwrap(),
        m6.last().unwrap(),
    );

    Ok(())
}
