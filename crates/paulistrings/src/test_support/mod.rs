//! Shared fixtures for the crate's own tests, benches and examples, re-exported flat from the submodules.
//!
//! Compiled only under `cfg(test)` or the `test-utils` feature (the crate's self-dev-dependency); `#[doc(hidden)]`, not public API.

mod circuits;
mod coefficient;
mod compare;
#[cfg(feature = "cuda")]
mod gpu;
mod oracle;
mod random;
mod rows;
mod transport;

pub use circuits::*;
pub use coefficient::*;
pub use compare::*;
#[cfg(feature = "cuda")]
pub use gpu::{
    assert_invariants_device, from_host_with_nvrtc_options, refine_to, set_fault_plan, DeviceSplit,
    FaultPlan, WireFault,
};
pub use oracle::*;
pub use random::*;
pub use rows::{circuit_generators, count_remote_deltas, GeneratorWeight};
pub use transport::*;

pub use crate::engine::bucketed::apply_layer_bucketed;
pub use crate::engine::partitioned::in_process::InProcessTransport;

pub use crate::pauli_sum::storage::{
    desired_bits, rebucketed_bits, DEFAULT_HASH_SEED, DEFAULT_MIN_BUCKETS,
    DEFAULT_TARGET_BUCKET_LEN,
};

/// Rough estimate of the cost of one `Instant::now()` read, in nanoseconds, for this hardware class; used by the `phase_breakdown` probe's overhead line (`timer_reads() * TIMER_READ_OVERHEAD_NS`).
#[cfg(feature = "phase-timing")]
pub const TIMER_READ_OVERHEAD_NS: u64 = 25;

const ZERO: num_complex::Complex64 = num_complex::Complex64::new(0.0, 0.0);

/// Returns early from the calling test when no CUDA device is available, so a device test is never `#[ignore]`d — it just does nothing on a box without a GPU.
/// Otherwise it holds a guard on [`DEVICE_TESTS`] for the rest of the test: shared by default, `require_cuda!(exclusive)` for a test whose wall bounds and wire timeouts must not compete with other device tests for the driver.
#[cfg(feature = "cuda")]
#[macro_export]
macro_rules! require_cuda {
    () => { $crate::require_cuda!(@read) };
    (exclusive) => { $crate::require_cuda!(@write) };
    (@$lock:ident) => {
        if !$crate::gpu::cuda_available() {
            return;
        }
        let _device_tests = $crate::test_support::DEVICE_TESTS
            .$lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    };
}

/// Orders the device tests of one test binary; see [`require_cuda`].
#[cfg(feature = "cuda")]
pub static DEVICE_TESTS: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// `transport` with its largest message capped at `bytes`, so a test reaches the chunked exchange.
#[cfg(feature = "mpi")]
pub fn mpi_with_chunk_bytes(
    transport: crate::mpi::MpiTransport,
    bytes: usize,
) -> crate::mpi::MpiTransport {
    transport.with_chunk_bytes(bytes)
}

/// The MPI library version string this build probed through `mpicc`.
#[cfg(feature = "mpi")]
pub fn mpi_build_library_version() -> &'static str {
    crate::mpi::MpiTransport::build_library_version()
}

/// A cut of `sum` at its median octave keeping about half of that octave, the first terms of it in bucket order, as an `ApproxTopN` layer pass would leave pending.
#[cfg(test)]
pub(crate) fn median_octave_cut<const W: usize, C: crate::Coefficient>(
    sum: &crate::PauliSum<W, C>,
) -> crate::truncation::octave_window::BoundaryCut {
    use crate::truncation::builtin::{octave, QuotaSplit};
    let mut octaves: Vec<usize> = sum.iter().map(|(_, _, c)| octave(c.norm_sqr())).collect();
    octaves.sort_unstable();
    let bin = octaves.get(octaves.len() / 2).copied().unwrap_or(0);
    let mut boundary_counts = Vec::with_capacity(sum.num_buckets());
    let mut above = 0usize;
    for b in 0..sum.num_buckets() {
        let coefficients = sum.bucket(b).2;
        boundary_counts.push(coefficients.iter().filter(|c| octave(c.norm_sqr()) == bin).count());
        above += coefficients.iter().filter(|c| octave(c.norm_sqr()) > bin).count();
    }
    let total = boundary_counts.iter().sum::<usize>();
    let quota = total / 2;
    let split = QuotaSplit::new(|b| boundary_counts[b], boundary_counts.len(), total, quota);
    crate::truncation::octave_window::BoundaryCut { bin, split, kept: above + quota }
}

#[cfg(test)]
mod tests;
