//! One layer applied to a [`HashSum`] in place, coset by coset.
//!
//! Each coset runs three passes: every source entry is rescaled in place by its identity amplitude while its other outputs go to per-member side buffers, the side buffers are accumulated into the member tables, and the touched tables are filtered.
//! The in-place rescale is safe because an entry's own slot is read only by itself; every cross-key contribution waits in a side buffer until the coset's first pass is done.

use std::sync::Mutex;
use std::time::Instant;

use num_complex::Complex64;
use rayon::prelude::*;

use super::fused::{local_source, pass1_local, pass1_rotation, pass2, rotation_source, FusedPlan, Row};
use super::orbit::{
    leaders_local, leaders_rotation, process_local, process_rotation, rescale_singletons, Leader,
    OrbitPlan,
};
use super::table::{accumulate, Entry, HashSum, Table};
use crate::channel::prepared::{LocalPtm, Prepared, PreparedRotation, LOCAL_DIM};
use crate::engine::coset::Gf2Span;
use crate::pauli_string::PauliString;
use crate::phase::Phase;
use crate::truncation::TruncationPolicy;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// How a layer moves contributions between keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// Rescale in place, buffer every cross-key row, accumulate the buffers, then filter.
    Scatter,
    /// Read, transform and write back each closed group of keys ([`orbit`](super::orbit)).
    Orbit,
    /// Rotations and Cliffords land each cross-key row with one lookup ([`fused`](super::fused)); any other channel runs [`Kernel::Orbit`].
    Fused,
}

/// Nanoseconds spent per pass, summed over cosets and workers.
#[derive(Clone, Copy, Debug, Default)]
pub struct PassTimes {
    /// In-place rescale plus side-buffer emission.
    pub emit_ns: u64,
    /// Side-buffer rows accumulated into the tables.
    pub accumulate_ns: u64,
    /// The filter pass.
    pub retain_ns: u64,
    /// Side-buffer rows, summed over layers.
    pub rows_emitted: u64,
}

impl PassTimes {
    fn absorb(&mut self, other: &Self) {
        self.emit_ns += other.emit_ns;
        self.accumulate_ns += other.accumulate_ns;
        self.retain_ns += other.retain_ns;
        self.rows_emitted += other.rows_emitted;
    }
}

/// One coset task's working set.
#[derive(Default)]
struct CosetWork<const W: usize> {
    tables: Vec<Table<W>>,
    side: Vec<Vec<Entry<W>>>,
    leaders: Vec<Leader<W>>,
    rows: Vec<Vec<Row<W>>>,
    betas: Vec<u32>,
    times: PassTimes,
}

/// Retained per-worker working sets for [`apply_layer_hashed`].
#[derive(Default)]
pub struct HashScratch<const W: usize> {
    serial: CosetWork<W>,
    workers: Vec<Mutex<CosetWork<W>>>,
}

impl<const W: usize> HashScratch<W> {
    /// An empty scratch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Drain the pass timers.
    pub fn take_times(&mut self) -> PassTimes {
        let mut total = std::mem::take(&mut self.serial.times);
        for worker in &self.workers {
            total.absorb(&std::mem::take(&mut worker.lock().unwrap().times));
        }
        total
    }
}

/// One non-identity output of a support pattern.
#[derive(Clone, Copy)]
struct Emit<const W: usize> {
    mask_x: [u64; W],
    mask_z: [u64; W],
    coordinate: u32,
    amplitude: Complex64,
}

/// A local PTM tabulated by input support pattern: the identity amplitude and the list of nonzero other outputs.
struct LocalPlan<'p, const W: usize> {
    ptm: &'p LocalPtm<W>,
    identity: [Complex64; LOCAL_DIM],
    /// Patterns whose only output is themselves at amplitude one, whose entries the emit pass leaves untouched.
    untouched: [bool; LOCAL_DIM],
    emits: Vec<Emit<W>>,
    offsets: [u16; LOCAL_DIM + 1],
}

impl<'p, const W: usize> LocalPlan<'p, W> {
    fn new(ptm: &'p LocalPtm<W>, span: &Gf2Span) -> Self {
        let mut identity = [ZERO; LOCAL_DIM];
        let mut emits = Vec::new();
        let mut offsets = [0u16; LOCAL_DIM + 1];
        for s in 0..LOCAL_DIM {
            for delta in ptm.deltas() {
                let amplitude = delta.amplitude[s];
                if amplitude == ZERO {
                    continue;
                }
                if delta.local_delta == 0 {
                    identity[s] = amplitude;
                } else {
                    emits.push(Emit {
                        mask_x: delta.mask_x,
                        mask_z: delta.mask_z,
                        coordinate: span.coordinate_of(delta.bucket_delta),
                        amplitude,
                    });
                }
            }
            offsets[s + 1] = emits.len() as u16;
        }
        let mut untouched = [false; LOCAL_DIM];
        for s in 0..LOCAL_DIM {
            untouched[s] = identity[s] == ONE && offsets[s] == offsets[s + 1];
        }
        Self { ptm, identity, untouched, emits, offsets }
    }
}

enum Plan<'p, const W: usize> {
    Local(LocalPlan<'p, W>),
    Rotation { rotation: &'p PreparedRotation<W>, generator_coordinate: u32 },
    OrbitLocal(OrbitPlan<'p, W>),
    OrbitRotation { rotation: &'p PreparedRotation<W>, generator_coordinate: u32 },
    FusedLocal(FusedPlan<'p, W>),
    FusedRotation { rotation: &'p PreparedRotation<W>, generator_coordinate: u32 },
}

/// Apply one prepared channel to `sum` in place, folding `keep_term` into the filter pass.
pub fn apply_layer_hashed<const W: usize, T>(
    sum: &mut HashSum<W>,
    prepared: &Prepared<W>,
    policy: &T,
    scratch: &mut HashScratch<W>,
    kernel: Kernel,
) where
    T: TruncationPolicy<W> + ?Sized,
{
    let span = Gf2Span::new(&prepared.bucket_deltas(), sum.hash.bits());
    let plan = match (prepared, kernel) {
        (Prepared::Local(ptm), Kernel::Scatter) => Plan::Local(LocalPlan::new(ptm, &span)),
        (Prepared::Local(ptm), Kernel::Orbit) => {
            Plan::OrbitLocal(OrbitPlan::new(ptm, &sum.hash, &span))
        }
        (Prepared::Rotation(rotation), Kernel::Scatter) => Plan::Rotation {
            rotation,
            generator_coordinate: span.coordinate_of(rotation.bucket_delta_generator),
        },
        (Prepared::Local(ptm), Kernel::Fused) => match FusedPlan::new(ptm, &sum.hash, &span) {
            Some(fused) => Plan::FusedLocal(fused),
            None => Plan::OrbitLocal(OrbitPlan::new(ptm, &sum.hash, &span)),
        },
        (Prepared::Rotation(rotation), Kernel::Fused) => Plan::FusedRotation {
            rotation,
            generator_coordinate: span.coordinate_of(rotation.bucket_delta_generator),
        },
        (Prepared::Rotation(rotation), Kernel::Orbit) => Plan::OrbitRotation {
            rotation,
            generator_coordinate: span.coordinate_of(rotation.bucket_delta_generator),
        },
    };
    let num_cosets = span.num_cosets();
    let buckets = DisjointSlots::new(&mut sum.buckets);
    if num_cosets < 2 {
        for coset in 0..num_cosets as u32 {
            run_coset(coset, &span, &buckets, &plan, policy, &mut scratch.serial);
        }
    } else {
        let pool = rayon::current_num_threads().max(1);
        if scratch.workers.len() < pool {
            scratch.workers.resize_with(pool, Mutex::default);
        }
        let workers = &scratch.workers;
        (0..num_cosets as u32).into_par_iter().for_each(|coset| {
            let worker = rayon::current_thread_index().expect("runs on a Rayon pool worker");
            let mut work = workers[worker].lock().unwrap();
            run_coset(coset, &span, &buckets, &plan, policy, &mut work);
        });
    }
    sum.recount();
}

fn run_coset<const W: usize, T>(
    coset: u32,
    span: &Gf2Span,
    buckets: &DisjointSlots<'_, Table<W>>,
    plan: &Plan<'_, W>,
    policy: &T,
    work: &mut CosetWork<W>,
) where
    T: TruncationPolicy<W> + ?Sized,
{
    let representative = span.representative_at(coset);
    let CosetWork { tables, side, leaders, rows, betas, times } = work;
    betas.clear();
    betas.extend((0..span.coset_size() as u32).map(|j| span.member(representative, j)));
    tables.resize_with(betas.len(), Table::default);
    side.resize_with(betas.len(), Vec::new);
    rows.resize_with(betas.len(), Vec::new);
    for (slot, &beta) in tables.iter_mut().zip(betas.iter()) {
        // SAFETY: the cosets partition the buckets, so no other task reaches bucket `beta`.
        std::mem::swap(slot, unsafe { buckets.get(beta as usize) });
    }

    let start = Instant::now();
    if WARM.load(std::sync::atomic::Ordering::Relaxed) {
        warm(tables);
    }
    match plan {
        Plan::Local(local) => emit_local(tables, side, local),
        Plan::Rotation { rotation, generator_coordinate } => {
            emit_rotation(tables, side, rotation, *generator_coordinate)
        }
        Plan::OrbitLocal(orbit) => {
            rescale_singletons(tables, orbit, policy);
            leaders.clear();
            leaders_local(tables, orbit, leaders);
            let found = Instant::now();
            process_local(tables, orbit, leaders, policy);
            finish_orbit(times, start, found, leaders.len());
            swap_back(tables, betas, buckets);
            return;
        }
        Plan::FusedLocal(fused) => {
            pass1_local(tables, rows, fused, policy);
            let found = Instant::now();
            let emitted: usize = rows.iter().map(Vec::len).sum();
            for j in 0..tables.len() {
                pass2(tables, j, &mut rows[j], local_source(fused), policy);
            }
            finish_orbit(times, start, found, emitted);
            swap_back(tables, betas, buckets);
            return;
        }
        Plan::FusedRotation { rotation, generator_coordinate } => {
            pass1_rotation(tables, rows, rotation, *generator_coordinate);
            let found = Instant::now();
            let emitted: usize = rows.iter().map(Vec::len).sum();
            for j in 0..tables.len() {
                pass2(
                    tables,
                    j,
                    &mut rows[j],
                    rotation_source(rotation, *generator_coordinate),
                    policy,
                );
            }
            finish_orbit(times, start, found, emitted);
            swap_back(tables, betas, buckets);
            return;
        }
        Plan::OrbitRotation { rotation, generator_coordinate } => {
            leaders.clear();
            leaders_rotation(tables, rotation, *generator_coordinate, leaders);
            let found = Instant::now();
            process_rotation(tables, rotation, *generator_coordinate, leaders, policy);
            finish_orbit(times, start, found, leaders.len());
            swap_back(tables, betas, buckets);
            return;
        }
    }
    let emitted = Instant::now();
    for (table, rows) in tables.iter_mut().zip(side.iter_mut()) {
        times.rows_emitted += rows.len() as u64;
        for e in rows.drain(..) {
            accumulate(table, e);
        }
    }
    let accumulated = Instant::now();
    for table in tables.iter_mut() {
        table.retain(|e| {
            ((e.c.re != 0.0) | (e.c.im != 0.0)) && policy.keep_term(&e.x, &e.z, e.c)
        });
    }
    let retained = Instant::now();
    times.emit_ns += (emitted - start).as_nanos() as u64;
    times.accumulate_ns += (accumulated - emitted).as_nanos() as u64;
    times.retain_ns += (retained - accumulated).as_nanos() as u64;
    swap_back(tables, betas, buckets);
}

/// Whether each coset streams its tables in before the random-access passes; a prototype switch.
pub static WARM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Stream every entry of the coset's tables through the cache, so the random lookups that follow hit in L2 instead of missing to DRAM.
fn warm<const W: usize>(tables: &[Table<W>]) {
    let mut fold = 0u64;
    for table in tables {
        for e in table.iter() {
            fold ^= e.x[0];
        }
    }
    std::hint::black_box(fold);
}

/// The orbit kernel's leader pass counts as `emit`, its write-back as `accumulate`.
fn finish_orbit(times: &mut PassTimes, start: Instant, found: Instant, leaders: usize) {
    let done = Instant::now();
    times.emit_ns += (found - start).as_nanos() as u64;
    times.accumulate_ns += (done - found).as_nanos() as u64;
    times.rows_emitted += leaders as u64;
}

fn swap_back<const W: usize>(
    tables: &mut [Table<W>],
    betas: &[u32],
    buckets: &DisjointSlots<'_, Table<W>>,
) {
    for (slot, &beta) in tables.iter_mut().zip(betas.iter()) {
        // SAFETY: the cosets partition the buckets, so no other task reaches bucket `beta`.
        std::mem::swap(slot, unsafe { buckets.get(beta as usize) });
    }
}

fn emit_local<const W: usize>(
    tables: &mut [Table<W>],
    side: &mut [Vec<Entry<W>>],
    plan: &LocalPlan<'_, W>,
) {
    // A `retain`, so an entry the identity delta does not keep frees its slot before the side rows land and a permutation runs at constant occupancy.
    for (i, table) in tables.iter_mut().enumerate() {
        table.retain(|e| {
            let s = plan.ptm.support_bits(&e.x, &e.z);
            if plan.untouched[s] {
                return true;
            }
            let c = e.c;
            e.c = c * plan.identity[s];
            let emits = &plan.emits[plan.offsets[s] as usize..plan.offsets[s + 1] as usize];
            for emit in emits {
                let mut x = e.x;
                let mut z = e.z;
                for w in 0..W {
                    x[w] ^= emit.mask_x[w];
                    z[w] ^= emit.mask_z[w];
                }
                side[i ^ emit.coordinate as usize].push(Entry { x, z, c: c * emit.amplitude });
            }
            plan.identity[s] != ZERO
        });
    }
}

fn emit_rotation<const W: usize>(
    tables: &mut [Table<W>],
    side: &mut [Vec<Entry<W>>],
    rotation: &PreparedRotation<W>,
    generator_coordinate: u32,
) {
    for (i, table) in tables.iter_mut().enumerate() {
        table.for_each_mut(|e| {
            let key = PauliString::<W> { x: e.x, z: e.z };
            if key.commutes_with(&rotation.generator) {
                return;
            }
            let c = e.c;
            e.c = c * rotation.cos;
            let mut product = key;
            let phase = product.mul_assign(&rotation.generator);
            side[i ^ generator_coordinate as usize].push(Entry {
                x: product.x,
                z: product.z,
                c: (Phase::I + phase).apply(c) * rotation.sin,
            });
        });
    }
}

/// A slice shared by tasks that each touch pairwise distinct elements.
struct DisjointSlots<'a, T> {
    base: *mut T,
    len: usize,
    _slice: std::marker::PhantomData<&'a mut [T]>,
}

// SAFETY: `get`'s contract keeps every element reached by at most one task at a time.
unsafe impl<T: Send> Sync for DisjointSlots<'_, T> {}

impl<'a, T> DisjointSlots<'a, T> {
    fn new(slice: &'a mut [T]) -> Self {
        Self { base: slice.as_mut_ptr(), len: slice.len(), _slice: std::marker::PhantomData }
    }

    /// # Safety
    ///
    /// No other reference to element `index` may be live while the returned one is.
    #[allow(clippy::mut_from_ref)]
    unsafe fn get(&self, index: usize) -> &mut T {
        assert!(index < self.len);
        // SAFETY: in bounds, and exclusive by the caller's contract.
        unsafe { &mut *self.base.add(index) }
    }
}
