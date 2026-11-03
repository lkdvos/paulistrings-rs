//! One layer applied to a [`HashSum`](super::HashSum) in place, coset by coset.
//!
//! A layer walks each coset's entries in position order; the first present member of a closed key group ([`GroupPlan`]) the walk reaches reads the group, applies its block, writes its present members in place and queues its absent ones.
//! A done bit per position marks the members already written, so each group is read once, with its input coefficients.
//! Truncated entries are removed after the walk and queued ones appended after that, so every position stays valid while it runs; truncation is exact because a group's outputs are final when written.
//! A monomial PTM (a Clifford) maps distinct keys to distinct keys, so it relabels with no lookup: an entry that keeps its key is rescaled in place, one that moves is removed and queued, and its new key is free once the removals are done; a diagonal one only rescales.


use std::sync::Mutex;
use std::time::Instant;

use num_complex::Complex64;
use rayon::prelude::*;

use super::plan::{monomial, GroupPlan, Plan};
use super::sum::HashSum;
use super::table::{key_hash, Entry, Table};
use crate::channel::prepared::{Prepared, PreparedRotation, LOCAL_DIM};
use crate::engine::coset::Gf2Span;
use crate::pauli_string::PauliString;
use crate::phase::Phase;
use crate::truncation::TruncationPolicy;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Nanoseconds per pass, summed over cosets and workers.
#[derive(Clone, Copy, Debug, Default)]
pub struct PassTimes {
    /// The walk, or the relabelling of a monomial layer.
    pub walk_ns: u64,
    /// Removing the marked entries and appending the queued ones.
    pub settle_ns: u64,
    /// Keys appended, summed over layers.
    pub appended: u64,
}

/// One coset task's working set.
#[derive(Default)]
struct CosetWork<const W: usize> {
    tables: Vec<Table<W>>,
    done: Vec<Vec<u64>>,
    dead: Vec<Vec<u64>>,
    pending: Vec<Vec<Entry<W>>>,
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
            let times = std::mem::take(&mut worker.lock().unwrap().times);
            total.walk_ns += times.walk_ns;
            total.settle_ns += times.settle_ns;
            total.appended += times.appended;
        }
        total
    }
}

/// Apply one prepared channel to `sum` in place, truncating each written term under `policy`.
pub fn apply_layer_hashed<const W: usize, T>(
    sum: &mut HashSum<W>,
    prepared: &Prepared<W>,
    policy: &T,
    scratch: &mut HashScratch<W>,
) where
    T: TruncationPolicy<W> + ?Sized,
{
    let span = Gf2Span::new(&prepared.bucket_deltas(), sum.hash.bits());
    let plan = match prepared {
        Prepared::Local(ptm) => {
            let plan = GroupPlan::new(ptm, &sum.hash, &span);
            match monomial(&plan) {
                Some((sigma, factor)) if sigma.iter().enumerate().all(|(p, &q)| p == q as usize) => {
                    Plan::Diagonal { plan, factor }
                }
                Some((sigma, factor)) => Plan::Monomial { plan, sigma, factor },
                None => Plan::Local(plan),
            }
        }
        Prepared::Rotation(rotation) => Plan::Rotation {
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
    let CosetWork { tables, done, dead, pending, betas, times } = work;
    betas.clear();
    betas.extend((0..span.coset_size() as u32).map(|j| span.member(representative, j)));
    let m = betas.len();
    tables.resize_with(m, Table::default);
    done.resize_with(m, Vec::new);
    dead.resize_with(m, Vec::new);
    pending.resize_with(m, Vec::new);
    swap(tables, betas, buckets);

    let start = Instant::now();
    clear_bits(tables, dead);
    match plan {
        Plan::Diagonal { plan, factor } => rescale(tables, dead, plan, factor, policy),
        Plan::Monomial { plan, sigma, factor } => relabel(tables, dead, pending, plan, sigma, factor, policy),
        Plan::Local(plan) => {
            clear_bits(tables, done);
            walk_local(tables, done, dead, pending, plan, policy);
        }
        Plan::Rotation { rotation, generator_coordinate } => {
            clear_bits(tables, done);
            walk_rotation(tables, done, dead, pending, rotation, *generator_coordinate, policy);
        }
    }
    let walked = Instant::now();
    for b in 0..m {
        times.appended += pending[b].len() as u64;
        settle(&mut tables[b], &mut dead[b], &mut pending[b]);
    }
    times.walk_ns += (walked - start).as_nanos() as u64;
    times.settle_ns += walked.elapsed().as_nanos() as u64;
    swap(tables, betas, buckets);
}

/// Remove the entries marked in `dead`, left cleared, and append the absent keys in `pending`, left empty.
/// When that touches more entries than stay, compacting and re-indexing is cheaper than removing and appending one by one.
fn settle<const W: usize>(table: &mut Table<W>, dead: &mut [u64], pending: &mut Vec<Entry<W>>) {
    let removed: usize = dead.iter().map(|word| word.count_ones() as usize).sum();
    let staying = table.len() - removed;
    if removed + pending.len() > staying {
        let mut entries = table.take_entries();
        let mut position = 0;
        entries.retain(|_| {
            position += 1;
            !is_set(dead, position - 1)
        });
        dead.fill(0);
        entries.append(pending);
        table.set_entries(entries);
    } else {
        table.remove_marked(dead);
        table.reserve(pending.len());
        for e in pending.drain(..) {
            table.push_unique(e);
        }
    }
}

fn swap<const W: usize>(
    tables: &mut [Table<W>],
    betas: &[u32],
    buckets: &DisjointSlots<'_, Table<W>>,
) {
    for (slot, &beta) in tables.iter_mut().zip(betas.iter()) {
        // SAFETY: the cosets partition the buckets, so no other task reaches bucket `beta`.
        std::mem::swap(slot, unsafe { buckets.get(beta as usize) });
    }
}

/// One cleared bit per position of each table.
fn clear_bits<const W: usize>(tables: &[Table<W>], bits: &mut [Vec<u64>]) {
    for (table, bits) in tables.iter().zip(bits.iter_mut()) {
        bits.clear();
        bits.resize(table.len().div_ceil(64), 0);
    }
}

#[inline(always)]
fn kept<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    x: &[u64; W],
    z: &[u64; W],
    c: Complex64,
    policy: &T,
) -> bool {
    ((c.re != 0.0) | (c.im != 0.0)) && policy.keep_term(x, z, c)
}

#[inline(always)]
fn is_set(bits: &[u64], position: usize) -> bool {
    (bits[position / 64] >> (position % 64)) & 1 != 0
}

#[inline(always)]
fn set(bits: &mut [u64], position: usize) {
    bits[position / 64] |= 1 << (position % 64);
}

/// The dense column pass: every entry scaled by its pattern's factor, the identity's being one.
fn rescale<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    tables: &mut [Table<W>],
    dead: &mut [Vec<u64>],
    plan: &GroupPlan<'_, W>,
    factor: &[Complex64; LOCAL_DIM],
    policy: &T,
) {
    for (table, dead) in tables.iter_mut().zip(dead.iter_mut()) {
        for (position, (x, z, c)) in table.coefficients_mut().enumerate() {
            *c *= factor[plan.ptm.support_bits(x, z)];
            if !kept(x, z, *c, policy) {
                set(dead, position);
            }
        }
    }
}

/// Rescale the entries that keep their key, and mark the others dead, queueing them relabelled in `pending` of their new table.
fn relabel<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    tables: &mut [Table<W>],
    dead: &mut [Vec<u64>],
    pending: &mut [Vec<Entry<W>>],
    plan: &GroupPlan<'_, W>,
    sigma: &[u8; LOCAL_DIM],
    factor: &[Complex64; LOCAL_DIM],
    policy: &T,
) {
    for b in 0..tables.len() {
        for (position, (x, z, c)) in tables[b].coefficients_mut().enumerate() {
            let p = plan.ptm.support_bits(x, z);
            let q = sigma[p] as usize;
            let scaled = *c * factor[p];
            if q == p {
                *c = scaled;
                if !kept(x, z, scaled, policy) {
                    set(&mut dead[b], position);
                }
                continue;
            }
            set(&mut dead[b], position);
            let (j, mut moved) = plan.shifted(b, x, z, p ^ q);
            moved.c = scaled;
            if kept(&moved.x, &moved.z, scaled, policy) {
                pending[j].push(moved);
            }
        }
    }
}

fn walk_local<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    tables: &mut [Table<W>],
    done: &mut [Vec<u64>],
    dead: &mut [Vec<u64>],
    pending: &mut [Vec<Entry<W>>],
    plan: &GroupPlan<'_, W>,
    policy: &T,
) {
    let mut keys = [(0usize, Entry { x: [0; W], z: [0; W], c: ZERO }); LOCAL_DIM];
    let mut at = [None::<usize>; LOCAL_DIM];
    for b in 0..tables.len() {
        for i in 0..tables[b].len() {
            if is_set(&done[b], i) {
                continue;
            }
            let e = tables[b].entries()[i];
            let p = plan.ptm.support_bits(&e.x, &e.z);
            let component = plan.component[p] as usize;
            if plan.skip[component] {
                continue;
            }
            let n = plan.sizes[component] as usize;
            let block = &plan.blocks[plan.offsets[component]..plan.offsets[component] + n * n];
            if n == 1 {
                let c = e.c * block[0];
                if kept(&e.x, &e.z, c, policy) {
                    *tables[b].coefficient_mut(i) = c;
                } else {
                    set(&mut dead[b], i);
                }
                continue;
            }
            for a in 0..n {
                let q = plan.members[component][a] as usize;
                if q == p {
                    keys[a] = (b, e);
                    at[a] = Some(i);
                    continue;
                }
                let (j, mut key) = plan.shifted(b, &e.x, &e.z, p ^ q);
                at[a] = tables[j].find(key_hash(&key.x, &key.z), &key.x, &key.z);
                if let Some(position) = at[a] {
                    key.c = tables[j].entries()[position].c;
                    set(&mut done[j], position);
                }
                keys[a] = (j, key);
            }
            for a in 0..n {
                let mut c = ZERO;
                for m in 0..n {
                    c += block[a * n + m] * keys[m].1.c;
                }
                let (j, key) = keys[a];
                match (at[a], kept(&key.x, &key.z, c, policy)) {
                    (Some(position), true) => *tables[j].coefficient_mut(position) = c,
                    (Some(position), false) => set(&mut dead[j], position),
                    (None, true) => pending[j].push(Entry { c, ..key }),
                    (None, false) => {}
                }
            }
        }
    }
}

fn walk_rotation<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    tables: &mut [Table<W>],
    done: &mut [Vec<u64>],
    dead: &mut [Vec<u64>],
    pending: &mut [Vec<Entry<W>>],
    rotation: &PreparedRotation<W>,
    generator_coordinate: u32,
    policy: &T,
) {
    let generator = &rotation.generator;
    for b in 0..tables.len() {
        let j = b ^ generator_coordinate as usize;
        for i in 0..tables[b].len() {
            if is_set(&done[b], i) {
                continue;
            }
            let e = tables[b].entries()[i];
            let v = PauliString::<W> { x: e.x, z: e.z };
            if v.commutes_with(generator) {
                continue;
            }
            let mut w = v;
            let phase_v_to_w = Phase::I + w.mul_assign(generator);
            let mut back = w;
            let phase_w_to_v = Phase::I + back.mul_assign(generator);
            let partner = tables[j].find(key_hash(&w.x, &w.z), &w.x, &w.z);
            let cw = match partner {
                Some(position) => {
                    set(&mut done[j], position);
                    tables[j].entries()[position].c
                }
                None => ZERO,
            };
            let new_v = e.c * rotation.cos + phase_w_to_v.apply(cw) * rotation.sin;
            let new_w = cw * rotation.cos + phase_v_to_w.apply(e.c) * rotation.sin;
            if kept(&v.x, &v.z, new_v, policy) {
                *tables[b].coefficient_mut(i) = new_v;
            } else {
                set(&mut dead[b], i);
            }
            match (partner, kept(&w.x, &w.z, new_w, policy)) {
                (Some(position), true) => *tables[j].coefficient_mut(position) = new_w,
                (Some(position), false) => set(&mut dead[j], position),
                (None, true) => pending[j].push(Entry { x: w.x, z: w.z, c: new_w }),
                (None, false) => {}
            }
        }
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
