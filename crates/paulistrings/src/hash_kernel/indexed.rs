//! [`IndexedSum`], whose buckets are dense entry lists behind a hash index of positions, and the layer kernel that updates them in place.
//!
//! A layer walks each coset's entries in position order; the first present member of a closed key group ([`OrbitPlan`]) the walk reaches reads the group, applies its block, writes its present members in place and queues its absent ones.
//! A done bit per position marks the members already written, so each group is read once, with its input coefficients.
//! Truncated entries are removed after the walk and queued ones appended after that, so every position stays valid while it runs; truncation is exact because a group's outputs are final when written.
//! A monomial PTM (a Clifford) maps distinct keys to distinct keys, so it relabels every entry with no lookup and re-indexes; a diagonal one only rescales.

use std::sync::Mutex;
use std::time::Instant;

use num_complex::Complex64;
use rayon::prelude::*;

use super::indexed_table::IndexedTable;
use super::layer::{DisjointSlots, PassTimes};
use super::orbit::OrbitPlan;
use super::table::{key_hash, Entry};
use crate::channel::prepared::{Prepared, PreparedRotation, LOCAL_DIM};
use crate::channel::Prepare;
use crate::circuit::Circuit;
use crate::engine::coset::Gf2Span;
use crate::engine::Direction;
use crate::pauli_string::PauliString;
use crate::pauli_sum::storage::rebucketed_bits;
use crate::pauli_sum::{Gf2Hash, PauliSum};
use crate::phase::Phase;
use crate::truncation::TruncationPolicy;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// A weighted sum of Pauli strings partitioned by a GF(2)-linear hash into indexed dense buckets, updated in place by each layer.
pub struct IndexedSum<const W: usize> {
    pub(crate) buckets: Vec<IndexedTable<W>>,
    pub(crate) hash: Gf2Hash<W>,
    pub(crate) num_qubits: usize,
    pub(crate) len: usize,
}

impl<const W: usize> IndexedSum<W> {
    /// The terms of `sum`, under its hash and bucket count.
    pub fn from_pauli_sum(sum: &PauliSum<W>) -> Self {
        let buckets = sum
            .buckets()
            .par_iter()
            .map(|columns| {
                IndexedTable::from_entries(
                    (0..columns.len())
                        .map(|i| Entry { x: columns.x[i], z: columns.z[i], c: columns.coeff[i] })
                        .collect(),
                )
            })
            .collect();
        Self { buckets, hash: sum.hash().clone(), num_qubits: sum.num_qubits(), len: sum.len() }
    }

    /// The terms as a canonical [`PauliSum`] under the same hash.
    pub fn to_pauli_sum(&self) -> PauliSum<W> {
        let mut terms: Vec<Entry<W>> =
            self.buckets.iter().flat_map(|table| table.entries().iter().copied()).collect();
        terms.par_sort_unstable_by(|a, b| (a.x, a.z).cmp(&(b.x, b.z)));
        let x: Vec<[u64; W]> = terms.iter().map(|e| e.x).collect();
        let z: Vec<[u64; W]> = terms.iter().map(|e| e.z).collect();
        let c: Vec<Complex64> = terms.iter().map(|e| e.c).collect();
        PauliSum::from_key_sorted(&x, &z, &c, self.hash.clone(), self.num_qubits)
    }

    /// Number of terms.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the sum has no terms.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of buckets.
    pub fn num_buckets(&self) -> usize {
        self.buckets.len()
    }

    /// Resident bytes of entries and indices.
    pub fn bytes(&self) -> usize {
        self.buckets.iter().map(IndexedTable::bytes).sum()
    }

    /// The partitioning hash.
    pub fn hash(&self) -> &Gf2Hash<W> {
        &self.hash
    }

    fn retain(&mut self, keep: impl Fn(&[u64; W], &[u64; W], Complex64) -> bool + Sync) {
        self.buckets.par_iter_mut().for_each(|table| table.retain(|e| keep(&e.x, &e.z, e.c)));
        self.recount();
    }

    fn recount(&mut self) {
        self.len = self.buckets.iter().map(IndexedTable::len).sum();
    }

    /// Double the bucket count by the hash's next row.
    fn refine(&mut self) {
        self.hash.refine();
        let new_bit = self.hash.bits() - 1;
        let hash = &self.hash;
        let upper: Vec<IndexedTable<W>> = self
            .buckets
            .par_iter_mut()
            .map(|table| table.split_off_where(|e| (hash.bucket_of(&e.x, &e.z) >> new_bit) & 1 == 1))
            .collect();
        self.buckets.extend(upper);
    }

    /// Halve the bucket count, merging bucket pairs `(i, i + B/2)`.
    fn coarsen(&mut self) {
        self.hash.coarsen();
        let half = self.buckets.len() / 2;
        let upper = self.buckets.split_off(half);
        self.buckets.par_iter_mut().zip(upper).for_each(|(lower, upper)| lower.absorb(upper));
    }

    fn rebucket(&mut self, target: usize, min_buckets: usize) {
        let want = rebucketed_bits(self.len, self.hash.bits(), target, min_buckets)
            .min(self.hash.max_bits());
        while self.hash.bits() < want {
            self.refine();
        }
        while self.hash.bits() > want {
            self.coarsen();
        }
    }
}

/// Bucket sizing for [`propagate_indexed`]; any `min_buckets >= 1` is accepted.
#[derive(Clone, Copy, Debug)]
pub struct IndexedOptions {
    /// Terms per bucket the rebucketing targets.
    pub target_bucket_len: usize,
    /// Floor on the bucket count once the sum is worth splitting.
    pub min_buckets: usize,
}

/// Propagate `sum` through `circuit` in place under a per-term `policy`.
///
/// # Panics
///
/// If `policy` has a layer pass, or a channel's preparation declines.
pub fn propagate_indexed<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: &mut IndexedSum<W>,
    policy: &T,
    direction: Direction,
    scratch: &mut IndexedScratch<W>,
    options: IndexedOptions,
) where
    T: TruncationPolicy<W> + ?Sized,
{
    assert!(!policy.finalizes_layer(), "the hash-kernel prototype supports per-term policies only");
    let adjoint = matches!(direction, Direction::Heisenberg);
    // An entry no group touches is never re-filtered, so the input must already pass.
    sum.retain(|x, z, c| policy.keep_term(x, z, c));
    let n = circuit.channels.len();
    for application_index in 0..n {
        let circuit_index = match direction {
            Direction::Forward => application_index,
            Direction::Heisenberg => n - 1 - application_index,
        };
        sum.rebucket(options.target_bucket_len, options.min_buckets);
        let prepared = circuit.channels[circuit_index]
            .prepare(&sum.hash, adjoint)
            .expect("channel preparation declined");
        apply_layer_indexed(sum, &prepared, policy, scratch);
    }
}

/// One coset task's working set.
#[derive(Default)]
struct IndexedWork<const W: usize> {
    tables: Vec<IndexedTable<W>>,
    done: Vec<Vec<u64>>,
    dead: Vec<Vec<u64>>,
    pending: Vec<Vec<Entry<W>>>,
    staying: Vec<Vec<Entry<W>>>,
    betas: Vec<u32>,
    times: PassTimes,
}

/// Retained per-worker working sets for [`apply_layer_indexed`].
#[derive(Default)]
pub struct IndexedScratch<const W: usize> {
    serial: IndexedWork<W>,
    workers: Vec<Mutex<IndexedWork<W>>>,
}

impl<const W: usize> IndexedScratch<W> {
    /// An empty scratch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Drain the pass timers: `emit` is the walk, `retain` the removals, `accumulate` the appends or the re-indexing.
    pub fn take_times(&mut self) -> PassTimes {
        let mut total = std::mem::take(&mut self.serial.times);
        for worker in &self.workers {
            let times = std::mem::take(&mut worker.lock().unwrap().times);
            total.emit_ns += times.emit_ns;
            total.accumulate_ns += times.accumulate_ns;
            total.retain_ns += times.retain_ns;
            total.rows_emitted += times.rows_emitted;
        }
        total
    }
}

enum Plan<'p, const W: usize> {
    /// Every pattern maps to itself: a rescale walk with no lookups.
    Diagonal { plan: OrbitPlan<'p, W>, factor: [Complex64; LOCAL_DIM] },
    /// Pattern `p` maps to `sigma[p]` alone, a bijection: a relabelling walk with no lookups.
    Monomial { plan: OrbitPlan<'p, W>, sigma: [u8; LOCAL_DIM], factor: [Complex64; LOCAL_DIM] },
    Local(OrbitPlan<'p, W>),
    Rotation { rotation: &'p PreparedRotation<W>, generator_coordinate: u32 },
}

/// The permutation and factors of a PTM taking each pattern to exactly one pattern, bijectively.
fn monomial<const W: usize>(
    plan: &OrbitPlan<'_, W>,
) -> Option<([u8; LOCAL_DIM], [Complex64; LOCAL_DIM])> {
    let dim = 1usize << (2 * plan.ptm.k());
    let mut sigma: [u8; LOCAL_DIM] = core::array::from_fn(|p| p as u8);
    let mut factor = [ZERO; LOCAL_DIM];
    let mut hit = [false; LOCAL_DIM];
    for p in 0..dim {
        let component = plan.component[p] as usize;
        let n = plan.sizes[component] as usize;
        let r = plan.rank[p] as usize;
        let block = &plan.blocks[plan.offsets[component]..plan.offsets[component] + n * n];
        let mut outputs = (0..n).filter(|&a| block[a * n + r] != ZERO);
        let a = outputs.next()?;
        if outputs.next().is_some() {
            return None;
        }
        sigma[p] = plan.members[component][a];
        factor[p] = block[a * n + r];
        if std::mem::replace(&mut hit[sigma[p] as usize], true) {
            return None;
        }
    }
    Some((sigma, factor))
}

/// Apply one prepared channel to `sum` in place, truncating each written term under `policy`.
pub fn apply_layer_indexed<const W: usize, T>(
    sum: &mut IndexedSum<W>,
    prepared: &Prepared<W>,
    policy: &T,
    scratch: &mut IndexedScratch<W>,
) where
    T: TruncationPolicy<W> + ?Sized,
{
    let span = Gf2Span::new(&prepared.bucket_deltas(), sum.hash.bits());
    let plan = match prepared {
        Prepared::Local(ptm) => {
            let plan = OrbitPlan::new(ptm, &sum.hash, &span);
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
    buckets: &DisjointSlots<'_, IndexedTable<W>>,
    plan: &Plan<'_, W>,
    policy: &T,
    work: &mut IndexedWork<W>,
) where
    T: TruncationPolicy<W> + ?Sized,
{
    let representative = span.representative_at(coset);
    let IndexedWork { tables, done, dead, pending, staying, betas, times } = work;
    betas.clear();
    betas.extend((0..span.coset_size() as u32).map(|j| span.member(representative, j)));
    let m = betas.len();
    tables.resize_with(m, IndexedTable::default);
    done.resize_with(m, Vec::new);
    dead.resize_with(m, Vec::new);
    pending.resize_with(m, Vec::new);
    staying.resize_with(m, Vec::new);
    swap(tables, betas, buckets);

    let start = Instant::now();
    if let Plan::Monomial { plan, sigma, factor } = plan {
        relabel(tables, staying, pending, plan, sigma, factor, policy);
        let walked = Instant::now();
        for b in 0..m {
            times.rows_emitted += pending[b].len() as u64;
            let mut entries = std::mem::take(&mut staying[b]);
            entries.append(&mut pending[b]);
            tables[b].set_entries(entries);
        }
        times.emit_ns += (walked - start).as_nanos() as u64;
        times.accumulate_ns += walked.elapsed().as_nanos() as u64;
        swap(tables, betas, buckets);
        return;
    }
    clear_bits(tables, dead);
    match plan {
        Plan::Diagonal { plan, factor } => rescale(tables, dead, plan, factor, policy),
        Plan::Local(plan) => {
            clear_bits(tables, done);
            walk_local(tables, done, dead, pending, plan, policy);
        }
        Plan::Rotation { rotation, generator_coordinate } => {
            clear_bits(tables, done);
            walk_rotation(tables, done, dead, pending, rotation, *generator_coordinate, policy);
        }
        Plan::Monomial { .. } => unreachable!(),
    }
    let walked = Instant::now();
    for (table, dead) in tables.iter_mut().zip(dead.iter_mut()) {
        table.remove_marked(dead);
    }
    let removed = Instant::now();
    for (table, pending) in tables.iter_mut().zip(pending.iter_mut()) {
        times.rows_emitted += pending.len() as u64;
        table.reserve(pending.len());
        for e in pending.drain(..) {
            table.push_unique(e);
        }
    }
    let appended = Instant::now();
    times.emit_ns += (walked - start).as_nanos() as u64;
    times.retain_ns += (removed - walked).as_nanos() as u64;
    times.accumulate_ns += (appended - removed).as_nanos() as u64;
    swap(tables, betas, buckets);
}

fn swap<const W: usize>(
    tables: &mut [IndexedTable<W>],
    betas: &[u32],
    buckets: &DisjointSlots<'_, IndexedTable<W>>,
) {
    for (slot, &beta) in tables.iter_mut().zip(betas.iter()) {
        // SAFETY: the cosets partition the buckets, so no other task reaches bucket `beta`.
        std::mem::swap(slot, unsafe { buckets.get(beta as usize) });
    }
}

/// One cleared bit per position of each table.
fn clear_bits<const W: usize>(tables: &[IndexedTable<W>], bits: &mut [Vec<u64>]) {
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
    tables: &mut [IndexedTable<W>],
    dead: &mut [Vec<u64>],
    plan: &OrbitPlan<'_, W>,
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

/// Split each table's entries into those that keep their key, rescaled in `staying`, and those that move, relabelled into `pending` of their new table.
fn relabel<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    tables: &mut [IndexedTable<W>],
    staying: &mut [Vec<Entry<W>>],
    pending: &mut [Vec<Entry<W>>],
    plan: &OrbitPlan<'_, W>,
    sigma: &[u8; LOCAL_DIM],
    factor: &[Complex64; LOCAL_DIM],
    policy: &T,
) {
    for b in 0..tables.len() {
        let mut entries = tables[b].take_entries();
        entries.retain_mut(|e| {
            let p = plan.ptm.support_bits(&e.x, &e.z);
            let q = sigma[p] as usize;
            let c = e.c * factor[p];
            if q == p {
                e.c = c;
                return kept(&e.x, &e.z, c, policy);
            }
            let (j, mut moved) = plan.shifted(b, &e.x, &e.z, p ^ q);
            moved.c = c;
            if kept(&moved.x, &moved.z, c, policy) {
                pending[j].push(moved);
            }
            false
        });
        staying[b] = entries;
    }
}

fn walk_local<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    tables: &mut [IndexedTable<W>],
    done: &mut [Vec<u64>],
    dead: &mut [Vec<u64>],
    pending: &mut [Vec<Entry<W>>],
    plan: &OrbitPlan<'_, W>,
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
    tables: &mut [IndexedTable<W>],
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
