//! The orbit kernel: each closed group of keys a layer mixes is read, transformed and written back in place, with no side buffers.
//!
//! Keys that differ only on the channel's support mix only among themselves, along the connected components of its pattern graph, so a component's present members are its complete input.
//! A component is processed once, from its leader: the present member of lowest rank, found before any write so the leaders are those of the input.
//! Truncation is decided as each output is written, and an entry no component touches keeps its coefficient, so `keep_term` must be idempotent on unchanged terms.

use num_complex::Complex64;

use super::table::{key_hash, Entry, Table};
use crate::channel::prepared::{LocalPtm, PreparedRotation, LOCAL_DIM};
use crate::engine::coset::Gf2Span;
use crate::pauli_string::PauliString;
use crate::pauli_sum::Gf2Hash;
use crate::phase::Phase;
use crate::truncation::TruncationPolicy;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// A local PTM as the blocks of its pattern graph's connected components.
pub(super) struct OrbitPlan<'p, const W: usize> {
    ptm: &'p LocalPtm<W>,
    /// Component of each support pattern.
    component: [u8; LOCAL_DIM],
    /// Position of each pattern among its component's members.
    rank: [u8; LOCAL_DIM],
    /// Members of each component, ascending; `sizes[c]` are meaningful.
    members: [[u8; LOCAL_DIM]; LOCAL_DIM],
    sizes: [u8; LOCAL_DIM],
    /// Components whose block is the identity, left untouched.
    skip: [bool; LOCAL_DIM],
    /// Whether some one-member component is not skipped.
    has_singletons: bool,
    /// Component `c`'s block, row-major `sizes[c]²` from `offsets[c]`: entry `(a, b)` takes member `b` to member `a`.
    blocks: Vec<Complex64>,
    offsets: [usize; LOCAL_DIM],
    /// Per local delta `d`: the XOR masks and the coset coordinate of its bucket delta.
    mask_x: [[u64; W]; LOCAL_DIM],
    mask_z: [[u64; W]; LOCAL_DIM],
    coordinate: [u32; LOCAL_DIM],
}

impl<'p, const W: usize> OrbitPlan<'p, W> {
    pub(super) fn new(ptm: &'p LocalPtm<W>, hash: &Gf2Hash<W>, span: &Gf2Span) -> Self {
        let dim = 1usize << (2 * ptm.k());
        // `transfer[q][p]`: the amplitude taking pattern `p` to pattern `q`.
        let mut transfer = [[ZERO; LOCAL_DIM]; LOCAL_DIM];
        for delta in ptm.deltas() {
            for p in 0..dim {
                transfer[p ^ delta.local_delta as usize][p] += delta.amplitude[p];
            }
        }

        let mut parent: [usize; LOCAL_DIM] = core::array::from_fn(|p| p);
        fn root(parent: &mut [usize; LOCAL_DIM], mut p: usize) -> usize {
            while parent[p] != p {
                parent[p] = parent[parent[p]];
                p = parent[p];
            }
            p
        }
        for q in 0..dim {
            for p in 0..dim {
                if transfer[q][p] != ZERO {
                    let (a, b) = (root(&mut parent, p), root(&mut parent, q));
                    parent[a.max(b)] = a.min(b);
                }
            }
        }

        let mut component = [0u8; LOCAL_DIM];
        let mut rank = [0u8; LOCAL_DIM];
        let mut members = [[0u8; LOCAL_DIM]; LOCAL_DIM];
        let mut sizes = [0u8; LOCAL_DIM];
        let mut id_of_root = [u8::MAX; LOCAL_DIM];
        let mut count = 0usize;
        for p in 0..dim {
            let r = root(&mut parent, p);
            if id_of_root[r] == u8::MAX {
                id_of_root[r] = count as u8;
                count += 1;
            }
            let c = id_of_root[r] as usize;
            component[p] = c as u8;
            rank[p] = sizes[c];
            members[c][sizes[c] as usize] = p as u8;
            sizes[c] += 1;
        }

        let mut blocks = Vec::new();
        let mut offsets = [0usize; LOCAL_DIM];
        let mut skip = [false; LOCAL_DIM];
        for c in 0..count {
            offsets[c] = blocks.len();
            let n = sizes[c] as usize;
            let mut identity = true;
            for a in 0..n {
                for b in 0..n {
                    let amplitude = transfer[members[c][a] as usize][members[c][b] as usize];
                    identity &= amplitude == if a == b { ONE } else { ZERO };
                    blocks.push(amplitude);
                }
            }
            skip[c] = identity;
        }
        let has_singletons = (0..count).any(|c| sizes[c] == 1 && !skip[c]);

        let mut mask_x = [[0u64; W]; LOCAL_DIM];
        let mut mask_z = [[0u64; W]; LOCAL_DIM];
        let mut coordinate = [u32::MAX; LOCAL_DIM];
        for d in 0..dim {
            let (x, z) = ptm.lift(d as u8);
            mask_x[d] = x;
            mask_z[d] = z;
            let bucket_delta = hash.bucket_of(&x, &z);
            if span.representative_of(bucket_delta) == 0 {
                coordinate[d] = span.coordinate_of(bucket_delta);
            }
        }

        Self {
            ptm,
            component,
            rank,
            members,
            sizes,
            skip,
            has_singletons,
            blocks,
            offsets,
            mask_x,
            mask_z,
            coordinate,
        }
    }

    /// `key ⊕ d` and the coset member it lives in, from member `i`.
    #[inline(always)]
    fn shifted(&self, i: usize, x: &[u64; W], z: &[u64; W], d: usize) -> (usize, Entry<W>) {
        debug_assert!(self.coordinate[d] != u32::MAX, "a component delta outside the span");
        let mut e = Entry { x: *x, z: *z, c: ZERO };
        for w in 0..W {
            e.x[w] ^= self.mask_x[d][w];
            e.z[w] ^= self.mask_z[d][w];
        }
        (i ^ self.coordinate[d] as usize, e)
    }
}

/// A component to process: a member's table, key and pattern.
#[derive(Clone, Copy, Debug)]
pub(super) struct Leader<const W: usize> {
    table: u32,
    pattern: u8,
    x: [u64; W],
    z: [u64; W],
}

fn find<const W: usize>(table: &Table<W>, key: &Entry<W>) -> Option<Complex64> {
    table.find(key_hash(&key.x, &key.z), |e| e.same_key(key)).map(|e| e.c)
}

/// Write `key`'s new coefficient `c`, given whether it was present: update, insert or erase.
#[inline(always)]
fn write_back<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    table: &mut Table<W>,
    key: Entry<W>,
    present: bool,
    c: Complex64,
    policy: &T,
) {
    let keep = ((c.re != 0.0) | (c.im != 0.0)) && policy.keep_term(&key.x, &key.z, c);
    let h = key_hash(&key.x, &key.z);
    match (present, keep) {
        (true, true) => {
            table.find_mut(h, |e| e.same_key(&key)).expect("present").c = c;
        }
        (true, false) => {
            if let Ok(entry) = table.find_entry(h, |e| e.same_key(&key)) {
                entry.remove();
            }
        }
        (false, true) => {
            table.insert_unique(h, Entry { c, ..key }, super::table::entry_hash);
        }
        (false, false) => {}
    }
}

/// Rescale the one-member components in place, where nothing else reads the entry.
pub(super) fn rescale_singletons<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    tables: &mut [Table<W>],
    plan: &OrbitPlan<'_, W>,
    policy: &T,
) {
    if !plan.has_singletons {
        return;
    }
    for table in tables.iter_mut() {
        table.retain(|e| {
            let p = plan.ptm.support_bits(&e.x, &e.z);
            let c = plan.component[p] as usize;
            if plan.skip[c] || plan.sizes[c] != 1 {
                return true;
            }
            e.c *= plan.blocks[plan.offsets[c]];
            ((e.c.re != 0.0) | (e.c.im != 0.0)) && policy.keep_term(&e.x, &e.z, e.c)
        });
    }
}

pub(super) fn leaders_local<const W: usize>(
    tables: &[Table<W>],
    plan: &OrbitPlan<'_, W>,
    leaders: &mut Vec<Leader<W>>,
) {
    for_each_interleaved(tables, |i, e| {
        {
            let p = plan.ptm.support_bits(&e.x, &e.z);
            let c = plan.component[p] as usize;
            if plan.skip[c] || plan.sizes[c] == 1 {
                return;
            }
            let leads = (0..plan.rank[p] as usize).all(|a| {
                let (j, key) = plan.shifted(i, &e.x, &e.z, p ^ plan.members[c][a] as usize);
                find(&tables[j], &key).is_none()
            });
            if leads {
                leaders.push(Leader { table: i as u32, pattern: p as u8, x: e.x, z: e.z });
            }
        }
    });
}

/// Every entry of every table, round-robin across the tables, so the leaders alternate between the tables they drain and fill and no table holds a component's inputs and outputs at once for long.
fn for_each_interleaved<const W: usize>(tables: &[Table<W>], mut f: impl FnMut(usize, &Entry<W>)) {
    if let [table] = tables {
        table.iter().for_each(|e| f(0, e));
        return;
    }
    let mut iterators: Vec<_> = tables.iter().map(|table| table.iter()).collect();
    let mut live = iterators.len();
    while live > 0 {
        live = 0;
        for (i, iterator) in iterators.iter_mut().enumerate() {
            if let Some(e) = iterator.next() {
                f(i, e);
                live += 1;
            }
        }
    }
}

pub(super) fn process_local<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    tables: &mut [Table<W>],
    plan: &OrbitPlan<'_, W>,
    leaders: &[Leader<W>],
    policy: &T,
) {
    let mut slots = [(0usize, Entry { x: [0; W], z: [0; W], c: ZERO }); LOCAL_DIM];
    let mut old = [None::<Complex64>; LOCAL_DIM];
    for leader in leaders {
        let p = leader.pattern as usize;
        let c = plan.component[p] as usize;
        let n = plan.sizes[c] as usize;
        for a in 0..n {
            let d = p ^ plan.members[c][a] as usize;
            slots[a] = plan.shifted(leader.table as usize, &leader.x, &leader.z, d);
            old[a] = find(&tables[slots[a].0], &slots[a].1);
        }
        let block = &plan.blocks[plan.offsets[c]..plan.offsets[c] + n * n];
        for a in 0..n {
            let mut sum = ZERO;
            for b in 0..n {
                if let Some(value) = old[b] {
                    sum += block[a * n + b] * value;
                }
            }
            slots[a].1.c = sum;
        }
        // Erasures before insertions, so a permutation frees a slot before it claims one and the table never grows past its input.
        for present in [true, false] {
            for a in (0..n).filter(|&a| old[a].is_some() == present) {
                let (j, key) = slots[a];
                write_back(&mut tables[j], key, present, key.c, policy);
            }
        }
    }
}

/// The two-member components `{v, v·P}` of a wide rotation; commuting keys are untouched.
pub(super) fn leaders_rotation<const W: usize>(
    tables: &[Table<W>],
    rotation: &PreparedRotation<W>,
    generator_coordinate: u32,
    leaders: &mut Vec<Leader<W>>,
) {
    let generator = &rotation.generator;
    for_each_interleaved(tables, |i, e| {
        {
            let key = PauliString::<W> { x: e.x, z: e.z };
            if key.commutes_with(generator) {
                return;
            }
            let mut partner = Entry { x: e.x, z: e.z, c: ZERO };
            for w in 0..W {
                partner.x[w] ^= generator.x[w];
                partner.z[w] ^= generator.z[w];
            }
            let leads = (e.x, e.z) < (partner.x, partner.z)
                || find(&tables[i ^ generator_coordinate as usize], &partner).is_none();
            if leads {
                leaders.push(Leader { table: i as u32, pattern: 0, x: e.x, z: e.z });
            }
        }
    });
}

pub(super) fn process_rotation<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    tables: &mut [Table<W>],
    rotation: &PreparedRotation<W>,
    generator_coordinate: u32,
    leaders: &[Leader<W>],
    policy: &T,
) {
    for leader in leaders {
        let i = leader.table as usize;
        let j = i ^ generator_coordinate as usize;
        let v = PauliString::<W> { x: leader.x, z: leader.z };
        let mut w = v;
        let phase_v_to_w = Phase::I + w.mul_assign(&rotation.generator);
        let mut back = w;
        let phase_w_to_v = Phase::I + back.mul_assign(&rotation.generator);
        let key_v = Entry { x: v.x, z: v.z, c: ZERO };
        let key_w = Entry { x: w.x, z: w.z, c: ZERO };
        let old_v = find(&tables[i], &key_v);
        let old_w = find(&tables[j], &key_w);
        let (cv, cw) = (old_v.unwrap_or(ZERO), old_w.unwrap_or(ZERO));
        let new_v = cv * rotation.cos + phase_w_to_v.apply(cw) * rotation.sin;
        let new_w = cw * rotation.cos + phase_v_to_w.apply(cv) * rotation.sin;
        // Erasures before insertions, as in `process_local`.
        if old_v.is_some() {
            write_back(&mut tables[i], key_v, true, new_v, policy);
            write_back(&mut tables[j], key_w, old_w.is_some(), new_w, policy);
        } else {
            write_back(&mut tables[j], key_w, old_w.is_some(), new_w, policy);
            write_back(&mut tables[i], key_v, false, new_v, policy);
        }
    }
}
