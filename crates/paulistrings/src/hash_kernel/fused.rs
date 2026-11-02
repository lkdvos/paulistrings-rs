//! The fused kernel for channels where every key receives at most one cross-key row: Pauli rotations, Cliffords, and their noisy rescalings.
//!
//! Pass 1 rescales each entry in place by its identity amplitude and buffers its other rows; pass 2 lands each row with one lookup and truncates the target at once, its value being final.
//! An entry pass 1 keeps but that receives no row is checked where that is known: in pass 1 when its pattern has no predecessor, and in pass 2, from its partner's row finding the partner absent, when the two form a mutual pair.
//! The exact counterpart of PauliPropagation.jl's `fused=true` rotation path, which truncates in pass 1 before an incoming row can land.

use num_complex::Complex64;

use super::table::{key_hash, Entry, Table};
use crate::channel::prepared::{LocalPtm, PreparedRotation, LOCAL_DIM};
use crate::engine::coset::Gf2Span;
use crate::pauli_string::PauliString;
use crate::phase::Phase;
use crate::truncation::TruncationPolicy;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// A buffered cross-key row, with the local delta back to its source.
#[derive(Clone, Copy, Debug)]
pub(super) struct Row<const W: usize> {
    entry: Entry<W>,
    /// Local delta from target to source; `u8::MAX` for a wide rotation's generator.
    delta: u8,
    /// Whether the source stays in its table and receives only from this row's target, so the target's absence settles the source.
    check_source: bool,
}

/// A local PTM whose pattern graph has in-degree at most one off the diagonal.
pub(super) struct FusedPlan<'p, const W: usize> {
    ptm: &'p LocalPtm<W>,
    identity: [Complex64; LOCAL_DIM],
    untouched: [bool; LOCAL_DIM],
    /// Pass 1 decides truncation: the pattern has no predecessor, or its identity amplitude is zero.
    final_after_pass1: [bool; LOCAL_DIM],
    /// Out-edges per pattern: `(target pattern, amplitude)`.
    edges: [[(u8, Complex64); LOCAL_DIM]; LOCAL_DIM],
    degree: [u8; LOCAL_DIM],
    check_source: [bool; LOCAL_DIM],
    mask_x: [[u64; W]; LOCAL_DIM],
    mask_z: [[u64; W]; LOCAL_DIM],
    coordinate: [u32; LOCAL_DIM],
}

impl<'p, const W: usize> FusedPlan<'p, W> {
    /// `None` unless every pattern has at most one predecessor and every kept pattern with one forms a mutual pair with it whose partner is kept too.
    pub(super) fn new(
        ptm: &'p LocalPtm<W>,
        hash: &crate::pauli_sum::Gf2Hash<W>,
        span: &Gf2Span,
    ) -> Option<Self> {
        let dim = 1usize << (2 * ptm.k());
        let mut transfer = [[ZERO; LOCAL_DIM]; LOCAL_DIM];
        for delta in ptm.deltas() {
            for p in 0..dim {
                transfer[p ^ delta.local_delta as usize][p] += delta.amplitude[p];
            }
        }
        let identity: [Complex64; LOCAL_DIM] = core::array::from_fn(|p| transfer[p][p]);
        let mut predecessor = [None::<usize>; LOCAL_DIM];
        for q in 0..dim {
            for p in (0..dim).filter(|&p| p != q && transfer[q][p] != ZERO) {
                if predecessor[q].replace(p).is_some() {
                    return None;
                }
            }
        }
        let mut final_after_pass1 = [true; LOCAL_DIM];
        for q in 0..dim {
            if let Some(p) = predecessor[q] {
                if identity[q] != ZERO {
                    let mutual = transfer[p][q] != ZERO && identity[p] != ZERO;
                    if !mutual {
                        return None;
                    }
                    final_after_pass1[q] = false;
                }
            }
        }
        let mut edges = [[(0u8, ZERO); LOCAL_DIM]; LOCAL_DIM];
        let mut degree = [0u8; LOCAL_DIM];
        let mut check_source = [false; LOCAL_DIM];
        for p in 0..dim {
            for q in (0..dim).filter(|&q| q != p && transfer[q][p] != ZERO) {
                edges[p][degree[p] as usize] = (q as u8, transfer[q][p]);
                degree[p] += 1;
            }
        }
        for q in 0..dim {
            // A row into `q` from `p` finding `q` absent leaves `p` with nothing incoming when `p`'s only predecessor is `q` and `q` was never erased in pass 1.
            if let Some(p) = predecessor[q] {
                check_source[q] = identity[p] != ZERO && identity[q] != ZERO;
            }
        }
        let untouched: [bool; LOCAL_DIM] =
            core::array::from_fn(|p| identity[p] == ONE && degree[p] == 0);
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
        Some(Self {
            ptm,
            identity,
            untouched,
            final_after_pass1,
            edges,
            degree,
            check_source,
            mask_x,
            mask_z,
            coordinate,
        })
    }
}

#[inline(always)]
fn kept<const W: usize, T: TruncationPolicy<W> + ?Sized>(e: &Entry<W>, policy: &T) -> bool {
    ((e.c.re != 0.0) | (e.c.im != 0.0)) && policy.keep_term(&e.x, &e.z, e.c)
}

pub(super) fn pass1_local<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    tables: &mut [Table<W>],
    side: &mut [Vec<Row<W>>],
    plan: &FusedPlan<'_, W>,
    policy: &T,
) {
    for (i, table) in tables.iter_mut().enumerate() {
        table.retain(|e| {
            let p = plan.ptm.support_bits(&e.x, &e.z);
            if plan.untouched[p] {
                return true;
            }
            let c = e.c;
            for &(q, amplitude) in &plan.edges[p][..plan.degree[p] as usize] {
                let d = p ^ q as usize;
                let mut row = Entry { x: e.x, z: e.z, c: c * amplitude };
                for w in 0..W {
                    row.x[w] ^= plan.mask_x[d][w];
                    row.z[w] ^= plan.mask_z[d][w];
                }
                side[i ^ plan.coordinate[d] as usize].push(Row {
                    entry: row,
                    delta: d as u8,
                    check_source: plan.check_source[q as usize],
                });
            }
            e.c = c * plan.identity[p];
            if plan.final_after_pass1[p] {
                kept(e, policy)
            } else {
                true
            }
        });
    }
}

pub(super) fn pass1_rotation<const W: usize>(
    tables: &mut [Table<W>],
    side: &mut [Vec<Row<W>>],
    rotation: &PreparedRotation<W>,
    generator_coordinate: u32,
) {
    let generator = &rotation.generator;
    let erase = rotation.cos == 0.0;
    for (i, table) in tables.iter_mut().enumerate() {
        table.retain(|e| {
            let key = PauliString::<W> { x: e.x, z: e.z };
            if key.commutes_with(generator) {
                return true;
            }
            let c = e.c;
            let mut product = key;
            let phase = Phase::I + product.mul_assign(generator);
            side[i ^ generator_coordinate as usize].push(Row {
                entry: Entry { x: product.x, z: product.z, c: phase.apply(c) * rotation.sin },
                delta: u8::MAX,
                check_source: !erase,
            });
            e.c = c * rotation.cos;
            // A kept anticommuting key's only incoming row is from its partner, so pass 2 settles it.
            !erase
        });
    }
}

/// Land table `j`'s rows, truncating each target at once, and settle the sources whose partner proved absent.
pub(super) fn pass2<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    tables: &mut [Table<W>],
    j: usize,
    rows: &mut Vec<Row<W>>,
    source_of: impl Fn(usize, &Entry<W>, u8) -> (usize, Entry<W>),
    policy: &T,
) {
    for row in rows.drain(..) {
        let target = row.entry;
        let h = key_hash(&target.x, &target.z);
        tables[j].reserve(1);
        match tables[j].find_or_slot(h, &target) {
            Ok(slot) => {
                tables[j].get_mut(slot).c += target.c;
                if !kept(tables[j].get(slot), policy) {
                    tables[j].erase(slot);
                }
            }
            Err(slot) => {
                if kept(&target, policy) {
                    tables[j].insert_at(slot, h, target);
                }
                if row.check_source {
                    let (i, source) = source_of(j, &target, row.delta);
                    if let Some(slot) = tables[i].find(key_hash(&source.x, &source.z), &source) {
                        if !kept(tables[i].get(slot), policy) {
                            tables[i].erase(slot);
                        }
                    }
                }
            }
        }
    }
}

/// The source of a local row: `target ⊕ d`, in member `j ⊕ coord(d)`.
pub(super) fn local_source<'a, const W: usize>(
    plan: &'a FusedPlan<'_, W>,
) -> impl Fn(usize, &Entry<W>, u8) -> (usize, Entry<W>) + 'a {
    move |j, target, d| {
        let d = d as usize;
        let mut source = *target;
        for w in 0..W {
            source.x[w] ^= plan.mask_x[d][w];
            source.z[w] ^= plan.mask_z[d][w];
        }
        (j ^ plan.coordinate[d] as usize, source)
    }
}

/// The source of a wide rotation's row: `target ⊕ P`, in member `j ⊕ coord(P)`.
pub(super) fn rotation_source<const W: usize>(
    rotation: &PreparedRotation<W>,
    generator_coordinate: u32,
) -> impl Fn(usize, &Entry<W>, u8) -> (usize, Entry<W>) + '_ {
    move |j, target, _| {
        let mut source = *target;
        for w in 0..W {
            source.x[w] ^= rotation.generator.x[w];
            source.z[w] ^= rotation.generator.z[w];
        }
        (j ^ generator_coordinate as usize, source)
    }
}
