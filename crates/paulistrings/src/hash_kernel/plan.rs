//! [`GroupPlan`], a local PTM as the blocks of its closed key groups, and the [`Plan`] a layer runs.
//!
//! Keys that differ only on the channel's support mix only among themselves, along the connected components of its pattern graph, so a component's present members are its complete input.

use num_complex::Complex64;

use super::table::Entry;
use crate::channel::prepared::{LocalPtm, PreparedRotation, LOCAL_DIM};
use crate::engine::coset::Gf2Span;
use crate::pauli_sum::Gf2Hash;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// A local PTM as the blocks of its pattern graph's connected components, the closed groups of keys a layer mixes.
pub(super) struct GroupPlan<'p, const W: usize> {
    pub(super) ptm: &'p LocalPtm<W>,
    /// Component of each support pattern.
    pub(super) component: [u8; LOCAL_DIM],
    /// Position of each pattern among its component's members.
    pub(super) rank: [u8; LOCAL_DIM],
    /// Members of each component, ascending; `sizes[c]` are meaningful.
    pub(super) members: [[u8; LOCAL_DIM]; LOCAL_DIM],
    pub(super) sizes: [u8; LOCAL_DIM],
    /// Components whose block is the identity, left untouched.
    pub(super) skip: [bool; LOCAL_DIM],
    /// Component `c`'s block, row-major `sizes[c]²` from `offsets[c]`: entry `(a, b)` takes member `b` to member `a`.
    pub(super) blocks: Vec<Complex64>,
    pub(super) offsets: [usize; LOCAL_DIM],
    /// Per local delta `d`: the XOR masks and the coset coordinate of its bucket delta.
    pub(super) mask_x: [[u64; W]; LOCAL_DIM],
    pub(super) mask_z: [[u64; W]; LOCAL_DIM],
    pub(super) coordinate: [u32; LOCAL_DIM],
}

impl<'p, const W: usize> GroupPlan<'p, W> {
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
            blocks,
            offsets,
            mask_x,
            mask_z,
            coordinate,
        }
    }

    /// `key ⊕ d` and the coset member it lives in, from member `i`.
    #[inline(always)]
    pub(super) fn shifted(&self, i: usize, x: &[u64; W], z: &[u64; W], d: usize) -> (usize, Entry<W>) {
        debug_assert!(self.coordinate[d] != u32::MAX, "a component delta outside the span");
        let mut e = Entry { x: *x, z: *z, c: ZERO };
        for w in 0..W {
            e.x[w] ^= self.mask_x[d][w];
            e.z[w] ^= self.mask_z[d][w];
        }
        (i ^ self.coordinate[d] as usize, e)
    }
}

pub(super) enum Plan<'p, const W: usize> {
    /// Every pattern maps to itself: a rescale walk with no lookups.
    Diagonal { plan: GroupPlan<'p, W>, factor: [Complex64; LOCAL_DIM] },
    /// Pattern `p` maps to `sigma[p]` alone, a bijection: a relabelling walk with no lookups.
    Monomial { plan: GroupPlan<'p, W>, sigma: [u8; LOCAL_DIM], factor: [Complex64; LOCAL_DIM] },
    Local(GroupPlan<'p, W>),
    Rotation { rotation: &'p PreparedRotation<W>, generator_coordinate: u32 },
}

/// The permutation and factors of a PTM taking each pattern to exactly one pattern, bijectively.
pub(super) fn monomial<const W: usize>(
    plan: &GroupPlan<'_, W>,
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
