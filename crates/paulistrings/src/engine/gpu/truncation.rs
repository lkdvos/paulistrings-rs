//! A [`BuiltinTruncation`] lowered for the device: the per-term [`KeepProgram`] the fused layer evaluates, and the tree whose layer pass K7 runs.

use cudarc::driver::DeviceRepr;

use super::error::GpuError;
use crate::truncation::BuiltinTruncation;

/// Nodes a [`KeepProgram`] holds; must match `KEEP_NODES` in `kernels/prelude.cuh`.
pub(crate) const KEEP_NODES: usize = 15;

const OP_KEEP: u32 = 0;
const OP_COEFF: u32 = 1;
const OP_WEIGHT: u32 = 2;
const OP_AND: u32 = 3;
const OP_OR: u32 = 4;

/// The per-term half of a tree as a postfix program, passed to K3 and K5 by value as `KeepProg`.
///
/// Node `i` is `op[i]` with operand `arg[i]`: `Coeff` carries `eps.to_bits()`, `Weight` carries `k`, `Keep`/`And`/`Or` none.
/// `TopN`/`ApproxTopN` leaves are `Keep` here, and `Keep` operands of `And`/`Or` are folded away, so a tree with no per-term filter is the one-node `Keep`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct KeepProgram {
    len: u32,
    op: [u32; KEEP_NODES],
    arg: [u64; KEEP_NODES],
}

// SAFETY: plain `repr(C)` data with the field order, types and padding of `KeepProg` in kernels/prelude.cuh.
unsafe impl DeviceRepr for KeepProgram {}

/// `t` with its layer passes replaced by `Keep` and every `Keep` operand folded.
fn per_term(t: &BuiltinTruncation) -> BuiltinTruncation {
    use BuiltinTruncation as T;
    match t {
        T::Keep | T::TopN(_) | T::ApproxTopN(_) => T::Keep,
        T::Coeff(eps) => T::Coeff(*eps),
        T::Weight(k) => T::Weight(*k),
        T::And(a, b) => match (per_term(a), per_term(b)) {
            (T::Keep, x) | (x, T::Keep) => x,
            (x, y) => T::And(Box::new(x), Box::new(y)),
        },
        T::Or(a, b) => match (per_term(a), per_term(b)) {
            (T::Keep, _) | (_, T::Keep) => T::Keep,
            (x, y) => T::Or(Box::new(x), Box::new(y)),
        },
    }
}

fn emit(t: &BuiltinTruncation, out: &mut Vec<(u32, u64)>) {
    use BuiltinTruncation as T;
    match t {
        T::Coeff(eps) => out.push((OP_COEFF, eps.to_bits())),
        T::Weight(k) => out.push((OP_WEIGHT, u64::from(*k))),
        T::And(a, b) | T::Or(a, b) => {
            emit(a, out);
            emit(b, out);
            out.push((
                if matches!(t, T::And(..)) {
                    OP_AND
                } else {
                    OP_OR
                },
                0,
            ));
        }
        T::Keep | T::TopN(_) | T::ApproxTopN(_) => out.push((OP_KEEP, 0)),
    }
}

impl KeepProgram {
    /// The program that keeps every nonzero term.
    pub(crate) const KEEP: Self = Self {
        len: 1,
        op: [OP_KEEP; KEEP_NODES],
        arg: [0; KEEP_NODES],
    };

    /// Lower the per-term half of `tree`; [`GpuError::Unsupported`] past [`KEEP_NODES`] nodes.
    pub(crate) fn lower(tree: &BuiltinTruncation) -> Result<Self, GpuError> {
        let mut nodes = Vec::new();
        emit(&per_term(tree), &mut nodes);
        if nodes.len() > KEEP_NODES {
            return Err(GpuError::Unsupported(
                "a per-term truncation program longer than 15 nodes",
            ));
        }
        let mut p = Self {
            len: nodes.len() as u32,
            op: [OP_KEEP; KEEP_NODES],
            arg: [0; KEEP_NODES],
        };
        for (i, (op, arg)) in nodes.into_iter().enumerate() {
            p.op[i] = op;
            p.arg[i] = arg;
        }
        Ok(p)
    }
}

/// A policy as the device runs it: the per-term program and the tree whose layer pass [`DevicePartition`](super::partition::DevicePartition) walks.
#[derive(Clone, Debug)]
pub(crate) struct DevicePolicy {
    pub(crate) tree: BuiltinTruncation,
    pub(crate) keep: KeepProgram,
}

impl DevicePolicy {
    pub(crate) fn lower(tree: BuiltinTruncation) -> Result<Self, GpuError> {
        Ok(Self {
            keep: KeepProgram::lower(&tree)?,
            tree,
        })
    }

    pub(crate) fn keep_all() -> Self {
        Self {
            tree: BuiltinTruncation::Keep,
            keep: KeepProgram::KEEP,
        }
    }
}

/// Visit the layer-pass leaves of `tree` in the order the host runs them: `And` both sides first then second, `Or` neither.
pub(crate) fn layer_pass_leaves(
    tree: &BuiltinTruncation,
    visit: &mut dyn FnMut(&BuiltinTruncation),
) {
    use BuiltinTruncation as T;
    match tree {
        T::TopN(_) | T::ApproxTopN(_) => visit(tree),
        T::And(a, b) => {
            layer_pass_leaves(a, visit);
            layer_pass_leaves(b, visit);
        }
        T::Keep | T::Coeff(_) | T::Weight(_) | T::Or(_, _) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::BuiltinTruncation as T;
    use super::*;
    use crate::truncation::{And, ApproxTopN, CoefficientThreshold, TopN, WeightCutoff};
    use crate::TruncationPolicy;
    use num_complex::Complex64;

    impl KeepProgram {
        /// The host mirror of `keep_eval` in kernels/prelude.cuh.
        fn eval<const W: usize>(&self, x: &[u64; W], z: &[u64; W], c: Complex64) -> bool {
            let m = c.norm_sqr();
            let weight: u64 = (0..W).map(|i| u64::from((x[i] | z[i]).count_ones())).sum();
            let mut st = 0u32;
            for i in 0..self.len as usize {
                st = match self.op[i] {
                    OP_COEFF => {
                        let eps = f64::from_bits(self.arg[i]);
                        (st << 1) | u32::from(eps < 0.0 || m > eps * eps)
                    }
                    OP_WEIGHT => (st << 1) | u32::from(weight <= self.arg[i]),
                    OP_AND => ((st >> 2) << 1) | (st & (st >> 1) & 1),
                    OP_OR => ((st >> 2) << 1) | ((st | (st >> 1)) & 1),
                    _ => (st << 1) | 1,
                };
            }
            st & 1 != 0
        }

        /// The tree this program encodes.
        fn decode(&self) -> T {
            let mut stack: Vec<T> = Vec::new();
            for i in 0..self.len as usize {
                let node = match self.op[i] {
                    OP_COEFF => T::Coeff(f64::from_bits(self.arg[i])),
                    OP_WEIGHT => T::Weight(self.arg[i] as u32),
                    OP_AND | OP_OR => {
                        let b = Box::new(stack.pop().expect("operand"));
                        let a = Box::new(stack.pop().expect("operand"));
                        if self.op[i] == OP_AND {
                            T::And(a, b)
                        } else {
                            T::Or(a, b)
                        }
                    }
                    _ => T::Keep,
                };
                stack.push(node);
            }
            assert_eq!(stack.len(), 1, "a well-formed program leaves one value");
            stack.pop().unwrap()
        }

        fn nodes(&self) -> Vec<(u32, u64)> {
            (0..self.len as usize)
                .map(|i| (self.op[i], self.arg[i]))
                .collect()
        }
    }

    fn and(a: T, b: T) -> T {
        T::And(Box::new(a), Box::new(b))
    }

    fn or(a: T, b: T) -> T {
        T::Or(Box::new(a), Box::new(b))
    }

    #[test]
    fn lowering_emits_the_hand_written_postfix() {
        let lower = |t: &T| KeepProgram::lower(t).unwrap().nodes();
        assert_eq!(lower(&T::Keep), vec![(OP_KEEP, 0)]);
        assert_eq!(lower(&T::Coeff(1e-3)), vec![(OP_COEFF, 1e-3f64.to_bits())]);
        assert_eq!(
            lower(&and(T::Coeff(0.5), T::Weight(4))),
            vec![(OP_COEFF, 0.5f64.to_bits()), (OP_WEIGHT, 4), (OP_AND, 0)]
        );
        assert_eq!(
            lower(&or(T::Weight(1), and(T::Coeff(0.25), T::Weight(3)))),
            vec![
                (OP_WEIGHT, 1),
                (OP_COEFF, 0.25f64.to_bits()),
                (OP_WEIGHT, 3),
                (OP_AND, 0),
                (OP_OR, 0)
            ]
        );
        assert_eq!(
            lower(&and(T::Coeff(1e-3), T::ApproxTopN(10))),
            vec![(OP_COEFF, 1e-3f64.to_bits())]
        );
        assert_eq!(
            lower(&and(T::ApproxTopN(10), T::ApproxTopN(3))),
            vec![(OP_KEEP, 0)]
        );
        assert_eq!(lower(&or(T::Coeff(1e-3), T::TopN(10))), vec![(OP_KEEP, 0)]);
        assert_eq!(KeepProgram::lower(&T::Keep).unwrap(), KeepProgram::KEEP);
    }

    /// Weights 0..3 at `W = 2`, one of them across the word boundary, against coefficients around 0.1 and 0.5.
    fn grid() -> Vec<([u64; 2], [u64; 2], Complex64)> {
        let keys = [
            ([0u64, 0], [0u64, 0]),
            ([0, 1], [0, 0]),
            ([0b01, 0], [0b10, 0]),
            ([0b011, 1], [0b110, 0]),
        ];
        let cs = [0.0, 0.05, 0.1, 0.3, 0.5, 2.0].map(|r| Complex64::new(-r, r / 2.0));
        keys.iter()
            .flat_map(|&(x, z)| cs.iter().map(move |&c| (x, z, c)))
            .collect()
    }

    #[test]
    fn lowering_round_trips_and_evaluates_as_the_tree() {
        let trees = [
            T::Keep,
            T::Coeff(0.1),
            T::Coeff(-1.0),
            T::Weight(2),
            and(T::Coeff(0.1), T::Weight(1)),
            or(T::Coeff(0.5), T::Weight(0)),
            or(
                and(T::Coeff(0.1), T::ApproxTopN(3)),
                and(T::Weight(2), T::Coeff(0.3)),
            ),
            and(
                or(T::Weight(0), T::Coeff(1.0)),
                or(T::TopN(3), T::Weight(3)),
            ),
            and(T::Keep, or(T::Keep, T::Coeff(0.2))),
        ];
        for tree in &trees {
            let p = KeepProgram::lower(tree).unwrap();
            assert_eq!(p.decode(), per_term(tree), "{tree:?}: decode");
            for (x, z, c) in grid() {
                assert_eq!(
                    p.eval(&x, &z, c),
                    <T as TruncationPolicy<2>>::keep_term(tree, &x, &z, c),
                    "{tree:?}: x={x:?} z={z:?} c={c}"
                );
            }
        }
    }

    #[test]
    fn a_program_past_fifteen_nodes_is_unsupported() {
        let chain =
            |leaves: usize| (1..leaves).fold(T::Coeff(0.0), |acc, i| and(acc, T::Weight(i as u32)));
        assert_eq!(KeepProgram::lower(&chain(8)).unwrap().len, 15);
        assert!(matches!(
            KeepProgram::lower(&chain(9)),
            Err(GpuError::Unsupported(_))
        ));
        // Folded `Keep` operands do not count against the limit.
        let padded = (0..20).fold(chain(8), |acc, _| and(acc, T::ApproxTopN(5)));
        assert_eq!(KeepProgram::lower(&padded).unwrap().len, 15);
    }

    #[test]
    fn every_builtin_lowers_including_top_n() {
        let tree = <_ as TruncationPolicy<1>>::device_policy(&And(
            CoefficientThreshold(1e-3),
            And(ApproxTopN(500), WeightCutoff(4)),
        ))
        .unwrap();
        let p = DevicePolicy::lower(tree).unwrap();
        assert_eq!(p.keep.decode(), and(T::Coeff(1e-3), T::Weight(4)));
        let top = <_ as TruncationPolicy<1>>::device_policy(&TopN(10)).unwrap();
        assert_eq!(DevicePolicy::lower(top).unwrap().keep, KeepProgram::KEEP);
    }

    #[test]
    fn layer_pass_leaves_follow_and_and_skip_or() {
        let leaves = |t: &T| {
            let mut v = Vec::new();
            layer_pass_leaves(t, &mut |l| v.push(l.clone()));
            v
        };
        assert_eq!(leaves(&T::ApproxTopN(3)), vec![T::ApproxTopN(3)]);
        assert_eq!(
            leaves(&and(T::Coeff(0.1), T::ApproxTopN(3))),
            vec![T::ApproxTopN(3)]
        );
        assert_eq!(
            leaves(&and(T::ApproxTopN(9), and(T::Weight(2), T::ApproxTopN(3)))),
            vec![T::ApproxTopN(9), T::ApproxTopN(3)]
        );
        assert!(leaves(&or(T::ApproxTopN(3), T::Coeff(0.1))).is_empty());
        assert!(leaves(&T::Coeff(0.1)).is_empty());
    }
}
