use super::BuiltinTruncation as T;
use super::*;
use crate::test_support::{and, or};
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
    let tree = T::from(And(
        CoefficientThreshold(1e-3),
        And(ApproxTopN(500), WeightCutoff(4)),
    ));
    let p = KeepProgram::lower(&tree).unwrap();
    assert_eq!(p.decode(), and(T::Coeff(1e-3), T::Weight(4)));
    assert_eq!(
        KeepProgram::lower(&T::from(TopN(10))).unwrap(),
        KeepProgram::KEEP
    );
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
    let collapse = T::from(crate::truncation::CollapseSample::new(4, 1));
    assert_eq!(
        leaves(&and(T::Coeff(0.1), collapse.clone())),
        vec![collapse.clone()]
    );
    assert_eq!(KeepProgram::lower(&collapse).unwrap(), KeepProgram::KEEP);
}
