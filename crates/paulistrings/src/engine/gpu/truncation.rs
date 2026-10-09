//! A [`BuiltinTruncation`] lowered for the device: the per-term [`KeepProgram`] the fused layer evaluates, and the walk over the tree's layer pass.

use cudarc::driver::DeviceRepr;

use super::error::GpuError;
use crate::truncation::BuiltinTruncation;

/// Nodes a [`KeepProgram`] holds; must match `KEEP_NODES` in `kernels/prelude.cuh`.
const KEEP_NODES: usize = 15;

const OP_KEEP: u32 = 0;
const OP_COEFF: u32 = 1;
const OP_WEIGHT: u32 = 2;
const OP_AND: u32 = 3;
const OP_OR: u32 = 4;

/// The per-term half of a tree as a postfix program, passed to K3 and K5 by value as `KeepProg`.
///
/// Node `i` is `op[i]` with operand `arg[i]`: `Coefficient` carries `eps.to_bits()`, `Weight` carries `k`, `Keep`/`And`/`Or` none.
/// `TopN`/`ApproxTopN`/`CollapseSample` leaves are `Keep` here, and `Keep` operands of `And`/`Or` are folded away, so a tree with no per-term filter is the one-node `Keep`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct KeepProgram {
    len: u32,
    op: [u32; KEEP_NODES],
    arg: [u64; KEEP_NODES],
}

// SAFETY: plain `repr(C)` data with the field order, types and padding of `KeepProg` in kernels/prelude.cuh.
unsafe impl DeviceRepr for KeepProgram {}

/// `tree` with its layer passes replaced by `Keep` and every `Keep` operand folded.
fn per_term(tree: &BuiltinTruncation) -> BuiltinTruncation {
    use BuiltinTruncation as T;
    match tree {
        T::Keep | T::TopN(_) | T::ApproxTopN(_) | T::CollapseSample(_) => T::Keep,
        T::Coefficient(eps) => T::Coefficient(*eps),
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

fn emit(tree: &BuiltinTruncation, out: &mut Vec<(u32, u64)>) {
    use BuiltinTruncation as T;
    match tree {
        T::Coefficient(eps) => out.push((OP_COEFF, eps.to_bits())),
        T::Weight(k) => out.push((OP_WEIGHT, u64::from(*k))),
        T::And(a, b) | T::Or(a, b) => {
            emit(a, out);
            emit(b, out);
            out.push((
                if matches!(tree, T::And(..)) {
                    OP_AND
                } else {
                    OP_OR
                },
                0,
            ));
        }
        T::Keep | T::TopN(_) | T::ApproxTopN(_) | T::CollapseSample(_) => out.push((OP_KEEP, 0)),
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
        let mut program = Self {
            len: nodes.len() as u32,
            op: [OP_KEEP; KEEP_NODES],
            arg: [0; KEEP_NODES],
        };
        for (i, (op, arg)) in nodes.into_iter().enumerate() {
            program.op[i] = op;
            program.arg[i] = arg;
        }
        Ok(program)
    }
}

/// Visit the layer-pass leaves of `tree` in the order the host runs them: `And` both sides first then second, `Or` neither.
pub(crate) fn layer_pass_leaves(
    tree: &BuiltinTruncation,
    visit: &mut dyn FnMut(&BuiltinTruncation),
) {
    use BuiltinTruncation as T;
    match tree {
        T::TopN(_) | T::ApproxTopN(_) | T::CollapseSample(_) => visit(tree),
        T::And(a, b) => {
            layer_pass_leaves(a, visit);
            layer_pass_leaves(b, visit);
        }
        T::Keep | T::Coefficient(_) | T::Weight(_) | T::Or(_, _) => {}
    }
}

#[cfg(test)]
mod tests;
