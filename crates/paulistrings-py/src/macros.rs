//! Width-dispatch macros for the `{W1, W2, W4, W8, W16}` monomorphization
//! enums (`PauliSumImpl`, `CircuitImpl`). See `sum.rs` / `circuit.rs`.
//!
//! Each hand-written dispatch match was five near-identical arms differing
//! only in the variant name; these macros write that shape once. They are
//! deliberately narrow (single dispatch, same-enum pairs, cross-enum pairs,
//! num_qubits-keyed construction) rather than one maximally general macro —
//! a macro that tries to cover every shape stops being readable at the call
//! site, which is the thing being optimized for here.

/// Dispatch a single width-monomorphized enum value: `Self::W1($bound) =>
/// $body`, `Self::W2($bound) => $body`, ... Resolves `Self` lexically at the
/// expansion site, so it works inside any `impl` block over one of the
/// width-dispatch enums (`PauliSumImpl`, `CircuitImpl`).
///
/// `$body` may be a plain expression (method call, field access, ...); the
/// macro imposes no further shape on it.
macro_rules! for_each_width {
    ($self:expr, |$bound:ident| $body:expr) => {
        match $self {
            Self::W1($bound) => $body,
            Self::W2($bound) => $body,
            Self::W4($bound) => $body,
            Self::W8($bound) => $body,
            Self::W16($bound) => $body,
        }
    };
}

/// Dispatch a pair of values of the *same* width-dispatch enum, requiring
/// both to be the same width variant. Yields `Some($body)` when they match
/// and `None` on a width mismatch (used by `PauliSum::overlap`, where a
/// mismatch means the two sums were monomorphized at different widths).
macro_rules! for_each_width_pair {
    (($a:expr, $b:expr), |$x:ident, $y:ident| $body:expr) => {
        match ($a, $b) {
            (Self::W1($x), Self::W1($y)) => Some($body),
            (Self::W2($x), Self::W2($y)) => Some($body),
            (Self::W4($x), Self::W4($y)) => Some($body),
            (Self::W8($x), Self::W8($y)) => Some($body),
            (Self::W16($x), Self::W16($y)) => Some($body),
            _ => None,
        }
    };
}

/// Cross-enum width dispatch for `PauliSum::propagate`: pairs a
/// `PauliSumImpl` with the `CircuitImpl` of the same width, binds the active
/// width to a local `const $w: usize` in scope for `$body` (needed for
/// `SpecPolicy::<W>`, which const-generic turbofish can't infer from
/// context), and rewraps the result in the matching `PauliSumImpl` variant.
///
/// The `else` arm handles the width-mismatch case, which is unreachable in
/// practice — both `PauliSumImpl::empty_for` and `CircuitImpl::new_for` map
/// `num_qubits` to the same dispatch band — but `propagate` still surfaces it
/// as a `PyResult` error rather than panicking, so the caller supplies a
/// `return Err(...)` (or similar) rather than the macro assuming one.
macro_rules! for_each_width_propagate {
    ($sum:expr, $circuit:expr, |$s:ident, $c:ident, $w:ident| $body:expr, else $mismatch:expr) => {
        match ($sum, $circuit) {
            (PauliSumImpl::W1($s), crate::circuit::CircuitImpl::W1($c)) => {
                const $w: usize = 1;
                PauliSumImpl::W1($body)
            }
            (PauliSumImpl::W2($s), crate::circuit::CircuitImpl::W2($c)) => {
                const $w: usize = 2;
                PauliSumImpl::W2($body)
            }
            (PauliSumImpl::W4($s), crate::circuit::CircuitImpl::W4($c)) => {
                const $w: usize = 4;
                PauliSumImpl::W4($body)
            }
            (PauliSumImpl::W8($s), crate::circuit::CircuitImpl::W8($c)) => {
                const $w: usize = 8;
                PauliSumImpl::W8($body)
            }
            (PauliSumImpl::W16($s), crate::circuit::CircuitImpl::W16($c)) => {
                const $w: usize = 16;
                PauliSumImpl::W16($body)
            }
            _ => $mismatch,
        }
    };
}

/// Pick a width band from a runtime `num_qubits` and construct `Self` in it:
/// `0..=64 => Some(Self::W1($body))`, ..., `513..=1024 => Some(Self::W16($body))`,
/// `None` above 1024. Binds the active width to a local `const $w: usize` in
/// scope for `$body`, mirroring [`for_each_width_propagate`].
///
/// Because the macro expands to `Some(Self::W1($body))` etc., a fallible
/// `$body` containing `?` still works correctly: `?` early-returns from the
/// *enclosing function* (not the macro) on error, bypassing the `Some(...)`
/// wrapper entirely, so `from_strings_dict` uses this directly with no
/// separate fallible variant.
macro_rules! for_num_qubits {
    ($n:expr, |$w:ident| $body:expr) => {
        match $n {
            0..=64 => {
                const $w: usize = 1;
                Some(Self::W1($body))
            }
            65..=128 => {
                const $w: usize = 2;
                Some(Self::W2($body))
            }
            129..=256 => {
                const $w: usize = 4;
                Some(Self::W4($body))
            }
            257..=512 => {
                const $w: usize = 8;
                Some(Self::W8($body))
            }
            513..=1024 => {
                const $w: usize = 16;
                Some(Self::W16($body))
            }
            _ => None,
        }
    };
}
