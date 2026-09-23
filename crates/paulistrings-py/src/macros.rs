//! Width-dispatch macros for the `{W1, W2, W4, W8, W16}` monomorphization enums (`PauliSumImpl`, `CircuitImpl`, `GpuPauliSumImpl`). See `sum.rs` / `circuit.rs`.
//! Deliberately narrow (single dispatch, same-enum pairs, cross-enum pairs, num_qubits-keyed construction) rather than one maximally general macro, for call-site readability.

/// Dispatch a single width-monomorphized enum value across `Self::{W1..W16}`. Resolves `Self` lexically at the expansion site, so it works inside any `impl` block over `PauliSumImpl`/`CircuitImpl`.
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

/// Dispatch a pair of values of the same width-dispatch enum, requiring both to be the same width variant. `Some($body)` on a match, `None` on a width mismatch (used by `PauliSum::overlap`).
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

/// [`for_each_width_pair`] for an operation whose result is itself width-carrying (`PauliSum::__add__`, `PauliString::mul`).
/// The body cannot name its own variant, so the arm binds the variant's constructor as `$wrap`; `$body` applies it to the core value it produced. `Some($body)` on a match, `None` on a width mismatch.
macro_rules! for_each_width_pair_rewrap {
    (($a:expr, $b:expr), |$x:ident, $y:ident, $wrap:ident| $body:expr) => {
        match ($a, $b) {
            (Self::W1($x), Self::W1($y)) => {
                let $wrap = Self::W1;
                Some($body)
            }
            (Self::W2($x), Self::W2($y)) => {
                let $wrap = Self::W2;
                Some($body)
            }
            (Self::W4($x), Self::W4($y)) => {
                let $wrap = Self::W4;
                Some($body)
            }
            (Self::W8($x), Self::W8($y)) => {
                let $wrap = Self::W8;
                Some($body)
            }
            (Self::W16($x), Self::W16($y)) => {
                let $wrap = Self::W16;
                Some($body)
            }
            _ => None,
        }
    };
}

/// Cross-enum width dispatch for the propagate entry points: pairs a `$enum` value (`PauliSumImpl`, `GpuPauliSumImpl`) with the `CircuitImpl` of the same width and binds the active width to a local `const $w: usize` for `$body` (needed for `SpecPolicy::<W>`).
/// `$wrap` is bound to the matching `$enum` variant's constructor, for a body whose result is itself width-carrying; an in-place body ignores it.
/// The `else` arm handles the width-mismatch case, unreachable in practice but surfaced as an error rather than a panic, so the caller supplies the `return Err(...)`.
macro_rules! for_each_width_propagate {
    ($enum:ident, $sum:expr, $circuit:expr, |$s:ident, $c:ident, $w:ident, $wrap:ident| $body:expr, else $mismatch:expr) => {
        match ($sum, $circuit) {
            ($enum::W1($s), crate::circuit::CircuitImpl::W1($c)) => {
                const $w: usize = 1;
                let $wrap = $enum::W1;
                $body
            }
            ($enum::W2($s), crate::circuit::CircuitImpl::W2($c)) => {
                const $w: usize = 2;
                let $wrap = $enum::W2;
                $body
            }
            ($enum::W4($s), crate::circuit::CircuitImpl::W4($c)) => {
                const $w: usize = 4;
                let $wrap = $enum::W4;
                $body
            }
            ($enum::W8($s), crate::circuit::CircuitImpl::W8($c)) => {
                const $w: usize = 8;
                let $wrap = $enum::W8;
                $body
            }
            ($enum::W16($s), crate::circuit::CircuitImpl::W16($c)) => {
                const $w: usize = 16;
                let $wrap = $enum::W16;
                $body
            }
            _ => $mismatch,
        }
    };
}

/// Single dispatch from one width enum into the same variant of another (`PauliSumImpl` <-> `GpuPauliSumImpl`); a `?` in `$body` returns from the enclosing function.
#[cfg(feature = "cuda")]
macro_rules! for_each_width_convert {
    ($from:ident => $to:ident, $value:expr, |$s:ident| $body:expr) => {
        match $value {
            $from::W1($s) => $to::W1($body),
            $from::W2($s) => $to::W2($body),
            $from::W4($s) => $to::W4($body),
            $from::W8($s) => $to::W8($body),
            $from::W16($s) => $to::W16($body),
        }
    };
}

/// Pick a width band from a runtime `num_qubits` and construct `Self` in it: `0..=64 => Some(Self::W1($body))`, ..., `None` above 1024. Binds the active width to a local `const $w: usize`, mirroring [`for_each_width_propagate`].
/// A fallible `$body` containing `?` still works: `?` early-returns from the enclosing function, bypassing the `Some(...)` wrapper, so `from_strings_dict` uses this directly with no separate fallible variant.
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
