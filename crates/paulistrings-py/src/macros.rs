//! Dispatch macros for the monomorphization enums: the dtype enums `{F64, F32, C128, C64}` (`PauliSumImpl`, `SplitImpl`) over the width enums `{W1, W2, W4, W8, W16}` (`WidthSum`, `WidthSplit`, `CircuitImpl`, `GpuPauliSumImpl`).
//! A call dispatches the dtype first and the width inside it, so each macro stays single-axis.
//! Deliberately narrow (single dispatch, same-enum pairs, cross-enum pairs, num_qubits-keyed construction) rather than one maximally general macro, for call-site readability.

/// Dispatch a single width-monomorphized enum value across `Self::{W1..W16}`. Resolves `Self` lexically at the expansion site, so it works inside any `impl` block over `WidthSum`/`CircuitImpl`.
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

/// Methods that forward to the same-named method of whichever variant `self` holds, through the dispatch macro `$dispatch` (`for_each_dtype` or `for_each_width`).
macro_rules! forward {
    ($dispatch:ident; $($(#[$attr:meta])* $vis:vis fn $name:ident(&self $(, $arg:ident: $ty:ty)*) -> $ret:ty;)*) => {
        $($(#[$attr])* $vis fn $name(&self $(, $arg: $ty)*) -> $ret {
            $dispatch!(self, |s| s.$name($($arg),*))
        })*
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

/// Cross-enum width dispatch for the propagate entry points: pairs a `$enum` value (`WidthSum`, `WidthSplit`, `GpuPauliSumImpl`) with the `CircuitImpl` of the same width and binds the active width to a local `const $w: usize` for a `$body` that names the width.
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

/// Single dispatch from one width enum into the same variant of another (`WidthSum` <-> `GpuPauliSumImpl`, or onto itself); a `?` in `$body` returns from the enclosing function.
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

/// Dispatch a dtype enum value across `Self::{F64, F32, C128, C64}`, binding the width enum it wraps.
macro_rules! for_each_dtype {
    ($self:expr, |$bound:ident| $body:expr) => {
        match $self {
            Self::F64($bound) => $body,
            Self::F32($bound) => $body,
            Self::C128($bound) => $body,
            Self::C64($bound) => $body,
        }
    };
}

/// Single dispatch from one dtype enum into the same variant of another (`PauliSumImpl` <-> `SplitImpl`, or onto itself); a `?` in `$body` returns from the enclosing function.
macro_rules! for_each_dtype_convert {
    ($from:ident => $to:ident, $value:expr, |$s:ident| $body:expr) => {
        match $value {
            $from::F64($s) => $to::F64($body),
            $from::F32($s) => $to::F32($body),
            $from::C128($s) => $to::C128($body),
            $from::C64($s) => $to::C64($body),
        }
    };
}

/// [`for_each_dtype`] over a pair of the same dtype enum, binding `$wrap` to the variant's constructor; `None` on a dtype mismatch, which a caller that converted both operands to one dtype never sees.
macro_rules! for_each_dtype_pair {
    (($a:expr, $b:expr), |$x:ident, $y:ident, $wrap:ident| $body:expr) => {
        match ($a, $b) {
            (Self::F64($x), Self::F64($y)) => {
                let $wrap = Self::F64;
                Some($body)
            }
            (Self::F32($x), Self::F32($y)) => {
                let $wrap = Self::F32;
                Some($body)
            }
            (Self::C128($x), Self::C128($y)) => {
                let $wrap = Self::C128;
                Some($body)
            }
            (Self::C64($x), Self::C64($y)) => {
                let $wrap = Self::C64;
                Some($body)
            }
            _ => None,
        }
    };
}

/// Construct a `$enum` at a runtime [`crate::dtype::Dtype`]: `$body` sees the coefficient type as a local alias `$c` and the matching variant's constructor as `$wrap`, mirroring [`for_num_qubits`].
macro_rules! for_dtype {
    ($enum:ident, $dtype:expr, |$c:ident, $wrap:ident| $body:expr) => {
        match $dtype {
            crate::dtype::Dtype::F64 => {
                type $c = f64;
                let $wrap = $enum::F64;
                $body
            }
            crate::dtype::Dtype::F32 => {
                type $c = f32;
                let $wrap = $enum::F32;
                $body
            }
            crate::dtype::Dtype::C128 => {
                type $c = num_complex::Complex64;
                let $wrap = $enum::C128;
                $body
            }
            crate::dtype::Dtype::C64 => {
                type $c = num_complex::Complex32;
                let $wrap = $enum::C64;
                $body
            }
        }
    };
}

/// Evaluate `$body` with `$policy` bound to the policy a host propagate runs `$tree` (a `&BuiltinTruncation`) as, one monomorphized engine per arm; see [`crate::truncation_dispatch::Lowered`].
macro_rules! with_lowered_policy {
    ($tree:expr, |$policy:ident| $body:expr) => {{
        use crate::truncation_dispatch::{record, Lowered};
        match Lowered::of($tree) {
            Lowered::Coefficient(lowered) => {
                let $policy = record(&lowered);
                $body
            }
            Lowered::CoefficientAndWeight(lowered) => {
                let $policy = record(&lowered);
                $body
            }
            Lowered::Tree(tree) => {
                let $policy = record(tree);
                $body
            }
        }
    }};
}
