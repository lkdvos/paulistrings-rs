//! The `phase-timing` instrumentation macros, which expand to nothing without the feature.

/// `let mut $stamp = Stamp::now();`, starting a chain of [`lap!`]s.
macro_rules! stamp {
    ($stamp:ident) => {
        #[cfg(feature = "phase-timing")]
        let mut $stamp = $crate::engine::stats::Stamp::now();
    };
}

/// Add the time since the last stamp to the `u64` counter `$slot`, and re-arm.
macro_rules! lap {
    ($stamp:ident, $slot:expr) => {
        #[cfg(feature = "phase-timing")]
        $stamp.lap(&mut $slot);
    };
}

/// Add `$amount` to the `u64` counter `$slot`.
macro_rules! tally {
    ($slot:expr, $amount:expr) => {
        #[cfg(feature = "phase-timing")]
        {
            $slot += ($amount) as u64;
        }
    };
}

/// A `let` or a block that exists only under the feature.
macro_rules! timed {
    (let $($declaration:tt)*) => {
        #[cfg(feature = "phase-timing")]
        let $($declaration)*;
    };
    ($($statement:tt)*) => {
        #[cfg(feature = "phase-timing")]
        {
            $($statement)*
        }
    };
}

pub(crate) use {lap, stamp, tally, timed};
