//! [`Knob`]: one `PAULISTRINGS_*` environment variable, read once per process.

use std::fmt::Debug;
use std::sync::OnceLock;

/// A runtime knob: `name` parsed on first use and cached for the process, its effective value logged at DEBUG on `log_target`.
/// Unset or empty means the default; an unparsable value warns and falls back to the default.
/// Every rank of a group must see the same environment.
pub(crate) struct Knob<T: 'static> {
    name: &'static str,
    log_target: &'static str,
    parse: fn(&str) -> Option<T>,
    default: fn() -> T,
    value: OnceLock<T>,
}

impl<T: Debug + Send + Sync> Knob<T> {
    pub(crate) const fn new(
        name: &'static str,
        log_target: &'static str,
        parse: fn(&str) -> Option<T>,
        default: fn() -> T,
    ) -> Self {
        Self { name, log_target, parse, default, value: OnceLock::new() }
    }

    /// The effective value, read from the environment on the first call only.
    pub(crate) fn get(&self) -> &T {
        self.value.get_or_init(|| self.resolve(std::env::var(self.name).ok().as_deref()))
    }

    /// The value `raw` stands for, without touching the environment or the cache; tests call it directly.
    pub(crate) fn resolve(&self, raw: Option<&str>) -> T {
        let value = match raw.map(str::trim).filter(|text| !text.is_empty()) {
            None => (self.default)(),
            Some(text) => (self.parse)(text).unwrap_or_else(|| {
                log::warn!(
                    target: self.log_target,
                    "{}={text:?} is not a valid value; using the default",
                    self.name
                );
                (self.default)()
            }),
        };
        log::debug!(target: self.log_target, "{} = {value:?}", self.name);
        value
    }
}

/// A positive integer, the parser of count knobs.
pub(crate) fn parse_positive(text: &str) -> Option<usize> {
    text.parse::<usize>().ok().filter(|&n| n > 0)
}

#[cfg(test)]
mod tests;
