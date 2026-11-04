//! Per-bucket counts of the rows the next layer exports, taken by the coset loop as it writes each bucket, so an exchanging layer's export needs no count pass of its own.
//!
//! Contract: the counts answer for the sum the coset loop left, under whatever cut the layer pass after it leaves pending, and anything that changes the sum or its buckets in between drops them ([`LayerOctaves::drop_ahead`]).

use crate::channel::prepared::{LocalPtm, Prepared};
use crate::coefficient::Coefficient;
use crate::engine::plan::RemoteDelta;
use crate::pauli_string::PauliString;
use crate::pauli_sum::storage::BucketColumns;
use crate::truncation::builtin::octave;
use crate::truncation::octave_window::{BucketOctaves, OctaveWindow, WINDOW_OCTAVES};

use super::LayerOctaves;

/// One bucket's counts ahead: the window counts of its terms that emit a row, and one mark bit per term that does, so the fill visits only those.
#[derive(Clone, Debug)]
pub(crate) struct BucketAhead {
    emitting: BucketOctaves,
    /// Bit `row % 64` of word `row / 64`.
    marks: Vec<u64>,
}

impl Default for BucketAhead {
    fn default() -> Self {
        Self { emitting: [0; WINDOW_OCTAVES + 2], marks: Vec::new() }
    }
}

impl BucketAhead {
    /// The rows marked as emitting, ascending.
    pub(crate) fn marked_rows(&self) -> impl Iterator<Item = usize> + '_ {
        self.marks.iter().enumerate().flat_map(|(word_index, &word)| {
            let mut word = word;
            std::iter::from_fn(move || {
                (word != 0).then(|| {
                    let bit = word.trailing_zeros() as usize;
                    word &= word - 1;
                    word_index * 64 + bit
                })
            })
        })
    }
}

/// Which terms one remote delta of a layer emits a row for, the test the export's count pass applies per term.
#[derive(Clone, Copy)]
pub(crate) enum EmitTest<'p, const W: usize, C: Coefficient = num_complex::Complex64> {
    /// A table entry nonzero on every support pattern.
    Every,
    /// A table entry, by its amplitude on the term's support pattern.
    Table { ptm: &'p LocalPtm<W, C>, entry: usize },
    /// A wide rotation's generator pass, on the terms that anticommute with the generator.
    Generator { generator: &'p PauliString<W> },
}

impl<'p, const W: usize, C: Coefficient> EmitTest<'p, W, C> {
    /// The test of `remote`, a delta of `prepared`.
    pub(crate) fn of(prepared: &'p Prepared<W, C>, remote: &RemoteDelta) -> Self {
        match prepared {
            Prepared::Local(ptm) if ptm.entry_is_dense(remote.entry) => EmitTest::Every,
            Prepared::Local(ptm) => EmitTest::Table { ptm, entry: remote.entry },
            Prepared::Rotation(rotation) => {
                debug_assert_eq!(
                    remote.entry, 1,
                    "a rotation's only remote entry is the generator pass"
                );
                EmitTest::Generator { generator: &rotation.generator }
            }
        }
    }

    /// Whether the term `(x, z)` emits a row.
    #[cfg(test)]
    pub(crate) fn emits(&self, x: &[u64; W], z: &[u64; W]) -> bool {
        match self {
            EmitTest::Every => true,
            EmitTest::Table { ptm, entry } => {
                ptm.deltas()[*entry].amplitude[ptm.support_bits(x, z)] != C::ZERO
            }
            EmitTest::Generator { generator } => {
                !PauliString::<W> { x: *x, z: *z }.commutes_with(generator)
            }
        }
    }
}

impl OctaveWindow {
    /// A coefficient's slot as [`count`](Self::count) bins it, and as the counts ahead bin it, where a NaN, which every cut drops, sits below the window.
    #[inline(always)]
    fn slots_of<C: Coefficient>(&self, coeff: C) -> (usize, usize) {
        let norm_sqr = coeff.norm_sqr();
        let slot = self.slot(octave(norm_sqr));
        (slot, slot * usize::from(!norm_sqr.is_nan()))
    }

    /// Count and mark the terms of `columns` that `test` emits for into `ahead`, and count all of them into `counts` if given, both binned under this window and overwritten.
    pub(crate) fn count_emitting<const W: usize, C: Coefficient>(
        &self,
        columns: &BucketColumns<W, C>,
        test: &EmitTest<'_, W, C>,
        counts: Option<&mut BucketOctaves>,
        ahead: &mut BucketAhead,
    ) {
        match *test {
            EmitTest::Every => self.count_rows(columns, |_, _| true, counts, ahead),
            EmitTest::Table { ptm, entry } => {
                let amplitude = &ptm.deltas()[entry].amplitude;
                let emits =
                    |x: &[u64; W], z: &[u64; W]| amplitude[ptm.support_bits(x, z)] != C::ZERO;
                self.count_rows(columns, emits, counts, ahead);
            }
            EmitTest::Generator { generator } => {
                let emits = |x: &[u64; W], z: &[u64; W]| {
                    !PauliString::<W> { x: *x, z: *z }.commutes_with(generator)
                };
                self.count_rows(columns, emits, counts, ahead);
            }
        }
    }

    /// [`count_emitting`](Self::count_emitting) under one test, monomorphized so the loop carries no dispatch.
    #[inline(always)]
    fn count_rows<const W: usize, C: Coefficient>(
        &self,
        columns: &BucketColumns<W, C>,
        emits: impl Fn(&[u64; W], &[u64; W]) -> bool,
        counts: Option<&mut BucketOctaves>,
        ahead: &mut BucketAhead,
    ) {
        let BucketAhead { emitting, marks } = ahead;
        *emitting = [0; WINDOW_OCTAVES + 2];
        marks.clear();
        marks.resize(columns.len().div_ceil(64), 0);
        let rows = columns.x.iter().zip(&columns.z).zip(&columns.coeff).enumerate();
        match counts {
            Some(counts) => {
                *counts = [0; WINDOW_OCTAVES + 2];
                for (row, ((x, z), &coeff)) in rows {
                    let (slot, emitting_slot) = self.slots_of(coeff);
                    let emits = emits(x, z);
                    counts[slot] += 1;
                    emitting[emitting_slot] += u32::from(emits);
                    marks[row / 64] |= u64::from(emits) << (row % 64);
                }
            }
            None => {
                for (row, ((x, z), &coeff)) in rows {
                    let (_, emitting_slot) = self.slots_of(coeff);
                    let emits = emits(x, z);
                    emitting[emitting_slot] += u32::from(emits);
                    marks[row / 64] |= u64::from(emits) << (row % 64);
                }
            }
        }
    }
}

impl LayerOctaves {
    /// Forget the counts ahead; whatever changes the sum or its buckets outside a coset loop calls it.
    pub(crate) fn drop_ahead(&mut self) {
        self.ahead_window = None;
    }

    /// Whether the last coset loop counted the rows this layer exports.
    pub(crate) fn has_ahead(&self) -> bool {
        self.ahead_window.is_some()
    }

    /// Bucket `b`'s exported rows among the terms the pending cut keeps, from the counts ahead; `None` when they cannot tell, at the one bucket a cut's boundary quota splits.
    pub(crate) fn rows_ahead(&self, b: usize) -> Option<u32> {
        let window = self.ahead_window?;
        let emitting = &self.ahead[b].emitting;
        let Some(cut) = &self.pending else {
            return Some(emitting.iter().sum());
        };
        // A pending cut was decided from the counts the same coset loop took, under the same window, so its octave is inside it.
        let slot = window.slot(cut.bin);
        debug_assert!(
            (1..=WINDOW_OCTAVES).contains(&slot),
            "the cut's octave is outside the window"
        );
        let above: u32 = emitting[slot + 1..].iter().sum();
        match cut.quotas[b] {
            0 => Some(above),
            quota if quota == self.per_bucket[b][slot] as usize => Some(above + emitting[slot]),
            _ => None,
        }
    }

    /// Bucket `b`'s counts ahead where [`rows_ahead`](Self::rows_ahead) answers for it: there the pending cut keeps every boundary term of `b` or none, so its [`BoundaryCut::filter`](crate::truncation::octave_window::BoundaryCut::filter) asked of the marked rows alone keeps what it keeps asked of every row.
    pub(crate) fn bucket_ahead(&self, b: usize) -> Option<&BucketAhead> {
        self.rows_ahead(b).map(|_| &self.ahead[b])
    }
}

#[cfg(test)]
mod tests;
