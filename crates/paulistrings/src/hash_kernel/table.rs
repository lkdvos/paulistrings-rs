//! [`Table`], one bucket: a dense entry list plus a linear-probing index of positions into it, and the in-bucket key hash.
//!
//! Contracts the kernel relies on: entries are dense in `0..len`; a push appends and leaves every position valid; [`Table::swap_remove`] moves the last entry into the hole, so positions are stable only until a removal.
//! Only coefficients may be written through [`Table::coefficient_mut`]; a key changes only by removal and push, or by [`Table::set_entries`].

use num_complex::Complex64;

/// One term of a bucket.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Entry<const W: usize> {
    pub(crate) x: [u64; W],
    pub(crate) z: [u64; W],
    pub(crate) c: Complex64,
}

/// The 64×64→128 multiply folded to 64 bits.
#[inline(always)]
fn folded_multiply(a: u64, b: u64) -> u64 {
    let product = (a as u128) * (b as u128);
    (product as u64) ^ ((product >> 64) as u64)
}

const KEY_SEEDS: [u64; 4] =
    [0x243F_6A88_85A3_08D3, 0x1319_8A2E_0370_7344, 0xA409_3822_299F_31D0, 0x082E_FA98_EC4E_6C89];

/// The in-bucket key hash; the bucket index is a GF(2)-linear function of the key, so this one must be nonlinear in every bit.
#[inline(always)]
pub(crate) fn key_hash<const W: usize>(x: &[u64; W], z: &[u64; W]) -> u64 {
    let mut h = 0u64;
    for w in 0..W {
        h = h.rotate_left(29)
            ^ folded_multiply(x[w] ^ KEY_SEEDS[(2 * w) % 4], z[w] ^ KEY_SEEDS[(2 * w + 1) % 4]);
    }
    h
}

const EMPTY: u8 = 0;
const DELETED: u8 = 1;

/// Live entries plus tombstones may fill at most `MAX_LOAD_NUMERATOR / 8` of the slots before a rebuild.
const MAX_LOAD_NUMERATOR: usize = 7;

const MIN_CAPACITY: usize = GROUP;

/// Tags probed at once; `tags` mirrors its first `GROUP` bytes past the end, so a group read never wraps.
const GROUP: usize = 16;

/// Bit `k` set iff byte `k` of the group at `tags[0..GROUP]` equals `value`.
#[inline(always)]
fn match_byte(tags: &[u8], value: u8) -> u32 {
    debug_assert!(tags.len() >= GROUP);
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::{_mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_set1_epi8};
        // SAFETY: SSE2 is part of the x86_64 baseline, and the slice holds `GROUP` readable bytes.
        unsafe {
            let group = _mm_loadu_si128(tags.as_ptr().cast());
            _mm_movemask_epi8(_mm_cmpeq_epi8(group, _mm_set1_epi8(value as i8))) as u32
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (0..GROUP).fold(0, |bits, k| bits | (((tags[k] == value) as u32) << k))
    }
}

/// Bit `k` set iff byte `k` of the group is free (empty or a tombstone): its high bit is clear.
#[inline(always)]
fn match_free(tags: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::{_mm_loadu_si128, _mm_movemask_epi8};
        // SAFETY: as `match_byte`.
        unsafe { !(_mm_movemask_epi8(_mm_loadu_si128(tags.as_ptr().cast())) as u32) & 0xFFFF }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (0..GROUP).fold(0, |bits, k| bits | (((tags[k] < 0x80) as u32) << k))
    }
}

#[inline(always)]
fn tag_of(hash: u64) -> u8 {
    0x80 | (hash >> 57) as u8
}

/// One bucket.
#[derive(Clone, Debug, Default)]
pub(crate) struct Table<const W: usize> {
    entries: Vec<Entry<W>>,
    /// The index slot holding each position.
    slot_of: Vec<u32>,
    tags: Vec<u8>,
    positions: Vec<u32>,
    tombstones: usize,
}

impl<const W: usize> Table<W> {
    /// The table of `entries`, whose keys are pairwise distinct.
    pub(crate) fn from_entries(entries: Vec<Entry<W>>) -> Self {
        let mut table = Self { entries, ..Self::default() };
        table.rebuild(Self::capacity_for(table.entries.len()));
        table
    }

    fn capacity_for(n: usize) -> usize {
        (n * 8).div_ceil(MAX_LOAD_NUMERATOR).next_power_of_two().max(MIN_CAPACITY)
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Resident bytes, the footprint once divided into `len`.
    pub(crate) fn bytes(&self) -> usize {
        self.entries.capacity() * std::mem::size_of::<Entry<W>>()
            + self.slot_of.capacity() * 4
            + self.tags.len()
            + self.positions.len() * 4
    }

    pub(crate) fn entries(&self) -> &[Entry<W>] {
        &self.entries
    }

    #[inline(always)]
    pub(crate) fn coefficient_mut(&mut self, position: usize) -> &mut Complex64 {
        &mut self.entries[position].c
    }

    /// Every key with its coefficient, writable, in position order.
    pub(crate) fn coefficients_mut(
        &mut self,
    ) -> impl Iterator<Item = (&[u64; W], &[u64; W], &mut Complex64)> + '_ {
        self.entries.iter_mut().map(|e| (&e.x, &e.z, &mut e.c))
    }

    #[inline(always)]
    fn mask(&self) -> usize {
        self.positions.len() - 1
    }

    /// Make room for `additional` pushes without an index rebuild in between.
    #[inline(always)]
    pub(crate) fn reserve(&mut self, additional: usize) {
        let limit = self.positions.len() * MAX_LOAD_NUMERATOR / 8;
        if self.entries.len() + self.tombstones + additional > limit {
            let needed = self.entries.len() + additional;
            self.rebuild(Self::capacity_for(needed + needed / 8).max(self.positions.len()));
        }
        self.entries.reserve(additional);
        self.slot_of.reserve(additional);
    }

    /// Re-index every entry at `capacity` index slots, clearing tombstones.
    #[cold]
    fn rebuild(&mut self, capacity: usize) {
        debug_assert!(capacity.is_power_of_two());
        self.tags.clear();
        self.tags.resize(capacity + GROUP, EMPTY);
        self.positions.clear();
        self.positions.resize(capacity, 0);
        self.tombstones = 0;
        self.slot_of.clear();
        for position in 0..self.entries.len() {
            let e = &self.entries[position];
            let slot = self.free_slot(key_hash(&e.x, &e.z));
            self.claim(slot, key_hash(&e.x, &e.z), position);
            self.slot_of.push(slot as u32);
        }
    }

    /// The first free index slot on `hash`'s probe sequence.
    #[inline(always)]
    fn free_slot(&self, hash: u64) -> usize {
        let mask = self.mask();
        let mut base = hash as usize & mask;
        loop {
            let free = match_free(&self.tags[base..base + GROUP]);
            if free != 0 {
                return (base + free.trailing_zeros() as usize) & mask;
            }
            base = (base + GROUP) & mask;
        }
    }

    #[inline(always)]
    fn set_tag(&mut self, slot: usize, tag: u8) {
        self.tags[slot] = tag;
        // The mirror of slots `0..GROUP`; for any other slot this rewrites the slot itself.
        let mirror = ((slot.wrapping_sub(GROUP)) & self.mask()) + GROUP;
        self.tags[mirror] = tag;
    }

    #[inline(always)]
    fn claim(&mut self, slot: usize, hash: u64, position: usize) {
        debug_assert!(self.tags[slot] < 0x80);
        self.tombstones -= (self.tags[slot] == DELETED) as usize;
        self.set_tag(slot, tag_of(hash));
        self.positions[slot] = position as u32;
    }

    /// The position of the entry with key `(x, z)`.
    #[inline(always)]
    pub(crate) fn find(&self, hash: u64, x: &[u64; W], z: &[u64; W]) -> Option<usize> {
        if self.positions.is_empty() {
            return None;
        }
        let tag = tag_of(hash);
        let mask = self.mask();
        let mut base = hash as usize & mask;
        loop {
            let group = &self.tags[base..base + GROUP];
            let empty = match_byte(group, EMPTY);
            // Linear probing stops at the first empty slot, so only matches before it count.
            let mut candidates = match_byte(group, tag) & (empty & empty.wrapping_neg()).wrapping_sub(1);
            while candidates != 0 {
                let slot = (base + candidates.trailing_zeros() as usize) & mask;
                let position = self.positions[slot] as usize;
                let e = &self.entries[position];
                if e.x == *x && e.z == *z {
                    return Some(position);
                }
                candidates &= candidates - 1;
            }
            if empty != 0 {
                return None;
            }
            base = (base + GROUP) & mask;
        }
    }

    /// Move the entries out, leaving the table empty with its allocations.
    pub(crate) fn take_entries(&mut self) -> Vec<Entry<W>> {
        std::mem::take(&mut self.entries)
    }

    /// Replace the entries, whose keys are pairwise distinct, and re-index them.
    pub(crate) fn set_entries(&mut self, entries: Vec<Entry<W>>) {
        self.entries = entries;
        self.rebuild(Self::capacity_for(self.entries.len()));
    }

    /// Append `e`, whose key is absent.
    #[inline(always)]
    pub(crate) fn push_unique(&mut self, e: Entry<W>) {
        self.reserve(1);
        let hash = key_hash(&e.x, &e.z);
        let slot = self.free_slot(hash);
        let position = self.entries.len();
        self.claim(slot, hash, position);
        self.slot_of.push(slot as u32);
        self.entries.push(e);
    }

    /// Remove the entry at `position`, moving the last entry into its place.
    #[inline(always)]
    pub(crate) fn swap_remove(&mut self, position: usize) {
        let slot = self.slot_of[position] as usize;
        // Branch-free: no probe passes an empty slot, so the slot may be emptied when its successor is empty.
        let tombstone = self.tags[(slot + 1) & self.mask()] != EMPTY;
        self.set_tag(slot, if tombstone { DELETED } else { EMPTY });
        self.tombstones += tombstone as usize;
        self.entries.swap_remove(position);
        self.slot_of.swap_remove(position);
        if position < self.entries.len() {
            self.positions[self.slot_of[position] as usize] = position as u32;
        }
    }

    /// Remove every entry whose bit is set in `dead`, which is left cleared.
    pub(crate) fn remove_marked(&mut self, dead: &mut [u64]) {
        // Descending, so the entry each removal moves down is never one still to be removed.
        for word in (0..dead.len()).rev() {
            let mut bits = std::mem::take(&mut dead[word]);
            while bits != 0 {
                let bit = 63 - bits.leading_zeros() as usize;
                bits &= !(1 << bit);
                self.swap_remove(word * 64 + bit);
            }
        }
    }

    /// Keep the entries `keep` accepts, in one pass.
    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&Entry<W>) -> bool) {
        let mut dead = vec![0u64; self.entries.len().div_ceil(64)];
        for (position, e) in self.entries.iter().enumerate() {
            dead[position / 64] |= (!keep(e) as u64) << (position % 64);
        }
        self.remove_marked(&mut dead);
    }

    /// Split off the entries `take` accepts, re-indexing both halves at their own size.
    pub(crate) fn split_off_where(&mut self, take: impl Fn(&Entry<W>) -> bool) -> Self {
        let (moving, staying): (Vec<_>, Vec<_>) =
            std::mem::take(&mut self.entries).into_iter().partition(|e| take(e));
        *self = Self::from_entries(staying);
        Self::from_entries(moving)
    }

    /// Append every entry of `other`, whose keys are all absent here.
    pub(crate) fn absorb(&mut self, other: Self) {
        self.reserve(other.len());
        for e in other.entries {
            self.push_unique(e);
        }
    }
}

#[cfg(test)]
mod tests;
