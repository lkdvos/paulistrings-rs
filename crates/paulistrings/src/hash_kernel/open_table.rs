//! [`Table`], one bucket's linear-probing hash table: entries in slot order, a tag byte per slot for probing, and an occupancy bitmap for walks.
//!
//! Contracts the kernels rely on: an insert never moves a live entry and an erase leaves a tombstone, so a slot index stays valid until the next [`Table::reserve`] that rebuilds; a walk visits live slots in ascending order, a word of the bitmap at a time.
//! Tombstones are cleared by rebuilding at the same capacity; the table grows only when its live entries need it.

use super::table::{key_hash, Entry};

pub(super) const EMPTY: u8 = 0;
pub(super) const DELETED: u8 = 1;

/// Live entries plus tombstones may fill at most `MAX_LOAD_NUMERATOR / 8` of the slots before a rebuild.
pub(super) const MAX_LOAD_NUMERATOR: usize = 7;

pub(super) const MIN_CAPACITY: usize = GROUP;

/// Tags probed at once; `tags` mirrors its first `GROUP` bytes past the end, so a group read never wraps.
pub(super) const GROUP: usize = 16;

/// Bit `k` set iff byte `k` of the group at `tags[0..GROUP]` equals `value`.
#[inline(always)]
pub(super) fn match_byte(tags: &[u8], value: u8) -> u32 {
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

/// Slots a walk prefetches ahead of the one it is on; `PAULISTRINGS_PREFETCH` overrides it for the prototype's measurements, 0 disabling.
pub(crate) static PREFETCH_SLOTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(32);

#[inline(always)]
pub(super) fn prefetch<T>(slots: &[T], slot: usize) {
    #[cfg(target_arch = "x86_64")]
    if slot < slots.len() {
        // SAFETY: a prefetch of an in-bounds address has no architectural effect.
        unsafe {
            core::arch::x86_64::_mm_prefetch(
                slots.as_ptr().add(slot).cast(),
                core::arch::x86_64::_MM_HINT_T0,
            )
        }
    }
}

/// Bit `k` set iff byte `k` of the group is free (empty or a tombstone): its high bit is clear.
#[inline(always)]
pub(super) fn match_free(tags: &[u8]) -> u32 {
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
pub(super) fn tag_of(hash: u64) -> u8 {
    0x80 | (hash >> 57) as u8
}

/// One bucket.
#[derive(Clone, Debug, Default)]
pub(crate) struct Table<const W: usize> {
    slots: Vec<Entry<W>>,
    tags: Vec<u8>,
    /// Bit `i % 64` of word `i / 64` is set iff slot `i` is live.
    live: Vec<u64>,
    len: usize,
    tombstones: usize,
}

impl<const W: usize> Table<W> {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// A table that holds `n` entries without rebuilding.
    pub(crate) fn with_capacity(n: usize) -> Self {
        let mut table = Self::new();
        table.allocate(Self::capacity_for(n));
        table
    }

    fn capacity_for(n: usize) -> usize {
        (n * 8).div_ceil(MAX_LOAD_NUMERATOR).next_power_of_two().max(MIN_CAPACITY)
    }

    fn allocate(&mut self, capacity: usize) {
        debug_assert!(capacity.is_power_of_two());
        let zero = Entry { x: [0; W], z: [0; W], c: num_complex::Complex64::new(0.0, 0.0) };
        self.slots = vec![zero; capacity];
        self.tags = vec![EMPTY; capacity + GROUP];
        self.live = vec![0; capacity.div_ceil(64)];
        self.len = 0;
        self.tombstones = 0;
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Number of slots.
    pub(crate) fn capacity(&self) -> usize {
        self.slots.len()
    }

    #[inline(always)]
    fn mask(&self) -> usize {
        self.slots.len() - 1
    }

    /// Make room for `additional` inserts without a rebuild in between; invalidates slot indices when it rebuilds.
    #[inline(always)]
    pub(crate) fn reserve(&mut self, additional: usize) {
        let limit = self.slots.len() * MAX_LOAD_NUMERATOR / 8;
        if self.len + self.tombstones + additional > limit {
            // An eighth of headroom past the request, so a table at its limit does not rebuild on every insert.
            let needed = self.len + additional;
            self.rebuild(needed + needed / 8);
        }
    }

    #[cold]
    fn rebuild(&mut self, needed: usize) {
        let capacity = Self::capacity_for(needed).max(self.slots.len());
        let old = std::mem::take(self);
        self.allocate(capacity);
        for e in old.iter() {
            self.insert_unique(key_hash(&e.x, &e.z), *e);
        }
    }

    /// The live slot holding `key`'s key.
    #[inline(always)]
    pub(crate) fn find(&self, hash: u64, key: &Entry<W>) -> Option<usize> {
        if self.slots.is_empty() {
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
                if self.slots[slot].same_key(key) {
                    return Some(slot);
                }
                candidates &= candidates - 1;
            }
            if empty != 0 {
                return None;
            }
            base = (base + GROUP) & mask;
        }
    }

    /// `Ok(slot)` holding `key`'s key, or `Err(slot)` where it would be inserted; call [`Self::reserve`] first.
    #[inline(always)]
    pub(crate) fn find_or_slot(&self, hash: u64, key: &Entry<W>) -> Result<usize, usize> {
        debug_assert!(self.len + self.tombstones < self.slots.len());
        let tag = tag_of(hash);
        let mask = self.mask();
        let mut base = hash as usize & mask;
        let mut free = usize::MAX;
        loop {
            let group = &self.tags[base..base + GROUP];
            let empty = match_byte(group, EMPTY);
            let before_empty = (empty & empty.wrapping_neg()).wrapping_sub(1);
            let mut candidates = match_byte(group, tag) & before_empty;
            while candidates != 0 {
                let slot = (base + candidates.trailing_zeros() as usize) & mask;
                if self.slots[slot].same_key(key) {
                    return Ok(slot);
                }
                candidates &= candidates - 1;
            }
            let free_here = match_free(group);
            if free == usize::MAX && free_here != 0 {
                free = (base + free_here.trailing_zeros() as usize) & mask;
            }
            if empty != 0 {
                return Err(free);
            }
            base = (base + GROUP) & mask;
        }
    }

    /// Write `tag` at `slot` and its mirror.
    #[inline(always)]
    fn set_tag(&mut self, slot: usize, tag: u8) {
        self.tags[slot] = tag;
        // The mirror of slots `0..GROUP`; for any other slot this rewrites the slot itself.
        let mirror = ((slot.wrapping_sub(GROUP)) & self.mask()) + GROUP;
        self.tags[mirror] = tag;
    }

    /// Put `e` at a free `slot` from [`Self::find_or_slot`].
    #[inline(always)]
    pub(crate) fn insert_at(&mut self, slot: usize, hash: u64, e: Entry<W>) {
        debug_assert!(self.tags[slot] < 0x80);
        self.tombstones -= (self.tags[slot] == DELETED) as usize;
        self.set_tag(slot, tag_of(hash));
        self.slots[slot] = e;
        self.live[slot / 64] |= 1 << (slot % 64);
        self.len += 1;
    }

    /// Insert `e`, whose key is absent.
    #[inline(always)]
    pub(crate) fn insert_unique(&mut self, hash: u64, e: Entry<W>) {
        self.reserve(1);
        let mask = self.mask();
        let mut base = hash as usize & mask;
        loop {
            let free = match_free(&self.tags[base..base + GROUP]);
            if free != 0 {
                let slot = (base + free.trailing_zeros() as usize) & mask;
                self.insert_at(slot, hash, e);
                return;
            }
            base = (base + GROUP) & mask;
        }
    }

    /// Add `e.c` onto `e`'s key, inserting it if absent.
    #[inline(always)]
    pub(crate) fn accumulate(&mut self, e: Entry<W>) {
        self.reserve(1);
        let hash = key_hash(&e.x, &e.z);
        match self.find_or_slot(hash, &e) {
            Ok(slot) => self.slots[slot].c += e.c,
            Err(slot) => self.insert_at(slot, hash, e),
        }
    }

    #[inline(always)]
    pub(crate) fn get(&self, slot: usize) -> &Entry<W> {
        debug_assert!(self.tags[slot] >= 0x80);
        &self.slots[slot]
    }

    #[inline(always)]
    pub(crate) fn get_mut(&mut self, slot: usize) -> &mut Entry<W> {
        debug_assert!(self.tags[slot] >= 0x80);
        &mut self.slots[slot]
    }

    /// Erase the live entry at `slot`, leaving a tombstone, or nothing when the next slot is empty and no probe can pass through.
    #[inline(always)]
    pub(crate) fn erase(&mut self, slot: usize) {
        debug_assert!(self.tags[slot] >= 0x80);
        self.live[slot / 64] &= !(1 << (slot % 64));
        self.len -= 1;
        // Branch-free: no probe passes an empty slot, so the slot may be emptied when its successor is empty.
        let tombstone = self.tags[(slot + 1) & self.mask()] != EMPTY;
        self.set_tag(slot, if tombstone { DELETED } else { EMPTY });
        self.tombstones += tombstone as usize;
    }

    /// Live entries in slot order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &Entry<W>> + '_ {
        let ahead = PREFETCH_SLOTS.load(std::sync::atomic::Ordering::Relaxed);
        self.live.iter().enumerate().flat_map(move |(word, &bits)| {
            BitIter(bits).map(move |bit| {
                let slot = word * 64 + bit;
                if ahead != 0 {
                    prefetch(&self.slots, slot + ahead);
                }
                &self.slots[slot]
            })
        })
    }

    /// `f` on every live entry in slot order; `false` erases it.
    #[inline(always)]
    pub(crate) fn retain(&mut self, mut f: impl FnMut(&mut Entry<W>) -> bool) {
        let ahead = PREFETCH_SLOTS.load(std::sync::atomic::Ordering::Relaxed);
        for word in 0..self.live.len() {
            let mut bits = self.live[word];
            let mut erased = 0u64;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let slot = word * 64 + bit;
                if ahead != 0 {
                    prefetch(&self.slots, slot + ahead);
                }
                erased |= ((!f(&mut self.slots[slot])) as u64) << bit;
            }
            while erased != 0 {
                let bit = erased.trailing_zeros() as usize;
                erased &= erased - 1;
                self.erase(word * 64 + bit);
            }
        }
    }

    /// `f` on every live entry in slot order.
    #[inline(always)]
    pub(crate) fn for_each_mut(&mut self, mut f: impl FnMut(&mut Entry<W>)) {
        let ahead = PREFETCH_SLOTS.load(std::sync::atomic::Ordering::Relaxed);
        for word in 0..self.live.len() {
            let mut bits = self.live[word];
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if ahead != 0 {
                    prefetch(&self.slots, word * 64 + bit + ahead);
                }
                f(&mut self.slots[word * 64 + bit]);
            }
        }
    }

    /// Move out every entry `take` accepts, in slot order.
    pub(crate) fn extract(&mut self, take: impl Fn(&Entry<W>) -> bool) -> Vec<Entry<W>> {
        let mut out = Vec::new();
        self.retain(|e| {
            let moving = take(e);
            if moving {
                out.push(*e);
            }
            !moving
        });
        out
    }

    /// Rebuild at the smallest capacity holding the live entries.
    pub(crate) fn shrink_to_fit(&mut self) {
        let capacity = Self::capacity_for(self.len);
        if capacity < self.slots.len() || self.tombstones > 0 {
            let old = std::mem::take(self);
            self.allocate(capacity);
            for e in old.iter() {
                self.insert_unique(key_hash(&e.x, &e.z), *e);
            }
        }
    }
}

struct BitIter(u64);

impl Iterator for BitIter {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }
        let bit = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1;
        Some(bit)
    }
}

#[cfg(test)]
mod tests;
