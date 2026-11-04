//! [`Table`], an accumulating hash table of terms: a dense entry list plus a linear-probing index of positions into it, and the in-bucket key hash.
//!
//! Entries are dense in `0..len` in insertion order and a key is never removed, so a position stays valid until [`Table::clear`].

use crate::coefficient::Coefficient;

/// One term.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Entry<const W: usize, C: Coefficient> {
    pub(crate) x: [u64; W],
    pub(crate) z: [u64; W],
    pub(crate) c: C,
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

/// Live entries may fill at most `MAX_LOAD_NUMERATOR / 8` of the slots before a rebuild.
const MAX_LOAD_NUMERATOR: usize = 7;

/// Tags probed at once; `tags` mirrors its first `GROUP` bytes past the end, so a group read never wraps.
const GROUP: usize = 16;

const MIN_CAPACITY: usize = GROUP;

/// Bit `k` set iff byte `k` of the group at `tags[0..GROUP]` equals `value`.
#[inline(always)]
fn match_byte(tags: &[u8], value: u8) -> u32 {
    debug_assert!(tags.len() >= GROUP);
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::{
            _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_set1_epi8,
        };
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

#[inline(always)]
fn tag_of(hash: u64) -> u8 {
    0x80 | (hash >> 57) as u8
}

/// An accumulating table of terms with pairwise distinct keys.
#[derive(Clone, Debug, Default)]
pub(crate) struct Table<const W: usize, C: Coefficient> {
    entries: Vec<Entry<W, C>>,
    tags: Vec<u8>,
    positions: Vec<u32>,
}

impl<const W: usize, C: Coefficient> Table<W, C> {
    fn capacity_for(n: usize) -> usize {
        (n * 8).div_ceil(MAX_LOAD_NUMERATOR).next_power_of_two().max(MIN_CAPACITY)
    }

    pub(crate) fn entries(&self) -> &[Entry<W, C>] {
        &self.entries
    }

    /// Forget every entry, keeping the allocations.
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.tags.fill(EMPTY);
    }

    #[inline(always)]
    fn mask(&self) -> usize {
        self.positions.len() - 1
    }

    /// Make room for `additional` pushes without an index rebuild in between.
    #[inline(always)]
    pub(crate) fn reserve(&mut self, additional: usize) {
        let needed = self.entries.len() + additional;
        if needed > self.positions.len() * MAX_LOAD_NUMERATOR / 8 {
            self.rebuild(Self::capacity_for(needed + needed / 8));
        }
        self.entries.reserve(additional);
    }

    /// Re-index every entry at `capacity` index slots.
    #[cold]
    fn rebuild(&mut self, capacity: usize) {
        debug_assert!(capacity.is_power_of_two());
        self.tags.clear();
        self.tags.resize(capacity + GROUP, EMPTY);
        self.positions.clear();
        self.positions.resize(capacity, 0);
        for position in 0..self.entries.len() {
            let e = &self.entries[position];
            let hash = key_hash(&e.x, &e.z);
            let slot = self.free_slot(hash);
            self.claim(slot, hash, position);
        }
    }

    /// The first empty index slot on `hash`'s probe sequence.
    #[inline(always)]
    fn free_slot(&self, hash: u64) -> usize {
        let mask = self.mask();
        let mut base = hash as usize & mask;
        loop {
            let free = match_byte(&self.tags[base..base + GROUP], EMPTY);
            if free != 0 {
                return (base + free.trailing_zeros() as usize) & mask;
            }
            base = (base + GROUP) & mask;
        }
    }

    #[inline(always)]
    fn claim(&mut self, slot: usize, hash: u64, position: usize) {
        debug_assert_eq!(self.tags[slot], EMPTY);
        let tag = tag_of(hash);
        self.tags[slot] = tag;
        // The mirror of slots `0..GROUP`; for any other slot this rewrites the slot itself.
        let mirror = ((slot.wrapping_sub(GROUP)) & self.mask()) + GROUP;
        self.tags[mirror] = tag;
        self.positions[slot] = position as u32;
    }

    /// Add `c` to the coefficient of key `(x, z)`, inserting the key when absent.
    #[inline(always)]
    pub(crate) fn accumulate(&mut self, x: [u64; W], z: [u64; W], c: C) {
        self.reserve(1);
        let hash = key_hash(&x, &z);
        let tag = tag_of(hash);
        let mask = self.mask();
        let mut base = hash as usize & mask;
        loop {
            let group = &self.tags[base..base + GROUP];
            let empty = match_byte(group, EMPTY);
            // Linear probing with no removal stops at the first empty slot, so only matches before it count.
            let mut candidates =
                match_byte(group, tag) & (empty & empty.wrapping_neg()).wrapping_sub(1);
            while candidates != 0 {
                let slot = (base + candidates.trailing_zeros() as usize) & mask;
                let e = &mut self.entries[self.positions[slot] as usize];
                if e.x == x && e.z == z {
                    e.c += c;
                    return;
                }
                candidates &= candidates - 1;
            }
            if empty != 0 {
                let slot = (base + empty.trailing_zeros() as usize) & mask;
                let position = self.entries.len();
                self.claim(slot, hash, position);
                self.entries.push(Entry { x, z, c });
                return;
            }
            base = (base + GROUP) & mask;
        }
    }
}

#[cfg(test)]
mod tests;
