use super::*;
use num_complex::Complex64;
use std::collections::HashMap;

fn entry(k: u64, c: f64) -> Entry<2> {
    Entry { x: [k, k >> 3], z: [k.wrapping_mul(0x9E37_79B9_7F4A_7C15), 0], c: Complex64::new(c, 0.0) }
}

fn hash(e: &Entry<2>) -> u64 {
    key_hash(&e.x, &e.z)
}

fn contents(table: &Table<2>) -> HashMap<([u64; 2], [u64; 2]), f64> {
    table.iter().map(|e| ((e.x, e.z), e.c.re)).collect()
}

/// A tiny xorshift, so the mixed workload is reproducible.
fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn a_mixed_workload_matches_a_reference_map() {
    let mut table = Table::<2>::new();
    let mut reference = HashMap::new();
    let mut state = 0x1234_5678u64;
    for step in 0..200_000 {
        let k = next(&mut state) % 3000;
        let e = entry(k, (step % 7) as f64 + 1.0);
        match next(&mut state) % 4 {
            0 | 1 => {
                table.accumulate(e);
                *reference.entry((e.x, e.z)).or_insert(0.0) += e.c.re;
            }
            2 => {
                if let Some(slot) = table.find(hash(&e), &e) {
                    table.erase(slot);
                }
                reference.remove(&(e.x, e.z));
            }
            _ => {
                let threshold = (next(&mut state) % 40) as f64;
                table.retain(|e| e.c.re < threshold || e.x[0] % 2 == 0);
                reference.retain(|key, c| *c < threshold || key.0[0] % 2 == 0);
            }
        }
        if step % 997 == 0 {
            assert_eq!(table.len(), reference.len(), "step {step}");
            assert_eq!(contents(&table), reference, "step {step}");
            assert!(table.len() + table.tombstones <= table.capacity() * MAX_LOAD_NUMERATOR / 8);
        }
    }
    assert_eq!(contents(&table), reference);
}

#[test]
fn churn_at_constant_size_never_grows_the_table() {
    // A permutation layer: every key is erased and a fresh one inserted, at constant occupancy.
    let mut table = Table::<2>::with_capacity(1000);
    for k in 0..1000 {
        table.insert_unique(hash(&entry(k, 1.0)), entry(k, 1.0));
    }
    let capacity = table.capacity();
    for round in 1..50u64 {
        let moved = table.extract(|_| true);
        for e in moved {
            let fresh = entry(e.x[0] + 1000 * round, 1.0);
            table.insert_unique(hash(&fresh), fresh);
        }
        assert_eq!(table.len(), 1000);
    }
    assert_eq!(table.capacity(), capacity);
}

#[test]
fn slot_indices_survive_inserts_and_erases_after_a_reserve() {
    let mut table = Table::<2>::new();
    for k in 0..100 {
        table.insert_unique(hash(&entry(k, k as f64)), entry(k, k as f64));
    }
    table.reserve(10);
    let slot = table.find(hash(&entry(42, 0.0)), &entry(42, 0.0)).unwrap();
    for k in 100..110 {
        table.insert_unique(hash(&entry(k, 0.0)), entry(k, 0.0));
    }
    let other = table.find(hash(&entry(7, 0.0)), &entry(7, 0.0)).unwrap();
    table.erase(other);
    assert_eq!(table.get(slot).c.re, 42.0);
}

#[test]
fn walks_visit_slots_in_ascending_order() {
    let mut table = Table::<2>::new();
    for k in 0..500 {
        table.insert_unique(hash(&entry(k, 1.0)), entry(k, 1.0));
    }
    let mut slots = Vec::new();
    for word in 0..table.live.len() {
        let mut bits = table.live[word];
        while bits != 0 {
            slots.push(word * 64 + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
    assert!(slots.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(slots.len(), 500);
}
