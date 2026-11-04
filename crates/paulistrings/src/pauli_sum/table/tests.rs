use super::*;
use std::collections::HashMap;

type Key = ([u64; 2], [u64; 2]);

fn key(k: u64) -> Key {
    ([k, k >> 3], [k.wrapping_mul(0x9E37_79B9_7F4A_7C15), 0])
}

fn contents(table: &Table<2, f64>) -> HashMap<Key, f64> {
    table.entries().iter().map(|e| ((e.x, e.z), e.c)).collect()
}

/// Every entry is reached by exactly one live index slot, and the load bound holds.
fn assert_consistent(table: &Table<2, f64>) {
    let capacity = table.positions.len();
    let mut seen = vec![false; table.entries().len()];
    for slot in 0..capacity {
        if table.tags[slot] >= 0x80 {
            let position = table.positions[slot] as usize;
            assert!(!seen[position], "position {position} indexed twice");
            seen[position] = true;
            let e = &table.entries()[position];
            assert_eq!(table.tags[slot], tag_of(key_hash(&e.x, &e.z)));
        }
    }
    assert!(seen.iter().all(|&s| s), "an entry has no index slot");
    for k in 0..GROUP {
        assert_eq!(table.tags[capacity + k], table.tags[k], "mirror byte {k}");
    }
    assert!(table.entries().len() <= capacity * MAX_LOAD_NUMERATOR / 8);
}

/// A tiny xorshift, so the workload is reproducible.
fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn accumulation_matches_a_reference_map() {
    let mut table = Table::<2, f64>::default();
    let mut reference: HashMap<Key, f64> = HashMap::new();
    let mut state = 0x1234_5678u64;
    for step in 0..100_000u64 {
        let (x, z) = key(next(&mut state) % 3000);
        let c = (step % 7) as f64 + 1.0;
        table.accumulate(x, z, c);
        *reference.entry((x, z)).or_insert(0.0) += c;
        if step % 9973 == 0 {
            assert_eq!(contents(&table), reference, "step {step}");
            assert_consistent(&table);
        }
    }
    assert_eq!(contents(&table), reference);
    assert_consistent(&table);
}

#[test]
fn clear_keeps_the_capacity_and_forgets_every_key() {
    let mut table = Table::<2, f64>::default();
    for k in 0..5000 {
        let (x, z) = key(k);
        table.accumulate(x, z, 1.0);
    }
    let capacity = table.positions.len();
    table.clear();
    assert_eq!(table.entries().len(), 0);
    assert_eq!(table.positions.len(), capacity);
    for k in 0..5000 {
        let (x, z) = key(k + 1);
        table.accumulate(x, z, 2.0);
    }
    assert_eq!(table.positions.len(), capacity);
    assert!(table.entries().iter().all(|e| e.c == 2.0));
    assert_consistent(&table);
}

#[test]
fn entries_keep_insertion_order() {
    let mut table = Table::<2, f64>::default();
    for k in [5u64, 3, 5, 9, 3, 1] {
        let (x, z) = key(k);
        table.accumulate(x, z, k as f64);
    }
    let order: Vec<f64> = table.entries().iter().map(|e| e.c).collect();
    assert_eq!(order, vec![10.0, 6.0, 9.0, 1.0]);
}
