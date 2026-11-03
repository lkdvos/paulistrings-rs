use super::*;
use std::collections::HashMap;

fn entry(k: u64, c: f64) -> Entry<2> {
    Entry { x: [k, k >> 3], z: [k.wrapping_mul(0x9E37_79B9_7F4A_7C15), 0], c: Complex64::new(c, 0.0) }
}

fn find(table: &IndexedTable<2>, e: &Entry<2>) -> Option<usize> {
    table.find(key_hash(&e.x, &e.z), &e.x, &e.z)
}

fn contents(table: &IndexedTable<2>) -> HashMap<([u64; 2], [u64; 2]), f64> {
    table.entries().iter().map(|e| ((e.x, e.z), e.c.re)).collect()
}

/// Every position is found by its own key, and every index slot points back at it.
fn assert_consistent(table: &IndexedTable<2>) {
    assert_eq!(table.slot_of.len(), table.len());
    for (position, e) in table.entries().iter().enumerate() {
        assert_eq!(find(table, e), Some(position));
        assert_eq!(table.positions[table.slot_of[position] as usize] as usize, position);
    }
    let live = table.tags[..table.positions.len()].iter().filter(|&&t| t >= 0x80).count();
    assert_eq!(live, table.len());
    assert!(table.len() + table.tombstones <= table.positions.len() * MAX_LOAD_NUMERATOR / 8);
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
    let mut table = IndexedTable::<2>::default();
    let mut reference = HashMap::new();
    let mut state = 0x1234_5678u64;
    for step in 0..100_000 {
        let k = next(&mut state) % 3000;
        let e = entry(k, (step % 7) as f64 + 1.0);
        match next(&mut state) % 4 {
            0 | 1 => {
                match find(&table, &e) {
                    Some(position) => *table.coefficient_mut(position) += e.c,
                    None => table.push_unique(e),
                }
                *reference.entry((e.x, e.z)).or_insert(0.0) += e.c.re;
            }
            2 => {
                if let Some(position) = find(&table, &e) {
                    table.swap_remove(position);
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
            assert_eq!(contents(&table), reference, "step {step}");
            assert_consistent(&table);
        }
    }
    assert_eq!(contents(&table), reference);
    assert_consistent(&table);
}

#[test]
fn removing_a_scattered_set_keeps_the_rest_findable() {
    let mut table = IndexedTable::from_entries((0..1000).map(|k| entry(k, k as f64)).collect());
    let mut dead = vec![0u64; 1000usize.div_ceil(64)];
    let mut removed = 0;
    for p in (0..1000).filter(|p| p % 3 == 0 || p % 7 == 5) {
        dead[p / 64] |= 1 << (p % 64);
        removed += 1;
    }
    table.remove_marked(&mut dead);
    assert!(dead.iter().all(|&w| w == 0));
    assert_eq!(table.len(), 1000 - removed);
    let mut remaining: Vec<u64> = table.entries().iter().map(|e| e.c.re as u64).collect();
    remaining.sort_unstable();
    let want: Vec<u64> = (0..1000).filter(|p| !(p % 3 == 0 || p % 7 == 5)).collect();
    assert_eq!(remaining, want);
    assert_consistent(&table);
}

#[test]
fn churn_at_constant_size_never_grows_the_index() {
    let mut table = IndexedTable::from_entries((0..500).map(|k| entry(k, 1.0)).collect());
    let capacity = table.positions.len();
    for round in 1..200u64 {
        table.swap_remove(0);
        table.push_unique(entry(500 * round + round, 1.0));
    }
    assert_eq!(table.positions.len(), capacity);
    assert_consistent(&table);
}

#[test]
fn splitting_and_absorbing_round_trip() {
    let mut table = IndexedTable::from_entries((0..777).map(|k| entry(k, k as f64)).collect());
    let upper = table.split_off_where(|e| e.x[0] % 2 == 1);
    assert_eq!((table.len(), upper.len()), (389, 388));
    assert_consistent(&table);
    assert_consistent(&upper);
    table.absorb(upper);
    assert_eq!(table.len(), 777);
    assert_consistent(&table);
}

#[test]
fn replaced_entries_are_reindexed() {
    let mut table = IndexedTable::from_entries((0..300).map(|k| entry(k, 1.0)).collect());
    let mut entries = table.take_entries();
    entries.retain(|e| e.x[0] % 4 != 0);
    entries.extend((1000..1500).map(|k| entry(k, 2.0)));
    table.set_entries(entries);
    assert_eq!(table.len(), 225 + 500);
    assert_eq!(find(&table, &entry(4, 1.0)), None);
    assert_consistent(&table);
}
