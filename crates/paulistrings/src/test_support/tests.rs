use super::HEAVY_HEX_127_EDGES;

/// Pins the transcribed copy: 144 undirected edges over qubits `0..126`, sorted and unique as `(lo, hi)`, degree histogram 2 × 1, 89 × 2, 36 × 3.
#[test]
fn heavy_hex_127_edges_match_the_source_lattice() {
    assert_eq!(HEAVY_HEX_127_EDGES.len(), 144);
    let mut degree = [0usize; 127];
    let mut prev = (0u32, 0u32);
    for (i, &(a, b)) in HEAVY_HEX_127_EDGES.iter().enumerate() {
        assert!(a < b, "edge {i} is not (lo, hi): ({a}, {b})");
        assert!(b < 127, "edge {i} names qubit {b} outside 0..126");
        if i > 0 {
            assert!(prev < (a, b), "edge {i} breaks the sorted-unique order");
        }
        prev = (a, b);
        degree[a as usize] += 1;
        degree[b as usize] += 1;
    }
    let mut histogram = [0usize; 4];
    for d in degree {
        histogram[d] += 1;
    }
    assert_eq!(histogram, [0, 2, 89, 36]);
}
