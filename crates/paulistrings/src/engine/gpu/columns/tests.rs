use super::*;
use crate::engine::gpu::device;

fn stream() -> Arc<CudaStream> {
    device::context(0).expect("context").default_stream()
}

#[test]
fn reserve_of_an_absurd_term_count_is_out_of_memory() {
    crate::require_cuda!();
    let s = stream();
    let mut columns = DeviceColumns::<16>::with_capacity(&s, 0, 1, 1).expect("tiny alloc");
    let terms = 4_000_000_000usize;
    match columns.reserve(terms, 1) {
        Err(GpuError::OutOfMemory { device, bytes }) => {
            assert_eq!(device, Some(0));
            assert_eq!(bytes, Some(terms as u64 * (16 * 16 + 24)));
        }
        other => panic!("expected OutOfMemory, got {other:?}"),
    }
    assert_eq!(
        columns.term_capacity(),
        0,
        "empty columns release their term columns before growing"
    );
}

#[test]
fn a_failed_reserve_leaves_live_columns_untouched() {
    crate::require_cuda!();
    let s = stream();
    let mut columns = DeviceColumns::<16>::with_capacity(&s, 0, 1, 1).expect("tiny alloc");
    columns.len = 1;
    assert!(matches!(columns.reserve(4_000_000_000, 1), Err(GpuError::OutOfMemory { .. })));
    assert_eq!(columns.term_capacity(), 1, "a failed reserve keeps live columns as they were");
}

#[test]
fn reserve_past_the_u32_index_range_is_unsupported() {
    crate::require_cuda!();
    let s = stream();
    let mut columns = DeviceColumns::<1>::with_capacity(&s, 0, 1, 1).expect("tiny alloc");
    assert!(matches!(columns.reserve(1usize << 33, 1), Err(GpuError::Unsupported(_))));
}

#[test]
fn reserve_keeps_the_live_terms_and_csr() {
    crate::require_cuda!();
    let s = stream();
    let mut columns = DeviceColumns::<2>::with_capacity(&s, 0, 3, 2).expect("alloc");
    let x: Vec<u64> = (0..6).collect();
    let z: Vec<u64> = (10..16).collect();
    let c: Vec<f64> = (0..6).map(|i| i as f64 + 0.5).collect();
    let g: Vec<u64> = vec![7, 8, 9];
    s.memcpy_htod(&x, &mut columns.x).unwrap();
    s.memcpy_htod(&z, &mut columns.z).unwrap();
    s.memcpy_htod(bytemuck::cast_slice::<f64, u8>(&c), &mut columns.coeff).unwrap();
    s.memcpy_htod(&g, &mut columns.fingerprint).unwrap();
    s.memcpy_htod(&[0u32, 1, 3], &mut columns.start).unwrap();
    s.memcpy_htod(&[1u32, 2], &mut columns.lengths).unwrap();
    columns.len = 3;
    columns.buckets = 2;
    columns.reserve(1000, 64).expect("grow");
    assert_eq!(columns.term_capacity(), 1000);
    columns.reserve(10, 4096).expect("grow the CSR alone");
    assert_eq!(columns.term_capacity(), 1000);
    assert_eq!(s.clone_dtoh(&columns.x.slice(0..6)).unwrap(), x);
    assert_eq!(s.clone_dtoh(&columns.z.slice(0..6)).unwrap(), z);
    let coeff = s.clone_dtoh(&columns.coeff.slice(0..6 * size_of::<f64>())).unwrap();
    assert_eq!(
        coeff
            .chunks_exact(8)
            .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>(),
        c
    );
    assert_eq!(s.clone_dtoh(&columns.fingerprint.slice(0..3)).unwrap(), g);
    assert_eq!(s.clone_dtoh(&columns.start.slice(0..3)).unwrap(), vec![0, 1, 3]);
    assert_eq!(s.clone_dtoh(&columns.lengths.slice(0..2)).unwrap(), vec![1, 2]);
    assert_eq!((columns.len, columns.buckets), (3, 2));
}
