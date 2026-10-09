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
            assert_eq!(device, 0);
            assert_eq!(bytes, terms as u64 * (16 * 16 + 24));
        }
        other => panic!("expected OutOfMemory, got {other:?}"),
    }
    assert_eq!(
        columns.term_capacity(),
        1,
        "a failed reserve leaves the columns untouched"
    );
}

#[test]
fn reserve_past_the_u32_index_range_is_unsupported() {
    crate::require_cuda!();
    let s = stream();
    let mut columns = DeviceColumns::<1>::with_capacity(&s, 0, 1, 1).expect("tiny alloc");
    assert!(matches!(
        columns.reserve(1usize << 33, 1),
        Err(GpuError::Unsupported(_))
    ));
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
    s.memcpy_htod(&c, &mut columns.coeff).unwrap();
    s.memcpy_htod(&g, &mut columns.g).unwrap();
    s.memcpy_htod(&[0u32, 1, 3], &mut columns.start).unwrap();
    s.memcpy_htod(&[1u32, 2], &mut columns.lens).unwrap();
    columns.len = 3;
    columns.buckets = 2;
    columns.reserve(1000, 64).expect("grow");
    assert_eq!(columns.term_capacity(), 1000);
    columns.reserve(10, 4096).expect("grow the CSR alone");
    assert_eq!(columns.term_capacity(), 1000);
    assert_eq!(s.clone_dtoh(&columns.x.slice(0..6)).unwrap(), x);
    assert_eq!(s.clone_dtoh(&columns.z.slice(0..6)).unwrap(), z);
    assert_eq!(s.clone_dtoh(&columns.coeff.slice(0..6)).unwrap(), c);
    assert_eq!(s.clone_dtoh(&columns.g.slice(0..3)).unwrap(), g);
    assert_eq!(
        s.clone_dtoh(&columns.start.slice(0..3)).unwrap(),
        vec![0, 1, 3]
    );
    assert_eq!(s.clone_dtoh(&columns.lens.slice(0..2)).unwrap(), vec![1, 2]);
    assert_eq!((columns.len, columns.buckets), (3, 2));
}
