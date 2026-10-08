use super::*;
use crate::engine::gpu::device;

fn stream() -> Arc<CudaStream> {
    device::context(0).expect("context").default_stream()
}

#[test]
fn reserve_of_an_absurd_term_count_is_out_of_memory() {
    crate::require_cuda!();
    let s = stream();
    let mut cols = DeviceColumns::<16>::with_capacity(&s, 0, 1, 1).expect("tiny alloc");
    let terms = 4_000_000_000usize;
    match cols.reserve(terms, 1) {
        Err(GpuError::OutOfMemory { device, bytes }) => {
            assert_eq!(device, 0);
            assert_eq!(bytes, terms as u64 * (16 * 16 + 24));
        }
        other => panic!("expected OutOfMemory, got {other:?}"),
    }
    assert_eq!(
        cols.term_capacity(),
        1,
        "a failed reserve leaves the columns untouched"
    );
}

#[test]
fn reserve_past_the_u32_index_range_is_unsupported() {
    crate::require_cuda!();
    let s = stream();
    let mut cols = DeviceColumns::<1>::with_capacity(&s, 0, 1, 1).expect("tiny alloc");
    assert!(matches!(
        cols.reserve(1usize << 33, 1),
        Err(GpuError::Unsupported(_))
    ));
}

#[test]
fn reserve_keeps_the_live_terms_and_csr() {
    crate::require_cuda!();
    let s = stream();
    let mut cols = DeviceColumns::<2>::with_capacity(&s, 0, 3, 2).expect("alloc");
    let x: Vec<u64> = (0..6).collect();
    let z: Vec<u64> = (10..16).collect();
    let c: Vec<f64> = (0..6).map(|i| i as f64 + 0.5).collect();
    let g: Vec<u64> = vec![7, 8, 9];
    s.memcpy_htod(&x, &mut cols.x).unwrap();
    s.memcpy_htod(&z, &mut cols.z).unwrap();
    s.memcpy_htod(&c, &mut cols.coeff).unwrap();
    s.memcpy_htod(&g, &mut cols.g).unwrap();
    s.memcpy_htod(&[0u32, 1, 3], &mut cols.start).unwrap();
    s.memcpy_htod(&[1u32, 2], &mut cols.lens).unwrap();
    cols.len = 3;
    cols.buckets = 2;
    cols.reserve(1000, 64).expect("grow");
    assert_eq!(cols.term_capacity(), 1000);
    cols.reserve(10, 4096).expect("grow the CSR alone");
    assert_eq!(cols.term_capacity(), 1000);
    assert_eq!(s.clone_dtoh(&cols.x.slice(0..6)).unwrap(), x);
    assert_eq!(s.clone_dtoh(&cols.z.slice(0..6)).unwrap(), z);
    assert_eq!(s.clone_dtoh(&cols.coeff.slice(0..6)).unwrap(), c);
    assert_eq!(s.clone_dtoh(&cols.g.slice(0..3)).unwrap(), g);
    assert_eq!(
        s.clone_dtoh(&cols.start.slice(0..3)).unwrap(),
        vec![0, 1, 3]
    );
    assert_eq!(s.clone_dtoh(&cols.lens.slice(0..2)).unwrap(), vec![1, 2]);
    assert_eq!((cols.len, cols.buckets), (3, 2));
}
