use super::*;
use crate::pauli_string::PauliString;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::pauli_sum::storage::DEFAULT_HASH_SEED;
use crate::phase::Phase;
use crate::test_support::{assert_same_terms, low_weight_sum, rand_sum};

/// The device columns in storage order, gathered by nothing: valid while the columns are compact.
struct Raw<const W: usize> {
    x: Vec<[u64; W]>,
    z: Vec<[u64; W]>,
    g: Vec<u64>,
}

impl<const W: usize> GpuSum<W> {
    fn download_raw(&self) -> Raw<W> {
        let n = self.columns.len;
        let s = &self.stream;
        let words = |v: Vec<u64>| -> Vec<[u64; W]> {
            v.chunks_exact(W)
                .map(|c| std::array::from_fn(|w| c[w]))
                .collect()
        };
        let x = s.clone_dtoh(&self.columns.x.slice(0..n * W)).unwrap();
        let z = s.clone_dtoh(&self.columns.z.slice(0..n * W)).unwrap();
        let g = s.clone_dtoh(&self.columns.g.slice(0..n)).unwrap();
        s.synchronize().unwrap();
        Raw {
            x: words(x),
            z: words(z),
            g,
        }
    }
}

/// Same hash and bucket count, and every bucket bitwise equal including coefficient bit patterns.
fn assert_same_buckets<const W: usize>(got: &PauliSum<W>, want: &PauliSum<W>, what: &str) {
    got.assert_invariants();
    assert!(got.hash().same_rows_as(want.hash()), "{what}: hash rows");
    assert_eq!(got.hash().bits(), want.hash().bits(), "{what}: bucket bits");
    assert_eq!(got.num_qubits(), want.num_qubits(), "{what}: num_qubits");
    for b in 0..want.num_buckets() {
        let (gx, gz, gc) = got.bucket(b);
        let (wx, wz, wc) = want.bucket(b);
        assert_eq!(gx, wx, "{what}: bucket {b} x");
        assert_eq!(gz, wz, "{what}: bucket {b} z");
        let bits = |c: &[Complex64]| -> Vec<(u64, u64)> {
            c.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
        };
        assert_eq!(bits(gc), bits(wc), "{what}: bucket {b} coeff bits");
    }
    assert_same_terms(got, want, what);
}

fn round_trip<const W: usize>(sum: &PauliSum<W>, what: &str) {
    let dev = GpuSum::from_host(sum, 0).expect("upload");
    assert_eq!(dev.len(), sum.len(), "{what}: len");
    assert_eq!(dev.bits(), sum.hash().bits(), "{what}: bits");
    dev.assert_invariants_device()
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    assert_same_buckets(&dev.to_host().expect("download"), sum, what);
}

fn one_term<const W: usize>(num_qubits: usize) -> PauliSum<W> {
    let mut accumulator = BuildAccumulator::<W>::new(num_qubits);
    accumulator.add_term(
        PauliString::<W>::y(num_qubits - 1),
        Phase::ONE,
        Complex64::new(0.5, -0.25),
    );
    accumulator.finalize()
}

fn single_bucket<const W: usize>(num_qubits: usize) -> PauliSum<W> {
    let sum = rand_sum::<W>(5000, num_qubits, 0x5B).with_hash(Gf2Hash::new(
        num_qubits,
        0,
        DEFAULT_HASH_SEED,
    ));
    assert_eq!(sum.num_buckets(), 1);
    sum
}

fn round_trips<const W: usize>() {
    let nq = 64 * W;
    round_trip(&rand_sum::<W>(10_000, nq, 0x1111), "rand 1e4");
    round_trip(&low_weight_sum::<W>(20_000, nq, 2, 0x5151), "low weight");
    round_trip(&PauliSum::<W>::empty(nq), "empty");
    round_trip(&one_term::<W>(nq), "one term");
    round_trip(&single_bucket::<W>(nq), "single bucket");
}

#[test]
fn round_trip_is_bitwise() {
    crate::require_cuda!();
    round_trips::<1>();
    round_trips::<2>();
}

#[test]
fn round_trip_is_bitwise_at_w4_and_one_million_terms() {
    crate::require_cuda!();
    round_trip(&rand_sum::<4>(10_000, 250, 0x4444), "W=4 rand 1e4");
    round_trip(&rand_sum::<2>(1_000_000, 128, 0xCAFE), "rand 1e6");
}

#[test]
fn to_host_twice_reuses_the_staging_and_agrees() {
    crate::require_cuda!();
    let sum = rand_sum::<1>(3000, 64, 0x77);
    let dev = GpuSum::from_host(&sum, 0).expect("upload");
    assert_same_buckets(&dev.to_host().unwrap(), &sum, "first");
    assert_same_buckets(&dev.to_host().unwrap(), &sum, "second");
}

fn check_fingerprints<const W: usize>(sum: &PauliSum<W>, mask: u64, dev: &GpuSum<W>) {
    let raw = dev.download_raw();
    assert_eq!(raw.g.len(), sum.len());
    let fingerprints = FingerprintRows::<W>::new(sum.hash().seed());
    for i in 0..raw.g.len() {
        assert_eq!(
            raw.g[i],
            fingerprints.fingerprint(&raw.x[i], &raw.z[i]) & mask,
            "term {i}"
        );
    }
}

fn device_fingerprints<const W: usize>() {
    let nq = 64 * W;
    for sum in [
        rand_sum::<W>(100_000, nq, 0xF00D),
        low_weight_sum::<W>(100_000, nq, 2, 0x6262),
    ] {
        let dev = GpuSum::from_host(&sum, 0).expect("upload");
        check_fingerprints(&sum, !0, &dev);
    }
}

#[test]
fn device_fingerprint_matches_host() {
    crate::require_cuda!();
    device_fingerprints::<1>();
    device_fingerprints::<2>();
}

fn refine_against_host<const W: usize>(sum: PauliSum<W>, what: &str) {
    let mut host = sum.clone();
    let mut dev = GpuSum::from_host(&sum, 0).expect("upload");
    dev.refine().expect("refine");
    host.refine();
    dev.assert_invariants_device()
        .unwrap_or_else(|e| panic!("{what} refine: {e}"));
    assert_same_buckets(&dev.to_host().unwrap(), &host, &format!("{what} refine"));
    let target = dev.bits() + 4;
    dev.refine_to(target).expect("refine_to");
    for _ in 0..4 {
        host.refine();
    }
    assert_eq!(dev.bits(), target);
    dev.assert_invariants_device()
        .unwrap_or_else(|e| panic!("{what} refine_to: {e}"));
    assert_same_buckets(&dev.to_host().unwrap(), &host, &format!("{what} refine_to"));
    let target = dev.bits() + 6;
    dev.refine_to(target).expect("refine_to across two passes");
    for _ in 0..6 {
        host.refine();
    }
    dev.assert_invariants_device()
        .unwrap_or_else(|e| panic!("{what} two-pass refine_to: {e}"));
    assert_same_buckets(
        &dev.to_host().unwrap(),
        &host,
        &format!("{what} two passes"),
    );
}

fn refines<const W: usize>() {
    let nq = 64 * W;
    refine_against_host(rand_sum::<W>(100_000, nq, 0x31), "rand");
    refine_against_host(low_weight_sum::<W>(50_000, nq, 2, 0x32), "low weight");
    refine_against_host(single_bucket::<W>(nq), "single bucket");
    refine_against_host(PauliSum::<W>::empty(nq), "empty");
}

#[test]
fn refine_matches_host() {
    crate::require_cuda!();
    refines::<1>();
    refines::<2>();
}

#[test]
fn refine_to_beyond_b_max_bits_is_unsupported() {
    crate::require_cuda!();
    let sum = rand_sum::<1>(1000, 64, 0x9);
    let mut dev = GpuSum::from_host(&sum, 0).expect("upload");
    let bits = dev.bits();
    assert!(matches!(
        dev.refine_to(B_MAX_BITS + 1),
        Err(GpuError::Unsupported(_))
    ));
    assert_eq!(dev.bits(), bits);
    dev.refine_to(bits).expect("a no-op");
    assert_same_buckets(&dev.to_host().unwrap(), &sum, "unchanged");
}

#[test]
fn eight_bit_fingerprints_still_round_trip_and_refine() {
    crate::require_cuda!();
    let options = ["-DFP_BITS=8".to_string()];
    let sum = rand_sum::<2>(50_000, 128, 0x88);
    let mut dev = GpuSum::from_host_with_options(&sum, 0, &options).expect("upload");
    check_fingerprints(&sum, 0xFF, &dev);
    dev.assert_invariants_device().expect("invariants");
    assert_same_buckets(&dev.to_host().unwrap(), &sum, "FP_BITS=8 round trip");
    let mut host = sum.clone();
    dev.refine_to(dev.bits() + 3).expect("refine_to");
    for _ in 0..3 {
        host.refine();
    }
    dev.assert_invariants_device()
        .expect("invariants after refine");
    assert_same_buckets(&dev.to_host().unwrap(), &host, "FP_BITS=8 refine");
}

/// Device order within a bucket is free, so a download must restore the host's lex order itself.
#[test]
fn to_host_re_sorts_a_bucket_out_of_lex_order() {
    crate::require_cuda!();
    let sum = rand_sum::<2>(20_000, 128, 0x43);
    let b0 = (0..sum.num_buckets())
        .find(|&b| sum.bucket_len(b) >= 3)
        .unwrap();
    let r0: usize = (0..b0).map(|b| sum.bucket_len(b)).sum();
    let (bx, bz, bc) = sum.bucket(b0);
    let l = bx.len();
    let rev = |col: &[[u64; 2]]| -> Vec<u64> { col.iter().rev().flat_map(|k| *k).collect() };
    let c: Vec<f64> = bc.iter().rev().flat_map(|c| [c.re, c.im]).collect();
    let mut dev = GpuSum::from_host(&sum, 0).expect("upload");
    let s = dev.stream.clone();
    s.memcpy_htod(&rev(bx), &mut dev.columns.x.slice_mut(2 * r0..2 * (r0 + l)))
        .unwrap();
    s.memcpy_htod(&rev(bz), &mut dev.columns.z.slice_mut(2 * r0..2 * (r0 + l)))
        .unwrap();
    s.memcpy_htod(&c, &mut dev.columns.coeff.slice_mut(2 * r0..2 * (r0 + l)))
        .unwrap();
    let fingerprints = FingerprintRows::<2>::new(sum.hash().seed());
    let g: Vec<u64> = (0..l)
        .rev()
        .map(|i| fingerprints.fingerprint(&bx[i], &bz[i]))
        .collect();
    s.memcpy_htod(&g, &mut dev.columns.g.slice_mut(r0..r0 + l))
        .unwrap();
    dev.assert_invariants_device()
        .expect("order within a bucket is free on the device");
    assert_same_buckets(&dev.to_host().unwrap(), &sum, "reversed bucket");
}

#[test]
fn invariants_kernel_reports_corruption() {
    crate::require_cuda!();
    let sum = rand_sum::<1>(20_000, 64, 0x42);
    let full: Vec<usize> = (0..sum.num_buckets())
        .filter(|&b| sum.bucket_len(b) >= 2)
        .take(2)
        .collect();
    let (b0, b1) = (full[0], full[1]);

    let mut dev = GpuSum::from_host(&sum, 0).expect("upload");
    let (bx, bz, _) = sum.bucket(b0);
    let r0: usize = (0..b0).map(|b| sum.bucket_len(b)).sum();
    dev.stream
        .memcpy_htod(&bx[0], &mut dev.columns.x.slice_mut(r0 + 1..r0 + 2))
        .unwrap();
    dev.stream
        .memcpy_htod(&bz[0], &mut dev.columns.z.slice_mut(r0 + 1..r0 + 2))
        .unwrap();
    let fingerprints = FingerprintRows::<1>::new(sum.hash().seed());
    let g = [fingerprints.fingerprint(&bx[0], &bz[0])];
    dev.stream
        .memcpy_htod(&g, &mut dev.columns.g.slice_mut(r0 + 1..r0 + 2))
        .unwrap();
    let msg = dev.assert_invariants_device().unwrap_err();
    assert!(msg.contains("0 misplaced, 1 duplicate keys"), "{msg}");
    assert!(msg.contains(&format!("bucket {b0})")), "{msg}");

    let mut dev = GpuSum::from_host(&sum, 0).expect("upload");
    let (cx, cz, _) = sum.bucket(b1);
    dev.stream
        .memcpy_htod(&cx[0], &mut dev.columns.x.slice_mut(r0..r0 + 1))
        .unwrap();
    dev.stream
        .memcpy_htod(&cz[0], &mut dev.columns.z.slice_mut(r0..r0 + 1))
        .unwrap();
    let msg = dev.assert_invariants_device().unwrap_err();
    assert!(msg.contains("1 misplaced"), "{msg}");
    assert!(msg.contains("1 stale fingerprints"), "{msg}");

    let mut dev = GpuSum::from_host(&sum, 0).expect("upload");
    dev.columns.len += 1;
    let msg = dev.assert_invariants_device().unwrap_err();
    assert!(msg.contains("bucket lengths sum to"), "{msg}");
}
