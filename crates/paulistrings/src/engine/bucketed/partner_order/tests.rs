use super::*;
use crate::test_support::Xs64;
use num_complex::Complex64;

/// `keys` sorted and deduplicated, XORed with the mask in source order, as the rotation scan emits them.
fn partner_run<const W: usize>(
    mut keys: Vec<([u64; W], [u64; W])>,
    mask: ([u64; W], [u64; W]),
) -> BucketColumns<W> {
    keys.sort_unstable();
    keys.dedup();
    let mut run = BucketColumns::default();
    for (row, (x, z)) in keys.into_iter().enumerate() {
        let x = core::array::from_fn(|w| x[w] ^ mask.0[w]);
        let z = core::array::from_fn(|w| z[w] ^ mask.1[w]);
        run.push(x, z, Complex64::new(row as f64, 0.0));
    }
    run
}

fn rows<const W: usize>(run: &BucketColumns<W>) -> Vec<([u64; W], [u64; W], Complex64)> {
    (0..run.len()).map(|row| (run.x[row], run.z[row], run.coeff[row])).collect()
}

fn sorted_by_key<const W: usize>(run: &BucketColumns<W>) -> Vec<([u64; W], [u64; W], Complex64)> {
    let mut rows = rows(run);
    rows.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    rows
}

/// One x-bit: the two halves of each prefix group swap, `[10, 11, 00, 01]` becoming `[00, 01, 10, 11]`.
#[test]
fn a_single_bit_mask_swaps_sibling_halves() {
    let keys = (0..4u64).map(|k| ([k], [0])).collect();
    let run = partner_run::<1>(keys, ([0b10], [0]));
    assert_eq!(run.x, vec![[0b10], [0b11], [0b00], [0b01]]);
    let mut out = BucketColumns::default();
    PartnerOrder::new(&[0b10], &[0]).order(run.view(), &mut out, &mut Vec::new());
    assert_eq!(out.x, vec![[0b00], [0b01], [0b10], [0b11]]);
    assert_eq!(out.coeff, [2.0, 3.0, 0.0, 1.0].map(Complex64::from));
}

/// The output is the run sorted by key, its coefficients travelling with their keys, for every mask of up to four bits.
fn assert_orders_like_a_sort<const W: usize>(seed: u64, bits_in_mask: usize, key_bits: u32) {
    let mut rng = Xs64::new(seed);
    let narrow = |v: u64| if key_bits >= 64 { v } else { v & ((1u64 << key_bits) - 1) };
    for trial in 0..200 {
        let mut mask = ([0u64; W], [0u64; W]);
        for _ in 0..bits_in_mask {
            let word = (rng.next_u64() % W as u64) as usize;
            let bit = rng.next_u64() % u64::from(key_bits.min(64));
            if rng.next_u64() & 1 == 0 {
                mask.0[word] |= 1 << bit;
            } else {
                mask.1[word] |= 1 << bit;
            }
        }
        let len = (rng.next_u64() % 300) as usize;
        let keys = (0..len)
            .map(|_| {
                (
                    core::array::from_fn(|_| narrow(rng.next_u64())),
                    core::array::from_fn(|_| narrow(rng.next_u64())),
                )
            })
            .collect();
        let run = partner_run::<W>(keys, mask);
        let mut out = BucketColumns::default();
        PartnerOrder::new(&mask.0, &mask.1).order(run.view(), &mut out, &mut Vec::new());
        assert_eq!(rows(&out), sorted_by_key(&run), "seed {seed} trial {trial} mask {mask:?}");
    }
}

#[test]
fn orders_like_a_sort() {
    for bits in 0..=4 {
        // Narrow keys make long shared prefixes and large groups; wide keys make singletons.
        assert_orders_like_a_sort::<1>(0x5EED ^ bits as u64, bits, 6);
        assert_orders_like_a_sort::<1>(0xFACE ^ bits as u64, bits, 64);
        assert_orders_like_a_sort::<2>(0xD00D ^ bits as u64, bits, 4);
        assert_orders_like_a_sort::<2>(0xBEAD ^ bits as u64, bits, 64);
    }
}

/// The empty run and a mask on the most significant bit of the last key word.
#[test]
fn edge_bits_and_empty_runs() {
    let mut out = BucketColumns::<2>::default();
    PartnerOrder::new(&[0, 0], &[0, 1 << 63]).order(
        BucketColumns::default().view(),
        &mut out,
        &mut Vec::new(),
    );
    assert_eq!(out.len(), 0);
    let keys = (0..8u64).map(|k| ([k, 0], [0, k << 61])).collect();
    let run = partner_run::<2>(keys, ([1 << 63, 0], [0, 1 << 63]));
    PartnerOrder::new(&[1 << 63, 0], &[0, 1 << 63]).order(run.view(), &mut out, &mut Vec::new());
    assert_eq!(rows(&out), sorted_by_key(&run));
}
