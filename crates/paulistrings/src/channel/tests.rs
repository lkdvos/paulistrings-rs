use super::*;
use crate::test_support::alloc_bufs;

#[test]
fn support_mask_packs_cross_word_qubits() {
    // Qubit 70 at W=2 lands in word 1, bit 6.
    let mask: [u64; 2] = support_mask(&[5, 70]);
    assert_eq!(mask[0], 1u64 << 5);
    assert_eq!(mask[1], 1u64 << 6);
}

#[test]
fn support_mask_is_order_and_duplicate_insensitive() {
    let a: [u64; 1] = support_mask(&[3, 1, 3]);
    let b: [u64; 1] = support_mask(&[1, 3]);
    assert_eq!(a, b);
}

/// The default `debug_name` must survive erasure to `dyn Channel<W>` — that is how the engine sees every channel — and must trim both the module path and any generic arguments.
#[test]
fn debug_name_through_dyn_trims_path_and_generics() {
    use crate::pauli_string::PauliString;

    let channels: Vec<(Box<dyn Channel<1>>, &str)> = vec![
        (Box::new(Clifford1Q::h(0)), "Clifford1Q"),
        (Box::new(Clifford2Q::cnot(0, 1)), "Clifford2Q"),
        (
            // Generic in `W`, so `type_name` carries a `<1>` to trim.
            Box::new(PauliRotation::new(PauliString::<1>::z(0), 0.3)),
            "PauliRotation",
        ),
        (
            Box::new(Depolarizing {
                support: [0],
                p: 0.1,
            }),
            "Depolarizing",
        ),
        (
            Box::new(Dephasing {
                support: [0],
                p: 0.1,
            }),
            "Dephasing",
        ),
        (
            Box::new(PauliChannel {
                support: [0],
                px: 0.1,
                py: 0.1,
                pz: 0.1,
            }),
            "PauliChannel",
        ),
        (
            Box::new(Depolarizing2Q {
                support: [0, 1],
                p: 0.1,
            }),
            "Depolarizing2Q",
        ),
        (
            Box::new(AmplitudeDamping {
                support: [0],
                gamma: 0.1,
            }),
            "AmplitudeDamping",
        ),
        (Box::new(IdentityChannel), "IdentityChannel"),
    ];
    for (channel, expected) in &channels {
        assert_eq!(channel.debug_name(), *expected);
    }
}

#[test]
fn support_mask_of_empty_is_zero() {
    let mask: [u64; 2] = support_mask(&[]);
    assert_eq!(mask, [0, 0]);
}

#[test]
fn push_writes_at_cursor_w1() {
    let (mut x, mut z, mut c, mut len) = alloc_bufs::<1>(4);
    {
        let mut buffer = OutputBuffer::<1> {
            x: &mut x,
            z: &mut z,
            coeff: &mut c,
            len: &mut len,
        };
        buffer.push([0xAA], [0xBB], Complex64::new(1.0, 2.0));
        buffer.push([0xCC], [0xDD], Complex64::new(3.0, 4.0));
        assert_eq!(*buffer.len, 2);
    }
    assert_eq!(x[0], [0xAA]);
    assert_eq!(z[0], [0xBB]);
    assert_eq!(c[0], Complex64::new(1.0, 2.0));
    assert_eq!(x[1], [0xCC]);
    assert_eq!(z[1], [0xDD]);
    assert_eq!(c[1], Complex64::new(3.0, 4.0));
    // remaining slots untouched
    assert_eq!(x[2], [0]);
    assert_eq!(x[3], [0]);
    assert_eq!(c[3], Complex64::new(0.0, 0.0));
}

#[test]
fn push_writes_at_cursor_w2() {
    let (mut x, mut z, mut c, mut len) = alloc_bufs::<2>(3);
    {
        let mut buffer = OutputBuffer::<2> {
            x: &mut x,
            z: &mut z,
            coeff: &mut c,
            len: &mut len,
        };
        buffer.push([0x11, 0x22], [0x33, 0x44], Complex64::new(5.0, 6.0));
        assert_eq!(*buffer.len, 1);
    }
    assert_eq!(x[0], [0x11, 0x22]);
    assert_eq!(z[0], [0x33, 0x44]);
    assert_eq!(c[0], Complex64::new(5.0, 6.0));
}

#[test]
#[should_panic]
fn push_panics_when_full() {
    let (mut x, mut z, mut c, mut len) = alloc_bufs::<1>(2);
    let mut buffer = OutputBuffer::<1> {
        x: &mut x,
        z: &mut z,
        coeff: &mut c,
        len: &mut len,
    };
    buffer.push([0; 1], [0; 1], Complex64::new(1.0, 0.0));
    buffer.push([0; 1], [0; 1], Complex64::new(1.0, 0.0));
    buffer.push([0; 1], [0; 1], Complex64::new(1.0, 0.0));
}

#[test]
fn clear_resets_cursor() {
    let (mut x, mut z, mut c, mut len) = alloc_bufs::<1>(4);
    {
        let mut buffer = OutputBuffer::<1> {
            x: &mut x,
            z: &mut z,
            coeff: &mut c,
            len: &mut len,
        };
        buffer.push([0xAA], [0xBB], Complex64::new(1.0, 0.0));
        buffer.push([0xCC], [0xDD], Complex64::new(2.0, 0.0));
        assert_eq!(*buffer.len, 2);
        buffer.clear();
        assert_eq!(*buffer.len, 0);
        buffer.push([0xEE], [0xFF], Complex64::new(3.0, 0.0));
        assert_eq!(*buffer.len, 1);
    }
    // The post-clear push lands at slot 0, overwriting the prior contents.
    assert_eq!(x[0], [0xEE]);
    assert_eq!(z[0], [0xFF]);
    assert_eq!(c[0], Complex64::new(3.0, 0.0));
    // Slot 1 was written before the clear and is left as-is.
    assert_eq!(x[1], [0xCC]);
    assert_eq!(z[1], [0xDD]);
}

#[test]
fn reuse_does_not_grow_backing_vecs() {
    let cap = 4;
    let mut x: Vec<[u64; 1]> = vec![[0u64; 1]; cap];
    let mut z: Vec<[u64; 1]> = vec![[0u64; 1]; cap];
    let mut c: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); cap];
    assert_eq!(x.capacity(), cap);
    assert_eq!(z.capacity(), cap);
    assert_eq!(c.capacity(), cap);
    let mut len: usize;
    for i in 0..100u64 {
        len = 0;
        let mut buffer = OutputBuffer::<1> {
            x: &mut x,
            z: &mut z,
            coeff: &mut c,
            len: &mut len,
        };
        buffer.push([i], [0], Complex64::new(i as f64, 0.0));
        buffer.push([i + 1], [0], Complex64::new((i + 1) as f64, 0.0));
    }
    assert_eq!(x.capacity(), cap);
    assert_eq!(z.capacity(), cap);
    assert_eq!(c.capacity(), cap);
}
