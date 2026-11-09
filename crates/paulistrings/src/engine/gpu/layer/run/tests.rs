use super::*;
use crate::channel::rotation::PauliRotation;
use crate::channel::{Channel, Prepare, PtmChannel};
use crate::pauli_string::PauliString;
use crate::pauli_sum::storage::DEFAULT_HASH_SEED;
use num_complex::Complex64;

fn local<const W: usize>(channel: &dyn Channel<W>, hash: &Gf2Hash<W>) -> LocalPtm<W> {
    match channel.prepare(hash, true).expect("prepared") {
        Prepared::Local(ptm) => ptm,
        Prepared::Rotation(_) => panic!("a one-qubit channel tabulates"),
    }
}

/// `a[s → u]` of a one-qubit table, zero where no entry takes `s` to `u`.
fn amplitude<const W: usize>(ptm: &LocalPtm<W>, s: usize, u: usize) -> Complex64 {
    ptm.deltas()
        .iter()
        .find(|entry| entry.local_delta as usize == s ^ u)
        .map_or(Complex64::ZERO, |entry| entry.amplitude[s])
}

/// The joined table takes `(s, t)` to `(u, v)` with `a[s → u]·b[t → v]`, the lower qubit's table first whichever order they come in, every entry under the hash of its mask.
fn check_tensor<const W: usize>(high_qubit: usize, low_qubit: usize, nq: usize) {
    let hash = Gf2Hash::<W>::new(nq, 7, DEFAULT_HASH_SEED);
    let high = local(&PauliRotation::new(PauliString::<W>::x(high_qubit), 0.7), &hash);
    let low = local(&PtmChannel::amplitude_damping(low_qubit, 0.2), &hash);
    let joined = high.tensor(&low, &hash).expect("one-qubit tables on different qubits");
    assert_eq!(joined.qubits().collect::<Vec<_>>(), [low_qubit, high_qubit]);
    let deltas: Vec<u8> = joined.deltas().iter().map(|entry| entry.local_delta).collect();
    assert!(deltas.windows(2).all(|pair| pair[0] < pair[1]), "ascending: {deltas:?}");
    for entry in joined.deltas() {
        assert_eq!(entry.bucket_delta, hash.bucket_of(&entry.mask_x, &entry.mask_z));
    }
    for s in 0..16 {
        for u in 0..16 {
            let want = amplitude(&low, s & 3, u & 3) * amplitude(&high, s >> 2, u >> 2);
            let got = amplitude(&joined, s, u);
            assert!((got - want).norm() < 1e-15, "{s} -> {u}: {got} vs {want}");
        }
    }
    assert!(high.tensor(&high, &hash).is_none(), "one qubit twice");
    let two = local(&PtmChannel::cnot(low_qubit, high_qubit), &hash);
    assert!(two.tensor(&low, &hash).is_none(), "a two-qubit table");
}

#[test]
fn tensor_joins_one_qubit_tables_on_different_qubits() {
    check_tensor::<1>(5, 2, 8);
    check_tensor::<2>(70, 3, 100);
}
