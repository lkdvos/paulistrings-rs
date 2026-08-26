//! Sort-merge propagation pipeline: scan → bucket → merge. See §5.

#![allow(unused)]

use num_complex::Complex64;
use rayon::prelude::*;

use crate::channel::{Channel, OutputBuffer};
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;

/// Empirical threshold below which a hashmap-based fast path beats sort-merge
/// (§8.3). Subject to benchmarking.
pub const SMALL_SUM_THRESHOLD: usize = 4096;

/// Apply a single channel to a `PauliSum`, producing the next layer.
///
/// Implements the three-phase pipeline:
///   1. **Scan** — `n_in × MAX_FANOUT` data-parallel channel applications.
///   2. **Sort** — stable lex sort of the populated prefix (Phase 6
///      placeholder for the bucket-scatter optimization deferred to
///      Phase 11; see slice plan §6.2).
///   3. **Merge** — segmented reduction; `keep_term` integration arrives
///      in Phase 7.
pub fn apply_layer<const W: usize, C, T>(
    input: &PauliSum<W>,
    channel: &C,
    policy: &T,
) -> PauliSum<W>
where
    C: Channel<W> + ?Sized,
    T: TruncationPolicy<W> + ?Sized,
{
    apply_layer_inner(input, channel, policy, /* adjoint = */ false)
}

/// Apply a channel's adjoint to a `PauliSum`. Used by `propagate` in
/// `Direction::Heisenberg` mode; structurally identical to `apply_layer`
/// but routes through `Channel::apply_adjoint`.
pub fn apply_layer_adjoint<const W: usize, C, T>(
    input: &PauliSum<W>,
    channel: &C,
    policy: &T,
) -> PauliSum<W>
where
    C: Channel<W> + ?Sized,
    T: TruncationPolicy<W> + ?Sized,
{
    apply_layer_inner(input, channel, policy, /* adjoint = */ true)
}

fn apply_layer_inner<const W: usize, C, T>(
    input: &PauliSum<W>,
    channel: &C,
    policy: &T,
    adjoint: bool,
) -> PauliSum<W>
where
    C: Channel<W> + ?Sized,
    T: TruncationPolicy<W> + ?Sized,
{
    let n_in = input.len();
    let mf = channel.max_fanout();
    let cap = n_in * mf;
    let mut out_x: Vec<[u64; W]> = vec![[0u64; W]; cap];
    let mut out_z: Vec<[u64; W]> = vec![[0u64; W]; cap];
    let mut out_coeff: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); cap];
    let len = if adjoint {
        scan_phase_adjoint(input, channel, &mut out_x, &mut out_z, &mut out_coeff)
    } else {
        scan_phase(input, channel, &mut out_x, &mut out_z, &mut out_coeff)
    };
    sort_phase(&mut out_x, &mut out_z, &mut out_coeff, len);
    let result = merge_phase::<W, T>(&out_x, &out_z, &out_coeff, len, input.num_qubits(), policy);
    #[cfg(debug_assertions)]
    result.assert_invariants();
    result
}

/// Phase 1 of `apply_layer`: walk the input `PauliSum` and write each input
/// term's outputs into a flat scratch buffer.
///
/// The caller pre-sizes the three slices to at least
/// `input.len() * channel.max_fanout()`. Each input `i` is assigned the
/// disjoint slot `[i*mf, (i+1)*mf)` in the buffer; per-input fills run in
/// parallel via `rayon::par_chunks_mut`, after which a sequential compaction
/// pass packs the populated prefixes contiguously. The slot layout depends
/// only on `(i, mf)` — not on the rayon thread count or scheduling — so the
/// final byte layout is deterministic across thread pool sizes (slice 8.1).
///
/// The function returns the actual number of output terms written (which may
/// be less than `n_in * max_fanout` when the channel produces variable
/// per-input fanout, e.g. `PauliRotation`).
///
/// `apply_fn` selects between forward (`Channel::apply`) and Heisenberg
/// (`Channel::apply_adjoint`); the two scan-phase callers differ only in
/// that one method call. It is `Fn + Sync` so the rayon worker threads can
/// share it.
///
/// Output is *not* sorted — that's slice 6.2's job.
fn scan_phase_with<const W: usize, C, F>(
    input: &PauliSum<W>,
    channel: &C,
    out_x: &mut [[u64; W]],
    out_z: &mut [[u64; W]],
    out_coeff: &mut [Complex64],
    apply_fn: F,
) -> usize
where
    C: Channel<W> + ?Sized,
    F: Fn(&C, &[u64; W], &[u64; W], Complex64, &mut OutputBuffer<'_, W>) + Sync,
{
    let mf = channel.max_fanout();
    let n_in = input.len();
    debug_assert!(out_x.len() >= n_in * mf);
    debug_assert_eq!(out_x.len(), out_z.len());
    debug_assert_eq!(out_x.len(), out_coeff.len());
    if n_in == 0 || mf == 0 {
        return 0;
    }
    let cap = n_in * mf;
    let lens: Vec<usize> = out_x[..cap]
        .par_chunks_mut(mf)
        .zip(out_z[..cap].par_chunks_mut(mf))
        .zip(out_coeff[..cap].par_chunks_mut(mf))
        .enumerate()
        .map(|(i, ((sx, sz), sc))| {
            let mut local_len = 0usize;
            let mut buf = OutputBuffer::<W> {
                x: sx,
                z: sz,
                coeff: sc,
                len: &mut local_len,
            };
            apply_fn(channel, &input.x[i], &input.z[i], input.coeff[i], &mut buf);
            local_len
        })
        .collect();
    compact_in_place(out_x, out_z, out_coeff, &lens, mf)
}

/// Pack the per-input populated prefixes `[i*mf, i*mf + lens[i])` into a
/// contiguous prefix `[0..total)` of the three SoA buffers. Source and
/// destination overlap, but the source index is always `>=` the destination,
/// so `slice::copy_within` (memmove semantics) is correct.
fn compact_in_place<const W: usize>(
    out_x: &mut [[u64; W]],
    out_z: &mut [[u64; W]],
    out_coeff: &mut [Complex64],
    lens: &[usize],
    mf: usize,
) -> usize {
    let mut write = 0usize;
    for (i, &len_i) in lens.iter().enumerate() {
        if len_i == 0 {
            continue;
        }
        let src = i * mf;
        debug_assert!(write <= src);
        if write != src {
            out_x.copy_within(src..src + len_i, write);
            out_z.copy_within(src..src + len_i, write);
            out_coeff.copy_within(src..src + len_i, write);
        }
        write += len_i;
    }
    write
}

/// Forward scan: dispatches each input through `Channel::apply`.
pub(crate) fn scan_phase<const W: usize, C: Channel<W> + ?Sized>(
    input: &PauliSum<W>,
    channel: &C,
    out_x: &mut [[u64; W]],
    out_z: &mut [[u64; W]],
    out_coeff: &mut [Complex64],
) -> usize {
    scan_phase_with(
        input,
        channel,
        out_x,
        out_z,
        out_coeff,
        |c, x, z, co, out| c.apply(x, z, co, out),
    )
}

/// Heisenberg scan: dispatches each input through `Channel::apply_adjoint`.
pub(crate) fn scan_phase_adjoint<const W: usize, C: Channel<W> + ?Sized>(
    input: &PauliSum<W>,
    channel: &C,
    out_x: &mut [[u64; W]],
    out_z: &mut [[u64; W]],
    out_coeff: &mut [Complex64],
) -> usize {
    scan_phase_with(
        input,
        channel,
        out_x,
        out_z,
        out_coeff,
        |c, x, z, co, out| c.apply_adjoint(x, z, co, out),
    )
}

/// Phase 2: stably sort the populated prefix `[0..len)` of the scratch
/// buffers by `(x, z)` lex order — the same key the `PauliString` `Ord`
/// impl uses (`x[0]..x[W-1]` then `z[0]..z[W-1]`).
///
/// Stability matters once truncation policies depend on insertion order
/// (Phase 7), and matches the design doc's "within-bucket relative order
/// inherited from input" contract (§5). The implementation is `O(n log n)`
/// instead of the design-doc's `O(n)` bucket scatter; the bucket
/// optimization is deferred to Phase 11 with profile data.
pub(crate) fn sort_phase<const W: usize>(
    out_x: &mut [[u64; W]],
    out_z: &mut [[u64; W]],
    out_coeff: &mut [Complex64],
    len: usize,
) {
    debug_assert!(out_x.len() >= len);
    debug_assert_eq!(out_x.len(), out_z.len());
    debug_assert_eq!(out_x.len(), out_coeff.len());
    if len < 2 {
        return;
    }
    let mut perm: Vec<usize> = (0..len).collect();
    // `[u64; W]`'s built-in `Ord` is lex over array elements, identical to
    // `PauliString::cmp`'s loop body.
    perm.sort_by(|&a, &b| {
        out_x[a]
            .cmp(&out_x[b])
            .then_with(|| out_z[a].cmp(&out_z[b]))
    });
    let new_x: Vec<[u64; W]> = perm.iter().map(|&i| out_x[i]).collect();
    let new_z: Vec<[u64; W]> = perm.iter().map(|&i| out_z[i]).collect();
    let new_c: Vec<Complex64> = perm.iter().map(|&i| out_coeff[i]).collect();
    out_x[..len].copy_from_slice(&new_x);
    out_z[..len].copy_from_slice(&new_z);
    out_coeff[..len].copy_from_slice(&new_c);
}

/// Empirical threshold below which the parallel merge's overhead dominates.
/// Below this, `merge_phase` collapses to a single sequential chunk.
const SMALL_MERGE_THRESHOLD: usize = 1024;

/// SoA triple emitted by a per-chunk merge — the same shape as `PauliSum`'s
/// internal storage, deferred into a `Vec` until concatenation.
type ChunkOutput<const W: usize> = (Vec<[u64; W]>, Vec<[u64; W]>, Vec<Complex64>);

/// Phase 3: segmented reduction over the sorted scratch into a fresh
/// `PauliSum`.
///
/// The input slices are the populated prefix `[0..len)` of the sort_phase
/// output; they must be sorted by `(x, z)` (`debug_assert`-checked). Adjacent
/// runs of equal keys have their coefficients summed; runs whose summed
/// coefficient is exactly `0+0i` are dropped, and `policy.keep_term` is
/// consulted on the *summed* coefficient — terms it rejects are dropped here
/// rather than in a post-pass (slice 7.1).
///
/// The reduction is parallelized via chunked segment-aware merging
/// (slice 8.2 / design doc §9): the populated prefix is partitioned into
/// `rayon::current_num_threads()` chunks whose boundaries are advanced
/// forward to the next run break, so every run is fully contained in exactly
/// one chunk. Each chunk runs the same per-run reduction independently;
/// the boundary "reconciliation pass" is the alignment step itself, so the
/// chunk results are concatenated without further merging.
pub(crate) fn merge_phase<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    sorted_x: &[[u64; W]],
    sorted_z: &[[u64; W]],
    sorted_coeff: &[Complex64],
    len: usize,
    num_qubits: usize,
    policy: &T,
) -> PauliSum<W> {
    let nchunks = if len < SMALL_MERGE_THRESHOLD {
        1
    } else {
        rayon::current_num_threads().max(1)
    };
    merge_phase_with_nchunks::<W, T>(
        sorted_x,
        sorted_z,
        sorted_coeff,
        len,
        num_qubits,
        policy,
        nchunks,
    )
}

/// `merge_phase` with an explicit chunk count. Public to the crate so tests
/// can pin `nchunks` and force runs to straddle boundaries; `merge_phase`
/// itself derives `nchunks` from `rayon::current_num_threads()`.
pub(crate) fn merge_phase_with_nchunks<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    sorted_x: &[[u64; W]],
    sorted_z: &[[u64; W]],
    sorted_coeff: &[Complex64],
    len: usize,
    num_qubits: usize,
    policy: &T,
    nchunks: usize,
) -> PauliSum<W> {
    debug_assert!(sorted_x.len() >= len);
    debug_assert_eq!(sorted_x.len(), sorted_z.len());
    debug_assert_eq!(sorted_x.len(), sorted_coeff.len());
    if len == 0 {
        return PauliSum::<W> {
            x: Vec::new(),
            z: Vec::new(),
            coeff: Vec::new(),
            num_qubits,
        };
    }
    let bounds = align_chunk_boundaries(sorted_x, sorted_z, len, nchunks.max(1));
    let chunk_results: Vec<ChunkOutput<W>> = bounds
        .par_iter()
        .map(|&(start, end)| {
            merge_chunk::<W, T>(sorted_x, sorted_z, sorted_coeff, start, end, policy)
        })
        .collect();
    let total: usize = chunk_results.iter().map(|(cx, _, _)| cx.len()).sum();
    let mut x = Vec::with_capacity(total);
    let mut z = Vec::with_capacity(total);
    let mut coeff = Vec::with_capacity(total);
    for (cx, cz, cc) in chunk_results {
        x.extend(cx);
        z.extend(cz);
        coeff.extend(cc);
    }
    PauliSum::<W> {
        x,
        z,
        coeff,
        num_qubits,
    }
}

/// Run the segmented reduction on the sub-range `[start..end)` of the sorted
/// scratch. Caller must ensure runs are not split: `(sorted_x[start-1],
/// sorted_z[start-1]) != (sorted_x[start], sorted_z[start])` whenever
/// `start > 0`, and similarly at `end`. With that invariant, the chunk's
/// reduction is identical to the sequential merge on the same sub-range —
/// `keep_term` and the zero-drop check operate on fully-summed coefficients.
fn merge_chunk<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    sorted_x: &[[u64; W]],
    sorted_z: &[[u64; W]],
    sorted_coeff: &[Complex64],
    start: usize,
    end: usize,
    policy: &T,
) -> ChunkOutput<W> {
    let mut x: Vec<[u64; W]> = Vec::new();
    let mut z: Vec<[u64; W]> = Vec::new();
    let mut coeff: Vec<Complex64> = Vec::new();
    merge_into::<W, T>(
        sorted_x,
        sorted_z,
        sorted_coeff,
        start,
        end,
        &mut x,
        &mut z,
        &mut coeff,
        policy,
    );
    (x, z, coeff)
}

/// The segmented reduction itself, appending into caller-owned columns.
///
/// Extracted from `merge_chunk` so the bucketed engine can reuse it verbatim:
/// per-bucket, this writes straight into the destination bucket's columns
/// instead of growing three un-preallocated `Vec`s and copying again
/// (v0.2 §6 step 4). `merge_chunk` now delegates here, so there is exactly one
/// implementation of the reduction, zero-drop and `keep_term` semantics — the
/// existing merge tests cover both callers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn merge_into<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    sorted_x: &[[u64; W]],
    sorted_z: &[[u64; W]],
    sorted_coeff: &[Complex64],
    start: usize,
    end: usize,
    dst_x: &mut Vec<[u64; W]>,
    dst_z: &mut Vec<[u64; W]>,
    dst_coeff: &mut Vec<Complex64>,
    policy: &T,
) {
    let zero = Complex64::new(0.0, 0.0);
    let mut i = start;
    while i < end {
        let key_x = sorted_x[i];
        let key_z = sorted_z[i];
        let mut acc = sorted_coeff[i];
        let mut j = i + 1;
        while j < end && sorted_x[j] == key_x && sorted_z[j] == key_z {
            acc += sorted_coeff[j];
            j += 1;
        }
        debug_assert!(
            i == 0 || (sorted_x[i - 1], sorted_z[i - 1]) <= (key_x, key_z),
            "merge_into: scratch is not sorted at index {}",
            i,
        );
        if acc != zero && policy.keep_term(&key_x, &key_z, acc) {
            dst_x.push(key_x);
            dst_z.push(key_z);
            dst_coeff.push(acc);
        }
        i = j;
    }
}

/// Partition `[0..len)` into `nchunks` non-empty sub-ranges whose interior
/// boundaries land at run breaks. The "natural" boundary `len * k / nchunks`
/// is advanced forward (or to `len`) until the keys at `t-1` and `t` differ.
/// Boundaries that collapse onto each other are deduped, so the returned
/// vector may contain fewer than `nchunks` chunks.
fn align_chunk_boundaries<const W: usize>(
    sorted_x: &[[u64; W]],
    sorted_z: &[[u64; W]],
    len: usize,
    nchunks: usize,
) -> Vec<(usize, usize)> {
    if len == 0 {
        return Vec::new();
    }
    if nchunks <= 1 {
        return vec![(0, len)];
    }
    let mut bounds: Vec<usize> = Vec::with_capacity(nchunks + 1);
    bounds.push(0);
    for k in 1..nchunks {
        let mut t = (len * k) / nchunks;
        // Advance to the start of a fresh run (or to `len`).
        while t > 0 && t < len && sorted_x[t] == sorted_x[t - 1] && sorted_z[t] == sorted_z[t - 1] {
            t += 1;
        }
        bounds.push(t);
    }
    bounds.push(len);
    bounds.dedup();
    bounds.windows(2).map(|w| (w[0], w[1])).collect()
}

#[cfg(all(test, debug_assertions))]
mod tests {
    use super::*;
    use crate::channel::{Clifford1Q, IdentityChannel, PauliRotation};
    use crate::pauli_string::PauliString;
    use crate::truncation::CoefficientThreshold;

    const TOL: f64 = 1e-12;

    #[allow(clippy::type_complexity)]
    fn alloc_bufs<const W: usize>(n: usize) -> (Vec<[u64; W]>, Vec<[u64; W]>, Vec<Complex64>) {
        (
            vec![[0u64; W]; n],
            vec![[0u64; W]; n],
            vec![Complex64::new(0.0, 0.0); n],
        )
    }

    fn approx_eq(a: Complex64, b: Complex64, tol: f64) -> bool {
        (a - b).norm() <= tol
    }

    /// `IdentityChannel` writes input through unchanged, in input order.
    #[test]
    fn scan_identity_passes_through_w1() {
        let input = PauliSum::<1>::from_strings(&[
            ("X", Complex64::new(1.0, 0.0)),
            ("Z", Complex64::new(2.0, 0.0)),
        ]);
        let id = IdentityChannel::new();
        let cap = input.len() * <IdentityChannel as Channel<1>>::max_fanout(&id);
        let (mut bx, mut bz, mut bc) = alloc_bufs::<1>(cap);
        let total = scan_phase(&input, &id, &mut bx, &mut bz, &mut bc);
        assert_eq!(total, 2);
        for i in 0..total {
            assert_eq!(bx[i], input.x()[i]);
            assert_eq!(bz[i], input.z()[i]);
            assert_eq!(bc[i], input.coeff()[i]);
        }
    }

    /// `H` conjugates `Z → X` and `X → Z` (with phase +1), preserving coeffs.
    /// Output is in input order; sort happens in slice 6.2.
    #[test]
    fn scan_clifford1q_h_w1() {
        let input = PauliSum::<1>::from_strings(&[
            ("Z", Complex64::new(3.0, 0.0)),
            ("X", Complex64::new(5.0, 0.0)),
        ]);
        let h = Clifford1Q::h(0);
        let cap = input.len() * <Clifford1Q as Channel<1>>::max_fanout(&h);
        let (mut bx, mut bz, mut bc) = alloc_bufs::<1>(cap);
        let total = scan_phase(&input, &h, &mut bx, &mut bz, &mut bc);
        assert_eq!(total, 2);
        // Input order: from_strings sorts lex by (x, z) — Z (x=0,z=1) sorts
        // before X (x=1,z=0). So scan output[0] = H·Z = X, output[1] = H·X = Z.
        assert_eq!(bx[0], PauliString::<1>::x(0).x);
        assert_eq!(bz[0], PauliString::<1>::x(0).z);
        assert_eq!(bc[0], Complex64::new(3.0, 0.0));
        assert_eq!(bx[1], PauliString::<1>::z(0).x);
        assert_eq!(bz[1], PauliString::<1>::z(0).z);
        assert_eq!(bc[1], Complex64::new(5.0, 0.0));
    }

    /// `PauliRotation` has `MAX_FANOUT = 2` but emits a single term when the
    /// input commutes with the generator. The scan packs outputs contiguously
    /// — no gaps left by a 1-output input.
    #[test]
    fn scan_pauli_rotation_packs_variable_fanout() {
        // Rotation around Z. Input "X" anticommutes (fanout 2); "Z" commutes
        // (fanout 1). Total = 3 outputs from 2 inputs.
        let p = PauliString::<1>::z(0);
        let theta = std::f64::consts::FRAC_PI_3;
        let rot = PauliRotation::new(p, theta);
        let input = PauliSum::<1>::from_strings(&[
            ("X", Complex64::new(1.0, 0.0)),
            ("Z", Complex64::new(2.0, 0.0)),
        ]);
        let cap = input.len() * <PauliRotation<1> as Channel<1>>::max_fanout(&rot);
        let (mut bx, mut bz, mut bc) = alloc_bufs::<1>(cap);
        let total = scan_phase(&input, &rot, &mut bx, &mut bz, &mut bc);
        assert_eq!(total, 3);
        // from_strings sorts: Z (x=0,z=1) < X (x=1,z=0). So input[0] = Z,
        // input[1] = X.
        // Output[0]: rot(Z) = Z (commutes, fanout 1) with coeff 2.
        assert_eq!(bx[0], PauliString::<1>::z(0).x);
        assert_eq!(bz[0], PauliString::<1>::z(0).z);
        assert!(approx_eq(bc[0], Complex64::new(2.0, 0.0), TOL));
        // Output[1]: cos·X with coeff 1.
        assert_eq!(bx[1], PauliString::<1>::x(0).x);
        assert_eq!(bz[1], PauliString::<1>::x(0).z);
        assert!(approx_eq(bc[1], Complex64::new(theta.cos(), 0.0), TOL));
        // Output[2]: sin·Y (X·Z = -iY, Phase::I + 3 = 0, so coeff = sin·1 = sin·1).
        // Working: input.coeff=1, Phase::I + mul_phase=Phase::I + 3 = 0 ⇒ Phase::ONE.
        // So bc[2] = 1·sin(θ).
        assert_eq!(bx[2], PauliString::<1>::y(0).x);
        assert_eq!(bz[2], PauliString::<1>::y(0).z);
        assert!(approx_eq(bc[2], Complex64::new(theta.sin(), 0.0), TOL));
    }

    /// Multi-word: input on qubit 64 (word 1), `H` flips X↔Z within word 1.
    #[test]
    fn scan_w2_word_boundary() {
        let input = PauliSum::<2> {
            x: vec![[0u64, 1u64]], // X on qubit 64
            z: vec![[0u64; 2]],
            coeff: vec![Complex64::new(1.5, 0.0)],
            num_qubits: 65,
        };
        let h = Clifford1Q::h(64);
        let cap = input.len() * <Clifford1Q as Channel<2>>::max_fanout(&h);
        let (mut bx, mut bz, mut bc) = alloc_bufs::<2>(cap);
        let total = scan_phase(&input, &h, &mut bx, &mut bz, &mut bc);
        assert_eq!(total, 1);
        // H·X = Z, on qubit 64 (word 1, bit 0).
        assert_eq!(bx[0], [0u64, 0u64]);
        assert_eq!(bz[0], [0u64, 1u64]);
        assert_eq!(bc[0], Complex64::new(1.5, 0.0));
    }

    /// Empty input → zero outputs; doesn't read the buffer.
    #[test]
    fn scan_empty_input() {
        let input = PauliSum::<1>::empty(4);
        let id = IdentityChannel::new();
        let mut bx: Vec<[u64; 1]> = vec![];
        let mut bz: Vec<[u64; 1]> = vec![];
        let mut bc: Vec<Complex64> = vec![];
        let total = scan_phase(&input, &id, &mut bx, &mut bz, &mut bc);
        assert_eq!(total, 0);
    }

    /// Slice 8.1: same input under different rayon thread-pool sizes must
    /// produce byte-identical output through `apply_layer`. Per-input slot
    /// bounds depend only on `(i, mf)`, so the scan output is deterministic;
    /// the merge phase processes runs in input order regardless of thread
    /// count, so floating-point summation is bit-stable too.
    #[test]
    fn scan_determinism_across_thread_counts() {
        // 16 distinct 2-qubit terms, fanout-1 channel (`H` on qubit 0).
        let labels = ["I", "X", "Y", "Z"];
        let mut owned: Vec<(String, Complex64)> = Vec::new();
        let mut k: i64 = 0;
        for a in &labels {
            for b in &labels {
                k += 1;
                owned.push((
                    format!("{}{}", a, b),
                    Complex64::new(k as f64 * 0.13, k as f64 * 0.07),
                ));
            }
        }
        let strings: Vec<(&str, Complex64)> = owned.iter().map(|(s, c)| (s.as_str(), *c)).collect();
        let input = PauliSum::<1>::from_strings(&strings);
        let h = Clifford1Q::h(0);
        let policy = AlwaysKeep;

        let run = |n: usize| -> PauliSum<1> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap();
            pool.install(|| apply_layer(&input, &h, &policy))
        };
        let r1 = run(1);
        let r2 = run(2);
        let r4 = run(4);

        assert_eq!(r1.x(), r2.x());
        assert_eq!(r1.z(), r2.z());
        assert_eq!(r1.coeff(), r2.coeff());
        assert_eq!(r1.x(), r4.x());
        assert_eq!(r1.z(), r4.z());
        assert_eq!(r1.coeff(), r4.coeff());
    }

    /// Hand-built unsorted scratch becomes sorted; coeffs follow their keys.
    /// Lex on `(x, z)`: I < Z < X < Y per word (since X has x=1>0 and Z has z=1
    /// only, x[0] dominates).
    #[test]
    fn sort_phase_orders_by_lex_key() {
        // Three terms in non-sorted order: X, I, Z. Expected sort: I, Z, X.
        let mut x: Vec<[u64; 1]> = vec![[1], [0], [0]];
        let mut z: Vec<[u64; 1]> = vec![[0], [0], [1]];
        let mut c: Vec<Complex64> = vec![
            Complex64::new(7.0, 0.0), // X tag
            Complex64::new(8.0, 0.0), // I tag
            Complex64::new(9.0, 0.0), // Z tag
        ];
        sort_phase(&mut x, &mut z, &mut c, 3);
        assert_eq!(x[0], [0]);
        assert_eq!(z[0], [0]);
        assert_eq!(c[0], Complex64::new(8.0, 0.0));
        assert_eq!(x[1], [0]);
        assert_eq!(z[1], [1]);
        assert_eq!(c[1], Complex64::new(9.0, 0.0));
        assert_eq!(x[2], [1]);
        assert_eq!(z[2], [0]);
        assert_eq!(c[2], Complex64::new(7.0, 0.0));
    }

    /// A pre-sorted scratch survives sort_phase intact.
    #[test]
    fn sort_phase_preserves_already_sorted() {
        let mut x: Vec<[u64; 1]> = vec![[0], [0], [1]];
        let mut z: Vec<[u64; 1]> = vec![[0], [1], [0]];
        let mut c: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];
        sort_phase(&mut x, &mut z, &mut c, 3);
        assert_eq!(x, vec![[0u64], [0u64], [1u64]]);
        assert_eq!(z, vec![[0u64], [1u64], [0u64]]);
        assert_eq!(
            c,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ]
        );
    }

    /// Stable sort: two outputs with identical (x, z) keep their input
    /// relative order. Distinguishable via their coefficients.
    #[test]
    fn sort_phase_is_stable_on_equal_keys() {
        // Three Z terms with coeffs 1, 2, 3 in input order, plus one X
        // separating coefficient 1 and 2 to force a non-trivial permutation.
        let mut x: Vec<[u64; 1]> = vec![[0], [1], [0], [0]];
        let mut z: Vec<[u64; 1]> = vec![[1], [0], [1], [1]];
        let mut c: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),  // Z (first)
            Complex64::new(99.0, 0.0), // X
            Complex64::new(2.0, 0.0),  // Z (second)
            Complex64::new(3.0, 0.0),  // Z (third)
        ];
        sort_phase(&mut x, &mut z, &mut c, 4);
        // Sorted order: Z, Z, Z, X. Z coeffs must come out in input order
        // 1, 2, 3 — the stability check.
        assert_eq!(x[0], [0]);
        assert_eq!(z[0], [1]);
        assert_eq!(c[0], Complex64::new(1.0, 0.0));
        assert_eq!(x[1], [0]);
        assert_eq!(z[1], [1]);
        assert_eq!(c[1], Complex64::new(2.0, 0.0));
        assert_eq!(x[2], [0]);
        assert_eq!(z[2], [1]);
        assert_eq!(c[2], Complex64::new(3.0, 0.0));
        assert_eq!(x[3], [1]);
        assert_eq!(z[3], [0]);
        assert_eq!(c[3], Complex64::new(99.0, 0.0));
    }

    /// W=2: lex priority is `x[0]` first (low word), then `x[1]`, then `z[0]`,
    /// `z[1]`. Word 0 dominates word 1 in cmp.
    #[test]
    fn sort_phase_w2_cross_word_priority() {
        // Term A: x=[0, 99], z=[0, 0]   (X-bits in word 1)
        // Term B: x=[1, 0],  z=[0, 0]   (X-bit in word 0, low value)
        // Lex cmp: A.x[0]=0 < B.x[0]=1, so A < B.
        let mut x: Vec<[u64; 2]> = vec![[1, 0], [0, 99]];
        let mut z: Vec<[u64; 2]> = vec![[0, 0], [0, 0]];
        let mut c: Vec<Complex64> = vec![
            Complex64::new(11.0, 0.0), // B
            Complex64::new(22.0, 0.0), // A
        ];
        sort_phase(&mut x, &mut z, &mut c, 2);
        assert_eq!(x[0], [0, 99]);
        assert_eq!(c[0], Complex64::new(22.0, 0.0));
        assert_eq!(x[1], [1, 0]);
        assert_eq!(c[1], Complex64::new(11.0, 0.0));
    }

    /// `len < 2` is a no-op short-circuit. Verify behavior on empty and
    /// single-element prefixes.
    #[test]
    fn sort_phase_len_lt_2_is_noop() {
        let mut x: Vec<[u64; 1]> = vec![[5]];
        let mut z: Vec<[u64; 1]> = vec![[7]];
        let mut c: Vec<Complex64> = vec![Complex64::new(1.0, 2.0)];
        sort_phase(&mut x, &mut z, &mut c, 1);
        assert_eq!(x[0], [5]);
        assert_eq!(z[0], [7]);
        assert_eq!(c[0], Complex64::new(1.0, 2.0));

        let mut empty_x: Vec<[u64; 1]> = vec![];
        let mut empty_z: Vec<[u64; 1]> = vec![];
        let mut empty_c: Vec<Complex64> = vec![];
        sort_phase(&mut empty_x, &mut empty_z, &mut empty_c, 0);
        assert!(empty_x.is_empty());
    }

    /// Truncation policy that always keeps terms — exercises the trait bound
    /// without affecting merge_phase output (Phase 6 doesn't fold keep_term
    /// into the merge yet).
    struct AlwaysKeep;
    impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

    #[test]
    fn merge_phase_empty_input() {
        let x: Vec<[u64; 1]> = vec![];
        let z: Vec<[u64; 1]> = vec![];
        let c: Vec<Complex64> = vec![];
        let out = merge_phase::<1, _>(&x, &z, &c, 0, 4, &AlwaysKeep);
        assert!(out.is_empty());
        assert_eq!(out.num_qubits(), 4);
        out.assert_invariants();
    }

    #[test]
    fn merge_phase_distinct_keys_pass_through() {
        // Sorted: I, Z, X.
        let x: Vec<[u64; 1]> = vec![[0], [0], [1]];
        let z: Vec<[u64; 1]> = vec![[0], [1], [0]];
        let c: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];
        let out = merge_phase::<1, _>(&x, &z, &c, 3, 1, &AlwaysKeep);
        assert_eq!(out.len(), 3);
        assert_eq!(out.x(), &[[0u64], [0u64], [1u64]]);
        assert_eq!(out.z(), &[[0u64], [1u64], [0u64]]);
        assert_eq!(
            out.coeff(),
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ]
        );
        out.assert_invariants();
    }

    #[test]
    fn merge_phase_combines_adjacent_duplicates() {
        // Three Z entries (coeffs 1, 2, 3) followed by one X (coeff 7).
        let x: Vec<[u64; 1]> = vec![[0], [0], [0], [1]];
        let z: Vec<[u64; 1]> = vec![[1], [1], [1], [0]];
        let c: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(7.0, 0.0),
        ];
        let out = merge_phase::<1, _>(&x, &z, &c, 4, 1, &AlwaysKeep);
        assert_eq!(out.len(), 2);
        // Z with summed coeff 6, then X with 7.
        assert_eq!(out.x()[0], [0]);
        assert_eq!(out.z()[0], [1]);
        assert_eq!(out.coeff()[0], Complex64::new(6.0, 0.0));
        assert_eq!(out.x()[1], [1]);
        assert_eq!(out.z()[1], [0]);
        assert_eq!(out.coeff()[1], Complex64::new(7.0, 0.0));
        out.assert_invariants();
    }

    #[test]
    fn merge_phase_drops_exact_zero_runs() {
        // Z with coeffs +1 and -1 → cancels to zero, dropped. X with 5
        // survives.
        let x: Vec<[u64; 1]> = vec![[0], [0], [1]];
        let z: Vec<[u64; 1]> = vec![[1], [1], [0]];
        let c: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(5.0, 0.0),
        ];
        let out = merge_phase::<1, _>(&x, &z, &c, 3, 1, &AlwaysKeep);
        assert_eq!(out.len(), 1);
        assert_eq!(out.x()[0], [1]);
        assert_eq!(out.z()[0], [0]);
        assert_eq!(out.coeff()[0], Complex64::new(5.0, 0.0));
        out.assert_invariants();
    }

    /// Slice 7.1: the policy's `keep_term` runs inside the merge loop.
    /// `CoefficientThreshold(1e-6)` drops the X term (coeff 1e-9) but keeps
    /// the Z term (coeff 0.5).
    #[test]
    fn merge_phase_drops_below_threshold() {
        let x: Vec<[u64; 1]> = vec![[0], [1]]; // Z, then X
        let z: Vec<[u64; 1]> = vec![[1], [0]];
        let c: Vec<Complex64> = vec![Complex64::new(0.5, 0.0), Complex64::new(1e-9, 0.0)];
        let out = merge_phase::<1, _>(&x, &z, &c, 2, 1, &CoefficientThreshold(1e-6));
        assert_eq!(out.len(), 1);
        assert_eq!(out.x()[0], [0]);
        assert_eq!(out.z()[0], [1]);
        assert!(approx_eq(out.coeff()[0], Complex64::new(0.5, 0.0), TOL));
        out.assert_invariants();
    }

    /// Slice 7.1: the threshold is checked *after* coefficients are summed,
    /// not on individual scratch entries. Two Z terms with coeffs 0.5 and
    /// -0.4999999 sum to 1e-7, which is below threshold 1e-6 — drop the
    /// merged term, even though each summand individually exceeds 1e-6.
    #[test]
    fn merge_phase_threshold_applied_after_summation() {
        let x: Vec<[u64; 1]> = vec![[0], [0]];
        let z: Vec<[u64; 1]> = vec![[1], [1]];
        let c: Vec<Complex64> = vec![Complex64::new(0.5, 0.0), Complex64::new(-0.4999999, 0.0)];
        let out = merge_phase::<1, _>(&x, &z, &c, 2, 1, &CoefficientThreshold(1e-6));
        assert_eq!(out.len(), 0);
        out.assert_invariants();
    }

    /// `CoefficientThreshold(0.0)` keeps every (non-zero) summed term — the
    /// no-op case for the new keep_term path.
    #[test]
    fn merge_phase_zero_threshold_keeps_everything() {
        let x: Vec<[u64; 1]> = vec![[0], [0], [1]];
        let z: Vec<[u64; 1]> = vec![[0], [1], [0]];
        let c: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];
        let out = merge_phase::<1, _>(&x, &z, &c, 3, 1, &CoefficientThreshold(0.0));
        assert_eq!(out.len(), 3);
        out.assert_invariants();
    }

    /// Buffer with a populated prefix `[0..len)` and trailing junk: the
    /// junk must not be merged.
    #[test]
    fn merge_phase_ignores_trailing_junk() {
        let x: Vec<[u64; 1]> = vec![[0], [99], [99]]; // junk at idx 1, 2
        let z: Vec<[u64; 1]> = vec![[1], [99], [99]];
        let c: Vec<Complex64> = vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(99.0, 0.0),
            Complex64::new(99.0, 0.0),
        ];
        let out = merge_phase::<1, _>(&x, &z, &c, 1, 1, &AlwaysKeep);
        assert_eq!(out.len(), 1);
        assert_eq!(out.x()[0], [0]);
        assert_eq!(out.z()[0], [1]);
        assert_eq!(out.coeff()[0], Complex64::new(2.0, 0.0));
    }

    /// Slice 8.2 stress: a run of identical keys straddles the `len/2`
    /// boundary. With `nchunks = 2`, the natural boundary at index 4 lands
    /// inside the Z-run; alignment must advance it forward to index 7 so
    /// the Z-run merges into a single term. Lex `(x, z)` sort puts
    /// `I < Z < X`.
    #[test]
    fn merge_phase_run_spans_chunk_boundary() {
        // Sorted: I, I, Z, Z, Z, Z, Z, X, X   (len = 9, mid = 4 inside Z-run)
        let x: Vec<[u64; 1]> = vec![[0], [0], [0], [0], [0], [0], [0], [1], [1]];
        let z: Vec<[u64; 1]> = vec![[0], [0], [1], [1], [1], [1], [1], [0], [0]];
        let c: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 0.0),
        ];
        let out = merge_phase_with_nchunks::<1, _>(&x, &z, &c, 9, 1, &AlwaysKeep, 2);
        assert_eq!(out.len(), 3);
        assert_eq!(out.x()[0], [0]);
        assert_eq!(out.z()[0], [0]);
        assert_eq!(out.coeff()[0], Complex64::new(2.0, 0.0));
        assert_eq!(out.x()[1], [0]);
        assert_eq!(out.z()[1], [1]);
        assert_eq!(out.coeff()[1], Complex64::new(2.5, 0.0));
        assert_eq!(out.x()[2], [1]);
        assert_eq!(out.z()[2], [0]);
        assert_eq!(out.coeff()[2], Complex64::new(4.0, 0.0));
        out.assert_invariants();
    }

    /// Slice 8.2: when the natural midpoint already falls at a run break,
    /// alignment leaves it alone — chunks split exactly between runs.
    #[test]
    fn merge_phase_aligned_boundary_no_shift() {
        // Sorted: I, I, I, X, X, X   (len = 6, mid = 3 lands on run break)
        let x: Vec<[u64; 1]> = vec![[0], [0], [0], [1], [1], [1]];
        let z: Vec<[u64; 1]> = vec![[0], [0], [0], [0], [0], [0]];
        let c: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
        ];
        let out = merge_phase_with_nchunks::<1, _>(&x, &z, &c, 6, 1, &AlwaysKeep, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out.x()[0], [0]);
        assert_eq!(out.z()[0], [0]);
        assert_eq!(out.coeff()[0], Complex64::new(6.0, 0.0));
        assert_eq!(out.x()[1], [1]);
        assert_eq!(out.z()[1], [0]);
        assert_eq!(out.coeff()[1], Complex64::new(15.0, 0.0));
        out.assert_invariants();
    }

    /// Slice 8.2: degenerate input where every key collapses to one run.
    /// All chunk boundaries advance to `len`; only one effective chunk.
    #[test]
    fn merge_phase_all_same_key_with_nchunks() {
        let x: Vec<[u64; 1]> = vec![[0], [0], [0], [0], [0]];
        let z: Vec<[u64; 1]> = vec![[1], [1], [1], [1], [1]];
        let c: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let out = merge_phase_with_nchunks::<1, _>(&x, &z, &c, 5, 1, &AlwaysKeep, 4);
        assert_eq!(out.len(), 1);
        assert_eq!(out.x()[0], [0]);
        assert_eq!(out.z()[0], [1]);
        assert_eq!(out.coeff()[0], Complex64::new(5.0, 0.0));
        out.assert_invariants();
    }
}
