// K11: the per-term half of `PauliSum::assert_invariants` on device, one warp per bucket.
// bad[0..4) count misplaced, not strictly ascending, out-of-range and stale-fingerprint rows; bad[4] is the lowest offending bucket.

extern "C" __global__ void k_check_invariants(const u64* __restrict__ x, const u64* __restrict__ z,
                                              const u64* __restrict__ g, const u32* __restrict__ start,
                                              const u32* __restrict__ len, const u64* __restrict__ hash_rows,
                                              u32 bits, u32 buckets, const u64* __restrict__ fp_rows,
                                              u32 num_qubits, u32* __restrict__ bad) {
    const u32 beta = blockIdx.x * (blockDim.x / WARP) + warp_id();
    if (beta >= buckets) return;
    const u32 s = start[beta], n = len[beta];
    u32 misplaced = 0, unordered = 0, out_of_range = 0, stale = 0;
    for (u32 i = lane_id(); i < n; i += WARP) {
        const size_t r = (size_t)s + i;
        const Key k = load_key(x, z, r);
        misplaced += bucket_of(k, hash_rows, bits) != beta;
        unordered += i > 0 && !key_lt(load_key(x, z, r - 1), k);
        u64 beyond = 0;
#pragma unroll
        for (int w = 0; w < W; ++w) beyond |= (k.x[w] | k.z[w]) & ~word_mask(num_qubits, w);
        out_of_range += beyond != 0;
        stale += fingerprint(k, fp_rows) != g[r];
    }
    if (misplaced | unordered | out_of_range | stale) {
        atomicAdd(&bad[0], misplaced);
        atomicAdd(&bad[1], unordered);
        atomicAdd(&bad[2], out_of_range);
        atomicAdd(&bad[3], stale);
        atomicMin(&bad[4], beta);
    }
}
