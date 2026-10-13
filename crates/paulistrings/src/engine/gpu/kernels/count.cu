// k_count: cnt[beta * E + e] = rows of bucket beta that entry e emits, one warp per bucket.
// k_rows: rows[p] = sum over local entries of cnt[(bucket_at[p] ^ bucket_delta[e]) * E + e] plus, per received entry, the length of segment p of its block.

#define TABLE_ARGS u32 mode, u32 E, u32 support_size, u32 qubit_0, u32 qubit_1, double rotation_cos, double rotation_sin, const double* __restrict__ amplitude, const u64* __restrict__ mask, const u32* __restrict__ nonzero_patterns
#define MAKE_TABLE(T) Table T; T.mode = mode; T.entries = E; T.support_size = support_size; T.qubit_0 = qubit_0; T.qubit_1 = qubit_1; T.rotation_cos = rotation_cos; T.rotation_sin = rotation_sin; T.amplitude = amplitude; T.mask = mask; T.nonzero_patterns = nonzero_patterns

extern "C" __global__ void k_count(const u64* __restrict__ x, const u64* __restrict__ z,
                                   const u32* __restrict__ in_start, const u32* __restrict__ in_len, u32 B,
                                   TABLE_ARGS, u32* __restrict__ cnt) {
    MAKE_TABLE(T);
    const u32 lane = lane_id();
    const u32 beta = blockIdx.x * (blockDim.x / WARP) + warp_id();
    if (beta >= B) return;
    const u32 start = in_start[beta], end = start + in_len[beta];
    u32 c[MAX_ENTRIES];
#pragma unroll
    for (int e = 0; e < MAX_ENTRIES; ++e) c[e] = 0;
    for (u32 r = start + lane; r < end; r += WARP) {
        const Key k = load_key(x, z, r);
        // Unrolled with a constant index so `c` stays in registers rather than local memory.
#pragma unroll
        for (int e = 0; e < MAX_ENTRIES; ++e) {
            if ((u32)e < E) c[e] += entry_emits(T, k, (u32)e) ? 1u : 0u;
        }
    }
#pragma unroll
    for (int e = 0; e < MAX_ENTRIES; ++e) {
        u32 v = c[e];
#pragma unroll
        for (u32 o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
        if (lane == 0 && (u32)e < E) cnt[(size_t)beta * E + e] = v;
    }
}

extern "C" __global__ void k_rows(const u32* __restrict__ cnt, const u32* __restrict__ bucket_at,
                                  const u32* __restrict__ bucket_delta, const u32* __restrict__ received_slot,
                                  const u32* __restrict__ recv_off, u32* __restrict__ rows, u32 B, u32 E) {
    const u32 p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= B) return;
    const u32 beta = bucket_at[p];
    u32 r = 0;
    for (u32 e = 0; e < E; ++e) {
        const u32 k = received_slot[e];
        if (k == NO_REMOTE) {
            r += cnt[(size_t)(beta ^ bucket_delta[e]) * E + e];
        } else {
            const size_t o = (size_t)k * (B + 1) + p;
            r += recv_off[o + 1] - recv_off[o];
        }
    }
    rows[p] = r;
}
