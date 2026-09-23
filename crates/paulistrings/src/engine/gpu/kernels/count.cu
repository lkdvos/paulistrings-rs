// K1: cnt[beta * E + e] = rows of bucket beta that entry e emits, one warp per bucket.
// K2: rows[p] = sum_e cnt[(bucket_at[p] ^ bd[e]) * E + e], the record count of the fused block for position p.

#define TABLE_ARGS u32 mode, u32 E, u32 kq, u32 q0, u32 q1, double rcos, double rsin, const double* __restrict__ amp, const u64* __restrict__ mask, const u32* __restrict__ nz
#define MAKE_TABLE(T) Table T; T.mode = mode; T.entries = E; T.kq = kq; T.q0 = q0; T.q1 = q1; T.rot_cos = rcos; T.rot_sin = rsin; T.amp = amp; T.mask = mask; T.nz = nz

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
                                  const u32* __restrict__ bd, u32* __restrict__ rows, u32 B, u32 E) {
    const u32 p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= B) return;
    const u32 beta = bucket_at[p];
    u32 r = 0;
    for (u32 e = 0; e < E; ++e) r += cnt[(size_t)(beta ^ bd[e]) * E + e];
    rows[p] = r;
}
