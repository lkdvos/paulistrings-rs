// K6: refine by `delta` hash rows in one counting pass, one warp per source bucket.
// Row r of bucket beta moves to beta | (s << bits_old), s the parities of rows bits_old.., keeping its order within the new bucket.

#define REFINE_MAX_DELTA 4
#define REFINE_MAX_SUB (1 << REFINE_MAX_DELTA)

__device__ __forceinline__ void load_new_rows(u64* srows, const u64* hash_rows, u32 bits_old, u32 delta) {
    for (u32 i = threadIdx.x; i < 2 * delta * W; i += blockDim.x) srows[i] = hash_rows[2 * bits_old * W + i];
    __syncthreads();
}

__device__ __forceinline__ u32 refine_sub(const Key& k, const u64* srows, u32 delta) {
    u32 s = 0;
    for (u32 j = 0; j < delta; ++j) s |= row_parity(k, srows, j) << j;
    return s;
}

extern "C" __global__ void k_refine_count(const u64* __restrict__ x, const u64* __restrict__ z,
                                          const u32* __restrict__ in_start, const u32* __restrict__ in_len,
                                          const u64* __restrict__ hash_rows, u32 b_old, u32 bits_old, u32 delta,
                                          u32* __restrict__ new_len) {
    __shared__ u64 srows[2 * REFINE_MAX_DELTA * W];
    load_new_rows(srows, hash_rows, bits_old, delta);
    const u32 lane = lane_id();
    const u32 beta = blockIdx.x * (blockDim.x / WARP) + warp_id();
    if (beta >= b_old) return;
    const u32 start = in_start[beta], end = start + in_len[beta];
    u32 c[REFINE_MAX_SUB];
#pragma unroll
    for (int s = 0; s < REFINE_MAX_SUB; ++s) c[s] = 0;
    for (u32 r = start + lane; r < end; r += WARP) {
        const u32 d = refine_sub(load_key(x, z, r), srows, delta);
#pragma unroll
        for (int s = 0; s < REFINE_MAX_SUB; ++s) c[s] += (d == (u32)s);
    }
    const u32 nsub = 1u << delta;
#pragma unroll
    for (int s = 0; s < REFINE_MAX_SUB; ++s) {
        u32 v = c[s];
#pragma unroll
        for (u32 o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
        if (lane == 0 && (u32)s < nsub) new_len[beta | ((u32)s << bits_old)] = v;
    }
}

extern "C" __global__ void k_refine_scatter(const u64* __restrict__ x, const u64* __restrict__ z,
                                            const double* __restrict__ c, const u64* __restrict__ g,
                                            const u32* __restrict__ in_start, const u32* __restrict__ in_len,
                                            const u64* __restrict__ hash_rows, u32 b_old, u32 bits_old, u32 delta,
                                            const u32* __restrict__ new_start, u64* __restrict__ ox,
                                            u64* __restrict__ oz, double* __restrict__ oc, u64* __restrict__ og) {
    __shared__ u64 srows[2 * REFINE_MAX_DELTA * W];
    load_new_rows(srows, hash_rows, bits_old, delta);
    const u32 lane = lane_id();
    const u32 beta = blockIdx.x * (blockDim.x / WARP) + warp_id();
    if (beta >= b_old) return;
    const u32 start = in_start[beta], len = in_len[beta];
    const u32 nsub = 1u << delta;
    u32 run[REFINE_MAX_SUB];
#pragma unroll
    for (int s = 0; s < REFINE_MAX_SUB; ++s) run[s] = 0;
    // Chunks of 32 are taken in source order and ranked by lane within each chunk, which is what makes the split stable.
    for (u32 r0 = 0; r0 < len; r0 += WARP) {
        const u32 r = r0 + lane;
        const bool ok = r < len;
        Key k;
        u32 d = 0;
        if (ok) {
            k = load_key(x, z, start + r);
            d = refine_sub(k, srows, delta);
        }
        for (u32 s = 0; s < nsub; ++s) {
            const bool mine = ok && d == s;
            const u32 bal = __ballot_sync(~0u, mine);
            if (mine) {
                const size_t dst = (size_t)new_start[beta | (s << bits_old)] + run[s] + __popc(bal & lanemask_lt());
                const size_t src = (size_t)start + r;
#pragma unroll
                for (int w = 0; w < W; ++w) {
                    ox[dst * W + w] = k.x[w];
                    oz[dst * W + w] = k.z[w];
                }
                oc[2 * dst] = c[2 * src];
                oc[2 * dst + 1] = c[2 * src + 1];
                og[dst] = g[src];
            }
            run[s] += __popc(bal);
        }
    }
}
