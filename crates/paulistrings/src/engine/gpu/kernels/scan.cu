// Device-wide exclusive scan and max of a u32 column: per-block scan, scan of the block sums, add-back.
// SCAN_BLOCK elements per block and at most SCAN_BLOCK blocks, so n <= 2^24, beyond 2^B_MAX_BITS + 1.

#define SCAN_THREADS 1024
#define SCAN_WARPS (SCAN_THREADS / WARP)
#define SCAN_ITEMS 4
#define SCAN_BLOCK (SCAN_THREADS * SCAN_ITEMS)

extern "C" __global__ void k_scan_block(const u32* __restrict__ in, u32* __restrict__ out,
                                        u32* __restrict__ block_sum, u32* __restrict__ block_max, u32 n) {
    __shared__ u32 sv[SCAN_BLOCK];
    __shared__ u32 swarp[36];
    __shared__ u32 smax[SCAN_WARPS];
    const u32 base = blockIdx.x * SCAN_BLOCK;
    const u32 tid = threadIdx.x;
    u32 mx = 0;
#pragma unroll
    for (int c = 0; c < SCAN_ITEMS; ++c) {
        u32 i = tid * SCAN_ITEMS + c;
        u32 v = (base + i < n) ? in[base + i] : 0u;
        sv[i] = v;
        mx = max(mx, v);
    }
#pragma unroll
    for (u32 o = 16; o > 0; o >>= 1) mx = max(mx, __shfl_down_sync(~0u, mx, o));
    if (lane_id() == 0) smax[warp_id()] = mx;
    __syncthreads();
    u32 total = block_exscan<SCAN_ITEMS, u32>(sv, SCAN_BLOCK, swarp);
#pragma unroll
    for (int c = 0; c < SCAN_ITEMS; ++c) {
        u32 i = tid * SCAN_ITEMS + c;
        if (base + i < n) out[base + i] = sv[i];
    }
    if (tid == 0) {
        u32 m2 = 0;
        for (int w = 0; w < SCAN_WARPS; ++w) m2 = max(m2, smax[w]);
        block_sum[blockIdx.x] = total;
        block_max[blockIdx.x] = m2;
    }
}

// One block: scans block_sum in place and writes out[n] = total, tot_max = (total, max).
extern "C" __global__ void k_scan_single(u32* __restrict__ block_sum, const u32* __restrict__ block_max,
                                         u32 nblocks, u32* __restrict__ out, u32 n, u32* __restrict__ tot_max) {
    __shared__ u32 sv[SCAN_BLOCK];
    __shared__ u32 swarp[36];
    __shared__ u32 smax[SCAN_WARPS];
    const u32 tid = threadIdx.x;
    u32 mx = 0;
#pragma unroll
    for (int c = 0; c < SCAN_ITEMS; ++c) {
        u32 i = tid * SCAN_ITEMS + c;
        sv[i] = (i < nblocks) ? block_sum[i] : 0u;
        if (i < nblocks) mx = max(mx, block_max[i]);
    }
#pragma unroll
    for (u32 o = 16; o > 0; o >>= 1) mx = max(mx, __shfl_down_sync(~0u, mx, o));
    if (lane_id() == 0) smax[warp_id()] = mx;
    __syncthreads();
    u32 total = block_exscan<SCAN_ITEMS, u32>(sv, SCAN_BLOCK, swarp);
#pragma unroll
    for (int c = 0; c < SCAN_ITEMS; ++c) {
        u32 i = tid * SCAN_ITEMS + c;
        if (i < nblocks) block_sum[i] = sv[i];
    }
    if (tid == 0) {
        u32 m2 = 0;
        for (int w = 0; w < SCAN_WARPS; ++w) m2 = max(m2, smax[w]);
        out[n] = total;
        tot_max[0] = total;
        tot_max[1] = m2;
    }
}

extern "C" __global__ void k_scan_add(u32* __restrict__ out, const u32* __restrict__ block_off, u32 n) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += block_off[i / SCAN_BLOCK];
}
