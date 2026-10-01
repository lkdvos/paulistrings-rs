// K0: the GF(2)-linear fingerprint g = G·(x, z) of FP_ROWS rows, masked to FP_BITS.

__device__ __forceinline__ u64 fingerprint(const Key& k, const u64* fp_rows) {
    u64 out = 0;
#pragma unroll 4
    for (u32 r = 0; r < FP_ROWS; ++r) out |= (u64)row_parity(k, fp_rows, r) << r;
    return out & fp_mask();
}

extern "C" __global__ void k_fingerprint(const u64* __restrict__ x, const u64* __restrict__ z, u32 m,
                                         const u64* __restrict__ fp_rows, u64* __restrict__ g) {
    __shared__ u64 srows[FP_ROWS * 2 * W];
    for (u32 i = threadIdx.x; i < FP_ROWS * 2 * W; i += blockDim.x) srows[i] = fp_rows[i];
    __syncthreads();
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    g[i] = fingerprint(load_key(x, z, i), srows);
}
