// k_fingerprint: the GF(2)-linear fingerprint g = G·(x, z) of FINGERPRINT_ROWS rows, masked to FINGERPRINT_BITS.

__device__ __forceinline__ u64 fingerprint(const Key& k, const u64* fingerprint_rows) {
    u64 out = 0;
#pragma unroll 4
    for (u32 r = 0; r < FINGERPRINT_ROWS; ++r) out |= (u64)row_parity(k, fingerprint_rows, r) << r;
    return out & fingerprint_mask();
}

extern "C" __global__ void k_fingerprint(const u64* __restrict__ x, const u64* __restrict__ z, u32 m,
                                         const u64* __restrict__ fingerprint_rows, u64* __restrict__ g) {
    __shared__ u64 srows[FINGERPRINT_ROWS * 2 * W];
    for (u32 i = threadIdx.x; i < FINGERPRINT_ROWS * 2 * W; i += blockDim.x) srows[i] = fingerprint_rows[i];
    __syncthreads();
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    g[i] = fingerprint(load_key(x, z, i), srows);
}
