// Device images of `Gf2Hash::bucket_of` and `PartitionRows::partition_of` over rows in the prelude's layout.

__device__ __forceinline__ u32 gf2_image(const Key& k, const u64* rows, u32 nrows) {
    u32 acc = 0;
    for (u32 r = 0; r < nrows; ++r) acc |= row_parity(k, rows, r) << r;
    return acc;
}

__device__ __forceinline__ u32 bucket_of(const Key& k, const u64* hash_rows, u32 bits) {
    return gf2_image(k, hash_rows, bits);
}

__device__ __forceinline__ u32 partition_of(const Key& k, const u64* part_rows, u32 bits) {
    return gf2_image(k, part_rows, bits);
}

extern "C" __global__ void k_bucket_of(const u64* __restrict__ x, const u64* __restrict__ z, u32 m,
                                       const u64* __restrict__ hash_rows, u32 bits, u32* __restrict__ out) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    out[i] = bucket_of(load_key(x, z, i), hash_rows, bits);
}

extern "C" __global__ void k_partition_of(const u64* __restrict__ x, const u64* __restrict__ z, u32 m,
                                          const u64* __restrict__ part_rows, u32 bits, u32* __restrict__ out) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    out[i] = partition_of(load_key(x, z, i), part_rows, bits);
}
