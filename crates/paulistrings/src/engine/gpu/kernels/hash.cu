// The device image of `Gf2Hash::bucket_of` over rows in the prelude's layout.

__device__ __forceinline__ u32 bucket_of(const Key& k, const u64* hash_rows, u32 bits) {
    u32 acc = 0;
    for (u32 r = 0; r < bits; ++r) acc |= row_parity(k, hash_rows, r) << r;
    return acc;
}
