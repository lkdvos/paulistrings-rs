// K10: one exchange block per remote delta, CSR by the receiver's destination position (ARCHITECTURE.md §Partitioning).
// counts[p] is entry e's row count from source bucket bucket_at[p] ^ bd; the fill writes segment p warp-compacted in source order, rows bitwise the local emitters'.

extern "C" __global__ void k_export_counts(const u32* __restrict__ cnt, const u32* __restrict__ bucket_at, u32 bd,
                                           u32 e, u32 E, u32 B, u32* __restrict__ counts) {
    const u32 p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= B) return;
    counts[p] = cnt[(size_t)(bucket_at[p] ^ bd) * E + e];
}

extern "C" __global__ void k_export_fill(const u64* __restrict__ x, const u64* __restrict__ z,
                                         const double* __restrict__ c, const u32* __restrict__ in_start,
                                         const u32* __restrict__ in_len, const u32* __restrict__ bucket_at,
                                         TABLE_ARGS, u32 bd, u32 e, u32 B, const u32* __restrict__ offsets,
                                         u64* __restrict__ ox, u64* __restrict__ oz, double* __restrict__ oc) {
    MAKE_TABLE(T);
    const u32 lane = lane_id();
    const u32 p = blockIdx.x * (blockDim.x / WARP) + warp_id();
    if (p >= B) return;
    const u32 sb = bucket_at[p] ^ bd;
    const u32 start = in_start[sb], len = in_len[sb];
    u32 w = offsets[p];
    for (u32 r0 = 0; r0 < len; r0 += WARP) {
        const u32 r = r0 + lane;
        bool ok = false;
        Key k;
        double pr = 0.0, pi = 0.0;
        if (r < len) {
            const size_t src = (size_t)start + r;
            k = load_key(x, z, src);
            ok = entry_emits(T, k, e);
            if (ok) entry_product(T, k, e, c[2 * src], c[2 * src + 1], pr, pi);
        }
        const u32 bal = __ballot_sync(~0u, ok);
        if (ok) {
            const size_t d = (size_t)w + __popc(bal & lanemask_lt());
#pragma unroll
            for (int i = 0; i < W; ++i) {
                ox[d * W + i] = k.x[i] ^ mask[(e * 2) * W + i];
                oz[d * W + i] = k.z[i] ^ mask[(e * 2 + 1) * W + i];
            }
            oc[2 * d] = pr;
            oc[2 * d + 1] = pi;
        }
        w += __popc(bal);
    }
}
