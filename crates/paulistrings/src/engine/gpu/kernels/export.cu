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

// K12: the sender-side merge of one partner's blocks; K3 runs between the counts and the split over the partner's sub-table.
// sub[beta * K + j] = cnt[beta * E + sel[j]], the count table of the sub-table from the layer's.
extern "C" __global__ void k_premerge_counts(const u32* __restrict__ cnt, u32 E, const u32* __restrict__ sel, u32 K,
                                             u32 B, u32* __restrict__ sub) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * K) return;
    sub[i] = cnt[(size_t)(i / K) * E + sel[i % K]];
}

// lens[j * n + i]: the merged rows of position p0 + i handed to block j, greedily in block order and never more than block j's unmerged count there, so no segment outgrows the tag's offset field.
extern "C" __global__ void k_premerge_split(const u32* __restrict__ out_len_pos, const u32* __restrict__ sub,
                                            const u32* __restrict__ bucket_at, const u32* __restrict__ bd, u32 K,
                                            u32 p0, u32 n, u32* __restrict__ lens) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u32 p = p0 + i;
    const u32 beta = bucket_at[p];
    u32 m = out_len_pos[p];
    for (u32 j = 0; j < K; ++j) {
        const u32 t = min(m, sub[(size_t)(beta ^ bd[j]) * K + j]);
        lens[(size_t)j * n + i] = t;
        m -= t;
    }
}

// Block j's share of every position in the batch, from the batch-relative arena to dst0 + loff[j * n + i] - loff[j * n]; one warp per position.
extern "C" __global__ void k_premerge_copy(const u64* __restrict__ ax, const u64* __restrict__ az,
                                           const double* __restrict__ ac, const u64* __restrict__ ag,
                                           const u32* __restrict__ seg_start, const u32* __restrict__ lens,
                                           const u32* __restrict__ loff, u32 j, u32 p0, u32 n, u32 dst0, u32 with_g,
                                           u64* __restrict__ ox, u64* __restrict__ oz, double* __restrict__ oc,
                                           u64* __restrict__ og) {
    const u32 lane = lane_id();
    const u32 i = blockIdx.x * (blockDim.x / WARP) + warp_id();
    if (i >= n) return;
    u32 skip = 0;
    for (u32 k = 0; k < j; ++k) skip += lens[(size_t)k * n + i];
    const size_t src0 = (size_t)(seg_start[p0 + i] - seg_start[p0]) + skip;
    const size_t d0 = (size_t)dst0 + (loff[(size_t)j * n + i] - loff[(size_t)j * n]);
    const u32 len = lens[(size_t)j * n + i];
    for (u32 r = lane; r < len; r += WARP) {
        const size_t s = src0 + r, d = d0 + r;
#pragma unroll
        for (int w = 0; w < W; ++w) {
            ox[d * W + w] = ax[s * W + w];
            oz[d * W + w] = az[s * W + w];
        }
        oc[2 * d] = ac[2 * s];
        oc[2 * d + 1] = ac[2 * s + 1];
        if (with_g) og[d] = ag[s];
    }
}
