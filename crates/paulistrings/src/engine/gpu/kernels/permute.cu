// K12-K14: the permutation layer for a table under which no two input keys emit one output key (`DevicePrepared::permutation`, ARCHITECTURE.md §GPU-Readiness).
// K12 counts the surviving rows per (source bucket, entry), K13 sizes every output bucket from those counts, and K14 scatters each row to bucket beta ^ bd[e] with no sort, no arena and no compaction.
// Output bucket beta' holds, entry by entry, the survivors of source bucket beta' ^ bd[e] in source order; nothing reads that order.

// The row entry `e` emits for source row `src`, and whether it survives: `entry_of[s]` picks the entry, then the product and the keep program decide exactly as the fused layer's `survives`.
__device__ __forceinline__ bool perm_row(const Table& T, const u32* __restrict__ entry_of,
                                         const u64* __restrict__ x, const u64* __restrict__ z,
                                         const double* __restrict__ c, size_t src, const KeepProg& prog,
                                         Key& k, u32& e, double& pr, double& pi) {
    k = load_key(x, z, src);
    const u32 s = support_bits(k, T.kq, T.q0, T.q1);
    e = entry_of[s];
    if (e == NO_ENTRY) return false;
    entry_product(T, k, e, c[2 * src], c[2 * src + 1], pr, pi);
    if (pr == 0.0 && pi == 0.0) return false;
    return keep_eval(prog, [&]() -> Key {
        Key o = k;
#pragma unroll
        for (int w = 0; w < W; ++w) {
            o.x[w] ^= T.mask[(e * 2) * W + w];
            o.z[w] ^= T.mask[(e * 2 + 1) * W + w];
        }
        return o;
    }, pr, pi);
}

// K12: cnt[beta * E + e] = surviving rows of bucket beta that entry e emits, one block per source bucket at K14's width.
extern "C" __global__ void __launch_bounds__(THREADS) k_perm_count(
    const u64* __restrict__ x, const u64* __restrict__ z, const double* __restrict__ c,
    const u32* __restrict__ in_start, const u32* __restrict__ in_len, TABLE_ARGS,
    const u32* __restrict__ entry_of, const __grid_constant__ KeepProg prog, u32* __restrict__ cnt) {
    MAKE_TABLE(T);
    __shared__ u32 s_cnt[MAX_ENTRIES];
    const u32 beta = blockIdx.x;
    const u32 tid = threadIdx.x, lane = lane_id();
    if (tid < MAX_ENTRIES) s_cnt[tid] = 0;
    __syncthreads();
    const u32 start = in_start[beta], end = start + in_len[beta];
    u32 cn[MAX_ENTRIES];
#pragma unroll
    for (int j = 0; j < MAX_ENTRIES; ++j) cn[j] = 0;
    for (u32 r = start + tid; r < end; r += blockDim.x) {
        Key k;
        u32 e;
        double pr, pi;
        const bool ok = perm_row(T, entry_of, x, z, c, r, prog, k, e, pr, pi);
        // Unrolled with a constant index so `cn` stays in registers rather than local memory.
#pragma unroll
        for (int j = 0; j < MAX_ENTRIES; ++j) cn[j] += (ok && e == (u32)j) ? 1u : 0u;
    }
#pragma unroll
    for (int j = 0; j < MAX_ENTRIES; ++j) {
        u32 v = cn[j];
#pragma unroll
        for (u32 o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
        if (lane == 0 && v != 0) atomicAdd(&s_cnt[j], v);
    }
    __syncthreads();
    if (tid < E) cnt[(size_t)beta * E + tid] = s_cnt[tid];
}

// K13: out_len[beta'] = sum over entries of cnt[(beta' ^ bd[e]) * E + e], one thread per output bucket.
extern "C" __global__ void k_perm_lens(const u32* __restrict__ cnt, const u32* __restrict__ bd, u32 B, u32 E,
                                       u32* __restrict__ out_len) {
    const u32 beta = blockIdx.x * blockDim.x + threadIdx.x;
    if (beta >= B) return;
    u32 n = 0;
    for (u32 e = 0; e < E; ++e) n += cnt[(size_t)(beta ^ bd[e]) * E + e];
    out_len[beta] = n;
}

// K14: one block per source bucket, any multiple of 32 threads up to THREADS wide.
// Rows go in tiles of blockDim.x; within a tile a row's slot is its rank among the tile's rows of the same entry, from a warp match and a per-entry prefix over the warps, so the scatter is deterministic without atomics.
extern "C" __global__ void __launch_bounds__(THREADS) k_perm_scatter(
    const u64* __restrict__ x, const u64* __restrict__ z, const double* __restrict__ c, const u64* __restrict__ g,
    const u32* __restrict__ in_start, const u32* __restrict__ in_len, TABLE_ARGS,
    const u32* __restrict__ entry_of, const u32* __restrict__ bd, const u64* __restrict__ gm,
    const u32* __restrict__ cnt, const u32* __restrict__ out_start, const __grid_constant__ KeepProg prog,
    u64* __restrict__ ox, u64* __restrict__ oz, double* __restrict__ oc, u64* __restrict__ og) {
    MAKE_TABLE(T);
    __shared__ u32 s_base[MAX_ENTRIES];
    __shared__ u32 s_run[MAX_ENTRIES];
    __shared__ u32 s_cnt[THREADS / WARP][MAX_ENTRIES];
    const u32 beta = blockIdx.x;
    const u32 tid = threadIdx.x, lane = lane_id(), warp = warp_id(), nwarps = blockDim.x / WARP;
    const u32 start = in_start[beta], len = in_len[beta];
    if (len == 0) return;
    // Entry e of this bucket lands in output bucket beta ^ bd[e], after that bucket's rows from the entries before e.
    if (tid < MAX_ENTRIES) {
        u32 base = 0;
        if (tid < E) {
            const u32 dst = beta ^ bd[tid];
            base = out_start[dst];
            for (u32 f = 0; f < tid; ++f) base += cnt[(size_t)(dst ^ bd[f]) * E + f];
        }
        s_base[tid] = base;
        s_run[tid] = 0;
    }
    __syncthreads();
    for (u32 r0 = 0; r0 < len; r0 += blockDim.x) {
        const u32 r = r0 + tid;
        Key k;
        u32 e = NO_ENTRY;
        double pr = 0.0, pi = 0.0;
        bool ok = false;
        if (r < len) ok = perm_row(T, entry_of, x, z, c, (size_t)start + r, prog, k, e, pr, pi);
        if (!ok) e = NO_ENTRY;
        // The previous tile's lanes read this warp's row of s_cnt before it is cleared.
        __syncwarp();
        if (lane < MAX_ENTRIES) s_cnt[warp][lane] = 0;
        __syncwarp();
        const u32 peers = __match_any_sync(~0u, e);
        const u32 rank = __popc(peers & lanemask_lt());
        if (ok && rank == 0) s_cnt[warp][e] = __popc(peers);
        __syncthreads();
        if (tid < MAX_ENTRIES) {
            u32 run = s_run[tid];
            for (u32 w = 0; w < nwarps; ++w) {
                const u32 v = s_cnt[w][tid];
                s_cnt[w][tid] = run;
                run += v;
            }
            s_run[tid] = run;
        }
        __syncthreads();
        if (ok) {
            const size_t dst = (size_t)s_base[e] + s_cnt[warp][e] + rank;
#pragma unroll
            for (int w = 0; w < W; ++w) {
                ox[dst * W + w] = k.x[w] ^ T.mask[(e * 2) * W + w];
                oz[dst * W + w] = k.z[w] ^ T.mask[(e * 2 + 1) * W + w];
            }
            oc[2 * dst] = pr;
            oc[2 * dst + 1] = pi;
            og[dst] = g[(size_t)start + r] ^ (gm[e] & fp_mask());
        }
    }
}
