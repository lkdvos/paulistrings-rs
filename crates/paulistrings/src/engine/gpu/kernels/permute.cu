// The permutation layer for a table under which no two input keys emit one output key (`DevicePrepared::permutation`).
// k_permute_count counts the surviving rows per (source bucket, entry), k_permute_lengths sizes every output bucket from those counts, and k_permute_scatter scatters each row to bucket beta ^ bucket_delta[e] with no sort, no arena and no compaction.
// Output bucket beta' holds, entry by entry, the survivors of source bucket beta' ^ bucket_delta[e] in source order; nothing reads that order.

// The row entry `e` emits for source row `src`, and whether it survives: `entry_of[s]` picks the entry, then the product and the keep program decide exactly as the fused layer's `survives`.
__device__ __forceinline__ bool permute_row(const Table& T, const u32* __restrict__ entry_of,
                                         const u64* __restrict__ x, const u64* __restrict__ z,
                                         const real_t* __restrict__ c, size_t src, const KeepProg& prog,
                                         Key& k, u32& e, Coeff& product) {
    k = load_key(x, z, src);
    const u32 s = support_bits(k, T.support_size, T.qubit_0, T.qubit_1);
    e = entry_of[s];
    if (e == NO_ENTRY) return false;
    product = entry_product(T, k, e, coeff_load(c, src));
    if (coeff_is_zero(product)) return false;
    return keep_eval(prog, [&]() -> Key {
        Key o = k;
#pragma unroll
        for (int w = 0; w < W; ++w) {
            o.x[w] ^= T.mask[(e * 2) * W + w];
            o.z[w] ^= T.mask[(e * 2 + 1) * W + w];
        }
        return o;
    }, product);
}

// k_permute_count: cnt[beta * E + e] = surviving rows of bucket beta that entry e emits, one block per source bucket at k_permute_scatter's width.
extern "C" __global__ void __launch_bounds__(THREADS) k_permute_count(
    const u64* __restrict__ x, const u64* __restrict__ z, const real_t* __restrict__ c,
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
        Coeff product;
        const bool ok = permute_row(T, entry_of, x, z, c, r, prog, k, e, product);
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

// k_permute_lengths: out_len[beta'] = sum over entries of cnt[(beta' ^ bucket_delta[e]) * E + e], one thread per output bucket.
extern "C" __global__ void k_permute_lengths(const u32* __restrict__ cnt, const u32* __restrict__ bucket_delta, u32 B, u32 E,
                                       u32* __restrict__ out_len) {
    const u32 beta = blockIdx.x * blockDim.x + threadIdx.x;
    if (beta >= B) return;
    u32 n = 0;
    for (u32 e = 0; e < E; ++e) n += cnt[(size_t)(beta ^ bucket_delta[e]) * E + e];
    out_len[beta] = n;
}

// k_permute_scatter: one block per source bucket, any multiple of 32 threads up to THREADS wide.
// Rows go in tiles of blockDim.x; within a tile a row's slot is its rank among the tile's rows of the same entry, from a warp match and a per-entry prefix over the warps, so the scatter is deterministic without atomics.
extern "C" __global__ void __launch_bounds__(THREADS) k_permute_scatter(
    const u64* __restrict__ x, const u64* __restrict__ z, const real_t* __restrict__ c, const u64* __restrict__ g,
    const u32* __restrict__ in_start, const u32* __restrict__ in_len, TABLE_ARGS,
    const u32* __restrict__ entry_of, const u32* __restrict__ bucket_delta, const u64* __restrict__ fingerprint_delta,
    const u32* __restrict__ cnt, const u32* __restrict__ out_start, const __grid_constant__ KeepProg prog,
    u64* __restrict__ ox, u64* __restrict__ oz, real_t* __restrict__ oc, u64* __restrict__ og) {
    MAKE_TABLE(T);
    __shared__ u32 s_base[MAX_ENTRIES];
    __shared__ u32 s_run[MAX_ENTRIES];
    __shared__ u32 s_cnt[THREADS / WARP][MAX_ENTRIES];
    const u32 beta = blockIdx.x;
    const u32 tid = threadIdx.x, lane = lane_id(), warp = warp_id(), nwarps = blockDim.x / WARP;
    const u32 start = in_start[beta], len = in_len[beta];
    if (len == 0) return;
    // Entry e of this bucket lands in output bucket beta ^ bucket_delta[e], after that bucket's rows from the entries before e.
    if (tid < MAX_ENTRIES) {
        u32 base = 0;
        if (tid < E) {
            const u32 dst = beta ^ bucket_delta[tid];
            base = out_start[dst];
            for (u32 f = 0; f < tid; ++f) base += cnt[(size_t)(dst ^ bucket_delta[f]) * E + f];
        }
        s_base[tid] = base;
        s_run[tid] = 0;
    }
    __syncthreads();
    for (u32 r0 = 0; r0 < len; r0 += blockDim.x) {
        const u32 r = r0 + tid;
        Key k;
        u32 e = NO_ENTRY;
        Coeff product = coeff_zero();
        bool ok = false;
        if (r < len) ok = permute_row(T, entry_of, x, z, c, (size_t)start + r, prog, k, e, product);
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
            coeff_store(oc, dst, product);
            og[dst] = g[(size_t)start + r] ^ (fingerprint_delta[e] & fingerprint_mask());
        }
    }
}

// A remote permutation layer's received rows, already shifted and multiplied by their sender, follow the local rows of their output bucket: a global permutation never sends a key a local row also reaches.
// recv_off is `K × (B + 1)` CSR offsets by destination position, block k's rows from rbase[k]; one warp per position.

// The received rows of position `p` that survive, counted (`out` null) or written after `local_len[beta]` rows of output bucket beta.
__device__ __forceinline__ u32 received_rows(const u32* __restrict__ recv_off, const u32* __restrict__ rbase, u32 K, u32 B,
                                             u32 p, const u64* __restrict__ rx, const u64* __restrict__ rz,
                                             const real_t* __restrict__ rc, const u64* __restrict__ rg,
                                             const KeepProg& prog, size_t dst, u64* __restrict__ ox,
                                             u64* __restrict__ oz, real_t* __restrict__ oc, u64* __restrict__ og) {
    const u32 lane = lane_id();
    u32 taken = 0;
    for (u32 k = 0; k < K; ++k) {
        const size_t o = (size_t)k * (B + 1) + p;
        const u32 lo = recv_off[o], len = recv_off[o + 1] - lo;
        const size_t first = (size_t)rbase[k] + lo;
        for (u32 r0 = 0; r0 < len; r0 += WARP) {
            const u32 r = r0 + lane;
            bool ok = false;
            Key key;
            Coeff c = coeff_zero();
            if (r < len) {
                key = load_key(rx, rz, first + r);
                c = coeff_load(rc, first + r);
                ok = survives(prog, key, c);
            }
            const u32 bal = __ballot_sync(~0u, ok);
            if (ox != nullptr && ok) {
                const size_t d = dst + taken + __popc(bal & lanemask_lt());
                store_key(ox, oz, d, key);
                coeff_store(oc, d, c);
                og[d] = rg[first + r];
            }
            taken += __popc(bal);
        }
    }
    return taken;
}

// k_permute_received_lengths: out_len[beta] = local_len[beta] plus the surviving received rows of beta's position.
extern "C" __global__ void k_permute_received_lengths(const u32* __restrict__ recv_off, const u32* __restrict__ rbase, u32 K,
                                                      u32 B, const u64* __restrict__ rx, const u64* __restrict__ rz,
                                                      const real_t* __restrict__ rc, const __grid_constant__ KeepProg prog,
                                                      const u32* __restrict__ bucket_at, const u32* __restrict__ local_len,
                                                      u32* __restrict__ out_len) {
    const u32 p = blockIdx.x * (blockDim.x / WARP) + warp_id();
    if (p >= B) return;
    const u32 n = received_rows(recv_off, rbase, K, B, p, rx, rz, rc, nullptr, prog, 0, nullptr, nullptr, nullptr, nullptr);
    if (lane_id() == 0) {
        const u32 beta = bucket_at[p];
        out_len[beta] = local_len[beta] + n;
    }
}

// k_permute_received_scatter: the surviving received rows of each position, in block then row order, after its bucket's local rows.
extern "C" __global__ void k_permute_received_scatter(const u32* __restrict__ recv_off, const u32* __restrict__ rbase, u32 K,
                                                      u32 B, const u64* __restrict__ rx, const u64* __restrict__ rz,
                                                      const real_t* __restrict__ rc, const u64* __restrict__ rg,
                                                      const __grid_constant__ KeepProg prog, const u32* __restrict__ bucket_at,
                                                      const u32* __restrict__ local_len, const u32* __restrict__ out_start,
                                                      u64* __restrict__ ox, u64* __restrict__ oz, real_t* __restrict__ oc,
                                                      u64* __restrict__ og) {
    const u32 p = blockIdx.x * (blockDim.x / WARP) + warp_id();
    if (p >= B) return;
    const u32 beta = bucket_at[p];
    const size_t dst = (size_t)out_start[beta] + local_len[beta];
    received_rows(recv_off, rbase, K, B, p, rx, rz, rc, rg, prog, dst, ox, oz, oc, og);
}
