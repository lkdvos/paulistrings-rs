// `total` gathers `[output terms, longest output bucket]`; with `histogram`, `hist` gathers `k_octave_histogram`'s `[len, bins…]` of the output from a shared histogram after the table, two 16-bit bins a word.
// k_join: the fused layer of a two-entry table with one identity-mask entry (every rotation), one block per output position, writing its bucket straight into the output columns at the position's pre-dedup offset; contracts on `JoinTable` in prepared.rs.
// The base entry's rows of bucket beta are unique keys, and so are the partner entry's rows from beta ^ delta, so the only merge is a partner landing on a base row: a shared open-addressing table over the hittable base rows joins them, and every other row is written as it is read.

// A slot is 0 (empty), (tag:18 | base row + 1:13), or once a partner hits it (1:1 | partner row + 1:13 | base row + 1:13).
#define JOIN_INDEX_BITS 13u
#define JOIN_INDEX_MASK ((1u << JOIN_INDEX_BITS) - 1u)
#define JOIN_TAG_MASK 0x3FFFFu
#define JOIN_HIT 0x80000000u
static_assert(MAX_BUCKET_LEN < (1u << JOIN_INDEX_BITS), "a row index plus one fits a slot field");
static_assert(2 * MAX_BUCKET_LEN < (1u << 16), "a block's output rows fit a 16-bit bin");

__device__ __forceinline__ u32 join_home(u64 f, u32 table_bits) {
    const u32 h = ((u32)f ^ (u32)(f >> 32)) * 0x9E3779B1u;
    return table_bits == 0 ? 0u : h >> (32u - table_bits);
}

__device__ __forceinline__ u32 join_tag(u64 f) { return (u32)(f >> 46) & JOIN_TAG_MASK; }

// One output row of the block, claimed in warp order, and its bin in the block's histogram.
__device__ __forceinline__ void join_emit(u32* s_out, bool keep, u32* s_hist, u32 histogram, u32& span_lo,
                                          u32& span_hi, u64* __restrict__ out_x, u64* __restrict__ out_z,
                                          real_t* __restrict__ out_c, u64* __restrict__ out_g, size_t out0, const Key& k,
                                          u64 f, Coeff v) {
    const size_t slot = out0 + warp_claim(s_out, keep);
    if (!keep) return;
    if (histogram) {
        const u32 bin = approx_bin(v);
        atomicAdd(&s_hist[bin >> 1], 1u << (16 * (bin & 1)));
        span_lo = min(span_lo, bin >> 1);
        span_hi = max(span_hi, bin >> 1);
    }
    store_key(out_x, out_z, slot, k);
    out_g[slot] = f;
    coeff_store(out_c, slot, v);
}

// k_join_rows: rows[p] = the lengths of the two source buckets position p reads, which bound its output, so sizing a join reads no key; it zeroes `k_join`'s `total`.
extern "C" __global__ void k_join_rows(const u32* __restrict__ in_len, const u32* __restrict__ bucket_at,
                                       const u32* __restrict__ bucket_delta, u32* __restrict__ rows, u32 B,
                                       u32* __restrict__ total) {
    const u32 p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < 2) total[p] = 0;
    if (p >= B) return;
    const u32 beta = bucket_at[p];
    rows[p] = in_len[beta ^ bucket_delta[0]] + in_len[beta ^ bucket_delta[1]];
}

extern "C" __global__ void __launch_bounds__(JOIN_THREADS) k_join(
    const u64* __restrict__ x, const u64* __restrict__ z, const real_t* __restrict__ c, const u64* __restrict__ g,
    const u32* __restrict__ in_start, const u32* __restrict__ in_len, const u32* __restrict__ bucket_at,
    const u32* __restrict__ seg_start, u32 mode, u32 support_size, u32 qubit_0, u32 qubit_1, double rotation_cos,
    double rotation_sin, const real_t* __restrict__ amplitude, const u64* __restrict__ mask,
    const u32* __restrict__ nonzero_patterns, const u32* __restrict__ bucket_delta,
    const u64* __restrict__ fingerprint_delta, u32 base_entry, u32 hit_patterns, u32 launch_bits,
    const __grid_constant__ KeepProg prog, u64* __restrict__ out_x, u64* __restrict__ out_z,
    real_t* __restrict__ out_c, u64* __restrict__ out_g, u32* __restrict__ out_start, u32* __restrict__ out_len,
    u32* __restrict__ total, u32 histogram, u64* __restrict__ hist) {
    extern __shared__ __align__(16) u32 slots[];
    __shared__ real_t s_amplitude[2 * LOCAL_DIM * COEFF_PARTS];
    __shared__ u64 s_mask[4 * W];
    __shared__ u32 s_nonzero[2];
    __shared__ u32 s_out;
    __shared__ u32 s_span[2];
    const u32 tid = threadIdx.x, lane = lane_id(), warp = warp_id();
    const u32 warps = JOIN_THREADS / WARP;
    const u32 p = blockIdx.x;
    const u32 beta = bucket_at[p];
    const u32 seg0 = seg_start[p];
    if (seg_start[p + 1] == seg0) {
        if (tid == 0) {
            out_start[beta] = seg0;
            out_len[beta] = 0;
        }
        return;
    }
    const u32 partner_entry = base_entry ^ 1u;
    const u32 base_start = in_start[beta], base_len = in_len[beta];
    // A block's table holds its own base bucket, at most half full, within the launch's `launch_bits`.
    const u32 table_bits = min(launch_bits, 32u - __clz(2u * max(base_len, 16u) - 1u));
    const u32 n_slots = 1u << table_bits;
    u32* const s_hist = slots + (1u << launch_bits);
    for (u32 i = tid; i < n_slots; i += JOIN_THREADS) slots[i] = 0;
    if (histogram) {
        for (u32 i = tid; i < APPROX_BINS / 8; i += JOIN_THREADS) reinterpret_cast<uint4*>(s_hist)[i] = make_uint4(0, 0, 0, 0);
    }
    // The histogram words the block's rows touch, so the fold reads only those.
    u32 span_lo = ~0u, span_hi = 0;
    if (mode == MODE_LOCAL) for (u32 i = tid; i < 2 * LOCAL_DIM * COEFF_PARTS; i += JOIN_THREADS) s_amplitude[i] = amplitude[i];
    for (u32 i = tid; i < 4 * W; i += JOIN_THREADS) s_mask[i] = mask[i];
    if (tid < 2) s_nonzero[tid] = nonzero_patterns[tid];
    if (tid == 0) {
        s_out = 0;
        s_span[0] = ~0u;
        s_span[1] = 0;
    }
    __syncthreads();
    Table T;
    T.mode = mode;
    T.entries = 2;
    T.support_size = support_size;
    T.qubit_0 = qubit_0;
    T.qubit_1 = qubit_1;
    T.rotation_cos = rotation_cos;
    T.rotation_sin = rotation_sin;
    T.amplitude = s_amplitude;
    T.mask = s_mask;
    T.nonzero_patterns = s_nonzero;
    // A rotation's partner rows anticommute, as do the base rows they can land on; a local table's land on its output patterns.
    auto hittable = [&](const Key& k) -> bool {
        if (mode == MODE_LOCAL) return ((hit_patterns >> support_bits(k, support_size, qubit_0, qubit_1)) & 1u) != 0;
        return anticommutes_generator(T, k);
    };
    const size_t out0 = seg0;

    // Base rows: a hittable one enters the table, every other one is final.
    for (u32 r0 = warp * WARP; r0 < base_len; r0 += warps * WARP) {
        const u32 r = r0 + lane;
        bool emit = false, insert = false;
        Key k;
        if (r < base_len) {
            k = load_key(x, z, base_start + r);
            emit = entry_emits(T, k, base_entry);
            insert = emit && hittable(k);
        }
        bool keep = false;
        Coeff v = coeff_zero();
        u64 f = 0;
        if (emit && !insert) {
            v = entry_product(T, k, base_entry, coeff_load(c, base_start + r));
            f = g[base_start + r];
            keep = survives(prog, k, v);
        }
        if (insert) {
            const u64 fi = g[base_start + r];
            const u32 entry = (join_tag(fi) << JOIN_INDEX_BITS) | (r + 1u);
            u32 s = join_home(fi, table_bits);
            while (atomicCAS(&slots[s], 0u, entry) != 0u) s = (s + 1u) & (n_slots - 1u);
        }
        join_emit(&s_out, keep, s_hist, histogram, span_lo, span_hi, out_x, out_z, out_c, out_g, out0, k, f, v);
    }
    __syncthreads();

    // Partner rows: a hit marks its slot, a miss is a new key, written as read.
    const u32 sb = beta ^ bucket_delta[partner_entry];
    const u32 partner_start = in_start[sb], partner_len = in_len[sb];
    const u64 fd = fingerprint_delta[partner_entry] & fingerprint_mask();
    for (u32 r0 = warp * WARP; r0 < partner_len; r0 += warps * WARP) {
        const u32 r = r0 + lane;
        bool emit = false;
        Key q;
        if (r < partner_len) {
            q = load_key(x, z, partner_start + r);
            emit = entry_emits(T, q, partner_entry);
        }
        bool miss = false;
        u64 f = 0;
        Key k = q;
        if (emit) {
#pragma unroll
            for (int w = 0; w < W; ++w) {
                q.x[w] ^= s_mask[(partner_entry * 2) * W + w];
                q.z[w] ^= s_mask[(partner_entry * 2 + 1) * W + w];
            }
            f = g[partner_start + r] ^ fd;
            const u32 tag = join_tag(f);
            u32 s = join_home(f, table_bits);
            miss = true;
            for (;;) {
                const u32 e = slots[s];
                if (e == 0u) break;
                if ((e & JOIN_HIT) == 0u && (e >> JOIN_INDEX_BITS) == tag) {
                    const u32 b = (e & JOIN_INDEX_MASK) - 1u;
                    if (key_eq(load_key(x, z, base_start + b), q)) {
                        slots[s] = JOIN_HIT | ((r + 1u) << JOIN_INDEX_BITS) | (b + 1u);
                        miss = false;
                        break;
                    }
                }
                s = (s + 1u) & (n_slots - 1u);
            }
        }
        bool keep = false;
        Coeff v = coeff_zero();
        if (miss) {
            v = entry_product(T, k, partner_entry, coeff_load(c, partner_start + r));
            keep = survives(prog, q, v);
        }
        join_emit(&s_out, keep, s_hist, histogram, span_lo, span_hi, out_x, out_z, out_c, out_g, out0, q, f, v);
    }
    __syncthreads();

    // The table's base rows, each summed with its partner if one hit.
    for (u32 s0 = warp * WARP; s0 < n_slots; s0 += warps * WARP) {
        const u32 e = slots[s0 + lane];
        bool keep = false;
        Key k;
        Coeff v = coeff_zero();
        u64 f = 0;
        if (e != 0u) {
            const u32 b = base_start + (e & JOIN_INDEX_MASK) - 1u;
            k = load_key(x, z, b);
            f = g[b];
            v = entry_product(T, k, base_entry, coeff_load(c, b));
            if (e & JOIN_HIT) {
                const u32 r = partner_start + ((e >> JOIN_INDEX_BITS) & JOIN_INDEX_MASK) - 1u;
                v = coeff_add(v, entry_product(T, load_key(x, z, r), partner_entry, coeff_load(c, r)));
            }
            keep = survives(prog, k, v);
        }
        join_emit(&s_out, keep, s_hist, histogram, span_lo, span_hi, out_x, out_z, out_c, out_g, out0, k, f, v);
    }
    if (histogram) {
#pragma unroll
        for (u32 o = 16; o > 0; o >>= 1) {
            span_lo = min(span_lo, __shfl_xor_sync(~0u, span_lo, o));
            span_hi = max(span_hi, __shfl_xor_sync(~0u, span_hi, o));
        }
        if (lane == 0 && span_lo <= span_hi) {
            atomicMin(&s_span[0], span_lo);
            atomicMax(&s_span[1], span_hi);
        }
    }
    __syncthreads();
    if (histogram) {
        const u32 lo = s_span[0], hi = s_span[1];
        for (u32 i = lo + tid; lo <= hi && i <= hi; i += JOIN_THREADS) {
            const u32 pair = s_hist[i];
            if ((pair & 0xFFFFu) != 0) atomicAdd(&hist[1 + 2 * i], (u64)(pair & 0xFFFFu));
            if ((pair >> 16) != 0) atomicAdd(&hist[2 + 2 * i], (u64)(pair >> 16));
        }
        if (tid == 0 && s_out != 0) atomicAdd(&hist[0], (u64)s_out);
    }
    if (tid == 0) {
        out_start[beta] = seg0;
        out_len[beta] = s_out;
        if (s_out != 0) {
            atomicAdd(&total[0], s_out);
            atomicMax(&total[1], s_out);
        }
    }
}
