// A run of rotation tables applied layer by layer inside shared memory, `keep` asked after every layer; launch and contracts in engine/gpu/layer/rotation_run.rs.
// A string's class is its key with the run's qubits cleared: every layer moves a string within its class, so classes are independent and one block holds whole classes.
// k_rotation_run_classify counts each touched string into its class bucket and each untouched survivor into its bucket, k_rotation_run_gather lists the touched strings by class bucket,
// k_rotation_run_apply runs every layer over one class bucket and appends the survivors to the arena, and k_rotation_run_place / k_rotation_run_scatter write the tight output.

#if !defined(RR_ROWS) || !defined(RR_THREADS)
#error "RR_ROWS and RR_THREADS must be defined"
#endif
#define RR_MAX_LAYERS 8
#define RR_MAX_QUBITS 16
#define RR_BITS (2 * RR_MAX_QUBITS)
// An empty slot; a slot otherwise holds a row index plus one.
#define RR_EMPTY 0u

// `RotationRunInfo` in rotation_run.rs, passed by value.
struct RunInfo {
    u32 layers;
    u32 qubits_len;
    u32 qubits[RR_MAX_QUBITS];
    // Per layer: the pattern shift of the table's first and second qubit (equal for a one-qubit table), the generator's flip in pattern bits,
    // the table's support patterns the generator reaches, and whether the identity entry is a rounding residue there.
    u32 low[RR_MAX_LAYERS];
    u32 high[RR_MAX_LAYERS];
    u32 two[RR_MAX_LAYERS];
    u32 flip[RR_MAX_LAYERS];
    u32 active[RR_MAX_LAYERS];
    u32 clifford[RR_MAX_LAYERS];
    // h and g of each pattern bit's key bit.
    u32 bucket[RR_BITS];
    u64 fingerprint[RR_BITS];
};
static_assert(sizeof(RunInfo) == 648, "RunInfo matches RotationRunInfo");

__device__ __forceinline__ u32 rr_bit(const u64 (&v)[W], u32 q) {
    u64 word = 0;
#pragma unroll
    for (int w = 0; w < W; ++w) word = ((u32)w == (q >> 6)) ? v[w] : word;
    return (u32)((word >> (q & 63)) & 1ull);
}

// Bit 2j is the x-bit of run qubit j, bit 2j+1 its z-bit.
__device__ __forceinline__ u32 rr_pattern(const RunInfo& R, const Key& k) {
    u32 p = 0;
    for (u32 j = 0; j < R.qubits_len; ++j) {
        const u32 q = R.qubits[j];
        p |= rr_bit(k.x, q) << (2 * j) | rr_bit(k.z, q) << (2 * j + 1);
    }
    return p;
}

// The table's support pattern of run pattern p, packed like `LocalPtm::support_bits`.
__device__ __forceinline__ u32 rr_support(const RunInfo& R, u32 i, u32 p) {
    const u32 s = (p >> R.low[i]) & 3u;
    return R.two[i] ? s | ((p >> R.high[i]) & 3u) << 2 : s;
}

__device__ __forceinline__ bool rr_touched(const RunInfo& R, u32 p) {
    for (u32 i = 0; i < R.layers; ++i) {
        if ((R.active[i] >> rr_support(R, i, p)) & 1u) return true;
    }
    return false;
}

__device__ __forceinline__ u64 rr_fingerprint(const RunInfo& R, u32 d) {
    u64 f = 0;
    for (u32 b = 0; b < 2 * R.qubits_len; ++b) f ^= ((d >> b) & 1u) ? R.fingerprint[b] : 0ull;
    return f & fingerprint_mask();
}

__device__ __forceinline__ u32 rr_bucket(const RunInfo& R, u32 d) {
    u32 h = 0;
    for (u32 b = 0; b < 2 * R.qubits_len; ++b) h ^= ((d >> b) & 1u) ? R.bucket[b] : 0u;
    return h;
}

// splitmix64's finalizer, so a class bucket draws on every fingerprint bit.
__device__ __forceinline__ u64 rr_mix(u64 v) {
    v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
    v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
    return v ^ (v >> 31);
}

__device__ __forceinline__ u32 rr_class_bucket(u64 class_fingerprint, u32 class_bits) {
    return class_bits == 0 ? 0u : (u32)(rr_mix(class_fingerprint) >> (64u - class_bits));
}

// The key with the run's qubits cleared, then set from pattern p.
__device__ __forceinline__ Key rr_with_pattern(const RunInfo& R, Key k, u32 p) {
    for (u32 j = 0; j < R.qubits_len; ++j) {
        const u32 q = R.qubits[j];
        const u64 bit = 1ull << (q & 63);
#pragma unroll
        for (int w = 0; w < W; ++w) {
            if ((u32)w == (q >> 6)) {
                k.x[w] = (k.x[w] & ~bit) | (((p >> (2 * j)) & 1u) ? bit : 0ull);
                k.z[w] = (k.z[w] & ~bit) | (((p >> (2 * j + 1)) & 1u) ? bit : 0ull);
            }
        }
    }
    return k;
}

__device__ __forceinline__ bool rr_same_class(const RunInfo& R, const Key& a, const Key& b) {
    return key_eq(rr_with_pattern(R, a, 0), rr_with_pattern(R, b, 0));
}

__device__ __forceinline__ bool rr_survives(const KeepProg& prog, const Key& k, Coeff c) {
    return !coeff_is_zero(c) && keep_eval(prog, [&]() -> Key { return k; }, c);
}

// k_rotation_run_classify: per bucket blockIdx.x, class_count[class bucket] += its touched strings and inert[beta] = its untouched strings that `keep` passes.
extern "C" __global__ void k_rotation_run_classify(
    const u64* __restrict__ x, const u64* __restrict__ z, const real_t* __restrict__ c, const u64* __restrict__ g,
    const u32* __restrict__ in_start, const u32* __restrict__ in_len, const __grid_constant__ RunInfo info,
    const __grid_constant__ KeepProg prog, u32 class_bits, u32* __restrict__ class_count, u32* __restrict__ inert) {
    __shared__ u32 s_inert;
    const u32 beta = blockIdx.x;
    const u32 start = in_start[beta], len = in_len[beta];
    if (threadIdx.x == 0) s_inert = 0;
    __syncthreads();
    u32 kept = 0;
    for (u32 r0 = 0; r0 < len; r0 += blockDim.x) {
        const u32 r = r0 + threadIdx.x;
        u32 cb = NO_ENTRY;
        if (r < len) {
            const Key k = load_key(x, z, (size_t)start + r);
            const u32 p = rr_pattern(info, k);
            if (rr_touched(info, p)) {
                cb = rr_class_bucket((g[(size_t)start + r] ^ rr_fingerprint(info, p)) & fingerprint_mask(), class_bits);
            } else {
                kept += rr_survives(prog, k, coeff_load(c, (size_t)start + r)) ? 1u : 0u;
            }
        }
        const u32 peers = __match_any_sync(~0u, cb);
        if (cb != NO_ENTRY && (peers & lanemask_lt()) == 0) atomicAdd(&class_count[cb], (u32)__popc(peers));
    }
    if (kept) atomicAdd(&s_inert, kept);
    __syncthreads();
    if (threadIdx.x == 0) inert[beta] = s_inert;
}

// k_rotation_run_gather: every touched string of bucket blockIdx.x, with its bucket, at a slot claimed from its class bucket's cursor.
extern "C" __global__ void k_rotation_run_gather(
    const u64* __restrict__ x, const u64* __restrict__ z, const real_t* __restrict__ c, const u64* __restrict__ g,
    const u32* __restrict__ in_start, const u32* __restrict__ in_len, const __grid_constant__ RunInfo info, u32 class_bits,
    u32* __restrict__ cursor, u64* __restrict__ ox, u64* __restrict__ oz, real_t* __restrict__ oc, u64* __restrict__ og,
    u32* __restrict__ obucket) {
    const u32 beta = blockIdx.x;
    const u32 start = in_start[beta], len = in_len[beta];
    for (u32 r0 = 0; r0 < len; r0 += blockDim.x) {
        const u32 r = r0 + threadIdx.x;
        u32 cb = NO_ENTRY;
        Key k;
        u64 f = 0;
        if (r < len) {
            k = load_key(x, z, (size_t)start + r);
            const u32 p = rr_pattern(info, k);
            if (rr_touched(info, p)) {
                f = g[(size_t)start + r];
                cb = rr_class_bucket((f ^ rr_fingerprint(info, p)) & fingerprint_mask(), class_bits);
            }
        }
        const u32 peers = __match_any_sync(~0u, cb);
        const u32 leader = __ffs(peers) - 1;
        u32 base = 0;
        if (cb != NO_ENTRY && lane_id() == leader) base = atomicAdd(&cursor[cb], (u32)__popc(peers));
        base = __shfl_sync(~0u, base, leader);
        if (cb != NO_ENTRY) {
            const size_t slot = (size_t)base + __popc(peers & lanemask_lt());
#pragma unroll
            for (int w = 0; w < W; ++w) {
                ox[slot * W + w] = k.x[w];
                oz[slot * W + w] = k.z[w];
            }
            coeff_store(oc, slot, coeff_load(c, (size_t)start + r));
            og[slot] = f;
            obucket[slot] = beta;
        }
    }
}

__device__ __forceinline__ u32 rr_home(u64 key, u32 slots) { return (u32)(rr_mix(key) >> 32) & (slots - 1u); }

__device__ __forceinline__ u64 rr_row_key(u32 cls, u32 p) { return (u64)cls << 32 | p; }

// The row of class `cls` at pattern `p`, or RR_ROWS when absent.
__device__ __forceinline__ u32 rr_find(const volatile u32* slots, u32 n_slots, const u16* cls_of, const u32* pat, u32 cls, u32 p) {
    u32 s = rr_home(rr_row_key(cls, p), n_slots);
    for (;;) {
        const u32 e = slots[s];
        if (e == RR_EMPTY) return RR_ROWS;
        if (cls_of[e - 1] == cls && pat[e - 1] == p) return e - 1;
        s = (s + 1u) & (n_slots - 1u);
    }
}

__device__ __forceinline__ void rr_insert(u32* slots, u32 n_slots, u32 cls, u32 p, u32 row) {
    u32 s = rr_home(rr_row_key(cls, p), n_slots);
    while (atomicCAS(&slots[s], RR_EMPTY, row + 1u) != RR_EMPTY) s = (s + 1u) & (n_slots - 1u);
}

// Shared bytes of k_rotation_run_apply: two coefficient columns, the class fingerprints of the input half, patterns, row and class tables, and class indices.
#define RR_SHARED_BYTES (2u * RR_ROWS * COEFF_PARTS * sizeof(real_t) + (RR_ROWS / 2u) * 8u + RR_ROWS * 4u + 2u * RR_ROWS * 4u + RR_ROWS * 4u + RR_ROWS * 2u)

// k_rotation_run_apply: class bucket blockIdx.x, rows [class_start[cb], class_start[cb + 1]) of the gathered columns, through every layer;
// each layer sums a string's identity part and its partner's generator part, and asks `keep` about every string it changed (about every string after the first layer);
// survivors go to the arena at a block slot claimed from counters[0], and total[beta'] counts each into its output bucket.
// counters[1] is set when a class bucket outgrows the block, counters[0] past arena_cap means the arena was too small; either way nothing past the cap is written.
extern "C" __global__ void __launch_bounds__(RR_THREADS) k_rotation_run_apply(
    const u64* __restrict__ x, const u64* __restrict__ z, const real_t* __restrict__ c, const u64* __restrict__ g,
    const u32* __restrict__ row_bucket, const u32* __restrict__ class_start, const __grid_constant__ RunInfo info,
    const real_t* __restrict__ amplitude, const __grid_constant__ KeepProg prog, u64* __restrict__ ax, u64* __restrict__ az,
    real_t* __restrict__ ac, u64* __restrict__ ag, u32* __restrict__ abucket, u32 arena_cap, u32* __restrict__ counters,
    u32* __restrict__ total) {
    extern __shared__ __align__(16) unsigned char rr_shared[];
    __shared__ real_t s_amp[RR_MAX_LAYERS * 2 * LOCAL_DIM * COEFF_PARTS];
    __shared__ u32 s_rows, s_out, s_base, s_overflow;
    const u32 cb = blockIdx.x;
    const u32 seg0 = class_start[cb];
    const u32 n_in = class_start[cb + 1] - seg0;
    if (n_in == 0) return;
    if (n_in > RR_ROWS / 2) {
        if (threadIdx.x == 0) atomicOr(&counters[1], 1u);
        return;
    }
    real_t* coef = (real_t*)rr_shared;
    real_t* coef_new = coef + RR_ROWS * COEFF_PARTS;
    u64* class_fp = (u64*)(coef_new + RR_ROWS * COEFF_PARTS);
    u32* pat = (u32*)(class_fp + RR_ROWS / 2);
    u32* row_slots = pat + RR_ROWS;
    u32* class_slots = row_slots + 2 * RR_ROWS;
    u16* cls_of = (u16*)(class_slots + RR_ROWS);
    // At most half full once every row is in.
    u32 row_n_slots = 64;
    while (row_n_slots < 8u * n_in && row_n_slots < 2u * RR_ROWS) row_n_slots <<= 1;
    const u32 row_cap = min((u32)RR_ROWS, row_n_slots / 2u);
    u32 class_n_slots = 32;
    while (class_n_slots < 2u * n_in) class_n_slots <<= 1;
    const u32 layers = info.layers;
    for (u32 i = threadIdx.x; i < row_n_slots; i += blockDim.x) row_slots[i] = RR_EMPTY;
    for (u32 i = threadIdx.x; i < class_n_slots; i += blockDim.x) class_slots[i] = RR_EMPTY;
    for (u32 i = threadIdx.x; i < layers * 2 * LOCAL_DIM * COEFF_PARTS; i += blockDim.x) s_amp[i] = amplitude[i];
    if (threadIdx.x == 0) {
        s_rows = n_in;
        s_out = 0;
        s_overflow = 0;
    }
    for (u32 r = threadIdx.x; r < n_in; r += blockDim.x) {
        const Key k = load_key(x, z, (size_t)seg0 + r);
        const u32 p = rr_pattern(info, k);
        pat[r] = p;
        class_fp[r] = (g[(size_t)seg0 + r] ^ rr_fingerprint(info, p)) & fingerprint_mask();
        coeff_store(coef, r, coeff_load(c, (size_t)seg0 + r));
    }
    __syncthreads();
    // A class is named by its first row to claim a slot; equal fingerprints are confirmed on the cleared keys.
    for (u32 r = threadIdx.x; r < n_in; r += blockDim.x) {
        const u64 f = class_fp[r];
        u32 s = rr_home(f, class_n_slots);
        u32 cls = r;
        Key k;
        bool loaded = false;
        for (;;) {
            u32 e = ((volatile u32*)class_slots)[s];
            if (e == RR_EMPTY) {
                e = atomicCAS(&class_slots[s], RR_EMPTY, r + 1u);
                if (e == RR_EMPTY) break;
            }
            const u32 other = e - 1u;
            if (class_fp[other] == f) {
                if (!loaded) {
                    k = load_key(x, z, (size_t)seg0 + r);
                    loaded = true;
                }
                if (rr_same_class(info, k, load_key(x, z, (size_t)seg0 + other))) {
                    cls = other;
                    break;
                }
            }
            s = (s + 1u) & (class_n_slots - 1u);
        }
        cls_of[r] = (u16)cls;
    }
    __syncthreads();
    for (u32 r = threadIdx.x; r < n_in; r += blockDim.x) rr_insert(row_slots, row_n_slots, cls_of[r], pat[r], r);
    __syncthreads();

    for (u32 i = 0; i < layers; ++i) {
        const u32 n0 = s_rows;
        const u32 flip = info.flip[i], active = info.active[i];
        const bool clifford = info.clifford[i] != 0;
        const real_t* a0 = s_amp + (2 * i) * LOCAL_DIM * COEFF_PARTS;
        const real_t* a1 = s_amp + (2 * i + 1) * LOCAL_DIM * COEFF_PARTS;
        for (u32 r = threadIdx.x; r < n0; r += blockDim.x) {
            const u32 p = pat[r];
            const u32 s = rr_support(info, i, p);
            if (((active >> s) & 1u) == 0) continue;
            const u32 cls = cls_of[r];
            const u32 q = p ^ flip;
            const Coeff v = coeff_load(coef, r);
            Coeff sum = clifford ? coeff_zero() : coeff_mul(v, coeff_load(a0, s));
            const u32 j = rr_find(row_slots, row_n_slots, cls_of, pat, cls, q);
            if (j != RR_ROWS) {
                sum = coeff_add(sum, coeff_mul(coeff_load(coef, j), coeff_load(a1, rr_support(info, i, q))));
            } else {
                const Coeff partner = coeff_mul(v, coeff_load(a1, s));
                if (!coeff_is_zero(partner)) {
                    const u32 t = atomicAdd(&s_rows, 1u);
                    if (t < row_cap) {
                        pat[t] = q;
                        cls_of[t] = (u16)cls;
                        coeff_store(coef_new, t, partner);
                        __threadfence_block();
                        rr_insert(row_slots, row_n_slots, cls, q, t);
                    } else {
                        s_overflow = 1;
                    }
                }
            }
            coeff_store(coef_new, r, sum);
        }
        __syncthreads();
        if (s_overflow) {
            if (threadIdx.x == 0) atomicOr(&counters[1], 1u);
            return;
        }
        const u32 n1 = s_rows;
        for (u32 r = threadIdx.x; r < n1; r += blockDim.x) {
            const u32 p = pat[r];
            const bool changed = r >= n0 || ((active >> rr_support(info, i, p)) & 1u) != 0;
            if (!changed && i > 0) continue;
            const Coeff v = changed ? coeff_load(coef_new, r) : coeff_load(coef, r);
            bool keep = !coeff_is_zero(v);
            if (keep && !(prog.len == 1 && prog.op[0] == OP_KEEP)) {
                const u32 rep = cls_of[r];
                keep = keep_eval(prog, [&]() -> Key { return rr_with_pattern(info, load_key(x, z, (size_t)seg0 + rep), p); }, v);
            }
            coeff_store(coef, r, keep ? v : coeff_zero());
        }
        __syncthreads();
    }

    const u32 n = s_rows;
    u32 mine = 0;
    for (u32 r = threadIdx.x; r < n; r += blockDim.x) mine += coeff_is_zero(coeff_load(coef, r)) ? 0u : 1u;
    if (mine) atomicAdd(&s_out, mine);
    __syncthreads();
    if (threadIdx.x == 0) {
        s_base = atomicAdd(&counters[0], s_out);
        s_out = 0;
    }
    __syncthreads();
    const u32 base = s_base;
    for (u32 r0 = 0; r0 < n; r0 += blockDim.x) {
        const u32 r = r0 + threadIdx.x;
        const Coeff v = r < n ? coeff_load(coef, r) : coeff_zero();
        const bool live = r < n && !coeff_is_zero(v);
        const u32 bal = __ballot_sync(~0u, live);
        u32 o = 0;
        if (lane_id() == 0 && bal != 0) o = atomicAdd(&s_out, (u32)__popc(bal));
        o = __shfl_sync(~0u, o, 0) + __popc(bal & lanemask_lt());
        if (!live) continue;
        const u32 rep = cls_of[r];
        const u32 p = pat[r];
        const u32 d = pat[rep] ^ p;
        const u32 beta = row_bucket[(size_t)seg0 + rep] ^ rr_bucket(info, d);
        atomicAdd(&total[beta], 1u);
        const size_t slot = (size_t)base + o;
        if (slot >= arena_cap) continue;
        const Key k = rr_with_pattern(info, load_key(x, z, (size_t)seg0 + rep), p);
#pragma unroll
        for (int w = 0; w < W; ++w) {
            ax[slot * W + w] = k.x[w];
            az[slot * W + w] = k.z[w];
        }
        coeff_store(ac, slot, v);
        ag[slot] = (g[(size_t)seg0 + rep] ^ rr_fingerprint(info, d)) & fingerprint_mask();
        abucket[slot] = beta;
    }
}

// k_rotation_run_place: bucket blockIdx.x's untouched survivors at the front of its output, its output length, and cursor[beta] past them for k_rotation_run_scatter.
extern "C" __global__ void k_rotation_run_place(
    const u64* __restrict__ x, const u64* __restrict__ z, const real_t* __restrict__ c, const u64* __restrict__ g,
    const u32* __restrict__ in_start, const u32* __restrict__ in_len, const __grid_constant__ RunInfo info,
    const __grid_constant__ KeepProg prog, const u32* __restrict__ inert, const u32* __restrict__ total,
    const u32* __restrict__ out_start, u32* __restrict__ out_len, u32* __restrict__ cursor, u64* __restrict__ ox,
    u64* __restrict__ oz, real_t* __restrict__ oc, u64* __restrict__ og) {
    __shared__ u32 s_next;
    const u32 beta = blockIdx.x;
    const u32 start = in_start[beta], len = in_len[beta];
    const u32 dst0 = out_start[beta];
    if (threadIdx.x == 0) {
        s_next = 0;
        out_len[beta] = total[beta];
        cursor[beta] = dst0 + inert[beta];
    }
    __syncthreads();
    for (u32 r0 = 0; r0 < len; r0 += blockDim.x) {
        const u32 r = r0 + threadIdx.x;
        bool keep = false;
        Key k;
        Coeff v = coeff_zero();
        if (r < len) {
            k = load_key(x, z, (size_t)start + r);
            if (!rr_touched(info, rr_pattern(info, k))) {
                v = coeff_load(c, (size_t)start + r);
                keep = rr_survives(prog, k, v);
            }
        }
        const u32 o = join_claim(&s_next, keep);
        if (keep) {
            const size_t slot = (size_t)dst0 + o;
#pragma unroll
            for (int w = 0; w < W; ++w) {
                ox[slot * W + w] = k.x[w];
                oz[slot * W + w] = k.z[w];
            }
            coeff_store(oc, slot, v);
            og[slot] = g[(size_t)start + r];
        }
    }
}

// k_rotation_run_scatter: arena row i to a slot claimed from its bucket's cursor.
extern "C" __global__ void k_rotation_run_scatter(
    const u64* __restrict__ ax, const u64* __restrict__ az, const real_t* __restrict__ ac, const u64* __restrict__ ag,
    const u32* __restrict__ abucket, u32 n, u32* __restrict__ cursor, u64* __restrict__ ox, u64* __restrict__ oz,
    real_t* __restrict__ oc, u64* __restrict__ og) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u32 slot = atomicAdd(&cursor[abucket[i]], 1u);
#pragma unroll
    for (int w = 0; w < W; ++w) {
        ox[(size_t)slot * W + w] = ax[(size_t)i * W + w];
        oz[(size_t)slot * W + w] = az[(size_t)i * W + w];
    }
    coeff_store(oc, slot, coeff_load(ac, i));
    og[slot] = ag[i];
}
