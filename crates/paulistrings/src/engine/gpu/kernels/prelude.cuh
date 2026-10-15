// Definitions every kernel family shares; NVRTC compiles each family behind this file with -DW=<w>.
#ifndef PAULISTRINGS_PRELUDE_CUH
#define PAULISTRINGS_PRELUDE_CUH

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

#ifndef W
#error "W must be defined"
#endif

// The collision nets compile with a shorter fingerprint; 64 is the unmasked default.
#ifndef FINGERPRINT_BITS
#define FINGERPRINT_BITS 64
#endif

#define FINGERPRINT_ROWS 64
#define WARP 32

// THREADS (the fused-layer block width per W), LAYER_CAP (records per fused block) and MAX_BUCKET_LEN come from module.rs as -D options.
#if !defined(THREADS) || !defined(LAYER_CAP) || !defined(MAX_BUCKET_LEN)
#error "THREADS, LAYER_CAP and MAX_BUCKET_LEN must be defined"
#endif
#define WARPS (THREADS / WARP)

// The record tag layout `entry:4 | offset-in-source-bucket:log2(MAX_BUCKET_LEN)`, in 16 bits.
#define MAX_ENTRIES 16
__host__ __device__ constexpr u32 log2_exact(u32 v) { return v <= 1u ? 0u : 1u + log2_exact(v >> 1); }
static_assert((MAX_BUCKET_LEN & (MAX_BUCKET_LEN - 1u)) == 0u, "MAX_BUCKET_LEN is a power of two");
static_assert(MAX_BUCKET_LEN * MAX_ENTRIES <= 65536u, "a tag fits 16 bits");
constexpr u32 TAG_OFFSET_BITS = log2_exact(MAX_BUCKET_LEN);
#define TAG_OFFSET_MASK (MAX_BUCKET_LEN - 1u)
#define LOCAL_DIM 16
// `received_slot[e]` for an entry whose rows come from a local bucket; any other value is the received block's slot.
#define NO_REMOTE 0xFFFFFFFFu
// `entry_of[s]` for a support pattern no entry of a permutation table emits for (permute.cu).
#define NO_ENTRY 0xFFFFFFFFu

// Prepared-table kinds: a `LocalPtm` and a wide `PreparedRotation`.
#define MODE_LOCAL 0
#define MODE_ROTATION 1

// The per-term truncation program, `KeepProgram` in truncation.rs: postfix over a one-bit-per-entry stack; KEEP_NODES and the OP_* codes come from module.rs as -D options.
#if !defined(KEEP_NODES) || !defined(OP_KEEP) || !defined(OP_COEFF) || !defined(OP_WEIGHT) || !defined(OP_AND) || !defined(OP_OR)
#error "KEEP_NODES and the OP_* codes must be defined"
#endif

struct KeepProg {
    u32 len;
    u32 op[KEEP_NODES];
    u64 arg[KEEP_NODES];
};

struct Key {
    u64 x[W];
    u64 z[W];
};

// One prepared channel as the kernels see it; `amplitude` holds 16 entries of LOCAL_DIM complex amplitudes as f64 pairs.
struct Table {
    u32 mode;
    u32 entries;
    u32 support_size;
    u32 qubit_0;
    u32 qubit_1;
    double rotation_cos;
    double rotation_sin;
    const double* amplitude;
    const u64* mask;
    const u32* nonzero_patterns;
};

__device__ __forceinline__ u32 lane_id() { return threadIdx.x & 31u; }
__device__ __forceinline__ u32 warp_id() { return threadIdx.x >> 5; }
__device__ __forceinline__ u32 lanemask_lt() { return (1u << lane_id()) - 1u; }

__device__ __forceinline__ Key load_key(const u64* __restrict__ x, const u64* __restrict__ z, size_t i) {
    Key k;
#pragma unroll
    for (int w = 0; w < W; ++w) {
        k.x[w] = x[i * W + w];
        k.z[w] = z[i * W + w];
    }
    return k;
}

__device__ __forceinline__ bool key_eq(const Key& a, const Key& b) {
    bool eq = true;
#pragma unroll
    for (int w = 0; w < W; ++w) eq &= (a.x[w] == b.x[w]) & (a.z[w] == b.z[w]);
    return eq;
}

// The host's `([u64; W], [u64; W])` tuple order: x words from index 0, then z words.
__device__ __forceinline__ bool key_lt(const Key& a, const Key& b) {
    for (int w = 0; w < W; ++w) {
        if (a.x[w] != b.x[w]) return a.x[w] < b.x[w];
    }
    for (int w = 0; w < W; ++w) {
        if (a.z[w] != b.z[w]) return a.z[w] < b.z[w];
    }
    return false;
}

// Row r of every uploaded GF(2) matrix is the pair (x-mask at rows[2rW..], z-mask at rows[(2r+1)W..]).
__device__ __forceinline__ u32 row_parity(const Key& k, const u64* rows, u32 r) {
    u64 acc = 0;
#pragma unroll
    for (int w = 0; w < W; ++w) acc ^= (k.x[w] & rows[(2 * r) * W + w]) ^ (k.z[w] & rows[(2 * r + 1) * W + w]);
    return (u32)(__popcll(acc) & 1);
}

// FINGERPRINT_ZERO_LOW is the test hook that clears the low word instead, so every block takes the g_hi32 fallback and resolves there.
__device__ __forceinline__ u64 fingerprint_mask() {
#if defined(FINGERPRINT_ZERO_LOW)
    return ~0xFFFFFFFFull;
#elif FINGERPRINT_BITS >= 64
    return ~0ull;
#else
    return (1ull << FINGERPRINT_BITS) - 1ull;
#endif
}

// Live-qubit mask of word w, as `word_mask` in pauli_string.rs.
__device__ __forceinline__ u64 word_mask(u32 num_qubits, int w) {
    const u32 lo = 64u * (u32)w;
    if (num_qubits >= lo + 64u) return ~0ull;
    if (num_qubits <= lo) return 0ull;
    return (1ull << (num_qubits - lo)) - 1ull;
}

// `LocalPtm::support_bits`: bit 2j is the x-bit of support qubit j, bit 2j+1 its z-bit.
__device__ __forceinline__ u32 support_bits(const Key& k, u32 support_size, u32 qubit_0, u32 qubit_1) {
    u32 s = 0;
    if (support_size > 0) {
        s |= (u32)((k.x[qubit_0 >> 6] >> (qubit_0 & 63)) & 1ull);
        s |= (u32)((k.z[qubit_0 >> 6] >> (qubit_0 & 63)) & 1ull) << 1;
    }
    if (support_size > 1) {
        s |= (u32)((k.x[qubit_1 >> 6] >> (qubit_1 & 63)) & 1ull) << 2;
        s |= (u32)((k.z[qubit_1 >> 6] >> (qubit_1 & 63)) & 1ull) << 3;
    }
    return s;
}

__device__ __forceinline__ u32 pauli_weight(const Key& k) {
    u32 wgt = 0;
#pragma unroll
    for (int w = 0; w < W; ++w) wgt += __popcll(k.x[w] | k.z[w]);
    return wgt;
}

// Whether k anticommutes with the generator, a rotation table's entry 1, read in place so no local array is spilled.
__device__ __forceinline__ bool anticommutes_generator(const Table& T, const Key& k) {
    u64 acc = 0;
#pragma unroll
    for (int w = 0; w < W; ++w) acc ^= (k.x[w] & T.mask[3 * W + w]) ^ (k.z[w] & T.mask[2 * W + w]);
    return (__popcll(acc) & 1) != 0;
}

// `PauliString::mul_assign`'s i^delta for k · P over the generator, delta mod 4.
__device__ __forceinline__ u32 product_phase_generator(const Table& T, const Key& k) {
    u32 delta = 0;
#pragma unroll
    for (int w = 0; w < W; ++w) {
        const u64 a = k.x[w], b = k.z[w], c = T.mask[2 * W + w], d = T.mask[3 * W + w];
        delta += 2u * (u32)__popcll(b & c) + (u32)__popcll(a & b) + (u32)__popcll(c & d) - (u32)__popcll((a ^ c) & (b ^ d));
    }
    return delta & 3u;
}

// Whether entry e emits a row for key k: the table-entry filter, never a product test (signed-zero contract).
__device__ __forceinline__ bool entry_emits(const Table& T, const Key& k, u32 e) {
    if (T.mode == MODE_LOCAL) {
        const u32 s = support_bits(k, T.support_size, T.qubit_0, T.qubit_1);
        return ((T.nonzero_patterns[e] >> s) & 1u) != 0;
    }
    return e == 0 || anticommutes_generator(T, k);
}

// The row entry e emits for (k, c): bitwise `DeltaEntry::emit` for a local table, `PreparedRotation::emit_generator` and the identity pass of the rotation arm otherwise.
__device__ __forceinline__ void entry_product(const Table& T, const Key& k, u32 e, double cr, double ci, double& pr, double& pi) {
    if (T.mode == MODE_LOCAL) {
        const u32 s = support_bits(k, T.support_size, T.qubit_0, T.qubit_1);
        const double ar = T.amplitude[(e * LOCAL_DIM + s) * 2], ai = T.amplitude[(e * LOCAL_DIM + s) * 2 + 1];
        pr = cr * ar - ci * ai;
        pi = cr * ai + ci * ar;
        return;
    }
    if (e == 0) {
        if (anticommutes_generator(T, k)) {
            pr = cr * T.rotation_cos;
            pi = ci * T.rotation_cos;
        } else {
            pr = cr;
            pi = ci;
        }
        return;
    }
    const u32 ph = (1u + product_phase_generator(T, k)) & 3u;
    double tr, ti;
    switch (ph) {
        case 0: tr = cr; ti = ci; break;
        case 1: tr = -ci; ti = cr; break;
        case 2: tr = -cr; ti = -ci; break;
        default: tr = ci; ti = -cr; break;
    }
    pr = tr * T.rotation_sin;
    pi = ti * T.rotation_sin;
}

// `TruncationPolicy::keep_term` of the lowered tree on the summed coefficient; `key()` is called at most once, and only by a weight node.
template <class KeyFn>
__device__ __forceinline__ bool keep_eval(const KeepProg& P, KeyFn key, double cr, double ci) {
    if (P.len == 1 && P.op[0] == OP_KEEP) return true;
    const double m = cr * cr + ci * ci;
    u32 st = 0;
    int wgt = -1;
    for (u32 i = 0; i < P.len; ++i) {
        u32 v;
        switch (P.op[i]) {
            case OP_COEFF: {
                const double eps = __longlong_as_double((long long)P.arg[i]);
                st = (st << 1) | (u32)(eps < 0.0 || m > eps * eps);
                break;
            }
            case OP_WEIGHT:
                if (wgt < 0) wgt = (int)pauli_weight(key());
                st = (st << 1) | (u32)((u64)wgt <= P.arg[i]);
                break;
            case OP_AND:
                v = st & (st >> 1) & 1u;
                st = ((st >> 2) << 1) | v;
                break;
            case OP_OR:
                v = (st | (st >> 1)) & 1u;
                st = ((st >> 2) << 1) | v;
                break;
            default:
                st = (st << 1) | 1u;
        }
    }
    return (st & 1u) != 0;
}

// Exclusive scan of v[0..n) in shared memory, ITEMS contiguous entries per thread; returns the total to every thread.
// swarp holds 33 words of scratch and is free again on return.
template <int ITEMS, typename T>
__device__ __forceinline__ u32 block_exclusive_scan(T* v, u32 n, u32* swarp) {
    const u32 tid = threadIdx.x, lane = lane_id(), warp = warp_id();
    const u32 base = tid * ITEMS;
    u32 loc[ITEMS];
    u32 sum = 0;
#pragma unroll
    for (int c = 0; c < ITEMS; ++c) {
        loc[c] = (base + c < n) ? (u32)v[base + c] : 0u;
        sum += loc[c];
    }
    u32 inc = sum;
#pragma unroll
    for (u32 o = 1; o < 32; o <<= 1) {
        u32 t = __shfl_up_sync(~0u, inc, o);
        if (lane >= o) inc += t;
    }
    if (lane == 31) swarp[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        u32 v2 = (lane < (blockDim.x >> 5)) ? swarp[lane] : 0u;
        u32 inc2 = v2;
#pragma unroll
        for (u32 o = 1; o < 32; o <<= 1) {
            u32 t = __shfl_up_sync(~0u, inc2, o);
            if (lane >= o) inc2 += t;
        }
        swarp[lane] = inc2 - v2;
        if (lane == 31) swarp[32] = inc2;
    }
    __syncthreads();
    u32 run = swarp[warp] + inc - sum;
#pragma unroll
    for (int c = 0; c < ITEMS; ++c) {
        if (base + c < n) v[base + c] = (T)run;
        run += loc[c];
    }
    u32 total = swarp[32];
    __syncthreads();
    return total;
}

#endif
