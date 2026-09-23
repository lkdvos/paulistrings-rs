// Definitions every kernel family shares; NVRTC compiles each family behind this file with -DW=<w>.
#ifndef PAULISTRINGS_PRELUDE_CUH
#define PAULISTRINGS_PRELUDE_CUH

typedef unsigned long long u64;
typedef unsigned int u32;

#ifndef W
#error "W must be defined"
#endif

// The collision nets compile with a shorter fingerprint; 64 is the unmasked default.
#ifndef FP_BITS
#define FP_BITS 64
#endif

#define FP_ROWS 64
#define WARP 32

struct Key {
    u64 x[W];
    u64 z[W];
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

__device__ __forceinline__ u64 fp_mask() {
#if FP_BITS >= 64
    return ~0ull;
#else
    return (1ull << FP_BITS) - 1ull;
#endif
}

// Live-qubit mask of word w, as `word_mask` in bucket/hash.rs.
__device__ __forceinline__ u64 word_mask(u32 num_qubits, int w) {
    const u32 lo = 64u * (u32)w;
    if (num_qubits >= lo + 64u) return ~0ull;
    if (num_qubits <= lo) return 0ull;
    return (1ull << (num_qubits - lo)) - 1ull;
}

// Exclusive scan of v[0..n) in shared memory, ITEMS contiguous entries per thread; returns the total to every thread.
// swarp holds 33 words of scratch and is free again on return.
template <int ITEMS, typename T>
__device__ __forceinline__ u32 block_exscan(T* v, u32 n, u32* swarp) {
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
