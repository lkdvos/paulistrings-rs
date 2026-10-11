// K7: `ApproxTopN`'s octave histogram of |c|^2 (ARCHITECTURE.md §Truncation).
// K8: exact `TopN`'s radix-select over the u64 bit pattern of |c|^2, which orders a non-negative f64 as its value.
// Both retain through `k_boundary_counts` and `k_retain`: the key `bits >> shift` (shift 52 is the octave, 0 the exact value) above `boundary`, plus a quota of terms at it in bucket order.

#define APPROX_BINS 2048

// out[0] += terms, out[1 + k] += population of octave k (`norm_sqr` bits >> 52); one warp per bucket, a shared histogram per block, integer atomics only.
extern "C" __global__ void k_octave_hist(const double* __restrict__ c, const u32* __restrict__ start,
                                         const u32* __restrict__ lens, u32 B, u64* __restrict__ out) {
    __shared__ u32 h[APPROX_BINS];
    __shared__ u32 terms;
    for (u32 i = threadIdx.x; i < APPROX_BINS; i += blockDim.x) h[i] = 0;
    if (threadIdx.x == 0) terms = 0;
    __syncthreads();
    const u32 lane = lane_id();
    const u32 stride = gridDim.x * (blockDim.x / WARP);
    for (u32 beta = blockIdx.x * (blockDim.x / WARP) + warp_id(); beta < B; beta += stride) {
        const u32 s = start[beta], len = lens[beta];
        for (u32 r = lane; r < len; r += WARP) {
            const size_t i = (size_t)s + r;
            const double cr = c[2 * i], ci = c[2 * i + 1];
            const u64 bits = (u64)__double_as_longlong(cr * cr + ci * ci);
            // A NaN with its sign bit set would index past the top bin; it shares it instead.
            atomicAdd(&h[min((u32)(bits >> 52), (u32)(APPROX_BINS - 1))], 1u);
        }
        if (lane == 0) atomicAdd(&terms, len);
    }
    __syncthreads();
    for (u32 i = threadIdx.x; i < APPROX_BINS; i += blockDim.x) {
        if (h[i] != 0) atomicAdd(&out[1 + i], (u64)h[i]);
    }
    if (threadIdx.x == 0 && terms != 0) atomicAdd(&out[0], (u64)terms);
}

// Histogram of one 8-bit digit of |c|^2's bit pattern at `shift`, over terms whose higher bits already match `prefix` under `fixed_mask` (empty for the first pass, `fixed_mask = 0`).
extern "C" __global__ void k_radix_hist(const double* __restrict__ c, const u32* __restrict__ start,
                                        const u32* __restrict__ lens, u32 B, u64 fixed_mask, u64 prefix,
                                        u32 shift, u64* __restrict__ out) {
    __shared__ u32 h[256];
    for (u32 i = threadIdx.x; i < 256; i += blockDim.x) h[i] = 0;
    __syncthreads();
    const u32 lane = lane_id();
    const u32 stride = gridDim.x * (blockDim.x / WARP);
    for (u32 beta = blockIdx.x * (blockDim.x / WARP) + warp_id(); beta < B; beta += stride) {
        const u32 s = start[beta], len = lens[beta];
        for (u32 r = lane; r < len; r += WARP) {
            const size_t i = (size_t)s + r;
            const double cr = c[2 * i], ci = c[2 * i + 1];
            const u64 bits = (u64)__double_as_longlong(cr * cr + ci * ci);
            if ((bits & fixed_mask) == prefix) {
                const u32 digit = (u32)((bits >> shift) & 0xFFu);
                atomicAdd(&h[digit], 1u);
            }
        }
    }
    __syncthreads();
    for (u32 i = threadIdx.x; i < 256; i += blockDim.x) {
        if (h[i] != 0) atomicAdd(&out[i], (u64)h[i]);
    }
}

// The exact bits of the single term matching `mask`/`prefix`; only correct when exactly one term matches (a radix-select digit that narrowed to a singleton).
extern "C" __global__ void k_radix_extract(const double* __restrict__ c, const u32* __restrict__ start,
                                           const u32* __restrict__ lens, u32 B, u64 mask, u64 prefix,
                                           u64* __restrict__ out) {
    const u32 lane = lane_id();
    const u32 stride = gridDim.x * (blockDim.x / WARP);
    for (u32 beta = blockIdx.x * (blockDim.x / WARP) + warp_id(); beta < B; beta += stride) {
        const u32 s = start[beta], len = lens[beta];
        for (u32 r = lane; r < len; r += WARP) {
            const size_t i = (size_t)s + r;
            const double cr = c[2 * i], ci = c[2 * i + 1];
            const u64 bits = (u64)__double_as_longlong(cr * cr + ci * ci);
            if ((bits & mask) == prefix) out[0] = bits;
        }
    }
}

// The key of |c|^2 for a boundary comparison, or `NO_KEY` for a NaN, which sits above +inf's bits and is never kept.
#define NO_KEY 0xFFFFFFFFFFFFFFFFull
__device__ __forceinline__ u64 boundary_key(double cr, double ci, u32 shift) {
    const u64 bits = (u64)__double_as_longlong(cr * cr + ci * ci);
    return bits <= 0x7ff0000000000000ull ? bits >> shift : NO_KEY;
}

// counts[beta] = terms of bucket beta whose key equals `boundary`; one warp per bucket, no atomics.
extern "C" __global__ void k_boundary_counts(const double* __restrict__ c, const u32* __restrict__ start,
                                             const u32* __restrict__ lens, u32 B, u64 boundary, u32 shift,
                                             u32* __restrict__ counts) {
    const u32 lane = lane_id();
    const u32 beta = blockIdx.x * (blockDim.x / WARP) + warp_id();
    if (beta >= B) return;
    const u32 s = start[beta], len = lens[beta];
    u32 count = 0;
    for (u32 r0 = 0; r0 < len; r0 += WARP) {
        const u32 r = r0 + lane;
        bool at = false;
        if (r < len) {
            const size_t i = (size_t)s + r;
            at = boundary_key(c[2 * i], c[2 * i + 1], shift) == boundary;
        }
        count += __popc(__ballot_sync(~0u, at));
    }
    if (lane == 0) counts[beta] = count;
}

// Keep the terms whose key exceeds `boundary`, and those equal to it while `boundary_before[beta]` plus the bucket's earlier ones stays below `quota`; order-preserving, one warp per bucket into the same start offsets.
extern "C" __global__ void k_retain(const u64* __restrict__ x, const u64* __restrict__ z,
                                    const double* __restrict__ c, const u64* __restrict__ g,
                                    const u32* __restrict__ in_start, const u32* __restrict__ in_len, u32 B,
                                    u64 boundary, u32 shift, const u32* __restrict__ boundary_before, u64 quota,
                                    u64* __restrict__ ox, u64* __restrict__ oz, double* __restrict__ oc,
                                    u64* __restrict__ og, u32* __restrict__ out_start, u32* __restrict__ out_len) {
    const u32 lane = lane_id();
    const u32 beta = blockIdx.x * (blockDim.x / WARP) + warp_id();
    if (beta >= B) return;
    const u32 start = in_start[beta], len = in_len[beta];
    u64 taken = boundary_before[beta];
    u32 run = 0;
    for (u32 r0 = 0; r0 < len; r0 += WARP) {
        const u32 r = r0 + lane;
        bool above = false, at = false;
        double cr = 0.0, ci = 0.0;
        if (r < len) {
            const size_t src = (size_t)start + r;
            cr = c[2 * src];
            ci = c[2 * src + 1];
            const u64 key = boundary_key(cr, ci, shift);
            above = key != NO_KEY && key > boundary;
            at = key == boundary;
        }
        const u32 at_ballot = __ballot_sync(~0u, at);
        const bool ok = above || (at && taken + __popc(at_ballot & lanemask_lt()) < quota);
        taken += __popc(at_ballot);
        const u32 bal = __ballot_sync(~0u, ok);
        if (ok) {
            const size_t src = (size_t)start + r;
            const size_t dst = (size_t)start + run + __popc(bal & lanemask_lt());
#pragma unroll
            for (int w = 0; w < W; ++w) {
                ox[dst * W + w] = x[src * W + w];
                oz[dst * W + w] = z[src * W + w];
            }
            oc[2 * dst] = cr;
            oc[2 * dst + 1] = ci;
            og[dst] = g[src];
        }
        run += __popc(bal);
    }
    if (lane == 0) {
        out_start[beta] = start;
        out_len[beta] = run;
    }
}
