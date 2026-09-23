// K7: `ApproxTopN` on device, the octave histogram of |c|^2 and the order-preserving retain against its edge (ARCHITECTURE.md §Truncation).

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

// Keep the terms with |c|^2 >= threshold, bitwise, order-preserving, one warp per bucket into the same start offsets (the K5 pattern).
extern "C" __global__ void k_retain(const u64* __restrict__ x, const u64* __restrict__ z,
                                    const double* __restrict__ c, const u64* __restrict__ g,
                                    const u32* __restrict__ in_start, const u32* __restrict__ in_len, u32 B,
                                    double threshold, u64* __restrict__ ox, u64* __restrict__ oz,
                                    double* __restrict__ oc, u64* __restrict__ og, u32* __restrict__ out_start,
                                    u32* __restrict__ out_len) {
    const u32 lane = lane_id();
    const u32 beta = blockIdx.x * (blockDim.x / WARP) + warp_id();
    if (beta >= B) return;
    const u32 start = in_start[beta], len = in_len[beta];
    u32 run = 0;
    for (u32 r0 = 0; r0 < len; r0 += WARP) {
        const u32 r = r0 + lane;
        bool ok = false;
        double cr = 0.0, ci = 0.0;
        if (r < len) {
            const size_t src = (size_t)start + r;
            cr = c[2 * src];
            ci = c[2 * src + 1];
            ok = cr * cr + ci * ci >= threshold;
        }
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
