// `CollapseSample`'s systematic resample (engine/gpu/collapse.rs): a stretch is `|c|` in fixed point, `round(|c|·2^shift)` with `Σ < 2^62`, so every prefix is exact and a bucket's last stretch ends exactly where the next bucket's first begins.
// `frame` is `[offset, whole]`: the stretches before this sum's first and the group's fixed-point total; `total` the `f64` estimate of `Σ|c|` the shift is taken from.

#define COLLAPSE_SCAN_ITEMS 4
#define COLLAPSE_SCAN_BLOCK (1024 * COLLAPSE_SCAN_ITEMS)

// `|c|` in double; a complex type takes `sqrt(|c|^2)` rather than the host's slower `hypot`, an ulp apart, which only moves a point that sat on a boundary.
__device__ __forceinline__ double coeff_magnitude(Coeff c) {
#if COEFF_COMPLEX
    return sqrt(coeff_norm_sqr(c));
#else
    return fabs((double)c.part[0]);
#endif
}

// The shift that puts `total` below 2^61, so the rounded stretches of a sum whose estimate is off by rounding still total below 2^62.
__device__ __forceinline__ int collapse_shift(double total) {
    int exponent;
    frexp(total, &exponent);
    return 61 - exponent;
}

__device__ __forceinline__ u64 collapse_stretch(double magnitude, int shift) {
    return __double2ull_rn(ldexp(magnitude, shift));
}

// How many of the `samples` points `(u + k)·whole/samples` lie below `boundary`, `per_unit = samples/whole`; nondecreasing in `boundary`, all of them at `whole`.
__device__ __forceinline__ u32 collapse_below(u64 boundary, u64 whole, double per_unit, double u, u32 samples) {
    if (boundary >= whole) return samples;
    const double count = ceil((double)boundary * per_unit - u);
    return count <= 0.0 ? 0u : (u32)fmin(count, (double)samples);
}

// `total[0] += Σ|c|`, a block's partial sum per atomic.
extern "C" __global__ void k_collapse_total(const real_t* __restrict__ c, const u32* __restrict__ start,
                                            const u32* __restrict__ lens, u32 B, double* __restrict__ total) {
    __shared__ double warp_sum[32];
    const u32 lane = lane_id();
    const u32 stride = gridDim.x * (blockDim.x / WARP);
    double sum = 0.0;
    for (u32 beta = blockIdx.x * (blockDim.x / WARP) + warp_id(); beta < B; beta += stride) {
        const u32 s = start[beta], len = lens[beta];
        for (u32 r = lane; r < len; r += WARP) sum += coeff_magnitude(coeff_load(c, (size_t)s + r));
    }
#pragma unroll
    for (u32 o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(~0u, sum, o);
    if (lane == 0) warp_sum[warp_id()] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        double block = 0.0;
        for (u32 w = 0; w < blockDim.x / WARP; ++w) block += warp_sum[w];
        if (block != 0.0) atomicAdd(total, block);
    }
}

// `weights[β]`, bucket β's fixed-point `Σ|c|`; one warp per bucket.
extern "C" __global__ void k_collapse_weights(const real_t* __restrict__ c, const u32* __restrict__ start,
                                              const u32* __restrict__ lens, u32 B, const double* __restrict__ total,
                                              u64* __restrict__ weights) {
    const u32 lane = lane_id();
    const u32 beta = blockIdx.x * (blockDim.x / WARP) + warp_id();
    if (beta >= B) return;
    const int shift = collapse_shift(*total);
    const u32 s = start[beta], len = lens[beta];
    u64 sum = 0;
    for (u32 r = lane; r < len; r += WARP) sum += collapse_stretch(coeff_magnitude(coeff_load(c, (size_t)s + r)), shift);
#pragma unroll
    for (u32 o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(~0u, sum, o);
    if (lane == 0) weights[beta] = sum;
}

extern "C" __global__ void k_collapse_scan_block(const u64* __restrict__ in, u64* __restrict__ out,
                                                 u64* __restrict__ block_sum, u32 n) {
    __shared__ u64 swarp[33];
    const u32 base = blockIdx.x * COLLAPSE_SCAN_BLOCK + threadIdx.x * COLLAPSE_SCAN_ITEMS;
    u64 items[COLLAPSE_SCAN_ITEMS];
#pragma unroll
    for (int i = 0; i < COLLAPSE_SCAN_ITEMS; ++i) items[i] = base + i < n ? in[base + i] : 0ull;
    const u64 total = block_exclusive_scan_items<COLLAPSE_SCAN_ITEMS, u64>(items, swarp);
#pragma unroll
    for (int i = 0; i < COLLAPSE_SCAN_ITEMS; ++i) {
        if (base + i < n) out[base + i] = items[i];
    }
    if (threadIdx.x == 0) block_sum[blockIdx.x] = total;
}

// One block: scans `block_sum` in place, writes `out[n]` and the one-sum frame `[0, total]`.
extern "C" __global__ void k_collapse_scan_single(u64* __restrict__ block_sum, u32 nblocks, u64* __restrict__ out,
                                                  u32 n, u64* __restrict__ frame) {
    __shared__ u64 swarp[33];
    const u32 base = threadIdx.x * COLLAPSE_SCAN_ITEMS;
    u64 items[COLLAPSE_SCAN_ITEMS];
#pragma unroll
    for (int i = 0; i < COLLAPSE_SCAN_ITEMS; ++i) items[i] = base + i < nblocks ? block_sum[base + i] : 0ull;
    const u64 total = block_exclusive_scan_items<COLLAPSE_SCAN_ITEMS, u64>(items, swarp);
#pragma unroll
    for (int i = 0; i < COLLAPSE_SCAN_ITEMS; ++i) {
        if (base + i < nblocks) block_sum[base + i] = items[i];
    }
    if (threadIdx.x == 0) {
        out[n] = total;
        frame[0] = 0;
        frame[1] = total;
    }
}

extern "C" __global__ void k_collapse_scan_add(u64* __restrict__ out, const u64* __restrict__ block_off, u32 n) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += block_off[i / COLLAPSE_SCAN_BLOCK];
}

// Bucket β's drawn terms, in its order and at its start offset, each `m·h·c/|c|` with `h = whole·2^-shift / samples`; `kept[0] +=` the terms written.
// A lane's stretch is `[lo, hi)` and it holds the points below `hi` that are not below `lo`, so a point on a boundary belongs to the later stretch and an empty stretch holds none.
extern "C" __global__ void k_collapse_pick(const u64* __restrict__ x, const u64* __restrict__ z,
                                           const real_t* __restrict__ c, const u64* __restrict__ g,
                                           const u32* __restrict__ in_start, const u32* __restrict__ in_len, u32 B,
                                           const u64* __restrict__ bucket_offset, const double* __restrict__ total,
                                           const u64* __restrict__ frame, double u, u32 samples,
                                           u64* __restrict__ ox, u64* __restrict__ oz, real_t* __restrict__ oc,
                                           u64* __restrict__ og, u32* __restrict__ out_start, u32* __restrict__ out_len,
                                           u64* __restrict__ kept) {
    __shared__ u32 block_kept;
    if (threadIdx.x == 0) block_kept = 0;
    __syncthreads();
    const u32 lane = lane_id();
    const u32 beta = blockIdx.x * (blockDim.x / WARP) + warp_id();
    if (beta < B) {
        const int shift = collapse_shift(*total);
        const u64 whole = frame[1];
        const double step = ldexp((double)whole, -shift) / (double)samples;
        const double per_unit = (double)samples / (double)whole;
        const u32 start = in_start[beta], len = in_len[beta];
        u64 running = frame[0] + bucket_offset[beta];
        u32 run = 0;
        for (u32 r0 = 0; r0 < len; r0 += WARP) {
            const u32 r = r0 + lane;
            Coeff v = coeff_zero();
            double magnitude = 0.0;
            u64 stretch = 0;
            if (r < len) {
                v = coeff_load(c, (size_t)start + r);
                magnitude = coeff_magnitude(v);
                stretch = collapse_stretch(magnitude, shift);
            }
            const u64 inc = warp_inclusive_scan(stretch);
            const u64 hi = running + inc;
            const u32 m = stretch == 0 ? 0u
                                       : collapse_below(hi, whole, per_unit, u, samples) -
                                             collapse_below(hi - stretch, whole, per_unit, u, samples);
            running += __shfl_sync(~0u, inc, 31);
            const bool drawn = m > 0;
            const u32 ballot = __ballot_sync(~0u, drawn);
            if (drawn) {
                const size_t src = (size_t)start + r;
                const size_t dst = (size_t)start + run + __popc(ballot & lanemask_lt());
                store_key(ox, oz, dst, load_key(x, z, src));
                const double scale = (double)m * step;
                Coeff out;
                FOR_PARTS(j) out.part[j] = (real_t)((double)v.part[j] / magnitude * scale);
                coeff_store(oc, dst, out);
                og[dst] = g[src];
            }
            run += __popc(ballot);
        }
        if (lane == 0) {
            out_start[beta] = start;
            out_len[beta] = run;
            if (run != 0) atomicAdd(&block_kept, run);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && block_kept != 0) atomicAdd(kept, (u64)block_kept);
}
