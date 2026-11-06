// k_octave_histogram: `ApproxTopN`'s octave histogram of |c|^2.
// k_radix_histogram and k_radix_extract: exact `TopN`'s radix-select over the u64 bit pattern of |c|^2, which orders a non-negative f64 as its value.
// Exact `TopN` retains through `k_boundary_counts` and `k_retain`: the key `bits >> shift` (shift 0, the exact value) above `boundary`, plus a quota of terms at it in bucket order; `ApproxTopN` cuts in place through `k_cut` at the octave (shift 52).

#define APPROX_BINS 2048

// out[0] += terms, out[1 + k] += population of octave k (`norm_sqr` bits >> 52); one warp per bucket, a shared histogram per block, integer atomics only.
extern "C" __global__ void k_octave_histogram(const real_t* __restrict__ c, const u32* __restrict__ start,
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
            const u64 bits = (u64)__double_as_longlong(coeff_norm_sqr(coeff_load(c, i)));
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
extern "C" __global__ void k_radix_histogram(const real_t* __restrict__ c, const u32* __restrict__ start,
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
            const u64 bits = (u64)__double_as_longlong(coeff_norm_sqr(coeff_load(c, i)));
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
extern "C" __global__ void k_radix_extract(const real_t* __restrict__ c, const u32* __restrict__ start,
                                           const u32* __restrict__ lens, u32 B, u64 mask, u64 prefix,
                                           u64* __restrict__ out) {
    const u32 lane = lane_id();
    const u32 stride = gridDim.x * (blockDim.x / WARP);
    for (u32 beta = blockIdx.x * (blockDim.x / WARP) + warp_id(); beta < B; beta += stride) {
        const u32 s = start[beta], len = lens[beta];
        for (u32 r = lane; r < len; r += WARP) {
            const size_t i = (size_t)s + r;
            const u64 bits = (u64)__double_as_longlong(coeff_norm_sqr(coeff_load(c, i)));
            if ((bits & mask) == prefix) out[0] = bits;
        }
    }
}

// The key of |c|^2 for a boundary comparison, or `NO_KEY` for a NaN, which sits above +inf's bits and is never kept.
#define NO_KEY 0xFFFFFFFFFFFFFFFFull
__device__ __forceinline__ u64 boundary_key(Coeff v, u32 shift) {
    const u64 bits = (u64)__double_as_longlong(coeff_norm_sqr(v));
    return bits <= 0x7ff0000000000000ull ? bits >> shift : NO_KEY;
}

// counts[beta] = terms of bucket beta whose key equals `boundary`; one warp per bucket, no atomics.
extern "C" __global__ void k_boundary_counts(const real_t* __restrict__ c, const u32* __restrict__ start,
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
            at = boundary_key(coeff_load(c, i), shift) == boundary;
        }
        count += __popc(__ballot_sync(~0u, at));
    }
    if (lane == 0) counts[beta] = count;
}

// Keep the terms whose key exceeds `boundary`, and those equal to it while `boundary_before[beta]` plus the bucket's earlier ones stays below `quota`; order-preserving, one warp per bucket into the same start offsets.
extern "C" __global__ void k_retain(const u64* __restrict__ x, const u64* __restrict__ z,
                                    const real_t* __restrict__ c, const u64* __restrict__ g,
                                    const u32* __restrict__ in_start, const u32* __restrict__ in_len, u32 B,
                                    u64 boundary, u32 shift, const u32* __restrict__ boundary_before, u64 quota,
                                    u64* __restrict__ ox, u64* __restrict__ oz, real_t* __restrict__ oc,
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
        Coeff v = coeff_zero();
        if (r < len) {
            const size_t src = (size_t)start + r;
            v = coeff_load(c, src);
            const u64 key = boundary_key(v, shift);
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
            store_key(ox, oz, dst, load_key(x, z, src));
            coeff_store(oc, dst, v);
            og[dst] = g[src];
        }
        run += __popc(bal);
    }
    if (lane == 0) {
        out_start[beta] = start;
        out_len[beta] = run;
    }
}

// `ApproxTopN`'s cut in place: keep the terms whose key exceeds `boundary`, and of those equal to it the first `quota` in bucket order; `counters[0]` sums the kept terms.
// One warp per bucket: a flag per row (0 dropped, 1 kept, 2 at the boundary), then the dropped rows below the bucket's kept count take the kept rows above it, so the bucket keeps its start and its kept terms fill its front, in no particular order.
// The boundary terms ahead of a bucket come from a decoupled look-back over `prefix` (zeroed, one entry per bucket); a warp takes its bucket from `counters[1]` in launch order, so every lower bucket's warp is already running and the look-back never waits on an unscheduled one.
#define CUT_QUEUE 64
#define CUT_WARPS 8
#define CUT_AGGREGATE (1ull << 62)
#define CUT_INCLUSIVE (2ull << 62)
#define CUT_VALUE (CUT_AGGREGATE - 1ull)

// The boundary terms of buckets `0..beta`, publishing bucket beta's `at_n` first as its aggregate and then as an inclusive prefix; lane 0's result.
__device__ __forceinline__ unsigned long long cut_look_back(unsigned long long* prefix, u32 beta, u32 at_n) {
    volatile unsigned long long* p = prefix;
    if (beta == 0) {
        p[0] = CUT_INCLUSIVE | at_n;
        return 0;
    }
    p[beta] = CUT_AGGREGATE | at_n;
    unsigned long long before = 0;
    for (u32 j = beta - 1;; --j) {
        unsigned long long v;
        do v = p[j]; while (v == 0);
        before += v & CUT_VALUE;
        if (v & CUT_INCLUSIVE) break;
    }
    p[beta] = CUT_INCLUSIVE | (before + at_n);
    return before;
}

__device__ __forceinline__ void cut_move(u64* x, u64* z, real_t* c, u64* g, size_t dst, size_t src) {
    store_key(x, z, dst, load_key(x, z, src));
    coeff_store(c, dst, coeff_load(c, src));
    g[dst] = g[src];
}

// Appends the rows of `[from, min(from + WARP, end))` that `want` picks to `q[n..]`; returns how many.
template <class Want>
__device__ __forceinline__ u32 cut_scan(u32* q, u32 n, u32 from, u32 end, Want want) {
    const u32 r = from + lane_id();
    const bool hit = r < end && want(r);
    const u32 bal = __ballot_sync(~0u, hit);
    if (hit) q[n + __popc(bal & lanemask_lt())] = r;
    return __popc(bal);
}

extern "C" __global__ void __launch_bounds__(CUT_WARPS * WARP) k_cut(
    u64* x, u64* z, real_t* c, u64* g, const u32* __restrict__ start, u32* __restrict__ lens, u32 B, u64 boundary,
    u32 shift, u64 quota, unsigned long long* prefix, u32 ordered, unsigned long long* __restrict__ counters,
    u8* __restrict__ flag) {
    __shared__ u32 queue[CUT_WARPS][2][CUT_QUEUE];
    const u32 lane = lane_id(), w = warp_id();
    if (blockIdx.x * (blockDim.x / WARP) + w >= B) return;
    u32 beta = 0;
    if (lane == 0) beta = (u32)atomicAdd(&counters[1], 1ull);
    beta = __shfl_sync(~0u, beta, 0);
    const u32 s = start[beta], len = lens[beta];
    u8* f = flag + s;
    u32 above_n = 0, at_n = 0;
    for (u32 r0 = 0; r0 < len; r0 += WARP) {
        const u32 r = r0 + lane;
        u8 v = 0;
        if (r < len) {
            const u64 key = boundary_key(coeff_load(c, (size_t)s + r), shift);
            v = key == boundary ? 2 : (key != NO_KEY && key > boundary ? 1 : 0);
            f[r] = v;
        }
        above_n += __popc(__ballot_sync(~0u, v == 1));
        at_n += __popc(__ballot_sync(~0u, v == 2));
    }
    unsigned long long first = 0;
    if (ordered && lane == 0) first = cut_look_back(prefix, beta, at_n);
    first = __shfl_sync(~0u, first, 0);
    const u32 at_kept = first >= quota ? 0u : (u32)min((unsigned long long)at_n, quota - first);
    const u32 kept = above_n + at_kept;
    __syncwarp();
    if (at_kept != 0 && at_kept != at_n) {
        // The bucket that straddles the quota keeps its first `at_kept` boundary rows.
        u32 rank = 0;
        for (u32 r0 = 0; r0 < len; r0 += WARP) {
            const u32 r = r0 + lane;
            const bool at = r < len && f[r] == 2;
            const u32 bal = __ballot_sync(~0u, at);
            if (at) f[r] = rank + __popc(bal & lanemask_lt()) < at_kept ? 1 : 0;
            rank += __popc(bal);
        }
        __syncwarp();
    }
    const u8 at_stays = at_kept == at_n ? 2 : 1;
    const auto keeps = [&](u32 r) -> bool {
        const u8 v = f[r];
        return v == 1 || v == at_stays;
    };
    u32* holes = queue[w][0];
    u32* movers = queue[w][1];
    u32 hole_at = 0, mover_at = kept, nh = 0, nm = 0;
    // Holes and movers are equally many, so the queues run dry together; neither runs at all when the bucket keeps everything or nothing.
    for (bool open = kept != 0 && kept != len; open;) {
        while (nh < WARP && hole_at < kept) {
            nh += cut_scan(holes, nh, hole_at, kept, [&](u32 r) { return !keeps(r); });
            hole_at += WARP;
        }
        while (nm < WARP && mover_at < len) {
            nm += cut_scan(movers, nm, mover_at, len, keeps);
            mover_at += WARP;
        }
        __syncwarp();
        // A queue holds up to `2 * WARP - 1` rows, a round moves at most one per lane.
        const u32 t = min(min(nh, nm), (u32)WARP);
        if (t == 0) break;
        if (lane < t) cut_move(x, z, c, g, (size_t)s + holes[lane], (size_t)s + movers[lane]);
        const u32 h0 = lane + t < nh ? holes[lane + t] : 0, h1 = lane + t + WARP < nh ? holes[lane + t + WARP] : 0;
        const u32 m0 = lane + t < nm ? movers[lane + t] : 0, m1 = lane + t + WARP < nm ? movers[lane + t + WARP] : 0;
        __syncwarp();
        if (lane + t < nh) holes[lane] = h0;
        if (lane + t + WARP < nh) holes[lane + WARP] = h1;
        if (lane + t < nm) movers[lane] = m0;
        if (lane + t + WARP < nm) movers[lane + WARP] = m1;
        __syncwarp();
        nh -= t;
        nm -= t;
    }
    if (lane == 0) {
        lens[beta] = kept;
        if (kept != 0) atomicAdd(&counters[0], (unsigned long long)kept);
    }
}
