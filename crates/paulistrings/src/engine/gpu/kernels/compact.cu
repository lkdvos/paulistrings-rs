// K4: append one batch of output positions [p0, p0 + gridDim.x) from the batch-relative arena to the tight columns at running + dst_off, recording each bucket's start and length.

extern "C" __global__ void k_compact(const u64* __restrict__ ax, const u64* __restrict__ az,
                                     const double* __restrict__ ac, const u64* __restrict__ ag,
                                     const u32* __restrict__ seg_start, const u32* __restrict__ dst_off,
                                     const u32* __restrict__ out_len_pos, const u32* __restrict__ bucket_at,
                                     u32 p0, u32 running, u64* __restrict__ ox, u64* __restrict__ oz,
                                     double* __restrict__ oc, u64* __restrict__ og,
                                     u32* __restrict__ out_start, u32* __restrict__ out_len) {
    const u32 p = p0 + blockIdx.x;
    const u32 n = out_len_pos[p];
    const u32 src0 = seg_start[p] - seg_start[p0];
    const u32 dst0 = running + dst_off[blockIdx.x];
    if (threadIdx.x == 0) {
        const u32 beta = bucket_at[p];
        out_start[beta] = dst0;
        out_len[beta] = n;
    }
    for (u32 k = threadIdx.x; k < n; k += blockDim.x) {
        const size_t s = src0 + k, d = dst0 + k;
#pragma unroll
        for (int w = 0; w < W; ++w) {
            ox[d * W + w] = ax[s * W + w];
            oz[d * W + w] = az[s * W + w];
        }
        oc[2 * d] = ac[2 * s];
        oc[2 * d + 1] = ac[2 * s + 1];
        og[d] = ag[s];
    }
}
