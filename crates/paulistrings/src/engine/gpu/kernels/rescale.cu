// K5: the key-preserving fast path, `rescale_in_place` in engine/bucketed.rs: c' = c * amp[s], dropping exact zeros and rows `keep_term` rejects, order-preserving, one warp per bucket into the same start offsets.

extern "C" __global__ void k_rescale(const u64* __restrict__ x, const u64* __restrict__ z,
                                     const double* __restrict__ c, const u64* __restrict__ g,
                                     const u32* __restrict__ in_start, const u32* __restrict__ in_len, u32 B,
                                     u32 kq, u32 q0, u32 q1, const double* __restrict__ amp,
                                     u32 keep_kind, double keep_eps, u32 keep_k,
                                     u64* __restrict__ ox, u64* __restrict__ oz, double* __restrict__ oc,
                                     u64* __restrict__ og, u32* __restrict__ out_start, u32* __restrict__ out_len) {
    const u32 lane = lane_id();
    const u32 beta = blockIdx.x * (blockDim.x / WARP) + warp_id();
    if (beta >= B) return;
    const u32 start = in_start[beta], len = in_len[beta];
    u32 run = 0;
    for (u32 r0 = 0; r0 < len; r0 += WARP) {
        const u32 r = r0 + lane;
        bool ok = false;
        Key k;
        double pr = 0.0, pi = 0.0;
        if (r < len) {
            const size_t src = (size_t)start + r;
            k = load_key(x, z, src);
            const u32 s = support_bits(k, kq, q0, q1);
            const double cr = c[2 * src], ci = c[2 * src + 1];
            const double ar = amp[s * 2], ai = amp[s * 2 + 1];
            pr = cr * ar - ci * ai;
            pi = cr * ai + ci * ar;
            ok = (pr != 0.0 || pi != 0.0) && keep_term(keep_kind, keep_eps, keep_k, k, pr, pi);
        }
        const u32 bal = __ballot_sync(~0u, ok);
        if (ok) {
            const size_t dst = (size_t)start + run + __popc(bal & lanemask_lt());
#pragma unroll
            for (int w = 0; w < W; ++w) {
                ox[dst * W + w] = k.x[w];
                oz[dst * W + w] = k.z[w];
            }
            oc[2 * dst] = pr;
            oc[2 * dst + 1] = pi;
            og[dst] = g[(size_t)start + r];
        }
        run += __popc(bal);
    }
    if (lane == 0) {
        out_start[beta] = start;
        out_len[beta] = run;
    }
}
