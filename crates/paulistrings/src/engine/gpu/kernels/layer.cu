// K3: the fused layer, one block per output position (ARCHITECTURE.md §Engine, §GPU-Readiness).
// Records (g_lo32, tag) for every emitting (source bucket, entry) pair, local or received, live in shared memory; a 16-bit index array is radix-sorted over them, equal-g32 runs are checked on the full key, and a segmented sum deduplicates before the surviving rows are written to the arena at seg_start[p].
// Fallbacks are block-uniform: a colliding adjacent pair (equal g_lo32, different key) triggers eight more passes over g_hi32, and a pair still colliding after that a full lex-key sort.

struct Smem {
    u32* sg;        // n_cap
    u16* stag;      // n_cap
    u16* ia;        // n_cap
    u16* ib;        // n_cap
    u16* scnt;      // n_cap / 2 : [digit][it][warp]
    u32* swarp;     // 36
    double* s_amp;  // 512
    u64* s_mask;    // 32 * W
    u64* s_gm;      // 16
    u32* s_bd;      // 16
    u32* s_nz;      // 16
    u32* s_sb;      // 16
    u32* s_rem;     // 16
    u32* s_len;     // 16
    u32* s_ebase;   // 18
    u32* s_flag;    // 4
    double* s_segr; // 32
    double* s_segi; // 32
    u32* s_segf;    // 32
};

__device__ __forceinline__ Smem carve(u8* base, u32 n_cap) {
    Smem s;
    s.sg = (u32*)base;
    s.stag = (u16*)(s.sg + n_cap);
    s.ia = s.stag + n_cap;
    s.ib = s.ia + n_cap;
    s.scnt = s.ib + n_cap;
    s.swarp = (u32*)(s.scnt + n_cap / 2);
    s.s_amp = (double*)(s.swarp + 36);
    s.s_mask = (u64*)(s.s_amp + 512);
    s.s_gm = s.s_mask + 32 * W;
    s.s_bd = (u32*)(s.s_gm + 16);
    s.s_nz = s.s_bd + 16;
    s.s_sb = s.s_nz + 16;
    s.s_rem = s.s_sb + 16;
    s.s_len = s.s_rem + 16;
    s.s_ebase = s.s_len + 16;
    s.s_flag = s.s_ebase + 18;
    s.s_segr = (double*)(s.s_flag + 4);
    s.s_segi = s.s_segr + 32;
    s.s_segf = (u32*)(s.s_segi + 32);
    return s;
}

// One stable counting-sort pass on 4-bit digits; item i = it * THREADS + tid holds record ia[i].
// Ranks come from warp ballots and group counts [digit][it][warp] are scanned digit-major, so the pass is stable and deterministic.
// Only the first n_it chunks hold live records (block-uniform); the pad records beyond them never move and are never read.
// FULL is the n_it == C instantiation with the chunk test compiled out.
template <int C, bool FULL, class DigitFn>
__device__ __forceinline__ void radix_pass_n(u16* ia, u16* ib, u16* scnt, u32* swarp, u32 n_it, DigitFn digit) {
    const u32 tid = threadIdx.x, warp = warp_id();
    const u32 n_groups = C * WARPS;
    for (u32 k = tid; k < 16 * n_groups; k += THREADS) scnt[k] = 0;
    __syncthreads();
    u32 rank[C], dig[C];
    u16 rec[C];
#pragma unroll
    for (int it = 0; it < C; ++it) {
        if (!FULL && (u32)it >= n_it) break;
        const u32 i = it * THREADS + tid;
        const u16 j = ia[i];
        const u32 d = digit((u32)j);
        u32 m = ~0u;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const u32 bal = __ballot_sync(~0u, (d >> b) & 1u);
            m &= ((d >> b) & 1u) ? bal : ~bal;
        }
        const u32 lt = m & lanemask_lt();
        rank[it] = __popc(lt);
        dig[it] = d;
        rec[it] = j;
        if (lt == 0) scnt[d * n_groups + it * WARPS + warp] = (u16)__popc(m);
    }
    __syncthreads();
    block_exscan<(C >= 2 ? C / 2 : 1), u16>(scnt, 16 * n_groups, swarp);
#pragma unroll
    for (int it = 0; it < C; ++it) {
        if (!FULL && (u32)it >= n_it) break;
        const u32 dest = (u32)scnt[dig[it] * n_groups + it * WARPS + warp] + rank[it];
        ib[dest] = rec[it];
    }
    __syncthreads();
}

template <int C, class DigitFn>
__device__ __forceinline__ void radix_pass(u16* ia, u16* ib, u16* scnt, u32* swarp, u32 n_it, DigitFn digit) {
    if (n_it >= (u32)C) radix_pass_n<C, true>(ia, ib, scnt, swarp, n_it, digit);
    else radix_pass_n<C, false>(ia, ib, scnt, swarp, n_it, digit);
}

template <int C, bool SEGSCAN>
__device__ __forceinline__ void layer_body(
    const u64* __restrict__ x, const u64* __restrict__ z, const double* __restrict__ c, const u64* __restrict__ g,
    const u32* __restrict__ in_start, const u32* __restrict__ in_len, const u32* __restrict__ bucket_at,
    const u32* __restrict__ cnt, const u32* __restrict__ seg_start,
    u32 mode, u32 E, u32 kq, u32 q0, u32 q1, double rcos, double rsin,
    const double* __restrict__ amp, const u64* __restrict__ mask, const u32* __restrict__ nz,
    const u32* __restrict__ bd, const u64* __restrict__ gm,
    const u32* __restrict__ rem, const u32* __restrict__ recv_off, const u32* __restrict__ rbase, u32 B,
    const u64* __restrict__ rx, const u64* __restrict__ rz, const double* __restrict__ rc, const u64* __restrict__ rg,
    const KeepProg& prog, u32 p0,
    u64* __restrict__ out_x, u64* __restrict__ out_z, double* __restrict__ out_c, u64* __restrict__ out_g,
    u32* __restrict__ out_len_pos, u32* __restrict__ fallback) {
    extern __shared__ __align__(16) u8 smem_raw[];
    const u32 n_cap = C * THREADS;
    Smem S = carve(smem_raw, n_cap);
    const u32 tid = threadIdx.x, lane = lane_id(), warp = warp_id();
    const u32 p = p0 + blockIdx.x;
    const u32 beta = bucket_at[p];
    const u32 seg0 = seg_start[p] - seg_start[p0];
    const u32 n_rows = seg_start[p + 1] - seg_start[p];
    if (n_rows == 0) {
        if (tid == 0) out_len_pos[p] = 0;
        return;
    }

    // A rotation table never reads `amp`.
    if (mode == MODE_LOCAL) for (u32 i = tid; i < 512; i += THREADS) S.s_amp[i] = amp[i];
    for (u32 i = tid; i < 32 * W; i += THREADS) S.s_mask[i] = mask[i];
    // A received entry's source is segment p of its block: rows already shifted and multiplied by the sender.
    if (tid < MAX_ENTRIES) {
        S.s_gm[tid] = gm[tid] & fp_mask();
        S.s_bd[tid] = bd[tid];
        S.s_nz[tid] = nz[tid];
        const u32 k = rem[tid];
        S.s_rem[tid] = k;
        u32 sb_start = 0, sb_len = 0;
        if (tid < E) {
            if (k == NO_REMOTE) {
                const u32 sb = beta ^ bd[tid];
                sb_start = in_start[sb];
                sb_len = in_len[sb];
            } else {
                const size_t o = (size_t)k * (B + 1) + p;
                const u32 lo = recv_off[o];
                sb_start = rbase[k] + lo;
                sb_len = recv_off[o + 1] - lo;
            }
        }
        S.s_sb[tid] = sb_start;
        S.s_len[tid] = sb_len;
    }
    __syncthreads();
    if (tid == 0) {
        u32 run = 0;
        for (u32 e = 0; e < E; ++e) {
            S.s_ebase[e] = run;
            run += (S.s_rem[e] == NO_REMOTE) ? cnt[(size_t)(beta ^ S.s_bd[e]) * E + e] : S.s_len[e];
        }
        S.s_ebase[E] = run;
        S.s_flag[0] = 0;
    }
    __syncthreads();
    Table T;
    T.mode = mode;
    T.entries = E;
    T.kq = kq;
    T.q0 = q0;
    T.q1 = q1;
    T.rot_cos = rcos;
    T.rot_sin = rsin;
    T.amp = S.s_amp;
    T.mask = S.s_mask;
    T.nz = S.s_nz;

    // Build: one warp per entry, warp-compacted in source order; a narrow block takes several entries per warp.
    for (u32 e = warp; e < E; e += WARPS) {
        const u32 start = S.s_sb[e];
        const u32 len = S.s_len[e];
        const bool received = S.s_rem[e] != NO_REMOTE;
        u32 wbase = S.s_ebase[e];
        const u64 gme = S.s_gm[e];
        for (u32 r0 = 0; r0 < len; r0 += WARP) {
            const u32 r = r0 + lane;
            bool ok = false;
            u64 gv = 0;
            if (r < len) {
                if (received) {
                    ok = true;
                    gv = rg[start + r];
                } else {
                    const Key k = load_key(x, z, start + r);
                    ok = entry_emits(T, k, e);
                    if (ok) gv = g[start + r] ^ gme;
                }
            }
            const u32 bal = __ballot_sync(~0u, ok);
            if (ok) {
                const u32 idx = wbase + __popc(bal & lanemask_lt());
                S.sg[idx] = (u32)gv;
                S.stag[idx] = (u16)((e << TAG_OFF_BITS) | r);
            }
            wbase += __popc(bal);
        }
    }
    for (u32 i = n_rows + tid; i < n_cap; i += THREADS) {
        S.sg[i] = 0xFFFFFFFFu;
        S.stag[i] = 0xFFFFu;
    }
    for (u32 i = tid; i < n_cap; i += THREADS) S.ia[i] = (u16)i;
    __syncthreads();

    u16* ia = S.ia;
    u16* ib = S.ib;
    u16* stag = S.stag;
    u32* sg = S.sg;
    const u32* s_sb = S.s_sb;
    const u32* s_rem = S.s_rem;
    const u64* s_mask = S.s_mask;
    const u64* s_gm = S.s_gm;

    auto src_of = [&](u32 j, u32& e, bool& received) -> u32 {
        const u32 t = stag[j];
        e = t >> TAG_OFF_BITS;
        received = s_rem[e] != NO_REMOTE;
        return s_sb[e] + (t & TAG_OFF_MASK);
    };
    auto out_key = [&](u32 j) -> Key {
        u32 e;
        bool received;
        const u32 src = src_of(j, e, received);
        if (received) return load_key(rx, rz, src);
        Key k = load_key(x, z, src);
#pragma unroll
        for (int w = 0; w < W; ++w) {
            k.x[w] ^= s_mask[(e * 2) * W + w];
            k.z[w] ^= s_mask[(e * 2 + 1) * W + w];
        }
        return k;
    };
    auto g_full = [&](u32 j) -> u64 {
        u32 e;
        bool received;
        const u32 src = src_of(j, e, received);
        return received ? rg[src] : (g[src] ^ s_gm[e]);
    };

    const u32 n_it = (n_rows + THREADS - 1) / THREADS;
    for (int pass = 0; pass < 8; ++pass) {
        radix_pass<C>(ia, ib, S.scnt, S.swarp, n_it, [&](u32 j) -> u32 { return (sg[j] >> (4 * pass)) & 15u; });
        u16* t = ia; ia = ib; ib = t;
    }

    // Adjacent equal-g32 pairs with different keys mean the 32-bit prefix did not separate distinct keys.
    for (u32 i = tid + 1; i < n_rows; i += THREADS) {
        const u32 j0 = ia[i - 1], j1 = ia[i];
        if (sg[j0] == sg[j1] && !key_eq(out_key(j0), out_key(j1))) atomicOr(S.s_flag, 1u);
    }
    __syncthreads();
    // Pad records (j >= n_rows) must stay last: every fallback digit source gives them the maximum digit.
    const bool bykey = S.s_flag[0] != 0;
    if (bykey) {
        if (tid == 0) atomicAdd(fallback, 1u);
        for (int pass = 0; pass < 8; ++pass) {
            radix_pass<C>(ia, ib, S.scnt, S.swarp, n_it, [&](u32 j) -> u32 {
                return j >= n_rows ? 15u : (u32)(g_full(j) >> (32 + 4 * pass)) & 15u;
            });
            u16* t = ia; ia = ib; ib = t;
        }
        if (tid == 0) S.s_flag[0] = 0;
        __syncthreads();
        for (u32 i = tid + 1; i < n_rows; i += THREADS) {
            const u32 j0 = ia[i - 1], j1 = ia[i];
            if (g_full(j0) == g_full(j1) && !key_eq(out_key(j0), out_key(j1))) atomicOr(S.s_flag, 1u);
        }
        __syncthreads();
        if (S.s_flag[0] != 0) {
            if (tid == 0) atomicAdd(fallback + 1, 1u);
            for (int pass = 0; pass < 32 * W; ++pass) {
                const u32 wi = pass / 16, nib = pass % 16;
                radix_pass<C>(ia, ib, S.scnt, S.swarp, n_it, [&](u32 j) -> u32 {
                    if (j >= n_rows) return 15u;
                    const Key k = out_key(j);
                    const u64 word = (wi < W) ? k.z[W - 1 - wi] : k.x[2 * W - 1 - wi];
                    return (u32)(word >> (4 * nib)) & 15u;
                });
                u16* t = ia; ia = ib; ib = t;
            }
        }
    }

    // Heads of equal-key runs, into ib; only ib[0..n_rows) is read below.
    for (u32 i = tid; i < n_rows; i += THREADS) {
        u16 h = 1;
        if (i > 0) {
            const u32 j0 = ia[i - 1], j1 = ia[i];
            h = bykey ? (u16)(!key_eq(out_key(j0), out_key(j1))) : (u16)(sg[j0] != sg[j1]);
        }
        ib[i] = h;
    }
    __syncthreads();

    // A received row's product is the identity: the sender already applied the entry.
    auto product = [&](u32 j, double& pr, double& pi) {
        u32 e;
        bool received;
        const u32 src = src_of(j, e, received);
        if (received) {
            pr = rc[2 * (size_t)src];
            pi = rc[2 * (size_t)src + 1];
            return;
        }
        const Key k = load_key(x, z, src);
        entry_product(T, k, e, c[2 * (size_t)src], c[2 * (size_t)src + 1], pr, pi);
    };
    auto write_row = [&](u32 slot, u32 j, double ar, double ai) {
        u32 e;
        bool received;
        const u32 src = src_of(j, e, received);
        if (received) {
            const Key k = load_key(rx, rz, src);
#pragma unroll
            for (int w = 0; w < W; ++w) {
                out_x[(size_t)slot * W + w] = k.x[w];
                out_z[(size_t)slot * W + w] = k.z[w];
            }
            out_g[slot] = rg[src];
        } else {
            const Key k = load_key(x, z, src);
#pragma unroll
            for (int w = 0; w < W; ++w) {
                out_x[(size_t)slot * W + w] = k.x[w] ^ s_mask[(e * 2) * W + w];
                out_z[(size_t)slot * W + w] = k.z[w] ^ s_mask[(e * 2 + 1) * W + w];
            }
            out_g[slot] = g[src] ^ s_gm[e];
        }
        out_c[2 * (size_t)slot] = ar;
        out_c[2 * (size_t)slot + 1] = ai;
    };
    auto survives = [&](u32 j, double ar, double ai) -> bool {
        if (ar == 0.0 && ai == 0.0) return false;
        return keep_eval(prog, [&]() -> Key { return out_key(j); }, ar, ai);
    };

    if (SEGSCAN) {
        // Block-parallel segmented sum: every item forms its product, and a segmented inclusive scan in sorted order lands each run's total on its last item.
        const u32 base = tid * C;
        double pr[C], pi[C];
        bool head[C];
#pragma unroll
        for (int it = 0; it < C; ++it) {
            const u32 i = base + it;
            pr[it] = 0.0;
            pi[it] = 0.0;
            head[it] = true;
            if (i < n_rows) {
                product(ia[i], pr[it], pi[it]);
                head[it] = ib[i] != 0;
            }
        }
        double tr = 0.0, ti = 0.0;
        bool tf = false;
#pragma unroll
        for (int it = 0; it < C; ++it) {
            if (head[it]) { tr = pr[it]; ti = pi[it]; tf = true; }
            else { tr += pr[it]; ti += pi[it]; }
        }
        // (sum, has_head) segmented scan: a range with a head absorbs nothing from its left.
        double sr = tr, si = ti;
        bool sf = tf;
#pragma unroll
        for (u32 o = 1; o < 32; o <<= 1) {
            const double r2 = __shfl_up_sync(~0u, sr, o);
            const double i2 = __shfl_up_sync(~0u, si, o);
            const u32 f2 = __shfl_up_sync(~0u, (u32)sf, o);
            if (lane >= o && !sf) { sr += r2; si += i2; sf = f2 != 0; }
        }
        double er = __shfl_up_sync(~0u, sr, 1), ei = __shfl_up_sync(~0u, si, 1);
        u32 ef = __shfl_up_sync(~0u, (u32)sf, 1);
        if (lane == 0) { er = 0.0; ei = 0.0; ef = 0; }
        if (lane == 31) { S.s_segr[warp] = sr; S.s_segi[warp] = si; S.s_segf[warp] = sf ? 1u : 0u; }
        __syncthreads();
        if (warp == 0) {
            const u32 nw = blockDim.x >> 5;
            double vr = lane < nw ? S.s_segr[lane] : 0.0, vi = lane < nw ? S.s_segi[lane] : 0.0;
            bool vf = lane < nw ? S.s_segf[lane] != 0 : true;
#pragma unroll
            for (u32 o = 1; o < 32; o <<= 1) {
                const double r2 = __shfl_up_sync(~0u, vr, o);
                const double i2 = __shfl_up_sync(~0u, vi, o);
                const u32 f2 = __shfl_up_sync(~0u, (u32)vf, o);
                if (lane >= o && !vf) { vr += r2; vi += i2; vf = f2 != 0; }
            }
            double xr = __shfl_up_sync(~0u, vr, 1), xi = __shfl_up_sync(~0u, vi, 1);
            if (lane == 0) { xr = 0.0; xi = 0.0; }
            S.s_segr[lane] = xr;
            S.s_segi[lane] = xi;
        }
        __syncthreads();
        double run_r = er, run_i = ei;
        if (!ef) { run_r += S.s_segr[warp]; run_i += S.s_segi[warp]; }
        bool keep[C];
#pragma unroll
        for (int it = 0; it < C; ++it) {
            const u32 i = base + it;
            if (head[it]) { run_r = pr[it]; run_i = pi[it]; }
            else { run_r += pr[it]; run_i += pi[it]; }
            const bool last = (i + 1 == n_rows) || (i + 1 < n_rows && ib[i + 1] != 0);
            keep[it] = (i < n_rows) && last && survives(ia[i], run_r, run_i);
            pr[it] = run_r;
            pi[it] = run_i;
        }
        __syncthreads();
#pragma unroll
        for (int it = 0; it < C; ++it) sg[base + it] = keep[it] ? 1u : 0u;
        __syncthreads();
        const u32 total = block_exscan<C, u32>(sg, n_cap, S.swarp);
        if (tid == 0) out_len_pos[p] = total;
#pragma unroll
        for (int it = 0; it < C; ++it) {
            if (!keep[it]) continue;
            const u32 i = base + it;
            write_row(seg0 + sg[i], ia[i], pr[it], pi[it]);
        }
    } else {
        // Segmented sum per head, sequential over the run, products recomputed from the input.
        double accr[C], acci[C];
        bool keep[C];
#pragma unroll
        for (int it = 0; it < C; ++it) {
            const u32 i = it * THREADS + tid;
            keep[it] = false;
            accr[it] = 0.0;
            acci[it] = 0.0;
            if (i < n_rows && ib[i]) {
                u32 j = i;
                bool first = true;
                double ar = 0.0, ai = 0.0;
                do {
                    double pr, pi;
                    product(ia[j], pr, pi);
                    if (first) { ar = pr; ai = pi; first = false; }
                    else { ar += pr; ai += pi; }
                    ++j;
                } while (j < n_rows && !ib[j]);
                accr[it] = ar;
                acci[it] = ai;
                keep[it] = survives(ia[i], ar, ai);
            }
        }
        __syncthreads();
#pragma unroll
        for (int it = 0; it < C; ++it) sg[it * THREADS + tid] = keep[it] ? 1u : 0u;
        __syncthreads();
        const u32 total = block_exscan<C, u32>(sg, n_cap, S.swarp);
        if (tid == 0) out_len_pos[p] = total;
#pragma unroll
        for (int it = 0; it < C; ++it) {
            if (!keep[it]) continue;
            const u32 i = it * THREADS + tid;
            write_row(seg0 + sg[i], ia[i], accr[it], acci[it]);
        }
    }
}

#define LAYER_KERNEL(C, SEG, NAME)                                                                         \
    extern "C" __global__ void __launch_bounds__(THREADS) NAME(                                            \
        const u64* x, const u64* z, const double* c, const u64* g, const u32* in_start, const u32* in_len, \
        const u32* bucket_at, const u32* cnt, const u32* seg_start, u32 mode, u32 E, u32 kq, u32 q0,       \
        u32 q1, double rcos, double rsin, const double* amp, const u64* mask, const u32* nz,               \
        const u32* bd, const u64* gm, const u32* rem, const u32* recv_off, const u32* rbase, u32 B,         \
        const u64* rx, const u64* rz, const double* rc, const u64* rg,                                     \
        const __grid_constant__ KeepProg prog, u32 p0, u64* out_x,                                         \
        u64* out_z, double* out_c, u64* out_g, u32* out_len_pos, u32* fallback) {                          \
        layer_body<C, SEG>(x, z, c, g, in_start, in_len, bucket_at, cnt, seg_start, mode, E, kq, q0, q1,   \
                           rcos, rsin, amp, mask, nz, bd, gm, rem, recv_off, rbase, B, rx, rz, rc, rg,     \
                           prog, p0, out_x, out_z, out_c, out_g, out_len_pos, fallback);                   \
    }

#define LAYER_PAIR(C) LAYER_KERNEL(C, false, k_layer_serial_##C) LAYER_KERNEL(C, true, k_layer_segscan_##C)

// Every items-per-thread variant whose record capacity fits CAP at this block width.
LAYER_PAIR(1)
LAYER_PAIR(2)
LAYER_PAIR(4)
LAYER_PAIR(8)
#if THREADS * 16 <= CAP
LAYER_PAIR(16)
#endif
#if THREADS * 32 <= CAP
LAYER_PAIR(32)
#endif
