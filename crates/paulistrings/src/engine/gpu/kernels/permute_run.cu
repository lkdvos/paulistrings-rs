// A run of one-to-one steps (`PermutationStep`, engine/bucketed/permutation_run.rs) walked per string in one pass; launch and contracts in engine/gpu/layer/run.rs.
// Per step, RUN_INFO_WORDS words of `info` (qubit_0, qubit_1, the second qubit's pattern mask, the inert patterns, 16 packed local deltas, 16 bucket deltas), then 16 fingerprint deltas and 16 amplitudes.
// k_permute_run_count counts every surviving string into its destination bucket, and k_permute_run_scatter walks again and writes each to a slot claimed from that bucket's cursor, so a bucket's order is the arrival order, which nothing reads.

#define RUN_INFO_WORDS 24

struct RunSteps {
    const u32* info;
    const u64* fingerprint;
    const real_t* amplitude;
    u32 n;
};

__device__ __forceinline__ void run_flip(u64 (&v)[W], u32 q, u32 bit) {
#pragma unroll
    for (int w = 0; w < W; ++w) v[w] ^= ((u32)w == (q >> 6)) ? ((u64)bit << (q & 63)) : 0ull;
}

// Walk (k, c) of bucket beta with fingerprint g through every step as `walk_bucket` does, asking `survives` after each step that changes the string and once more about a string none changes; false when it drops.
__device__ __forceinline__ bool run_walk(const RunSteps& R, const KeepProg& prog, Key& k, Coeff& c, u32& beta, u64& g) {
    bool tested = false;
    for (u32 i = 0; i < R.n; ++i) {
        const u32* step = R.info + (size_t)i * RUN_INFO_WORDS;
        const u32 q0 = step[0], q1 = step[1];
        const u32 s = key_bit(k.x, q0) | key_bit(k.z, q0) << 1 | ((key_bit(k.x, q1) | key_bit(k.z, q1) << 1) & step[2]) << 2;
        if ((step[3] >> s) & 1u) continue;
        if (!tested && i > 0 && !survives(prog, k, c)) return false;
        const u32 d = (step[4 + (s >> 2)] >> (8 * (s & 3))) & 0xFFu;
        run_flip(k.x, q0, d & 1u);
        run_flip(k.z, q0, (d >> 1) & 1u);
        run_flip(k.x, q1, (d >> 2) & 1u);
        run_flip(k.z, q1, (d >> 3) & 1u);
        beta ^= step[8 + s];
        g ^= R.fingerprint[(size_t)i * LOCAL_DIM + s];
        c = coeff_mul(c, coeff_load(R.amplitude, (size_t)i * LOCAL_DIM + s));
        if (!survives(prog, k, c)) return false;
        tested = true;
    }
    return tested || R.n == 0 || survives(prog, k, c);
}

#define RUN_STEP_ARGS const u32* __restrict__ info, const u64* __restrict__ fingerprint, const real_t* __restrict__ amplitude, u32 n_steps
#define MAKE_RUN(R) const RunSteps R = {info, fingerprint, amplitude, n_steps}

// k_permute_run_count: out_len[beta'] += the rows of bucket blockIdx.x that survive the run and end in bucket beta'; out_len starts zeroed.
extern "C" __global__ void __launch_bounds__(THREADS) k_permute_run_count(
    const u64* __restrict__ x, const u64* __restrict__ z, const real_t* __restrict__ c,
    const u32* __restrict__ in_start, const u32* __restrict__ in_len, RUN_STEP_ARGS,
    const __grid_constant__ KeepProg prog, u32* __restrict__ out_len) {
    MAKE_RUN(R);
    const u32 beta = blockIdx.x;
    const u32 start = in_start[beta], len = in_len[beta];
    for (u32 r0 = 0; r0 < len; r0 += blockDim.x) {
        const u32 r = r0 + threadIdx.x;
        u32 destination = NO_ENTRY;
        if (r < len) {
            Key k = load_key(x, z, (size_t)start + r);
            Coeff v = coeff_load(c, (size_t)start + r);
            u32 b = beta;
            u64 g = 0;
            if (run_walk(R, prog, k, v, b, g)) destination = b;
        }
        count_by_destination(out_len, destination);
    }
}

// k_permute_run_scatter: as k_permute_run_count, writing each survivor at a slot claimed from cursor[beta'], which starts at the output bucket's start.
extern "C" __global__ void __launch_bounds__(THREADS) k_permute_run_scatter(
    const u64* __restrict__ x, const u64* __restrict__ z, const real_t* __restrict__ c, const u64* __restrict__ g_in,
    const u32* __restrict__ in_start, const u32* __restrict__ in_len, RUN_STEP_ARGS,
    const __grid_constant__ KeepProg prog, u32* __restrict__ cursor,
    u64* __restrict__ ox, u64* __restrict__ oz, real_t* __restrict__ oc, u64* __restrict__ og) {
    MAKE_RUN(R);
    const u32 beta = blockIdx.x;
    const u32 start = in_start[beta], len = in_len[beta];
    for (u32 r0 = 0; r0 < len; r0 += blockDim.x) {
        const u32 r = r0 + threadIdx.x;
        u32 destination = NO_ENTRY;
        Key k;
        Coeff v = coeff_zero();
        u64 g = 0;
        if (r < len) {
            k = load_key(x, z, (size_t)start + r);
            v = coeff_load(c, (size_t)start + r);
            u32 b = beta;
            if (run_walk(R, prog, k, v, b, g)) destination = b;
        }
        const size_t slot = claim_by_destination(cursor, destination);
        if (destination != NO_ENTRY) {
            store_key(ox, oz, slot, k);
            coeff_store(oc, slot, v);
            og[slot] = g_in[(size_t)start + r] ^ (g & fingerprint_mask());
        }
    }
}
