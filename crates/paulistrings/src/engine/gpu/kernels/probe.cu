// Proves the NVRTC and launch plumbing for one width; nothing else.
extern "C" __global__ void k_probe(u64 *out, int n) {
    int i = threadIdx.x;
    if (i < n) {
        out[i] = (u64)i * (u64)W;
    }
}
