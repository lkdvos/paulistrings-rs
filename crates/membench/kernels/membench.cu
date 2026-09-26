// STREAM read / write / copy / triad over f64 arrays, grid-stride, one launch per pass.
typedef unsigned long long u64;

extern "C" __global__ void k_fill(double* __restrict__ a, double v, u64 n) {
    for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) a[i] = v;
}

// Block partial sums land in out[blockIdx.x]; the host never reads them, they only keep the loads alive.
extern "C" __global__ void k_read(const double* __restrict__ a, double* __restrict__ out, u64 n) {
    __shared__ double s[32];
    double acc = 0.0;
    for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) acc += a[i];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(~0u, acc, o);
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        double v = threadIdx.x < (blockDim.x >> 5) ? s[threadIdx.x] : 0.0;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
        if (threadIdx.x == 0) out[blockIdx.x] = v;
    }
}

extern "C" __global__ void k_write(double* __restrict__ a, double v, u64 n) {
    for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) a[i] = v;
}

extern "C" __global__ void k_copy(double* __restrict__ a, const double* __restrict__ b, u64 n) {
    for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) a[i] = b[i];
}

extern "C" __global__ void k_triad(double* __restrict__ a, const double* __restrict__ b, const double* __restrict__ c,
                                   double v, u64 n) {
    for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) a[i] = b[i] + v * c[i];
}
