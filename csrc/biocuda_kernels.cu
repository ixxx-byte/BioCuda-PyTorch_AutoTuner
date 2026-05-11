#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void xor_permute_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, int64_t n, int mask) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int group = tid & ~31;
    int lane = tid & 31;
    y[tid] = x[group + (lane ^ mask)];
}

template <typename scalar_t>
__global__ void reverse32_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, int64_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int group = tid & ~31;
    int lane = tid & 31;
    y[tid] = x[group + (31 - lane)];
}

__global__ void hamming_popc_kernel(const unsigned int* __restrict__ a,
                                    const unsigned int* __restrict__ b,
                                    unsigned long long* __restrict__ partial,
                                    int64_t n_words) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned long long acc = 0ULL;
    for (int64_t i = tid; i < n_words; i += stride) {
        acc += static_cast<unsigned long long>(__popc(a[i] ^ b[i]));
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(partial, acc);
    }
}

__global__ void sw_score_kernel(const unsigned char* __restrict__ A,
                                const unsigned char* __restrict__ B,
                                int la,
                                int lb,
                                int match,
                                int mismatch,
                                int gap_open,
                                int gap_extend,
                                int* __restrict__ out_score) {
    extern __shared__ int smem[];
    int* H = smem;
    int* E = H + (la + 1);
    int* warp_best = E + (la + 1);
    int tid = threadIdx.x;
    int bs = blockDim.x;
    int n_warps = (bs + 31) / 32;
    int warp_id = tid / 32;
    int lane = tid & 31;
    int best = 0;

    for (int i = tid; i <= la; i += bs) {
        H[i] = 0;
        E[i] = 0;
    }
    if (tid < n_warps) warp_best[tid] = 0;
    __syncthreads();

    if (tid == 0) {
        for (int j = 1; j <= lb; ++j) {
            unsigned char bj = B[j - 1];
            int prev_left = 0;
            int prev_diag = H[0];
            int F = -1073741824;
            for (int i = 1; i <= la; ++i) {
                int old_H = H[i];
                int diag = prev_diag;
                prev_diag = old_H;
                int s = (A[i - 1] == bj) ? match : mismatch;
                F = max(prev_left - gap_open - gap_extend, F - gap_extend);
                int Eij = max(old_H - gap_open - gap_extend, E[i] - gap_extend);
                int v = max(0, max(diag + s, max(F, Eij)));
                prev_left = v;
                E[i] = Eij;
                H[i] = v;
                if (v > best) best = v;
            }
        }
    }
    __syncthreads();

    if (tid == 0) warp_best[0] = best;
    __syncthreads();
    int my_best = (lane == 0 && warp_id < n_warps) ? warp_best[warp_id] : 0;
    for (int off = 16; off > 0; off >>= 1) {
        int other = __shfl_xor_sync(0xffffffffu, my_best, off);
        if (other > my_best) my_best = other;
    }
    if (tid == 0) atomicMax(out_score, my_best);
}

__global__ void affine_scan32_kernel(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ out_a,
                                     float* __restrict__ out_b,
                                     int64_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int lane = threadIdx.x & 31;
    float av = a[tid];
    float bv = b[tid];

    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        float a_prev = __shfl_up_sync(0xffffffffu, av, off);
        float b_prev = __shfl_up_sync(0xffffffffu, bv, off);
        if (lane >= off) {
            bv = av * b_prev + bv;
            av = av * a_prev;
        }
    }
    out_a[tid] = av;
    out_b[tid] = bv;
}

torch::Tensor hamming_popc_cuda(torch::Tensor a, torch::Tensor b, int64_t block_size) {
    auto out = torch::zeros({1}, torch::TensorOptions().device(a.device()).dtype(torch::kUInt64));
    int threads = static_cast<int>(block_size);
    int blocks = std::max<int64_t>(1, std::min<int64_t>(1024, (a.numel() + threads - 1) / threads));
    hamming_popc_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const unsigned int*>(a.data_ptr()),
        reinterpret_cast<const unsigned int*>(b.data_ptr()),
        reinterpret_cast<unsigned long long*>(out.data_ptr()),
        a.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor sw_score_cuda(torch::Tensor a, torch::Tensor b, int64_t match, int64_t mismatch, int64_t gap_open, int64_t gap_extend, int64_t block_size) {
    TORCH_CHECK(a.numel() <= 65535, "current SW kernel is intended for short sequences; la must fit shared-memory indexing");
    int block = static_cast<int>(block_size);
    int n_warps = (block + 31) / 32;
    int smem_bytes = static_cast<int>((2 * (a.numel() + 1) + n_warps) * sizeof(int));
    auto out = torch::zeros({1}, torch::TensorOptions().device(a.device()).dtype(torch::kInt32));
    sw_score_kernel<<<1, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const unsigned char*>(a.data_ptr()),
        reinterpret_cast<const unsigned char*>(b.data_ptr()),
        static_cast<int>(a.numel()),
        static_cast<int>(b.numel()),
        static_cast<int>(match),
        static_cast<int>(mismatch),
        static_cast<int>(gap_open),
        static_cast<int>(gap_extend),
        reinterpret_cast<int*>(out.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor reverse32_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int threads = 256;
    int blocks = std::max<int64_t>(1, (x.numel() + threads - 1) / threads);
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Byte, x.scalar_type(), "reverse32_cuda", [&] {
        reverse32_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), x.numel());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor xor_permute_cuda(torch::Tensor x, int64_t mask) {
    auto y = torch::empty_like(x);
    int threads = 256;
    int blocks = std::max<int64_t>(1, (x.numel() + threads - 1) / threads);
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Byte, x.scalar_type(), "xor_permute_cuda", [&] {
        xor_permute_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), x.numel(), static_cast<int>(mask));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor affine_scan32_cuda(torch::Tensor a, torch::Tensor b) {
    auto out_a = torch::empty_like(a);
    auto out_b = torch::empty_like(b);
    int threads = 256;
    int blocks = std::max<int64_t>(1, (a.numel() + threads - 1) / threads);
    affine_scan32_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out_a.data_ptr<float>(), out_b.data_ptr<float>(), a.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return torch::stack({out_a, out_b}, 0);
}
