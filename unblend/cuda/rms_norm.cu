// RoFormer RMSNorm over the contiguous last dimension.
// CUDA port of ``unblend/metal/rms_norm.metal``.
//
// One block handles one row. Inputs and affine weights may be FP32, FP16,
// or BF16, but the sum-of-squares reduction and affine arithmetic stay in
// FP32 to match ``RMSNorm.forward``.

#include "bindings.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "kernels.cuh"

namespace {

constexpr unsigned int kMaxTgs = 1024;

template <typename SCALAR_T>
__global__ void rms_norm_kernel(
    SCALAR_T* __restrict__  out,
    const SCALAR_T* __restrict__  in_,
    const SCALAR_T* __restrict__  gamma,
    unsigned int dim,
    float scale
) {
    __shared__ float shared_sqsum[kMaxTgs];
    __shared__ float multiplier;

    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int row = blockIdx.x;
    const unsigned long long base = (unsigned long long)row * dim;

    float local_sqsum = 0.0f;
    for (unsigned int i = tid; i < dim; i += tgs) {
        const float value = static_cast<float>(in_[base + i]);
        local_sqsum += value * value;
    }
    shared_sqsum[tid] = local_sqsum;
    __syncthreads();

    for (unsigned int stride = tgs >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sqsum[tid] += shared_sqsum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // F.normalize divides by max(L2 norm, 1e-12), then RoFormer
        // multiplies by sqrt(dim). Keep that exact convention rather than
        // introducing the additive epsilon used by other RMSNorm variants.
        const float norm = sqrtf(shared_sqsum[0]);
        multiplier = scale / fmaxf(norm, 1.0e-12f);
    }
    __syncthreads();

    for (unsigned int i = tid; i < dim; i += tgs) {
        const float value = static_cast<float>(in_[base + i]);
        const float gain = static_cast<float>(gamma[i]);
        out[base + i] = static_cast<SCALAR_T>(value * multiplier * gain);
    }
}

template <typename SCALAR_T>
void rms_norm_impl(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& gamma,
    int64_t dim, double scale, int64_t tgs
) {
    const int64_t rows = in_.numel() / dim;
    const dim3 grid((unsigned int)rows);
    const dim3 block((unsigned int)tgs);
    rms_norm_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<SCALAR_T>(), in_.const_data_ptr<SCALAR_T>(),
        gamma.const_data_ptr<SCALAR_T>(), (unsigned int)dim, (float)scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void rms_norm(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& gamma,
    int64_t dim, double scale, int64_t tgs
) {
    TORCH_CHECK(in_.is_cuda() && out.is_cuda() && gamma.is_cuda(),
                "rms_norm: tensors must be CUDA");
    TORCH_CHECK(
        in_.scalar_type() == out.scalar_type() &&
            gamma.scalar_type() == in_.scalar_type(),
        "rms_norm: dtype mismatch");
    TORCH_CHECK(tgs >= 1 && tgs <= 1024 && (tgs & (tgs - 1)) == 0,
                "rms_norm: block size must be a power of two <= 1024");
    UNBLEND_DISPATCH(rms_norm_impl, in_, out, in_, gamma, dim, scale, tgs)
}
