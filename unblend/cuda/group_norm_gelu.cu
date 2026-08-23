// GroupNorm fused with GELU activation.
// CUDA port of ``unblend/metal/group_norm_gelu.metal``.
//
// Saves the round-trip that PyTorch would otherwise spend on the explicit
// ``functional.gelu(...)`` op after every ``norm1`` call inside HEncLayer /
// HDecLayer / DConv. We use the tanh approximation (the same form as
// ``F.gelu(approximate='tanh')``) to match the Metal kernels bit-for-bit in
// spirit; the per-element gap from PyTorch's default exact-erf GELU peaks at
// ~1e-3 (near |x|≈2), which is below FP16/BF16 output precision, so this
// path is numerically equivalent to the reference at the dtypes it runs in.
// (The FP32 fallback in ``unblend/cuda/__init__.py`` uses exact erf.)
//
// ``apply_norm_gelu`` is the third stage of the multi-stage path; its
// mean/scale come from ``finalize_meanvar`` in ``group_norm.cu``.
// Vector/scalar path selection and the reduction helpers are shared via
// ``kernels.cuh``.

#include "bindings.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>

#include "kernels.cuh"

namespace {

__device__ __forceinline__ float gelu_tanh(float y) {
    // sqrt(2/pi) = 0.7978845608028654
    const float inner = 0.7978845608028654f * (y + 0.044715f * y * y * y);
    return 0.5f * y * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float4 gelu_tanh4(const float4& y) {
    const float inner_x = 0.7978845608028654f * (y.x + 0.044715f * y.x * y.x * y.x);
    const float inner_y = 0.7978845608028654f * (y.y + 0.044715f * y.y * y.y * y.y);
    const float inner_z = 0.7978845608028654f * (y.z + 0.044715f * y.z * y.z * y.z);
    const float inner_w = 0.7978845608028654f * (y.w + 0.044715f * y.w * y.w * y.w);
    return make_float4(
        0.5f * y.x * (1.0f + tanhf(inner_x)),
        0.5f * y.y * (1.0f + tanhf(inner_y)),
        0.5f * y.z * (1.0f + tanhf(inner_z)),
        0.5f * y.w * (1.0f + tanhf(inner_w))
    );
}

template <typename SCALAR_T>
__global__ void group_norm_g1_gelu_kernel(
    SCALAR_T* __restrict__  out,
    const SCALAR_T* __restrict__  in_,
    const SCALAR_T* __restrict__ inject,  // optional second input added first
    const SCALAR_T* __restrict__  weight,
    const SCALAR_T* __restrict__  bias,
    unsigned int C,
    unsigned int N,
    float eps
) {
    const bool has_inj = inject != nullptr;
    __shared__ float sh_sum[MAX_WARPS];
    __shared__ float sh_sq[MAX_WARPS];
    __shared__ float bcast[2];

    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int b = blockIdx.x;

    const unsigned int total = C * N;
    // 64-bit base offsets: b * total overflows 32 bits on huge inputs.
    const SCALAR_T* __restrict__  in_b = in_ + (unsigned long long)b * total;
    SCALAR_T* __restrict__  out_b = out + (unsigned long long)b * total;

    const unsigned long long jbase =
        has_inj ? (unsigned long long)b * total : 0ull;
    const SCALAR_T* __restrict__ j_b =
        has_inj ? inject + jbase : nullptr;
    const Scalar4<SCALAR_T>* __restrict__ j4 =
        has_inj ? reinterpret_cast<const Scalar4<SCALAR_T>*>(j_b) : nullptr;
    float K = static_cast<float>(in_b[0]);
    if (has_inj) {
        K += static_cast<float>(inject[jbase]);
    }
    float s = 0.0f, sq = 0.0f;
    gn_accumulate_sumsq(in_b, j_b, total, K, tid, tgs, s, sq);
    gn_reduce_finalize(s, sq, K, total, eps, sh_sum, sh_sq, bcast);
    const float mean = bcast[0];
    const float scale = bcast[1];

    if ((N & 3u) == 0u) {
        const Scalar4<SCALAR_T>* __restrict__ in4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(in_b);
        Scalar4<SCALAR_T>* __restrict__ out4 =
            reinterpret_cast<Scalar4<SCALAR_T>*>(out_b);
        const unsigned int Nv = N >> 2;
        const unsigned int nv = C * Nv;
        for (unsigned int i = tid; i < nv; i += tgs) {
            const unsigned int c = i / Nv;
            const float w = static_cast<float>(weight[c]);
            const float bv = static_cast<float>(bias[c]);
            float4 v = unpack4(in4[i]);
            if (has_inj) {
                const float4 wj = unpack4(j4[i]);
                v.x += wj.x; v.y += wj.y; v.z += wj.z; v.w += wj.w;
            }
            const float4 r = make_float4(
                (v.x - mean) * scale * w + bv,
                (v.y - mean) * scale * w + bv,
                (v.z - mean) * scale * w + bv,
                (v.w - mean) * scale * w + bv
            );
            out4[i] = pack4<SCALAR_T>(gelu_tanh4(r));
        }
    } else {
        for (unsigned int i = tid; i < total; i += tgs) {
            const unsigned int c = i / N;
            const float w = static_cast<float>(weight[c]);
            const float bv = static_cast<float>(bias[c]);
            float v = static_cast<float>(in_b[i]);
            if (has_inj) {
                v += static_cast<float>(inject[i]);
            }
            const float y = (v - mean) * scale * w + bv;
            out_b[i] = static_cast<SCALAR_T>(gelu_tanh(y));
        }
    }
}

template <typename SCALAR_T>
__global__ void apply_norm_gelu_kernel(
    SCALAR_T* __restrict__  out,
    const SCALAR_T* __restrict__  in_,
    const SCALAR_T* __restrict__ inject,  // optional second input added first
    const float* __restrict__  meanvar,
    const SCALAR_T* __restrict__  weight,
    const SCALAR_T* __restrict__  bias,
    unsigned int total_per_b,
    unsigned int num_tiles,
    unsigned int N
) {
    const bool has_inj = inject != nullptr;
    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int bt = blockIdx.x;
    const unsigned int b = bt / num_tiles;
    const unsigned int t = bt % num_tiles;

    const float mean = meanvar[(unsigned long long)b * 2 + 0];
    const float scale = meanvar[(unsigned long long)b * 2 + 1];

    const SCALAR_T* __restrict__  x_b = in_ + (unsigned long long)b * total_per_b;
    const SCALAR_T* __restrict__ j_b =
        has_inj ? inject + (unsigned long long)b * total_per_b : nullptr;
    SCALAR_T* __restrict__  out_b = out + (unsigned long long)b * total_per_b;

    if ((N & 3u) == 0u) {
        const Scalar4<SCALAR_T>* __restrict__ in4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(x_b);
        const Scalar4<SCALAR_T>* __restrict__ j4 =
            has_inj ? reinterpret_cast<const Scalar4<SCALAR_T>*>(j_b) : nullptr;
        Scalar4<SCALAR_T>* __restrict__ out4 =
            reinterpret_cast<Scalar4<SCALAR_T>*>(out_b);
        const unsigned int Nv = N >> 2;
        const unsigned int nv = total_per_b >> 2;
        const unsigned int start =
            (unsigned int)((unsigned long long)t * nv / num_tiles);
        const unsigned int end =
            (unsigned int)((unsigned long long)(t + 1) * nv / num_tiles);
        for (unsigned int i = start + tid; i < end; i += tgs) {
            const unsigned int c = i / Nv;
            const float w = static_cast<float>(weight[c]);
            const float bv = static_cast<float>(bias[c]);
            float4 v = unpack4(in4[i]);
            if (has_inj) {
                const float4 wj = unpack4(j4[i]);
                v.x += wj.x; v.y += wj.y; v.z += wj.z; v.w += wj.w;
            }
            const float4 r = make_float4(
                (v.x - mean) * scale * w + bv,
                (v.y - mean) * scale * w + bv,
                (v.z - mean) * scale * w + bv,
                (v.w - mean) * scale * w + bv
            );
            out4[i] = pack4<SCALAR_T>(gelu_tanh4(r));
        }
    } else {
        const unsigned int start =
            (unsigned int)((unsigned long long)t * total_per_b / num_tiles);
        const unsigned int end =
            (unsigned int)((unsigned long long)(t + 1) * total_per_b / num_tiles);
        for (unsigned int i = start + tid; i < end; i += tgs) {
            const unsigned int c = i / N;
            const float w = static_cast<float>(weight[c]);
            const float bv = static_cast<float>(bias[c]);
            float v = static_cast<float>(x_b[i]);
            if (has_inj) {
                v += static_cast<float>(j_b[i]);
            }
            const float y = (v - mean) * scale * w + bv;
            out_b[i] = static_cast<SCALAR_T>(gelu_tanh(y));
        }
    }
}

template <typename SCALAR_T>
void group_norm_g1_gelu_impl(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& inject,
    const at::Tensor& weight, const at::Tensor& bias, int64_t C, int64_t N,
    double eps, int64_t tgs
) {
    const dim3 grid((unsigned int)in_.size(0));
    const dim3 block((unsigned int)tgs);
    group_norm_g1_gelu_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<SCALAR_T>(), in_.const_data_ptr<SCALAR_T>(),
        inject.numel() > 0 ? inject.const_data_ptr<SCALAR_T>() : nullptr,
        weight.const_data_ptr<SCALAR_T>(), bias.const_data_ptr<SCALAR_T>(),
        (unsigned int)C, (unsigned int)N, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SCALAR_T>
void apply_norm_gelu_impl(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& inject,
    const at::Tensor& meanvar, const at::Tensor& weight, const at::Tensor& bias,
    int64_t total_per_b, int64_t num_tiles, int64_t N, int64_t tgs
) {
    const dim3 grid((unsigned int)(in_.size(0) * num_tiles));
    const dim3 block((unsigned int)tgs);
    apply_norm_gelu_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<SCALAR_T>(), in_.const_data_ptr<SCALAR_T>(),
        inject.numel() > 0 ? inject.const_data_ptr<SCALAR_T>() : nullptr,
        meanvar.const_data_ptr<float>(), weight.const_data_ptr<SCALAR_T>(),
        bias.const_data_ptr<SCALAR_T>(), (unsigned int)total_per_b,
        (unsigned int)num_tiles, (unsigned int)N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SCALAR_T>
__global__ void add_gelu_kernel(
    SCALAR_T* __restrict__ out,
    const SCALAR_T* __restrict__ a,
    const SCALAR_T* __restrict__ b,
    unsigned int total
) {
    if ((total & 3u) == 0u) {
        const Scalar4<SCALAR_T>* __restrict__ a4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(a);
        const Scalar4<SCALAR_T>* __restrict__ b4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(b);
        Scalar4<SCALAR_T>* __restrict__ o4 =
            reinterpret_cast<Scalar4<SCALAR_T>*>(out);
        const unsigned int nv = total >> 2;
        const unsigned int tgs = blockDim.x;
        for (unsigned int i = blockIdx.x * tgs + threadIdx.x; i < nv;
             i += gridDim.x * tgs) {
            const float4 va = unpack4(a4[i]);
            const float4 vb = unpack4(b4[i]);
            o4[i] = pack4<SCALAR_T>(gelu_tanh4(make_float4(
                va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w)));
        }
    } else {
        const unsigned int tgs = blockDim.x;
        for (unsigned int i = blockIdx.x * tgs + threadIdx.x; i < total;
             i += gridDim.x * tgs) {
            out[i] = static_cast<SCALAR_T>(
                gelu_tanh(static_cast<float>(a[i]) + static_cast<float>(b[i]))
            );
        }
    }
}

template <typename SCALAR_T>
void add_gelu_impl(
    const at::Tensor& out, const at::Tensor& a, const at::Tensor& b
) {
    const int64_t total = a.numel();
    const unsigned int tgs = 256;
    const unsigned int max_blocks = 4096;
    const unsigned int blocks =
        (unsigned int)std::min<int64_t>((total / 4 + tgs - 1) / tgs, max_blocks);
    add_gelu_kernel<SCALAR_T><<<std::max(blocks, 1u), tgs, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<SCALAR_T>(), a.const_data_ptr<SCALAR_T>(),
        b.const_data_ptr<SCALAR_T>(), (unsigned int)total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void group_norm_g1_gelu(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& inject,
    const at::Tensor& weight, const at::Tensor& bias, int64_t C, int64_t N,
    double eps, int64_t tgs
) {
    UNBLEND_CHECKS(group_norm_g1_gelu)
    UNBLEND_DISPATCH(group_norm_g1_gelu_impl, in_, out, in_, inject, weight, bias, C, N, eps, tgs)
}

void apply_norm_gelu(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& inject,
    const at::Tensor& meanvar, const at::Tensor& weight, const at::Tensor& bias,
    int64_t total_per_b, int64_t num_tiles, int64_t N, int64_t tgs
) {
    UNBLEND_CHECKS(apply_norm_gelu)
    UNBLEND_DISPATCH(apply_norm_gelu_impl, in_, out, in_, inject, meanvar, weight, bias, total_per_b, num_tiles, N, tgs)
}
void add_gelu(
    const at::Tensor& out, const at::Tensor& a, const at::Tensor& b
) {
    TORCH_CHECK(a.is_cuda() && out.is_cuda() && b.is_cuda(),
                "add_gelu: tensors must be CUDA");
    TORCH_CHECK(
        a.scalar_type() == out.scalar_type() && b.scalar_type() == a.scalar_type(),
        "add_gelu: dtype mismatch");
    UNBLEND_DISPATCH(add_gelu_impl, a, out, a, b)
}
