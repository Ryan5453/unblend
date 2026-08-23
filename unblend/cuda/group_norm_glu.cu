// GroupNorm fused with GLU (channel halving).
// CUDA port of ``unblend/metal/group_norm_glu.metal``.
//
// Input shape (B, 2C, N), output (B, C, N). The reduction is over all 2C
// input channels (so the per-batch mean matches ``F.group_norm`` exactly);
// for each output channel we read ``a = norm(in[c])`` and
// ``b = norm(in[c + C])`` and combine via ``a * sigmoid(b)`` without ever
// writing the post-norm full-size tensor to memory.
//
// ``apply_norm_glu`` is the third stage of the multi-stage path, reading
// ``meanvar`` produced by ``finalize_meanvar`` in ``group_norm.cu``.
// Crucially it tiles the OUTPUT space (size C*N), not the input — each
// output element pulls its two input channels by absolute offset so the
// tile boundaries don't matter. Vector/scalar path selection and the
// reduction helpers are shared via ``kernels.cuh``.

#include "bindings.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "kernels.cuh"

namespace {

__device__ __forceinline__ float4 sigmoid4(const float4& g) {
    return make_float4(
        1.0f / (1.0f + expf(-g.x)),
        1.0f / (1.0f + expf(-g.y)),
        1.0f / (1.0f + expf(-g.z)),
        1.0f / (1.0f + expf(-g.w))
    );
}

template <typename SCALAR_T>
__global__ void group_norm_g1_glu_kernel(
    SCALAR_T* __restrict__  out,       // (B, C/2, N)
    const SCALAR_T* __restrict__  in_, // (B, C,   N)
    const SCALAR_T* __restrict__  weight,
    const SCALAR_T* __restrict__  bias,
    unsigned int C,
    unsigned int N,
    float eps
) {
    __shared__ float sh_sum[MAX_WARPS];
    __shared__ float sh_sq[MAX_WARPS];
    __shared__ float bcast[2];

    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int b = blockIdx.x;

    const unsigned int C_half = C >> 1;
    const unsigned int total_in = C * N;
    const unsigned int total_out = C_half * N;
    // 64-bit base offsets: b * total overflows 32 bits on huge inputs.
    const SCALAR_T* __restrict__  in_b = in_ + (unsigned long long)b * total_in;
    SCALAR_T* __restrict__  out_b = out + (unsigned long long)b * total_out;

    float K = static_cast<float>(in_b[0]);
    float s = 0.0f, sq = 0.0f;
    gn_accumulate_sumsq(in_b, total_in, K, tid, tgs, s, sq);
    gn_reduce_finalize(s, sq, K, total_in, eps, sh_sum, sh_sq, bcast);
    const float mean = bcast[0];
    const float scale = bcast[1];

    if ((N & 3u) == 0u) {
        const Scalar4<SCALAR_T>* in4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(in_b);
        Scalar4<SCALAR_T>* out4 = reinterpret_cast<Scalar4<SCALAR_T>*>(out_b);
        const unsigned int Nv = N >> 2;
        const unsigned int nv = C_half * Nv;
        const unsigned int boff = C_half * Nv;  // vector offset of channel c + C_half
        for (unsigned int i = tid; i < nv; i += tgs) {
            const unsigned int c = i / Nv;
            const float wa = static_cast<float>(weight[c]);
            const float ba = static_cast<float>(bias[c]);
            const float wb = static_cast<float>(weight[c + C_half]);
            const float bb = static_cast<float>(bias[c + C_half]);
            const float4 va = unpack4(in4[i]);
            const float4 vg = unpack4(in4[i + boff]);
            const float4 a = make_float4(
                (va.x - mean) * scale * wa + ba,
                (va.y - mean) * scale * wa + ba,
                (va.z - mean) * scale * wa + ba,
                (va.w - mean) * scale * wa + ba
            );
            const float4 g = make_float4(
                (vg.x - mean) * scale * wb + bb,
                (vg.y - mean) * scale * wb + bb,
                (vg.z - mean) * scale * wb + bb,
                (vg.w - mean) * scale * wb + bb
            );
            const float4 sig = sigmoid4(g);
            out4[i] = pack4<SCALAR_T>(
                make_float4(a.x * sig.x, a.y * sig.y, a.z * sig.z, a.w * sig.w)
            );
        }
    } else {
        for (unsigned int i = tid; i < total_out; i += tgs) {
            const unsigned int c_out = i / N;
            const unsigned int sp = i % N;
            const unsigned int idx_a = c_out * N + sp;
            const unsigned int idx_b = (c_out + C_half) * N + sp;
            const float wa = static_cast<float>(weight[c_out]);
            const float ba = static_cast<float>(bias[c_out]);
            const float wb = static_cast<float>(weight[c_out + C_half]);
            const float bb = static_cast<float>(bias[c_out + C_half]);
            const float a =
                (static_cast<float>(in_b[idx_a]) - mean) * scale * wa + ba;
            const float b_val =
                (static_cast<float>(in_b[idx_b]) - mean) * scale * wb + bb;
            const float sig = 1.0f / (1.0f + expf(-b_val));
            out_b[i] = static_cast<SCALAR_T>(a * sig);
        }
    }
}

template <typename SCALAR_T>
__global__ void apply_norm_glu_kernel(
    SCALAR_T* __restrict__  out,       // (B, C/2 * N)
    const SCALAR_T* __restrict__  in_, // (B, C * N)
    const float* __restrict__  meanvar,
    const SCALAR_T* __restrict__  weight,
    const SCALAR_T* __restrict__  bias,
    unsigned int total_in_per_b,  // C * N
    unsigned int total_out_per_b, // C_half * N
    unsigned int num_tiles,
    unsigned int N,
    unsigned int C_half
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int bt = blockIdx.x;
    const unsigned int b = bt / num_tiles;
    const unsigned int t = bt % num_tiles;

    const float mean = meanvar[(unsigned long long)b * 2 + 0];
    const float scale = meanvar[(unsigned long long)b * 2 + 1];

    const SCALAR_T* __restrict__  x_b = in_ + (unsigned long long)b * total_in_per_b;
    SCALAR_T* __restrict__  out_b = out + (unsigned long long)b * total_out_per_b;

    if ((N & 3u) == 0u) {
        const Scalar4<SCALAR_T>* in4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(x_b);
        Scalar4<SCALAR_T>* out4 = reinterpret_cast<Scalar4<SCALAR_T>*>(out_b);
        const unsigned int Nv = N >> 2;
        const unsigned int nv = total_out_per_b >> 2;
        const unsigned int boff = C_half * Nv;
        const unsigned int start =
            (unsigned int)((unsigned long long)t * nv / num_tiles);
        const unsigned int end =
            (unsigned int)((unsigned long long)(t + 1) * nv / num_tiles);
        for (unsigned int i = start + tid; i < end; i += tgs) {
            const unsigned int c = i / Nv;
            const float wa = static_cast<float>(weight[c]);
            const float ba = static_cast<float>(bias[c]);
            const float wb = static_cast<float>(weight[c + C_half]);
            const float bb = static_cast<float>(bias[c + C_half]);
            const float4 va = unpack4(in4[i]);
            const float4 vg = unpack4(in4[i + boff]);
            const float4 a = make_float4(
                (va.x - mean) * scale * wa + ba,
                (va.y - mean) * scale * wa + ba,
                (va.z - mean) * scale * wa + ba,
                (va.w - mean) * scale * wa + ba
            );
            const float4 g = make_float4(
                (vg.x - mean) * scale * wb + bb,
                (vg.y - mean) * scale * wb + bb,
                (vg.z - mean) * scale * wb + bb,
                (vg.w - mean) * scale * wb + bb
            );
            const float4 sig = sigmoid4(g);
            out4[i] = pack4<SCALAR_T>(
                make_float4(a.x * sig.x, a.y * sig.y, a.z * sig.z, a.w * sig.w)
            );
        }
    } else {
        // Tile over the OUTPUT space; each output element pulls its two
        // input channels independently regardless of where they sit in the
        // input tile.
        const unsigned int start =
            (unsigned int)((unsigned long long)t * total_out_per_b / num_tiles);
        const unsigned int end =
            (unsigned int)((unsigned long long)(t + 1) * total_out_per_b / num_tiles);
        for (unsigned int i = start + tid; i < end; i += tgs) {
            const unsigned int c_out = i / N;
            const unsigned int sp = i % N;
            const unsigned int idx_a = c_out * N + sp;
            const unsigned int idx_b = (c_out + C_half) * N + sp;
            const float wa = static_cast<float>(weight[c_out]);
            const float ba = static_cast<float>(bias[c_out]);
            const float wb = static_cast<float>(weight[c_out + C_half]);
            const float bb = static_cast<float>(bias[c_out + C_half]);
            const float a =
                (static_cast<float>(x_b[idx_a]) - mean) * scale * wa + ba;
            const float b_val =
                (static_cast<float>(x_b[idx_b]) - mean) * scale * wb + bb;
            const float sig = 1.0f / (1.0f + expf(-b_val));
            out_b[i] = static_cast<SCALAR_T>(a * sig);
        }
    }
}

template <typename SCALAR_T>
void group_norm_g1_glu_impl(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& weight,
    const at::Tensor& bias, int64_t C, int64_t N, double eps, int64_t tgs
) {
    const dim3 grid((unsigned int)in_.size(0));
    const dim3 block((unsigned int)tgs);
    group_norm_g1_glu_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<SCALAR_T>(), in_.const_data_ptr<SCALAR_T>(),
        weight.const_data_ptr<SCALAR_T>(), bias.const_data_ptr<SCALAR_T>(),
        (unsigned int)C, (unsigned int)N, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SCALAR_T>
void apply_norm_glu_impl(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& meanvar,
    const at::Tensor& weight, const at::Tensor& bias, int64_t total_in_per_b,
    int64_t total_out_per_b, int64_t num_tiles, int64_t N, int64_t C_half,
    int64_t tgs
) {
    const dim3 grid((unsigned int)(in_.size(0) * num_tiles));
    const dim3 block((unsigned int)tgs);
    apply_norm_glu_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<SCALAR_T>(), in_.const_data_ptr<SCALAR_T>(),
        meanvar.const_data_ptr<float>(), weight.const_data_ptr<SCALAR_T>(),
        bias.const_data_ptr<SCALAR_T>(), (unsigned int)total_in_per_b,
        (unsigned int)total_out_per_b, (unsigned int)num_tiles,
        (unsigned int)N, (unsigned int)C_half);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void group_norm_g1_glu(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& weight,
    const at::Tensor& bias, int64_t C, int64_t N, double eps, int64_t tgs
) {
    UNBLEND_CHECKS(group_norm_g1_glu)
    UNBLEND_DISPATCH(group_norm_g1_glu_impl, in_, out, in_, weight, bias, C, N, eps, tgs)
}

void apply_norm_glu(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& meanvar,
    const at::Tensor& weight, const at::Tensor& bias, int64_t total_in_per_b,
    int64_t total_out_per_b, int64_t num_tiles, int64_t N, int64_t C_half,
    int64_t tgs
) {
    UNBLEND_CHECKS(apply_norm_glu)
    UNBLEND_DISPATCH(apply_norm_glu_impl, in_, out, in_, meanvar, weight, bias, total_in_per_b, total_out_per_b, num_tiles, N, C_half, tgs)
}
