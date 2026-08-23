// GroupNorm with num_groups=1: single-stage kernels and the reduction
// primitives shared with every other ``apply_*`` kernel in this folder.
// CUDA port of ``unblend/metal/group_norm.metal`` — kernel names, argument
// order, and semantics match the Metal originals one-to-one.
//
// ``group_norm_g1`` runs one block per batch element — best for shapes with
// many batch elements (DConv internals). ``group_norm_g1_chlast`` is its
// channel-LAST twin for the transformer's ``MyGroupNorm`` (input
// ``(B, T, C)`` flattened to ``(B, T*C)``; affine index is ``i % C``).
//
// ``partial_reduce`` + ``finalize_meanvar`` are the first two stages of the
// multi-stage path used when a single-stage launch would leave the GPU idle
// (small batch, large per-batch work). Apply kernels in the other .cu files
// read the (B, 2) ``meanvar`` buffer the finalize stage writes.
// ``apply_norm`` / ``apply_norm_chlast`` are the plain (no activation)
// third stages.
//
// Loads/stores use Scalar4 vectors when alignment permits (see kernels.cuh);
// the apply loops additionally need the affine index to be constant within
// each vector, i.e. N % 4 == 0 for channel-first and C % 4 == 0 for
// channel-last, and fall back to scalar loops otherwise.

#include "bindings.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "kernels.cuh"

namespace {

template <typename SCALAR_T>
__global__ void group_norm_g1_kernel(
    SCALAR_T* __restrict__  out,
    const SCALAR_T* __restrict__  in_,
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

    const unsigned int total = C * N;
    // Batch base offsets in 64-bit: b * total overflows 32 bits past ~4G
    // elements into the buffer.
    const SCALAR_T* __restrict__  in_b = in_ + (unsigned long long)b * total;
    SCALAR_T* __restrict__  out_b = out + (unsigned long long)b * total;

    float K = static_cast<float>(in_b[0]);
    float s = 0.0f, sq = 0.0f;
    gn_accumulate_sumsq(in_b, total, K, tid, tgs, s, sq);
    gn_reduce_finalize(s, sq, K, total, eps, sh_sum, sh_sq, bcast);
    const float mean = bcast[0];
    const float scale = bcast[1];

    if ((N & 3u) == 0u) {
        const Scalar4<SCALAR_T>* in4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(in_b);
        Scalar4<SCALAR_T>* out4 = reinterpret_cast<Scalar4<SCALAR_T>*>(out_b);
        const unsigned int Nv = N >> 2;
        const unsigned int nv = C * Nv;
        for (unsigned int i = tid; i < nv; i += tgs) {
            const unsigned int c = i / Nv;
            const float w = static_cast<float>(weight[c]);
            const float bv = static_cast<float>(bias[c]);
            const float4 v = unpack4(in4[i]);
            const float4 r = make_float4(
                (v.x - mean) * scale * w + bv,
                (v.y - mean) * scale * w + bv,
                (v.z - mean) * scale * w + bv,
                (v.w - mean) * scale * w + bv
            );
            out4[i] = pack4<SCALAR_T>(r);
        }
    } else {
        for (unsigned int i = tid; i < total; i += tgs) {
            const unsigned int c = i / N;
            const float w = static_cast<float>(weight[c]);
            const float bv = static_cast<float>(bias[c]);
            const float v = static_cast<float>(in_b[i]);
            out_b[i] = static_cast<SCALAR_T>((v - mean) * scale * w + bv);
        }
    }
}

template <typename SCALAR_T>
__global__ void group_norm_g1_chlast_kernel(
    SCALAR_T* __restrict__  out,
    const SCALAR_T* __restrict__  in_,
    const SCALAR_T* __restrict__  weight,
    const SCALAR_T* __restrict__  bias,
    unsigned int C,
    unsigned int total,  // T * C per batch
    float eps
) {
    __shared__ float sh_sum[MAX_WARPS];
    __shared__ float sh_sq[MAX_WARPS];
    __shared__ float bcast[2];

    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int b = blockIdx.x;

    const SCALAR_T* __restrict__  in_b = in_ + (unsigned long long)b * total;
    SCALAR_T* __restrict__  out_b = out + (unsigned long long)b * total;

    float K = static_cast<float>(in_b[0]);
    float s = 0.0f, sq = 0.0f;
    gn_accumulate_sumsq(in_b, total, K, tid, tgs, s, sq);
    gn_reduce_finalize(s, sq, K, total, eps, sh_sum, sh_sq, bcast);
    const float mean = bcast[0];
    const float scale = bcast[1];

    if ((C & 3u) == 0u) {
        // C % 4 == 0 implies total % 4 == 0 (total = T*C), so vector loads
        // stay aligned and each Scalar4 spans channels 4k..4k+3.
        const Scalar4<SCALAR_T>* in4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(in_b);
        Scalar4<SCALAR_T>* out4 = reinterpret_cast<Scalar4<SCALAR_T>*>(out_b);
        const Scalar4<SCALAR_T>* w4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(weight);
        const Scalar4<SCALAR_T>* b4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(bias);
        const unsigned int Cv = C >> 2;
        const unsigned int nv = total >> 2;
        for (unsigned int i = tid; i < nv; i += tgs) {
            const unsigned int cv = i % Cv;
            const float4 w = unpack4(w4[cv]);
            const float4 bv = unpack4(b4[cv]);
            const float4 v = unpack4(in4[i]);
            const float4 r = make_float4(
                (v.x - mean) * scale * w.x + bv.x,
                (v.y - mean) * scale * w.y + bv.y,
                (v.z - mean) * scale * w.z + bv.z,
                (v.w - mean) * scale * w.w + bv.w
            );
            out4[i] = pack4<SCALAR_T>(r);
        }
    } else {
        for (unsigned int i = tid; i < total; i += tgs) {
            const unsigned int c = i % C;
            const float w = static_cast<float>(weight[c]);
            const float bv = static_cast<float>(bias[c]);
            const float v = static_cast<float>(in_b[i]);
            out_b[i] = static_cast<SCALAR_T>((v - mean) * scale * w + bv);
        }
    }
}

template <typename SCALAR_T>
__global__ void partial_reduce_kernel(
    const SCALAR_T* __restrict__  in_,
    const SCALAR_T* __restrict__ inject,  // optional second input added first
    float* __restrict__  scratch,  // (B, num_tiles, 2)
    unsigned int total_per_b,
    unsigned int num_tiles
) {
    const bool has_inj = inject != nullptr;
    __shared__ float sh_sum[MAX_WARPS];
    __shared__ float sh_sq[MAX_WARPS];
    __shared__ float bcast[2];

    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int bt = blockIdx.x;  // b * num_tiles + t
    const unsigned int b = bt / num_tiles;
    const unsigned int t = bt % num_tiles;

    const SCALAR_T* __restrict__  x_b = in_ + (unsigned long long)b * total_per_b;
    const SCALAR_T* __restrict__  j_b =
        has_inj ? inject + (unsigned long long)b * total_per_b : nullptr;

    // Shift by the batch's first element (shared across all tiles of this
    // batch) so the partial sums feeding the variance don't lose precision
    // to cancellation on large-DC inputs. finalize_meanvar adds K back for
    // the mean; the variance it derives from these shifted sums is
    // unaffected.
    float K = static_cast<float>(x_b[0]);
    if (has_inj) {
        K += static_cast<float>(j_b[0]);
    }
    float local_sum = 0.0f;
    float local_sqsum = 0.0f;
    if ((total_per_b & 3u) == 0u) {
        // Tile the vector space. The partial sums are position-agnostic, so
        // tiling (total/4) vectors instead of total scalars changes nothing
        // downstream — finalize just sums every tile's partials.
        const Scalar4<SCALAR_T>* __restrict__ x4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(x_b);
        const Scalar4<SCALAR_T>* __restrict__ j4 =
            has_inj ? reinterpret_cast<const Scalar4<SCALAR_T>*>(j_b) : nullptr;
        const unsigned int nv = total_per_b >> 2;
        const unsigned int start =
            (unsigned int)((unsigned long long)t * nv / num_tiles);
        const unsigned int end =
            (unsigned int)((unsigned long long)(t + 1) * nv / num_tiles);
        for (unsigned int i = start + tid; i < end; i += tgs) {
            float4 v = unpack4(x4[i]);
            if (has_inj) {
                const float4 wj = unpack4(j4[i]);
                v.x += wj.x; v.y += wj.y; v.z += wj.z; v.w += wj.w;
            }
            v.x -= K; v.y -= K; v.z -= K; v.w -= K;
            local_sum += v.x + v.y + v.z + v.w;
            local_sqsum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        // Even tile boundaries — the last tile picks up any remainder.
        const unsigned int start =
            (unsigned int)((unsigned long long)t * total_per_b / num_tiles);
        const unsigned int end =
            (unsigned int)((unsigned long long)(t + 1) * total_per_b / num_tiles);
        for (unsigned int i = start + tid; i < end; i += tgs) {
            float v = static_cast<float>(x_b[i]) - K;
            if (has_inj) {
                v += static_cast<float>(j_b[i]);
            }
            local_sum += v;
            local_sqsum += v * v;
        }
    }
    block_reduce_sumsq(local_sum, local_sqsum, sh_sum, sh_sq, bcast);
    if (tid == 0) {
        scratch[((unsigned long long)b * num_tiles + t) * 2 + 0] = bcast[0];
        scratch[((unsigned long long)b * num_tiles + t) * 2 + 1] = bcast[1];
    }
}

template <typename SCALAR_T>
__global__ void finalize_meanvar_kernel(
    const float* __restrict__  scratch,  // (B, num_tiles, 2) — shifted (sum_d, sqsum_d)
    float* __restrict__  meanvar,        // (B, 2) — (mean, rsqrt(var+eps))
    unsigned int total_per_b,
    unsigned int num_tiles,
    float eps,
    const SCALAR_T* __restrict__  in_,     // input, for the shift reference K
    const SCALAR_T* __restrict__ inject    // optional second input, same role
) {
    __shared__ float sh_sum[MAX_WARPS];
    __shared__ float sh_sq[MAX_WARPS];
    __shared__ float bcast[2];

    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int b = blockIdx.x;

    float local_sum = 0.0f;
    float local_sqsum = 0.0f;
    for (unsigned int t = tid; t < num_tiles; t += tgs) {
        local_sum += scratch[((unsigned long long)b * num_tiles + t) * 2 + 0];
        local_sqsum += scratch[((unsigned long long)b * num_tiles + t) * 2 + 1];
    }
    // sh_sum / sh_sq are sums of (x - K); recover the true mean by adding K
    // back. The variance is computed from the shifted sums, where
    // cancellation is negligible. K is the same reference partial_reduce
    // used: the batch's first element.
    const unsigned long long kbase = (unsigned long long)b * total_per_b;
    float K = static_cast<float>(in_[kbase]);
    if (inject != nullptr) {
        K += static_cast<float>(inject[kbase]);
    }
    gn_reduce_finalize(
        local_sum, local_sqsum, K, total_per_b, eps, sh_sum, sh_sq, bcast
    );
    if (tid == 0) {
        meanvar[(unsigned long long)b * 2 + 0] = bcast[0];
        meanvar[(unsigned long long)b * 2 + 1] = bcast[1];
    }
}

template <typename SCALAR_T>
__global__ void apply_norm_kernel(
    SCALAR_T* __restrict__  out,
    const SCALAR_T* __restrict__  in_,
    const float* __restrict__  meanvar,  // (B, 2)
    const SCALAR_T* __restrict__  weight,
    const SCALAR_T* __restrict__  bias,
    unsigned int total_per_b,
    unsigned int num_tiles,
    unsigned int N  // spatial size
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int bt = blockIdx.x;
    const unsigned int b = bt / num_tiles;
    const unsigned int t = bt % num_tiles;

    const float mean = meanvar[(unsigned long long)b * 2 + 0];
    const float scale = meanvar[(unsigned long long)b * 2 + 1];

    const SCALAR_T* __restrict__  x_b = in_ + (unsigned long long)b * total_per_b;
    SCALAR_T* __restrict__  out_b = out + (unsigned long long)b * total_per_b;

    if ((N & 3u) == 0u) {
        const Scalar4<SCALAR_T>* in4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(x_b);
        Scalar4<SCALAR_T>* out4 = reinterpret_cast<Scalar4<SCALAR_T>*>(out_b);
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
            const float4 v = unpack4(in4[i]);
            const float4 r = make_float4(
                (v.x - mean) * scale * w + bv,
                (v.y - mean) * scale * w + bv,
                (v.z - mean) * scale * w + bv,
                (v.w - mean) * scale * w + bv
            );
            out4[i] = pack4<SCALAR_T>(r);
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
            const float v = static_cast<float>(x_b[i]);
            out_b[i] = static_cast<SCALAR_T>((v - mean) * scale * w + bv);
        }
    }
}

// Channel-last multi-stage third stage (transformer MyGroupNorm shapes).
template <typename SCALAR_T>
__global__ void apply_norm_chlast_kernel(
    SCALAR_T* __restrict__  out,
    const SCALAR_T* __restrict__  in_,
    const float* __restrict__  meanvar,  // (B, 2)
    const SCALAR_T* __restrict__  weight,
    const SCALAR_T* __restrict__  bias,
    unsigned int total_per_b,  // T * C
    unsigned int num_tiles,
    unsigned int C
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int bt = blockIdx.x;
    const unsigned int b = bt / num_tiles;
    const unsigned int t = bt % num_tiles;

    const float mean = meanvar[(unsigned long long)b * 2 + 0];
    const float scale = meanvar[(unsigned long long)b * 2 + 1];

    const SCALAR_T* __restrict__  x_b = in_ + (unsigned long long)b * total_per_b;
    SCALAR_T* __restrict__  out_b = out + (unsigned long long)b * total_per_b;

    if ((C & 3u) == 0u) {
        const Scalar4<SCALAR_T>* in4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(x_b);
        Scalar4<SCALAR_T>* out4 = reinterpret_cast<Scalar4<SCALAR_T>*>(out_b);
        const Scalar4<SCALAR_T>* w4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(weight);
        const Scalar4<SCALAR_T>* b4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(bias);
        const unsigned int Cv = C >> 2;
        const unsigned int nv = total_per_b >> 2;
        const unsigned int start =
            (unsigned int)((unsigned long long)t * nv / num_tiles);
        const unsigned int end =
            (unsigned int)((unsigned long long)(t + 1) * nv / num_tiles);
        for (unsigned int i = start + tid; i < end; i += tgs) {
            const unsigned int cv = i % Cv;
            const float4 w = unpack4(w4[cv]);
            const float4 bv = unpack4(b4[cv]);
            const float4 v = unpack4(in4[i]);
            const float4 r = make_float4(
                (v.x - mean) * scale * w.x + bv.x,
                (v.y - mean) * scale * w.y + bv.y,
                (v.z - mean) * scale * w.z + bv.z,
                (v.w - mean) * scale * w.w + bv.w
            );
            out4[i] = pack4<SCALAR_T>(r);
        }
    } else {
        const unsigned int start =
            (unsigned int)((unsigned long long)t * total_per_b / num_tiles);
        const unsigned int end =
            (unsigned int)((unsigned long long)(t + 1) * total_per_b / num_tiles);
        for (unsigned int i = start + tid; i < end; i += tgs) {
            const unsigned int c = i % C;
            const float w = static_cast<float>(weight[c]);
            const float bv = static_cast<float>(bias[c]);
            const float v = static_cast<float>(x_b[i]);
            out_b[i] = static_cast<SCALAR_T>((v - mean) * scale * w + bv);
        }
    }
}

// ---------------------------------------------------------------------------
// Launchers: dtype dispatch + launch configuration on the current stream.
// ---------------------------------------------------------------------------

template <typename SCALAR_T>
void group_norm_g1_impl(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& weight,
    const at::Tensor& bias, int64_t C, int64_t N, double eps, int64_t tgs
) {
    const dim3 grid((unsigned int)in_.size(0));
    const dim3 block((unsigned int)tgs);
    group_norm_g1_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<SCALAR_T>(), in_.const_data_ptr<SCALAR_T>(),
        weight.const_data_ptr<SCALAR_T>(), bias.const_data_ptr<SCALAR_T>(),
        (unsigned int)C, (unsigned int)N, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SCALAR_T>
void group_norm_g1_chlast_impl(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& weight,
    const at::Tensor& bias, int64_t C, int64_t total, double eps, int64_t tgs
) {
    const dim3 grid((unsigned int)in_.size(0));
    const dim3 block((unsigned int)tgs);
    group_norm_g1_chlast_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<SCALAR_T>(), in_.const_data_ptr<SCALAR_T>(),
        weight.const_data_ptr<SCALAR_T>(), bias.const_data_ptr<SCALAR_T>(),
        (unsigned int)C, (unsigned int)total, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SCALAR_T>
void partial_reduce_impl(
    const at::Tensor& in_, const at::Tensor& inject, const at::Tensor& scratch,
    int64_t total_per_b, int64_t num_tiles, int64_t tgs
) {
    const dim3 grid((unsigned int)(in_.size(0) * num_tiles));
    const dim3 block((unsigned int)tgs);
    partial_reduce_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        in_.const_data_ptr<SCALAR_T>(),
        inject.numel() > 0 ? inject.const_data_ptr<SCALAR_T>() : nullptr,
        scratch.data_ptr<float>(),
        (unsigned int)total_per_b, (unsigned int)num_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SCALAR_T>
void finalize_meanvar_impl(
    const at::Tensor& scratch, const at::Tensor& meanvar, int64_t total_per_b,
    int64_t num_tiles, double eps, const at::Tensor& in_,
    const at::Tensor& inject, int64_t tgs
) {
    const dim3 grid((unsigned int)meanvar.size(0));
    const dim3 block((unsigned int)tgs);
    finalize_meanvar_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        scratch.const_data_ptr<float>(), meanvar.data_ptr<float>(),
        (unsigned int)total_per_b, (unsigned int)num_tiles, (float)eps,
        in_.const_data_ptr<SCALAR_T>(),
        inject.numel() > 0 ? inject.const_data_ptr<SCALAR_T>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SCALAR_T>
void apply_norm_impl(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& meanvar,
    const at::Tensor& weight, const at::Tensor& bias, int64_t total_per_b,
    int64_t num_tiles, int64_t N, int64_t tgs
) {
    const dim3 grid((unsigned int)(in_.size(0) * num_tiles));
    const dim3 block((unsigned int)tgs);
    apply_norm_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<SCALAR_T>(), in_.const_data_ptr<SCALAR_T>(),
        meanvar.const_data_ptr<float>(), weight.const_data_ptr<SCALAR_T>(),
        bias.const_data_ptr<SCALAR_T>(), (unsigned int)total_per_b,
        (unsigned int)num_tiles, (unsigned int)N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SCALAR_T>
void apply_norm_chlast_impl(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& meanvar,
    const at::Tensor& weight, const at::Tensor& bias, int64_t total_per_b,
    int64_t num_tiles, int64_t C, int64_t tgs
) {
    const dim3 grid((unsigned int)(in_.size(0) * num_tiles));
    const dim3 block((unsigned int)tgs);
    apply_norm_chlast_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<SCALAR_T>(), in_.const_data_ptr<SCALAR_T>(),
        meanvar.const_data_ptr<float>(), weight.const_data_ptr<SCALAR_T>(),
        bias.const_data_ptr<SCALAR_T>(), (unsigned int)total_per_b,
        (unsigned int)num_tiles, (unsigned int)C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void group_norm_g1(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& weight,
    const at::Tensor& bias, int64_t C, int64_t N, double eps, int64_t tgs
) {
    UNBLEND_CHECKS(group_norm_g1)
    UNBLEND_DISPATCH(group_norm_g1_impl, in_, out, in_, weight, bias, C, N, eps, tgs)
}

void group_norm_g1_chlast(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& weight,
    const at::Tensor& bias, int64_t C, int64_t total, double eps, int64_t tgs
) {
    UNBLEND_CHECKS(group_norm_g1_chlast)
    UNBLEND_DISPATCH(group_norm_g1_chlast_impl, in_, out, in_, weight, bias, C, total, eps, tgs)
}

void partial_reduce(
    const at::Tensor& in_, const at::Tensor& inject, const at::Tensor& scratch,
    int64_t total_per_b, int64_t num_tiles, int64_t tgs
) {
    TORCH_CHECK(in_.is_cuda() && scratch.is_cuda(), "partial_reduce: tensors must be CUDA");
    TORCH_CHECK(scratch.scalar_type() == at::kFloat, "partial_reduce: FP32 scratch required");
    UNBLEND_DISPATCH(partial_reduce_impl, in_, in_, inject, scratch, total_per_b, num_tiles, tgs)
}

void finalize_meanvar(
    const at::Tensor& scratch, const at::Tensor& meanvar, int64_t total_per_b,
    int64_t num_tiles, double eps, const at::Tensor& in_,
    const at::Tensor& inject, int64_t tgs
) {
    TORCH_CHECK(in_.is_cuda() && scratch.is_cuda() && meanvar.is_cuda(),
                "finalize_meanvar: tensors must be CUDA");
    TORCH_CHECK(scratch.scalar_type() == at::kFloat && meanvar.scalar_type() == at::kFloat,
                "finalize_meanvar: FP32 buffers required");
    UNBLEND_DISPATCH(finalize_meanvar_impl, in_, scratch, meanvar, total_per_b, num_tiles, eps, in_, inject, tgs)
}

void apply_norm(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& meanvar,
    const at::Tensor& weight, const at::Tensor& bias, int64_t total_per_b,
    int64_t num_tiles, int64_t N, int64_t tgs
) {
    UNBLEND_CHECKS(apply_norm)
    UNBLEND_DISPATCH(apply_norm_impl, in_, out, in_, meanvar, weight, bias, total_per_b, num_tiles, N, tgs)
}

void apply_norm_chlast(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& meanvar,
    const at::Tensor& weight, const at::Tensor& bias, int64_t total_per_b,
    int64_t num_tiles, int64_t C, int64_t tgs
) {
    UNBLEND_CHECKS(apply_norm_chlast)
    UNBLEND_DISPATCH(apply_norm_chlast_impl, in_, out, in_, meanvar, weight, bias, total_per_b, num_tiles, C, tgs)
}
