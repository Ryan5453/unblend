// GroupNorm with num_groups=1: single-stage kernels and the reduction
// primitives shared with every other ``apply_*`` kernel in this folder.
// CUDA port of ``unblend/metal/group_norm.metal``: kernel names and semantics
// match the Metal originals, except that ``partial_reduce`` also takes an
// optional ``inject`` input.
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
// channel-last, and use scalar loops otherwise.

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
    __shared__ float sh[GN_SHARED_FLOATS];

    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int b = blockIdx.x;

    const unsigned int total = C * N;
    // Batch base offsets in 64-bit: b * total overflows 32 bits past ~4G
    // elements into the buffer.
    const SCALAR_T* __restrict__  in_b = in_ + (unsigned long long)b * total;
    SCALAR_T* __restrict__  out_b = out + (unsigned long long)b * total;

    const float2 ms = gn_reduce_finalize(
        gn_thread_partial(in_b, total, 0u, total, tid, tgs), total, eps, sh
    );
    const float mean = ms.x;
    const float scale = ms.y;

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
    __shared__ float sh[GN_SHARED_FLOATS];

    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int b = blockIdx.x;

    const SCALAR_T* __restrict__  in_b = in_ + (unsigned long long)b * total;
    SCALAR_T* __restrict__  out_b = out + (unsigned long long)b * total;

    const float2 ms = gn_reduce_finalize(
        gn_thread_partial(in_b, total, 0u, total, tid, tgs), total, eps, sh
    );
    const float mean = ms.x;
    const float scale = ms.y;

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
    __shared__ float sh[GN_SHARED_FLOATS];

    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int bt = blockIdx.x;  // b * num_tiles + t
    const unsigned int b = bt / num_tiles;
    const unsigned int t = bt % num_tiles;

    const SCALAR_T* __restrict__  x_b = in_ + (unsigned long long)b * total_per_b;
    const SCALAR_T* __restrict__  j_b =
        has_inj ? inject + (unsigned long long)b * total_per_b : nullptr;

    // Each tile writes its own (mean, M2), not K-shifted sums:
    // finalize_meanvar merges the tiles with the same identities
    // block_merge_moments uses across threads, so the multi-stage statistics
    // match the single-stage ones.
    const uint2 r = gn_tile_bounds(t, num_tiles, total_per_b);
    block_merge_moments(
        gn_thread_partial(x_b, j_b, total_per_b, r.x, r.y, tid, tgs), r.y - r.x,
        false, 0.0f, sh
    );
    if (tid == 0) {
        scratch[((unsigned long long)b * num_tiles + t) * 2 + 0] = sh[GN_BCAST];
        scratch[((unsigned long long)b * num_tiles + t) * 2 + 1] = sh[GN_BCAST + 1];
    }
}

// Reads only the per-tile moments, so unlike the other stages it doesn't
// depend on the input dtype.
__global__ void finalize_meanvar_kernel(
    const float* __restrict__  scratch,  // (B, num_tiles, 2) — per-tile (mean, M2)
    float* __restrict__  meanvar,        // (B, 2) — (mean, rsqrt(var+eps))
    unsigned int total_per_b,
    unsigned int num_tiles,
    float eps
) {
    __shared__ float sh[GN_SHARED_FLOATS];

    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int b = blockIdx.x;

    // Fold this thread's tiles into one partial shifted by its first tile's
    // mean: a tile of n elements with (mean, M2) contributes
    // s += n (mean - K) and sq += M2 + n (mean - K)^2.
    GnPartial p = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int t = tid; t < num_tiles; t += tgs) {
        const uint2 r = gn_tile_bounds(t, num_tiles, total_per_b);
        const float n = static_cast<float>(r.y - r.x);
        const float mean = scratch[((unsigned long long)b * num_tiles + t) * 2 + 0];
        const float m2 = scratch[((unsigned long long)b * num_tiles + t) * 2 + 1];
        if (t == tid) {
            p.K = mean;
        }
        const float d = mean - p.K;
        p.n += n;
        p.s += n * d;
        p.sq += m2 + n * d * d;
    }
    const float2 ms = gn_reduce_finalize(p, total_per_b, eps, sh);
    if (tid == 0) {
        meanvar[(unsigned long long)b * 2 + 0] = ms.x;
        meanvar[(unsigned long long)b * 2 + 1] = ms.y;
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
    int64_t num_tiles, double eps, int64_t tgs
) {
    TORCH_CHECK(scratch.is_cuda() && meanvar.is_cuda(),
                "finalize_meanvar: tensors must be CUDA");
    TORCH_CHECK(scratch.scalar_type() == at::kFloat && meanvar.scalar_type() == at::kFloat,
                "finalize_meanvar: FP32 buffers required");
    const dim3 grid((unsigned int)meanvar.size(0));
    const dim3 block((unsigned int)tgs);
    finalize_meanvar_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        scratch.const_data_ptr<float>(), meanvar.data_ptr<float>(),
        (unsigned int)total_per_b, (unsigned int)num_tiles, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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
