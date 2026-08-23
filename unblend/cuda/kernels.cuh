// Shared prelude included by every kernel translation unit in this folder
// and by bindings.cpp. CUDA port of ``unblend/metal/common.metal``.
//
// Provides the SCALAR_T template vocabulary (Scalar4 packed loads), the
// FP32 conversion helpers, and the two-level warp/block reduction helpers
// that mirror the Metal simdgroup reductions:
//
//   gn_accumulate_sumsq  — K-shifted (sum, sum-of-squares) accumulation,
//                          vectorized 4-wide when element count % 4 == 0
//   block_reduce_sumsq   — warp shuffle reduce + one cross-warp stage
//                          (two __syncthreads, like the Metal twin's two
//                          threadgroup barriers)
//   gn_reduce_finalize   — block_reduce_sumsq + mean/rsqrt(var+eps) math
//
// All reductions accumulate in FP32; the storage type (float/half/bf16)
// only crosses device memory at load and store. The same kernels serve
// FP32, FP16, and BF16 via C++ templates — unlike the Metal side there is
// no per-dtype recompilation; bindings.cpp instantiates and dispatches.

#pragma once

#ifndef __CUDACC__
#error "kernels.cuh is device-only; include bindings.h from host code"
#endif

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

// Upper bound on warps per block (1024 threads / 32 lanes) — mirrors
// MAX_SIMDGROUPS in common.metal.
#define MAX_WARPS 32

// ---------------------------------------------------------------------------
// Packed 4-element storage types
// ---------------------------------------------------------------------------
//
// Alignment matches what the Python side guarantees via the storage-offset
// check (_kernel_arg): 16 bytes for float4, 8 bytes for the 2-byte types.
// Over-aligned types would break the reinterpret_cast on buffers whose base
// is only 8-byte aligned.

template <typename T>
struct Scalar4;

template <>
struct alignas(16) Scalar4<float> {
    float x, y, z, w;
};

template <>
struct alignas(8) Scalar4<c10::Half> {
    c10::Half x, y, z, w;
};

template <>
struct alignas(8) Scalar4<c10::BFloat16> {
    c10::BFloat16 x, y, z, w;
};

// Unpack a packed vector to float4 for compute.
__device__ __forceinline__ float4 unpack4(const Scalar4<float>& v) {
    return make_float4(v.x, v.y, v.z, v.w);
}

__device__ __forceinline__ float4 unpack4(const Scalar4<c10::Half>& v) {
    return make_float4(
        static_cast<float>(v.x),
        static_cast<float>(v.y),
        static_cast<float>(v.z),
        static_cast<float>(v.w)
    );
}

__device__ __forceinline__ float4 unpack4(const Scalar4<c10::BFloat16>& v) {
    return make_float4(
        static_cast<float>(v.x),
        static_cast<float>(v.y),
        static_cast<float>(v.z),
        static_cast<float>(v.w)
    );
}

// Pack a float4 back to the storage type (round-to-nearest). Explicitly
// specialized per storage type — call as ``pack4<SCALAR_T>(v)``.
template <typename SCALAR_T>
__device__ __forceinline__ Scalar4<SCALAR_T> pack4(const float4& v);

template <>
__device__ __forceinline__ Scalar4<float> pack4<float>(const float4& v) {
    return Scalar4<float>{v.x, v.y, v.z, v.w};
}

template <>
__device__ __forceinline__ Scalar4<c10::Half> pack4<c10::Half>(const float4& v) {
    return Scalar4<c10::Half>{
        static_cast<c10::Half>(v.x),
        static_cast<c10::Half>(v.y),
        static_cast<c10::Half>(v.z),
        static_cast<c10::Half>(v.w)
    };
}

template <>
__device__ __forceinline__ Scalar4<c10::BFloat16> pack4<c10::BFloat16>(const float4& v) {
    return Scalar4<c10::BFloat16>{
        static_cast<c10::BFloat16>(v.x),
        static_cast<c10::BFloat16>(v.y),
        static_cast<c10::BFloat16>(v.z),
        static_cast<c10::BFloat16>(v.w)
    };
}

// ---------------------------------------------------------------------------
// Reduction helpers (FP32 throughout)
// ---------------------------------------------------------------------------

// Butterfly shuffle reduce across a full warp. Full-mask participation is
// safe: every kernel launches whole blocks, so all 32 lanes are active.
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// Accumulate K-shifted (sum, sqsum) partials for x[0:total] into s/sq,
// strided by thread. ``inj`` is an optional second input added elementwise
// before accumulation (the HTDemucs encoder's conv-output + inject pattern);
// pass nullptr when absent. Uses Scalar4 vector loads when the element count
// is divisible by 4 (which also keeps every batch's base pointer 8-byte
// aligned); otherwise scalar loads. The shift by K makes the one-pass
// E[x^2] - E[x]^2 variance robust to large DC offsets (variance is
// shift-invariant; the caller adds K back to the mean).
template <typename SCALAR_T>
__device__ __forceinline__ void gn_accumulate_sumsq(
    const SCALAR_T* __restrict__  x,
    const SCALAR_T* __restrict__ inj,
    unsigned int total,
    float K,
    unsigned int tid,
    unsigned int tgs,
    float& s,
    float& sq
) {
    if ((total & 3u) == 0u) {
        const Scalar4<SCALAR_T>* __restrict__ x4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(x);
        const Scalar4<SCALAR_T>* __restrict__ j4 =
            inj == nullptr ? nullptr
                           : reinterpret_cast<const Scalar4<SCALAR_T>*>(inj);
        const unsigned int nv = total >> 2;
        for (unsigned int i = tid; i < nv; i += tgs) {
            float4 v = unpack4(x4[i]);
            if (j4 != nullptr) {
                const float4 w = unpack4(j4[i]);
                v.x += w.x;
                v.y += w.y;
                v.z += w.z;
                v.w += w.w;
            }
            v.x -= K; v.y -= K; v.z -= K; v.w -= K;
            s += v.x + v.y + v.z + v.w;
            sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (unsigned int i = tid; i < total; i += tgs) {
            float v = static_cast<float>(x[i]) - K;
            if (inj != nullptr) {
                v += static_cast<float>(inj[i]);
            }
            s += v;
            sq += v * v;
        }
    }
}

// Overload for the no-inject call sites. ``nullptr`` cannot be passed
// directly to the two-pointer form above: template argument deduction cannot
// infer ``SCALAR_T`` from ``std::nullptr_t``, so the cast lives here.
template <typename SCALAR_T>
__device__ __forceinline__ void gn_accumulate_sumsq(
    const SCALAR_T* __restrict__  x,
    unsigned int total,
    float K,
    unsigned int tid,
    unsigned int tgs,
    float& s,
    float& sq
) {
    gn_accumulate_sumsq(
        x, static_cast<const SCALAR_T*>(nullptr), total, K, tid, tgs, s, sq
    );
}

// Reduce per-thread (sum, sqsum) partials across the block. On return
// bcast[0]/bcast[1] hold the block totals, visible to every thread.
// Two-level scheme mirroring tg_reduce_sumsq in common.metal: warp shuffle
// within each 32-lane warp, then one shuffle across the per-warp partials.
__device__ __forceinline__ void block_reduce_sumsq(
    float s,
    float sq,
    float* __restrict__  sh_sum,
    float* __restrict__  sh_sq,
    float* __restrict__  bcast
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int wid = threadIdx.x >> 5;
    const unsigned int tgs = blockDim.x;

    s = warp_sum(s);
    sq = warp_sum(sq);
    if (lane == 0) {
        sh_sum[wid] = s;
        sh_sq[wid] = sq;
    }
    __syncthreads();
    if (wid == 0) {
        const unsigned int nwarp = (tgs + 31u) >> 5;
        float ts = lane < nwarp ? sh_sum[lane] : 0.0f;
        float tq = lane < nwarp ? sh_sq[lane] : 0.0f;
        ts = warp_sum(ts);
        tq = warp_sum(tq);
        if (lane == 0) {
            bcast[0] = ts;
            bcast[1] = tq;
        }
    }
    __syncthreads();
}

// block_reduce_sumsq, then convert the shifted totals into the
// normalization constants: bcast[0] = mean (K added back),
// bcast[1] = rsqrt(var + eps).
__device__ __forceinline__ void gn_reduce_finalize(
    float s,
    float sq,
    float K,
    unsigned int total,
    float eps,
    float* __restrict__  sh_sum,
    float* __restrict__  sh_sq,
    float* __restrict__  bcast
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int wid = threadIdx.x >> 5;
    const unsigned int tgs = blockDim.x;

    s = warp_sum(s);
    sq = warp_sum(sq);
    if (lane == 0) {
        sh_sum[wid] = s;
        sh_sq[wid] = sq;
    }
    __syncthreads();
    if (wid == 0) {
        const unsigned int nwarp = (tgs + 31u) >> 5;
        float ts = lane < nwarp ? sh_sum[lane] : 0.0f;
        float tq = lane < nwarp ? sh_sq[lane] : 0.0f;
        ts = warp_sum(ts);
        tq = warp_sum(tq);
        if (lane == 0) {
            const float invN = 1.0f / static_cast<float>(total);
            const float mean_d = ts * invN;
            const float var = fmaxf(tq * invN - mean_d * mean_d, 0.0f);
            bcast[0] = K + mean_d;
            bcast[1] = rsqrtf(var + eps);
        }
    }
    __syncthreads();
}


