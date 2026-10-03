// Shared prelude included by every kernel translation unit in this folder
// and by bindings.cpp. CUDA port of ``unblend/metal/common.metal``.
//
// Provides the SCALAR_T template vocabulary (Scalar4 packed loads), the
// FP32 conversion helpers, and the two-level warp/block reduction helpers
// that mirror the Metal simdgroup reductions:
//
//   gn_thread_partial    — per-thread shifted (n, K, sum, sum-of-squares),
//                          vectorized 4-wide when element count % 4 == 0
//   gn_tile_bounds       — multi-stage tile ranges (32-bit arithmetic)
//   block_merge_moments  — per-thread partials -> block (mean, M2), via a
//                          warp stage and one cross-warp stage
//   gn_reduce_finalize   — block_merge_moments + (mean, rsqrt(var+eps))
//
// All reductions accumulate in FP32; the storage type (float/half/bf16)
// only crosses device memory at load and store. The same kernels serve
// FP32, FP16, and BF16 via C++ templates, instantiated once and selected at
// launch by UNBLEND_DISPATCH (bindings.h) rather than recompiled per dtype.

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

// Butterfly shuffle reduce across a full warp. Full-mask participation
// requires blocks of at least 32 threads, which the launcher guarantees
// (_MIN_TGS in __init__.py).
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// Shared-memory floats the reduction helpers need: per-warp (n, mean, M2)
// partials plus a 2-float broadcast slot at GN_BCAST. Kernels declare
// ``__shared__ float sh[GN_SHARED_FLOATS];`` and pass ``sh``.
#define GN_BCAST (3 * MAX_WARPS)
#define GN_SHARED_FLOATS (3 * MAX_WARPS + 2)

// One thread's share of a reduction: ``n`` elements accumulated as the
// shifted sums ``s = sum(x - K)`` and ``sq = sum((x - K)^2)``.
struct GnPartial {
    float n;
    float K;
    float s;
    float sq;
};

// x[i] (+ inj[i] when present) in FP32.
template <typename SCALAR_T>
__device__ __forceinline__ float gn_load(
    const SCALAR_T* __restrict__  x,
    const SCALAR_T* __restrict__ inj,
    unsigned int i
) {
    float v = static_cast<float>(x[i]);
    if (inj != nullptr) {
        v += static_cast<float>(inj[i]);
    }
    return v;
}

// Partial for the elements of x[lo:hi) (plus the optional elementwise
// second input ``inj``, the HTDemucs encoder's conv-output + inject
// pattern; nullptr when absent) visited by thread ``tid`` of a
// ``tgs``-thread stride. Uses Scalar4 vector loads when ``total`` (the
// per-batch element count ``x`` is based on) is divisible by 4, which also
// keeps every batch's base pointer 8-byte aligned; ``lo``/``hi`` must then
// be multiples of 4 (see gn_tile_bounds). Otherwise scalar loads.
//
// The shift ``K`` is the mean of the thread's first 4 elements (its first
// Scalar4, or its first 4 strided scalars). Shifting keeps the one-pass
// sums accurate under large DC offsets. The shift is per thread rather than
// one shared x[0] because a shared outlier shift puts its offset into every
// term: an fp16 x[0] = 3000 over N = 750k lost ~5% of the variance to fp32
// cancellation. Per thread, an outlier only lands in one thread's shift,
// diluted 4x there. block_merge_moments combines the partials without
// reintroducing the cancellation. Mirrors gn_thread_partial in
// common.metal.
template <typename SCALAR_T>
__device__ __forceinline__ GnPartial gn_thread_partial(
    const SCALAR_T* __restrict__  x,
    const SCALAR_T* __restrict__ inj,
    unsigned int total,
    unsigned int lo,
    unsigned int hi,
    unsigned int tid,
    unsigned int tgs
) {
    GnPartial p = {0.0f, 0.0f, 0.0f, 0.0f};
    unsigned int cnt = 0u;
    if ((total & 3u) == 0u) {
        const Scalar4<SCALAR_T>* __restrict__ x4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(x);
        const Scalar4<SCALAR_T>* __restrict__ j4 =
            inj == nullptr ? nullptr
                           : reinterpret_cast<const Scalar4<SCALAR_T>*>(inj);
        const unsigned int vhi = hi >> 2;
        const unsigned int i0 = (lo >> 2) + tid;
        if (i0 < vhi) {
            float4 v = unpack4(x4[i0]);
            if (j4 != nullptr) {
                const float4 w = unpack4(j4[i0]);
                v.x += w.x;
                v.y += w.y;
                v.z += w.z;
                v.w += w.w;
            }
            p.K = 0.25f * (v.x + v.y + v.z + v.w);
        }
        for (unsigned int i = i0; i < vhi; i += tgs) {
            float4 v = unpack4(x4[i]);
            if (j4 != nullptr) {
                const float4 w = unpack4(j4[i]);
                v.x += w.x;
                v.y += w.y;
                v.z += w.z;
                v.w += w.w;
            }
            v.x -= p.K; v.y -= p.K; v.z -= p.K; v.w -= p.K;
            p.s += v.x + v.y + v.z + v.w;
            p.sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            cnt += 4u;
        }
    } else {
        const unsigned int i0 = lo + tid;
        if (i0 < hi) {
            // Independent loads (the loop re-reads them from cache); a thread
            // with fewer than 4 elements pads with its first.
            const unsigned int i1 = i0 + tgs, i2 = i1 + tgs, i3 = i2 + tgs;
            const float k0 = gn_load(x, inj, i0);
            const float k1 = i1 < hi ? gn_load(x, inj, i1) : k0;
            const float k2 = i2 < hi ? gn_load(x, inj, i2) : k0;
            const float k3 = i3 < hi ? gn_load(x, inj, i3) : k0;
            p.K = 0.25f * (k0 + k1 + k2 + k3);
        }
        for (unsigned int i = i0; i < hi; i += tgs) {
            const float v = gn_load(x, inj, i) - p.K;
            p.s += v;
            p.sq += v * v;
            cnt += 1u;
        }
    }
    p.n = static_cast<float>(cnt);
    return p;
}

// Overload for the no-inject call sites. ``nullptr`` cannot be passed
// directly to the two-pointer form above: template argument deduction cannot
// infer ``SCALAR_T`` from ``std::nullptr_t``, so the cast lives here.
template <typename SCALAR_T>
__device__ __forceinline__ GnPartial gn_thread_partial(
    const SCALAR_T* __restrict__  x,
    unsigned int total,
    unsigned int lo,
    unsigned int hi,
    unsigned int tid,
    unsigned int tgs
) {
    return gn_thread_partial(
        x, static_cast<const SCALAR_T*>(nullptr), total, lo, hi, tid, tgs
    );
}

// floor(t * n / num_tiles) in 32-bit arithmetic (t * n can overflow 32
// bits; the 64-bit divide that would avoid it is emulated). Exact while
// num_tiles <= 65536, so that t * r < 2^32; the host caps it at
// _MULTI_STAGE_MAX_TILES. Mirrors common.metal.
__device__ __forceinline__ unsigned int gn_split_point(
    unsigned int t, unsigned int num_tiles, unsigned int n
) {
    const unsigned int q = n / num_tiles;
    const unsigned int r = n - q * num_tiles;
    return t * q + (t * r) / num_tiles;
}

// Element range [lo, hi) of tile ``t`` of ``num_tiles`` over a batch of
// ``total`` elements. When total % 4 == 0 the tiles split the Scalar4
// vector space, so both bounds are multiples of 4. Shared by partial_reduce
// (which reduces the tile) and finalize_meanvar (which needs each tile's
// element count to merge the tile moments).
__device__ __forceinline__ uint2 gn_tile_bounds(
    unsigned int t, unsigned int num_tiles, unsigned int total
) {
    if ((total & 3u) == 0u) {
        const unsigned int nv = total >> 2;
        return make_uint2(
            gn_split_point(t, num_tiles, nv) << 2,
            gn_split_point(t + 1u, num_tiles, nv) << 2
        );
    }
    return make_uint2(
        gn_split_point(t, num_tiles, total),
        gn_split_point(t + 1u, num_tiles, total)
    );
}

// Merge every thread's partial into the block's statistics over ``count``
// elements (the sum of every thread's ``p.n``). On return sh[GN_BCAST]
// holds the mean and sh[GN_BCAST + 1] either the M2 (sum of squared
// deviations from the mean) or, with ``normalize``, the normalization scale
// rsqrt(M2 / count + eps); both visible to every thread.
//
// Each warp first takes its mean ``m`` and then the exact expansion
// sum (x - m)^2 = sq + (K - m) * (2 s + n (K - m)) per thread, so no
// per-thread division by ``n`` is needed; warp 0 then merges the per-warp
// (n, mean, M2) with the parallel-variance identity
// M2 = sum M2_w + n_w (mean_w - mean)^2. Every term summed is either
// non-negative or a thread-local expansion, so the merge adds no
// cancellation. Same barrier count as a plain two-level sum reduce.
// Mirrors tg_merge_moments in common.metal.
__device__ __forceinline__ void block_merge_moments(
    GnPartial p,
    unsigned int count,
    bool normalize,
    float eps,
    float* sh
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int wid = threadIdx.x >> 5;
    const unsigned int tgs = blockDim.x;
    float* sh_n = sh;
    float* sh_mean = sh + MAX_WARPS;
    float* sh_m2 = sh + 2 * MAX_WARPS;

    const float n_w = warp_sum(p.n);
    const float m_w = warp_sum(p.n * p.K + p.s) / fmaxf(n_w, 1.0f);
    const float D = p.K - m_w;
    const float m2_w = warp_sum(p.sq + D * (2.0f * p.s + p.n * D));
    if (lane == 0) {
        sh_n[wid] = n_w;
        sh_mean[wid] = m_w;
        sh_m2[wid] = m2_w;
    }
    __syncthreads();
    if (wid == 0) {
        const unsigned int nwarp = (tgs + 31u) >> 5;
        const float n = lane < nwarp ? sh_n[lane] : 0.0f;
        const float m = lane < nwarp ? sh_mean[lane] : 0.0f;
        const float m2 = lane < nwarp ? sh_m2[lane] : 0.0f;
        const float inv_count = 1.0f / static_cast<float>(count > 0u ? count : 1u);
        const float mean = warp_sum(n * m) * inv_count;
        const float d = m - mean;
        const float M2 = warp_sum(m2 + n * d * d);
        if (lane == 0) {
            sh[GN_BCAST] = mean;
            sh[GN_BCAST + 1] = normalize ? rsqrtf(M2 * inv_count + eps) : M2;
        }
    }
    __syncthreads();
}

// block_merge_moments over ``total`` elements, returning the normalization
// constants (mean, rsqrt(var + eps)) to every thread.
__device__ __forceinline__ float2 gn_reduce_finalize(
    GnPartial p,
    unsigned int total,
    float eps,
    float* sh
) {
    block_merge_moments(p, total, true, eps, sh);
    return make_float2(sh[GN_BCAST], sh[GN_BCAST + 1]);
}
