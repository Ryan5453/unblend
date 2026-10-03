// Shared prelude prepended (by unblend/metal/__init__.py) to every kernel
// source in this folder before compilation. Provides the SCALAR_T /
// SCALAR4_T defaults and the threadgroup reduction helpers.
//
// Reductions use a two-level simdgroup reduce (simd_sum within each
// 32-lane simdgroup, then one simd_sum across the per-simdgroup partials)
// instead of a shared-memory tree: 2 threadgroup barriers instead of
// log2(tgs), and GN_SHARED_FLOATS of threadgroup memory instead of tgs.
//
// All reductions accumulate in FP32; the low-precision type (half/bfloat)
// only crosses the device-memory boundary at load and store. The same
// source compiles for FP32, FP16 and BF16: the Python side prepends
// ``#define SCALAR_T`` / ``#define SCALAR4_T`` for the requested dtype.

#include <metal_stdlib>
using namespace metal;

#ifndef SCALAR_T
#define SCALAR_T half
#define SCALAR4_T half4
#endif

// Upper bound on simdgroups per threadgroup (1024 threads / 32 lanes).
#define MAX_SIMDGROUPS 32

// Threadgroup scratch the reduction helpers need: per-simdgroup (n, mean,
// M2) partials plus a 2-float broadcast slot. Kernels declare
// ``threadgroup float sh[GN_SHARED_FLOATS];`` and pass ``sh``.
#define GN_SHARED_FLOATS (3 * MAX_SIMDGROUPS + 2)

// One thread's share of a reduction: ``n`` elements accumulated as the
// shifted sums ``s = sum(x - K)`` and ``sq = sum((x - K)^2)``.
struct GnPartial {
    float n;
    float K;
    float s;
    float sq;
};

// Partial for the elements of ``x[lo:hi)`` visited by thread ``tid`` of a
// ``tgs``-thread stride. Uses SCALAR4_T vector loads when ``total`` (the
// per-batch element count ``x`` is based on) is divisible by 4, which also
// keeps every batch's base pointer 8-byte aligned; ``lo``/``hi`` must then
// be multiples of 4 (see gn_tile_bounds). Otherwise scalar loads.
//
// The shift ``K`` is the mean of the thread's first 4 elements (its first
// SCALAR4_T, or its first 4 strided scalars). Shifting keeps the one-pass
// sums accurate under large DC offsets. The shift is per thread rather than
// one shared ``x[0]`` because a shared outlier shift puts its offset into
// every term: an fp16 ``x[0] = 3000`` over N = 750k lost ~5% of the
// variance to fp32 cancellation. Per thread, an outlier only lands in one
// thread's shift, diluted 4x there. tg_merge_moments combines the partials
// without reintroducing the cancellation. (The count is an integer in the
// loop: a float counter measurably slowed the load loop.)
inline GnPartial gn_thread_partial(
    device const SCALAR_T* x,
    uint total,
    uint lo,
    uint hi,
    uint tid,
    uint tgs
) {
    GnPartial p = {0.0f, 0.0f, 0.0f, 0.0f};
    uint cnt = 0u;
    if ((total & 3u) == 0u) {
        device const SCALAR4_T* x4 = (device const SCALAR4_T*)x;
        const uint vhi = hi >> 2;
        const uint i0 = (lo >> 2) + tid;
        if (i0 < vhi) {
            const float4 v = float4(x4[i0]);
            p.K = 0.25f * (v.x + v.y + v.z + v.w);
        }
        for (uint i = i0; i < vhi; i += tgs) {
            float4 v = float4(x4[i]) - p.K;
            p.s  += v.x + v.y + v.z + v.w;
            p.sq += dot(v, v);
            cnt += 4u;
        }
    } else {
        const uint i0 = lo + tid;
        if (i0 < hi) {
            // Independent loads (the loop re-reads them from cache); a thread
            // with fewer than 4 elements pads with its first.
            const uint i1 = i0 + tgs, i2 = i1 + tgs, i3 = i2 + tgs;
            const float k0 = float(x[i0]);
            const float k1 = i1 < hi ? float(x[i1]) : k0;
            const float k2 = i2 < hi ? float(x[i2]) : k0;
            const float k3 = i3 < hi ? float(x[i3]) : k0;
            p.K = 0.25f * (k0 + k1 + k2 + k3);
        }
        for (uint i = i0; i < hi; i += tgs) {
            float v = float(x[i]) - p.K;
            p.s  += v;
            p.sq += v * v;
            cnt += 1u;
        }
    }
    p.n = float(cnt);
    return p;
}

// ``floor(t * n / num_tiles)`` in 32-bit arithmetic: ``t * n`` can overflow
// 32 bits, and the 64-bit divide that would avoid it is emulated on Apple
// GPUs and cost more than the whole tile reduction. Exact while
// ``num_tiles <= 65536`` (so that ``t * r < 2^32``); the host caps it at
// _MULTI_STAGE_MAX_TILES.
inline uint gn_split_point(uint t, uint num_tiles, uint n) {
    const uint q = n / num_tiles;
    const uint r = n - q * num_tiles;
    return t * q + (t * r) / num_tiles;
}

// Element range ``[lo, hi)`` of tile ``t`` of ``num_tiles`` over a batch of
// ``total`` elements. When ``total % 4 == 0`` the tiles split the SCALAR4_T
// vector space, so both bounds are multiples of 4. Shared by partial_reduce
// (which reduces the tile) and finalize_meanvar (which needs each tile's
// element count to merge the tile moments).
inline uint2 gn_tile_bounds(uint t, uint num_tiles, uint total) {
    if ((total & 3u) == 0u) {
        const uint nv = total >> 2;
        return uint2(
            gn_split_point(t, num_tiles, nv) << 2,
            gn_split_point(t + 1u, num_tiles, nv) << 2
        );
    }
    return uint2(
        gn_split_point(t, num_tiles, total),
        gn_split_point(t + 1u, num_tiles, total)
    );
}

// Merge every thread's partial into the threadgroup's statistics over
// ``count`` elements (the sum of every thread's ``p.n``). On return
// ``sh[GN_BCAST]`` holds the mean and ``sh[GN_BCAST + 1]`` either the M2
// (sum of squared deviations from the mean) or, with ``normalize``, the
// normalization scale ``rsqrt(M2 / count + eps)``; both visible to every
// thread. ``normalize`` is a compile-time constant at every call site.
//
// Each simdgroup first takes its mean ``m`` and then the exact expansion
// ``sum (x - m)^2 = sq + (K - m) * (2 s + n (K - m))`` per thread, so no
// per-thread division by ``n`` is needed; simdgroup 0 then merges the
// per-simdgroup (n, mean, M2) with the parallel-variance identity
// ``M2 = sum M2_s + n_s (mean_s - mean)^2``. Every term summed is either
// non-negative or a thread-local expansion, so the merge adds no
// cancellation. Same barrier count as a plain two-level sum reduce.
#define GN_BCAST (3 * MAX_SIMDGROUPS)

inline void tg_merge_moments(
    GnPartial p,
    uint count,
    bool normalize,
    float eps,
    uint lane,
    uint sid,
    uint tgs,
    threadgroup float* sh
) {
    threadgroup float* sh_n    = sh;
    threadgroup float* sh_mean = sh + MAX_SIMDGROUPS;
    threadgroup float* sh_m2   = sh + 2 * MAX_SIMDGROUPS;

    const float n_s = simd_sum(p.n);
    const float m_s = simd_sum(p.n * p.K + p.s) / max(n_s, 1.0f);
    const float D   = p.K - m_s;
    const float m2_s = simd_sum(p.sq + D * (2.0f * p.s + p.n * D));
    if (lane == 0) {
        sh_n[sid]    = n_s;
        sh_mean[sid] = m_s;
        sh_m2[sid]   = m2_s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0) {
        const uint nsimd = (tgs + 31) >> 5;
        const float n  = lane < nsimd ? sh_n[lane]    : 0.0f;
        const float m  = lane < nsimd ? sh_mean[lane] : 0.0f;
        const float m2 = lane < nsimd ? sh_m2[lane]   : 0.0f;
        const float inv_count = 1.0f / float(max(count, 1u));
        const float mean = simd_sum(n * m) * inv_count;
        const float d = m - mean;
        const float M2 = simd_sum(m2 + n * d * d);
        if (lane == 0) {
            sh[GN_BCAST]     = mean;
            sh[GN_BCAST + 1] = normalize ? rsqrt(M2 * inv_count + eps) : M2;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// tg_merge_moments over ``total`` elements, returning the normalization
// constants ``(mean, rsqrt(var + eps))`` to every thread.
inline float2 gn_reduce_finalize(
    GnPartial p,
    uint total,
    float eps,
    uint lane,
    uint sid,
    uint tgs,
    threadgroup float* sh
) {
    tg_merge_moments(p, total, true, eps, lane, sid, tgs, sh);
    return float2(sh[GN_BCAST], sh[GN_BCAST + 1]);
}

// ---------------------------------------------------------------------------
// Channel-walk fallback for the apply loops
// ---------------------------------------------------------------------------
//
// The vector paths in the apply kernels need the affine index to be constant
// across each SCALAR4_T, i.e. N % 4 == 0 for the channel-first layout. When
// it is not, the kernels walk the tile one channel at a time so the affine
// parameters load once per channel and no per-element ``i / N`` divide (a
// multi-instruction sequence on Apple GPUs) is needed. Within a channel the
// elements are contiguous, so a scalar head brings the cursor to a 4-element
// boundary, a SCALAR4_T body covers the middle, and a scalar tail finishes
// the channel.

// Bounds of channel ``c``'s slice of the tile range ``[lo, end)``:
//   hi   - end of this channel's slice
//   head - first 4-element-aligned index at or after ``lo``, clamped to ``hi``
//   vend - end of the 4-element-aligned body
#define GN_CHANNEL_BOUNDS(lo, end, N, c, hi, head, vend)  \
    const uint hi   = min((end), ((c) + 1u) * (N));       \
    const uint head = min(hi, ((lo) + 3u) & ~3u);         \
    const uint vend = head + (((hi - head) >> 2) << 2)

// Whether the vectorized body is safe for a per-batch stride of ``per_b``
// elements. Every buffer the apply kernels touch is indexed as
// ``batch_base + i`` with ``batch_base`` a multiple of the per-batch stride,
// so a 4-aligned ``i`` only yields a 4-aligned absolute offset -- and hence a
// legally aligned SCALAR4_T access -- when that stride is itself a multiple
// of 4 (C=6, N=85995, for example, is not).
#define GN_CHANNEL_VECTORIZABLE(per_b) (((per_b) & 3u) == 0u)
