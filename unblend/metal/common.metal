// Shared prelude prepended (by unblend/metal/__init__.py) to every kernel
// source in this folder before compilation. Provides the SCALAR_T /
// SCALAR4_T defaults and the threadgroup reduction helpers.
//
// Reductions use a two-level simdgroup reduce (simd_sum within each
// 32-lane simdgroup, then one simd_sum across the per-simdgroup partials)
// instead of a shared-memory tree: 2 threadgroup barriers instead of
// log2(tgs), and 32 floats of threadgroup memory instead of tgs.
//
// All reductions accumulate in FP32; the low-precision type (half/bfloat)
// only crosses the device-memory boundary at load and store. The same
// source compiles for both FP16 and BF16 — the Python side prepends
// ``#define SCALAR_T bfloat`` / ``#define SCALAR4_T bfloat4`` to switch.

#include <metal_stdlib>
using namespace metal;

#ifndef SCALAR_T
#define SCALAR_T half
#define SCALAR4_T half4
#endif

// Upper bound on simdgroups per threadgroup (1024 threads / 32 lanes).
#define MAX_SIMDGROUPS 32

// Accumulate K-shifted (sum, sqsum) partials for ``x[0:total]`` into
// ``s``/``sq``, strided by thread. Uses SCALAR4_T vector loads when the
// element count is divisible by 4 (which also keeps every batch's base
// pointer 8-byte aligned); otherwise scalar loads. The shift by K makes
// the one-pass ``E[d^2] - E[d]^2`` variance robust to large DC offsets
// (variance is shift-invariant; the caller adds K back to the mean).
inline void gn_accumulate_sumsq(
    device const SCALAR_T* x,
    uint total,
    float K,
    uint tid,
    uint tgs,
    thread float& s,
    thread float& sq
) {
    if ((total & 3u) == 0u) {
        device const SCALAR4_T* x4 = (device const SCALAR4_T*)x;
        const uint nv = total >> 2;
        for (uint i = tid; i < nv; i += tgs) {
            float4 v = float4(x4[i]) - K;
            s  += v.x + v.y + v.z + v.w;
            sq += dot(v, v);
        }
    } else {
        for (uint i = tid; i < total; i += tgs) {
            float v = float(x[i]) - K;
            s  += v;
            sq += v * v;
        }
    }
}

// Reduce per-thread (sum, sqsum) partials across the threadgroup. On
// return ``bcast[0]``/``bcast[1]`` hold the threadgroup totals, visible
// to every thread.
inline void tg_reduce_sumsq(
    float s,
    float sq,
    uint lane,
    uint sid,
    uint tgs,
    threadgroup float* sh_sum,
    threadgroup float* sh_sqsum,
    threadgroup float* bcast
) {
    s  = simd_sum(s);
    sq = simd_sum(sq);
    if (lane == 0) {
        sh_sum[sid]   = s;
        sh_sqsum[sid] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0) {
        const uint nsimd = (tgs + 31) >> 5;
        float ts = lane < nsimd ? sh_sum[lane]   : 0.0f;
        float tq = lane < nsimd ? sh_sqsum[lane] : 0.0f;
        ts = simd_sum(ts);
        tq = simd_sum(tq);
        if (lane == 0) {
            bcast[0] = ts;
            bcast[1] = tq;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// tg_reduce_sumsq, then convert the shifted totals into the normalization
// constants: ``bcast[0] = mean`` (K added back), ``bcast[1] = rsqrt(var+eps)``.
inline void gn_reduce_finalize(
    float s,
    float sq,
    float K,
    uint total,
    float eps,
    uint lane,
    uint sid,
    uint tgs,
    threadgroup float* sh_sum,
    threadgroup float* sh_sqsum,
    threadgroup float* bcast
) {
    s  = simd_sum(s);
    sq = simd_sum(sq);
    if (lane == 0) {
        sh_sum[sid]   = s;
        sh_sqsum[sid] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0) {
        const uint nsimd = (tgs + 31) >> 5;
        float ts = lane < nsimd ? sh_sum[lane]   : 0.0f;
        float tq = lane < nsimd ? sh_sqsum[lane] : 0.0f;
        ts = simd_sum(ts);
        tq = simd_sum(tq);
        if (lane == 0) {
            float invN   = 1.0f / float(total);
            float mean_d = ts * invN;
            float var    = max(tq * invN - mean_d * mean_d, 0.0f);
            bcast[0] = K + mean_d;
            bcast[1] = rsqrt(var + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
