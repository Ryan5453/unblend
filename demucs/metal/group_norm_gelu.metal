// GroupNorm fused with GELU activation.
//
// Saves the round-trip that PyTorch would otherwise spend on the explicit
// ``functional.gelu(...)`` op after every ``norm1`` call inside HEncLayer
// / HDecLayer / DConv. We use the tanh approximation (the same form as
// ``F.gelu(approximate='tanh')``) because the MPS shader toolchain exposes no
// ``erf`` builtin. The per-element gap from PyTorch's default exact-erf GELU
// peaks at ~1e-3 (near |x|≈2), which is below FP16/BF16 output precision, so
// this path is numerically equivalent to the reference at the dtypes it runs
// in. (The FP32 fallback in ``demucs/metal/__init__.py`` uses exact erf, since
// there erf is available and the difference is no longer sub-precision.)
//
// ``apply_norm_gelu`` is the third stage of the multi-stage path; its
// mean/scale come from ``finalize_meanvar`` over in ``group_norm.metal``.
// Same source compiles for FP16 (half) and BF16 (bfloat) — the Python side
// prepends ``#define SCALAR_T bfloat`` to switch.
// See ``demucs/metal/__init__.py`` for the Python-side wrappers.

#include <metal_stdlib>
using namespace metal;

#ifndef SCALAR_T
#define SCALAR_T half
#endif

inline float gelu_tanh(float y) {
    // sqrt(2/pi) = 0.7978845608028654
    float inner = 0.7978845608028654f * (y + 0.044715f * y * y * y);
    return 0.5f * y * (1.0f + tanh(inner));
}

kernel void group_norm_g1_gelu(
    device SCALAR_T*       out      [[buffer(0)]],
    device const SCALAR_T* in_      [[buffer(1)]],
    device const SCALAR_T* weight   [[buffer(2)]],
    device const SCALAR_T* bias     [[buffer(3)]],
    constant uint&     C        [[buffer(4)]],
    constant uint&     N        [[buffer(5)]],
    constant float&    eps      [[buffer(6)]],
    uint b   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    constexpr uint MAX_TGS = 1024;
    threadgroup float shared_sum[MAX_TGS];
    threadgroup float shared_sqsum[MAX_TGS];

    const uint total = C * N;
    device const SCALAR_T* in_b = in_ + b * total;
    device SCALAR_T*       out_b = out + b * total;

    // Shift by the batch's first element before summing so the one-pass
    // variance doesn't lose precision to cancellation on large-DC inputs
    // (see group_norm.metal:group_norm_g1 for the rationale).
    float K = float(in_b[0]);
    float local_sum = 0.0f, local_sqsum = 0.0f;
    for (uint i = tid; i < total; i += tgs) {
        float v = float(in_b[i]) - K;
        local_sum += v;
        local_sqsum += v * v;
    }
    shared_sum[tid] = local_sum;
    shared_sqsum[tid] = local_sqsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid]   += shared_sum[tid + stride];
            shared_sqsum[tid] += shared_sqsum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float bcast_mean, bcast_scale;
    if (tid == 0) {
        float invN = 1.0f / float(total);
        float mean_d = shared_sum[0] * invN;
        float var = max(shared_sqsum[0] * invN - mean_d * mean_d, 0.0f);
        bcast_mean = K + mean_d;
        bcast_scale = rsqrt(var + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = bcast_mean, scale = bcast_scale;

    for (uint i = tid; i < total; i += tgs) {
        uint  c_idx = i / N;
        float v     = float(in_b[i]);
        float w     = float(weight[c_idx]);
        float bv    = float(bias[c_idx]);
        float y     = (v - mean) * scale * w + bv;
        out_b[i]    = SCALAR_T(gelu_tanh(y));
    }
}

kernel void apply_norm_gelu(
    device SCALAR_T*        out          [[buffer(0)]],
    device const SCALAR_T*  in_          [[buffer(1)]],
    device const float* meanvar      [[buffer(2)]],
    device const SCALAR_T*  weight       [[buffer(3)]],
    device const SCALAR_T*  bias         [[buffer(4)]],
    constant uint&      total_per_b  [[buffer(5)]],
    constant uint&      num_tiles    [[buffer(6)]],
    constant uint&      N            [[buffer(7)]],
    uint bt  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    uint b = bt / num_tiles;
    uint t = bt % num_tiles;
    uint start = (uint)((ulong)t * (ulong)total_per_b / (ulong)num_tiles);
    uint end   = (uint)((ulong)(t + 1) * (ulong)total_per_b / (ulong)num_tiles);

    float mean  = meanvar[b * 2 + 0];
    float scale = meanvar[b * 2 + 1];

    device const SCALAR_T* x_b   = in_ + b * total_per_b;
    device SCALAR_T*       out_b = out + b * total_per_b;

    for (uint i = start + tid; i < end; i += tgs) {
        uint  c_idx = i / N;
        float v     = float(x_b[i]);
        float w     = float(weight[c_idx]);
        float bv    = float(bias[c_idx]);
        float y     = (v - mean) * scale * w + bv;
        out_b[i]    = SCALAR_T(gelu_tanh(y));
    }
}
