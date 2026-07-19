// GroupNorm fused with GELU activation.
//
// Saves the round-trip that PyTorch would otherwise spend on the explicit
// ``functional.gelu(...)`` op after every ``norm1`` call inside HEncLayer
// / HDecLayer / DConv. We use the tanh approximation (the same form as
// ``F.gelu(approximate='tanh')``) because the MPS shader toolchain exposes no
// ``erf`` builtin. The per-element gap from PyTorch's default exact-erf GELU
// peaks at ~1e-3 (near |x|≈2), which is below FP16/BF16 output precision, so
// this path is numerically equivalent to the reference at the dtypes it runs
// in. (The FP32 fallback in ``unblend/metal/__init__.py`` uses exact erf, since
// there erf is available and the difference is no longer sub-precision.)
//
// ``apply_norm_gelu`` is the third stage of the multi-stage path; its
// mean/scale come from ``finalize_meanvar`` over in ``group_norm.metal``.
// Vector/scalar path selection and the reduction helpers are shared via
// ``common.metal``, which the Python side prepends before compiling.

inline float gelu_tanh(float y) {
    // sqrt(2/pi) = 0.7978845608028654
    float inner = 0.7978845608028654f * (y + 0.044715f * y * y * y);
    return 0.5f * y * (1.0f + tanh(inner));
}

inline float4 gelu_tanh4(float4 y) {
    float4 inner = 0.7978845608028654f * (y + 0.044715f * y * y * y);
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
    uint b    [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sid  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float sh_sum[MAX_SIMDGROUPS];
    threadgroup float sh_sqsum[MAX_SIMDGROUPS];
    threadgroup float bcast[2];

    const uint total = C * N;
    // ulong base offsets: b * total overflows 32 bits on huge inputs.
    device const SCALAR_T* in_b  = in_ + (ulong)b * total;
    device SCALAR_T*       out_b = out + (ulong)b * total;

    float K = float(in_b[0]);
    float s = 0.0f, sq = 0.0f;
    gn_accumulate_sumsq(in_b, total, K, tid, tgs, s, sq);
    gn_reduce_finalize(s, sq, K, total, eps, lane, sid, tgs, sh_sum, sh_sqsum, bcast);
    const float mean  = bcast[0];
    const float scale = bcast[1];

    if ((N & 3u) == 0u) {
        device const SCALAR4_T* in4  = (device const SCALAR4_T*)in_b;
        device SCALAR4_T*       out4 = (device SCALAR4_T*)out_b;
        const uint Nv = N >> 2;
        const uint nv = C * Nv;
        for (uint i = tid; i < nv; i += tgs) {
            uint  c  = i / Nv;
            float w  = float(weight[c]);
            float bv = float(bias[c]);
            float4 y = (float4(in4[i]) - mean) * scale * w + bv;
            out4[i]  = SCALAR4_T(gelu_tanh4(y));
        }
    } else {
        for (uint i = tid; i < total; i += tgs) {
            uint  c  = i / N;
            float w  = float(weight[c]);
            float bv = float(bias[c]);
            float y  = (float(in_b[i]) - mean) * scale * w + bv;
            out_b[i] = SCALAR_T(gelu_tanh(y));
        }
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

    float mean  = meanvar[b * 2 + 0];
    float scale = meanvar[b * 2 + 1];

    device const SCALAR_T* x_b   = in_ + (ulong)b * total_per_b;
    device SCALAR_T*       out_b = out + (ulong)b * total_per_b;

    if ((N & 3u) == 0u) {
        device const SCALAR4_T* in4  = (device const SCALAR4_T*)x_b;
        device SCALAR4_T*       out4 = (device SCALAR4_T*)out_b;
        const uint Nv = N >> 2;
        const uint nv = total_per_b >> 2;
        uint start = (uint)((ulong)t * (ulong)nv / (ulong)num_tiles);
        uint end   = (uint)((ulong)(t + 1) * (ulong)nv / (ulong)num_tiles);
        for (uint i = start + tid; i < end; i += tgs) {
            uint  c  = i / Nv;
            float w  = float(weight[c]);
            float bv = float(bias[c]);
            float4 y = (float4(in4[i]) - mean) * scale * w + bv;
            out4[i]  = SCALAR4_T(gelu_tanh4(y));
        }
    } else {
        uint start = (uint)((ulong)t * (ulong)total_per_b / (ulong)num_tiles);
        uint end   = (uint)((ulong)(t + 1) * (ulong)total_per_b / (ulong)num_tiles);
        for (uint i = start + tid; i < end; i += tgs) {
            uint  c  = i / N;
            float w  = float(weight[c]);
            float bv = float(bias[c]);
            float y  = (float(x_b[i]) - mean) * scale * w + bv;
            out_b[i] = SCALAR_T(gelu_tanh(y));
        }
    }
}
