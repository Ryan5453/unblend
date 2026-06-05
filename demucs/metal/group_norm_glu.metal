// GroupNorm fused with GLU (channel halving).
//
// Input shape (B, 2C, N), output (B, C, N). The reduction is over all 2C
// input channels (so the per-batch mean matches ``F.group_norm`` exactly);
// for each output channel we read ``a = norm(in[c])`` and
// ``b = norm(in[c + C])`` and combine via ``a * sigmoid(b)`` without ever
// writing the post-norm full-size tensor to memory.
//
// ``apply_norm_glu`` is the third stage of the multi-stage path,
// reading ``meanvar`` produced by ``finalize_meanvar`` in
// ``group_norm.metal``. Crucially it tiles the OUTPUT space (size C*N),
// not the input — each output element pulls its two input channels by
// absolute offset so the tile boundaries don't matter.
// Same source compiles for FP16 (half) and BF16 (bfloat) — the Python side
// prepends ``#define SCALAR_T bfloat`` to switch.
// See ``demucs/metal/__init__.py`` for the Python-side wrappers.

#include <metal_stdlib>
using namespace metal;

#ifndef SCALAR_T
#define SCALAR_T half
#endif

kernel void group_norm_g1_glu(
    device SCALAR_T*       out      [[buffer(0)]],   // (B, C/2, N)
    device const SCALAR_T* in_      [[buffer(1)]],   // (B, C,   N)
    device const SCALAR_T* weight   [[buffer(2)]],   // (C,)
    device const SCALAR_T* bias     [[buffer(3)]],   // (C,)
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

    const uint C_half = C >> 1;
    const uint total_in  = C * N;
    const uint total_out = C_half * N;
    device const SCALAR_T* in_b = in_ + b * total_in;
    device SCALAR_T*       out_b = out + b * total_out;

    float local_sum = 0.0f, local_sqsum = 0.0f;
    for (uint i = tid; i < total_in; i += tgs) {
        float v = float(in_b[i]);
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
        float invN = 1.0f / float(total_in);
        float mean = shared_sum[0] * invN;
        float meanSq = shared_sqsum[0] * invN;
        float var = max(meanSq - mean * mean, 0.0f);
        bcast_mean = mean;
        bcast_scale = rsqrt(var + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = bcast_mean, scale = bcast_scale;

    for (uint i = tid; i < total_out; i += tgs) {
        uint c_out = i / N;
        uint sp    = i % N;
        uint idx_a = c_out          * N + sp;
        uint idx_b = (c_out + C_half) * N + sp;
        float wa = float(weight[c_out]);
        float ba = float(bias[c_out]);
        float wb = float(weight[c_out + C_half]);
        float bb = float(bias[c_out + C_half]);
        float a = (float(in_b[idx_a]) - mean) * scale * wa + ba;
        float b_val = (float(in_b[idx_b]) - mean) * scale * wb + bb;
        float sig = 1.0f / (1.0f + exp(-b_val));
        out_b[i] = SCALAR_T(a * sig);
    }
}

kernel void apply_norm_glu(
    device SCALAR_T*        out          [[buffer(0)]],   // (B, C/2 * N)
    device const SCALAR_T*  in_          [[buffer(1)]],   // (B, C * N)
    device const float* meanvar      [[buffer(2)]],
    device const SCALAR_T*  weight       [[buffer(3)]],
    device const SCALAR_T*  bias         [[buffer(4)]],
    constant uint&      total_in_per_b  [[buffer(5)]], // C * N
    constant uint&      total_out_per_b [[buffer(6)]], // C_half * N
    constant uint&      num_tiles    [[buffer(7)]],
    constant uint&      N            [[buffer(8)]],
    constant uint&      C_half       [[buffer(9)]],
    uint bt  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    uint b = bt / num_tiles;
    uint t = bt % num_tiles;
    // Tile over the OUTPUT space; each output element pulls its two input
    // channels independently regardless of where they sit in the input tile.
    uint start = (uint)((ulong)t * (ulong)total_out_per_b / (ulong)num_tiles);
    uint end   = (uint)((ulong)(t + 1) * (ulong)total_out_per_b / (ulong)num_tiles);

    float mean  = meanvar[b * 2 + 0];
    float scale = meanvar[b * 2 + 1];

    device const SCALAR_T* x_b   = in_  + b * total_in_per_b;
    device SCALAR_T*       out_b = out  + b * total_out_per_b;

    for (uint i = start + tid; i < end; i += tgs) {
        uint c_out = i / N;
        uint sp    = i % N;
        uint idx_a = c_out          * N + sp;
        uint idx_b = (c_out + C_half) * N + sp;
        float wa = float(weight[c_out]);
        float ba = float(bias[c_out]);
        float wb = float(weight[c_out + C_half]);
        float bb = float(bias[c_out + C_half]);
        float a = (float(x_b[idx_a]) - mean) * scale * wa + ba;
        float b_val = (float(x_b[idx_b]) - mean) * scale * wb + bb;
        float sig = 1.0f / (1.0f + exp(-b_val));
        out_b[i] = SCALAR_T(a * sig);
    }
}
