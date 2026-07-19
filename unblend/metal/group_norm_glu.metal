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
// Vector/scalar path selection and the reduction helpers are shared via
// ``common.metal``, which the Python side prepends before compiling.

kernel void group_norm_g1_glu(
    device SCALAR_T*       out      [[buffer(0)]],   // (B, C/2, N)
    device const SCALAR_T* in_      [[buffer(1)]],   // (B, C,   N)
    device const SCALAR_T* weight   [[buffer(2)]],   // (C,)
    device const SCALAR_T* bias     [[buffer(3)]],   // (C,)
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

    const uint C_half = C >> 1;
    const uint total_in  = C * N;
    const uint total_out = C_half * N;
    // ulong base offsets: b * total overflows 32 bits on huge inputs.
    device const SCALAR_T* in_b  = in_ + (ulong)b * total_in;
    device SCALAR_T*       out_b = out + (ulong)b * total_out;

    float K = float(in_b[0]);
    float s = 0.0f, sq = 0.0f;
    gn_accumulate_sumsq(in_b, total_in, K, tid, tgs, s, sq);
    gn_reduce_finalize(s, sq, K, total_in, eps, lane, sid, tgs, sh_sum, sh_sqsum, bcast);
    const float mean  = bcast[0];
    const float scale = bcast[1];

    if ((N & 3u) == 0u) {
        device const SCALAR4_T* in4  = (device const SCALAR4_T*)in_b;
        device SCALAR4_T*       out4 = (device SCALAR4_T*)out_b;
        const uint Nv    = N >> 2;
        const uint nv    = C_half * Nv;
        const uint boff  = C_half * Nv;   // vector offset of channel c + C_half
        for (uint i = tid; i < nv; i += tgs) {
            uint  c  = i / Nv;
            float wa = float(weight[c]);
            float ba = float(bias[c]);
            float wb = float(weight[c + C_half]);
            float bb = float(bias[c + C_half]);
            float4 a = (float4(in4[i]) - mean) * scale * wa + ba;
            float4 g = (float4(in4[i + boff]) - mean) * scale * wb + bb;
            float4 sig = 1.0f / (1.0f + exp(-g));
            out4[i] = SCALAR4_T(a * sig);
        }
    } else {
        for (uint i = tid; i < total_out; i += tgs) {
            uint c_out = i / N;
            uint sp    = i % N;
            uint idx_a = c_out            * N + sp;
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

    float mean  = meanvar[b * 2 + 0];
    float scale = meanvar[b * 2 + 1];

    device const SCALAR_T* x_b   = in_  + (ulong)b * total_in_per_b;
    device SCALAR_T*       out_b = out  + (ulong)b * total_out_per_b;

    if ((N & 3u) == 0u) {
        device const SCALAR4_T* in4  = (device const SCALAR4_T*)x_b;
        device SCALAR4_T*       out4 = (device SCALAR4_T*)out_b;
        const uint Nv   = N >> 2;
        const uint nv   = total_out_per_b >> 2;
        const uint boff = C_half * Nv;
        uint start = (uint)((ulong)t * (ulong)nv / (ulong)num_tiles);
        uint end   = (uint)((ulong)(t + 1) * (ulong)nv / (ulong)num_tiles);
        for (uint i = start + tid; i < end; i += tgs) {
            uint  c  = i / Nv;
            float wa = float(weight[c]);
            float ba = float(bias[c]);
            float wb = float(weight[c + C_half]);
            float bb = float(bias[c + C_half]);
            float4 a = (float4(in4[i]) - mean) * scale * wa + ba;
            float4 g = (float4(in4[i + boff]) - mean) * scale * wb + bb;
            float4 sig = 1.0f / (1.0f + exp(-g));
            out4[i] = SCALAR4_T(a * sig);
        }
    } else {
        // Tile over the OUTPUT space; each output element pulls its two input
        // channels independently regardless of where they sit in the input tile.
        uint start = (uint)((ulong)t * (ulong)total_out_per_b / (ulong)num_tiles);
        uint end   = (uint)((ulong)(t + 1) * (ulong)total_out_per_b / (ulong)num_tiles);
        for (uint i = start + tid; i < end; i += tgs) {
            uint c_out = i / N;
            uint sp    = i % N;
            uint idx_a = c_out            * N + sp;
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
}
