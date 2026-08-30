// DConv envelope kernels: the post-conv2 path inside a DConv layer fused
// down to a single launch (or, for the multi-stage variant, two launches
// over the input followed by one over the output).
//
// ``norm_glu_ls_resid`` (single-stage) and ``apply_norm_glu_ls_resid``
// (multi-stage third stage) absorb GroupNorm into the same fused op:
//   output = residual + layer_scale * glu(group_norm(z))
// which replaces FOUR previously separate kernel launches (group_norm,
// glu, layerscale mul, residual add) with one. Used per DConv sub-layer.
// Vector/scalar path selection and the reduction helpers are shared via
// ``common.metal``, which the Python side prepends before compiling.

kernel void norm_glu_ls_resid(
    device SCALAR_T*       out          [[buffer(0)]],   // (B, C, N)
    device const SCALAR_T* z            [[buffer(1)]],   // (B, 2C, N)
    device const SCALAR_T* residual     [[buffer(2)]],   // (B, C, N)
    device const SCALAR_T* nweight      [[buffer(3)]],   // (2C,)
    device const SCALAR_T* nbias        [[buffer(4)]],   // (2C,)
    device const SCALAR_T* layer_scale  [[buffer(5)]],   // (C,)
    constant uint&     C2           [[buffer(6)]],   // 2*C
    constant uint&     N            [[buffer(7)]],
    constant float&    eps          [[buffer(8)]],
    uint b    [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sid  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float sh_sum[MAX_SIMDGROUPS];
    threadgroup float sh_sqsum[MAX_SIMDGROUPS];
    threadgroup float bcast[2];

    const uint C = C2 >> 1;
    const uint total_in  = C2 * N;
    const uint total_out = C  * N;
    // ulong base offsets: b * total overflows 32 bits on huge inputs.
    device const SCALAR_T* z_b = z + (ulong)b * total_in;
    device const SCALAR_T* r_b = residual + (ulong)b * total_out;
    device SCALAR_T*       o_b = out + (ulong)b * total_out;

    float K = float(z_b[0]);
    float s = 0.0f, sq = 0.0f;
    gn_accumulate_sumsq(z_b, total_in, K, tid, tgs, s, sq);
    gn_reduce_finalize(s, sq, K, total_in, eps, lane, sid, tgs, sh_sum, sh_sqsum, bcast);
    const float mean  = bcast[0];
    const float scale = bcast[1];

    if ((N & 3u) == 0u) {
        device const SCALAR4_T* z4 = (device const SCALAR4_T*)z_b;
        device const SCALAR4_T* r4 = (device const SCALAR4_T*)r_b;
        device SCALAR4_T*       o4 = (device SCALAR4_T*)o_b;
        const uint Nv   = N >> 2;
        const uint nv   = C * Nv;
        const uint boff = C * Nv;   // vector offset of channel c + C
        for (uint i = tid; i < nv; i += tgs) {
            uint  c  = i / Nv;
            float wa = float(nweight[c]);
            float ba = float(nbias[c]);
            float wb = float(nweight[c + C]);
            float bb = float(nbias[c + C]);
            float4 a = (float4(z4[i]) - mean) * scale * wa + ba;
            float4 g = (float4(z4[i + boff]) - mean) * scale * wb + bb;
            float4 sig = 1.0f / (1.0f + exp(-g));
            float ls = float(layer_scale[c]);
            o4[i] = SCALAR4_T(a * sig * ls + float4(r4[i]));
        }
    } else {
        // N % 4 != 0: walk the output channel by channel (see common.metal).
        // In output space the flat index IS idx_a and idx_b sits C*N past it,
        // so the old per-element ``i / N`` and ``i % N`` divides both go away.
        //
        // Unlike the multi-stage twin below, this fallback deliberately stays
        // scalar. This kernel is the most register-hungry in the folder (three
        // data buffers plus layer_scale), and adding the vectorized head/body/
        // tail here costs the N % 4 == 0 fast path above ~4% -- which is the
        // path htdemucs actually takes (N=336), while the single-stage
        // fallback never sees the large odd-N shapes (those have a big enough
        // per-batch count to be routed to the multi-stage path).
        const uint boff = C * N;
        uint c  = 0u;
        uint lo = 0u;
        while (lo < total_out) {
            const uint hi = min(total_out, (c + 1u) * N);
            const float wa = float(nweight[c]);
            const float ba = float(nbias[c]);
            const float wb = float(nweight[c + C]);
            const float bb = float(nbias[c + C]);
            const float ls = float(layer_scale[c]);
            for (uint i = lo + tid; i < hi; i += tgs) {
                float a = (float(z_b[i]) - mean) * scale * wa + ba;
                float g = (float(z_b[i + boff]) - mean) * scale * wb + bb;
                float sig = 1.0f / (1.0f + exp(-g));
                o_b[i] = SCALAR_T(a * sig * ls + float(r_b[i]));
            }
            lo = hi;
            ++c;
        }
    }
}

kernel void apply_norm_glu_ls_resid(
    device SCALAR_T*        out             [[buffer(0)]],
    device const SCALAR_T*  z               [[buffer(1)]],
    device const SCALAR_T*  residual        [[buffer(2)]],
    device const float* meanvar         [[buffer(3)]],
    device const SCALAR_T*  nweight         [[buffer(4)]],
    device const SCALAR_T*  nbias           [[buffer(5)]],
    device const SCALAR_T*  layer_scale     [[buffer(6)]],
    constant uint&      total_in_per_b  [[buffer(7)]],   // 2C * N
    constant uint&      total_out_per_b [[buffer(8)]],   // C  * N
    constant uint&      num_tiles       [[buffer(9)]],
    constant uint&      N               [[buffer(10)]],
    constant uint&      C               [[buffer(11)]],
    uint bt  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    uint b = bt / num_tiles;
    uint t = bt % num_tiles;

    float mean  = meanvar[b * 2 + 0];
    float scale = meanvar[b * 2 + 1];

    device const SCALAR_T* z_b = z        + (ulong)b * total_in_per_b;
    device const SCALAR_T* r_b = residual + (ulong)b * total_out_per_b;
    device SCALAR_T*       o_b = out      + (ulong)b * total_out_per_b;

    if ((N & 3u) == 0u) {
        device const SCALAR4_T* z4 = (device const SCALAR4_T*)z_b;
        device const SCALAR4_T* r4 = (device const SCALAR4_T*)r_b;
        device SCALAR4_T*       o4 = (device SCALAR4_T*)o_b;
        const uint Nv   = N >> 2;
        const uint nv   = total_out_per_b >> 2;
        const uint boff = C * Nv;
        uint start = (uint)((ulong)t * (ulong)nv / (ulong)num_tiles);
        uint end   = (uint)((ulong)(t + 1) * (ulong)nv / (ulong)num_tiles);
        for (uint i = start + tid; i < end; i += tgs) {
            uint  c  = i / Nv;
            float wa = float(nweight[c]);
            float ba = float(nbias[c]);
            float wb = float(nweight[c + C]);
            float bb = float(nbias[c + C]);
            float4 a = (float4(z4[i]) - mean) * scale * wa + ba;
            float4 g = (float4(z4[i + boff]) - mean) * scale * wb + bb;
            float4 sig = 1.0f / (1.0f + exp(-g));
            float ls = float(layer_scale[c]);
            o4[i] = SCALAR4_T(a * sig * ls + float4(r4[i]));
        }
    } else {
        // N % 4 != 0: walk the tile channel by channel (see common.metal). The
        // flat output index IS idx_a and idx_b sits C*N past it, so the old
        // per-element ``i / N`` and ``i % N`` divides both disappear.
        const uint boff = C * N;
        const bool vec_ok = GN_CHANNEL_VECTORIZABLE(total_out_per_b);
        uint start = (uint)((ulong)t * (ulong)total_out_per_b / (ulong)num_tiles);
        uint end   = (uint)((ulong)(t + 1) * (ulong)total_out_per_b / (ulong)num_tiles);
        uint c  = start / N;
        uint lo = start;
        while (lo < end) {
            GN_CHANNEL_BOUNDS(lo, end, N, c, hi, head, vend);
            const float wa = float(nweight[c]);
            const float ba = float(nbias[c]);
            const float wb = float(nweight[c + C]);
            const float bb = float(nbias[c + C]);
            const float ls = float(layer_scale[c]);
            const uint vstart = vec_ok ? head : hi;
            const uint vstop  = vec_ok ? vend : hi;

            for (uint i = lo + tid; i < vstart; i += tgs) {
                float a = (float(z_b[i]) - mean) * scale * wa + ba;
                float g = (float(z_b[i + boff]) - mean) * scale * wb + bb;
                float sig = 1.0f / (1.0f + exp(-g));
                o_b[i] = SCALAR_T(a * sig * ls + float(r_b[i]));
            }
            if (vec_ok) {
                device const SCALAR4_T* z4 = (device const SCALAR4_T*)z_b;
                device const SCALAR4_T* r4 = (device const SCALAR4_T*)r_b;
                device SCALAR4_T*       o4 = (device SCALAR4_T*)o_b;
                const uint vboff = boff >> 2;
                for (uint v = (vstart >> 2) + tid; v < (vstop >> 2); v += tgs) {
                    float4 a = (float4(z4[v]) - mean) * scale * wa + ba;
                    float4 g = (float4(z4[v + vboff]) - mean) * scale * wb + bb;
                    float4 sig = 1.0f / (1.0f + exp(-g));
                    o4[v] = SCALAR4_T(a * sig * ls + float4(r4[v]));
                }
            }
            for (uint i = vstop + tid; i < hi; i += tgs) {
                float a = (float(z_b[i]) - mean) * scale * wa + ba;
                float g = (float(z_b[i + boff]) - mean) * scale * wb + bb;
                float sig = 1.0f / (1.0f + exp(-g));
                o_b[i] = SCALAR_T(a * sig * ls + float(r_b[i]));
            }
            lo = hi;
            ++c;
        }
    }
}
