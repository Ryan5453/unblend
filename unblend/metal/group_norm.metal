// GroupNorm with num_groups=1: single-stage kernels and the reduction
// primitives shared with every other ``apply_*`` kernel in this folder.
//
// ``group_norm_g1`` runs one threadgroup per batch element — best for
// shapes with many batch elements (DConv internals). ``group_norm_g1_chlast``
// is its channel-LAST twin for the transformer's ``MyGroupNorm`` (input
// ``(B, T, C)`` flattened to ``(B, T*C)``; affine index is ``i % C``).
//
// ``partial_reduce`` + ``finalize_meanvar`` are the first two stages of
// the multi-stage path used when a single-stage launch would leave the
// GPU idle (small batch, large per-batch work). Apply kernels in the
// other ``.metal`` files read the (B, 2) ``meanvar`` buffer the finalize
// stage writes. ``apply_norm`` / ``apply_norm_chlast`` are the plain
// (no activation) third stages.
//
// Loads/stores use SCALAR4_T vectors when alignment permits (see
// ``common.metal``). The apply loops need the affine index to be constant
// within each vector: channel-first layouts with N % 4 != 0 use the
// channel walk from ``common.metal``, and channel-last layouts with
// C % 4 != 0 use a scalar loop. The reduction helpers live in
// ``common.metal``, which the Python side prepends to this file before
// compiling.

kernel void group_norm_g1(
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
    threadgroup float sh[GN_SHARED_FLOATS];

    const uint total = C * N;
    // Batch base offsets in ulong: b * total overflows 32 bits past ~4G
    // elements into the buffer.
    device const SCALAR_T* in_b  = in_ + (ulong)b * total;
    device SCALAR_T*       out_b = out + (ulong)b * total;

    const float2 ms = gn_reduce_finalize(
        gn_thread_partial(in_b, total, 0u, total, tid, tgs), total, eps,
        lane, sid, tgs, sh
    );
    const float mean  = ms.x;
    const float scale = ms.y;

    if ((N & 3u) == 0u) {
        device const SCALAR4_T* in4  = (device const SCALAR4_T*)in_b;
        device SCALAR4_T*       out4 = (device SCALAR4_T*)out_b;
        const uint Nv = N >> 2;
        const uint nv = C * Nv;
        for (uint i = tid; i < nv; i += tgs) {
            uint  c  = i / Nv;
            float w  = float(weight[c]);
            float bv = float(bias[c]);
            float4 v = float4(in4[i]);
            out4[i]  = SCALAR4_T((v - mean) * scale * w + bv);
        }
    } else {
        // N % 4 != 0: walk channel by channel so the affine params are hoisted
        // and no per-element ``i / N`` divide is needed (see common.metal).
        const bool vec_ok = GN_CHANNEL_VECTORIZABLE(total);
        uint c  = 0u;
        uint lo = 0u;
        while (lo < total) {
            GN_CHANNEL_BOUNDS(lo, total, N, c, hi, head, vend);
            const float w  = float(weight[c]);
            const float bv = float(bias[c]);
            const uint vstart = vec_ok ? head : hi;
            const uint vstop  = vec_ok ? vend : hi;

            for (uint i = lo + tid; i < vstart; i += tgs) {
                out_b[i] = SCALAR_T((float(in_b[i]) - mean) * scale * w + bv);
            }
            if (vec_ok) {
                device const SCALAR4_T* in4  = (device const SCALAR4_T*)in_b;
                device SCALAR4_T*       out4 = (device SCALAR4_T*)out_b;
                for (uint v = (vstart >> 2) + tid; v < (vstop >> 2); v += tgs) {
                    out4[v] = SCALAR4_T((float4(in4[v]) - mean) * scale * w + bv);
                }
            }
            for (uint i = vstop + tid; i < hi; i += tgs) {
                out_b[i] = SCALAR_T((float(in_b[i]) - mean) * scale * w + bv);
            }
            lo = hi;
            ++c;
        }
    }
}

// Channel-last single stage: input (B, T*C) with the affine broadcast over
// the trailing C axis. When C % 4 == 0 a SCALAR4_T load covers 4 consecutive
// channels, so the affine params vectorize too.
kernel void group_norm_g1_chlast(
    device SCALAR_T*       out      [[buffer(0)]],
    device const SCALAR_T* in_      [[buffer(1)]],
    device const SCALAR_T* weight   [[buffer(2)]],
    device const SCALAR_T* bias     [[buffer(3)]],
    constant uint&     C        [[buffer(4)]],
    constant uint&     total    [[buffer(5)]],   // T * C per batch
    constant float&    eps      [[buffer(6)]],
    uint b    [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sid  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float sh[GN_SHARED_FLOATS];

    device const SCALAR_T* in_b  = in_ + (ulong)b * total;
    device SCALAR_T*       out_b = out + (ulong)b * total;

    const float2 ms = gn_reduce_finalize(
        gn_thread_partial(in_b, total, 0u, total, tid, tgs), total, eps,
        lane, sid, tgs, sh
    );
    const float mean  = ms.x;
    const float scale = ms.y;

    if ((C & 3u) == 0u) {
        // C % 4 == 0 implies total % 4 == 0 (total = T*C), so vector loads
        // stay aligned and each SCALAR4_T spans channels 4k..4k+3.
        device const SCALAR4_T* in4  = (device const SCALAR4_T*)in_b;
        device SCALAR4_T*       out4 = (device SCALAR4_T*)out_b;
        device const SCALAR4_T* w4   = (device const SCALAR4_T*)weight;
        device const SCALAR4_T* b4   = (device const SCALAR4_T*)bias;
        const uint Cv = C >> 2;
        const uint nv = total >> 2;
        for (uint i = tid; i < nv; i += tgs) {
            uint   cv = i % Cv;
            float4 w  = float4(w4[cv]);
            float4 bv = float4(b4[cv]);
            float4 v  = float4(in4[i]);
            out4[i]   = SCALAR4_T((v - mean) * scale * w + bv);
        }
    } else {
        for (uint i = tid; i < total; i += tgs) {
            uint  c  = i % C;
            float w  = float(weight[c]);
            float bv = float(bias[c]);
            float v  = float(in_b[i]);
            out_b[i] = SCALAR_T((v - mean) * scale * w + bv);
        }
    }
}

kernel void partial_reduce(
    device const SCALAR_T*  in_           [[buffer(0)]],
    device float*       scratch       [[buffer(1)]],   // (B, num_tiles, 2)
    constant uint&      total_per_b   [[buffer(2)]],
    constant uint&      num_tiles     [[buffer(3)]],
    uint bt   [[threadgroup_position_in_grid]],        // b * num_tiles + t
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sid  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float sh[GN_SHARED_FLOATS];

    uint b = bt / num_tiles;
    uint t = bt % num_tiles;

    device const SCALAR_T* x_b = in_ + (ulong)b * total_per_b;

    // Each tile writes its own (mean, M2), not K-shifted sums: finalize_meanvar
    // merges the tiles with the same identities tg_merge_moments uses across
    // threads, so the multi-stage statistics match the single-stage ones.
    const uint2 r = gn_tile_bounds(t, num_tiles, total_per_b);
    tg_merge_moments(
        gn_thread_partial(x_b, total_per_b, r.x, r.y, tid, tgs), r.y - r.x,
        false, 0.0f, lane, sid, tgs, sh
    );
    if (tid == 0) {
        scratch[(b * num_tiles + t) * 2 + 0] = sh[GN_BCAST];
        scratch[(b * num_tiles + t) * 2 + 1] = sh[GN_BCAST + 1];
    }
}

kernel void finalize_meanvar(
    device const float* scratch       [[buffer(0)]],   // (B, num_tiles, 2) — per-tile (mean, M2)
    device float*       meanvar       [[buffer(1)]],   // (B, 2) — (mean, rsqrt(var+eps))
    constant uint&      total_per_b   [[buffer(2)]],
    constant uint&      num_tiles     [[buffer(3)]],
    constant float&     eps           [[buffer(4)]],
    uint b    [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sid  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float sh[GN_SHARED_FLOATS];

    // Fold this thread's tiles into one partial shifted by its first tile's
    // mean: a tile of n elements with (mean, M2) contributes
    // s += n (mean - K) and sq += M2 + n (mean - K)^2.
    GnPartial p = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint t = tid; t < num_tiles; t += tgs) {
        const uint2 r = gn_tile_bounds(t, num_tiles, total_per_b);
        const float n    = float(r.y - r.x);
        const float mean = scratch[(b * num_tiles + t) * 2 + 0];
        const float m2   = scratch[(b * num_tiles + t) * 2 + 1];
        if (t == tid) {
            p.K = mean;
        }
        const float d = mean - p.K;
        p.n  += n;
        p.s  += n * d;
        p.sq += m2 + n * d * d;
    }
    const float2 ms = gn_reduce_finalize(p, total_per_b, eps, lane, sid, tgs, sh);
    if (tid == 0) {
        meanvar[b * 2 + 0] = ms.x;
        meanvar[b * 2 + 1] = ms.y;
    }
}

kernel void apply_norm(
    device SCALAR_T*        out          [[buffer(0)]],
    device const SCALAR_T*  in_          [[buffer(1)]],
    device const float* meanvar      [[buffer(2)]],    // (B, 2)
    device const SCALAR_T*  weight       [[buffer(3)]],    // (C,)
    device const SCALAR_T*  bias         [[buffer(4)]],    // (C,)
    constant uint&      total_per_b  [[buffer(5)]],
    constant uint&      num_tiles    [[buffer(6)]],
    constant uint&      N            [[buffer(7)]],    // spatial size
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
            float4 v = float4(in4[i]);
            out4[i]  = SCALAR4_T((v - mean) * scale * w + bv);
        }
    } else {
        // N % 4 != 0: walk the tile channel by channel (see common.metal).
        const bool vec_ok = GN_CHANNEL_VECTORIZABLE(total_per_b);
        uint start = (uint)((ulong)t * (ulong)total_per_b / (ulong)num_tiles);
        uint end   = (uint)((ulong)(t + 1) * (ulong)total_per_b / (ulong)num_tiles);
        uint c  = start / N;
        uint lo = start;
        while (lo < end) {
            GN_CHANNEL_BOUNDS(lo, end, N, c, hi, head, vend);
            const float w  = float(weight[c]);
            const float bv = float(bias[c]);
            const uint vstart = vec_ok ? head : hi;
            const uint vstop  = vec_ok ? vend : hi;

            for (uint i = lo + tid; i < vstart; i += tgs) {
                out_b[i] = SCALAR_T((float(x_b[i]) - mean) * scale * w + bv);
            }
            if (vec_ok) {
                device const SCALAR4_T* in4  = (device const SCALAR4_T*)x_b;
                device SCALAR4_T*       out4 = (device SCALAR4_T*)out_b;
                for (uint v = (vstart >> 2) + tid; v < (vstop >> 2); v += tgs) {
                    out4[v] = SCALAR4_T((float4(in4[v]) - mean) * scale * w + bv);
                }
            }
            for (uint i = vstop + tid; i < hi; i += tgs) {
                out_b[i] = SCALAR_T((float(x_b[i]) - mean) * scale * w + bv);
            }
            lo = hi;
            ++c;
        }
    }
}

// Channel-last multi-stage third stage (transformer MyGroupNorm shapes).
kernel void apply_norm_chlast(
    device SCALAR_T*        out          [[buffer(0)]],
    device const SCALAR_T*  in_          [[buffer(1)]],
    device const float* meanvar      [[buffer(2)]],    // (B, 2)
    device const SCALAR_T*  weight       [[buffer(3)]],    // (C,)
    device const SCALAR_T*  bias         [[buffer(4)]],    // (C,)
    constant uint&      total_per_b  [[buffer(5)]],    // T * C
    constant uint&      num_tiles    [[buffer(6)]],
    constant uint&      C            [[buffer(7)]],
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

    if ((C & 3u) == 0u) {
        device const SCALAR4_T* in4  = (device const SCALAR4_T*)x_b;
        device SCALAR4_T*       out4 = (device SCALAR4_T*)out_b;
        device const SCALAR4_T* w4   = (device const SCALAR4_T*)weight;
        device const SCALAR4_T* b4   = (device const SCALAR4_T*)bias;
        const uint Cv = C >> 2;
        const uint nv = total_per_b >> 2;
        uint start = (uint)((ulong)t * (ulong)nv / (ulong)num_tiles);
        uint end   = (uint)((ulong)(t + 1) * (ulong)nv / (ulong)num_tiles);
        for (uint i = start + tid; i < end; i += tgs) {
            uint   cv = i % Cv;
            float4 w  = float4(w4[cv]);
            float4 bv = float4(b4[cv]);
            float4 v  = float4(in4[i]);
            out4[i]   = SCALAR4_T((v - mean) * scale * w + bv);
        }
    } else {
        uint start = (uint)((ulong)t * (ulong)total_per_b / (ulong)num_tiles);
        uint end   = (uint)((ulong)(t + 1) * (ulong)total_per_b / (ulong)num_tiles);
        for (uint i = start + tid; i < end; i += tgs) {
            uint  c  = i % C;
            float w  = float(weight[c]);
            float bv = float(bias[c]);
            float v  = float(x_b[i]);
            out_b[i] = SCALAR_T((v - mean) * scale * w + bv);
        }
    }
}
