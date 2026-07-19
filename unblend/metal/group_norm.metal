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
// ``common.metal``); the apply loops additionally need the affine index
// to be constant within each vector, i.e. N % 4 == 0 for channel-first
// and C % 4 == 0 for channel-last, and fall back to scalar loops
// otherwise. The reduction helpers live in ``common.metal``, which the
// Python side prepends to this file before compiling.

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
    threadgroup float sh_sum[MAX_SIMDGROUPS];
    threadgroup float sh_sqsum[MAX_SIMDGROUPS];
    threadgroup float bcast[2];

    const uint total = C * N;
    // Batch base offsets in ulong: b * total overflows 32 bits past ~4G
    // elements into the buffer.
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
            float4 v = float4(in4[i]);
            out4[i]  = SCALAR4_T((v - mean) * scale * w + bv);
        }
    } else {
        for (uint i = tid; i < total; i += tgs) {
            uint  c  = i / N;
            float w  = float(weight[c]);
            float bv = float(bias[c]);
            float v  = float(in_b[i]);
            out_b[i] = SCALAR_T((v - mean) * scale * w + bv);
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
    threadgroup float sh_sum[MAX_SIMDGROUPS];
    threadgroup float sh_sqsum[MAX_SIMDGROUPS];
    threadgroup float bcast[2];

    device const SCALAR_T* in_b  = in_ + (ulong)b * total;
    device SCALAR_T*       out_b = out + (ulong)b * total;

    float K = float(in_b[0]);
    float s = 0.0f, sq = 0.0f;
    gn_accumulate_sumsq(in_b, total, K, tid, tgs, s, sq);
    gn_reduce_finalize(s, sq, K, total, eps, lane, sid, tgs, sh_sum, sh_sqsum, bcast);
    const float mean  = bcast[0];
    const float scale = bcast[1];

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
    threadgroup float sh_sum[MAX_SIMDGROUPS];
    threadgroup float sh_sqsum[MAX_SIMDGROUPS];
    threadgroup float bcast[2];

    uint b = bt / num_tiles;
    uint t = bt % num_tiles;

    device const SCALAR_T* x_b = in_ + (ulong)b * total_per_b;

    // Shift by the batch's first element (shared across all tiles of this
    // batch) so the partial sums feeding the variance don't lose precision to
    // cancellation on large-DC inputs. finalize_meanvar adds K back for the
    // mean; the variance it derives from these shifted sums is unaffected.
    float K = float(x_b[0]);
    float local_sum = 0.0f;
    float local_sqsum = 0.0f;
    if ((total_per_b & 3u) == 0u) {
        // Tile the vector space. The partial sums are position-agnostic, so
        // tiling (total/4) vectors instead of total scalars changes nothing
        // downstream — finalize just sums every tile's partials.
        device const SCALAR4_T* x4 = (device const SCALAR4_T*)x_b;
        const uint nv = total_per_b >> 2;
        uint start = (uint)((ulong)t * (ulong)nv / (ulong)num_tiles);
        uint end   = (uint)((ulong)(t + 1) * (ulong)nv / (ulong)num_tiles);
        for (uint i = start + tid; i < end; i += tgs) {
            float4 v = float4(x4[i]) - K;
            local_sum   += v.x + v.y + v.z + v.w;
            local_sqsum += dot(v, v);
        }
    } else {
        // Even tile boundaries — the last tile picks up any remainder.
        uint start = (uint)((ulong)t * (ulong)total_per_b / (ulong)num_tiles);
        uint end   = (uint)((ulong)(t + 1) * (ulong)total_per_b / (ulong)num_tiles);
        for (uint i = start + tid; i < end; i += tgs) {
            float v = float(x_b[i]) - K;
            local_sum   += v;
            local_sqsum += v * v;
        }
    }
    tg_reduce_sumsq(local_sum, local_sqsum, lane, sid, tgs, sh_sum, sh_sqsum, bcast);
    if (tid == 0) {
        scratch[(b * num_tiles + t) * 2 + 0] = bcast[0];
        scratch[(b * num_tiles + t) * 2 + 1] = bcast[1];
    }
}

kernel void finalize_meanvar(
    device const float* scratch       [[buffer(0)]],   // (B, num_tiles, 2) — shifted (sum_d, sqsum_d)
    device float*       meanvar       [[buffer(1)]],   // (B, 2) — (mean, rsqrt(var+eps))
    constant uint&      total_per_b   [[buffer(2)]],
    constant uint&      num_tiles     [[buffer(3)]],
    constant float&     eps           [[buffer(4)]],
    device const SCALAR_T* in_        [[buffer(5)]],   // input, for the shift reference K
    uint b    [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sid  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float sh_sum[MAX_SIMDGROUPS];
    threadgroup float sh_sqsum[MAX_SIMDGROUPS];
    threadgroup float bcast[2];

    float local_sum = 0.0f;
    float local_sqsum = 0.0f;
    for (uint t = tid; t < num_tiles; t += tgs) {
        local_sum   += scratch[(b * num_tiles + t) * 2 + 0];
        local_sqsum += scratch[(b * num_tiles + t) * 2 + 1];
    }
    // sh_sum / sh_sqsum are sums of (x - K); recover the true mean by adding
    // K back. The variance is computed from the shifted sums, where
    // cancellation is negligible. K is the same reference partial_reduce
    // used: the batch's first element.
    float K = float(in_[(ulong)b * total_per_b]);
    gn_reduce_finalize(
        local_sum, local_sqsum, K, total_per_b, eps,
        lane, sid, tgs, sh_sum, sh_sqsum, bcast
    );
    if (tid == 0) {
        meanvar[b * 2 + 0] = bcast[0];
        meanvar[b * 2 + 1] = bcast[1];
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
        uint start = (uint)((ulong)t * (ulong)total_per_b / (ulong)num_tiles);
        uint end   = (uint)((ulong)(t + 1) * (ulong)total_per_b / (ulong)num_tiles);
        for (uint i = start + tid; i < end; i += tgs) {
            uint  c  = i / N;
            float w  = float(weight[c]);
            float bv = float(bias[c]);
            float v  = float(x_b[i]);
            out_b[i] = SCALAR_T((v - mean) * scale * w + bv);
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
