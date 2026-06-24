// GroupNorm with num_groups=1: single-stage and the reduction primitives
// shared with every other ``apply_*`` kernel in this folder.
//
// ``group_norm_g1`` runs one threadgroup per batch element — best for
// shapes with many batch elements and small per-batch work (DConv internals).
//
// ``partial_reduce`` + ``finalize_meanvar`` are the first two stages of
// the multi-stage path used for the outermost encoder/decoder GroupNorms,
// where ``B`` is small but per-batch work is huge. Apply kernels in the
// other ``.metal`` files (``group_norm_gelu``, ``group_norm_glu``,
// ``dconv_envelope``) read the (B, 2) ``meanvar`` buffer this finalize
// stage writes.
//
// ``apply_norm`` is the plain (no activation) third stage. Its mirror
// kernels with fused activations live alongside their respective single-
// stage variants in the other files.
//
// All reductions accumulate in FP32; the low-precision type (half/bfloat)
// only crosses the device-memory boundary at load and store, avoiding the
// implicit cast traffic that makes PyTorch's stock low-precision GroupNorm
// slow on MPS. The same source compiles for both FP16 and BF16 — the Python
// side prepends ``#define SCALAR_T bfloat`` to switch.
// See ``demucs/metal/__init__.py`` for the Python-side wrappers.

#include <metal_stdlib>
using namespace metal;

#ifndef SCALAR_T
#define SCALAR_T half
#endif

kernel void group_norm_g1(
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
    // Batch base offsets in ulong: b * total overflows 32 bits past ~4G
    // elements into the buffer.
    device const SCALAR_T* in_b = in_ + (ulong)b * total;
    device SCALAR_T*       out_b = out + (ulong)b * total;

    // Shift every value by a reference (the batch's first element) before
    // summing. Variance is shift-invariant, but the shifted data is small
    // relative to its mean, so the one-pass ``E[d^2] - E[d]^2`` no longer
    // loses most of its significant digits to catastrophic cancellation when
    // the input has a large DC offset. We add K back to recover the true mean.
    float K = float(in_b[0]);
    float local_sum = 0.0f;
    float local_sqsum = 0.0f;
    for (uint i = tid; i < total; i += tgs) {
        float v = float(in_b[i]) - K;
        local_sum   += v;
        local_sqsum += v * v;
    }
    shared_sum[tid]   = local_sum;
    shared_sqsum[tid] = local_sqsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgs >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid]   += shared_sum[tid + stride];
            shared_sqsum[tid] += shared_sqsum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float bcast_mean;
    threadgroup float bcast_scale;
    if (tid == 0) {
        float invN   = 1.0f / float(total);
        float mean_d = shared_sum[0]   * invN;
        float var    = max(shared_sqsum[0] * invN - mean_d * mean_d, 0.0f);
        bcast_mean  = K + mean_d;
        bcast_scale = rsqrt(var + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean  = bcast_mean;
    float scale = bcast_scale;

    for (uint i = tid; i < total; i += tgs) {
        uint  c_idx = i / N;
        float v     = float(in_b[i]);
        float w     = float(weight[c_idx]);
        float bv    = float(bias[c_idx]);
        float y     = (v - mean) * scale;
        out_b[i]    = SCALAR_T(y * w + bv);
    }
}

kernel void partial_reduce(
    device const SCALAR_T*  in_           [[buffer(0)]],
    device float*       scratch       [[buffer(1)]],   // (B, num_tiles, 2)
    constant uint&      total_per_b   [[buffer(2)]],
    constant uint&      num_tiles     [[buffer(3)]],
    uint bt  [[threadgroup_position_in_grid]],         // b * num_tiles + t
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    constexpr uint MAX_TGS = 1024;
    threadgroup float sh_sum[MAX_TGS];
    threadgroup float sh_sqsum[MAX_TGS];

    uint b = bt / num_tiles;
    uint t = bt % num_tiles;
    // Even tile boundaries — the last tile picks up any remainder.
    uint start = (uint)((ulong)t * (ulong)total_per_b / (ulong)num_tiles);
    uint end   = (uint)((ulong)(t + 1) * (ulong)total_per_b / (ulong)num_tiles);

    device const SCALAR_T* x_b = in_ + (ulong)b * total_per_b;

    // Shift by the batch's first element (shared across all tiles of this
    // batch) so the partial sums feeding the variance don't lose precision to
    // cancellation on large-DC inputs. finalize_meanvar adds K back for the
    // mean; the variance it derives from these shifted sums is unaffected.
    float K = float(x_b[0]);
    float local_sum = 0.0f;
    float local_sqsum = 0.0f;
    for (uint i = start + tid; i < end; i += tgs) {
        float v = float(x_b[i]) - K;
        local_sum   += v;
        local_sqsum += v * v;
    }
    sh_sum[tid]   = local_sum;
    sh_sqsum[tid] = local_sqsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgs >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_sum[tid]   += sh_sum[tid + stride];
            sh_sqsum[tid] += sh_sqsum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        scratch[(b * num_tiles + t) * 2 + 0] = sh_sum[0];
        scratch[(b * num_tiles + t) * 2 + 1] = sh_sqsum[0];
    }
}

kernel void finalize_meanvar(
    device const float* scratch       [[buffer(0)]],   // (B, num_tiles, 2) — shifted (sum_d, sqsum_d)
    device float*       meanvar       [[buffer(1)]],   // (B, 2) — (mean, rsqrt(var+eps))
    constant uint&      total_per_b   [[buffer(2)]],
    constant uint&      num_tiles     [[buffer(3)]],
    constant float&     eps           [[buffer(4)]],
    device const SCALAR_T* in_        [[buffer(5)]],   // input, for the shift reference K
    uint b   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    constexpr uint MAX_TGS = 1024;
    threadgroup float sh_sum[MAX_TGS];
    threadgroup float sh_sqsum[MAX_TGS];

    float local_sum = 0.0f;
    float local_sqsum = 0.0f;
    for (uint t = tid; t < num_tiles; t += tgs) {
        local_sum   += scratch[(b * num_tiles + t) * 2 + 0];
        local_sqsum += scratch[(b * num_tiles + t) * 2 + 1];
    }
    sh_sum[tid]   = local_sum;
    sh_sqsum[tid] = local_sqsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgs >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_sum[tid]   += sh_sum[tid + stride];
            sh_sqsum[tid] += sh_sqsum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        // sh_sum / sh_sqsum are sums of (x - K); recover the true mean by
        // adding K back. The variance is computed from the shifted sums, where
        // cancellation is negligible. K is the same reference partial_reduce
        // used: the batch's first element.
        float K      = float(in_[(ulong)b * total_per_b]);
        float invN   = 1.0f / float(total_per_b);
        float mean_d = sh_sum[0]   * invN;
        float var    = max(sh_sqsum[0] * invN - mean_d * mean_d, 0.0f);
        meanvar[b * 2 + 0] = K + mean_d;
        meanvar[b * 2 + 1] = rsqrt(var + eps);
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
    uint start = (uint)((ulong)t * (ulong)total_per_b / (ulong)num_tiles);
    uint end   = (uint)((ulong)(t + 1) * (ulong)total_per_b / (ulong)num_tiles);

    float mean  = meanvar[b * 2 + 0];
    float scale = meanvar[b * 2 + 1];

    device const SCALAR_T* x_b   = in_ + (ulong)b * total_per_b;
    device SCALAR_T*       out_b = out + (ulong)b * total_per_b;

    for (uint i = start + tid; i < end; i += tgs) {
        uint  c_idx = i / N;
        float v     = float(x_b[i]);
        float w     = float(weight[c_idx]);
        float bv    = float(bias[c_idx]);
        float y     = (v - mean) * scale;
        out_b[i]    = SCALAR_T(y * w + bv);
    }
}
