// DConv envelope kernels: the post-conv2 path inside a DConv layer fused
// down to a single launch (or, for the multi-stage variant, two launches
// over the input followed by one over the output).
//
// ``glu_layerscale_resid``: combines GLU + per-channel LayerScale +
// residual add for an already-normalised input.
//
// ``norm_glu_ls_resid`` (single-stage) and ``apply_norm_glu_ls_resid``
// (multi-stage third stage) absorb GroupNorm into the same fused op:
//   output = residual + layer_scale * glu(group_norm(z))
// which replaces FOUR previously separate kernel launches (group_norm,
// glu, layerscale mul, residual add) with one. Used per DConv sub-layer.
// Same source compiles for FP16 (half) and BF16 (bfloat) — the Python side
// prepends ``#define SCALAR_T bfloat`` to switch.
// See ``demucs/metal/__init__.py`` for the Python-side wrappers.

#include <metal_stdlib>
using namespace metal;

#ifndef SCALAR_T
#define SCALAR_T half
#endif

kernel void glu_layerscale_resid(
    device SCALAR_T*       out         [[buffer(0)]],   // (B, C, N)
    device const SCALAR_T* z           [[buffer(1)]],   // (B, 2C, N)
    device const SCALAR_T* residual    [[buffer(2)]],   // (B, C, N)
    device const SCALAR_T* layer_scale [[buffer(3)]],   // (C,)
    constant uint&     C           [[buffer(4)]],
    constant uint&     N           [[buffer(5)]],
    uint b   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    uint total_z = 2 * C * N;
    uint total_o = C * N;
    device const SCALAR_T* z_b = z + b * total_z;
    device const SCALAR_T* r_b = residual + b * total_o;
    device SCALAR_T*       o_b = out + b * total_o;

    for (uint i = tid; i < total_o; i += tgs) {
        uint c = i / N;
        uint sp = i % N;
        float a = float(z_b[c * N + sp]);
        float b_val = float(z_b[(c + C) * N + sp]);
        float sig = 1.0f / (1.0f + exp(-b_val));
        float ls = float(layer_scale[c]);
        float resid = float(r_b[i]);
        o_b[i] = SCALAR_T(a * sig * ls + resid);
    }
}

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
    uint b   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    constexpr uint MAX_TGS = 1024;
    threadgroup float sh_sum[MAX_TGS];
    threadgroup float sh_sqsum[MAX_TGS];

    const uint C = C2 >> 1;
    const uint total_in  = C2 * N;
    const uint total_out = C  * N;
    device const SCALAR_T* z_b = z + b * total_in;
    device const SCALAR_T* r_b = residual + b * total_out;
    device SCALAR_T*       o_b = out + b * total_out;

    float local_sum = 0.0f, local_sqsum = 0.0f;
    for (uint i = tid; i < total_in; i += tgs) {
        float v = float(z_b[i]);
        local_sum += v;
        local_sqsum += v * v;
    }
    sh_sum[tid] = local_sum;
    sh_sqsum[tid] = local_sqsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_sum[tid]   += sh_sum[tid + stride];
            sh_sqsum[tid] += sh_sqsum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float bcast_mean, bcast_scale;
    if (tid == 0) {
        float invN = 1.0f / float(total_in);
        float mean = sh_sum[0] * invN;
        float meanSq = sh_sqsum[0] * invN;
        float var = max(meanSq - mean * mean, 0.0f);
        bcast_mean = mean;
        bcast_scale = rsqrt(var + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = bcast_mean, scale = bcast_scale;

    for (uint i = tid; i < total_out; i += tgs) {
        uint c  = i / N;
        uint sp = i % N;
        uint idx_a = c          * N + sp;
        uint idx_b = (c + C)    * N + sp;
        float wa = float(nweight[c]);
        float ba = float(nbias[c]);
        float wb = float(nweight[c + C]);
        float bb = float(nbias[c + C]);
        float a = (float(z_b[idx_a]) - mean) * scale * wa + ba;
        float b_val = (float(z_b[idx_b]) - mean) * scale * wb + bb;
        float sig = 1.0f / (1.0f + exp(-b_val));
        float ls = float(layer_scale[c]);
        float resid = float(r_b[i]);
        o_b[i] = SCALAR_T(a * sig * ls + resid);
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
    uint start = (uint)((ulong)t * (ulong)total_out_per_b / (ulong)num_tiles);
    uint end   = (uint)((ulong)(t + 1) * (ulong)total_out_per_b / (ulong)num_tiles);

    float mean  = meanvar[b * 2 + 0];
    float scale = meanvar[b * 2 + 1];

    device const SCALAR_T* z_b   = z        + b * total_in_per_b;
    device const SCALAR_T* r_b   = residual + b * total_out_per_b;
    device SCALAR_T*       o_b   = out      + b * total_out_per_b;

    for (uint i = start + tid; i < end; i += tgs) {
        uint c  = i / N;
        uint sp = i % N;
        uint idx_a = c          * N + sp;
        uint idx_b = (c + C)    * N + sp;
        float wa = float(nweight[c]);
        float ba = float(nbias[c]);
        float wb = float(nweight[c + C]);
        float bb = float(nbias[c + C]);
        float a = (float(z_b[idx_a]) - mean) * scale * wa + ba;
        float b_val = (float(z_b[idx_b]) - mean) * scale * wb + bb;
        float sig = 1.0f / (1.0f + exp(-b_val));
        float ls = float(layer_scale[c]);
        float resid = float(r_b[i]);
        o_b[i] = SCALAR_T(a * sig * ls + resid);
    }
}

// Plain GLU along channel dim (dim=1) for already-normalised inputs.
//
// Input (B, 2C, *), output (B, C, *). Used by FusedHEncLayer/FusedHDecLayer
// when the layer's outer norm is Identity (HTDemucs default norm_starts=4
// with depth=4 means none of the outer encoder/decoder norms ever apply),
// in which case F.glu(rewrite_output, dim=1) is the only post-conv op left.
//
// Multi-threadgroup launch: ``num_tiles`` threadgroups per batch, each
// covering ``total_out / num_tiles`` output elements. Saturates the GPU
// even at the typical inference batch size of 1.
kernel void glu_channelwise(
    device SCALAR_T*       out         [[buffer(0)]],   // (B, C, N)
    device const SCALAR_T* in_         [[buffer(1)]],   // (B, 2C, N)
    constant uint&         C           [[buffer(2)]],   // output channels
    constant uint&         N           [[buffer(3)]],   // product of spatial dims
    constant uint&         num_tiles   [[buffer(4)]],
    uint bt  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    uint b = bt / num_tiles;
    uint t = bt % num_tiles;
    uint total_in  = 2 * C * N;
    uint total_out = C * N;
    uint start = (uint)((ulong)t * (ulong)total_out / (ulong)num_tiles);
    uint end   = (uint)((ulong)(t + 1) * (ulong)total_out / (ulong)num_tiles);

    device const SCALAR_T* in_b  = in_ + b * total_in;
    device SCALAR_T*       out_b = out + b * total_out;

    for (uint i = start + tid; i < end; i += tgs) {
        uint c  = i / N;
        uint sp = i % N;
        float a     = float(in_b[c          * N + sp]);
        float b_val = float(in_b[(c + C)    * N + sp]);
        float sig   = 1.0f / (1.0f + exp(-b_val));
        out_b[i]    = SCALAR_T(a * sig);
    }
}
