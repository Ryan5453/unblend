// Channel-last fused activation kernels for NHWC (channels_last) inference.
//
// Storage layout per batch element: element ``e = j * C + c`` with ``j``
// spanning the spatial dims and ``c`` the channels — i.e. the trailing axis
// is the channel axis. The affine index is therefore ``e % C`` everywhere,
// mirroring the channel-last GroupNorm kernels in ``group_norm.cu``.
//
// Two properties make these cheap:
//   * A 4-wide vector spans four consecutive CHANNELS, so when ``C % 4 == 0``
//     the weight/bias vectors load directly alongside the data vectors
//     (same condition as the plain chlast kernels).
//   * The GLU gate for output channel ``c`` sits at a constant offset
//     ``C_half`` within the same spatial row (``C_half == C2 / 2``), so in
//     vector units the gate vector is exactly ``Cv`` vectors past the value
//     vector.
//
// The reduction stages (``partial_reduce`` / ``finalize_meanvar`` from
// ``group_norm.cu``) are layout-agnostic — they reduce whole flat per-batch
// blocks — so only the single-stage and stage-3 apply kernels live here.
// GELU uses the tanh approximation matching the other files in this folder.

#include "bindings.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "kernels.cuh"

namespace {

__device__ __forceinline__ float gelu_tanh(float y) {
    const float inner = 0.7978845608028654f * (y + 0.044715f * y * y * y);
    return 0.5f * y * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float4 gelu_tanh4(const float4& y) {
    const float ix = 0.7978845608028654f * (y.x + 0.044715f * y.x * y.x * y.x);
    const float iy = 0.7978845608028654f * (y.y + 0.044715f * y.y * y.y * y.y);
    const float iz = 0.7978845608028654f * (y.z + 0.044715f * y.z * y.z * y.z);
    const float iw = 0.7978845608028654f * (y.w + 0.044715f * y.w * y.w * y.w);
    return make_float4(
        0.5f * y.x * (1.0f + tanhf(ix)),
        0.5f * y.y * (1.0f + tanhf(iy)),
        0.5f * y.z * (1.0f + tanhf(iz)),
        0.5f * y.w * (1.0f + tanhf(iw))
    );
}

__device__ __forceinline__ float sigmoid1(float g) {
    return 1.0f / (1.0f + expf(-g));
}

__device__ __forceinline__ float4 sigmoid4(const float4& g) {
    return make_float4(
        sigmoid1(g.x), sigmoid1(g.y), sigmoid1(g.z), sigmoid1(g.w)
    );
}

// ---------------------------------------------------------------------------
// GN + (optional inject add) + GELU
// ---------------------------------------------------------------------------

template <typename SCALAR_T>
__global__ void group_norm_g1_chlast_gelu_kernel(
    SCALAR_T* __restrict__ out,
    const SCALAR_T* __restrict__ in_,
    const SCALAR_T* __restrict__ inject,  // nullable second input
    const SCALAR_T* __restrict__ weight,
    const SCALAR_T* __restrict__ bias,
    unsigned int C,
    unsigned int total,  // X * C per batch
    float eps
) {
    const bool has_inj = inject != nullptr;
    __shared__ float sh_sum[MAX_WARPS];
    __shared__ float sh_sq[MAX_WARPS];
    __shared__ float bcast[2];

    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int b = blockIdx.x;

    const SCALAR_T* __restrict__ in_b = in_ + (unsigned long long)b * total;
    const SCALAR_T* __restrict__ j_b =
        has_inj ? inject + (unsigned long long)b * total : nullptr;
    SCALAR_T* __restrict__ out_b = out + (unsigned long long)b * total;

    const unsigned long long jbase =
        has_inj ? (unsigned long long)b * total : 0ull;
    float K = static_cast<float>(in_b[0]);
    if (has_inj) {
        K += static_cast<float>(inject[jbase]);
    }
    float s = 0.0f, sq = 0.0f;
    gn_accumulate_sumsq(in_b, j_b, total, K, tid, tgs, s, sq);
    gn_reduce_finalize(s, sq, K, total, eps, sh_sum, sh_sq, bcast);
    const float mean = bcast[0];
    const float scale = bcast[1];

    if ((C & 3u) == 0u) {
        const Scalar4<SCALAR_T>* __restrict__ in4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(in_b);
        const Scalar4<SCALAR_T>* __restrict__ j4 =
            has_inj ? reinterpret_cast<const Scalar4<SCALAR_T>*>(j_b) : nullptr;
        const Scalar4<SCALAR_T>* __restrict__ w4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(weight);
        const Scalar4<SCALAR_T>* __restrict__ b4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(bias);
        Scalar4<SCALAR_T>* __restrict__ out4 =
            reinterpret_cast<Scalar4<SCALAR_T>*>(out_b);
        const unsigned int Cv = C >> 2;
        const unsigned int nv = total >> 2;
        for (unsigned int iv = tid; iv < nv; iv += tgs) {
            const unsigned int cv = iv % Cv;
            const float4 w = unpack4(w4[cv]);
            const float4 bv = unpack4(b4[cv]);
            float4 v = unpack4(in4[iv]);
            if (has_inj) {
                const float4 wj = unpack4(j4[iv]);
                v.x += wj.x; v.y += wj.y; v.z += wj.z; v.w += wj.w;
            }
            const float4 r = make_float4(
                (v.x - mean) * scale * w.x + bv.x,
                (v.y - mean) * scale * w.y + bv.y,
                (v.z - mean) * scale * w.z + bv.z,
                (v.w - mean) * scale * w.w + bv.w
            );
            out4[iv] = pack4<SCALAR_T>(gelu_tanh4(r));
        }
    } else {
        for (unsigned int i = tid; i < total; i += tgs) {
            const unsigned int c = i % C;
            const float w = static_cast<float>(weight[c]);
            const float bv = static_cast<float>(bias[c]);
            float v = static_cast<float>(in_b[i]);
            if (has_inj) {
                v += static_cast<float>(j_b[i]);
            }
            const float y = (v - mean) * scale * w + bv;
            out_b[i] = static_cast<SCALAR_T>(gelu_tanh(y));
        }
    }
}

template <typename SCALAR_T>
__global__ void apply_norm_chlast_gelu_kernel(
    SCALAR_T* __restrict__ out,
    const SCALAR_T* __restrict__ in_,
    const SCALAR_T* __restrict__ inject,  // nullable second input
    const float* __restrict__ meanvar,
    const SCALAR_T* __restrict__ weight,
    const SCALAR_T* __restrict__ bias,
    unsigned int total_per_b,
    unsigned int num_tiles,
    unsigned int C
) {
    const bool has_inj = inject != nullptr;
    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int bt = blockIdx.x;
    const unsigned int b = bt / num_tiles;
    const unsigned int t = bt % num_tiles;

    const float mean = meanvar[(unsigned long long)b * 2 + 0];
    const float scale = meanvar[(unsigned long long)b * 2 + 1];

    const SCALAR_T* __restrict__ x_b = in_ + (unsigned long long)b * total_per_b;
    const SCALAR_T* __restrict__ j_b =
        has_inj ? inject + (unsigned long long)b * total_per_b : nullptr;
    SCALAR_T* __restrict__ out_b = out + (unsigned long long)b * total_per_b;

    if ((C & 3u) == 0u) {
        const Scalar4<SCALAR_T>* __restrict__ in4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(x_b);
        const Scalar4<SCALAR_T>* __restrict__ j4 =
            has_inj ? reinterpret_cast<const Scalar4<SCALAR_T>*>(j_b) : nullptr;
        const Scalar4<SCALAR_T>* __restrict__ w4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(weight);
        const Scalar4<SCALAR_T>* __restrict__ b4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(bias);
        Scalar4<SCALAR_T>* __restrict__ out4 =
            reinterpret_cast<Scalar4<SCALAR_T>*>(out_b);
        const unsigned int Cv = C >> 2;
        const unsigned int nv = total_per_b >> 2;
        const unsigned int start =
            (unsigned int)((unsigned long long)t * nv / num_tiles);
        const unsigned int end =
            (unsigned int)((unsigned long long)(t + 1) * nv / num_tiles);
        for (unsigned int iv = start + tid; iv < end; iv += tgs) {
            const unsigned int cv = iv % Cv;
            const float4 w = unpack4(w4[cv]);
            const float4 bv = unpack4(b4[cv]);
            float4 v = unpack4(in4[iv]);
            if (has_inj) {
                const float4 wj = unpack4(j4[iv]);
                v.x += wj.x; v.y += wj.y; v.z += wj.z; v.w += wj.w;
            }
            const float4 r = make_float4(
                (v.x - mean) * scale * w.x + bv.x,
                (v.y - mean) * scale * w.y + bv.y,
                (v.z - mean) * scale * w.z + bv.z,
                (v.w - mean) * scale * w.w + bv.w
            );
            out4[iv] = pack4<SCALAR_T>(gelu_tanh4(r));
        }
    } else {
        const unsigned int start =
            (unsigned int)((unsigned long long)t * total_per_b / num_tiles);
        const unsigned int end =
            (unsigned int)((unsigned long long)(t + 1) * total_per_b / num_tiles);
        for (unsigned int i = start + tid; i < end; i += tgs) {
            const unsigned int c = i % C;
            const float w = static_cast<float>(weight[c]);
            const float bv = static_cast<float>(bias[c]);
            float v = static_cast<float>(x_b[i]);
            if (has_inj) {
                v += static_cast<float>(j_b[i]);
            }
            const float y = (v - mean) * scale * w + bv;
            out_b[i] = static_cast<SCALAR_T>(gelu_tanh(y));
        }
    }
}

// ---------------------------------------------------------------------------
// GN + GLU (channel halving)
//
// Input has ``C2 = 2 * C`` channels; output has ``C``. Output element
// ``(j, c)`` reads value ``(j, c)`` and gate ``(j, c + C_half)`` from the
// input row — a constant ``C_half`` scalar offset (``Cv`` vectors).
// ---------------------------------------------------------------------------

template <typename SCALAR_T>
__global__ void group_norm_g1_chlast_glu_kernel(
    SCALAR_T* __restrict__ out,       // (B, X * C)
    const SCALAR_T* __restrict__ in_, // (B, X * C2)
    const SCALAR_T* __restrict__ weight,
    const SCALAR_T* __restrict__ bias,
    unsigned int C2,                  // input channel count (even)
    unsigned int X,                   // spatial size
    float eps
) {
    __shared__ float sh_sum[MAX_WARPS];
    __shared__ float sh_sq[MAX_WARPS];
    __shared__ float bcast[2];

    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int b = blockIdx.x;

    const unsigned int C = C2 >> 1;
    const unsigned int total_in = X * C2;
    const unsigned int total_out = X * C;
    const SCALAR_T* __restrict__ in_b = in_ + (unsigned long long)b * total_in;
    SCALAR_T* __restrict__ out_b = out + (unsigned long long)b * total_out;

    float K = static_cast<float>(in_b[0]);
    float s = 0.0f, sq = 0.0f;
    gn_accumulate_sumsq(in_b, total_in, K, tid, tgs, s, sq);
    gn_reduce_finalize(s, sq, K, total_in, eps, sh_sum, sh_sq, bcast);
    const float mean = bcast[0];
    const float scale = bcast[1];

    if ((C & 3u) == 0u) {
        const Scalar4<SCALAR_T>* __restrict__ in4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(in_b);
        const Scalar4<SCALAR_T>* __restrict__ w4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(weight);
        const Scalar4<SCALAR_T>* __restrict__ b4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(bias);
        Scalar4<SCALAR_T>* __restrict__ out4 =
            reinterpret_cast<Scalar4<SCALAR_T>*>(out_b);
        const unsigned int Cv = C >> 2;
        const unsigned int nv = total_out >> 2;
        const unsigned int boff = Cv;  // vector offset of the gate half-row
        for (unsigned int iv = tid; iv < nv; iv += tgs) {
            const unsigned int cv = iv % Cv;
            const unsigned int j = iv / Cv;
            const float4 wa = unpack4(w4[cv]);
            const float4 ba = unpack4(b4[cv]);
            const float4 wb = unpack4(w4[cv + Cv]);
            const float4 bb = unpack4(b4[cv + Cv]);
            const unsigned long long base =
                (unsigned long long)j * (C2 >> 2);
            const float4 va = unpack4(in4[base + cv]);
            const float4 vg = unpack4(in4[base + cv + boff]);
            const float4 a = make_float4(
                (va.x - mean) * scale * wa.x + ba.x,
                (va.y - mean) * scale * wa.y + ba.y,
                (va.z - mean) * scale * wa.z + ba.z,
                (va.w - mean) * scale * wa.w + ba.w
            );
            const float4 g = make_float4(
                (vg.x - mean) * scale * wb.x + bb.x,
                (vg.y - mean) * scale * wb.y + bb.y,
                (vg.z - mean) * scale * wb.z + bb.z,
                (vg.w - mean) * scale * wb.w + bb.w
            );
            const float4 sig = sigmoid4(g);
            out4[iv] = pack4<SCALAR_T>(
                make_float4(a.x * sig.x, a.y * sig.y, a.z * sig.z, a.w * sig.w)
            );
        }
    } else {
        for (unsigned int e = tid; e < total_out; e += tgs) {
            const unsigned int j = e / C;
            const unsigned int c = e % C;
            const unsigned long long a_idx =
                (unsigned long long)j * C2 + c;
            const unsigned long long g_idx = a_idx + C;
            const float wa = static_cast<float>(weight[c]);
            const float ba = static_cast<float>(bias[c]);
            const float wb = static_cast<float>(weight[c + C]);
            const float bb = static_cast<float>(bias[c + C]);
            const float a =
                (static_cast<float>(in_b[a_idx]) - mean) * scale * wa + ba;
            const float g =
                (static_cast<float>(in_b[g_idx]) - mean) * scale * wb + bb;
            out_b[e] = static_cast<SCALAR_T>(a * sigmoid1(g));
        }
    }
}

template <typename SCALAR_T>
__global__ void apply_norm_chlast_glu_kernel(
    SCALAR_T* __restrict__ out,       // (B, X * C)
    const SCALAR_T* __restrict__ in_, // (B, X * C2)
    const float* __restrict__ meanvar,
    const SCALAR_T* __restrict__ weight,
    const SCALAR_T* __restrict__ bias,
    unsigned int total_in_per_b,   // X * C2
    unsigned int total_out_per_b,  // X * C
    unsigned int num_tiles,
    unsigned int C                 // output channel count (C2 = 2 * C)
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int bt = blockIdx.x;
    const unsigned int b = bt / num_tiles;
    const unsigned int t = bt % num_tiles;

    const float mean = meanvar[(unsigned long long)b * 2 + 0];
    const float scale = meanvar[(unsigned long long)b * 2 + 1];

    const SCALAR_T* __restrict__ x_b = in_ + (unsigned long long)b * total_in_per_b;
    SCALAR_T* __restrict__ out_b = out + (unsigned long long)b * total_out_per_b;
    const unsigned int C_half = C;

    if ((C & 3u) == 0u) {
        const Scalar4<SCALAR_T>* __restrict__ in4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(x_b);
        const Scalar4<SCALAR_T>* __restrict__ w4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(weight);
        const Scalar4<SCALAR_T>* __restrict__ b4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(bias);
        Scalar4<SCALAR_T>* __restrict__ out4 =
            reinterpret_cast<Scalar4<SCALAR_T>*>(out_b);
        const unsigned int Cv = C >> 2;
        const unsigned int nv = total_out_per_b >> 2;
        const unsigned int boff = Cv;
        const unsigned int start =
            (unsigned int)((unsigned long long)t * nv / num_tiles);
        const unsigned int end =
            (unsigned int)((unsigned long long)(t + 1) * nv / num_tiles);
        for (unsigned int iv = start + tid; iv < end; iv += tgs) {
            const unsigned int cv = iv % Cv;
            const unsigned int j = iv / Cv;
            const float4 wa = unpack4(w4[cv]);
            const float4 ba = unpack4(b4[cv]);
            const float4 wb = unpack4(w4[cv + Cv]);
            const float4 bb = unpack4(b4[cv + Cv]);
            const unsigned long long base =
                (unsigned long long)j * ((2 * C) >> 2);
            const float4 va = unpack4(in4[base + cv]);
            const float4 vg = unpack4(in4[base + cv + boff]);
            const float4 a = make_float4(
                (va.x - mean) * scale * wa.x + ba.x,
                (va.y - mean) * scale * wa.y + ba.y,
                (va.z - mean) * scale * wa.z + ba.z,
                (va.w - mean) * scale * wa.w + ba.w
            );
            const float4 g = make_float4(
                (vg.x - mean) * scale * wb.x + bb.x,
                (vg.y - mean) * scale * wb.y + bb.y,
                (vg.z - mean) * scale * wb.z + bb.z,
                (vg.w - mean) * scale * wb.w + bb.w
            );
            const float4 sig = sigmoid4(g);
            out4[iv] = pack4<SCALAR_T>(
                make_float4(a.x * sig.x, a.y * sig.y, a.z * sig.z, a.w * sig.w)
            );
        }
    } else {
        const unsigned int start =
            (unsigned int)((unsigned long long)t * total_out_per_b / num_tiles);
        const unsigned int end =
            (unsigned int)((unsigned long long)(t + 1) * total_out_per_b / num_tiles);
        for (unsigned int e = start + tid; e < end; e += tgs) {
            const unsigned int j = e / C;
            const unsigned int c = e % C;
            const unsigned long long a_idx =
                (unsigned long long)j * (2 * C) + c;
            const unsigned long long g_idx = a_idx + C_half;
            const float wa = static_cast<float>(weight[c]);
            const float ba = static_cast<float>(bias[c]);
            const float wb = static_cast<float>(weight[c + C_half]);
            const float bb = static_cast<float>(bias[c + C_half]);
            const float a =
                (static_cast<float>(x_b[a_idx]) - mean) * scale * wa + ba;
            const float g =
                (static_cast<float>(x_b[g_idx]) - mean) * scale * wb + bb;
            out_b[e] = static_cast<SCALAR_T>(a * sigmoid1(g));
        }
    }
}

// ---------------------------------------------------------------------------
// DConv envelope: residual + layer_scale * glu(group_norm(z))
// ---------------------------------------------------------------------------

template <typename SCALAR_T>
__global__ void norm_glu_ls_resid_chlast_kernel(
    SCALAR_T* __restrict__ out,       // (B, X * C)
    const SCALAR_T* __restrict__ z,   // (B, X * C2)
    const SCALAR_T* __restrict__ resid,  // (B, X * C)
    const SCALAR_T* __restrict__ nweight,  // (C2,)
    const SCALAR_T* __restrict__ nbias,    // (C2,)
    const SCALAR_T* __restrict__ layer_scale,  // (C,)
    unsigned int C2,
    unsigned int X,
    float eps
) {
    __shared__ float sh_sum[MAX_WARPS];
    __shared__ float sh_sq[MAX_WARPS];
    __shared__ float bcast[2];

    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int b = blockIdx.x;

    const unsigned int C = C2 >> 1;
    const unsigned int total_in = X * C2;
    const unsigned int total_out = X * C;
    const SCALAR_T* __restrict__ z_b = z + (unsigned long long)b * total_in;
    const SCALAR_T* __restrict__ r_b = resid + (unsigned long long)b * total_out;
    SCALAR_T* __restrict__ o_b = out + (unsigned long long)b * total_out;

    float K = static_cast<float>(z_b[0]);
    float s = 0.0f, sq = 0.0f;
    gn_accumulate_sumsq(z_b, total_in, K, tid, tgs, s, sq);
    gn_reduce_finalize(s, sq, K, total_in, eps, sh_sum, sh_sq, bcast);
    const float mean = bcast[0];
    const float scale = bcast[1];

    if ((C & 3u) == 0u) {
        const Scalar4<SCALAR_T>* __restrict__ z4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(z_b);
        const Scalar4<SCALAR_T>* __restrict__ r4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(r_b);
        const Scalar4<SCALAR_T>* __restrict__ nw4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(nweight);
        const Scalar4<SCALAR_T>* __restrict__ nb4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(nbias);
        Scalar4<SCALAR_T>* __restrict__ o4 =
            reinterpret_cast<Scalar4<SCALAR_T>*>(o_b);
        const unsigned int Cv = C >> 2;
        const unsigned int nv = total_out >> 2;
        const unsigned int boff = Cv;
        for (unsigned int iv = tid; iv < nv; iv += tgs) {
            const unsigned int cv = iv % Cv;
            const unsigned int j = iv / Cv;
            const float4 wa = unpack4(nw4[cv]);
            const float4 ba = unpack4(nb4[cv]);
            const float4 wb = unpack4(nw4[cv + Cv]);
            const float4 bb = unpack4(nb4[cv + Cv]);
            const unsigned long long base =
                (unsigned long long)j * (C2 >> 2);
            const float4 vz = unpack4(z4[base + cv]);
            const float4 vg = unpack4(z4[base + cv + boff]);
            const float4 a = make_float4(
                (vz.x - mean) * scale * wa.x + ba.x,
                (vz.y - mean) * scale * wa.y + ba.y,
                (vz.z - mean) * scale * wa.z + ba.z,
                (vz.w - mean) * scale * wa.w + ba.w
            );
            const float4 g = make_float4(
                (vg.x - mean) * scale * wb.x + bb.x,
                (vg.y - mean) * scale * wb.y + bb.y,
                (vg.z - mean) * scale * wb.z + bb.z,
                (vg.w - mean) * scale * wb.w + bb.w
            );
            const float4 sig = sigmoid4(g);
            // Channel-last: each vector lane is a DIFFERENT channel, so the
            // per-channel LayerScale must be applied per lane.
            const unsigned int c0 = cv * 4u;
            const float4 ls = make_float4(
                static_cast<float>(layer_scale[c0]),
                static_cast<float>(layer_scale[c0 + 1]),
                static_cast<float>(layer_scale[c0 + 2]),
                static_cast<float>(layer_scale[c0 + 3])
            );
            const float4 vr = unpack4(r4[iv]);
            o4[iv] = pack4<SCALAR_T>(make_float4(
                a.x * sig.x * ls.x + vr.x,
                a.y * sig.y * ls.y + vr.y,
                a.z * sig.z * ls.z + vr.z,
                a.w * sig.w * ls.w + vr.w
            ));
        }
    } else {
        for (unsigned int e = tid; e < total_out; e += tgs) {
            const unsigned int j = e / C;
            const unsigned int c = e % C;
            const unsigned long long a_idx =
                (unsigned long long)j * C2 + c;
            const unsigned long long g_idx = a_idx + C;
            const float wa = static_cast<float>(nweight[c]);
            const float ba = static_cast<float>(nbias[c]);
            const float wb = static_cast<float>(nweight[c + C]);
            const float bb = static_cast<float>(nbias[c + C]);
            const float a =
                (static_cast<float>(z_b[a_idx]) - mean) * scale * wa + ba;
            const float g =
                (static_cast<float>(z_b[g_idx]) - mean) * scale * wb + bb;
            const float ls = static_cast<float>(layer_scale[c]);
            o_b[e] = static_cast<SCALAR_T>(
                a * sigmoid1(g) * ls + static_cast<float>(r_b[e])
            );
        }
    }
}

template <typename SCALAR_T>
__global__ void apply_norm_glu_ls_resid_chlast_kernel(
    SCALAR_T* __restrict__ out,
    const SCALAR_T* __restrict__ z,
    const SCALAR_T* __restrict__ resid,
    const float* __restrict__ meanvar,
    const SCALAR_T* __restrict__ nweight,
    const SCALAR_T* __restrict__ nbias,
    const SCALAR_T* __restrict__ layer_scale,
    unsigned int total_in_per_b,
    unsigned int total_out_per_b,
    unsigned int num_tiles,
    unsigned int C
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int bt = blockIdx.x;
    const unsigned int b = bt / num_tiles;
    const unsigned int t = bt % num_tiles;

    const float mean = meanvar[(unsigned long long)b * 2 + 0];
    const float scale = meanvar[(unsigned long long)b * 2 + 1];

    const SCALAR_T* __restrict__ z_b = z + (unsigned long long)b * total_in_per_b;
    const SCALAR_T* __restrict__ r_b =
        resid + (unsigned long long)b * total_out_per_b;
    SCALAR_T* __restrict__ o_b = out + (unsigned long long)b * total_out_per_b;

    if ((C & 3u) == 0u) {
        const Scalar4<SCALAR_T>* __restrict__ z4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(z_b);
        const Scalar4<SCALAR_T>* __restrict__ r4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(r_b);
        const Scalar4<SCALAR_T>* __restrict__ nw4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(nweight);
        const Scalar4<SCALAR_T>* __restrict__ nb4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(nbias);
        Scalar4<SCALAR_T>* __restrict__ o4 =
            reinterpret_cast<Scalar4<SCALAR_T>*>(o_b);
        const unsigned int Cv = C >> 2;
        const unsigned int nv = total_out_per_b >> 2;
        const unsigned int boff = Cv;
        const unsigned int start =
            (unsigned int)((unsigned long long)t * nv / num_tiles);
        const unsigned int end =
            (unsigned int)((unsigned long long)(t + 1) * nv / num_tiles);
        for (unsigned int iv = start + tid; iv < end; iv += tgs) {
            const unsigned int cv = iv % Cv;
            const unsigned int j = iv / Cv;
            const float4 wa = unpack4(nw4[cv]);
            const float4 ba = unpack4(nb4[cv]);
            const float4 wb = unpack4(nw4[cv + Cv]);
            const float4 bb = unpack4(nb4[cv + Cv]);
            const unsigned long long base =
                (unsigned long long)j * ((2 * C) >> 2);
            const float4 vz = unpack4(z4[base + cv]);
            const float4 vg = unpack4(z4[base + cv + boff]);
            const float4 a = make_float4(
                (vz.x - mean) * scale * wa.x + ba.x,
                (vz.y - mean) * scale * wa.y + ba.y,
                (vz.z - mean) * scale * wa.z + ba.z,
                (vz.w - mean) * scale * wa.w + ba.w
            );
            const float4 g = make_float4(
                (vg.x - mean) * scale * wb.x + bb.x,
                (vg.y - mean) * scale * wb.y + bb.y,
                (vg.z - mean) * scale * wb.z + bb.z,
                (vg.w - mean) * scale * wb.w + bb.w
            );
            const float4 sig = sigmoid4(g);
            // Channel-last: each vector lane is a DIFFERENT channel, so the
            // per-channel LayerScale must be applied per lane.
            const unsigned int c0 = cv * 4u;
            const float4 ls = make_float4(
                static_cast<float>(layer_scale[c0]),
                static_cast<float>(layer_scale[c0 + 1]),
                static_cast<float>(layer_scale[c0 + 2]),
                static_cast<float>(layer_scale[c0 + 3])
            );
            const float4 vr = unpack4(r4[iv]);
            o4[iv] = pack4<SCALAR_T>(make_float4(
                a.x * sig.x * ls.x + vr.x,
                a.y * sig.y * ls.y + vr.y,
                a.z * sig.z * ls.z + vr.z,
                a.w * sig.w * ls.w + vr.w
            ));
        }
    } else {
        const unsigned int start =
            (unsigned int)((unsigned long long)t * total_out_per_b / num_tiles);
        const unsigned int end =
            (unsigned int)((unsigned long long)(t + 1) * total_out_per_b / num_tiles);
        for (unsigned int e = start + tid; e < end; e += tgs) {
            const unsigned int j = e / C;
            const unsigned int c = e % C;
            const unsigned long long a_idx =
                (unsigned long long)j * (2 * C) + c;
            const unsigned long long g_idx = a_idx + C;
            const float wa = static_cast<float>(nweight[c]);
            const float ba = static_cast<float>(nbias[c]);
            const float wb = static_cast<float>(nweight[c + C]);
            const float bb = static_cast<float>(nbias[c + C]);
            const float a =
                (static_cast<float>(z_b[a_idx]) - mean) * scale * wa + ba;
            const float g =
                (static_cast<float>(z_b[g_idx]) - mean) * scale * wb + bb;
            const float ls = static_cast<float>(layer_scale[c]);
            o_b[e] = static_cast<SCALAR_T>(
                a * sigmoid1(g) * ls + static_cast<float>(r_b[e])
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

template <typename SCALAR_T>
void group_norm_g1_chlast_gelu_impl(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& inject,
    const at::Tensor& weight, const at::Tensor& bias, int64_t C, int64_t total,
    double eps, int64_t tgs
) {
    const dim3 grid((unsigned int)in_.size(0));
    const dim3 block((unsigned int)tgs);
    group_norm_g1_chlast_gelu_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<SCALAR_T>(), in_.const_data_ptr<SCALAR_T>(),
        inject.numel() > 0 ? inject.const_data_ptr<SCALAR_T>() : nullptr,
        weight.const_data_ptr<SCALAR_T>(), bias.const_data_ptr<SCALAR_T>(),
        (unsigned int)C, (unsigned int)total, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SCALAR_T>
void apply_norm_chlast_gelu_impl(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& inject,
    const at::Tensor& meanvar, const at::Tensor& weight, const at::Tensor& bias,
    int64_t total_per_b, int64_t num_tiles, int64_t C, int64_t tgs
) {
    const dim3 grid((unsigned int)(in_.size(0) * num_tiles));
    const dim3 block((unsigned int)tgs);
    apply_norm_chlast_gelu_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<SCALAR_T>(), in_.const_data_ptr<SCALAR_T>(),
        inject.numel() > 0 ? inject.const_data_ptr<SCALAR_T>() : nullptr,
        meanvar.const_data_ptr<float>(), weight.const_data_ptr<SCALAR_T>(),
        bias.const_data_ptr<SCALAR_T>(), (unsigned int)total_per_b,
        (unsigned int)num_tiles, (unsigned int)C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SCALAR_T>
void group_norm_g1_chlast_glu_impl(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& weight,
    const at::Tensor& bias, int64_t C2, int64_t X, double eps, int64_t tgs
) {
    const dim3 grid((unsigned int)in_.size(0));
    const dim3 block((unsigned int)tgs);
    group_norm_g1_chlast_glu_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<SCALAR_T>(), in_.const_data_ptr<SCALAR_T>(),
        weight.const_data_ptr<SCALAR_T>(), bias.const_data_ptr<SCALAR_T>(),
        (unsigned int)C2, (unsigned int)X, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SCALAR_T>
void apply_norm_chlast_glu_impl(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& meanvar,
    const at::Tensor& weight, const at::Tensor& bias, int64_t total_in_per_b,
    int64_t total_out_per_b, int64_t num_tiles, int64_t C, int64_t tgs
) {
    const dim3 grid((unsigned int)(in_.size(0) * num_tiles));
    const dim3 block((unsigned int)tgs);
    apply_norm_chlast_glu_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<SCALAR_T>(), in_.const_data_ptr<SCALAR_T>(),
        meanvar.const_data_ptr<float>(), weight.const_data_ptr<SCALAR_T>(),
        bias.const_data_ptr<SCALAR_T>(), (unsigned int)total_in_per_b,
        (unsigned int)total_out_per_b, (unsigned int)num_tiles,
        (unsigned int)C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SCALAR_T>
void norm_glu_ls_resid_chlast_impl(
    const at::Tensor& out, const at::Tensor& z, const at::Tensor& resid,
    const at::Tensor& nweight, const at::Tensor& nbias,
    const at::Tensor& layer_scale, int64_t C2, int64_t X, double eps,
    int64_t tgs
) {
    const dim3 grid((unsigned int)z.size(0));
    const dim3 block((unsigned int)tgs);
    norm_glu_ls_resid_chlast_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<SCALAR_T>(), z.const_data_ptr<SCALAR_T>(),
        resid.const_data_ptr<SCALAR_T>(), nweight.const_data_ptr<SCALAR_T>(),
        nbias.const_data_ptr<SCALAR_T>(),
        layer_scale.const_data_ptr<SCALAR_T>(), (unsigned int)C2,
        (unsigned int)X, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SCALAR_T>
void apply_norm_glu_ls_resid_chlast_impl(
    const at::Tensor& out, const at::Tensor& z, const at::Tensor& resid,
    const at::Tensor& meanvar, const at::Tensor& nweight,
    const at::Tensor& nbias, const at::Tensor& layer_scale,
    int64_t total_in_per_b, int64_t total_out_per_b, int64_t num_tiles,
    int64_t C, int64_t tgs
) {
    const dim3 grid((unsigned int)(z.size(0) * num_tiles));
    const dim3 block((unsigned int)tgs);
    apply_norm_glu_ls_resid_chlast_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<SCALAR_T>(), z.const_data_ptr<SCALAR_T>(),
        resid.const_data_ptr<SCALAR_T>(), meanvar.const_data_ptr<float>(),
        nweight.const_data_ptr<SCALAR_T>(), nbias.const_data_ptr<SCALAR_T>(),
        layer_scale.const_data_ptr<SCALAR_T>(), (unsigned int)total_in_per_b,
        (unsigned int)total_out_per_b, (unsigned int)num_tiles,
        (unsigned int)C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void group_norm_g1_chlast_gelu(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& inject,
    const at::Tensor& weight, const at::Tensor& bias, int64_t C, int64_t total,
    double eps, int64_t tgs
) {
    UNBLEND_CHECKS(group_norm_g1_chlast_gelu)
    UNBLEND_DISPATCH(group_norm_g1_chlast_gelu_impl, in_, out, in_, inject, weight, bias, C, total, eps, tgs)
}

void apply_norm_chlast_gelu(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& inject,
    const at::Tensor& meanvar, const at::Tensor& weight, const at::Tensor& bias,
    int64_t total_per_b, int64_t num_tiles, int64_t C, int64_t tgs
) {
    UNBLEND_CHECKS(apply_norm_chlast_gelu)
    UNBLEND_DISPATCH(apply_norm_chlast_gelu_impl, in_, out, in_, inject, meanvar, weight, bias, total_per_b, num_tiles, C, tgs)
}

void group_norm_g1_chlast_glu(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& weight,
    const at::Tensor& bias, int64_t C2, int64_t X, double eps, int64_t tgs
) {
    UNBLEND_CHECKS(group_norm_g1_chlast_glu)
    UNBLEND_DISPATCH(group_norm_g1_chlast_glu_impl, in_, out, in_, weight, bias, C2, X, eps, tgs)
}

void apply_norm_chlast_glu(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& meanvar,
    const at::Tensor& weight, const at::Tensor& bias, int64_t total_in_per_b,
    int64_t total_out_per_b, int64_t num_tiles, int64_t C, int64_t tgs
) {
    UNBLEND_CHECKS(apply_norm_chlast_glu)
    UNBLEND_DISPATCH(apply_norm_chlast_glu_impl, in_, out, in_, meanvar, weight, bias, total_in_per_b, total_out_per_b, num_tiles, C, tgs)
}

void norm_glu_ls_resid_chlast(
    const at::Tensor& out, const at::Tensor& z, const at::Tensor& resid,
    const at::Tensor& nweight, const at::Tensor& nbias,
    const at::Tensor& layer_scale, int64_t C2, int64_t X, double eps,
    int64_t tgs
) {
    TORCH_CHECK(z.is_cuda() && out.is_cuda() && resid.is_cuda(),
                "norm_glu_ls_resid_chlast: tensors must be CUDA");
    TORCH_CHECK(
        z.scalar_type() == out.scalar_type() &&
            resid.scalar_type() == z.scalar_type() &&
            nweight.scalar_type() == z.scalar_type() &&
            nbias.scalar_type() == z.scalar_type() &&
            layer_scale.scalar_type() == z.scalar_type(),
        "norm_glu_ls_resid_chlast: dtype mismatch");
    TORCH_CHECK(tgs >= 1 && tgs <= 1024,
                "norm_glu_ls_resid_chlast: bad block size");
    UNBLEND_DISPATCH(norm_glu_ls_resid_chlast_impl, z, out, z, resid, nweight, nbias, layer_scale, C2, X, eps, tgs)
}

void apply_norm_glu_ls_resid_chlast(
    const at::Tensor& out, const at::Tensor& z, const at::Tensor& resid,
    const at::Tensor& meanvar, const at::Tensor& nweight,
    const at::Tensor& nbias, const at::Tensor& layer_scale,
    int64_t total_in_per_b, int64_t total_out_per_b, int64_t num_tiles,
    int64_t C, int64_t tgs
) {
    TORCH_CHECK(z.is_cuda() && out.is_cuda() && resid.is_cuda(),
                "apply_norm_glu_ls_resid_chlast: tensors must be CUDA");
    TORCH_CHECK(
        z.scalar_type() == out.scalar_type() &&
            resid.scalar_type() == z.scalar_type() &&
            nweight.scalar_type() == z.scalar_type() &&
            nbias.scalar_type() == z.scalar_type() &&
            layer_scale.scalar_type() == z.scalar_type(),
        "apply_norm_glu_ls_resid_chlast: dtype mismatch");
    TORCH_CHECK(tgs >= 1 && tgs <= 1024,
                "apply_norm_glu_ls_resid_chlast: bad block size");
    UNBLEND_DISPATCH(apply_norm_glu_ls_resid_chlast_impl, z, out, z, resid, meanvar, nweight, nbias, layer_scale, total_in_per_b, total_out_per_b, num_tiles, C, tgs)
}
