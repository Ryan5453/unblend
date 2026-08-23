// DConv envelope kernels: the post-conv2 path inside a DConv layer fused
// down to a single launch (or, for the multi-stage variant, two launches
// over the input followed by one over the output).
// CUDA port of ``unblend/metal/dconv_envelope.metal``.
//
// ``norm_glu_ls_resid`` (single-stage) and ``apply_norm_glu_ls_resid``
// (multi-stage third stage) absorb GroupNorm into the same fused op:
//   output = residual + layer_scale * glu(group_norm(z))
// which replaces FOUR previously separate kernel launches (group_norm,
// glu, layerscale mul, residual add) with one. Used per DConv sub-layer.
// Vector/scalar path selection and the reduction helpers are shared via
// ``kernels.cuh``.

#include "bindings.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "kernels.cuh"

namespace {

template <typename SCALAR_T>
__global__ void norm_glu_ls_resid_kernel(
    SCALAR_T* __restrict__  out,        // (B, C, N)
    const SCALAR_T* __restrict__  z,    // (B, 2C, N)
    const SCALAR_T* __restrict__  residual,  // (B, C, N)
    const SCALAR_T* __restrict__  nweight,   // (2C,)
    const SCALAR_T* __restrict__  nbias,     // (2C,)
    const SCALAR_T* __restrict__  layer_scale,  // (C,)
    unsigned int C2,       // 2*C
    unsigned int N,
    float eps
) {
    __shared__ float sh_sum[MAX_WARPS];
    __shared__ float sh_sq[MAX_WARPS];
    __shared__ float bcast[2];

    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int b = blockIdx.x;

    const unsigned int C = C2 >> 1;
    const unsigned int total_in = C2 * N;
    const unsigned int total_out = C * N;
    // 64-bit base offsets: b * total overflows 32 bits on huge inputs.
    const SCALAR_T* __restrict__  z_b = z + (unsigned long long)b * total_in;
    const SCALAR_T* __restrict__  r_b = residual + (unsigned long long)b * total_out;
    SCALAR_T* __restrict__  o_b = out + (unsigned long long)b * total_out;

    float K = static_cast<float>(z_b[0]);
    float s = 0.0f, sq = 0.0f;
    gn_accumulate_sumsq(z_b, total_in, K, tid, tgs, s, sq);
    gn_reduce_finalize(s, sq, K, total_in, eps, sh_sum, sh_sq, bcast);
    const float mean = bcast[0];
    const float scale = bcast[1];

    if ((N & 3u) == 0u) {
        const Scalar4<SCALAR_T>* z4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(z_b);
        const Scalar4<SCALAR_T>* r4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(r_b);
        Scalar4<SCALAR_T>* o4 = reinterpret_cast<Scalar4<SCALAR_T>*>(o_b);
        const unsigned int Nv = N >> 2;
        const unsigned int nv = C * Nv;
        const unsigned int boff = C * Nv;  // vector offset of channel c + C
        for (unsigned int i = tid; i < nv; i += tgs) {
            const unsigned int c = i / Nv;
            const float wa = static_cast<float>(nweight[c]);
            const float ba = static_cast<float>(nbias[c]);
            const float wb = static_cast<float>(nweight[c + C]);
            const float bb = static_cast<float>(nbias[c + C]);
            const float4 vz = unpack4(z4[i]);
            const float4 a = make_float4(
                (vz.x - mean) * scale * wa + ba,
                (vz.y - mean) * scale * wa + ba,
                (vz.z - mean) * scale * wa + ba,
                (vz.w - mean) * scale * wa + ba
            );
            const float4 vg = unpack4(z4[i + boff]);
            const float4 g = make_float4(
                (vg.x - mean) * scale * wb + bb,
                (vg.y - mean) * scale * wb + bb,
                (vg.z - mean) * scale * wb + bb,
                (vg.w - mean) * scale * wb + bb
            );
            const float4 sig = make_float4(
                1.0f / (1.0f + expf(-g.x)),
                1.0f / (1.0f + expf(-g.y)),
                1.0f / (1.0f + expf(-g.z)),
                1.0f / (1.0f + expf(-g.w))
            );
            const float ls = static_cast<float>(layer_scale[c]);
            const float4 r = unpack4(r4[i]);
            o4[i] = pack4<SCALAR_T>(make_float4(
                a.x * sig.x * ls + r.x,
                a.y * sig.y * ls + r.y,
                a.z * sig.z * ls + r.z,
                a.w * sig.w * ls + r.w
            ));
        }
    } else {
        for (unsigned int i = tid; i < total_out; i += tgs) {
            const unsigned int c = i / N;
            const unsigned int sp = i % N;
            const unsigned int idx_a = c * N + sp;
            const unsigned int idx_b = (c + C) * N + sp;
            const float wa = static_cast<float>(nweight[c]);
            const float ba = static_cast<float>(nbias[c]);
            const float wb = static_cast<float>(nweight[c + C]);
            const float bb = static_cast<float>(nbias[c + C]);
            const float a =
                (static_cast<float>(z_b[idx_a]) - mean) * scale * wa + ba;
            const float b_val =
                (static_cast<float>(z_b[idx_b]) - mean) * scale * wb + bb;
            const float sig = 1.0f / (1.0f + expf(-b_val));
            const float ls = static_cast<float>(layer_scale[c]);
            o_b[i] = static_cast<SCALAR_T>(
                a * sig * ls + static_cast<float>(r_b[i])
            );
        }
    }
}

template <typename SCALAR_T>
__global__ void apply_norm_glu_ls_resid_kernel(
    SCALAR_T* __restrict__  out,
    const SCALAR_T* __restrict__  z,
    const SCALAR_T* __restrict__  residual,
    const float* __restrict__  meanvar,
    const SCALAR_T* __restrict__  nweight,
    const SCALAR_T* __restrict__  nbias,
    const SCALAR_T* __restrict__  layer_scale,
    unsigned int total_in_per_b,   // 2C * N
    unsigned int total_out_per_b,  // C  * N
    unsigned int num_tiles,
    unsigned int N,
    unsigned int C
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int tgs = blockDim.x;
    const unsigned int bt = blockIdx.x;
    const unsigned int b = bt / num_tiles;
    const unsigned int t = bt % num_tiles;

    const float mean = meanvar[(unsigned long long)b * 2 + 0];
    const float scale = meanvar[(unsigned long long)b * 2 + 1];

    const SCALAR_T* __restrict__  z_b = z + (unsigned long long)b * total_in_per_b;
    const SCALAR_T* __restrict__  r_b = residual + (unsigned long long)b * total_out_per_b;
    SCALAR_T* __restrict__  o_b = out + (unsigned long long)b * total_out_per_b;

    if ((N & 3u) == 0u) {
        const Scalar4<SCALAR_T>* z4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(z_b);
        const Scalar4<SCALAR_T>* r4 =
            reinterpret_cast<const Scalar4<SCALAR_T>*>(r_b);
        Scalar4<SCALAR_T>* o4 = reinterpret_cast<Scalar4<SCALAR_T>*>(o_b);
        const unsigned int Nv = N >> 2;
        const unsigned int nv = total_out_per_b >> 2;
        const unsigned int boff = C * Nv;
        const unsigned int start =
            (unsigned int)((unsigned long long)t * nv / num_tiles);
        const unsigned int end =
            (unsigned int)((unsigned long long)(t + 1) * nv / num_tiles);
        for (unsigned int i = start + tid; i < end; i += tgs) {
            const unsigned int c = i / Nv;
            const float wa = static_cast<float>(nweight[c]);
            const float ba = static_cast<float>(nbias[c]);
            const float wb = static_cast<float>(nweight[c + C]);
            const float bb = static_cast<float>(nbias[c + C]);
            const float4 vz = unpack4(z4[i]);
            const float4 a = make_float4(
                (vz.x - mean) * scale * wa + ba,
                (vz.y - mean) * scale * wa + ba,
                (vz.z - mean) * scale * wa + ba,
                (vz.w - mean) * scale * wa + ba
            );
            const float4 vg = unpack4(z4[i + boff]);
            const float4 g = make_float4(
                (vg.x - mean) * scale * wb + bb,
                (vg.y - mean) * scale * wb + bb,
                (vg.z - mean) * scale * wb + bb,
                (vg.w - mean) * scale * wb + bb
            );
            const float4 sig = make_float4(
                1.0f / (1.0f + expf(-g.x)),
                1.0f / (1.0f + expf(-g.y)),
                1.0f / (1.0f + expf(-g.z)),
                1.0f / (1.0f + expf(-g.w))
            );
            const float ls = static_cast<float>(layer_scale[c]);
            const float4 r = unpack4(r4[i]);
            o4[i] = pack4<SCALAR_T>(make_float4(
                a.x * sig.x * ls + r.x,
                a.y * sig.y * ls + r.y,
                a.z * sig.z * ls + r.z,
                a.w * sig.w * ls + r.w
            ));
        }
    } else {
        const unsigned int start =
            (unsigned int)((unsigned long long)t * total_out_per_b / num_tiles);
        const unsigned int end =
            (unsigned int)((unsigned long long)(t + 1) * total_out_per_b / num_tiles);
        for (unsigned int i = start + tid; i < end; i += tgs) {
            const unsigned int c = i / N;
            const unsigned int sp = i % N;
            const unsigned int idx_a = c * N + sp;
            const unsigned int idx_b = (c + C) * N + sp;
            const float wa = static_cast<float>(nweight[c]);
            const float ba = static_cast<float>(nbias[c]);
            const float wb = static_cast<float>(nweight[c + C]);
            const float bb = static_cast<float>(nbias[c + C]);
            const float a =
                (static_cast<float>(z_b[idx_a]) - mean) * scale * wa + ba;
            const float b_val =
                (static_cast<float>(z_b[idx_b]) - mean) * scale * wb + bb;
            const float sig = 1.0f / (1.0f + expf(-b_val));
            const float ls = static_cast<float>(layer_scale[c]);
            o_b[i] = static_cast<SCALAR_T>(
                a * sig * ls + static_cast<float>(r_b[i])
            );
        }
    }
}

template <typename SCALAR_T>
void norm_glu_ls_resid_impl(
    const at::Tensor& out, const at::Tensor& z, const at::Tensor& residual,
    const at::Tensor& nweight, const at::Tensor& nbias,
    const at::Tensor& layer_scale, int64_t C2, int64_t N, double eps,
    int64_t tgs
) {
    const dim3 grid((unsigned int)z.size(0));
    const dim3 block((unsigned int)tgs);
    norm_glu_ls_resid_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<SCALAR_T>(), z.const_data_ptr<SCALAR_T>(),
        residual.const_data_ptr<SCALAR_T>(),
        nweight.const_data_ptr<SCALAR_T>(), nbias.const_data_ptr<SCALAR_T>(),
        layer_scale.const_data_ptr<SCALAR_T>(), (unsigned int)C2,
        (unsigned int)N, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SCALAR_T>
void apply_norm_glu_ls_resid_impl(
    const at::Tensor& out, const at::Tensor& z, const at::Tensor& residual,
    const at::Tensor& meanvar, const at::Tensor& nweight,
    const at::Tensor& nbias, const at::Tensor& layer_scale,
    int64_t total_in_per_b, int64_t total_out_per_b, int64_t num_tiles,
    int64_t N, int64_t C, int64_t tgs
) {
    const dim3 grid((unsigned int)(z.size(0) * num_tiles));
    const dim3 block((unsigned int)tgs);
    apply_norm_glu_ls_resid_kernel<SCALAR_T><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<SCALAR_T>(), z.const_data_ptr<SCALAR_T>(),
        residual.const_data_ptr<SCALAR_T>(), meanvar.const_data_ptr<float>(),
        nweight.const_data_ptr<SCALAR_T>(), nbias.const_data_ptr<SCALAR_T>(),
        layer_scale.const_data_ptr<SCALAR_T>(), (unsigned int)total_in_per_b,
        (unsigned int)total_out_per_b, (unsigned int)num_tiles,
        (unsigned int)N, (unsigned int)C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void norm_glu_ls_resid(
    const at::Tensor& out, const at::Tensor& z, const at::Tensor& residual,
    const at::Tensor& nweight, const at::Tensor& nbias,
    const at::Tensor& layer_scale, int64_t C2, int64_t N, double eps,
    int64_t tgs
) {
    TORCH_CHECK(z.is_cuda() && out.is_cuda() && residual.is_cuda(),
                "norm_glu_ls_resid: tensors must be CUDA");
    TORCH_CHECK(
        z.scalar_type() == out.scalar_type() &&
            residual.scalar_type() == z.scalar_type() &&
            nweight.scalar_type() == z.scalar_type() &&
            nbias.scalar_type() == z.scalar_type() &&
            layer_scale.scalar_type() == z.scalar_type(),
        "norm_glu_ls_resid: dtype mismatch");
    TORCH_CHECK(tgs >= 1 && tgs <= 1024, "norm_glu_ls_resid: bad block size");
    UNBLEND_DISPATCH(norm_glu_ls_resid_impl, z, out, z, residual, nweight, nbias, layer_scale, C2, N, eps, tgs)
}

void apply_norm_glu_ls_resid(
    const at::Tensor& out, const at::Tensor& z, const at::Tensor& residual,
    const at::Tensor& meanvar, const at::Tensor& nweight,
    const at::Tensor& nbias, const at::Tensor& layer_scale,
    int64_t total_in_per_b, int64_t total_out_per_b, int64_t num_tiles,
    int64_t N, int64_t C, int64_t tgs
) {
    TORCH_CHECK(z.is_cuda() && out.is_cuda() && residual.is_cuda(),
                "apply_norm_glu_ls_resid: tensors must be CUDA");
    TORCH_CHECK(
        z.scalar_type() == out.scalar_type() &&
            residual.scalar_type() == z.scalar_type() &&
            nweight.scalar_type() == z.scalar_type() &&
            nbias.scalar_type() == z.scalar_type() &&
            layer_scale.scalar_type() == z.scalar_type(),
        "apply_norm_glu_ls_resid: dtype mismatch");
    TORCH_CHECK(tgs >= 1 && tgs <= 1024,
                "apply_norm_glu_ls_resid: bad block size");
    UNBLEND_DISPATCH(apply_norm_glu_ls_resid_impl, z, out, z, residual, meanvar, nweight, nbias, layer_scale, total_in_per_b, total_out_per_b, num_tiles, N, C, tgs)
}
