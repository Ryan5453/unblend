// pybind11 bindings for the Unblend CUDA kernels.
//
// Exposes one host function per kernel, named exactly like its Metal twin
// in ``unblend/metal/*.metal``. Each function validates arguments, derives
// the launch configuration (grid of B or B * num_tiles blocks; blockDim.x =
// tgs threads — the analogues of Metal's ``threads=``/``group_size=``), and
// dispatches on the input's scalar type to a float/half/bfloat16 template
// instantiation compiled into the sibling .cu files.

#include <torch/extension.h>

#include "bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("group_norm_g1", &group_norm_g1,
          "num_groups=1 GroupNorm, one block per batch element");
    m.def("group_norm_g1_chlast", &group_norm_g1_chlast,
          "num_groups=1 GroupNorm over (B, T*C) with channel-last affine");
    m.def("partial_reduce", &partial_reduce,
          "Multi-stage stage 1: per-tile shifted (sum, sqsum) partials");
    m.def("finalize_meanvar", &finalize_meanvar,
          "Multi-stage stage 2: per-batch (mean, rsqrt(var+eps))");
    m.def("apply_norm", &apply_norm,
          "Multi-stage stage 3: normalize + affine (channel-first)");
    m.def("apply_norm_chlast", &apply_norm_chlast,
          "Multi-stage stage 3: normalize + affine (channel-last)");
    m.def("group_norm_g1_gelu", &group_norm_g1_gelu,
          "gelu(group_norm(x)) fused, single-stage");
    m.def("apply_norm_gelu", &apply_norm_gelu,
          "gelu(normalize(x)) fused, multi-stage third stage");
    m.def("group_norm_g1_glu", &group_norm_g1_glu,
          "glu(group_norm(x), dim=1) fused, single-stage");
    m.def("apply_norm_glu", &apply_norm_glu,
          "glu(normalize(x), dim=1) fused, multi-stage third stage");
    m.def("norm_glu_ls_resid", &norm_glu_ls_resid,
          "residual + layer_scale * glu(group_norm(z)), single-stage");
    m.def("apply_norm_glu_ls_resid", &apply_norm_glu_ls_resid,
          "residual + layer_scale * glu(group_norm(z)), multi-stage third stage");
    m.def("rms_norm", &rms_norm,
          "RoFormer last-dimension RMSNorm, one block per row");
    m.def("add_gelu", &add_gelu,
          "gelu(a + b) elementwise for the norm-free encoder layers");
    m.def("roformer_rotary", &roformer_rotary,
          "Fused interleaved rotary-position embedding (RoFormer q/k rotation)");
    m.def("group_norm_g1_chlast_gelu", &group_norm_g1_chlast_gelu,
          "gelu(group_norm(x + inject?)) fused, channel-last, single-stage");
    m.def("apply_norm_chlast_gelu", &apply_norm_chlast_gelu,
          "gelu(normalize(x + inject?)) fused, channel-last, multi-stage");
    m.def("group_norm_g1_chlast_glu", &group_norm_g1_chlast_glu,
          "glu(group_norm(x), dim=-1) fused, channel-last, single-stage");
    m.def("apply_norm_chlast_glu", &apply_norm_chlast_glu,
          "glu(normalize(x)) fused, channel-last, multi-stage");
    m.def("norm_glu_ls_resid_chlast", &norm_glu_ls_resid_chlast,
          "residual + layer_scale * glu(group_norm(z)), channel-last, single-stage");
    m.def("apply_norm_glu_ls_resid_chlast", &apply_norm_glu_ls_resid_chlast,
          "residual + layer_scale * glu(group_norm(z)), channel-last, multi-stage");
}
