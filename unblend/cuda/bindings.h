// Host-side launcher declarations for the Unblend CUDA kernels.
//
// Every launcher mirrors the identically-named Metal kernel in
// ``unblend/metal/*.metal``: same arguments, same semantics, same
// single-kernel responsibility. The launch configuration (grid of B or
// B * num_tiles blocks, blockDim.x = tgs threads) is derived here from the
// tensor shapes and the heuristic parameters the Python side computes —
// the CUDA analogues of Metal's ``threads=``/``group_size=`` kwargs.
//
// Each launcher dispatches on the input's scalar type (float32/float16/
// bfloat16) to a template instantiation of its kernel; all kernels live in
// the sibling .cu files and include this header so signatures stay in sync.

#pragma once

#include <ATen/ATen.h>

// ---------------------------------------------------------------------------
// Shared launcher boilerplate (used by every .cu file in this folder)
// ---------------------------------------------------------------------------

// Dispatch on TENSOR's scalar type to the ``NAME`` template launcher
// instantiated for float / half / bfloat16.
#define UNBLEND_DISPATCH(NAME, TENSOR, ...)                                \
    switch ((TENSOR).scalar_type()) {                                      \
        case at::ScalarType::Float:                                        \
            NAME<float>(__VA_ARGS__);                                      \
            break;                                                         \
        case at::ScalarType::Half:                                         \
            NAME<c10::Half>(__VA_ARGS__);                                  \
            break;                                                         \
        case at::ScalarType::BFloat16:                                     \
            NAME<c10::BFloat16>(__VA_ARGS__);                              \
            break;                                                         \
        default:                                                           \
            TORCH_CHECK(                                                   \
                false, "unblend CUDA kernels: unsupported dtype ",         \
                (TENSOR).scalar_type());                                   \
    }

// Common argument validation for kernels shaped (out, in, weight, bias, ...).
#define UNBLEND_CHECKS(NAME)                                               \
    TORCH_CHECK(in_.is_cuda() && out.is_cuda(), #NAME ": tensors must be CUDA"); \
    TORCH_CHECK(                                                           \
        in_.scalar_type() == out.scalar_type() &&                          \
            weight.scalar_type() == in_.scalar_type() &&                   \
            bias.scalar_type() == in_.scalar_type(),                       \
        #NAME ": dtype mismatch");                                         \
    TORCH_CHECK(tgs >= 1 && tgs <= 1024, #NAME ": bad block size");


void group_norm_g1(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& weight,
    const at::Tensor& bias, int64_t C, int64_t N, double eps, int64_t tgs);

void group_norm_g1_chlast(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& weight,
    const at::Tensor& bias, int64_t C, int64_t total, double eps, int64_t tgs);

void partial_reduce(
    const at::Tensor& in_, const at::Tensor& inject, const at::Tensor& scratch,
    int64_t total_per_b, int64_t num_tiles, int64_t tgs);

void finalize_meanvar(
    const at::Tensor& scratch, const at::Tensor& meanvar, int64_t total_per_b,
    int64_t num_tiles, double eps, const at::Tensor& in_,
    const at::Tensor& inject, int64_t tgs);

void apply_norm(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& meanvar,
    const at::Tensor& weight, const at::Tensor& bias, int64_t total_per_b,
    int64_t num_tiles, int64_t N, int64_t tgs);

void apply_norm_chlast(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& meanvar,
    const at::Tensor& weight, const at::Tensor& bias, int64_t total_per_b,
    int64_t num_tiles, int64_t C, int64_t tgs);

void group_norm_g1_gelu(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& inject,
    const at::Tensor& weight, const at::Tensor& bias, int64_t C, int64_t N,
    double eps, int64_t tgs);

void apply_norm_gelu(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& inject,
    const at::Tensor& meanvar, const at::Tensor& weight, const at::Tensor& bias,
    int64_t total_per_b, int64_t num_tiles, int64_t N, int64_t tgs);

void group_norm_g1_glu(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& weight,
    const at::Tensor& bias, int64_t C, int64_t N, double eps, int64_t tgs);

void apply_norm_glu(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& meanvar,
    const at::Tensor& weight, const at::Tensor& bias, int64_t total_in_per_b,
    int64_t total_out_per_b, int64_t num_tiles, int64_t N, int64_t C_half,
    int64_t tgs);

void norm_glu_ls_resid(
    const at::Tensor& out, const at::Tensor& z, const at::Tensor& residual,
    const at::Tensor& nweight, const at::Tensor& nbias,
    const at::Tensor& layer_scale, int64_t C2, int64_t N, double eps,
    int64_t tgs);

void apply_norm_glu_ls_resid(
    const at::Tensor& out, const at::Tensor& z, const at::Tensor& residual,
    const at::Tensor& meanvar, const at::Tensor& nweight,
    const at::Tensor& nbias, const at::Tensor& layer_scale,
    int64_t total_in_per_b, int64_t total_out_per_b, int64_t num_tiles,
    int64_t N, int64_t C, int64_t tgs);

void rms_norm(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& gamma,
    int64_t dim, double scale, int64_t tgs);

void add_gelu(
    const at::Tensor& out, const at::Tensor& a, const at::Tensor& b);

void group_norm_g1_chlast_gelu(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& inject,
    const at::Tensor& weight, const at::Tensor& bias, int64_t C, int64_t total,
    double eps, int64_t tgs);

void apply_norm_chlast_gelu(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& inject,
    const at::Tensor& meanvar, const at::Tensor& weight, const at::Tensor& bias,
    int64_t total_per_b, int64_t num_tiles, int64_t C, int64_t tgs);

void group_norm_g1_chlast_glu(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& weight,
    const at::Tensor& bias, int64_t C2, int64_t X, double eps, int64_t tgs);

void apply_norm_chlast_glu(
    const at::Tensor& out, const at::Tensor& in_, const at::Tensor& meanvar,
    const at::Tensor& weight, const at::Tensor& bias, int64_t total_in_per_b,
    int64_t total_out_per_b, int64_t num_tiles, int64_t C, int64_t tgs);

void norm_glu_ls_resid_chlast(
    const at::Tensor& out, const at::Tensor& z, const at::Tensor& resid,
    const at::Tensor& nweight, const at::Tensor& nbias,
    const at::Tensor& layer_scale, int64_t C2, int64_t X, double eps,
    int64_t tgs);

void apply_norm_glu_ls_resid_chlast(
    const at::Tensor& out, const at::Tensor& z, const at::Tensor& resid,
    const at::Tensor& meanvar, const at::Tensor& nweight,
    const at::Tensor& nbias, const at::Tensor& layer_scale,
    int64_t total_in_per_b, int64_t total_out_per_b, int64_t num_tiles,
    int64_t C, int64_t tgs);

at::Tensor roformer_rotary(
    const at::Tensor& t, const at::Tensor& cos, const at::Tensor& sin);
