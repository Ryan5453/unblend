// Fused interleaved rotary-position embedding (RoFormer).
//
// ``RotaryEmbedding.rotate_queries_or_keys`` applies, per adjacent element
// pair ``(x1, x2)`` of the last axis, the complex product
// ``(x1 + i·x2)·e^{iθ}`` — in eager PyTorch that is four muls, a subtract,
// an add and an interleave copy: seven full-tensor passes per query/key.
// This kernel does it in one read and one write, accumulating in FP32,
// and reads STRIDED inputs natively so transposed-head layouts
// ([B, H, S, Dh] after ``transpose(1, 2)``) need no ``contiguous`` copy.
//
// Requires only that the last dimension has stride 1 (checked host-side);
// any leading-dim strides are honored via a mixed-radix coordinate walk.

#include "bindings.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>

namespace {

constexpr int MAX_DIMS = 8;

struct RotaryLayout {
    unsigned long long sizes[MAX_DIMS];
    unsigned long long strides[MAX_DIMS];
    int ndim;
};

template <typename SCALAR_T>
__global__ void roformer_rotary_kernel(
    SCALAR_T* __restrict__ out,              // contiguous [*, S, 2P]
    const SCALAR_T* __restrict__ x,          // strided [*, S, 2P]
    const SCALAR_T* __restrict__ cos_t,      // [S, P]
    const SCALAR_T* __restrict__ sin_t,      // [S, P]
    unsigned long long pairs,                // numel / 2
    unsigned int half,                       // P = dim / 2
    unsigned int seq,                        // S
    RotaryLayout layout
) {
    unsigned long long g =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long gstride =
        (unsigned long long)gridDim.x * blockDim.x;

    const unsigned int P = half;
    const unsigned int S = seq;
    const int ndim = layout.ndim;

    for (; g < pairs; g += gstride) {
        // Decompose g -> (lead, s, k) with k fastest.
        const unsigned int k = (unsigned int)(g % P);
        unsigned long long tmp = g / P;
        const unsigned int s = (unsigned int)(tmp % S);
        unsigned long long lead = tmp / S;

        // Address of x1 via mixed radix over the leading dims.
        unsigned long long addr =
            (unsigned long long)s * layout.strides[ndim - 2] +
            (unsigned long long)(2 * k);
        unsigned long long rem = lead;
        for (int d = ndim - 3; d >= 0; --d) {
            if (rem == 0) {
                break;  // all remaining digits are zero
            }
            const unsigned long long idx = rem % layout.sizes[d];
            rem /= layout.sizes[d];
            addr += idx * layout.strides[d];
        }

        const float c =
            static_cast<float>(cos_t[(unsigned long long)s * P + k]);
        const float sn =
            static_cast<float>(sin_t[(unsigned long long)s * P + k]);
        const float a = static_cast<float>(x[addr]);
        const float b = static_cast<float>(x[addr + 1]);
        // Replicate eager PyTorch's per-op rounding exactly: TensorIterator
        // computes each fp16/bf16 op in FP32 and stores back to the working
        // dtype, and downstream torch.compile graphs trace this same op
        // sequence. Matching the rounding order keeps the fused kernel
        // bit-identical to the unfused path — compiled-vs-eager parity and
        // long-chain divergence both depend on it.
        const float p1 = static_cast<float>(static_cast<SCALAR_T>(a * c));
        const float p2 = static_cast<float>(static_cast<SCALAR_T>(b * sn));
        const float p3 = static_cast<float>(static_cast<SCALAR_T>(a * sn));
        const float p4 = static_cast<float>(static_cast<SCALAR_T>(b * c));
        const float r0 = static_cast<float>(static_cast<SCALAR_T>(p1 - p2));
        const float r1 = static_cast<float>(static_cast<SCALAR_T>(p3 + p4));
        const unsigned long long ooff =
            ((unsigned long long)(lead * S + s)) * (2ull * P) + 2ull * k;
        out[ooff] = static_cast<SCALAR_T>(r0);
        out[ooff + 1] = static_cast<SCALAR_T>(r1);
    }
}

at::Tensor roformer_rotary_impl(
    const at::Tensor& t, const at::Tensor& cos, const at::Tensor& sin
) {
    // Output must be plain contiguous: the kernel writes in flat
    // [..., S, D] order regardless of the input's strides, and empty_like
    // would preserve a transposed input's stride pattern, scrambling reads.
    auto out = at::empty(t.sizes(), t.options());
    const int64_t dim = t.size(-1);
    const int64_t half = dim / 2;
    const unsigned long long pairs = t.numel() / 2ull;
    if (pairs == 0) {
        return out;
    }
    const unsigned int seq = (unsigned int)(cos.size(0));
    const int ndim = t.dim();

    RotaryLayout layout;
    layout.ndim = ndim;
    TORCH_CHECK(ndim <= MAX_DIMS, "roformer_rotary: too many dims");
    for (int d = 0; d < ndim; ++d) {
        layout.sizes[d] = (unsigned long long)t.size(d);
        layout.strides[d] = (unsigned long long)t.stride(d);
    }

    const unsigned int threads = 256;
    const unsigned int blocks = (unsigned int)std::min<unsigned long long>(
        (pairs + threads - 1) / threads, 8192ull
    );

    switch (t.scalar_type()) {
        case at::ScalarType::Float:
            roformer_rotary_kernel<float><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                out.data_ptr<float>(), t.const_data_ptr<float>(),
                cos.const_data_ptr<float>(), sin.const_data_ptr<float>(),
                pairs, (unsigned int)half, seq, layout);
            break;
        case at::ScalarType::Half:
            roformer_rotary_kernel<c10::Half><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                out.data_ptr<c10::Half>(), t.const_data_ptr<c10::Half>(),
                cos.const_data_ptr<c10::Half>(), sin.const_data_ptr<c10::Half>(),
                pairs, (unsigned int)half, seq, layout);
            break;
        case at::ScalarType::BFloat16:
            roformer_rotary_kernel<c10::BFloat16><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                out.data_ptr<c10::BFloat16>(), t.const_data_ptr<c10::BFloat16>(),
                cos.const_data_ptr<c10::BFloat16>(), sin.const_data_ptr<c10::BFloat16>(),
                pairs, (unsigned int)half, seq, layout);
            break;
        default:
            TORCH_CHECK(false, "roformer_rotary: unsupported dtype");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

}  // namespace

at::Tensor roformer_rotary(
    const at::Tensor& t, const at::Tensor& cos, const at::Tensor& sin
) {
    TORCH_CHECK(t.is_cuda(), "roformer_rotary: tensor must be CUDA");
    TORCH_CHECK(
        cos.scalar_type() == t.scalar_type() && sin.scalar_type() == t.scalar_type(),
        "roformer_rotary: dtype mismatch");
    TORCH_CHECK(t.size(-1) % 2 == 0, "roformer_rotary: last dim must be even");
    TORCH_CHECK(t.stride(-1) == 1, "roformer_rotary: last dim must be contiguous");
    return roformer_rotary_impl(t, cos, sin);
}
