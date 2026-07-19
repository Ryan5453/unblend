// RoFormer RMSNorm over the contiguous last dimension.
//
// One threadgroup handles one row. Inputs and affine weights may be FP32,
// FP16, or BF16, but the sum-of-squares reduction and affine arithmetic stay
// in FP32 to match ``RMSNorm.forward``. The Python side injects SCALAR_T.

#include <metal_stdlib>
using namespace metal;

#ifndef SCALAR_T
#define SCALAR_T half
#endif

kernel void rms_norm(
    device SCALAR_T*       out   [[buffer(0)]],
    device const SCALAR_T* in_   [[buffer(1)]],
    device const SCALAR_T* gamma [[buffer(2)]],
    constant uint&         dim   [[buffer(3)]],
    constant float&        scale [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    constexpr uint MAX_TGS = 1024;
    threadgroup float shared_sqsum[MAX_TGS];

    const ulong base = (ulong)row * dim;
    float local_sqsum = 0.0f;
    for (uint i = tid; i < dim; i += tgs) {
        const float value = float(in_[base + i]);
        local_sqsum += value * value;
    }
    shared_sqsum[tid] = local_sqsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgs >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sqsum[tid] += shared_sqsum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float multiplier;
    if (tid == 0) {
        // F.normalize divides by max(L2 norm, 1e-12), then RoFormer
        // multiplies by sqrt(dim). Keep that exact convention rather than
        // introducing the additive epsilon used by other RMSNorm variants.
        const float norm = sqrt(shared_sqsum[0]);
        multiplier = scale / max(norm, 1.0e-12f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < dim; i += tgs) {
        const float value = float(in_[base + i]);
        const float gain = float(gamma[i]);
        out[base + i] = SCALAR_T(value * multiplier * gain);
    }
}
