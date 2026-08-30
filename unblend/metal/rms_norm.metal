// RoFormer RMSNorm over the contiguous last dimension.
//
// One SIMDGROUP handles one row, so the sum-of-squares reduction is a single
// ``simd_sum`` with no threadgroup barriers and no threadgroup memory. (The
// previous shape — one 256-thread threadgroup per row reducing through a
// shared-memory tree — spent log2(tgs) barriers and 4 KB of threadgroup
// memory per row to reduce as little as one element per thread, which capped
// occupancy and left the kernel at ~3% of memory bandwidth.)
//
// A threadgroup packs ``tgs / 32`` simdgroups and therefore normalizes that
// many rows; the host launches ceil(rows / rows_per_tg) threadgroups and the
// kernel bounds-checks the tail.
//
// Inputs and affine weights may be FP32, FP16, or BF16, but the reduction and
// affine arithmetic stay in FP32 to match ``RMSNorm.forward``. The Python side
// injects SCALAR_T / SCALAR4_T.

#ifndef SCALAR_T
#define SCALAR_T half
#define SCALAR4_T half4
#endif

kernel void rms_norm(
    device SCALAR_T*       out   [[buffer(0)]],
    device const SCALAR_T* in_   [[buffer(1)]],
    device const SCALAR_T* gamma [[buffer(2)]],
    constant uint&         dim   [[buffer(3)]],
    constant float&        scale [[buffer(4)]],
    constant uint&         rows  [[buffer(5)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tgs  [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sid  [[simdgroup_index_in_threadgroup]]
) {
    // Whole simdgroups take the same branch, so returning here never strands
    // a lane inside the simd_sum below.
    const uint row = tg * (tgs >> 5) + sid;
    if (row >= rows) {
        return;
    }

    const ulong base = (ulong)row * dim;
    device const SCALAR_T* x_row = in_ + base;
    device SCALAR_T*       o_row = out + base;

    if ((dim & 3u) == 0u) {
        // dim % 4 == 0 makes every row base a multiple of 4 elements, so the
        // vector casts stay naturally aligned.
        device const SCALAR4_T* x4 = (device const SCALAR4_T*)x_row;
        device SCALAR4_T*       o4 = (device SCALAR4_T*)o_row;
        device const SCALAR4_T* g4 = (device const SCALAR4_T*)gamma;
        const uint nv = dim >> 2;

        float local_sqsum = 0.0f;
        for (uint i = lane; i < nv; i += 32) {
            const float4 v = float4(x4[i]);
            local_sqsum += dot(v, v);
        }
        // F.normalize divides by max(L2 norm, 1e-12), then RoFormer
        // multiplies by sqrt(dim). Keep that exact convention rather than
        // introducing the additive epsilon used by other RMSNorm variants.
        const float multiplier =
            scale / max(sqrt(simd_sum(local_sqsum)), 1.0e-12f);

        for (uint i = lane; i < nv; i += 32) {
            o4[i] = SCALAR4_T(float4(x4[i]) * multiplier * float4(g4[i]));
        }
    } else {
        float local_sqsum = 0.0f;
        for (uint i = lane; i < dim; i += 32) {
            const float value = float(x_row[i]);
            local_sqsum += value * value;
        }
        const float multiplier =
            scale / max(sqrt(simd_sum(local_sqsum)), 1.0e-12f);

        for (uint i = lane; i < dim; i += 32) {
            o_row[i] = SCALAR_T(float(x_row[i]) * multiplier * float(gamma[i]));
        }
    }
}
