// Fused interleaved rotary-position embedding (RoFormer).
//
// ``RotaryEmbedding.rotate_queries_or_keys`` applies, per adjacent element
// pair ``(x1, x2)`` of the last axis, the complex product
// ``(x1 + i·x2)·e^{iθ}``. In eager PyTorch that is an unflatten/unbind into
// two STRIDED half-tensors, four muls, a subtract, an add, then a stack and
// flatten to re-interleave — seven full-tensor passes per query/key, most of
// them over non-unit-stride views. This does it in one read and one write.
//
// Queries/keys reach this op as ``[B, H, S, Dh]`` transposed views, so the
// input is read STRIDED via a mixed-radix coordinate walk (the same scheme
// ``unblend/cuda/rotary.cu`` uses) rather than forcing a contiguous copy
// first. Only ``stride(-1) == 1`` is required.
//
// Two shapes are provided. ``roformer_rotary`` handles one pair per thread
// and works for any even last dimension. ``roformer_rotary_v1``/``_v2`` are
// the templated vector forms, moving 8 and 16 bytes per lane; the 16-byte
// form is what reaches memory-bandwidth peak on Apple GPUs (measured on an
// M2 Max: 107 GB/s scalar, 210 GB/s at 8 B/lane, 379 GB/s at 16 B/lane,
// against a ~380 GB/s ceiling). The host picks the widest form whose
// alignment preconditions hold and falls back down the list otherwise.
//
// Arithmetic deliberately rounds every intermediate back to SCALAR_T so the
// result is bit-identical to the unfused eager path, which computes each
// fp16/bf16 op in FP32 and stores back to the working dtype. All three forms
// are bit-identical to eager and to each other.

#ifndef SCALAR_T
#define SCALAR_T half
#define SCALAR4_T half4
#endif

// One rotated pair: (a, b) -> (a·cos - b·sin, a·sin + b·cos), rounding each
// product to SCALAR_T the way eager TensorIterator does. The caller applies
// the final store-rounding.
inline float2 rotary_pair(float a, float b, float c, float sn) {
    const float p1 = float(SCALAR_T(a * c));
    const float p2 = float(SCALAR_T(b * sn));
    const float p3 = float(SCALAR_T(a * sn));
    const float p4 = float(SCALAR_T(b * c));
    return float2(p1 - p2, p3 + p4);
}

// Offset of the pair/vector unit ``lead`` selects, walking the leading dims in
// mixed radix. ``unit_elems`` is how many elements one thread's unit spans.
inline ulong rotary_addr(
    device const long* sizes,
    device const long* strides,
    uint ndim,
    uint s,
    uint lead,
    uint elem_in_row
) {
    ulong addr = (ulong)s * (ulong)strides[ndim - 2] + (ulong)elem_in_row;
    uint rem = lead;
    for (int d = (int)ndim - 3; d >= 0; --d) {
        if (rem == 0u) {
            break;  // all remaining digits are zero
        }
        const uint sz = (uint)sizes[d];
        addr += (ulong)(rem % sz) * (ulong)strides[d];
        rem /= sz;
    }
    return addr;
}

// Scalar form: one pair per thread. Works for any even last dimension and any
// stride pattern with stride(-1) == 1.
kernel void roformer_rotary(
    device SCALAR_T*       out    [[buffer(0)]],   // contiguous [..., S, 2P]
    device const SCALAR_T* x      [[buffer(1)]],   // strided    [..., S, 2P]
    device const SCALAR_T* cos_t  [[buffer(2)]],   // [S, P]
    device const SCALAR_T* sin_t  [[buffer(3)]],   // [S, P]
    device const long*     layout [[buffer(4)]],   // sizes[ndim], then strides[ndim]
    constant uint&         ndim   [[buffer(5)]],
    constant uint&         P      [[buffer(6)]],   // dim / 2
    constant uint&         S      [[buffer(7)]],   // sequence length
    constant uint&         units  [[buffer(8)]],   // numel / 2
    uint gid [[thread_position_in_grid]],
    uint gsz [[threads_per_grid]]
) {
    device const long* sizes   = layout;
    device const long* strides = layout + ndim;

    for (uint g = gid; g < units; g += gsz) {
        const uint k    = g % P;
        uint tmp        = g / P;
        const uint s    = tmp % S;
        const uint lead = tmp / S;

        const ulong addr = rotary_addr(sizes, strides, ndim, s, lead, 2u * k);
        const ulong trig = (ulong)s * (ulong)P + (ulong)k;
        const ulong ooff = ((ulong)lead * (ulong)S + (ulong)s) * (ulong)(2u * P)
                         + (ulong)(2u * k);

        const float2 r = rotary_pair(
            float(x[addr]), float(x[addr + 1]),
            float(cos_t[trig]), float(sin_t[trig])
        );
        out[ooff]     = SCALAR_T(r.x);
        out[ooff + 1] = SCALAR_T(r.y);
    }
}

// Vector form: NV SCALAR4_T units (4*NV elements, 2*NV pairs) per thread.
// Requires P % (2*NV) == 0 and every walked stride to be a multiple of 4, both
// checked host-side.
template <uint NV>
kernel void rotary_vec(
    device SCALAR_T*       out    [[buffer(0)]],
    device const SCALAR_T* x      [[buffer(1)]],
    device const SCALAR_T* cos_t  [[buffer(2)]],
    device const SCALAR_T* sin_t  [[buffer(3)]],
    device const long*     layout [[buffer(4)]],
    constant uint&         ndim   [[buffer(5)]],
    constant uint&         P      [[buffer(6)]],
    constant uint&         S      [[buffer(7)]],
    constant uint&         units  [[buffer(8)]],   // numel / (4*NV)
    uint gid [[thread_position_in_grid]],
    uint gsz [[threads_per_grid]]
) {
    device const long* sizes   = layout;
    device const long* strides = layout + ndim;
    const uint Q = P / (2u * NV);   // units per row

    for (uint g = gid; g < units; g += gsz) {
        const uint kq   = g % Q;
        uint tmp        = g / Q;
        const uint s    = tmp % S;
        const uint lead = tmp / S;

        const ulong addr =
            rotary_addr(sizes, strides, ndim, s, lead, 4u * NV * kq);
        const ulong trig = (ulong)s * (ulong)P + (ulong)(2u * NV * kq);
        const ulong ooff = ((ulong)lead * (ulong)S + (ulong)s) * (ulong)(2u * P)
                         + (ulong)(4u * NV * kq);

        device const SCALAR4_T* xv = (device const SCALAR4_T*)(x + addr);
        device SCALAR4_T*       ov = (device SCALAR4_T*)(out + ooff);

        #pragma unroll
        for (uint j = 0; j < NV; ++j) {
            const float4 v = float4(xv[j]);
            const float2 q0 = rotary_pair(v.x, v.y,
                                          float(cos_t[trig + 2u * j]),
                                          float(sin_t[trig + 2u * j]));
            const float2 q1 = rotary_pair(v.z, v.w,
                                          float(cos_t[trig + 2u * j + 1u]),
                                          float(sin_t[trig + 2u * j + 1u]));
            ov[j] = SCALAR4_T(float4(q0.x, q0.y, q1.x, q1.y));
        }
    }
}

#define ROTARY_INST(n)                                                    \
    template [[host_name("roformer_rotary_v" #n)]]                        \
    kernel void rotary_vec<n>(                                            \
        device SCALAR_T*, device const SCALAR_T*, device const SCALAR_T*, \
        device const SCALAR_T*, device const long*, constant uint&,       \
        constant uint&, constant uint&, constant uint&, uint, uint)

ROTARY_INST(1);
ROTARY_INST(2);
