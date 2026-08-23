"""
Minimal transposed-layout debug for roformer_rotary: prints every value.
"""

import os
import sys

os.environ.setdefault(
    "UNBLEND_CACHE_DIR", "/projects/fahey.rya/unblend-bench/.model-cache"
)
REPO = "/projects/fahey.rya/unblend-cuda/repo"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

from unblend.cuda import _get_extension  # noqa: E402, F401


def main() -> None:
    """Tiny case, full dump."""
    ext = _get_extension()
    dev = torch.device("cuda")
    B, H, S, Dh = 2, 2, 2, 4
    packed = torch.arange(B * S * H * Dh, device=dev, dtype=torch.float32).reshape(
        B, S, H, Dh
    )
    strided = packed.transpose(1, 2)  # [B,H,S,Dh]
    angles = torch.randn(S, Dh // 2, device=dev)
    cos = angles.cos()
    sin = angles.sin()

    out = ext.roformer_rotary(strided, cos, sin)

    x1, x2 = strided[..., 0::2], strided[..., 1::2]
    ref = torch.empty_like(strided)
    ref[..., 0::2] = x1 * cos - x2 * sin
    ref[..., 1::2] = x1 * sin + x2 * cos

    print("strides:", strided.stride())
    for b in range(B):
        for h in range(H):
            for s in range(S):
                for d in range(Dh):
                    o = out[b, h, s, d].item()
                    r = ref[b, h, s, d].item()
                    if abs(o - r) > 1e-4:
                        print(
                            f"MISMATCH b{b} h{h} s{s} d{d}: kernel={o:.4f} ref={r:.4f}"
                        )
    print("sample kernel[0,0]:", out[0, 0].tolist())
    print("sample ref   [0,0]:", ref[0, 0].tolist())


if __name__ == "__main__":
    main()
