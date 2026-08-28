"""
CUDA kernels for low-precision inference on NVIDIA GPUs.
"""

from __future__ import annotations

import logging
import os
import threading
import warnings
from importlib import resources
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _pow2_tgs(max_threads: int, cap: int | None = None) -> int:
    """
    Largest power of two ``<= min(cap, max_threads)``.

    :param max_threads: The device's ``max_threads_per_block``.
    :param cap: Upper bound on the returned block size; defaults to the.
    :return: The largest power-of-two block size within the bounds.
    """
    if cap is None:
        cap = int(os.environ.get("UNBLEND_CUDA_TGS_CAP", "256"))
    limit = min(cap, max_threads)
    tgs = 1
    while tgs * 2 <= limit:
        tgs *= 2
    return tgs


_SOURCE_FILES: list[str] = [
    "group_norm.cu",
    "group_norm_gelu.cu",
    "group_norm_glu.cu",
    "dconv_envelope.cu",
    "rms_norm.cu",
    "chlast_act.cu",
    "rotary.cu",
]

_KERNEL_SOURCES: dict[str, str] = {
    "group_norm_g1": "group_norm.cu",
    "group_norm_g1_chlast": "group_norm.cu",
    "partial_reduce": "group_norm.cu",
    "finalize_meanvar": "group_norm.cu",
    "apply_norm": "group_norm.cu",
    "apply_norm_chlast": "group_norm.cu",
    "group_norm_g1_gelu": "group_norm_gelu.cu",
    "apply_norm_gelu": "group_norm_gelu.cu",
    "add_gelu": "group_norm_gelu.cu",
    "group_norm_g1_glu": "group_norm_glu.cu",
    "apply_norm_glu": "group_norm_glu.cu",
    "norm_glu_ls_resid": "dconv_envelope.cu",
    "apply_norm_glu_ls_resid": "dconv_envelope.cu",
    "rms_norm": "rms_norm.cu",
    "roformer_rotary": "rotary.cu",
    "group_norm_g1_chlast_gelu": "chlast_act.cu",
    "apply_norm_chlast_gelu": "chlast_act.cu",
    "group_norm_g1_chlast_glu": "chlast_act.cu",
    "apply_norm_chlast_glu": "chlast_act.cu",
    "norm_glu_ls_resid_chlast": "chlast_act.cu",
    "apply_norm_glu_ls_resid_chlast": "chlast_act.cu",
}

_LP_DTYPES = frozenset((torch.float16, torch.bfloat16))
_RMS_DTYPES = frozenset((torch.float32, *_LP_DTYPES))

_extension: Any = None
_extension_error: str | None = None
_extension_lock = threading.Lock()
_warmup_thread: threading.Thread | None = None
_build_notice_sent = False
_max_threads_cache: dict[int, int] = {}
_sm_count_cache: dict[int, int] = {}
_OPS_REGISTERED = False

_SWAPPABLE_BACKENDS = frozenset({"demucs", "scnet"})


def swappable_backends() -> frozenset[str]:
    """Eligible backend names for fused-kernel swaps.

    :return: Backend names.
    """
    return _SWAPPABLE_BACKENDS


_OP_NAMESPACE = "unblend_cuda"


def _ensure_custom_ops() -> None:
    """
    Register CUDA kernels as torch.library custom ops.
    """

    global _OPS_REGISTERED
    if _OPS_REGISTERED:
        return

    @torch.library.custom_op(f"{_OP_NAMESPACE}::group_norm_g1", mutates_args={"out"})
    def group_norm_g1(
        out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        C: int,
        N: int,
        eps: float,
        tgs: int,
    ) -> None:
        """
        Launch the ``group_norm_g1`` CUDA kernel (custom-op wrapper).

        :param out: Output buffer, written in place.
        :param x: Input tensor.
        :param weight: Affine weight.
        :param bias: Affine bias.
        :param C: Channel count of the reduction space.
        :param N: Spatial element count per batch element.
        :param eps: Variance epsilon.
        :param tgs: Block size (threads per block).
        """
        _get_kernel("group_norm_g1", x.dtype)(out, x, weight, bias, C, N, eps, tgs)

    @torch.library.custom_op(
        f"{_OP_NAMESPACE}::group_norm_g1_chlast", mutates_args={"out"}
    )
    def group_norm_g1_chlast(
        out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        C: int,
        total: int,
        eps: float,
        tgs: int,
    ) -> None:
        """
        Launch the ``group_norm_g1_chlast`` CUDA kernel (custom-op wrapper).

        :param out: Output buffer, written in place.
        :param x: Input tensor.
        :param weight: Affine weight.
        :param bias: Affine bias.
        :param C: Channel count of the reduction space.
        :param total: Total elements per batch element (``T * C``).
        :param eps: Variance epsilon.
        :param tgs: Block size (threads per block).
        """
        _get_kernel("group_norm_g1_chlast", x.dtype)(
            out, x, weight, bias, C, total, eps, tgs
        )

    @torch.library.custom_op(
        f"{_OP_NAMESPACE}::partial_reduce", mutates_args={"scratch"}
    )
    def partial_reduce(
        x: torch.Tensor,
        inject: torch.Tensor,
        scratch: torch.Tensor,
        total_per_b: int,
        num_tiles: int,
        tgs: int,
    ) -> None:
        """
        Launch the ``partial_reduce`` CUDA kernel (custom-op wrapper).

        :param x: Input tensor.
        :param inject: Optional second input added elementwise before normalizat...
        :param scratch: FP32 ``(B, num_tiles, 2)`` scratch buffer, written in place.
        :param total_per_b: Elements reduced per batch element.
        :param num_tiles: Tile count for the multi-stage launches.
        :param tgs: Block size (threads per block).
        """
        _get_kernel("partial_reduce", x.dtype)(
            x, inject, scratch, total_per_b, num_tiles, tgs
        )

    @torch.library.custom_op(
        f"{_OP_NAMESPACE}::finalize_meanvar", mutates_args={"meanvar"}
    )
    def finalize_meanvar(
        scratch: torch.Tensor,
        meanvar: torch.Tensor,
        total_per_b: int,
        num_tiles: int,
        eps: float,
        x: torch.Tensor,
        inject: torch.Tensor,
        tgs: int,
    ) -> None:
        """
        Launch the ``finalize_meanvar`` CUDA kernel (custom-op wrapper).

        :param scratch: FP32 ``(B, num_tiles, 2)`` scratch buffer, written in place.
        :param meanvar: FP32 ``(B, 2)`` mean/rsqrt(var+eps) buffer, written in pl...
        :param total_per_b: Elements reduced per batch element.
        :param num_tiles: Tile count for the multi-stage launches.
        :param eps: Variance epsilon.
        :param x: Input tensor.
        :param inject: Optional second input added elementwise before normalizat...
        :param tgs: Block size (threads per block).
        """
        _get_kernel("finalize_meanvar", x.dtype)(
            scratch, meanvar, total_per_b, num_tiles, eps, x, inject, tgs
        )

    @torch.library.custom_op(f"{_OP_NAMESPACE}::apply_norm", mutates_args={"out"})
    def apply_norm(
        out: torch.Tensor,
        x: torch.Tensor,
        meanvar: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        total_per_b: int,
        num_tiles: int,
        N: int,
        tgs: int,
    ) -> None:
        """
        Launch the ``apply_norm`` CUDA kernel (custom-op wrapper).

        :param out: Output buffer, written in place.
        :param x: Input tensor.
        :param meanvar: FP32 ``(B, 2)`` mean/rsqrt(var+eps) buffer, written in pl...
        :param weight: Affine weight.
        :param bias: Affine bias.
        :param total_per_b: Elements reduced per batch element.
        :param num_tiles: Tile count for the multi-stage launches.
        :param N: Spatial element count per batch element.
        :param tgs: Block size (threads per block).
        """
        _get_kernel("apply_norm", x.dtype)(
            out, x, meanvar, weight, bias, total_per_b, num_tiles, N, tgs
        )

    @torch.library.custom_op(
        f"{_OP_NAMESPACE}::apply_norm_chlast", mutates_args={"out"}
    )
    def apply_norm_chlast(
        out: torch.Tensor,
        x: torch.Tensor,
        meanvar: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        total_per_b: int,
        num_tiles: int,
        C: int,
        tgs: int,
    ) -> None:
        """
        Launch the ``apply_norm_chlast`` CUDA kernel (custom-op wrapper).

        :param out: Output buffer, written in place.
        :param x: Input tensor.
        :param meanvar: FP32 ``(B, 2)`` mean/rsqrt(var+eps) buffer, written in pl...
        :param weight: Affine weight.
        :param bias: Affine bias.
        :param total_per_b: Elements reduced per batch element.
        :param num_tiles: Tile count for the multi-stage launches.
        :param C: Channel count of the reduction space.
        :param tgs: Block size (threads per block).
        """
        _get_kernel("apply_norm_chlast", x.dtype)(
            out, x, meanvar, weight, bias, total_per_b, num_tiles, C, tgs
        )

    @torch.library.custom_op(
        f"{_OP_NAMESPACE}::group_norm_g1_gelu", mutates_args={"out"}
    )
    def group_norm_g1_gelu(
        out: torch.Tensor,
        x: torch.Tensor,
        inject: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        C: int,
        N: int,
        eps: float,
        tgs: int,
    ) -> None:
        """
        Launch the ``group_norm_g1_gelu`` CUDA kernel (custom-op wrapper).

        :param out: Output buffer, written in place.
        :param x: Input tensor.
        :param inject: Optional second input added elementwise before normalizat...
        :param weight: Affine weight.
        :param bias: Affine bias.
        :param C: Channel count of the reduction space.
        :param N: Spatial element count per batch element.
        :param eps: Variance epsilon.
        :param tgs: Block size (threads per block).
        """
        _get_kernel("group_norm_g1_gelu", x.dtype)(
            out, x, inject, weight, bias, C, N, eps, tgs
        )

    @torch.library.custom_op(f"{_OP_NAMESPACE}::apply_norm_gelu", mutates_args={"out"})
    def apply_norm_gelu(
        out: torch.Tensor,
        x: torch.Tensor,
        inject: torch.Tensor,
        meanvar: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        total_per_b: int,
        num_tiles: int,
        N: int,
        tgs: int,
    ) -> None:
        """
        Launch the ``apply_norm_gelu`` CUDA kernel (custom-op wrapper).

        :param out: Output buffer, written in place.
        :param x: Input tensor.
        :param inject: Optional second input added elementwise before normalizat...
        :param meanvar: FP32 ``(B, 2)`` mean/rsqrt(var+eps) buffer, written in pl...
        :param weight: Affine weight.
        :param bias: Affine bias.
        :param total_per_b: Elements reduced per batch element.
        :param num_tiles: Tile count for the multi-stage launches.
        :param N: Spatial element count per batch element.
        :param tgs: Block size (threads per block).
        """
        _get_kernel("apply_norm_gelu", x.dtype)(
            out, x, inject, meanvar, weight, bias, total_per_b, num_tiles, N, tgs
        )

    @torch.library.custom_op(
        f"{_OP_NAMESPACE}::group_norm_g1_glu", mutates_args={"out"}
    )
    def group_norm_g1_glu(
        out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        C: int,
        N: int,
        eps: float,
        tgs: int,
    ) -> None:
        """
        Launch the ``group_norm_g1_glu`` CUDA kernel (custom-op wrapper).

        :param out: Output buffer, written in place.
        :param x: Input tensor.
        :param weight: Affine weight.
        :param bias: Affine bias.
        :param C: Channel count of the reduction space.
        :param N: Spatial element count per batch element.
        :param eps: Variance epsilon.
        :param tgs: Block size (threads per block).
        """
        _get_kernel("group_norm_g1_glu", x.dtype)(out, x, weight, bias, C, N, eps, tgs)

    @torch.library.custom_op(f"{_OP_NAMESPACE}::apply_norm_glu", mutates_args={"out"})
    def apply_norm_glu(
        out: torch.Tensor,
        x: torch.Tensor,
        meanvar: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        total_in_per_b: int,
        total_out_per_b: int,
        num_tiles: int,
        N: int,
        C_half: int,
        tgs: int,
    ) -> None:
        """
        Launch the ``apply_norm_glu`` CUDA kernel (custom-op wrapper).

        :param out: Output buffer, written in place.
        :param x: Input tensor.
        :param meanvar: FP32 ``(B, 2)`` mean/rsqrt(var+eps) buffer, written in pl...
        :param weight: Affine weight.
        :param bias: Affine bias.
        :param total_in_per_b: Input-space elements per batch element.
        :param total_out_per_b: Output-space elements per batch element.
        :param num_tiles: Tile count for the multi-stage launches.
        :param N: Spatial element count per batch element.
        :param C_half: Half the GLU input channel count.
        :param tgs: Block size (threads per block).
        """
        _get_kernel("apply_norm_glu", x.dtype)(
            out,
            x,
            meanvar,
            weight,
            bias,
            total_in_per_b,
            total_out_per_b,
            num_tiles,
            N,
            C_half,
            tgs,
        )

    @torch.library.custom_op(
        f"{_OP_NAMESPACE}::norm_glu_ls_resid", mutates_args={"out"}
    )
    def norm_glu_ls_resid(
        out: torch.Tensor,
        z: torch.Tensor,
        residual: torch.Tensor,
        nweight: torch.Tensor,
        nbias: torch.Tensor,
        layer_scale: torch.Tensor,
        C2: int,
        N: int,
        eps: float,
        tgs: int,
    ) -> None:
        """
        Launch the ``norm_glu_ls_resid`` CUDA kernel (custom-op wrapper).

        :param out: Output buffer, written in place.
        :param z: GroupNorm/GLU input of ``2C`` channels.
        :param residual: Residual tensor added at the end.
        :param nweight: GroupNorm affine weight over the full ``2C`` input.
        :param nbias: GroupNorm affine bias over the full ``2C`` input.
        :param layer_scale: Per-output-channel LayerScale gains.
        :param C2: Full GLU input channel count (``2 * C``).
        :param N: Spatial element count per batch element.
        :param eps: Variance epsilon.
        :param tgs: Block size (threads per block).
        """
        _get_kernel("norm_glu_ls_resid", z.dtype)(
            out, z, residual, nweight, nbias, layer_scale, C2, N, eps, tgs
        )

    @torch.library.custom_op(
        f"{_OP_NAMESPACE}::apply_norm_glu_ls_resid", mutates_args={"out"}
    )
    def apply_norm_glu_ls_resid(
        out: torch.Tensor,
        z: torch.Tensor,
        residual: torch.Tensor,
        meanvar: torch.Tensor,
        nweight: torch.Tensor,
        nbias: torch.Tensor,
        layer_scale: torch.Tensor,
        total_in_per_b: int,
        total_out_per_b: int,
        num_tiles: int,
        N: int,
        C: int,
        tgs: int,
    ) -> None:
        """
        Launch the ``apply_norm_glu_ls_resid`` CUDA kernel (custom-op wrapper).

        :param out: Output buffer, written in place.
        :param z: GroupNorm/GLU input of ``2C`` channels.
        :param residual: Residual tensor added at the end.
        :param meanvar: FP32 ``(B, 2)`` mean/rsqrt(var+eps) buffer.
        :param nweight: GroupNorm affine weight over the full ``2C`` input.
        :param nbias: GroupNorm affine bias over the full ``2C`` input.
        :param layer_scale: Per-output-channel LayerScale gains.
        :param total_in_per_b: Input-space elements per batch element.
        :param total_out_per_b: Output-space elements per batch element.
        :param num_tiles: Tile count for the multi-stage launches.
        :param N: Spatial element count per batch element.
        :param C: Half the GLU input channel count (output channels).
        :param tgs: Block size (threads per block).
        """
        _get_kernel("apply_norm_glu_ls_resid", z.dtype)(
            out,
            z,
            residual,
            meanvar,
            nweight,
            nbias,
            layer_scale,
            total_in_per_b,
            total_out_per_b,
            num_tiles,
            N,
            C,
            tgs,
        )

    @torch.library.custom_op(
        f"{_OP_NAMESPACE}::group_norm_g1_chlast_gelu", mutates_args={"out"}
    )
    def group_norm_g1_chlast_gelu(
        out: torch.Tensor,
        x: torch.Tensor,
        inject: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        C: int,
        total: int,
        eps: float,
        tgs: int,
    ) -> None:
        """
        Launch the ``group_norm_g1_chlast_gelu`` CUDA kernel (custom-op wrapper).

        :param out: Output buffer, written in place.
        :param x: Input tensor in channel-last storage.
        :param inject: Optional second input added elementwise first.
        :param weight: Affine weight.
        :param bias: Affine bias.
        :param C: Channel count.
        :param total: Flat per-batch element count (``X * C``).
        :param eps: Variance epsilon.
        :param tgs: Block size (threads per block).
        """
        _get_kernel("group_norm_g1_chlast_gelu", x.dtype)(
            out, x, inject, weight, bias, C, total, eps, tgs
        )

    @torch.library.custom_op(
        f"{_OP_NAMESPACE}::apply_norm_chlast_gelu", mutates_args={"out"}
    )
    def apply_norm_chlast_gelu(
        out: torch.Tensor,
        x: torch.Tensor,
        inject: torch.Tensor,
        meanvar: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        total_per_b: int,
        num_tiles: int,
        C: int,
        tgs: int,
    ) -> None:
        """
        Launch the ``apply_norm_chlast_gelu`` CUDA kernel (custom-op wrapper).

        :param out: Output buffer, written in place.
        :param x: Input tensor in channel-last storage.
        :param inject: Optional second input added elementwise first.
        :param meanvar: FP32 ``(B, 2)`` mean/rsqrt(var+eps) buffer.
        :param weight: Affine weight.
        :param bias: Affine bias.
        :param total_per_b: Flat per-batch element count.
        :param num_tiles: Tile count for the multi-stage launches.
        :param C: Channel count.
        :param tgs: Block size (threads per block).
        """
        _get_kernel("apply_norm_chlast_gelu", x.dtype)(
            out, x, inject, meanvar, weight, bias, total_per_b, num_tiles, C, tgs
        )

    @torch.library.custom_op(
        f"{_OP_NAMESPACE}::group_norm_g1_chlast_glu", mutates_args={"out"}
    )
    def group_norm_g1_chlast_glu(
        out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        C2: int,
        X: int,
        eps: float,
        tgs: int,
    ) -> None:
        """
        Launch the ``group_norm_g1_chlast_glu`` CUDA kernel (custom-op wrapper).

        :param out: Output buffer, written in place.
        :param x: Input tensor with ``C2 = 2 * C`` channels in channel-last...
        :param weight: Affine weight over the full ``C2`` input channels.
        :param bias: Affine bias over the full ``C2`` input channels.
        :param C2: Input channel count (even).
        :param X: Spatial size.
        :param eps: Variance epsilon.
        :param tgs: Block size (threads per block).
        """
        _get_kernel("group_norm_g1_chlast_glu", x.dtype)(
            out, x, weight, bias, C2, X, eps, tgs
        )

    @torch.library.custom_op(
        f"{_OP_NAMESPACE}::apply_norm_chlast_glu", mutates_args={"out"}
    )
    def apply_norm_chlast_glu(
        out: torch.Tensor,
        x: torch.Tensor,
        meanvar: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        total_in_per_b: int,
        total_out_per_b: int,
        num_tiles: int,
        C: int,
        tgs: int,
    ) -> None:
        """
        Launch the ``apply_norm_chlast_glu`` CUDA kernel (custom-op wrapper).

        :param out: Output buffer, written in place.
        :param x: Input tensor with ``2 * C`` channels in channel-last stor...
        :param meanvar: FP32 ``(B, 2)`` mean/rsqrt(var+eps) buffer.
        :param weight: Affine weight over the input channels.
        :param bias: Affine bias over the input channels.
        :param total_in_per_b: Flat per-batch input element count.
        :param total_out_per_b: Flat per-batch output element count.
        :param num_tiles: Tile count for the multi-stage launches.
        :param C: Output channel count.
        :param tgs: Block size (threads per block).
        """
        _get_kernel("apply_norm_chlast_glu", x.dtype)(
            out,
            x,
            meanvar,
            weight,
            bias,
            total_in_per_b,
            total_out_per_b,
            num_tiles,
            C,
            tgs,
        )

    @torch.library.custom_op(
        f"{_OP_NAMESPACE}::norm_glu_ls_resid_chlast", mutates_args={"out"}
    )
    def norm_glu_ls_resid_chlast(
        out: torch.Tensor,
        z: torch.Tensor,
        resid: torch.Tensor,
        nweight: torch.Tensor,
        nbias: torch.Tensor,
        layer_scale: torch.Tensor,
        C2: int,
        X: int,
        eps: float,
        tgs: int,
    ) -> None:
        """
        Launch the ``norm_glu_ls_resid_chlast`` CUDA kernel (custom-op wrapper).

        :param out: Output buffer, written in place.
        :param z: GroupNorm/GLU input of ``C2`` channels, channel-last stor...
        :param resid: Residual tensor added at the end.
        :param nweight: GroupNorm affine weight over the ``C2`` input channels.
        :param nbias: GroupNorm affine bias over the ``C2`` input channels.
        :param layer_scale: Per-output-channel LayerScale gains.
        :param C2: Input channel count (even).
        :param X: Spatial size.
        :param eps: Variance epsilon.
        :param tgs: Block size (threads per block).
        """
        _get_kernel("norm_glu_ls_resid_chlast", z.dtype)(
            out, z, resid, nweight, nbias, layer_scale, C2, X, eps, tgs
        )

    @torch.library.custom_op(
        f"{_OP_NAMESPACE}::apply_norm_glu_ls_resid_chlast", mutates_args={"out"}
    )
    def apply_norm_glu_ls_resid_chlast(
        out: torch.Tensor,
        z: torch.Tensor,
        resid: torch.Tensor,
        meanvar: torch.Tensor,
        nweight: torch.Tensor,
        nbias: torch.Tensor,
        layer_scale: torch.Tensor,
        total_in_per_b: int,
        total_out_per_b: int,
        num_tiles: int,
        C: int,
        tgs: int,
    ) -> None:
        """
        Launch the ``apply_norm_glu_ls_resid_chlast`` CUDA kernel (custom-op wrapper).

        :param out: Output buffer, written in place.
        :param z: GroupNorm/GLU input of ``2 * C`` channels, channel-last s...
        :param resid: Residual tensor added at the end.
        :param meanvar: FP32 ``(B, 2)`` mean/rsqrt(var+eps) buffer.
        :param nweight: GroupNorm affine weight over the input channels.
        :param nbias: GroupNorm affine bias over the input channels.
        :param layer_scale: Per-output-channel LayerScale gains.
        :param total_in_per_b: Flat per-batch input element count.
        :param total_out_per_b: Flat per-batch output element count.
        :param num_tiles: Tile count for the multi-stage launches.
        :param C: Output channel count.
        :param tgs: Block size (threads per block).
        """
        _get_kernel("apply_norm_glu_ls_resid_chlast", z.dtype)(
            out,
            z,
            resid,
            meanvar,
            nweight,
            nbias,
            layer_scale,
            total_in_per_b,
            total_out_per_b,
            num_tiles,
            C,
            tgs,
        )

    @torch.library.custom_op(f"{_OP_NAMESPACE}::add_gelu", mutates_args={"out"})
    def add_gelu(
        out: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> None:
        """Launch the ``add_gelu`` CUDA kernel (custom-op wrapper).

        :param out: Output buffer, written in place.
        :param a: First addend.
        :param b: Second addend.
        :return: Nothing; ``out`` is written in place.
        """
        _get_kernel("add_gelu", a.dtype)(out, a, b)

    @torch.library.custom_op(f"{_OP_NAMESPACE}::rms_norm", mutates_args={"out"})
    def rms_norm(
        out: torch.Tensor,
        x: torch.Tensor,
        gamma: torch.Tensor,
        dim: int,
        scale: float,
        tgs: int,
    ) -> None:
        """
        Launch the ``rms_norm`` CUDA kernel (custom-op wrapper).

        :param out: Output buffer, written in place.
        :param x: Input tensor.
        :param gamma: RMSNorm gain.
        :param dim: RMSNorm feature dimension.
        :param scale: RoFormer's ``sqrt(dim)`` scale.
        :param tgs: Block size (threads per block).
        """
        _get_kernel("rms_norm", x.dtype)(out, x, gamma, dim, scale, tgs)

    _OPS_REGISTERED = True


_ensure_custom_ops()


def _launch(name: str, dtype: torch.dtype, *args: Any) -> None:
    """
    Launch a kernel, routing through the custom op under ``torch.compile``.

    :param name: Kernel name (a key of ``_KERNEL_SOURCES``).
    :param dtype: Tensor dtype of the first tensor argument.
    :param args: Full kernel argument list (tensors then scalars).
    """
    if torch.compiler.is_compiling():
        getattr(torch.ops.unblend_cuda, name)(*args)
    else:
        _get_kernel(name, dtype)(*args)


def _sm_count(device: torch.device) -> int:
    """
    Return the device's streaming-multiprocessor count, cached per index.

    :param device: CUDA device to query
    :return: ``multi_processor_count`` for the device
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    cached = _sm_count_cache.get(index)
    if cached is None:
        cached = torch.cuda.get_device_properties(index).multi_processor_count
        _sm_count_cache[index] = cached
    return cached


def _build_extension() -> Any:
    """
    Compile the CUDA kernel extension with ``torch.utils.cpp_extension``.

    :return: The compiled extension module exposing one function per k...
    """
    from torch.utils import cpp_extension

    sources: list[str] = []
    with resources.as_file(resources.files(__name__)) as pkg_dir:
        extra_include_paths = [str(pkg_dir)]
        for source_name in _SOURCE_FILES:
            sources.append(str(pkg_dir / source_name))
        sources.append(str(pkg_dir / "bindings.cpp"))

    suffix = ""
    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        major, minor = torch.cuda.get_device_capability(
            torch.device("cuda", torch.cuda.current_device())
        )
        suffix = f"_sm{major}{minor}"
    return cpp_extension.load(
        name="unblend_cuda_kernels" + suffix,
        sources=sources,
        extra_include_paths=extra_include_paths,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def _notify_build_start() -> None:
    """
    Warn once that a blocking nvcc build is starting.
    """

    global _build_notice_sent
    if _build_notice_sent:
        return
    _build_notice_sent = True
    warnings.warn(
        "Compiling unblend CUDA kernels (~2 min, once per GPU architecture; "
        "cached under TORCH_EXTENSIONS_DIR afterwards). Pass "
        "custom_kernels=False (or set UNBLEND_CUSTOM_KERNELS=0) to skip "
        "native kernels entirely.",
        stacklevel=2,
    )


def _get_extension() -> Any:
    """
    Return the compiled kernel extension, building it on first use.

    :return: The extension module.
    """
    global _extension, _extension_error
    with _extension_lock:
        if _extension is not None:
            return _extension
        if _extension_error is not None:
            raise RuntimeError(_extension_error)
        _notify_build_start()
        try:
            _extension = _build_extension()
        except Exception as exc:
            _extension_error = f"CUDA kernel compilation failed: {exc}"
            raise RuntimeError(_extension_error) from exc
        return _extension


def warmup_async() -> None:
    """
    Start building the kernel extension in a daemon thread, if needed.

    """
    global _warmup_thread
    if not torch.cuda.is_available():
        return
    with _extension_lock:
        if _extension is not None or _extension_error is not None:
            return
        if _warmup_thread is not None and _warmup_thread.is_alive():
            return
        _warmup_thread = threading.Thread(
            target=_get_extension,
            name="unblend-cuda-kernel-build",
            daemon=True,
        )
        _warmup_thread.start()


def _get_kernel(name: str, dtype: torch.dtype) -> Any:
    """
    Look up a CUDA kernel binding by ``(name, dtype)``.

    :param name: Kernel function name (a key of ``_KERNEL_SOURCES``).
    :param dtype: Scalar dtype the caller will run in (unused; see above).
    :return: The callable binding for ``name``.
    """
    ext = _get_extension()
    fn = getattr(ext, name, None)
    if fn is None:
        raise KeyError(
            f"Unknown CUDA kernel {name!r}; expected one of {sorted(_KERNEL_SOURCES)}"
        )
    return fn


def _max_threads(device: torch.device) -> int:
    """
    Return the device's maximum threads per block, cached per device index.

    :param device: CUDA device to query
    :return: ``max_threads_per_block`` for the device (1024 on all current
        hardware)
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    cached = _max_threads_cache.get(index)
    if cached is None:
        cached = torch.cuda.get_device_properties(index).max_threads_per_block
        _max_threads_cache[index] = cached
    return cached


def _is_cuda_lp(t: torch.Tensor) -> bool:
    """
    Report whether a tensor is on CUDA in a kernel-supported low-precision dtype.

    :param t: Tensor whose device and dtype are checked.
    :return: ``True`` if ``t`` is on CUDA and FP16/BF16 under inferenc...
    """
    return (
        t.device.type == "cuda"
        and t.dtype in _LP_DTYPES
        and not torch.is_grad_enabled()
    )


def _is_chlast_4d(t: torch.Tensor) -> bool:
    """
    Report whether a rank-4 tensor is stored in channels_last (NHWC) layout.

    :param t: Tensor to inspect.
    :return: True when ``t`` is rank-4 and contiguous in channels_last...
    """
    return (
        t.dim() == 4
        and not t.is_contiguous()
        and t.is_contiguous(memory_format=torch.channels_last)
    )


def _kernel_arg(t: torch.Tensor) -> torch.Tensor:
    """
    Prepare a tensor for kernel dispatch: dense in a supported layout with a 4-element-aligned storage offset.

    :param t: Tensor to prepare.
    :return: ``t`` itself if already safe, else an aligned dense copy.
    """
    if t.is_contiguous() or (
        t.dim() == 4 and t.is_contiguous(memory_format=torch.channels_last)
    ):
        return t if t.storage_offset() % 4 == 0 else t.clone()
    return t.contiguous()


def cuda_rms_norm(x: torch.Tensor, gamma: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Apply RoFormer's last-dimension RMSNorm with one fused CUDA kernel.

    :param x: Input tensor normalized over its final dimension.
    :param gamma: Learnable gain with length ``x.shape[-1]``.
    :param scale: RoFormer's ``sqrt(dim)`` normalization scale.
    :return: Normalized tensor with the same shape and dtype as ``x``.
    """
    if (
        x.device.type != "cuda"
        or x.dtype not in _RMS_DTYPES
        or x.numel() == 0
        or torch.is_grad_enabled()
    ):
        normalized = F.normalize(x.float(), dim=-1) * scale * gamma.float()
        return normalized.type(x.dtype)

    x_contig = x.contiguous()
    dim = x_contig.shape[-1]
    gamma_contig = gamma.to(device=x.device, dtype=x.dtype).contiguous()
    out = torch.empty_like(x_contig)

    try:
        _get_kernel("rms_norm", x.dtype)
    except RuntimeError as exc:
        warnings.warn(
            f"{exc}; falling back to native PyTorch RMSNorm.",
            RuntimeWarning,
        )
        normalized = F.normalize(x.float(), dim=-1) * scale * gamma.float()
        return normalized.type(x.dtype)

    tgs = _pow2_tgs(_max_threads(x.device))
    while tgs > dim:
        tgs //= 2
    _launch("rms_norm", x.dtype, out, x_contig, gamma_contig, dim, float(scale), tgs)
    return out.view_as(x)


class CUDAGroupNorm(nn.Module):
    """
    Replacement for ``nn.GroupNorm(num_groups=1)`` on CUDA in FP16/BF16.
    """

    _SINGLE_STAGE_LIMIT = 1_500_000

    _SINGLE_STAGE_MIN_BATCH_FACTOR = 4

    _SINGLE_STAGE_SMALL_PER_BATCH = 49_152

    _MULTI_STAGE_TILE_SIZE = 16_384
    _MULTI_STAGE_MAX_TILES = 4096

    @classmethod
    def _use_single_stage(cls, batch: int, per_batch: int) -> bool:
        """
        Decide between the single-stage and multi-stage kernel paths.

        :param batch: Number of batch elements (blocks a single-stage
            launch would fire)
        :param per_batch: Elements reduced per batch element (input space)
        :return: ``True`` to run the fused single-stage kernel
        """
        if per_batch > cls._SINGLE_STAGE_LIMIT:
            return False
        if per_batch <= cls._SINGLE_STAGE_SMALL_PER_BATCH:
            return True
        min_batch = cls._SINGLE_STAGE_MIN_BATCH_FACTOR * _sm_count(
            torch.device("cuda", torch.cuda.current_device())
        )
        return batch >= min_batch

    def _multi_stage_num_tiles(self, tile_space: int, B: int) -> int:
        """
        Size the multi-stage tiling so stages 1/3 saturate the GPU.

        :param tile_space: Element count ``num_tiles`` is sized against — the.
        :param B: Number of batch elements participating in the launch.
        :return: The power-of-two tile count to launch with.
        """
        num_tiles = min(
            self._MULTI_STAGE_MAX_TILES,
            max(
                1,
                (tile_space + self._MULTI_STAGE_TILE_SIZE - 1)
                // self._MULTI_STAGE_TILE_SIZE,
            ),
        )

        min_tiles = max(
            1,
            (4 * _sm_count(torch.device("cuda", torch.cuda.current_device())))
            // max(B, 1),
        )
        num_tiles = max(num_tiles, min_tiles)
        pow2 = 1
        while pow2 * 2 <= num_tiles:
            pow2 *= 2
        return pow2

    def _multi_stage_meanvar(
        self,
        x_contig: torch.Tensor,
        B: int,
        per_batch_in: int,
        tile_space: int,
        inject: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """
        Run multi-stage stages 1+2: per-tile partial reduce, then finalize per-batch ``(mean, rsqrt(var+eps))``.

        :param x_contig: Contiguous kernel-ready input, ``(B, per_batch_in)`` flat.
        :param B: Number of batch elements.
        :param per_batch_in: Elements reduced per batch element.
        :param tile_space: Element count ``num_tiles`` is sized against — the.
        :param inject: Optional second input added elementwise before the.
        :return: The ``(B, 2)`` FP32 meanvar buffer and ``num_tiles``.
        """
        num_tiles = self._multi_stage_num_tiles(tile_space, B)

        dtype = x_contig.dtype
        scratch = torch.empty(
            (B, num_tiles, 2), dtype=torch.float32, device=x_contig.device
        )
        meanvar = torch.empty((B, 2), dtype=torch.float32, device=x_contig.device)

        max_threads = _max_threads(x_contig.device)
        tgs1 = _pow2_tgs(max_threads)

        tgs2 = min(num_tiles, max_threads)
        pow2 = 1
        while pow2 * 2 <= tgs2:
            pow2 *= 2
        tgs2 = pow2

        inj_arg = (
            torch.empty(0, dtype=dtype, device=x_contig.device)
            if inject is None
            else _kernel_arg(inject)
        )
        _launch(
            "partial_reduce",
            dtype,
            x_contig,
            inj_arg,
            scratch,
            per_batch_in,
            num_tiles,
            tgs1,
        )
        _launch(
            "finalize_meanvar",
            dtype,
            scratch,
            meanvar,
            per_batch_in,
            num_tiles,
            float(self.eps),
            x_contig,
            inj_arg,
            tgs2,
        )
        return meanvar, num_tiles

    def __init__(self, gn: nn.GroupNorm) -> None:
        """
        Wrap a ``num_groups=1`` affine GroupNorm, snapshotting its affine params in FP32.

        :param gn: Source GroupNorm to replace; must have ``num_groups=1`` and ``affine=True``
        :raises ValueError: If ``gn`` has ``num_groups != 1`` or is not affine
        """
        super().__init__()
        if gn.num_groups != 1:
            raise ValueError(
                f"CUDAGroupNorm only supports num_groups=1; got {gn.num_groups}"
            )
        if not gn.affine:
            raise ValueError("CUDAGroupNorm requires affine=True")
        self.num_channels = gn.num_channels
        self.eps = gn.eps

        self.weight = nn.Parameter(gn.weight.detach().to(torch.float32).clone())
        self.bias = nn.Parameter(gn.bias.detach().to(torch.float32).clone())
        self.train(gn.training)

    @classmethod
    def from_groupnorm(cls, gn: nn.GroupNorm) -> "CUDAGroupNorm":
        """
        Build a :class:`CUDAGroupNorm` from an existing GroupNorm.

        :param gn: Source GroupNorm to wrap
        :return: A new :class:`CUDAGroupNorm` mirroring ``gn``
        """
        return cls(gn)

    def _lp_affine(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the affine weight/bias cast to the given dtype/device, cached per key.

        :param dtype: Target dtype for the cast affine parameters
        :param device: Target device for the cast affine parameters
        :return: The ``(weight, bias)`` tensors as contiguous ``dtype``/``device`` copies
        """
        cache = getattr(self, "_aff_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_aff_cache", cache)

        if not torch.compiler.is_compiling():
            versions = (
                id(self.weight),
                self.weight._version,
                id(self.bias),
                self.bias._version,
            )
            if getattr(self, "_aff_versions", None) != versions:
                cache.clear()
                object.__setattr__(self, "_aff_versions", versions)
        key = (dtype, device)
        cached = cache.get(key)
        if cached is None:
            w = self.weight.detach().to(device=device, dtype=dtype).contiguous()
            b = self.bias.detach().to(device=device, dtype=dtype).contiguous()
            cache[key] = (w, b)
            return w, b
        return cached

    def _clear_parameter_caches(self) -> None:
        """
        Discard all derived low-precision parameter copies.
        """
        for name in ("_aff_cache", "_ls_cache"):
            cache = getattr(self, name, None)
            if cache:
                cache.clear()
        for name in ("_aff_versions", "_ls_version"):
            if hasattr(self, name):
                object.__delattr__(self, name)

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> "CUDAGroupNorm":
        """
        Apply a module transform and invalidate device/dtype-specific caches.

        :param fn: Tensor transform supplied by ``nn.Module.to``/``half``.
        :param recurse: Whether to transform child modules recursively.
        :return: This module after the transform completes.
        """
        result = super()._apply(fn, recurse=recurse)
        self._clear_parameter_caches()
        return result

    def _load_from_state_dict(self, *args: object, **kwargs: object) -> None:
        """
        Reload parameters and invalidate the lazily-cast affine/LayerScale caches.

        :param args: Positional arguments forwarded to ``nn.Module._load_from_...
        :param kwargs: Keyword arguments forwarded to ``nn.Module._load_from_sta...
        """
        super()._load_from_state_dict(*args, **kwargs)
        self._clear_parameter_caches()

    def forward(self, x: torch.Tensor, gelu: bool = False) -> torch.Tensor:
        """
        Apply ``num_groups=1`` group normalization, using a fused CUDA kernel on FP16/BF16.

        :param x: Input tensor of shape ``(B, C, ...)``.
        :param gelu: Also apply GELU to the normalized output in the same.
        :return: Normalized, affine-transformed tensor with the same shape...
        """

        if not _is_cuda_lp(x):
            if x.dtype == torch.float32:
                y = F.group_norm(x, 1, self.weight, self.bias, self.eps)
                return F.gelu(y) if gelu else y
            y = F.group_norm(x.to(torch.float32), 1, self.weight, self.bias, self.eps)
            return F.gelu(y).to(x.dtype) if gelu else y.to(x.dtype)

        x_contig = _kernel_arg(x)
        B = x_contig.shape[0]
        C = x_contig.shape[1]
        N = 1
        for d in x_contig.shape[2:]:
            N *= d
        per_batch = C * N

        weight, bias = self._lp_affine(x.dtype, x.device)
        max_threads = _max_threads(x.device)

        suffix = "_chlast" if _is_chlast_4d(x_contig) else ""
        if self._use_single_stage(B, per_batch):
            tgs = _pow2_tgs(max_threads)
            while tgs > 1 and tgs > per_batch:
                tgs //= 2
            out = torch.empty_like(x_contig)
            if gelu:
                _launch(
                    f"group_norm_g1{suffix}_gelu",
                    x.dtype,
                    out,
                    x_contig,
                    torch.empty(0, dtype=x.dtype, device=x.device),
                    weight,
                    bias,
                    C,
                    per_batch if suffix else N,
                    float(self.eps),
                    tgs,
                )
            else:
                name = f"group_norm_g1{suffix}"
                if suffix:
                    _launch(
                        name,
                        x.dtype,
                        out,
                        x_contig,
                        weight,
                        bias,
                        C,
                        per_batch,
                        float(self.eps),
                        tgs,
                    )
                else:
                    _launch(
                        name,
                        x.dtype,
                        out,
                        x_contig,
                        weight,
                        bias,
                        C,
                        N,
                        float(self.eps),
                        tgs,
                    )
            return out.view_as(x)

        meanvar, num_tiles = self._multi_stage_meanvar(
            x_contig, B, per_batch, per_batch
        )
        out = torch.empty_like(x_contig)
        tgs3 = _pow2_tgs(max_threads)
        if gelu:
            _launch(
                f"apply_norm{suffix}_gelu",
                x.dtype,
                out,
                x_contig,
                torch.empty(0, dtype=x.dtype, device=x.device),
                meanvar,
                weight,
                bias,
                per_batch,
                num_tiles,
                C if suffix else N,
                tgs3,
            )
        else:
            name = f"apply_norm{suffix}"
            if suffix:
                _launch(
                    name,
                    x.dtype,
                    out,
                    x_contig,
                    meanvar,
                    weight,
                    bias,
                    per_batch,
                    num_tiles,
                    C,
                    tgs3,
                )
            else:
                _launch(
                    name,
                    x.dtype,
                    out,
                    x_contig,
                    meanvar,
                    weight,
                    bias,
                    per_batch,
                    num_tiles,
                    N,
                    tgs3,
                )
        return out.view_as(x)


class FusedGroupNormGelu(CUDAGroupNorm):
    """
    Drop-in for the ``gelu(group_norm(...))`` pattern.
    """

    def forward(
        self, x: torch.Tensor, inject: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Apply ``gelu(group_norm(x + inject))`` fused into one CUDA kernel on FP16/BF16.

        :param x: Input tensor of shape ``(B, C, ...)``.
        :param inject: Optional second input added elementwise before the.
        :return: GELU-activated normalized tensor with the same shape as `...
        """

        if not _is_cuda_lp(x):
            if inject is not None:
                x = x + inject
            if x.dtype == torch.float32:
                return F.gelu(
                    F.group_norm(x, 1, self.weight, self.bias, self.eps),
                )
            return F.gelu(
                F.group_norm(x.to(torch.float32), 1, self.weight, self.bias, self.eps),
            ).to(x.dtype)

        x_contig = _kernel_arg(x)
        inj_contig = _kernel_arg(inject) if inject is not None else None
        B = x_contig.shape[0]
        C = x_contig.shape[1]
        N = 1
        for d in x_contig.shape[2:]:
            N *= d
        per_batch = C * N
        weight, bias = self._lp_affine(x.dtype, x.device)
        max_threads = _max_threads(x.device)

        suffix = "_chlast" if _is_chlast_4d(x_contig) else ""
        if self._use_single_stage(B, per_batch):
            tgs = _pow2_tgs(max_threads)
            while tgs > 1 and tgs > per_batch:
                tgs //= 2
            out = torch.empty_like(x_contig)
            _launch(
                f"group_norm_g1{suffix}_gelu",
                x.dtype,
                out,
                x_contig,
                inj_contig
                if inj_contig is not None
                else torch.empty(0, dtype=x.dtype, device=x.device),
                weight,
                bias,
                C,
                per_batch if suffix else N,
                float(self.eps),
                tgs,
            )
            return out.view_as(x)

        meanvar, num_tiles = self._multi_stage_meanvar(
            x_contig, B, per_batch, per_batch, inject=inj_contig
        )
        out = torch.empty_like(x_contig)
        tgs3 = _pow2_tgs(max_threads)
        suffix = "_chlast" if _is_chlast_4d(x_contig) else ""
        _launch(
            f"apply_norm{suffix}_gelu",
            x.dtype,
            out,
            x_contig,
            inj_contig
            if inj_contig is not None
            else torch.empty(0, dtype=x.dtype, device=x.device),
            meanvar,
            weight,
            bias,
            per_batch,
            num_tiles,
            C if suffix else N,
            tgs3,
        )
        return out.view_as(x)


class FusedGroupNormGlu(CUDAGroupNorm):
    """
    Drop-in for the ``glu(group_norm(rewrite(...)), dim=1)`` pattern.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ``glu(group_norm(x), dim=1)`` fused into one CUDA kernel on FP16/BF16.

        :param x: Input tensor of shape ``(B, 2C, ...)`` with even channel count
        :return: GLU-gated normalized tensor of shape ``(B, C, ...)``
        :raises ValueError: If the input channel dimension is not even
        """

        if not _is_cuda_lp(x):
            if x.dtype == torch.float32:
                return F.glu(
                    F.group_norm(x, 1, self.weight, self.bias, self.eps),
                    dim=1,
                )
            return F.glu(
                F.group_norm(x.to(torch.float32), 1, self.weight, self.bias, self.eps),
                dim=1,
            ).to(x.dtype)

        x_contig = _kernel_arg(x)
        B = x_contig.shape[0]
        C_in = x_contig.shape[1]
        if C_in % 2 != 0:
            raise ValueError(
                f"FusedGroupNormGlu requires even input channels; got {C_in}"
            )
        C_half = C_in // 2
        N = 1
        for d in x_contig.shape[2:]:
            N *= d
        per_batch_in = C_in * N
        per_batch_out = C_half * N
        weight, bias = self._lp_affine(x.dtype, x.device)
        max_threads = _max_threads(x.device)

        suffix = "_chlast" if _is_chlast_4d(x_contig) else ""
        if self._use_single_stage(B, per_batch_in):
            tgs = _pow2_tgs(max_threads)
            while tgs > 1 and tgs > per_batch_out:
                tgs //= 2
            out_shape = (B, C_half) + tuple(x_contig.shape[2:])

            fmt = dict(memory_format=torch.channels_last) if suffix else {}
            out = torch.empty(out_shape, dtype=x.dtype, device=x.device, **fmt)
            if suffix:
                _launch(
                    "group_norm_g1_chlast_glu",
                    x.dtype,
                    out,
                    x_contig,
                    weight,
                    bias,
                    C_in,
                    N,
                    float(self.eps),
                    tgs,
                )
            else:
                _launch(
                    "group_norm_g1_glu",
                    x.dtype,
                    out,
                    x_contig,
                    weight,
                    bias,
                    C_in,
                    N,
                    float(self.eps),
                    tgs,
                )
            return out

        meanvar, num_tiles = self._multi_stage_meanvar(
            x_contig, B, per_batch_in, per_batch_out
        )
        out_shape = (B, C_half) + tuple(x_contig.shape[2:])

        fmt = dict(memory_format=torch.channels_last) if suffix else {}
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device, **fmt)
        tgs3 = _pow2_tgs(max_threads)

        if suffix:
            _launch(
                "apply_norm_chlast_glu",
                x.dtype,
                out,
                x_contig,
                meanvar,
                weight,
                bias,
                per_batch_in,
                per_batch_out,
                num_tiles,
                C_half,
                tgs3,
            )
        else:
            _launch(
                "apply_norm_glu",
                x.dtype,
                out,
                x_contig,
                meanvar,
                weight,
                bias,
                per_batch_in,
                per_batch_out,
                num_tiles,
                N,
                C_half,
                tgs3,
            )
        return out


class FusedNormGluLayerScaleResid(CUDAGroupNorm):
    """
    Single fused op for the DConv envelope after the second conv: ``residual + layer_scale * glu(group_norm(z), dim=1)``.
    """

    def __init__(
        self,
        gn: nn.GroupNorm,
        layer_scale_param: torch.Tensor,
    ) -> None:
        """
        Wrap a GroupNorm and snapshot the LayerScale param for the fused envelope op.

        :param gn: Source GroupNorm to replace; must have ``num_groups=1`` and ``affine=True``
        :param layer_scale_param: Per-channel LayerScale tensor of shape ``(C,)``
        :raises ValueError: If ``gn`` has ``num_groups != 1`` or is not affine
        """
        super().__init__(gn)
        self.layer_scale = nn.Parameter(
            layer_scale_param.detach().to(torch.float32).clone()
        )

    @classmethod
    def from_groupnorm_and_scale(
        cls,
        gn: nn.GroupNorm,
        layer_scale_param: torch.Tensor,
    ) -> "FusedNormGluLayerScaleResid":
        """
        Build a :class:`FusedNormGluLayerScaleResid` from a GroupNorm and LayerScale param.

        :param gn: Source GroupNorm to wrap
        :param layer_scale_param: Per-channel LayerScale tensor of shape ``(C,)``
        :return: A new :class:`FusedNormGluLayerScaleResid` combining both
        """
        return cls(gn, layer_scale_param)

    def _lp_layer_scale(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Return the LayerScale tensor cast to the given dtype/device, cached per key.

        :param dtype: Target dtype for the cast LayerScale
        :param device: Target device for the cast LayerScale
        :return: The LayerScale as a contiguous ``dtype``/``device`` copy
        """
        cache = getattr(self, "_ls_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_ls_cache", cache)

        if not torch.compiler.is_compiling():
            version = (id(self.layer_scale), self.layer_scale._version)
            if getattr(self, "_ls_version", None) != version:
                cache.clear()
                object.__setattr__(self, "_ls_version", version)
        key = (dtype, device)
        cached = cache.get(key)
        if cached is None:
            t = self.layer_scale.detach().to(device=device, dtype=dtype).contiguous()
            cache[key] = t
            return t
        return cached

    def forward(self, z: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Compute ``residual + layer_scale * glu(group_norm(z), dim=1)`` in one CUDA kernel.

        :param z: GroupNorm/GLU input of shape ``(B, 2C, ...)`` with even channel count
        :param residual: Residual tensor of shape ``(B, C, ...)`` to add
        :return: The fused result of shape ``(B, C, ...)``
        :raises ValueError: If the GLU input channel dimension is not even
        """

        if not _is_cuda_lp(z):
            if z.dtype == torch.float32:
                zn = F.group_norm(z, 1, self.weight, self.bias, self.eps)
                return residual + self.layer_scale[:, None] * F.glu(zn, dim=1)
            zn = F.group_norm(z.to(torch.float32), 1, self.weight, self.bias, self.eps)
            out = residual.to(torch.float32) + self.layer_scale[:, None] * F.glu(
                zn, dim=1
            )
            return out.to(z.dtype)

        z_c = _kernel_arg(z)
        r_c = _kernel_arg(residual)
        B = z_c.shape[0]
        C2 = z_c.shape[1]
        if C2 % 2 != 0:
            raise ValueError("GLU input channel dim must be even")
        C = C2 // 2
        N = 1
        for d in z_c.shape[2:]:
            N *= d
        per_batch_in = C2 * N
        per_batch_out = C * N
        weight, bias = self._lp_affine(z.dtype, z.device)
        ls = self._lp_layer_scale(z.dtype, z.device)
        out_shape = (B, C) + tuple(z_c.shape[2:])
        max_threads = _max_threads(z.device)

        suffix = "_chlast" if _is_chlast_4d(z_c) else ""

        fmt = dict(memory_format=torch.channels_last) if suffix else {}
        out = torch.empty(out_shape, dtype=z.dtype, device=z.device, **fmt)
        if self._use_single_stage(B, per_batch_in):
            tgs = _pow2_tgs(max_threads)

            while tgs > 1 and tgs > per_batch_out:
                tgs //= 2
            _launch(
                f"norm_glu_ls_resid{suffix}",
                z.dtype,
                out,
                z_c,
                r_c,
                weight,
                bias,
                ls,
                C2,
                N,
                float(self.eps),
                tgs,
            )
            return out

        meanvar, num_tiles = self._multi_stage_meanvar(
            z_c, B, per_batch_in, per_batch_out
        )
        tgs3 = _pow2_tgs(max_threads)

        if suffix:
            _launch(
                "apply_norm_glu_ls_resid_chlast",
                z.dtype,
                out,
                z_c,
                r_c,
                meanvar,
                weight,
                bias,
                ls,
                per_batch_in,
                per_batch_out,
                num_tiles,
                C,
                tgs3,
            )
        else:
            _launch(
                "apply_norm_glu_ls_resid",
                z.dtype,
                out,
                z_c,
                r_c,
                meanvar,
                weight,
                bias,
                ls,
                per_batch_in,
                per_batch_out,
                num_tiles,
                N,
                C,
                tgs3,
            )
        return out


class CUDAMyGroupNorm(CUDAGroupNorm):
    """
    Replacement for ``unblend.transformer.MyGroupNorm`` on CUDA in FP16/BF16.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize a ``(B, T, C)`` tensor per batch element without transposing.

        :param x: Input tensor of shape ``(B, T, C)``
        :return: Normalized, affine-transformed tensor with the same shape as ``x``
        """

        if x.dtype == torch.float32:
            x = x.transpose(1, 2)
            x = F.group_norm(x, 1, self.weight, self.bias, self.eps)
            return x.transpose(1, 2)

        if not _is_cuda_lp(x):
            x_t = x.transpose(1, 2)
            return (
                F.group_norm(x_t.to(torch.float32), 1, self.weight, self.bias, self.eps)
                .to(x.dtype)
                .transpose(1, 2)
            )

        x_contig = _kernel_arg(x)
        B = x_contig.shape[0]
        C = x_contig.shape[-1]
        per_batch = 1
        for d in x_contig.shape[1:]:
            per_batch *= d
        weight, bias = self._lp_affine(x.dtype, x.device)
        max_threads = _max_threads(x.device)

        if self._use_single_stage(B, per_batch):
            tgs = _pow2_tgs(max_threads)
            while tgs > 1 and tgs > per_batch:
                tgs //= 2
            out = torch.empty_like(x_contig)
            _launch(
                "group_norm_g1_chlast",
                x.dtype,
                out,
                x_contig,
                weight,
                bias,
                C,
                per_batch,
                float(self.eps),
                tgs,
            )
            return out.view_as(x)

        meanvar, num_tiles = self._multi_stage_meanvar(
            x_contig, B, per_batch, per_batch
        )
        out = torch.empty_like(x_contig)
        tgs3 = _pow2_tgs(max_threads)
        _launch(
            "apply_norm_chlast",
            x.dtype,
            out,
            x_contig,
            meanvar,
            weight,
            bias,
            per_batch,
            num_tiles,
            C,
            tgs3,
        )
        return out.view_as(x)


class FusedDConvLayer(nn.Module):
    """
    One DConv sub-layer (formerly an ``nn.Sequential`` of 7 ops) folded into 4 calls: ``conv1 → fused_norm_gelu → conv2 →...
    """

    def __init__(
        self,
        conv1: nn.Conv1d,
        norm1: nn.GroupNorm,
        conv2: nn.Conv1d,
        norm2: nn.GroupNorm,
        layer_scale_param: torch.Tensor,
    ) -> None:
        """
        Build a fused DConv sub-layer from its constituent convs/norms/scale.

        :param conv1: First pointwise convolution (``C -> hidden``).
        :param norm1: GroupNorm following ``conv1``; fused with the GELU.
        :param conv2: Second pointwise convolution (``hidden -> 2C``).
        :param norm2: GroupNorm following ``conv2``; fused into the envelope op.
        :param layer_scale_param: Per-channel LayerScale tensor of shape ``(C,)``.
        """
        super().__init__()
        self.conv1 = conv1
        self.norm1_gelu = FusedGroupNormGelu.from_groupnorm(norm1)
        self.conv2 = conv2
        self.norm2_envelope = FusedNormGluLayerScaleResid.from_groupnorm_and_scale(
            norm2, layer_scale_param
        )

    @classmethod
    def from_sequential(cls, seq: nn.Sequential) -> "FusedDConvLayer":
        """
        Build from the standard 7-op DConv ``nn.Sequential``.

        :param seq: The 7-op DConv ``nn.Sequential`` to fold.
        :return: A new :class:`FusedDConvLayer` mirroring ``seq``.
        """

        from ..transformer import LayerScale

        if len(seq) != 7:
            raise ValueError(f"expected 7-op DConv sequential, got {len(seq)}")
        conv1, norm1, act, conv2, norm2, glu, layer_scale = list(seq)
        if not isinstance(conv1, nn.Conv1d):
            raise TypeError("seq[0] must be nn.Conv1d")
        if not isinstance(norm1, nn.GroupNorm) or norm1.num_groups != 1:
            raise TypeError("seq[1] must be GroupNorm(num_groups=1)")
        if not isinstance(act, nn.GELU):
            raise TypeError("seq[2] must be nn.GELU")
        if not isinstance(conv2, nn.Conv1d):
            raise TypeError("seq[3] must be nn.Conv1d")
        if not isinstance(norm2, nn.GroupNorm) or norm2.num_groups != 1:
            raise TypeError("seq[4] must be GroupNorm(num_groups=1)")
        if not isinstance(glu, nn.GLU) or glu.dim != 1:
            raise TypeError("seq[5] must be nn.GLU(dim=1)")
        if not isinstance(layer_scale, LayerScale):
            raise TypeError("seq[6] must be LayerScale")
        return cls(conv1, norm1, conv2, norm2, layer_scale.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run ``conv1 -> fused_norm_gelu -> conv2 -> fused_norm_glu_ls_resid`` on ``x``.

        :param x: Input tensor of shape ``(B, C, T)``
        :return: Output tensor of shape ``(B, C, T)`` with the residual absorbed
        """
        h = self.conv1(x)
        h = self.norm1_gelu(h)
        h = self.conv2(h)
        return self.norm2_envelope(h, x)


class FusedDConv(nn.Module):
    """Drop-in for ``unblend.blocks.DConv`` whose layers are
    :class:`FusedDConvLayer`. Each layer already absorbs the residual add,
    so the outer loop just chains them.
    """

    def __init__(self, fused_layers: list[FusedDConvLayer]) -> None:
        """
        Hold the fused DConv sub-layers in order.

        :param fused_layers: The :class:`FusedDConvLayer` instances to chain
        """
        super().__init__()
        self.layers = nn.ModuleList(fused_layers)

    @classmethod
    def from_dconv(cls, dconv: nn.Module) -> "FusedDConv":
        """
        Build a :class:`FusedDConv` from a ``unblend.blocks.DConv``.

        :param dconv: Source DConv whose sub-sequentials are folded
        :return: A new :class:`FusedDConv` with one fused layer per sequential
        :raises TypeError: If ``dconv`` is not a ``DConv`` instance
        """

        from ..blocks import DConv

        if not isinstance(dconv, DConv):
            raise TypeError(f"expected DConv, got {type(dconv).__name__}")
        return cls([FusedDConvLayer.from_sequential(seq) for seq in dconv.layers])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Chain the fused DConv sub-layers, each absorbing its own residual add.

        :param x: Input tensor of shape ``(B, C, T)``
        :return: Output tensor of shape ``(B, C, T)``
        """
        for layer in self.layers:
            x = layer(x)
        return x


class FusedHEncLayer(nn.Module):
    """Replacement for ``unblend.blocks.HEncLayer`` that uses fused CUDA
    kernels for low-precision (FP16/BF16) inference. Same forward contract.

    We keep ``self.conv``, ``self.rewrite``, and the layer's ``empty`` /
    ``stride`` / ``freq`` / ``pad`` flags as on the original. The
    GroupNorms and the surrounding ``gelu``/``glu`` are folded into single
    fused calls; the inner DConv (if present) is replaced with FusedDConv.
    """

    def __init__(self, layer: nn.Module) -> None:
        """
        Build a fused encoder layer from an ``HEncLayer``, folding its norms/activations.

        :param layer: Source ``HEncLayer`` whose conv, rewrite, norms, flags, and
            inner DConv are carried forward or replaced with fused equivalents
        """
        super().__init__()
        from ..blocks import DConv

        self.kernel_size = layer.kernel_size
        self.stride = layer.stride
        self.empty = layer.empty
        self.freq = layer.freq
        self.norm = layer.norm
        self.pad = layer.pad

        if isinstance(layer.norm1, nn.GroupNorm) and layer.norm1.num_groups == 1:
            self.norm1 = FusedGroupNormGelu.from_groupnorm(layer.norm1)
            self._fused_gelu = True
        else:
            self.norm1 = layer.norm1
            self._fused_gelu = False

        self.conv = layer.conv
        if layer.empty:
            return

        self.rewrite = layer.rewrite
        if layer.rewrite is not None:
            if isinstance(layer.norm2, nn.GroupNorm) and layer.norm2.num_groups == 1:
                self.norm2 = FusedGroupNormGlu.from_groupnorm(layer.norm2)
                self._fused_glu = True
            else:
                self.norm2 = layer.norm2
                self._fused_glu = False
        else:
            self.norm2 = None
            self._fused_glu = False

        if layer.dconv is not None and isinstance(layer.dconv, DConv):
            try:
                self.dconv = FusedDConv.from_dconv(layer.dconv)
            except (TypeError, ValueError):
                self.dconv = layer.dconv
        else:
            self.dconv = layer.dconv

    def forward(
        self, x: torch.Tensor, inject: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Run the encoder layer's conv, optional inject, fused norm/activations, and DConv.

        :param x: Input tensor; ``(B, C, Fr, T)`` for frequency layers or ``(B, C, T)`` otherwise
        :param inject: Optional tensor added after the conv, matching its last dimension
        :return: The encoded tensor for the next stage
        """
        if not self.freq and x.dim() == 4:
            B, C, Fr, T = x.shape
            x = x.view(B, -1, T)
        if not self.freq:
            le = x.shape[-1]
            if not le % self.stride == 0:
                x = F.pad(x, (0, self.stride - (le % self.stride)))
        y = self.conv(x)
        if self.empty:
            return y
        if inject is not None:
            assert inject.shape[-1] == y.shape[-1]
            if inject.dim() == 3 and y.dim() == 4:
                inject = inject[:, :, None]

        if self._fused_gelu:
            y = self.norm1(y, inject=inject)
        elif type(self.norm1) is nn.Identity and inject is not None:
            out = torch.empty_like(_kernel_arg(y))
            _launch(
                "add_gelu",
                y.dtype,
                out,
                _kernel_arg(y),
                _kernel_arg(inject),
            )
            y = out
        elif inject is not None:
            y = F.gelu(self.norm1(y + inject))
        else:
            y = F.gelu(self.norm1(y))
        if self.dconv is not None:
            if self.freq:
                B, C, Fr, T = y.shape
                y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y)
            if self.freq:
                y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        if self.rewrite is not None:
            if self._fused_glu:
                z = self.norm2(self.rewrite(y))
            else:
                z = F.glu(self.norm2(self.rewrite(y)), dim=1)
        else:
            z = y
        return z


class FusedHDecLayer(nn.Module):
    """Replacement for ``unblend.blocks.HDecLayer`` using fused CUDA kernels.

    The ``glu(norm1(rewrite(...)))`` pattern is fused. We do NOT fuse the
    final ``gelu(norm2(conv_tr(...)))`` because the ``last`` flag (mutated
    by MultiWrap) decides whether GELU runs at all — keeping that switch
    in Python keeps things simple. ``norm2`` itself is still
    ``CUDAGroupNorm`` (handled by the outer swap pass).
    """

    def __init__(self, layer: nn.Module) -> None:
        """
        Build a fused decoder layer from an ``HDecLayer``, folding the GLU norm path.

        :param layer: Source ``HDecLayer`` whose conv_tr, rewrite, norms, flags, and
            inner DConv are carried forward or replaced with fused equivalents
        """
        super().__init__()
        from ..blocks import DConv

        self.pad = layer.pad
        self.last = layer.last
        self.freq = layer.freq
        self.chin = layer.chin
        self.empty = layer.empty
        self.stride = layer.stride
        self.kernel_size = layer.kernel_size
        self.norm = layer.norm
        self.context_freq = layer.context_freq

        self.conv_tr = layer.conv_tr
        self.norm2 = layer.norm2
        if layer.empty:
            return

        self.rewrite = layer.rewrite
        if layer.rewrite is not None:
            if isinstance(layer.norm1, nn.GroupNorm) and layer.norm1.num_groups == 1:
                self.norm1 = FusedGroupNormGlu.from_groupnorm(layer.norm1)
                self._fused_glu = True
            else:
                self.norm1 = layer.norm1
                self._fused_glu = False
        else:
            self.norm1 = None
            self._fused_glu = False

        if layer.dconv is not None and isinstance(layer.dconv, DConv):
            try:
                self.dconv = FusedDConv.from_dconv(layer.dconv)
            except (TypeError, ValueError):
                self.dconv = layer.dconv
        else:
            self.dconv = layer.dconv

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Add the skip connection, run the fused GLU norm, DConv, and transposed conv.

        :param x: Input tensor; reshaped to ``(B, chin, Fr, T)`` for frequency layers
        :param skip: Skip-connection tensor added to ``x``; ``None`` only for empty layers
        :param length: Target length used to crop the transposed-conv output for time layers
        :return: A ``(z, y)`` pair — the decoded output ``z`` and the pre-conv_tr tensor ``y``
        """
        if self.freq and x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, self.chin, -1, T)
        if not self.empty:
            x = x + skip
            if self.rewrite is not None:
                if self._fused_glu:
                    y = self.norm1(self.rewrite(x))
                else:
                    y = F.glu(self.norm1(self.rewrite(x)), dim=1)
            else:
                y = x
            if self.dconv is not None:
                if self.freq:
                    B, C, Fr, T = y.shape
                    y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
                y = self.dconv(y)
                if self.freq:
                    y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        else:
            y = x
            assert skip is None

        z_tr = self.conv_tr(y)
        if isinstance(self.norm2, CUDAGroupNorm):
            z = self.norm2(z_tr, gelu=not self.last)
        else:
            z = self.norm2(z_tr)
            if not self.last:
                z = F.gelu(z)
        if self.freq:
            if self.pad:
                z = z[..., self.pad : -self.pad, :]
        else:
            z = z[..., self.pad : self.pad + length]
        return z, y


def fused_roformer_rotary(
    t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply interleaved rotary rotation with one fused CUDA kernel.

    :param t: Queries or keys ``[..., seq, dim]``.
    :param cos: Rotation cosine table ``[seq, dim // 2]``.
    :param sin: Rotation sine table ``[seq, dim // 2]``.
    :return: Rotated tensor of the same shape and dtype.
    """
    if t.stride(-1) != 1 or t.dim() > 8:
        x1, x2 = t.unflatten(-1, (-1, 2)).unbind(dim=-1)
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(
            -2
        )
    return torch.ops.unblend_cuda.roformer_rotary(t, cos, sin)


@torch.library.custom_op(f"{_OP_NAMESPACE}::roformer_rotary", mutates_args=())
def roformer_rotary_op(
    t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Fused interleaved rotary rotation (custom-op wrapper).

    :param t: Queries or keys ``[..., seq, dim]``, last-dim contiguous.
    :param cos: Rotation cosine table ``[seq, dim // 2]``.
    :param sin: Rotation sine table ``[seq, dim // 2]``.
    :return: Rotated tensor, contiguous.
    """
    return _get_extension().roformer_rotary(t, cos, sin)


@roformer_rotary_op.register_fake
def _roformer_rotary_fake(
    t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Shape/dtype propagation for tracing.

    :param t: Input tensor.
    :param cos: Unused.
    :param sin: Unused.
    :return: Fresh tensor with ``t``'s size/dtype (contiguous).
    """
    return torch.empty_like(t)


def has_swappable_modules(model: nn.Module) -> bool:
    """
    Whether :func:`apply_cuda_optimizations` would replace anything.

    :param model: Model to inspect.
    :return: ``True`` if at least one module would be swapped.
    """
    from ..blocks import HDecLayer, HEncLayer
    from ..transformer import MyGroupNorm

    for module in model.modules():
        if isinstance(module, (HEncLayer, HDecLayer, MyGroupNorm)):
            return True
        if (
            isinstance(module, nn.GroupNorm)
            and module.num_groups == 1
            and module.affine
        ):
            return True
    return False


def apply_cuda_optimizations(model: nn.Module) -> dict[str, int]:
    """
    Replace memory-bound op chains with fused CUDA-kernel equivalents in-place.

    :param model: Model to mutate in place, swapping eligible submodules.
    :return: A mapping from swap kind to the number of modules replace...
    """
    from ..blocks import HDecLayer, HEncLayer
    from ..transformer import MyGroupNorm

    counts = {
        "h_enc_layer": 0,
        "h_dec_layer": 0,
        "fused_dconv": 0,
        "group_norm": 0,
        "my_group_norm": 0,
    }

    try:
        _get_extension()
    except Exception as exc:
        warnings.warn(
            f"CUDA kernel compilation failed ({exc}); skipping CUDA "
            "optimizations and keeping native PyTorch ops.",
            RuntimeWarning,
        )
        return counts

    def _walk_layers(mod: nn.Module) -> None:
        """
        Recursively replace ``HEncLayer``/``HDecLayer`` children with fused versions.

        :param mod: Module whose children are walked and swapped in place
        """
        for name, child in list(mod.named_children()):
            replacement: nn.Module | None = None
            if type(child) is HEncLayer:
                try:
                    replacement = FusedHEncLayer(child)
                    counts["h_enc_layer"] += 1

                    if isinstance(getattr(replacement, "dconv", None), FusedDConv):
                        counts["fused_dconv"] += 1
                except Exception as exc:
                    logger.debug(
                        "CUDA fusion failed for %s, leaving it unfused: %s",
                        type(child).__name__,
                        exc,
                        exc_info=True,
                    )
            elif type(child) is HDecLayer:
                try:
                    replacement = FusedHDecLayer(child)
                    counts["h_dec_layer"] += 1
                    if isinstance(getattr(replacement, "dconv", None), FusedDConv):
                        counts["fused_dconv"] += 1
                except Exception as exc:
                    logger.debug(
                        "CUDA fusion failed for %s, leaving it unfused: %s",
                        type(child).__name__,
                        exc,
                        exc_info=True,
                    )
            if replacement is not None:
                params = list(child.parameters())
                if params:
                    replacement.to(device=params[0].device)
                replacement.train(child.training)
                setattr(mod, name, replacement)
            else:
                _walk_layers(child)

    _walk_layers(model)

    def _walk_modules(mod: nn.Module) -> None:
        """
        Recursively swap remaining ``MyGroupNorm``/``GroupNorm`` children.

        :param mod: Module whose children are walked and swapped in place
        """
        for name, child in list(mod.named_children()):
            if isinstance(child, (CUDAGroupNorm, CUDAMyGroupNorm)):
                continue
            replacement: nn.Module | None = None
            if isinstance(child, MyGroupNorm):
                if child.num_groups == 1 and child.affine:
                    replacement = CUDAMyGroupNorm(child)
                    counts["my_group_norm"] += 1
            elif type(child) is nn.GroupNorm:
                if child.num_groups == 1 and child.affine:
                    replacement = CUDAGroupNorm.from_groupnorm(child)
                    counts["group_norm"] += 1

            if replacement is not None:
                params = list(child.parameters())
                if params:
                    replacement.to(device=params[0].device)
                replacement.train(child.training)
                setattr(mod, name, replacement)
            else:
                _walk_modules(child)

    _walk_modules(model)
    return counts


__all__ = [
    "CUDAGroupNorm",
    "fused_roformer_rotary",
    "has_swappable_modules",
    "cuda_rms_norm",
    "CUDAMyGroupNorm",
    "FusedGroupNormGelu",
    "FusedGroupNormGlu",
    "FusedNormGluLayerScaleResid",
    "FusedDConvLayer",
    "FusedDConv",
    "FusedHEncLayer",
    "FusedHDecLayer",
    "apply_cuda_optimizations",
]
