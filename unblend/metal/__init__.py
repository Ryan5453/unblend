"""
Custom Metal kernels for low-precision (FP16/BF16) inference on Apple Silicon (MPS).

PyTorch's MPS backend has slow paths for several FP16 ops that are
hot in HTDemucs — most notably ``aten::native_group_norm`` (3.5x slower in
FP16 than FP32) and ``aten::scaled_dot_product_attention`` (~50% slower in
FP16 due to implicit upcasts at op boundaries). This module ships drop-in
replacements that close those gaps:

- :class:`MetalGroupNorm` — wraps PyTorch's ``nn.GroupNorm`` with a single
  fused Metal kernel for FP16/BF16 inputs (with an FP32-cast fallback for
  the small-batch / large-per-batch shapes where the GPU stays
  under-utilised).
- :class:`MetalMyGroupNorm` — replaces the transformer's ``MyGroupNorm``
  with a transpose-avoiding channel-last Metal kernel: the op folds onto
  ``(B, T*C)`` with the affine broadcast over the trailing ``C`` axis,
  skipping both transposes and the ~6-op pointwise chain PyTorch would
  dispatch.
- :class:`MetalMultiheadAttention` — keeps Q/K/V/output linear projections
  in the input dtype (MPS matmul has a fast FP16/BF16 path) and runs the
  attention itself as a manual matmul → softmax → matmul in that same dtype,
  avoiding both the fused-SDPA per-call dispatch overhead and the FP32
  upcast. Masked/causal calls fall back to the wrapped MHA.

These are activated automatically by :func:`apply_metal_optimizations`,
which :class:`unblend.api.Separator` calls when the user opts into FP16 or
BF16 on MPS. RoFormer's checkpoint-compatible ``RMSNorm`` modules call
:func:`metal_rms_norm` directly during MPS inference; that kernel also handles
FP32 as well as the automatic FP16 dtype. CPU/CUDA paths use PyTorch's native
ops.
"""

from __future__ import annotations

import logging
import warnings
from importlib import resources
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _pow2_tgs(max_threads: int, cap: int = 256) -> int:
    """
    Largest power of two ``<= min(cap, max_threads)``.

    The simdgroup reductions no longer require a power-of-two threadgroup
    size, but pow-2 keeps the simdgroups full and the strided loops evenly
    balanced. Apple GPUs report ``max_threads_per_threadgroup == 1024``
    today, so capping at ``cap`` already yields a power of two.

    :param max_threads: The kernel's ``max_threads_per_threadgroup``
    :param cap: Upper bound on the returned threadgroup size
    :return: The largest power-of-two threadgroup size within the bounds
    """
    limit = min(cap, max_threads)
    tgs = 1
    while tgs * 2 <= limit:
        tgs *= 2
    return tgs


# ---------------------------------------------------------------------------
# Metal kernel sources
# ---------------------------------------------------------------------------
#
# The Metal Shading Language sources live in sibling ``.metal`` files,
# split by purpose:
#
#   common.metal           — shared prelude (SCALAR_T/SCALAR4_T defaults,
#                            simdgroup reduction helpers); prepended to
#                            every other file at compile time
#   group_norm.metal       — basic GroupNorm (channel-first + channel-last)
#                            + reduction primitives (``partial_reduce`` +
#                            ``finalize_meanvar``, shared by every
#                            multi-stage path below)
#   group_norm_gelu.metal  — GroupNorm fused with GELU
#   group_norm_glu.metal   — GroupNorm fused with GLU (channel halving)
#   dconv_envelope.metal   — DConv post-conv2 envelope
#                            (residual + layer_scale * glu(group_norm(z)))
#
# All reductions accumulate in FP32 inside the kernel; FP16 only crosses
# the device-memory boundary at load and store. This avoids the implicit
# cast traffic that makes PyTorch's stock FP16 GroupNorm slow on MPS.
# Loads/stores vectorize to half4/bfloat4 when alignment permits, with
# scalar fallbacks compiled into the same kernel (uniform runtime branch).
def _load_metal_source(name: str) -> str:
    """
    Read a Metal source file shipped alongside this package.

    :param name: Filename of the ``.metal`` source within this package
    :return: The file's contents decoded as UTF-8 text
    """
    return resources.files(__name__).joinpath(name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Lazy kernel compilation
# ---------------------------------------------------------------------------
#
# The kernels are split across four ``.metal`` files by purpose. Each file
# compiles to its own Metal library; kernels in different libraries
# communicate only through global memory buffers (e.g. the ``meanvar``
# scratch the multi-stage paths share between ``finalize_meanvar`` from
# ``group_norm.metal`` and the ``apply_*`` kernels in the activation /
# envelope files).

_KERNEL_SOURCES: dict[str, str] = {
    # group_norm.metal — basic GroupNorm + reduction primitives
    "group_norm_g1": "group_norm.metal",
    "group_norm_g1_chlast": "group_norm.metal",
    "partial_reduce": "group_norm.metal",
    "finalize_meanvar": "group_norm.metal",
    "apply_norm": "group_norm.metal",
    "apply_norm_chlast": "group_norm.metal",
    # group_norm_gelu.metal — GN fused with GELU
    "group_norm_g1_gelu": "group_norm_gelu.metal",
    "apply_norm_gelu": "group_norm_gelu.metal",
    # group_norm_glu.metal — GN fused with GLU (channel halving)
    "group_norm_g1_glu": "group_norm_glu.metal",
    "apply_norm_glu": "group_norm_glu.metal",
    # dconv_envelope.metal — DConv post-conv2 fusion paths
    "norm_glu_ls_resid": "dconv_envelope.metal",
    "apply_norm_glu_ls_resid": "dconv_envelope.metal",
    # rms_norm.metal — RoFormer last-dimension RMSNorm
    "rms_norm": "rms_norm.metal",
}
_HTDEMUCS_KERNELS = tuple(name for name in _KERNEL_SOURCES if name != "rms_norm")


_compiled_libraries: dict[tuple[str, torch.dtype], Any] = {}
_compiled_kernels: dict[tuple[str, torch.dtype], Any] = {}

# Map torch dtype -> Metal scalar typename injected into the kernel source via
# a ``#define SCALAR_T ...`` prepended at compile time. The .metal files were
# rewritten to use ``SCALAR_T`` instead of ``half`` so the same source compiles
# for FP32, FP16, or BF16 where supported by its wrapper.
_DTYPE_TO_METAL: dict[torch.dtype, str] = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "bfloat",
}

# The HTDemucs fusion kernels dispatch only for reduced precision. RoFormer's
# RMSNorm kernel deliberately also supports explicitly requested FP32.
_LP_DTYPES = frozenset((torch.float16, torch.bfloat16))
_RMS_DTYPES = frozenset(_DTYPE_TO_METAL)


def _is_metal_lp(t: torch.Tensor) -> bool:
    """
    Report whether a tensor is on MPS in a kernel-supported low-precision dtype.

    True only for a low-precision (FP16/BF16) tensor on MPS — the sole case
    the custom Metal kernels handle. Every other (device, dtype) combination
    falls back to PyTorch. Names the dispatch condition that every fused
    module's ``forward`` shares, so the fallback gate reads the same way in
    one place instead of being open-coded as ``device != "mps" or dtype not
    in _LP_DTYPES`` at each call site.

    :param t: Tensor whose device and dtype are checked
    :return: ``True`` if ``t`` is on MPS and FP16/BF16, ``False`` otherwise
    """
    return t.device.type == "mps" and t.dtype in _LP_DTYPES


def _kernel_arg(t: torch.Tensor) -> torch.Tensor:
    """
    Prepare a tensor for kernel dispatch: contiguous with a 4-element-aligned
    storage offset.

    The kernels reinterpret their buffers as ``half4``/``bfloat4`` on the
    vectorized paths, which needs 8-byte alignment. A contiguous tensor can
    still be a view at an odd element offset into a pooled allocation, so
    clone those (in practice the inputs are fresh conv/linear outputs at
    offset 0 and this never fires).

    :param t: Tensor to prepare
    :return: ``t`` itself if already safe, else a contiguous aligned copy
    """
    t = t.contiguous()
    if t.storage_offset() % 4:
        t = t.clone()
    return t


def _get_kernel(name: str, dtype: torch.dtype) -> Any:
    """
    Look up a Metal kernel by ``(name, dtype)``; compile its source file
    once per dtype and cache both the library and the per-kernel handle.

    The same ``.metal`` source compiles for ``float``, ``half``, or ``bfloat`` —
    we prepend ``#define SCALAR_T <type>`` and call
    ``torch.mps.compile_shader`` once per dtype per file. PyTorch's API
    doesn't deduplicate identical sources internally, so we cache here.

    :param name: Kernel function name (a key of ``_KERNEL_SOURCES``)
    :param dtype: Scalar dtype to compile for (FP32, FP16, or BF16)
    :return: The compiled, callable per-kernel handle for ``(name, dtype)``
    :raises RuntimeError: If ``torch.mps.compile_shader`` is unavailable
    :raises KeyError: If ``name`` is not a known Metal kernel
    :raises ValueError: If ``dtype`` has no corresponding Metal scalar type
    """
    cache_key = (name, dtype)
    cached = _compiled_kernels.get(cache_key)
    if cached is not None:
        return cached
    if not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError(
            "torch.mps.compile_shader unavailable; need PyTorch >= 2.6 for "
            "Metal kernel-backed inference."
        )
    source_file = _KERNEL_SOURCES.get(name)
    if source_file is None:
        raise KeyError(
            f"Unknown Metal kernel {name!r}; expected one of {sorted(_KERNEL_SOURCES)}"
        )
    metal_type = _DTYPE_TO_METAL.get(dtype)
    if metal_type is None:
        raise ValueError(
            f"Metal kernels are only built for {_RMS_DTYPES}; got {dtype!r}"
        )
    lib_key = (source_file, dtype)
    lib = _compiled_libraries.get(lib_key)
    if lib is None:
        # Prepend the shared prelude (includes, reduction helpers) and the
        # SCALAR_T/SCALAR4_T defines so the same source compiles for either
        # FP16 or BF16.
        src = _load_metal_source("common.metal") + _load_metal_source(source_file)
        lib = torch.mps.compile_shader(
            f"#define SCALAR_T {metal_type}\n#define SCALAR4_T {metal_type}4\n{src}"
        )
        _compiled_libraries[lib_key] = lib
    fn = getattr(lib, name)
    _compiled_kernels[cache_key] = fn
    return fn


def metal_rms_norm(
    x: torch.Tensor, gamma: torch.Tensor, scale: float
) -> torch.Tensor:
    """
    Apply RoFormer's last-dimension RMSNorm with one fused MPS kernel.

    The Metal reduction and affine arithmetic use FP32 for every storage
    dtype. Non-MPS and unsupported-dtype inputs take the exact PyTorch path;
    callers that need autograd should also use that fallback because custom
    ``compile_shader`` kernels are inference-only.

    :param x: Input tensor normalized over its final dimension.
    :param gamma: Learnable gain with length ``x.shape[-1]``.
    :param scale: RoFormer's ``sqrt(dim)`` normalization scale.
    :return: Normalized tensor with the same shape and dtype as ``x``.
    """
    if x.device.type != "mps" or x.dtype not in _RMS_DTYPES or x.numel() == 0:
        normalized = F.normalize(x.float(), dim=-1) * scale * gamma.float()
        return normalized.type(x.dtype)

    x_contig = x.contiguous()
    dim = x_contig.shape[-1]
    rows = x_contig.numel() // dim
    gamma_contig = gamma.to(device=x.device, dtype=x.dtype).contiguous()
    out = torch.empty_like(x_contig)

    kernel = _get_kernel("rms_norm", x.dtype)
    tgs = _pow2_tgs(kernel.max_threads_per_threadgroup)
    while tgs > dim:
        tgs //= 2
    kernel(
        out,
        x_contig,
        gamma_contig,
        dim,
        float(scale),
        threads=rows * tgs,
        group_size=tgs,
    )
    return out.view_as(x)


# ---------------------------------------------------------------------------
# Module replacements
# ---------------------------------------------------------------------------


class MetalGroupNorm(nn.Module):
    """Replacement for ``nn.GroupNorm(num_groups=1)`` on MPS in FP16/BF16.

    Two low-precision fast paths and one FP32 fallback:

    - **Single-stage**: one threadgroup per batch element. Fully fused;
      lowest launch overhead. Best for the high-B, low-per-batch shapes
      that dominate count in HTDemucs (DConv internals).
    - **Multi-stage**: three Metal kernels — partial-reduce per tile,
      finalise mean/scale per batch, apply per tile. Tiles parallelise
      across many threadgroups so the GPU is saturated even when
      ``B == 2``. Used for the outermost encoder/decoder GroupNorms and
      the transformer norms.
    - **PyTorch fallback** (``F.group_norm``): non-MPS devices and FP32
      inputs.

    :meth:`_use_single_stage` picks between the two kernel paths.
    """

    # Hard per-batch ceiling for the single-stage path: above this even a
    # large-B launch is better served by tiles.
    _SINGLE_STAGE_LIMIT = 1_500_000
    # The single-stage kernel launches exactly ``B`` threadgroups. Below
    # this many, the GPU runs mostly idle (measured 4-8 GB/s vs ~450 GB/s
    # peak on M-series at B=2), so unless the per-batch work is small we
    # tile. Swept empirically on M-series: at B >= 128 single-stage wins at
    # every per-batch size; below that the crossover sits between 32K and
    # 131K elements per batch.
    _SINGLE_STAGE_MIN_BATCH = 128
    # ...but for small per-batch work the multi-stage path's two extra
    # kernel launches cost more than the parallelism buys.
    _SINGLE_STAGE_SMALL_PER_BATCH = 49_152
    # Target tile size for the multi-stage path. Each stage-1/3 threadgroup
    # processes ``~TILE_SIZE`` elements; we pick ``num_tiles`` so each tile
    # is roughly this size, capped to avoid excessive scratch.
    _MULTI_STAGE_TILE_SIZE = 16_384
    _MULTI_STAGE_MAX_TILES = 4096

    @classmethod
    def _use_single_stage(cls, batch: int, per_batch: int) -> bool:
        """
        Decide between the single-stage and multi-stage kernel paths.

        :param batch: Number of batch elements (threadgroups a single-stage
            launch would fire)
        :param per_batch: Elements reduced per batch element (input space)
        :return: ``True`` to run the fused single-stage kernel
        """
        if per_batch > cls._SINGLE_STAGE_LIMIT:
            return False
        return (
            batch >= cls._SINGLE_STAGE_MIN_BATCH
            or per_batch <= cls._SINGLE_STAGE_SMALL_PER_BATCH
        )

    def _multi_stage_meanvar(
        self,
        x_contig: torch.Tensor,
        B: int,
        per_batch_in: int,
        tile_space: int,
    ) -> tuple[torch.Tensor, int]:
        """
        Run multi-stage stages 1+2: per-tile partial reduce, then finalize
        per-batch ``(mean, rsqrt(var+eps))``.

        :param x_contig: Contiguous kernel-ready input, ``(B, per_batch_in)`` flat
        :param B: Number of batch elements
        :param per_batch_in: Elements reduced per batch element
        :param tile_space: Element count ``num_tiles`` is sized against — the
            *output* space, so stage 3 (which the caller launches with the
            returned ``num_tiles``) gets evenly sized tiles
        :return: The ``(B, 2)`` FP32 meanvar buffer and ``num_tiles``
        """
        num_tiles = min(
            self._MULTI_STAGE_MAX_TILES,
            max(
                1,
                (tile_space + self._MULTI_STAGE_TILE_SIZE - 1)
                // self._MULTI_STAGE_TILE_SIZE,
            ),
        )
        # Snap num_tiles down to a power of two: keeps the stage-2
        # threadgroup sizing simple and each tile a touch larger.
        pow2 = 1
        while pow2 * 2 <= num_tiles:
            pow2 *= 2
        num_tiles = pow2

        dtype = x_contig.dtype
        scratch = torch.empty(
            (B, num_tiles, 2), dtype=torch.float32, device=x_contig.device
        )
        meanvar = torch.empty((B, 2), dtype=torch.float32, device=x_contig.device)

        k1 = _get_kernel("partial_reduce", dtype)
        k2 = _get_kernel("finalize_meanvar", dtype)
        tgs1 = _pow2_tgs(k1.max_threads_per_threadgroup)
        # Stage 2 reduces ``num_tiles`` floats per batch; size threadgroup to
        # at most ``num_tiles`` (pow-2 keeps the strided loop balanced).
        tgs2 = min(num_tiles, k2.max_threads_per_threadgroup)
        pow2 = 1
        while pow2 * 2 <= tgs2:
            pow2 *= 2
        tgs2 = pow2

        k1(
            x_contig,
            scratch,
            per_batch_in,
            num_tiles,
            threads=B * num_tiles * tgs1,
            group_size=tgs1,
        )
        k2(
            scratch,
            meanvar,
            per_batch_in,
            num_tiles,
            float(self.eps),
            x_contig,
            threads=B * tgs2,
            group_size=tgs2,
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
                f"MetalGroupNorm only supports num_groups=1; got {gn.num_groups}"
            )
        if not gn.affine:
            raise ValueError("MetalGroupNorm requires affine=True")
        self.num_channels = gn.num_channels
        self.eps = gn.eps
        # Affine parameters live in FP32 storage; we cast lazily and cache.
        self.weight = nn.Parameter(gn.weight.detach().to(torch.float32).clone())
        self.bias = nn.Parameter(gn.bias.detach().to(torch.float32).clone())

    @classmethod
    def from_groupnorm(cls, gn: nn.GroupNorm) -> "MetalGroupNorm":
        """
        Build a :class:`MetalGroupNorm` from an existing GroupNorm.

        :param gn: Source GroupNorm to wrap
        :return: A new :class:`MetalGroupNorm` mirroring ``gn``
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
        key = (dtype, device)
        cached = cache.get(key)
        if cached is None:
            w = self.weight.detach().to(device=device, dtype=dtype).contiguous()
            b = self.bias.detach().to(device=device, dtype=dtype).contiguous()
            cache[key] = (w, b)
            return w, b
        return cached

    def _load_from_state_dict(self, *args: object, **kwargs: object) -> None:
        """
        Reload parameters and invalidate the lazily-cast affine/LayerScale caches.

        The affine (and, in subclasses, LayerScale) params are cast to the input
        dtype/device on first forward and cached by ``(dtype, device)``. If a
        caller reloads weights via ``load_state_dict`` after a forward has run,
        those cached casts would be stale — so drop them here and let the next
        forward re-derive them from the updated parameters.

        :param args: Positional arguments forwarded to ``nn.Module._load_from_state_dict``.
        :param kwargs: Keyword arguments forwarded to ``nn.Module._load_from_state_dict``.
        """
        super()._load_from_state_dict(*args, **kwargs)
        for name in ("_aff_cache", "_ls_cache"):
            cache = getattr(self, name, None)
            if cache:
                cache.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ``num_groups=1`` group normalization, using a Metal kernel on MPS FP16/BF16.

        :param x: Input tensor of shape ``(B, C, ...)``
        :return: Normalized, affine-transformed tensor with the same shape as ``x``
        """
        # FP32 / non-MPS / unsupported low-precision dtype: defer to PyTorch.
        # The affine params are stored in FP32, so any non-FP32 input must be
        # upcast (the metal-LP path is handled below) — otherwise F.group_norm
        # raises on the dtype mismatch (e.g. an FP16 tensor on a non-MPS device).
        if not _is_metal_lp(x):
            if x.dtype == torch.float32:
                return F.group_norm(x, 1, self.weight, self.bias, self.eps)
            return F.group_norm(
                x.to(torch.float32), 1, self.weight, self.bias, self.eps
            ).to(x.dtype)

        x_contig = _kernel_arg(x)
        B = x_contig.shape[0]
        C = x_contig.shape[1]
        N = 1
        for d in x_contig.shape[2:]:
            N *= d
        per_batch = C * N

        weight, bias = self._lp_affine(x.dtype, x.device)

        if self._use_single_stage(B, per_batch):
            kernel = _get_kernel("group_norm_g1", x.dtype)
            tgs = _pow2_tgs(kernel.max_threads_per_threadgroup)
            while tgs > 1 and tgs > per_batch:
                tgs //= 2
            out = torch.empty_like(x_contig)
            kernel(
                out,
                x_contig,
                weight,
                bias,
                C,
                N,
                float(self.eps),
                threads=B * tgs,
                group_size=tgs,
            )
            return out.view_as(x)

        # Multi-stage path. Tile the per-batch work so that ``B * num_tiles``
        # threadgroups participate in stage 1 and 3 — that's enough to keep
        # the Apple GPU fully busy even when ``B == 2``.
        meanvar, num_tiles = self._multi_stage_meanvar(
            x_contig, B, per_batch, per_batch
        )
        out = torch.empty_like(x_contig)
        k3 = _get_kernel("apply_norm", x.dtype)
        tgs3 = _pow2_tgs(k3.max_threads_per_threadgroup)
        k3(
            out,
            x_contig,
            meanvar,
            weight,
            bias,
            per_batch,
            num_tiles,
            N,
            threads=B * num_tiles * tgs3,
            group_size=tgs3,
        )
        return out.view_as(x)


class FusedGroupNormGelu(MetalGroupNorm):
    """Drop-in for the ``gelu(group_norm(...))`` pattern.

    Same shape contract as ``nn.GroupNorm`` (input == output shape) but the
    forward applies GELU at the same time as the normalize+affine, saving a
    memory round-trip on the activation tensor. The Metal kernel uses the tanh
    GELU approximation (no ``erf`` builtin in the MPS shader toolchain; the gap
    is sub-FP16-precision); the FP32 fallback below uses exact erf.

    Activated automatically via :func:`apply_metal_optimizations` for the
    ``self.norm1`` slots in HEncLayer/HDecLayer/DConv where a GELU follows.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ``gelu(group_norm(x))`` fused into one Metal kernel on MPS FP16/BF16.

        :param x: Input tensor of shape ``(B, C, ...)``
        :return: GELU-activated normalized tensor with the same shape as ``x``
        """
        # FP32 / non-MPS / unsupported low-precision dtype: hand to PyTorch.
        # Affine params are FP32, so upcast any non-FP32 input to avoid a
        # dtype mismatch in F.group_norm.
        if not _is_metal_lp(x):
            if x.dtype == torch.float32:
                return F.gelu(
                    F.group_norm(x, 1, self.weight, self.bias, self.eps),
                )
            return F.gelu(
                F.group_norm(x.to(torch.float32), 1, self.weight, self.bias, self.eps),
            ).to(x.dtype)

        x_contig = _kernel_arg(x)
        B = x_contig.shape[0]
        C = x_contig.shape[1]
        N = 1
        for d in x_contig.shape[2:]:
            N *= d
        per_batch = C * N
        weight, bias = self._lp_affine(x.dtype, x.device)

        if self._use_single_stage(B, per_batch):
            kernel = _get_kernel("group_norm_g1_gelu", x.dtype)
            tgs = _pow2_tgs(kernel.max_threads_per_threadgroup)
            while tgs > 1 and tgs > per_batch:
                tgs //= 2
            out = torch.empty_like(x_contig)
            kernel(
                out,
                x_contig,
                weight,
                bias,
                C,
                N,
                float(self.eps),
                threads=B * tgs,
                group_size=tgs,
            )
            return out.view_as(x)

        # Multi-stage: reuse partial_reduce / finalize, then apply_norm_gelu.
        meanvar, num_tiles = self._multi_stage_meanvar(
            x_contig, B, per_batch, per_batch
        )
        out = torch.empty_like(x_contig)
        k3 = _get_kernel("apply_norm_gelu", x.dtype)
        tgs3 = _pow2_tgs(k3.max_threads_per_threadgroup)
        k3(
            out,
            x_contig,
            meanvar,
            weight,
            bias,
            per_batch,
            num_tiles,
            N,
            threads=B * num_tiles * tgs3,
            group_size=tgs3,
        )
        return out.view_as(x)


class FusedGroupNormGlu(MetalGroupNorm):
    """Drop-in for the ``glu(group_norm(rewrite(...)), dim=1)`` pattern.

    Input shape: ``(B, 2C, ...)``. Output shape: ``(B, C, ...)``. The
    GroupNorm reduces over all 2C input channels (so the per-batch mean
    matches ``F.group_norm`` exactly), then for each output channel pair
    we read ``a = norm(in[c])`` and ``b = norm(in[c + C])`` and combine
    via ``a * sigmoid(b)`` — all without ever writing the post-norm
    full-size tensor to memory.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ``glu(group_norm(x), dim=1)`` fused into one Metal kernel on MPS FP16/BF16.

        :param x: Input tensor of shape ``(B, 2C, ...)`` with even channel count
        :return: GLU-gated normalized tensor of shape ``(B, C, ...)``
        :raises ValueError: If the input channel dimension is not even
        """
        # Affine params are FP32, so upcast any non-FP32 input to avoid a
        # dtype mismatch in F.group_norm.
        if not _is_metal_lp(x):
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

        if self._use_single_stage(B, per_batch_in):
            kernel = _get_kernel("group_norm_g1_glu", x.dtype)
            tgs = _pow2_tgs(kernel.max_threads_per_threadgroup)
            while tgs > 1 and tgs > per_batch_out:
                tgs //= 2
            out_shape = (B, C_half) + tuple(x_contig.shape[2:])
            out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
            kernel(
                out,
                x_contig,
                weight,
                bias,
                C_in,
                N,
                float(self.eps),
                threads=B * tgs,
                group_size=tgs,
            )
            return out

        # Multi-stage: partial-reduce/finalize over the FULL input
        # (per_batch_in elements per batch), then apply_norm_glu over the
        # OUTPUT space (per_batch_out elements per batch).
        meanvar, num_tiles = self._multi_stage_meanvar(
            x_contig, B, per_batch_in, per_batch_out
        )
        out_shape = (B, C_half) + tuple(x_contig.shape[2:])
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        k3 = _get_kernel("apply_norm_glu", x.dtype)
        tgs3 = _pow2_tgs(k3.max_threads_per_threadgroup)
        # Stage 3 tiles the OUTPUT (per_batch_out elements) and writes
        # halve-channel output.
        k3(
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
            threads=B * num_tiles * tgs3,
            group_size=tgs3,
        )
        return out


class FusedNormGluLayerScaleResid(MetalGroupNorm):
    """Single fused op for the DConv envelope after the second conv:
    ``residual + layer_scale * glu(group_norm(z), dim=1)``.

    Replaces FOUR previously-separate ops (group_norm, glu, layerscale mul,
    residual add) with one Metal kernel — saves three intermediate tensor
    writes and the matching reads. ``z`` has 2C input channels, output has
    C channels.

    Inherits ``MetalGroupNorm`` for the affine-cache and dispatch logic;
    additionally owns the LayerScale parameter.
    """

    def __init__(self, gn: nn.GroupNorm, layer_scale_param: torch.Tensor) -> None:
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
        cls, gn: nn.GroupNorm, layer_scale_param: torch.Tensor
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
        key = (dtype, device)
        cached = cache.get(key)
        if cached is None:
            t = self.layer_scale.detach().to(device=device, dtype=dtype).contiguous()
            cache[key] = t
            return t
        return cached

    def forward(self, z: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Compute ``residual + layer_scale * glu(group_norm(z), dim=1)`` in one Metal kernel.

        :param z: GroupNorm/GLU input of shape ``(B, 2C, ...)`` with even channel count
        :param residual: Residual tensor of shape ``(B, C, ...)`` to add
        :return: The fused result of shape ``(B, C, ...)``
        :raises ValueError: If the GLU input channel dimension is not even
        """
        # FP32 / non-MPS / unsupported low-precision dtype: explicit eltwise.
        # weight/bias/layer_scale are FP32, so a non-FP32 input is computed in
        # FP32 and cast back (avoids a group_norm dtype mismatch and keeps the
        # output dtype equal to ``z``).
        if not _is_metal_lp(z):
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
        out = torch.empty(out_shape, dtype=z.dtype, device=z.device)

        if self._use_single_stage(B, per_batch_in):
            kernel = _get_kernel("norm_glu_ls_resid", z.dtype)
            tgs = _pow2_tgs(kernel.max_threads_per_threadgroup)
            # Don't launch more threads than the apply loop has output elements
            # (the reduction over per_batch_in still covers all input via the
            # strided loop). Mirrors the group_norm_glu single-stage clamp.
            while tgs > 1 and tgs > per_batch_out:
                tgs //= 2
            kernel(
                out,
                z_c,
                r_c,
                weight,
                bias,
                ls,
                C2,
                N,
                float(self.eps),
                threads=B * tgs,
                group_size=tgs,
            )
            return out

        # Multi-stage path: stages 1+2 reduce over the FULL (2C, N) input.
        meanvar, num_tiles = self._multi_stage_meanvar(
            z_c, B, per_batch_in, per_batch_out
        )
        k3 = _get_kernel("apply_norm_glu_ls_resid", z.dtype)
        tgs3 = _pow2_tgs(k3.max_threads_per_threadgroup)
        # Stage 3 tiles the OUTPUT space.
        k3(
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
            threads=B * num_tiles * tgs3,
            group_size=tgs3,
        )
        return out


class MetalMyGroupNorm(MetalGroupNorm):
    """Replacement for ``unblend.transformer.MyGroupNorm`` on MPS in FP16/BF16.

    The original transposes ``(B, T, C) -> (B, C, T)``, runs ``GroupNorm``
    with ``num_groups=1``, transposes back. Since ``num_groups=1`` is just
    per-batch normalisation, we fold the operation onto ``(B, T*C)``
    directly and skip the transposes — the channel-last (``_chlast``)
    kernels broadcast the affine over the trailing ``C`` axis (index
    ``i % C`` instead of channel-first's ``i / N``). Inherits the affine
    cache, dispatch heuristic, and multi-stage reduction from
    :class:`MetalGroupNorm`; only the apply kernels differ.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize a ``(B, T, C)`` tensor per batch element without transposing.

        :param x: Input tensor of shape ``(B, T, C)``
        :return: Normalized, affine-transformed tensor with the same shape as ``x``
        """
        # FP32: replicate the original's transpose-then-GroupNorm contract via
        # F.group_norm directly (affine params are FP32, so this matches).
        if x.dtype == torch.float32:
            x = x.transpose(1, 2)
            x = F.group_norm(x, 1, self.weight, self.bias, self.eps)
            return x.transpose(1, 2)

        # Non-FP32 on a non-MPS device, or an unsupported low-precision dtype —
        # fall back via FP32 cast plus the transpose dance the original did.
        # (The metal-LP path is handled below.)
        if not _is_metal_lp(x):
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

        if self._use_single_stage(B, per_batch):
            kernel = _get_kernel("group_norm_g1_chlast", x.dtype)
            tgs = _pow2_tgs(kernel.max_threads_per_threadgroup)
            while tgs > 1 and tgs > per_batch:
                tgs //= 2
            out = torch.empty_like(x_contig)
            kernel(
                out,
                x_contig,
                weight,
                bias,
                C,
                per_batch,
                float(self.eps),
                threads=B * tgs,
                group_size=tgs,
            )
            return out.view_as(x)

        meanvar, num_tiles = self._multi_stage_meanvar(
            x_contig, B, per_batch, per_batch
        )
        out = torch.empty_like(x_contig)
        k3 = _get_kernel("apply_norm_chlast", x.dtype)
        tgs3 = _pow2_tgs(k3.max_threads_per_threadgroup)
        k3(
            out,
            x_contig,
            meanvar,
            weight,
            bias,
            per_batch,
            num_tiles,
            C,
            threads=B * num_tiles * tgs3,
            group_size=tgs3,
        )
        return out.view_as(x)


class FusedDConvLayer(nn.Module):
    """One DConv sub-layer (formerly an ``nn.Sequential`` of 7 ops) folded
    into 4 calls: ``conv1 → fused_norm_gelu → conv2 → fused_norm_glu_ls_resid``.

    The original layer was::

        Conv1d(C, hidden) → GroupNorm → GELU → Conv1d(hidden, 2C)
            → GroupNorm → GLU → LayerScale → +residual_x

    We keep the two convs as PyTorch ops (MPS matmul is already fast) and
    fold every non-conv step into our two big fused Metal kernels. Saves
    ~5 intermediate tensor writes/reads per call. The residual addition is
    absorbed into the second fusion.
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

        :param conv1: First pointwise convolution (``C -> hidden``)
        :param norm1: GroupNorm following ``conv1``; fused with the GELU
        :param conv2: Second pointwise convolution (``hidden -> 2C``)
        :param norm2: GroupNorm following ``conv2``; fused into the envelope op
        :param layer_scale_param: Per-channel LayerScale tensor of shape ``(C,)``
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

        We expect the canonical layout (``norm=True``, which is the HTDemucs
        default). Anything else and the swap is skipped at the
        apply_metal_optimizations level.

        :param seq: The 7-op DConv ``nn.Sequential`` to fold
        :return: A new :class:`FusedDConvLayer` mirroring ``seq``
        :raises ValueError: If ``seq`` does not have exactly 7 ops
        :raises TypeError: If any op is not of the expected type for its slot
        """
        # Local import; ``LayerScale`` only ships with the transformer module.
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
        # Local import to break the unblend.blocks <-> metal cycle.
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
    """Replacement for ``unblend.blocks.HEncLayer`` that uses fused Metal
    kernels for low-precision (FP16/BF16) inference on MPS. Same forward contract.

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

        # Carry forward the geometry/flags ``apply.py`` and the model's
        # forward_core inspect on the layer instance.
        self.kernel_size = layer.kernel_size
        self.stride = layer.stride
        self.empty = layer.empty
        self.freq = layer.freq
        self.norm = layer.norm
        self.pad = layer.pad

        self.conv = layer.conv
        if layer.empty:
            return

        # norm1 + gelu fusion
        if isinstance(layer.norm1, nn.GroupNorm) and layer.norm1.num_groups == 1:
            self.norm1 = FusedGroupNormGelu.from_groupnorm(layer.norm1)
            self._fused_gelu = True
        else:
            self.norm1 = layer.norm1
            self._fused_gelu = False

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

        # DConv fusion
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
            y = y + inject
        # If norm1 absorbs gelu we just call it; otherwise apply gelu by hand.
        if self._fused_gelu:
            y = self.norm1(y)
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
    """Replacement for ``unblend.blocks.HDecLayer`` using fused Metal kernels.

    The ``glu(norm1(rewrite(...)))`` pattern is fused. We do NOT fuse the
    final ``gelu(norm2(conv_tr(...)))`` because the ``last`` flag (mutated
    by MultiWrap) decides whether GELU runs at all — keeping that switch
    in Python keeps things simple. ``norm2`` itself is still
    ``MetalGroupNorm`` (handled by the outer swap pass).
    """

    def __init__(self, layer: nn.Module) -> None:
        """
        Build a fused decoder layer from an ``HDecLayer``, folding the GLU norm path.

        :param layer: Source ``HDecLayer`` whose conv_tr, rewrite, norms, flags, and
            inner DConv are carried forward or replaced with fused equivalents
        """
        super().__init__()
        from ..blocks import DConv

        # Carry-forward attributes
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
        self.norm2 = layer.norm2  # left to be swapped to MetalGroupNorm separately
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
        z = self.norm2(self.conv_tr(y))
        if self.freq:
            if self.pad:
                z = z[..., self.pad : -self.pad, :]
        else:
            z = z[..., self.pad : self.pad + length]
        if not self.last:
            z = F.gelu(z)
        return z, y


class MetalMultiheadAttention(nn.Module):
    """Replacement for ``nn.MultiheadAttention`` on MPS in FP16/BF16.

    Q/K/V/output linear projections stay in the parameter dtype, and the
    attention itself runs as a manual matmul → softmax → matmul in that same
    dtype — MPSGraph's tuned GEMM beats both PyTorch's fused MPS SDPA
    (~2.5x) and every hand-written simdgroup-MMA kernel we measured at the
    cross-transformer's shapes (attention here is GEMM-throughput-bound;
    even a zero-memory-traffic FlashAttention-2 kernel measured slower than
    the full matmul path). Masked/causal calls fall back to the wrapped MHA.
    """

    def __init__(self, mha: nn.MultiheadAttention) -> None:
        """
        Wrap an ``nn.MultiheadAttention``, sharing its projection params and keeping it as fallback.

        :param mha: Source multihead attention whose weights are reused and which
            handles any path this wrapper does not optimise
        """
        super().__init__()
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = mha.embed_dim // mha.num_heads
        self.batch_first = mha.batch_first
        self._packed_qkv = mha.in_proj_weight is not None
        if self._packed_qkv:
            self.in_proj_weight = mha.in_proj_weight
            self.in_proj_bias = mha.in_proj_bias
        else:
            self.q_proj_weight = mha.q_proj_weight
            self.k_proj_weight = mha.k_proj_weight
            self.v_proj_weight = mha.v_proj_weight
            self.in_proj_bias = mha.in_proj_bias
        self.out_proj = mha.out_proj
        # Held for any path we don't optimise (masks, need_weights, etc.).
        self._fallback = mha

    @classmethod
    def from_mha(cls, mha: nn.MultiheadAttention) -> "MetalMultiheadAttention":
        """
        Build a :class:`MetalMultiheadAttention` from an existing multihead attention.

        :param mha: Source ``nn.MultiheadAttention`` to wrap
        :return: A new :class:`MetalMultiheadAttention` mirroring ``mha``
        """
        return cls(mha)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, None]:
        """
        Run multihead attention, keeping projections in the input dtype on MPS FP16/BF16.

        Optimised only for the batch-first, packed-QKV, unmasked, non-weight-returning
        case on MPS in FP16/BF16; every other configuration defers to the wrapped MHA.

        :param query: Query tensor of shape ``(B, Lq, embed_dim)`` (batch-first)
        :param key: Key tensor of shape ``(B, Lk, embed_dim)``
        :param value: Value tensor of shape ``(B, Lk, embed_dim)``
        :param key_padding_mask: Optional key padding mask; presence forces the fallback path
        :param need_weights: If ``True``, forces the fallback path (this wrapper returns no weights)
        :param attn_mask: Optional attention mask; presence forces the fallback path
        :param average_attn_weights: Forwarded to the fallback MHA when used
        :param is_causal: If ``True``, forces the fallback path
        :return: A ``(output, None)`` pair; the output has shape ``(B, Lq, embed_dim)``
        """
        # Bail to the wrapped MHA for shapes/configs we don't optimise. Masks
        # and is_causal go there too: nn.MultiheadAttention's mask contract
        # (bool True = *dis*allowed, 3D (B*H, L, S) layout) is the opposite of
        # F.scaled_dot_product_attention's, and the wrapped module is the only
        # path that reproduces it exactly. Demucs itself never passes masks.
        if (
            need_weights
            or key_padding_mask is not None
            or attn_mask is not None
            or is_causal
            or not self.batch_first
            or not self._packed_qkv
            or query.device.type != "mps"
            or query.dtype not in _LP_DTYPES
        ):
            return self._fallback(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )

        is_self_attn = query is key and key is value
        E = self.embed_dim
        H = self.num_heads
        D = self.head_dim
        B = query.size(0)
        Lq = query.size(1)

        # Projections in the input dtype (MPS BF16/FP16 matmul is fast). For
        # self-attention the packed (B, L, 3E) QKV is contiguous after the
        # linear; cross-attention does three separate linears.
        if is_self_attn:
            qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            Lk = Lq
        else:
            wq = self.in_proj_weight[:E]
            wk = self.in_proj_weight[E : 2 * E]
            wv = self.in_proj_weight[2 * E :]
            if self.in_proj_bias is not None:
                bq = self.in_proj_bias[:E]
                bk = self.in_proj_bias[E : 2 * E]
                bv = self.in_proj_bias[2 * E :]
            else:
                bq = bk = bv = None
            q = F.linear(query, wq, bq)
            k = F.linear(key, wk, bk)
            v = F.linear(value, wv, bv)
            Lk = k.size(1)

        q = q.view(B, Lq, H, D).transpose(1, 2)
        k = k.view(B, Lk, H, D).transpose(1, 2)
        v = v.view(B, Lk, H, D).transpose(1, 2)

        # Manual SDPA via matmul. PyTorch's MPS fused SDPA has measurable
        # per-call dispatch overhead at the cross-transformer's B=1 shapes;
        # decomposing it into matmul → softmax → matmul lets us:
        #   1. Run both matmuls in FP16/BF16 (MPS matmul has fast LP paths)
        #   2. Use ``F.softmax`` directly — PyTorch's MPS softmax on a
        #      low-precision input *already promotes to FP32 internally*,
        #      so explicitly casting first is redundant (verified: MAE
        #      identical between native and FP32-cast softmax in both dtypes).
        # Wins ~6 ms/forward for FP16 and ~7 ms/forward for BF16 vs the
        # FP32-cast-everything SDPA path. (Masked/causal calls never reach
        # here — they bailed to the wrapped MHA above.)
        #
        # Pre-scale q so the unscaled logits never materialise in
        # FP16/BF16 — mathematically identical, but the matmul output
        # stays D**0.5 smaller, removing the overflow exposure.
        scale = D**-0.5
        scores = torch.matmul(q * scale, k.transpose(-2, -1))
        scores = F.softmax(scores, dim=-1)
        out = torch.matmul(scores, v)

        out = out.transpose(1, 2).reshape(B, Lq, self.embed_dim)
        out = self.out_proj(out)
        return out, None


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def apply_metal_optimizations(model: nn.Module) -> dict[str, int]:
    """
    Replace low-precision-slow ops with Metal-backed equivalents in-place.

    Idempotent (safe to call multiple times) and only swaps modules whose
    forward semantics we know how to preserve. If the Metal shaders fail to
    compile, warns and returns without modifying the model.

    Two passes:

    1. **Layer-level fusion**: replace every ``HEncLayer`` / ``HDecLayer``
       with a fused version that absorbs the ``gelu(norm1(...))`` and
       ``glu(norm2(rewrite(...)))`` patterns into single Metal kernels.
       The fused layer also recursively replaces its inner ``DConv`` with
       a :class:`FusedDConv` whose sub-layers fuse all 7 of the original
       ``Conv → GroupNorm → GELU → Conv → GroupNorm → GLU → LayerScale``
       ops down to two Metal kernel launches plus the two convs.
    2. **Module-level fallback swap** for everything not already covered:
       remaining plain ``nn.GroupNorm`` (transformer), ``MyGroupNorm``,
       and ``nn.MultiheadAttention``.

    Two passes are needed because the layer fusion both *consumes* the
    GroupNorms it fuses and *leaves behind* a few that the second pass
    still needs to replace (e.g. ``HDecLayer.norm2``, transformer norms).

    :param model: Model to mutate in place, swapping eligible submodules
    :return: A mapping from swap kind to the number of modules replaced, for
        diagnostic logging
    """
    from ..blocks import HDecLayer, HEncLayer
    from ..transformer import MyGroupNorm

    counts = {
        "h_enc_layer": 0,
        "h_dec_layer": 0,
        "fused_dconv": 0,
        "group_norm": 0,
        "my_group_norm": 0,
        "multi_head_attention": 0,
    }

    # Eagerly compile every Metal kernel before touching the model, so a
    # broken shader toolchain surfaces here — where we can fall back wholesale
    # to native PyTorch ops (slower but correct) — instead of mid-forward
    # after modules were already swapped. Compilation is cached per
    # (source, dtype), so on success the first forward hits warm caches.
    dtypes = {p.dtype for p in model.parameters()} & _LP_DTYPES or _LP_DTYPES
    try:
        for kernel_name in _HTDEMUCS_KERNELS:
            for dtype in dtypes:
                _get_kernel(kernel_name, dtype)
    except Exception as exc:
        warnings.warn(
            f"Metal kernel compilation failed ({exc}); skipping Metal "
            "optimizations and keeping native PyTorch ops.",
            RuntimeWarning,
        )
        return counts

    # Pass 1: layer-level fusion.
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
                    # Count nested DConv replacement too, for visibility.
                    if isinstance(getattr(replacement, "dconv", None), FusedDConv):
                        counts["fused_dconv"] += 1
                except Exception as exc:
                    logger.debug(
                        "Metal fusion failed for %s, leaving it unfused: %s",
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
                        "Metal fusion failed for %s, leaving it unfused: %s",
                        type(child).__name__,
                        exc,
                        exc_info=True,
                    )
            if replacement is not None:
                params = list(child.parameters())
                if params:
                    replacement.to(device=params[0].device)
                setattr(mod, name, replacement)
            else:
                _walk_layers(child)

    _walk_layers(model)

    # Pass 2: replace remaining plain ops (transformer norms, MyGroupNorm,
    # MultiheadAttention, and any GroupNorm a fused layer didn't absorb).
    def _walk_modules(mod: nn.Module) -> None:
        """
        Recursively swap remaining ``MyGroupNorm``/``GroupNorm``/``MultiheadAttention`` children.

        :param mod: Module whose children are walked and swapped in place
        """
        for name, child in list(mod.named_children()):
            # Don't descend into modules this pass already produced (on this
            # or an earlier call): ``MetalMultiheadAttention`` keeps the
            # original MHA inside as ``_fallback``, which a repeat walk would
            # otherwise re-wrap into a nested wrapper.
            if isinstance(
                child, (MetalGroupNorm, MetalMyGroupNorm, MetalMultiheadAttention)
            ):
                continue
            replacement: nn.Module | None = None
            if isinstance(child, MyGroupNorm):
                if child.num_groups == 1 and child.affine:
                    replacement = MetalMyGroupNorm(child)
                    counts["my_group_norm"] += 1
            elif type(child) is nn.GroupNorm:
                if child.num_groups == 1 and child.affine:
                    replacement = MetalGroupNorm.from_groupnorm(child)
                    counts["group_norm"] += 1
            elif type(child) is nn.MultiheadAttention:
                replacement = MetalMultiheadAttention.from_mha(child)
                counts["multi_head_attention"] += 1
            if replacement is not None:
                params = list(child.parameters())
                if params:
                    replacement.to(device=params[0].device)
                setattr(mod, name, replacement)
            else:
                _walk_modules(child)

    _walk_modules(model)
    return counts


__all__ = [
    "MetalGroupNorm",
    "metal_rms_norm",
    "MetalMyGroupNorm",
    "MetalMultiheadAttention",
    "FusedGroupNormGelu",
    "FusedGroupNormGlu",
    "FusedNormGluLayerScaleResid",
    "FusedDConvLayer",
    "FusedDConv",
    "FusedHEncLayer",
    "FusedHDecLayer",
    "apply_metal_optimizations",
]
