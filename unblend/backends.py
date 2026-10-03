# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Base class and builder registry for separation architectures.
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch import nn


class ASSModel(nn.Module):
    """
    Base class for every source-separation architecture.

    Subclasses set ``sources``, ``samplerate``, ``audio_channels`` and
    ``max_allowed_segment``, take ``(batch, channels, samples)`` audio in
    ``forward`` and return ``(batch, stems, channels, samples)`` estimates.
    """

    #: Name of the hot-path method :meth:`enable_compiled_core` wraps.
    core_name = "forward_core"

    #: Whether the caller must apply track-level normalization. Most models
    #: are trained on raw audio and opt out.
    external_normalization = True

    #: Whether FP16 can't be trusted with a chunk that is mostly silence
    #: around a brief sound, so separation redoes such chunks in FP32. True
    #: for models that normalize each chunk by its own spread.
    sparse_chunks_need_fp32 = False

    def __init__(self) -> None:
        """
        Initialize the inference-interface defaults.
        """
        super().__init__()
        self.sources: list[str] = []
        self.samplerate: int = 44100
        self.audio_channels: int = 2
        self.max_allowed_segment: float = 10.0
        self._fixed_batch_shape: bool = False

    def __getstate__(self) -> dict:
        """
        Pickle and ``deepcopy`` with the hot path eager: a compiled core is
        bound to this instance, so a copy keeping it would run this model's
        weights (and pickling it fails outright). The copy can compile again.

        :return: The state to pickle.
        """
        state = super().__getstate__()
        if "_eager_core" in state:
            state.pop(self.core_name, None)
            state.pop("_eager_core")
            state["_fixed_batch_shape"] = False
        return state

    def prefill_inference_caches(self) -> None:
        """
        Fill lazily-built caches before CUDAGraph capture. Default no-op.
        """

    def enable_compiled_core(self) -> None:
        """
        Wrap the hot path in ``torch.compile``.

        STFT/iSTFT stay eager — they compile poorly and don't help
        steady-state throughput.
        """
        if not hasattr(self, "_eager_core"):
            self._eager_core = getattr(self, self.core_name)
        self.prefill_inference_caches()
        setattr(
            self,
            self.core_name,
            torch.compile(self._eager_core, mode="reduce-overhead"),
        )
        self._fixed_batch_shape = True

    def disable_compiled_core(self) -> None:
        """
        Restore the eager hot path so a retry does not double-wrap it.
        """
        eager = getattr(self, "_eager_core", None)
        if eager is not None:
            setattr(self, self.core_name, eager)
            del self._eager_core
        self._fixed_batch_shape = False


_BUILDERS: dict[str, "Callable[..., ASSModel]"] = {}
_ARCHITECTURES: dict[str, frozenset[str]] = {}


class CustomKernelModule(nn.Module):
    """
    Base for modules that carry a hand-written CUDA/Metal fast path.

    Subclasses check ``self.use_custom_kernels`` before dispatching to a kernel
    and fall back to stock PyTorch ops when it is cleared, so
    :func:`disable_custom_kernels` can put a whole model back on the reference
    path. That is what ``Separator(custom_kernels=False)`` and
    ``UNBLEND_CUSTOM_KERNELS=0`` do, and it is the supported way to tell
    whether a kernel is responsible for a numerical difference.
    """

    def __init__(self) -> None:
        """
        Enable this module's custom-kernel fast path by default.
        """
        super().__init__()
        self.use_custom_kernels: bool = True


def disable_custom_kernels(model: nn.Module) -> int:
    """
    Turn off the custom-kernel fast path on every module that has one.

    :param model: Model to walk; any :class:`CustomKernelModule` in it is
        switched to its reference implementation.
    :return: Number of modules switched off.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, CustomKernelModule):
            module.use_custom_kernels = False
            count += 1
    return count


def register_backend(
    name: str,
    builder: "Callable[..., ASSModel]",
    architectures: "Iterable[str]",
) -> None:
    """
    Register a backend.

    :param name: Backend name used by ``metadata.yaml``.
    :param builder: Builder callable.
    :param architectures: Architecture names the builder accepts.
    :raises ValueError: If an architecture is already owned by another backend.
    """
    names = frozenset(architectures)
    for architecture in names:
        owner = backend_for_architecture(architecture)
        if owner is not None and owner != name:
            raise ValueError(
                f"Architecture {architecture!r} is already registered to "
                f"backend {owner!r}."
            )
    _BUILDERS[name] = builder
    _ARCHITECTURES[name] = names


def backend_for_architecture(architecture: str) -> str | None:
    """
    Return the backend that builds an architecture.

    :param architecture: Architecture name.
    :return: Backend name, or ``None`` if unregistered.
    """
    for backend, names in _ARCHITECTURES.items():
        if architecture in names:
            return backend
    return None


def build(
    backend: str,
    architecture: str,
    config: dict,
    *,
    sources: list[str],
    samplerate: int,
    segment_samples: int,
    state: dict | None = None,
) -> ASSModel:
    """
    Construct a registered architecture.

    :param backend: Registered backend name.
    :param architecture: Architecture name.
    :param config: Constructor kwargs.
    :param sources: Output stem names.
    :param samplerate: Sample rate.
    :param segment_samples: Training chunk length in samples.
    :param state: Checkpoint state dict to load strictly, or ``None``.
    :return: The constructed model.
    :raises KeyError: If ``backend`` is not registered.
    """
    return _BUILDERS[backend](
        architecture,
        config,
        sources=sources,
        samplerate=samplerate,
        segment_samples=segment_samples,
        state=state,
    )


def tensor_version(t: torch.Tensor) -> int:
    """
    A tensor's in-place version counter, for invalidating derived copies.

    Inference tensors (created under ``torch.inference_mode``) have no counter;
    they report ``-1``, so only replacing the tensor, which callers track by
    identity (see :func:`tensor_record`), invalidates their copies.

    :param t: Tensor to inspect.
    :return: The version counter, or ``-1`` for an inference tensor.
    """
    return -1 if t.is_inference() else t._version


def state_without(module: torch.nn.Module, *names: str) -> dict:
    """
    A module's pickle state minus derived caches and their tensor records.

    ``copy.deepcopy`` and pickling both go through ``__getstate__``. A record
    copied along would name the copy's own parameters, but with the original's
    version counts, which the copy's counters restart below: after enough
    in-place edits the stale cache would match again. Dropping both makes the
    copy rebuild them.

    :param module: The module being pickled or copied.
    :param names: Attribute names to leave out.
    :return: The state to pickle.
    """
    state = torch.nn.Module.__getstate__(module)
    for name in names:
        state.pop(name, None)
    return state


def tensor_record(*tensors: torch.Tensor) -> tuple[tuple[torch.Tensor, int], ...]:
    """
    Record tensors and their versions, to tell later whether derived copies
    are stale.

    Holds the tensors themselves, not their ids: CPython reuses a freed
    object's id, so a replacement could otherwise pass for the original.

    :param tensors: The source tensors (e.g. a layer's weight and bias).
    :return: A record for :func:`tensor_record_matches`.
    """
    return tuple((t, tensor_version(t)) for t in tensors)


def tensor_record_matches(
    record: tuple[tuple[torch.Tensor, int], ...] | None, *tensors: torch.Tensor
) -> bool:
    """
    Whether ``tensors`` are the same objects, unchanged, as when recorded.

    :param record: A record from :func:`tensor_record`, or ``None``.
    :param tensors: The current source tensors.
    :return: True if every tensor is the recorded one at the recorded version.
    """
    return (
        record is not None
        and len(record) == len(tensors)
        and all(r is t and v == tensor_version(t) for (r, v), t in zip(record, tensors))
    )
