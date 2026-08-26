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
