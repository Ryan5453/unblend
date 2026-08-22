# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The contract every separation architecture implements.

Inference itself needs only four members — ``sources``, ``samplerate``,
``max_allowed_segment`` and ``forward`` — which is why ``apply_model`` can
drive a foreign architecture unchanged. Everything beyond that is optional
acceleration: cache prefill, a compiled hot path, and batch-shape preferences.

Those optional pieces used to live in ``Separator`` as ``isinstance`` branches
over ``HTDemucs``/``_RoformerBase``, which meant a third architecture had to
edit the API layer to be usable. They are declared here and implemented by each
architecture instead, so adding one touches its own module, the builder
registry, and ``metadata.json`` — and nothing else.

The protocol is structural on purpose: architectures are *not* required to
inherit from it. HTDemucs is reconstructed from legacy pickles that name its
class directly, so changing its bases would be a compatibility hazard for no
benefit. Callers use :func:`prefill_caches`/:func:`enable_compiled_core`/
:func:`disable_compiled_core`, which no-op for models that implement nothing.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class SeparationModel(Protocol):
    """
    Structural interface for a separation architecture.

    Only the four required members are needed to run inference; the optional
    hooks below exist so an architecture can opt into acceleration without the
    API layer knowing its name.
    """

    #: Output stem names, in the order ``forward`` emits them.
    sources: list[str]
    #: Sample rate the weights operate at.
    samplerate: int
    #: Longest chunk, in seconds, the model may be fed in one pass.
    max_allowed_segment: float

    def forward(self, mix: Tensor) -> Tensor:
        """
        Separate a batch of mixtures.

        :param mix: ``(batch, channels, samples)`` audio.
        :return: ``(batch, stems, channels, samples)`` estimates.
        """
        ...


def prefill_caches(model: torch.nn.Module) -> None:
    """
    Materialise any lazily-built inference caches before compilation.

    Rotary tables and positional embeddings are allocated on first use. Under
    CUDAGraph capture that allocation happens inside the graph and poisons it,
    so architectures that build caches lazily populate them here instead.

    :param model: The model to prepare; no-ops if it declares no caches.
    """
    hook = getattr(model, "prefill_inference_caches", None)
    if hook is not None:
        hook()


def enable_compiled_core(model: torch.nn.Module) -> None:
    """
    Swap in the architecture's compiled hot path, if it defines one.

    Deliberately narrower than compiling ``forward``: STFT/iSTFT are a poor
    fit for Inductor and inflate compile time without improving steady-state
    throughput, so each architecture chooses its own core.

    :param model: The model to compile; no-ops if it defines no compiled core.
    """
    hook = getattr(model, "enable_compiled_core", None)
    if hook is not None:
        hook()


def disable_compiled_core(model: torch.nn.Module) -> None:
    """
    Restore the eager hot path so a retry does not double-wrap it.

    :param model: The model to restore; always clears the fixed-batch flag.
    """
    hook = getattr(model, "disable_compiled_core", None)
    if hook is not None:
        hook()
    model._fixed_batch_shape = False


def prefers_power_of_two_batch(model: torch.nn.Module) -> bool:
    """
    Whether the model wants its compiled batch size rounded down to a power
    of two before calibration sweeps it.

    :param model: The model to query.
    :return: ``True`` if the architecture requests power-of-two batches.
    """
    return bool(getattr(model, "prefers_power_of_two_batch", False))


#: Builders for architectures whose weights ship as one verified Safetensors
#: checkpoint plus a config, keyed by ``metadata.json``'s ``backend`` field.
#: Registering here is what makes a new architecture loadable — the repository
#: dispatches through this table instead of naming families inline.
_BUILDERS: dict[str, "Callable[..., torch.nn.Module]"] = {}


def register_backend(name: str, builder: "Callable[..., torch.nn.Module]") -> None:
    """
    Register a single-checkpoint architecture family.

    :param name: The ``backend`` value used in ``metadata.json``.
    :param builder: Callable with ``build_*``'s signature.
    """
    _BUILDERS[name] = builder


def single_checkpoint_backends() -> frozenset[str]:
    """
    Backends whose weights are one Safetensors checkpoint plus a config.

    :return: The registered backend names.
    """
    return frozenset(_BUILDERS)


def build(
    backend: str,
    architecture: str,
    config: dict,
    *,
    sources: list[str],
    samplerate: int,
    segment_samples: int,
    state: dict | None = None,
) -> torch.nn.Module:
    """
    Construct a registered architecture and (optionally) load its weights.

    :param backend: Registered backend name.
    :param architecture: Architecture within that backend.
    :param config: Constructor kwargs from ``metadata.json``.
    :param sources: Output stem names.
    :param samplerate: Sample rate the checkpoint operates at.
    :param segment_samples: Training chunk length in samples.
    :param state: Checkpoint state dict to load strictly, or ``None``.
    :return: The constructed model in eval mode.
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
