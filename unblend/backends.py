# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The contract every separation architecture implements.

Inference itself needs only four members — ``sources``, ``samplerate``,
``max_allowed_segment`` and ``forward`` — which is why ``apply_model`` can
drive a foreign architecture unchanged. Everything beyond that is optional
acceleration: a compiled hot path and batch-shape preferences.

Those optional pieces used to live in ``Separator`` as ``isinstance`` branches
over ``HTDemucs``/``_RoformerBase``, which meant a third architecture had to
edit the API layer to be usable. They are declared here and implemented by each
architecture instead, so adding one touches its own module, the builder
registry, and ``metadata.yaml`` — and nothing else.

The protocol is structural on purpose: architectures are *not* required to
inherit from it. HTDemucs is reconstructed from legacy pickles that name its
class directly, so changing its bases would be a compatibility hazard for no
benefit. Callers use :func:`enable_compiled_core`/:func:`disable_compiled_core`,
which no-op for models that implement nothing. An architecture with lazily-built
inference caches (rotary tables, positional embeddings) populates them inside
its own ``enable_compiled_core``: allocating them under CUDAGraph capture would
poison the graph, and swapping in the compiled core is the moment that matters.
"""

from __future__ import annotations

from typing import Callable, Iterable, Protocol, runtime_checkable

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
#: checkpoint plus a config, keyed by ``metadata.yaml``'s ``backend`` field.
#: Registering here is what makes a new architecture loadable — the repository
#: dispatches through this table instead of naming families inline.
_BUILDERS: dict[str, "Callable[..., torch.nn.Module]"] = {}

#: The architecture names each backend can build. Populated by
#: ``register_backend`` so registry validation and the architecture -> backend
#: lookup below never hard-code an architecture list.
_ARCHITECTURES: dict[str, frozenset[str]] = {}


def register_backend(
    name: str,
    builder: "Callable[..., torch.nn.Module]",
    architectures: "Iterable[str]",
) -> None:
    """
    Register a single-checkpoint architecture family.

    :param name: The ``backend`` value used in ``metadata.yaml``.
    :param builder: Callable with ``build_*``'s signature.
    :param architectures: The ``architecture`` values ``builder`` accepts.
    :raises ValueError: If an architecture is already owned by another
        backend. Architecture names are globally unique so a registry entry
        can name only its architecture and have the backend derived from it.
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


def single_checkpoint_backends() -> frozenset[str]:
    """
    Backends whose weights are one Safetensors checkpoint plus a config.

    :return: The registered backend names.
    """
    return frozenset(_BUILDERS)


def architectures(backend: str) -> frozenset[str]:
    """
    Architectures a registered backend can build.

    :param backend: Backend name.
    :return: Its architecture names, empty if the backend is unregistered.
    """
    return _ARCHITECTURES.get(backend, frozenset())


def backend_for_architecture(architecture: str) -> str | None:
    """
    The backend that builds a given architecture.

    Lets a registry entry declare only ``architecture``: since every
    architecture belongs to exactly one loader family, ``backend`` is
    redundant information that can be derived instead of restated.

    :param architecture: Architecture name.
    :return: The owning backend, or ``None`` if nothing claims the name.
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
) -> torch.nn.Module:
    """
    Construct a registered architecture and (optionally) load its weights.

    :param backend: Registered backend name.
    :param architecture: Architecture within that backend.
    :param config: Constructor kwargs from ``metadata.yaml``.
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
