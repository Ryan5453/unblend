# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class UnblendError(Exception):
    """
    Base exception class for all unblend-specific errors.
    """

    pass


class LoadAudioError(UnblendError):
    """
    Exception raised when audio loading fails.
    """

    pass


class ModelLoadingError(UnblendError):
    """
    Exception raised when model loading fails.
    """

    pass


class ValidationError(UnblendError, ValueError):
    """
    Exception raised when a parameter value is invalid.

    Also a ``ValueError`` so callers catching the builtin keep working.
    """

    pass


# Backwards-compatible alias for the pre-rename name (when the package was
# ``demucs``). Kept so existing callers that catch ``DemucsError`` keep working.
DemucsError = UnblendError
