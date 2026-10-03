# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class UnblendError(Exception):
    """
    Base exception class for all unblend-specific errors.
    """


class LoadAudioError(UnblendError):
    """
    Exception raised when audio loading fails.
    """


class ModelLoadingError(UnblendError):
    """
    Exception raised when model loading fails.
    """


class ValidationError(UnblendError, ValueError):
    """
    Exception raised when a parameter value is invalid. Also a ``ValueError``,
    so code catching the built-in keeps working.
    """
