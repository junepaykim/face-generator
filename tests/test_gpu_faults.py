"""gpu.is_fatal: which errors end the process's use of the GPU. No torch needed."""

from __future__ import annotations

import pytest

from facegen.render.gpu import describe, is_fatal


class AcceleratorError(RuntimeError):
    """Stands in for torch.AcceleratorError: told by its name."""


class OutOfMemoryError(RuntimeError):
    """Stands in for torch.OutOfMemoryError."""


SEEN = (
    "CUDA error: an illegal memory access was encountered Search for `cudaErrorIllegalAddress'"
    " in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html"
)


@pytest.mark.parametrize(
    "exc",
    [
        AcceleratorError(SEEN),
        RuntimeError(SEEN),
        AcceleratorError(""),
        RuntimeError("CUDA error: device-side assert triggered"),
        RuntimeError("CUDA error: unspecified launch failure"),
        RuntimeError("cuDNN error: CUDNN_STATUS_EXECUTION_FAILED; cudaErrorLaunchFailure"),
    ],
)
def test_a_dead_context_is_fatal(exc: BaseException) -> None:
    assert is_fatal(exc)


@pytest.mark.parametrize(
    "exc",
    [
        OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("CUDA error: out of memory"),
        AttributeError("'NoneType' object has no attribute 'text_encoder'"),
        OSError("model not found in the local cache"),
        KeyboardInterrupt(),
        ValueError("age unknown"),
    ],
)
def test_other_errors_are_not(exc: BaseException) -> None:
    assert not is_fatal(exc)


def test_an_error_raised_from_a_fatal_one_is_fatal() -> None:
    try:
        try:
            raise AcceleratorError(SEEN)
        except RuntimeError as inner:
            raise ValueError("could not draw") from inner
    except ValueError as outer:
        assert is_fatal(outer)


def test_a_cycle_of_causes_ends() -> None:
    a, b = ValueError("a"), ValueError("b")
    a.__cause__, b.__cause__ = b, a
    assert not is_fatal(a)


def test_describe_never_raises() -> None:
    assert isinstance(describe(), str)
