"""Loader and public wrappers for the compiled BioCUDA PyTorch extension."""

from __future__ import annotations

import importlib

import torch

try:
    _C = importlib.import_module("biocuda_autotune._C")
except Exception:
    _C = None


def is_available() -> bool:
    return _C is not None and torch.cuda.is_available()


def _require_ext():
    if _C is None:
        raise RuntimeError(
            "BioCUDA CUDA extension is not built. Install with `pip install -e .` "
            "from an environment that has PyTorch, CUDA, and a working compiler."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("BioCUDA CUDA extension requires a CUDA device.")
    return _C


def hamming_popc(a: torch.Tensor, b: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    """G4 Hamming POPC kernel. Inputs must be contiguous CUDA uint32 tensors."""

    return _require_ext().hamming_popc(a.contiguous(), b.contiguous(), int(block_size))


def smith_waterman_score(
    a: torch.Tensor,
    b: torch.Tensor,
    match: int = 2,
    mismatch: int = -1,
    gap_open: int = 2,
    gap_extend: int = 1,
    block_size: int = 128,
) -> torch.Tensor:
    """G5 Smith-Waterman score kernel. Inputs are CUDA uint8 encoded sequences."""

    return _require_ext().sw_score(
        a.contiguous(),
        b.contiguous(),
        int(match),
        int(mismatch),
        int(gap_open),
        int(gap_extend),
        int(block_size),
    )


def xor_permute(x: torch.Tensor, mask: int) -> torch.Tensor:
    """G1 XOR permutation inside each 32-element group."""

    return _require_ext().xor_permute(x.contiguous(), int(mask))


def reverse32(x: torch.Tensor) -> torch.Tensor:
    """G2 reverse indexing inside each 32-element group."""

    return _require_ext().reverse32(x.contiguous())


def affine_scan32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """G7 affine monoid inclusive scan inside each 32-element group."""

    return _require_ext().affine_scan32(a.contiguous(), b.contiguous())
