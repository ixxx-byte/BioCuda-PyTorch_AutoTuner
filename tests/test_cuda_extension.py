import pytest
import torch

from biocuda_autotune import affine_scan32, hamming_popc, is_available, reverse32, xor_permute


pytestmark = pytest.mark.skipif(not is_available(), reason="BioCUDA CUDA extension is not built or CUDA is unavailable")


def test_xor_and_reverse32():
    x = torch.arange(64, device="cuda", dtype=torch.int32)
    assert torch.equal(xor_permute(x, 1)[0:32], x[torch.arange(32, device="cuda") ^ 1])
    assert torch.equal(reverse32(x)[0:32], torch.flip(x[0:32], dims=[0]))


def test_hamming_popc():
    a = torch.tensor([0xFFFFFFFF, 0x00000000], device="cuda", dtype=torch.uint32)
    b = torch.tensor([0x00000000, 0x00000000], device="cuda", dtype=torch.uint32)
    assert int(hamming_popc(a, b).cpu()[0]) == 32


def test_affine_scan32_shape():
    a = torch.ones(32, device="cuda", dtype=torch.float32)
    b = torch.ones(32, device="cuda", dtype=torch.float32)
    out = affine_scan32(a, b)
    assert tuple(out.shape) == (2, 32)
