# BioCUDA PyTorch AutoTuner

BioCUDA PyTorch AutoTuner packages the BioCUDA v39 optimizer as a normal PyTorch extension. It combines an import-safe Python formula engine with native CUDA/C++ kernels for the BioCUDA formulas that are real kernel primitives.

## Install

```bash
pip install -e .
```

If CUDA is not visible during installation, the package can still install without the native `_C` extension. To force native CUDA build:

```bash
set FORCE_CUDA=1
pip install -e .
```

To intentionally skip native compilation:

```bash
set BIOCUDA_BUILD_EXT=0
pip install -e .
```

## Quick Start

```python
import torch
from biocuda_autotune import BioCUDAAutoTune, coverage_report, hamming_popc, matmul

tuner = BioCUDAAutoTune(verbose=True)
print(tuner.summary())
print(coverage_report())

a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = matmul(a, b, tuner=tuner)

x = torch.randint(0, 2**31, (1 << 20,), device="cuda", dtype=torch.uint32)
y = torch.randint(0, 2**31, (1 << 20,), device="cuda", dtype=torch.uint32)
dist = hamming_popc(x, y)
```

## Native CUDA Coverage

Implemented as compiled PyTorch CUDA extension:

- G1 XOR permutation: `xor_permute`
- G2 reverse indexing: `reverse32`
- G4 Hamming POPC: `hamming_popc`
- G5 Smith-Waterman score: `smith_waterman_score`
- G7 affine scan: `affine_scan32`

The remaining formulas are exposed as Python formulas, autotuner logic, or measurement-required entries. Use `coverage_report()` to inspect the G1-G35 implementation map.

## Project Layout

- `biocuda_autotune/`: Python API, formula engine, autotuner, extension loader
- `csrc/`: C++/CUDA PyTorch extension sources
- `examples/`: quickstart script
- `tests/`: formula and CUDA extension smoke tests
