"""BioCUDA AutoTune PyTorch extension."""

from .autotune import BioCUDAAutoTune, BioCUDAOptimizer, quick_optimize
from .coverage import FORMULA_COVERAGE, coverage_report
from .core import FormulaEngine, GPUSpec, detect_gpu_spec
from .extension import affine_scan32, hamming_popc, is_available, reverse32, smith_waterman_score, xor_permute
from .ops import benchmark_matmul, matmul

__version__ = "0.1.0"

__all__ = [
    "BioCUDAAutoTune",
    "BioCUDAOptimizer",
    "FormulaEngine",
    "GPUSpec",
    "benchmark_matmul",
    "coverage_report",
    "detect_gpu_spec",
    "FORMULA_COVERAGE",
    "hamming_popc",
    "is_available",
    "matmul",
    "quick_optimize",
    "reverse32",
    "smith_waterman_score",
    "xor_permute",
    "affine_scan32",
    "__version__",
]
