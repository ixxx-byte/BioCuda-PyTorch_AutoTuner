"""BioCUDA AutoTune PyTorch extension."""

from .autotune import BioCUDAAutoTune, BioCUDAOptimizer, quick_optimize
from .benchmark_artifacts import BIO_KERNEL_BENCHMARKS, MATMUL_BENCHMARKS, NOTEBOOK_SUMMARY, notebook_summary
from .calibration import KernelUnavailable, estimate_tc_peak_from_tau, summarize_cycle_samples
from .coverage import FORMULA_COVERAGE, coverage_report
from .core import FormulaEngine, GPU_DB, GPUSpec, detect_gpu_spec
from .extension import affine_scan32, hamming_popc, is_available, reverse32, smith_waterman_score, xor_permute
from .falsification import FalsifyResult, TierMResult, exp3_regret_bound, hill_r2, kendall_tau
from .kernel_specs import KERNEL_SPECS, KernelSpec, dry_run_table, kernel_modules_count
from .ops import benchmark_matmul, matmul

__version__ = "0.1.0"

__all__ = [
    "BioCUDAAutoTune",
    "BioCUDAOptimizer",
    "BIO_KERNEL_BENCHMARKS",
    "FalsifyResult",
    "FormulaEngine",
    "GPU_DB",
    "GPUSpec",
    "KERNEL_SPECS",
    "KernelSpec",
    "KernelUnavailable",
    "MATMUL_BENCHMARKS",
    "NOTEBOOK_SUMMARY",
    "TierMResult",
    "benchmark_matmul",
    "coverage_report",
    "detect_gpu_spec",
    "dry_run_table",
    "estimate_tc_peak_from_tau",
    "exp3_regret_bound",
    "FORMULA_COVERAGE",
    "hamming_popc",
    "hill_r2",
    "is_available",
    "kendall_tau",
    "kernel_modules_count",
    "matmul",
    "notebook_summary",
    "quick_optimize",
    "reverse32",
    "smith_waterman_score",
    "summarize_cycle_samples",
    "xor_permute",
    "affine_scan32",
    "__version__",
]
