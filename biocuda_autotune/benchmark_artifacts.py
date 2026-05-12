"""Static benchmark artifacts captured from the restored notebook."""

NOTEBOOK_SUMMARY = {
    "version": "39.12.0-triton",
    "spec": "BioCUDA-v38 + Triton integration",
    "gpu_key": "T4",
    "gpu_name": "Tesla T4",
    "sm_arch": "sm_75",
    "arch": "turing",
    "detection_mode": "live",
    "cupy_available": True,
    "cupti_available": False,
    "triton_available": True,
    "kernel_modules_count": 10,
    "falsification_t0_n_c": {"passed": 173, "total": 173, "pass_rate": 1.0, "failures": []},
    "tier_m": {"passed": 4, "evaluated": 5, "skipped": 0, "total": 5},
    "tau_tc_calibration": {
        "gpu_key": "T4",
        "sm_arch": "sm_75",
        "mma_shape": [16, 8, 8],
        "iters": 4096,
        "cycles_samples": [110813, 110815, 110824, 110824, 110825, 110826, 110941],
        "cycles_median": 110824,
        "tau_tc_cycles_per_mma": 27.056640625,
        "flops_per_mma": 2048,
        "sm_clock_hz": 1590000000.0,
        "sm_count": 40,
        "w_max_used": 64,
        "tflops_estimated_peak": 65.1264,
    },
    "kernel_sanity_check": {
        "available": True,
        "hamming_cpu": 65505,
        "hamming_gpu": 65505,
        "hamming_match": True,
        "sw_score": 22,
        "sw_cpu": 22,
        "sw_match": True,
    },
    "g34_omega_size": 1536,
    "g19_reuse_thresholds": {
        "A_min_smem": 22.214285714285715,
        "A_min_shfl": 161.5,
        "A_min_l2": 1.9545454545454546,
    },
}

BIO_KERNEL_BENCHMARKS = [
    {"kernel": "Hamming", "size": "256K words / 1 MB", "default_us": 468.0, "biocuda_us": 474.4, "speedup": 0.987, "choice": "block=1024"},
    {"kernel": "Hamming", "size": "1024K words / 4 MB", "default_us": 1281.1, "biocuda_us": 1269.3, "speedup": 1.009, "choice": "block=1024"},
    {"kernel": "Hamming", "size": "4096K words / 16 MB", "default_us": 7915.8, "biocuda_us": 7992.3, "speedup": 0.990, "choice": "block=1024"},
    {"kernel": "Smith-Waterman", "size": "64 x 64", "default_us": 741.5, "biocuda_us": 747.6, "speedup": 0.992, "choice": "block=256"},
    {"kernel": "Smith-Waterman", "size": "128 x 128", "default_us": 2354.4, "biocuda_us": 2298.3, "speedup": 1.024, "choice": "block=256"},
    {"kernel": "Smith-Waterman", "size": "256 x 256", "default_us": 4432.6, "biocuda_us": 3858.2, "speedup": 1.149, "choice": "block=256"},
]

MATMUL_BENCHMARKS = [
    {"size": 512, "cublas_us": 90.2, "bio_pytorch_us": 60.4, "bio_pytorch_tflops": 4.44, "exp3_winner": "BC+PT"},
    {"size": 1024, "cublas_us": 937.4, "bio_pytorch_us": 79.1, "bio_pytorch_tflops": 27.16, "exp3_winner": "BC+PT"},
    {"size": 2048, "cublas_us": 5005.2, "bio_pytorch_us": 466.3, "bio_pytorch_tflops": 36.85, "exp3_winner": "BC+PT"},
    {"size": 4096, "cublas_us": 35842.3, "bio_pytorch_us": 6744.4, "bio_pytorch_tflops": 20.38, "exp3_winner": "BC+PT"},
]


def notebook_summary():
    return NOTEBOOK_SUMMARY.copy()
