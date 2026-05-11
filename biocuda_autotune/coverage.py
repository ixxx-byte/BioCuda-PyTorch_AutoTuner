"""Formula coverage map for BioCUDA v38/v39 packaging.

The spec contains hardware identities, models, measurement protocols, and
kernel algorithms. Only the algorithmic/hardware primitives should become CUDA
kernels; the measurement formulas require Nsight/CUPTI/NVML data.
"""

FORMULA_COVERAGE = {
    "G1": {"status": "cuda_kernel", "entry": "extension.xor_permute", "note": "XOR lane/group permutation"},
    "G2": {"status": "cuda_kernel", "entry": "extension.reverse32", "note": "reverse index per 32-lane group"},
    "G3": {"status": "python_formula", "entry": "FormulaEngine.g3_transactions"},
    "G4": {"status": "cuda_kernel", "entry": "extension.hamming_popc"},
    "G5": {"status": "cuda_kernel", "entry": "extension.smith_waterman_score"},
    "G6": {"status": "python_formula", "entry": "FormulaEngine.g6_roofline_crossover"},
    "G7": {"status": "cuda_kernel", "entry": "extension.affine_scan32"},
    "G8": {"status": "python_formula", "entry": "FormulaEngine.g8_hmm_flops"},
    "G9": {"status": "measurement_required", "entry": "Nsight/MPS counters"},
    "G10": {"status": "measurement_required", "entry": "MI screen over Nsight counters"},
    "G11": {"status": "python_formula", "entry": "Psi/odds helper formulas"},
    "G12": {"status": "python_formula", "entry": "FormulaEngine.g12_crossover"},
    "G13": {"status": "python_formula", "entry": "FormulaEngine.g13_hill"},
    "G14": {"status": "python_formula+measurement_required", "entry": "FormulaEngine.g14_* + CUPTI/NVML counters"},
    "G15": {"status": "python_formula+measurement_required", "entry": "FormulaEngine.g15_hill_occupancy"},
    "G16": {"status": "python_formula", "entry": "FormulaEngine.g16_occupancy"},
    "G17": {"status": "measurement_required", "entry": "warmup microbenchmark fit"},
    "G18": {"status": "python_formula", "entry": "FormulaEngine.g18_critical_path"},
    "G19": {"status": "python_formula", "entry": "FormulaEngine.g19_reuse"},
    "G20": {"status": "python_formula+test_protocol", "entry": "ECC/integrity checks"},
    "G21": {"status": "planned_cuda_kernel", "entry": "stochastic rounding kernels"},
    "G22": {"status": "python_formula", "entry": "EXP3 helpers"},
    "G23": {"status": "python_formula", "entry": "FormulaEngine.g23_e_addr"},
    "G24": {"status": "python_formula+measurement_required", "entry": "FormulaEngine.g24_h_l2_footprint"},
    "G25": {"status": "model", "entry": "partition objective"},
    "G26": {"status": "python_formula+measurement_required", "entry": "bank conflict model"},
    "G27": {"status": "python_formula", "entry": "FormulaEngine.g27_*"},
    "G28": {"status": "measurement_required", "entry": "L1I instruction count/Nsight"},
    "G29": {"status": "python_formula+measurement_required", "entry": "MPS pairwise calibration"},
    "G30": {"status": "python_formula+measurement_required", "entry": "runtime surrogate/CUPTI correction"},
    "G31": {"status": "python_formula+measurement_required", "entry": "mode utility from G17/G10"},
    "G32": {"status": "python_formula", "entry": "benchmark jitter index"},
    "G33": {"status": "python_formula+measurement_required", "entry": "co-schedule predictor"},
    "G34": {"status": "python_autotuner", "entry": "BioCUDAOptimizer.select_matmul_configs"},
    "G35": {"status": "python_formula", "entry": "FormulaEngine.g35_tc_partial"},
}


def coverage_report():
    return FORMULA_COVERAGE.copy()
