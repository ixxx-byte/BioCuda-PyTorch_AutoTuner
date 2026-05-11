# BioCUDA PyTorch AutoTuner

**BioCUDA PyTorch AutoTuner** is a PyTorch extension for bioinformatics-oriented GPU kernel tuning. It packages the BioCUDA v39.12 optimizer as a normal Python/CUDA project: an import-safe formula engine, a zero-shot autotuner, native CUDA/C++ kernels, Triton/PyTorch matmul paths, falsification hooks, and a transparent G1-G35 implementation map.

The project is built around the BioCUDA v38 specification: a deliberately conservative engineering system for GPU kernels used in sequence analysis, dynamic programming, profile models, prefix scans, and tensor-core scoring. The central idea is not "DNA is isomorphic to GPUs". It is narrower and more useful:

> Real bioinformatics kernels have recurring GPU cost structures. Those structures can be described with explicit formulas, classified honestly, pruned before launch, measured after launch, and falsified when the hardware disagrees.

This repository turns that idea into a PyTorch extension.

---

## Why This Exists

Bioinformatics GPU workloads are often a mix of:

- bitwise kernels such as Hamming distance and k-mer comparisons,
- dynamic programming kernels such as Smith-Waterman and edit distance,
- dense scoring kernels such as PWM and Profile-HMM forward passes,
- suffix-array / FM-index construction phases,
- small but latency-sensitive prefix scans,
- matrix multiply variants that compete across cuBLAS, Triton, PyTorch compile, and custom search.

Classic autotuning can benchmark a huge configuration space, but that gets expensive. In the benchmark that motivated this package, a native Triton search over 128 matmul configs took up to **1178 seconds** on Tesla T4. BioCUDA instead tries a zero-shot path:

1. Use hardware formulas to remove impossible or obviously poor configs.
2. Score feasible configs in cycles before compilation.
3. Select a small top-K.
4. Launch and validate with CUDA events, ECC checks, EXP3 winner selection, and jitter analysis.

The package is designed to be useful in two modes:

- **Practical mode:** call `hamming_popc`, `smith_waterman_score`, `matmul`, or `quick_optimize`.
- **Research mode:** inspect `FormulaEngine`, `coverage_report()`, falsification tiers, and benchmark stability.

---

## Core Principles From BioCUDA v38

BioCUDA v38 intentionally removed claims that were too broad or not falsifiable. The specification keeps only what can be tied to hardware, measurement, or explicit modelling assumptions.

### Classification System

| Class | Meaning |
|---|---|
| `[P]` | Proven from algebra or hardware specification; deterministic to verify. |
| `[P+]` | Hardware-verifiable within an explicit tolerance, usually <=10%. |
| `[S]` | Model with stated assumptions and known applicability limits. |
| `[E]` | Empirical parameter that must be measured or fitted on target hardware. |
| `[A]` | Analogy or mnemonic; useful for explanation, not prediction. |

### What v38 Keeps

- Hardware-precise formulas: G1, G4, G5, G16, G35.
- Falsification tiers: T0 algebra, N hardware checks, M model checks, C consistency checks.
- Working autotuning: G34 configuration search plus G22 EXP3-style online selection.
- Real bioinformatics use cases: Smith-Waterman, PWM, HMM, suffix array, Hamming distance, prefix scan.

### What v38 Removed

- Broad DNA-GPU "isomorphism" claims.
- M/D/1 theorem language for GPU scheduling.
- Shannon-cycles as a unit.
- Non-measured lambda derivations.
- Uncheckable biology analogies inside non-bio GPU formulas.

---

## Architecture

```text
biocuda_autotune/
  __init__.py          public API
  core.py              GPUSpec + FormulaEngine G1-G35 helpers
  autotune.py          BioCUDAAutoTune + BioCUDAOptimizer
  ops.py               PyTorch/Triton operations and benchmark helpers
  extension.py         compiled CUDA extension loader
  coverage.py          explicit G1-G35 implementation map

csrc/
  biocuda_ext.cpp      pybind11 / torch extension bindings
  biocuda_kernels.cu   CUDA kernels for kernel-level formulas

examples/
  biocuda_quickstart.py

tests/
  test_formula_engine.py
  test_cuda_extension.py
```

The native extension is named:

```python
biocuda_autotune._C
```

It is loaded through `biocuda_autotune.extension`, so the Python package can still import on machines without a CUDA compiler. CUDA-only functions raise a clear runtime error if `_C` is not built.

---

## Installation

### Editable Install

```bash
pip install -e .
```

### Force CUDA Extension Build

Use this when PyTorch cannot see CUDA during setup, but you know CUDA/NVCC are available:

```bash
set FORCE_CUDA=1
pip install -e .
```

### Pure Python Install

Use this when you only want the formula engine and autotuner logic:

```bash
set BIOCUDA_BUILD_EXT=0
pip install -e .
```

### Optional Triton Path

```bash
pip install -e ".[triton]"
```

---

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

Training helper:

```python
from biocuda_autotune import quick_optimize

model, optimizer, scaler = quick_optimize(
    model,
    learning_rate=1e-3,
    optimizer_name="AdamW",
    compile_mode="default",
)
```

---

## Native CUDA Kernels

Implemented as compiled PyTorch CUDA extension:

| Formula | API | What It Does |
|---|---|---|
| G1 | `xor_permute(x, mask)` | XOR permutation inside each 32-element warp-style group. |
| G2 | `reverse32(x)` | Reverse indexing inside each 32-element group. |
| G4 | `hamming_popc(a, b)` | Binary Hamming distance over `torch.uint32` words using POPC and warp reduction. |
| G5 | `smith_waterman_score(a, b)` | Short-sequence Smith-Waterman score over `torch.uint8` encoded symbols. |
| G7 | `affine_scan32(a, b)` | Affine monoid inclusive scan over 32-element groups. |

Example:

```python
import torch
from biocuda_autotune import hamming_popc, reverse32, xor_permute

x = torch.arange(32, device="cuda", dtype=torch.int32)
print(reverse32(x))
print(xor_permute(x, 31))

a = torch.randint(0, 2**31, (1 << 20,), device="cuda", dtype=torch.uint32)
b = torch.randint(0, 2**31, (1 << 20,), device="cuda", dtype=torch.uint32)
print(hamming_popc(a, b))
```

---

## The BioCUDA Formula System

BioCUDA v38 defines 35 formulas. Not all of them should be CUDA kernels. Some are algebraic identities, some are Python-side scoring models, and some require counters from Nsight, CUPTI, NVML, MPS, or dedicated microbenchmarks.

Use:

```python
from biocuda_autotune import coverage_report
coverage = coverage_report()
```

### Implementation Map

| Formula | Status In This Package | Role |
|---|---|---|
| G1 | CUDA kernel | XOR complement/permutation primitive. |
| G2 | CUDA kernel | Reverse indexing primitive. |
| G3 | Python formula | Cache-line transaction quantum. |
| G4 | CUDA kernel | Hamming POPC. |
| G5 | CUDA kernel | Smith-Waterman/edit-distance DP score primitive. |
| G6 | Python formula | PWM / tensor-core roofline crossover. |
| G7 | CUDA kernel | Affine prefix scan. |
| G8 | Python formula | HMM GEMM arithmetic model. |
| G9 | Measurement required | GPU surprisal from `sm__cycles_active`. |
| G10 | Measurement required | MI screen for Boltzmann approximation validity. |
| G11 | Python formula | Hardware odds ratio. |
| G12 | Python formula | Shared-memory staging crossover. |
| G13 | Python formula | Bottleneck instruction class / Hill helper. |
| G14 | Python + measurement | Memory transport, entropy, energy. |
| G15 | Python + measurement | Occupancy response / TC cooperativity. |
| G16 | Python formula | Resource-bound occupancy. |
| G17 | Measurement required | L2 warmup fit. |
| G18 | Python formula | Scoreboard DAG critical path. |
| G19 | Python formula | Hierarchy reuse thresholds. |
| G20 | Python + protocol | ECC/integrity checks. |
| G21 | Planned CUDA kernel | Stochastic rounding kernels. |
| G22 | Python formula | EXP3 update and regret bound. |
| G23 | Python formula | Address curvature. |
| G24 | Python + measurement | L2 footprint/reuse model. |
| G25 | Model | Partition objective. |
| G26 | Python + measurement | Bank conflict model. |
| G27 | Python formula | Suffix-array digit/scatter bounds. |
| G28 | Measurement required | L1I pressure protocol. |
| G29 | Python + measurement | Resource competition model. |
| G30 | Python + measurement | Runtime surrogate / CUPTI correction. |
| G31 | Python + measurement | Mode utility. |
| G32 | Python formula | Jitter index. |
| G33 | Python + measurement | L2 sharing / co-scheduling predictor. |
| G34 | Python autotuner | Configuration search and pruning. |
| G35 | Python formula | Tensor-core partial-tile efficiency. |

This split is intentional. It keeps the project honest: formulas requiring hardware counters are not silently replaced with invented constants.

---

## Hardware Foundation

The reference v38 constants are for H100 SXM5 / `sm_90`, but the package also contains specs for T4, V100, A100, RTX 4090, and generic CUDA fallback.

### H100 Reference Constants

| Symbol | Value |
|---|---:|
| Warp size `W` | 32 |
| SM count `N_SM` | 132 |
| Max warps / SM `W_max` | 64 |
| Issue slots `N_iss` | 4 |
| Registers / SM `R_max` | 65536 |
| Shared memory / SM `M_S` | 233472 bytes |
| L2 | ~50 MB |
| HBM bandwidth | ~3.35e12 bytes/s |
| Register granularity `g_reg` | 256 |
| Cache segment `L_seg` | 128 bytes |

### Latency Table

| Path | Latency | Throughput | Status |
|---|---:|---:|---|
| register / shfl | 4 cycles | 16 warps/cycle | ISA |
| DPX vimin3 | 2 cycles | 128 ops/cycle/SM | PTX ISA |
| TC mma FP16 full | 16 cycles | 512 FMA/cycle/SM | CUTLASS / ISA-derived |
| TC partial | `16 / eta` | `512 * eta` FMA | derived |
| shared memory | ~23 cycles | platform dependent | microbench |
| L2 hit | ~193 cycles | platform dependent | microbench |
| HBM miss | ~600 cycles | platform dependent | microbench |

The package treats `tau_smem`, `tau_l2`, and `tau_hbm` as platform constants that should be calibrated on real hardware.

---

## Core Math Highlights

### G1/G2: XOR And Reverse Indexing

```text
(C_m x)_l = x_(l xor m)
C_m^2 = I
(R x)_l = x_(31-l)
```

These are small primitives, but they matter because they replace shared-memory round trips with register/shuffle-level movement when the access pattern is compatible.

### G4: Hamming POPC

```text
d_H(x, y) = popc(x xor y)
```

For binary words, the reference latency is 8 cycles. For 2-bit DNA encoding:

```text
d_H_2bit(x, y) = popc(((x xor y) | ((x xor y) >> 1)) & M)
```

### G5: Edit Distance / Smith-Waterman Wavefront

```text
|A_d| = min(d+1, m, n, m+n-d-1)
T_antidiag_lb = tau_shfl + tau_DPX + tau_reg
```

The package exposes a short-sequence Smith-Waterman CUDA kernel. The v38 model is broader than this implementation: the spec describes DPX wavefront lower bounds; the current extension provides a correctness-oriented kernel primitive that can be expanded into a full tiled wavefront backend.

### G6/G35: Tensor-Core Scoring And Partial Tiles

BioCUDA uses tensor-core suitability as a scoring signal:

```text
I* = Phi_TC / B_HBM
Phi_TC = tc_per_sm * 512 * 2 * N_SM * f_clk
```

The `*2` is important: 512 FMA means 1024 FLOP.

Partial tile penalty:

```text
eta_partial(R, C) = R*C / (16*ceil(R/16) * 16*ceil(C/16))
```

For example:

```text
eta(20, 100) = 2000 / 3584 = 0.558
```

### G12/G19: Staging Thresholds

The two formulas encode the same threshold from opposite directions:

```text
G12(E=0) = tau_S / (tau_G - tau_S)
A_min_smem = (tau_G - tau_S) / tau_S
G12(E=0) * A_min_smem = 1
```

On H100 reference values:

```text
A_min_smem = (600 - 23) / 23 = 25.09x
```

On the benchmarked Tesla T4 run:

```text
A_min_smem = 22.214
A_min_shfl = 161.5
A_min_l2   = 1.955
```

### G34: Configuration Search

The abstract search space:

```text
Omega = K x T x L x P x S
|Omega| = 4 * 6 * 4 * 4 * 4 = 1536
```

The practical matmul selector uses BioCUDA's pre-launch scoring idea:

```text
score(C) =
  lambda_E  * tau_eff(C) * (1 - eta_exec)
+ lambda_I  * delta_tau_interact
+ lambda_R  * regret_cycles
+ lambda_En * E_energy_norm * tau_eff(C)
```

All terms are in cycles or dimensionless multipliers of cycles.

Default zero-shot lambdas:

| Lambda | Value | Meaning |
|---|---:|---|
| `lambda_E` | 0.6 | execution time term |
| `lambda_I` | 0.2 | interaction penalty |
| `lambda_R` | 0.1 | online regret |
| `lambda_En` | 0.1 | normalized energy |

---

## Benchmarks

The benchmark data below is from the BioCUDA v39.12 notebook run that motivated this package.

Environment:

| Item | Value |
|---|---|
| GPU | Tesla T4 |
| Architecture | Turing / `sm_75` |
| Triton | 3.6.0 |
| Method | CUDA events |
| Iterations | G21 Nyquist condition: 100 iterations for matmul |
| Stability | G32 jitter reported |
| Integrity | G20 ECC / relative error checks |
| L2 behavior | L2 flush before timed sections |

### Bio-Kernel Benchmark: Hamming + Smith-Waterman

| Kernel | Size | Default | BioCUDA Choice | BioCUDA Time | Speedup |
|---|---:|---:|---|---:|---:|
| Hamming | 256K words / 1 MB | 468.0 us | block=1024 | 474.4 us | 0.987x |
| Hamming | 1024K words / 4 MB | 1281.1 us | block=1024 | 1269.3 us | 1.009x |
| Hamming | 4096K words / 16 MB | 7915.8 us | block=1024 | 7992.3 us | 0.990x |
| Smith-Waterman | 64 x 64 | 741.5 us | block=256 | 747.6 us | 0.992x |
| Smith-Waterman | 128 x 128 | 2354.4 us | block=256 | 2298.3 us | 1.024x |
| Smith-Waterman | 256 x 256 | 4432.6 us | block=256 | 3858.2 us | 1.149x |

Interpretation:

- Hamming is already close to memory/launch limited on this setup. BioCUDA selects a larger block but only moves within noise-level improvement/regression.
- Smith-Waterman improves as the problem grows: the 256 x 256 case reaches **1.149x** and has much lower jitter than the default run.
- The SW default at 256 x 256 showed high jitter (`0.928`), while BioCUDA's selected config showed `0.003`.

### MatMul Benchmark: 8 Variants

All timings are microseconds.

| Size | cuBLAS | Triton 4 cfg | Native Triton 128 cfg | BioCUDA+Triton | BioCUDA+Opt Triton | torch.compile | BioCUDA+PyTorch | BioCUDA+Opt PyTorch | EXP3 Winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 512 | 90.2 | 286.9 | 228.8 | 154.1 | 152.2 | 141.3 | **60.4** | 144.2 | BC+PT |
| 1024 | 937.4 | 1918.7 | 782.3 | 954.3 | 965.9 | 204.8 | **79.1** | 209.5 | BC+PT |
| 2048 | 5005.2 | 16363.5 | 5733.7 | 7159.8 | 7157.8 | 1042.4 | **466.3** | 1069.1 | BC+PT |
| 4096 | 35842.3 | 136814.8 | 45900.9 | 57466.9 | 57395.0 | 7251.4 | **6744.4** | 7261.2 | BC+PT |

Relative to cuBLAS where `1.00x` means same runtime:

| Size | cuBLAS | Triton 4 cfg | Native Triton | BC+Triton | BC+Opt Triton | Opt PyTorch | BC+PyTorch | BC+Opt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1.00x | 3.18x | 2.54x | 1.71x | 1.69x | 1.57x | **0.67x** | 1.60x |
| 1024 | 1.00x | 2.05x | 0.83x | 1.02x | 1.03x | 0.22x | **0.08x** | 0.22x |
| 2048 | 1.00x | 3.27x | 1.15x | 1.43x | 1.43x | 0.21x | **0.09x** | 0.21x |
| 4096 | 1.00x | 3.82x | 1.28x | 1.60x | 1.60x | 0.20x | **0.19x** | 0.20x |

TFLOPS achieved:

| Size | cuBLAS | Triton 4 cfg | Native Triton | BC+Triton | BC+Opt Triton | Opt PyTorch | BC+PyTorch | BC+Opt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 2.98 | 0.94 | 1.17 | 1.74 | 1.76 | 1.90 | **4.44** | 1.86 |
| 1024 | 2.29 | 1.12 | 2.74 | 2.25 | 2.22 | 10.49 | **27.16** | 10.25 |
| 2048 | 3.43 | 1.05 | 3.00 | 2.40 | 2.40 | 16.48 | **36.85** | 16.07 |
| 4096 | 3.83 | 1.00 | 2.99 | 2.39 | 2.39 | 18.95 | **20.38** | 18.93 |

### Search Cost

| Size | Native Triton 128-config Search | BioCUDA Zero-Shot Top-6 Pruning |
|---:|---:|---:|
| 512 | 1028.59 s | 2.00 s |
| 1024 | 27.81 s | 0.74 s |
| 2048 | 151.07 s | 1.01 s |
| 4096 | 1178.17 s | 3.32 s |

This is the main autotuning story: BioCUDA does not always make the Triton kernel itself faster than every alternative, but it cuts search cost by orders of magnitude and gives a structured winner-selection layer across variants.

### Stability And Integrity

| Size | ECC / Rel Error | EXP3 Winner | Most Stable Variant |
|---:|---|---|---|
| 512 | PASS, rel_err=0.00e+00 | BC+PT, 98% | Tri_opt, jitter=0.0271 |
| 1024 | PASS, rel_err=1.31e-06 | BC+PT, 100% | cuBLAS, jitter=0.0078 |
| 2048 | PASS, rel_err=2.31e-06 | BC+PT, 92% | Tri_base, jitter=0.0042 |
| 4096 | PASS, rel_err=0.00e+00 | BC+PT, 100% | Tri_base, jitter=0.0009 |

G32 jitter summary:

| Size | Best Jitter | Worst Jitter |
|---:|---:|---:|
| 512 | 0.0271 | 3.1094 |
| 1024 | 0.0078 | 0.1520 |
| 2048 | 0.0042 | 0.3827 |
| 4096 | 0.0009 | 0.0179 |

### Calibration And Falsification Summary

| Metric | Result |
|---|---:|
| Version | 39.12.0-triton |
| GPU | Tesla T4 |
| Detection mode | live |
| CuPy available | true |
| CUPTI available | false |
| Triton available | true |
| Kernel modules | 10 |
| T0/N/C falsification pass rate | 173 / 173 = 100% |
| Tier M evaluated | 5 |
| Tier M passed | 4 |
| G34 omega size | 1536 |
| Kernel sanity: Hamming | CPU=65505, GPU=65505, match=true |
| Kernel sanity: Smith-Waterman | CPU=22, GPU=22, match=true |

Tier M details:

| Test | Passed | Measured | Predicted / Tolerance | Notes |
|---|---|---:|---:|---|
| M1 Kendall | no | 0.9286 | pred=1.0286, tol=0.1 | `tau_roof=0.929` |
| M2 Hill R2 | yes | 0.9585 | pred=0.95, tol=0.05 | `V=1.13 K=1.6 n=1.50` |
| M3 EXP3 regret | yes | 58.648 | pred=135.517, tol=5.0 | `T=512 K=8` |
| M4 Occupancy | yes | 0.030 | pred=0.1, tol=0.1 | `pred=1.000 meas=0.970` |
| M5 TC vs vendor | yes | 0.00194 | pred=0.15, tol=0.15 | `est=65.13 vendor=65.00` |

Tensor-core calibration on T4:

| Field | Value |
|---|---:|
| MMA shape | 16 x 8 x 8 |
| Iterations | 4096 |
| Median cycles | 110824 |
| tau_tc cycles / MMA | 27.0566 |
| FLOP / MMA | 2048 |
| SM clock | 1.59 GHz |
| SM count | 40 |
| Estimated FP16 TC peak | 65.1264 TFLOPS |
| Vendor reference | 65.0 TFLOPS |

---

## Active Formula Sets In The Benchmark

### Phase 1: Bio-Kernels

```text
G1-G7, G12, G14, G16, G18, G19, G23-G26, G28
```

Used for Hamming and Smith-Waterman configuration scoring.

### Phase 2: MatMul

```text
G3, G6, G12-G19, G23-G26, G28, G29, G33-G35
```

Used for matmul tile/config pruning, tensor-core suitability, occupancy, partial-tile penalty, and resource pressure.

### Phase 3: Cross-Kernel Analysis

```text
G9-G11, G17, G20-G22, G27, G29-G33
```

Used for hardware surprisal, EXP3 winner selection, jitter/stability, cross-kernel ranking, and post-run analysis.

### Prediction-Only In That Run

```text
G8  HMM
G27 suffix-array digit/scatter model
```

---

## Falsification Philosophy

BioCUDA is not written as a collection of magical heuristics. Each formula has a class and a failure mode.

### Tier T0: Algebraic Identities

Examples:

- `C_m^2 = I`
- `C_m C_n = C_(m xor n)`
- affine scan associativity
- `eta_partial(16,16) = 1`
- `H_mem in [0, ln 3]`
- lambda normalization equals 1

### Tier N: Hardware-Verifiable Checks

Examples:

- scan lower bound near 20 cycles,
- full warp scan near 40 cycles,
- Hamming binary latency near 8 cycles,
- tensor-core latency within tolerance,
- occupancy prediction close to measured active warps,
- G34 latency below practical threshold.

### Tier M: Model Checks

Examples:

- Hill fit quality,
- EXP3 regret,
- staging crossover,
- cross-kernel slowdown correlation,
- L2 warmup relation to G31/G29 lag.

Tier M can fail without invalidating the algebraic core. It tells you where calibration is needed.

---

## Current Limitations

- The native Smith-Waterman CUDA kernel is a short-sequence primitive, not yet a full production tiled wavefront implementation.
- G21 stochastic rounding is currently represented in formulas/tests but not yet a native CUDA kernel.
- Measurement-required formulas need external tooling: Nsight, CUPTI, NVML, MPS/MIG.
- The benchmark numbers above are from Tesla T4; do not assume the same winners on A100/H100 without recalibration.
- `BioCUDA + PyTorch` benchmark results are included as measured notebook output, but the exact PyTorch compiler/runtime path should be revalidated in this packaged project.

---

## Roadmap

- Add CUDA microbenchmarks for tau_shfl, tau_smem, tau_l2, tau_hbm, tau_tc.
- Add G21 stochastic rounding kernels.
- Expand Smith-Waterman from short-sequence primitive to tiled wavefront backend.
- Add Profile-HMM and PWM examples using G6/G8/G35.
- Add suffix-array digit-sort prototype using G7/G27.
- Add Nsight/CUPTI counter import utilities for G9/G10/G14/G30.
- Add benchmark scripts that reproduce the Tesla T4 tables from this README.
- Add wheels or documented build matrix for Linux/Windows CUDA environments.

---

## Development

Run formula tests:

```bash
pytest tests/test_formula_engine.py
```

Run CUDA extension smoke tests:

```bash
pytest tests/test_cuda_extension.py
```

Example script:

```bash
python examples/biocuda_quickstart.py
```

---

## Citation / Attribution

This repository packages the BioCUDA v38/v39.12 design as a PyTorch extension. If you use it in research or experiments, cite the repository and include:

- the target GPU,
- PyTorch version,
- CUDA version,
- Triton version if used,
- whether CUPTI/Nsight/MPS data was available,
- which G-formulas were used as kernels, formulas, or measured counters.

The project is intentionally falsifiable. If a model term fails on your GPU, report the hardware, counters, and benchmark script rather than treating the failure as noise.
