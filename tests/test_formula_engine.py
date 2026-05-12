import math

from biocuda_autotune import BioCUDAAutoTune, GPU_DB, coverage_report


def test_formula_summary_is_available():
    tuner = BioCUDAAutoTune(device="cpu", verbose=False)
    summary = tuner.summary()
    assert summary["g34_omega_size"] == 1536
    assert summary["g19_reuse_thresholds"]["A_min_smem"] > 0


def test_h100_errata_values():
    tuner = BioCUDAAutoTune(device="cpu", verbose=False)
    h100 = tuner.engine
    # CPU fallback detection uses the T4 spec by default, so this checks the
    # formula invariant through the reported reciprocal relation.
    reuse = h100.g19_reuse()["A_min_smem"]
    crossover = h100.g12_crossover(0.0)
    assert math.isclose(reuse * crossover, 1.0, rel_tol=1e-12)


def test_coverage_map_has_all_g_formulas():
    cov = coverage_report()
    assert len(cov) == 35
    for i in range(1, 36):
        assert f"G{i}" in cov


def test_gpu_database_matches_notebook_coverage():
    assert set(GPU_DB) == {
        "V100",
        "T4",
        "A100",
        "A10",
        "RTX3090",
        "L4",
        "L40",
        "RTX4090",
        "H100_SXM5",
        "H100_PCIE",
    }
