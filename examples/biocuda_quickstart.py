import torch

from biocuda_autotune import BioCUDAAutoTune, benchmark_matmul, coverage_report, hamming_popc, is_available, matmul


def main():
    tuner = BioCUDAAutoTune(verbose=True)
    print(tuner.summary())
    print("covered formulas:", len(coverage_report()))

    if torch.cuda.is_available():
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        c = matmul(a, b, tuner=tuner)
        print("matmul:", tuple(c.shape), c.dtype)
        print(benchmark_matmul(1024, iterations=25))
        if is_available():
            x = torch.randint(0, 2**31, (1 << 20,), device="cuda", dtype=torch.uint32)
            y = torch.randint(0, 2**31, (1 << 20,), device="cuda", dtype=torch.uint32)
            print("hamming:", int(hamming_popc(x, y).cpu()[0]))
        else:
            print("CUDA extension _C is not built; run `pip install -e .` in a CUDA build environment.")
    else:
        print("CUDA is not available; package import and formula engine are ready.")


if __name__ == "__main__":
    main()
