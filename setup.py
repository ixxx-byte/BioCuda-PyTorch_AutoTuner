import os

from setuptools import find_packages, setup

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except Exception:
    torch = None
    BuildExtension = None
    CUDAExtension = None


def build_extensions():
    if os.environ.get("BIOCUDA_BUILD_EXT", "1") == "0":
        return [], {}
    if torch is None or CUDAExtension is None:
        return [], {}
    if not torch.cuda.is_available() and os.environ.get("FORCE_CUDA", "0") != "1":
        return [], {}
    cxx_flags = ["/O2"] if os.name == "nt" else ["-O3"]
    ext = CUDAExtension(
        name="biocuda_autotune._C",
        sources=["csrc/biocuda_ext.cpp", "csrc/biocuda_kernels.cu"],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": ["-O3", "--use_fast_math"],
        },
    )
    return [ext], {"build_ext": BuildExtension}


ext_modules, cmdclass = build_extensions()


setup(
    name="biocuda-autotune",
    version="0.1.0",
    description="BioCUDA v39 zero-shot PyTorch autotuner and Triton matmul extension",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["torch>=2.0.0", "numpy>=1.19.0"],
    extras_require={"triton": ["triton>=2.0.0"]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
