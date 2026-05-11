#include <torch/extension.h>

torch::Tensor hamming_popc_cuda(torch::Tensor a, torch::Tensor b, int64_t block_size);
torch::Tensor sw_score_cuda(torch::Tensor a, torch::Tensor b, int64_t match, int64_t mismatch, int64_t gap_open, int64_t gap_extend, int64_t block_size);
torch::Tensor reverse32_cuda(torch::Tensor x);
torch::Tensor xor_permute_cuda(torch::Tensor x, int64_t mask);
torch::Tensor affine_scan32_cuda(torch::Tensor a, torch::Tensor b);

static void check_cuda_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

torch::Tensor hamming_popc(torch::Tensor a, torch::Tensor b, int64_t block_size) {
    check_cuda_tensor(a, "a");
    check_cuda_tensor(b, "b");
    TORCH_CHECK(a.scalar_type() == torch::kUInt32, "a must be torch.uint32");
    TORCH_CHECK(b.scalar_type() == torch::kUInt32, "b must be torch.uint32");
    TORCH_CHECK(a.numel() == b.numel(), "a and b must have the same number of words");
    return hamming_popc_cuda(a, b, block_size);
}

torch::Tensor sw_score(torch::Tensor a, torch::Tensor b, int64_t match, int64_t mismatch, int64_t gap_open, int64_t gap_extend, int64_t block_size) {
    check_cuda_tensor(a, "a");
    check_cuda_tensor(b, "b");
    TORCH_CHECK(a.scalar_type() == torch::kUInt8, "a must be torch.uint8");
    TORCH_CHECK(b.scalar_type() == torch::kUInt8, "b must be torch.uint8");
    return sw_score_cuda(a, b, match, mismatch, gap_open, gap_extend, block_size);
}

torch::Tensor reverse32(torch::Tensor x) {
    check_cuda_tensor(x, "x");
    TORCH_CHECK(x.numel() % 32 == 0, "x.numel() must be divisible by 32");
    return reverse32_cuda(x);
}

torch::Tensor xor_permute(torch::Tensor x, int64_t mask) {
    check_cuda_tensor(x, "x");
    TORCH_CHECK(mask >= 0 && mask < 32, "mask must be in [0, 31]");
    TORCH_CHECK(x.numel() % 32 == 0, "x.numel() must be divisible by 32");
    return xor_permute_cuda(x, mask);
}

torch::Tensor affine_scan32(torch::Tensor a, torch::Tensor b) {
    check_cuda_tensor(a, "a");
    check_cuda_tensor(b, "b");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be torch.float32");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be torch.float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
    TORCH_CHECK(a.numel() % 32 == 0, "a.numel() must be divisible by 32");
    return affine_scan32_cuda(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hamming_popc", &hamming_popc, "BioCUDA G4 Hamming POPC reduction (CUDA)",
          py::arg("a"), py::arg("b"), py::arg("block_size") = 256);
    m.def("sw_score", &sw_score, "BioCUDA G5 Smith-Waterman score (CUDA)",
          py::arg("a"), py::arg("b"), py::arg("match") = 2, py::arg("mismatch") = -1,
          py::arg("gap_open") = 2, py::arg("gap_extend") = 1, py::arg("block_size") = 128);
    m.def("reverse32", &reverse32, "BioCUDA G2 reverse indexing over each 32-lane group (CUDA)");
    m.def("xor_permute", &xor_permute, "BioCUDA G1 XOR permutation over each 32-lane group (CUDA)",
          py::arg("x"), py::arg("mask"));
    m.def("affine_scan32", &affine_scan32, "BioCUDA G7 affine monoid inclusive scan over each 32-lane group (CUDA)");
}
