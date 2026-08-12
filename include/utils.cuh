#pragma once

#include <cuda/cmath>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <memory>
#include <stdexcept>
#include <vector>

template <typename T> struct Del {
  void operator()(T *p) const {
    if (p)
      cudaFree(p);
  }
};

template <typename T> using CudaPtr = std::unique_ptr<T, Del<T>>;

template <typename T>
concept CudaScalar =
    std::same_as<T, int> || std::same_as<T, float> || std::same_as<T, half> ||
    std::same_as<T, __nv_fp8_storage_t>;

template <CudaScalar T = float> CudaPtr<T> make_cuda_unique(int n) {
  T *p = nullptr;
  cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&p), n * sizeof(T));
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
  return CudaPtr<T>(p);
}

template <CudaScalar T>
CudaPtr<T> make_cuda_vector(const std::vector<T> &src, bool copy_data = true) {
  auto dst = make_cuda_unique<T>(src.size());

  if (copy_data) {
    cudaMemcpy(dst.get(), src.data(), src.size() * sizeof(T),
               cudaMemcpyHostToDevice);
  }
  return dst;
}

template <CudaScalar T> CudaPtr<T> make_cuda_scalar(T value) {
  auto dst = make_cuda_unique<T>(1);
  cudaMemcpy(dst.get(), &value, sizeof(T), cudaMemcpyHostToDevice);
  return dst;
}

template <CudaScalar T>
void sync_from_device(std::vector<T> &dst, const CudaPtr<T> &src) {
  cudaMemcpy(dst.data(), src.get(), dst.size() * sizeof(T),
             cudaMemcpyDeviceToHost);
}

template <CudaScalar T> 
void sync_from_device(T *dst, const CudaPtr<T> &src) {
  cudaMemcpy(dst, src.get(), 1 * sizeof(T), cudaMemcpyDeviceToHost);
}