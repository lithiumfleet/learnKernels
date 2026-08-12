#include <cuda/cmath>
#include <iostream>
#include <memory>
#include <vector>

__global__ void histogram_i32_kernel(int *a, int *y, int N, int CNT) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    atomicAdd(&y[a[tid] % CNT], 1);
  }
}

struct CudaDeletor {
  void operator()(int *ptr) const {
    if (ptr)
      cudaFree(ptr);
  }
};

using Cudaptr = std::unique_ptr<int, CudaDeletor>;

Cudaptr make_cuda_unique(int N) {
  int *ptr = nullptr;
  cudaMalloc(&ptr, N * sizeof(int));
  return Cudaptr(ptr);
}

int main() {
  int N = 1025;
  int CNT = 7;
  auto ha = std::vector<int>(N);
  auto hy = std::vector<int>(CNT);

  for (int i = 0; i < N; i++)
    ha[i] = i;
  for (int i = 0; i < 5; i++)
    hy[i] = 0;

  auto da = make_cuda_unique(N);
  auto dy = make_cuda_unique(CNT);

  cudaMemcpy(da.get(), ha.data(), N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dy.get(), hy.data(), CNT * sizeof(int), cudaMemcpyHostToDevice);

  int blockSize = 128;
  int gridSize = cuda::ceil_div(N, blockSize);
  histogram_i32_kernel<<<gridSize, blockSize>>>(da.get(), dy.get(), N, CNT);

  cudaMemcpy(hy.data(), dy.get(), CNT * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < CNT; i++) {
    std::cout << hy[i] << " ";
  }

  return 0;
}