#include <cmath>
#include <cuda/cmath>
#include <iostream>
#include <memory>
#include <vector>

#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f

__global__ void sigmoid_f32_kernel(float *x, float *y, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < N) {
    float v = x[tid];
    v = fminf(fmaxf(v, MIN_EXP_F32), MAX_EXP_F32);
    y[tid] = 1.0f / (1.0f + expf(-v));
  }
}

struct Del {
  void operator()(float p[]) const {
    if (p)
      cudaFree(p);
  }
};

using cudaptr = std::unique_ptr<float[], Del>;

cudaptr make_cu_unique(int n) {
  float *p = nullptr;
  cudaMalloc(&p, n * sizeof(float));
  return cudaptr(p);
}

int main() {
  int N = 1028;
  auto hx = std::vector<float>(N);
  auto hy = std::vector<float>(N);

  hx[0] = 0.0f;
  for (int i = 1; i < 5; i++) {
    hx[i] = logf(i);
  }

  auto dx = make_cu_unique(N);
  auto dy = make_cu_unique(N);

  cudaMemcpy(dx.get(), hx.data(), N * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = cuda::ceil_div(N, blockSize);
  sigmoid_f32_kernel<<<gridSize, blockSize>>>(dx.get(), dy.get(), N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }

  cudaMemcpy(hy.data(), dy.get(), N * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 5; i++) {
    std::cout << hx[i] << " ";
  }
  std::cout << std::endl;

  for (int i = 0; i < 5; i++) {
    std::cout << hy[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}