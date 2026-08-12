#include <cuda/cmath>
#include <cuda_fp16.h>
#include <iostream>
#include <memory>
#include <vector>

#include "utils.cuh"

#define clamp(max, val, min) (val < min ? min : val > max ? max : val)
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)


__global__ void sigmoid_f16x8_pack_kernel(half *x, half *y, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  tid *= 8;

  if (tid + 8 <= N) {
    alignas(16) half bufx[8];
    *reinterpret_cast<float4 *>(&bufx[0]) =
        *reinterpret_cast<float4 *>(&x[tid]);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      half v = clamp(MAX_EXP_F16, bufx[i], MIN_EXP_F16);
      y[tid + i] = __float2half(1.0f) / (__float2half(1.0f) + hexp2(-v));
    }
  } else if (tid < N) {
    for (int i = tid; i < N; i++) {
      half v = clamp(MAX_EXP_F16, x[tid], MIN_EXP_F16);
      y[i] = __float2half(1.0f) / (__float2half(1.0f) + hexp2(-v));
    }
  }
}

int main() {
  int N = 1028;
  auto hx = std::vector<half>(N);
  auto hy = std::vector<half>(N);

  hx[0] = __float2half(0.0f);
  for (int i = 1; i < 5; i++) {
    hx[i] = __float2half(logf(i));
  }

  auto dx = make_cuda_unique(N);
  auto dy = make_cuda_unique(N);

  cudaMemcpy(dx.get(), hx.data(), N * sizeof(half), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int packN = cuda::ceil_div(N, 8);
  int gridSize = cuda::ceil_div(packN, blockSize);
  sigmoid_f16x8_pack_kernel<<<gridSize, blockSize>>>(dx.get(), dy.get(), N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }

  cudaMemcpy(hy.data(), dy.get(), N * sizeof(half), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 5; i++) {
    std::cout << __half2float(hx[i]) << " ";
  }
  std::cout << std::endl;

  for (int i = 0; i < 5; i++) {
    std::cout << __half2float(hy[i]) << " ";
  }
  std::cout << std::endl;

  return 0;
}