#include <cuda/cmath>
#include <cuda_fp16.h>
#include <iostream>
#include <ostream>
#include <vector>

#include "utils.cuh"

__global__ void relu_f16x8_pack_kernel(half *x, half *y, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  tid *= 8;

  int hzero = __float2half(0.0f);

  if (tid + 8 < N) {
    half xbuf[8], ybuf[8];
    *reinterpret_cast<float4 *>(xbuf) = *reinterpret_cast<float4 *>(&x[tid]);

#pragma unroll
    for (int i = 0; i < 8; i++) {
      ybuf[i] = __hmax(hzero, xbuf[i]);
    }

    *reinterpret_cast<float4 *>(&y[tid]) = *reinterpret_cast<float4 *>(ybuf);

  } else {
    for (int i = tid; i < N; i++) {
      y[i] = __hmax(hzero, x[i]);
    }
  }
}

int main() {
  int N = 1027;
  auto hx = std::vector<half>(N);
  auto hy = std::vector<half>(N);

  for (int i = 0; i < 5; i++) {
    hx[i] = __float2half(i * (i % 2 == 0 ? -1 : 1));
  }

  auto dx = make_cuda_unique(N);
  auto dy = make_cuda_unique(N);

  cudaMemcpy(dx.get(), hx.data(), N * sizeof(half), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int packN = 8;
  int gridSize = cuda::ceil_div(cuda::ceil_div(N, packN), blockSize);

  relu_f16x8_pack_kernel<<<gridSize, blockSize>>>(dx.get(), dy.get(), N);

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