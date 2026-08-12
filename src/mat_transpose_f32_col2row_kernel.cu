#include "utils.cuh"
#include <cassert>
#include <iostream>
#include <ostream>
#include <vector>

__global__ void mat_transpose_f32_col2row_kernel(float *x, float *y,
                                                 const int rn, const int cn) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int rid = tid / cn;
  int cid = tid % cn;
  assert(tid == rid * cn + cid);
  if (tid < rn * cn) {
    y[cid * rn + rid] = x[tid];
  }
}

int main() {
  int rn = 1024, cn = 512;
  auto hx = std::vector<float>(rn * cn);
  auto hy = std::vector<float>(cn * rn);

  for (int i = 0; i < rn; i++) {
    for (int j = 0; j < cn; j++) {
      hx[i * cn + j] = j;
    }
  }

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      std::cout << hx[i * rn + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto dx = make_cuda_unique<float>(rn * cn);
  auto dy = make_cuda_unique<float>(cn * rn);

  cudaMemcpy(dx.get(), hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = cuda::ceil_div(rn * cn, blockSize);
  mat_transpose_f32_col2row_kernel<<<gridSize, blockSize>>>(dx.get(), dy.get(),
                                                            rn, cn);
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << cudaGetErrorString(err) << std::endl;
  }

  cudaMemcpy(hy.data(), dy.get(), hy.size() * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      std::cout << hy[i * rn + j] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}