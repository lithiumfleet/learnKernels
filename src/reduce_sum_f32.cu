#include "utils.cuh"
#include <cassert>
#include <cuda/cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// this kernel works in warp level
__device__ float warp_reduce_sum_f32(float val) {
  for (int m = 16; m >= 1; m >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, m);
  }
  return val;
}

// this kernel works in block level
__global__ void reduce_sum_f32(float *a, float *y, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = tid < N ? a[tid] : 0.0f;

  sum = warp_reduce_sum_f32(sum);

  __shared__ float smem[32]; // on Nvidia GPU, blockDim.x * blockDim.y * blockDim.z <= 1024. so the max size of smem is 32.

  int laneId = threadIdx.x % 32;
  int warpId = threadIdx.x / 32; // the warp id in one block

  if (laneId == 0)
    smem[warpId] = sum;

  __syncthreads();

  if (warpId == 0) {
    int numWarps = (blockDim.x + 31) / 32;

    sum = laneId < numWarps ? smem[laneId] : 0.0f;

    sum = warp_reduce_sum_f32(sum);

    if (laneId == 0)
      atomicAdd(y, sum); // race through blocks
  }
}

int main() {
  int N = 1028;
  auto ha = std::vector<float>(N);
  for (int i = 0; i < N; i++)
    ha[i] = 1;
  float hy = 0;

  auto da = make_cuda_unique<float>(N);
  auto dy = make_cuda_unique<float>(1);

  cudaMemcpy(da.get(), ha.data(), ha.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dy.get(), &hy, 1 * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 64;
  int gridSize = cuda::ceil_div(N, blockSize);

  reduce_sum_f32<<<gridSize, blockSize>>>(da.get(), dy.get(), N);

  auto cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    std::cout << cudaGetErrorString(cudaErr) << std::endl;
    exit(1);
  }

  cudaMemcpy(&hy, dy.get(), 1 * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << hy << std::endl;

  return 0;
}