#include <cstdio>
#include <cuda/cmath>
#include <iostream>
#include <ostream>

__device__ float warpReduce(float val) {
  for (int i = 16; i >= 1; i /= 2)
    val += __shfl_down_sync(0xffffffff, val, i);
  return val;
}

__global__ void reduceSum(float *a, float *y, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float val = tid < N ? a[tid] : 0.0f;
  val = warpReduce(val);

  __shared__ float smem[32];

  int warpId = threadIdx.x / 32;
  int laneId = threadIdx.x % 32;
  if (laneId == 0) {
    smem[warpId] = val;
  }

  __syncthreads();
  if (warpId == 0) {
    int numWarps = (blockDim.x + 31) / 32;
    val = laneId < numWarps ? smem[laneId] : 0.0f;
    val = warpReduce(val);
    if (laneId == 0)
      atomicAdd(y, val);
  }
}

int main() {
  int N = 1025;
  float *a = nullptr;
  float *y = nullptr;
  cudaMallocManaged(&a, N * sizeof(float));
  cudaMallocManaged(&y, sizeof(float));

  for (int i = 0; i < N; i++)
    a[i] = 1.0f;

  int blockSize = 256;
  int gridSize = cuda::ceil_div(N, blockSize);

  auto config = cudaLaunchConfig_t{};
  config.gridDim.x = gridSize;
  config.blockDim.x = blockSize;

  auto err = cudaLaunchKernelEx(&config, reduceSum, a, y, N);
  if (err != cudaSuccess) {
    std::cout << cudaGetErrorString(err) << std::endl;
  }

  cudaDeviceSynchronize(); // printf in kernel must be synced.

  std::cout << *y << std::endl;
  cudaFreeHost(a);
  cudaFreeHost(y);
  return 0;
}