#include "utils.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <ostream>
#include <vector>

__device__ float warp_reduce_sum_f32(float val) {
  for (int off = 16; off >= 1; off /= 2) {
    val += __shfl_down_sync(0xffffffff, val, off);
  }
  return val;
}

__global__ void dot_prod_f32_f32_kernel(float *a, float *b, float *y, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = tid < N ? a[tid] * b[tid] : 0.0f;

  sum = warp_reduce_sum_f32(sum);

  __shared__ float smem[32];
  int warpId = threadIdx.x / 32;
  int laneId = threadIdx.x % 32;

  if (laneId == 0) {
    smem[warpId] = sum;
  }
  __syncthreads();

  if (warpId == 0) {
    int numWarps = (blockDim.x + 31) / 32;
    sum = laneId < numWarps ? smem[laneId] : 0.0f;
    sum = warp_reduce_sum_f32(sum);
    if (laneId == 0) {
      atomicAdd(y, sum);
    }
  }
}

int main() {
  int N = 8308;
  auto ha = std::vector<float>(N);
  auto hb = std::vector<float>(N);
  float hy = 0.0f;
  for (int i = 0; i < N; i++) {
    ha[i] = i % 2 ? 1 : 2;
    hb[i] = i % 2 ? 2 : 1;
  }

  auto da = make_cuda_vector(ha);
  auto db = make_cuda_vector(hb);
  auto dy = make_cuda_scalar(0.0f);

  int blockSize = 256;
  int gridSize = cuda::ceil_div(N, blockSize);
  dot_prod_f32_f32_kernel<<<gridSize, blockSize>>>(da.get(), db.get(), dy.get(),
                                                   N);

  sync_from_device(&hy, dy);

  std::cout << hy << std::endl;

  return 0;
}