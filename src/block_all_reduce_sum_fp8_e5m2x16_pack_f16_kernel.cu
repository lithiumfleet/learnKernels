#include <cuda_fp8.h>
#include <iostream>
#include <ostream>
#include <vector>

#include "utils.cuh"

__device__ half warp_reduce_sum_f16_f16(half val) {
  for (int offset = 1; offset < 32; offset *= 2) {
    val = __hadd(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

__global__ void
block_all_reduce_sum_fp8_e5m2x16_pack_f16_kernel(__nv_fp8_storage_t *a,
                                                 float *y, int N) {
  // thread loads fp8 using packing
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  tid *= 16;
  if (tid >= N) return; // add boundary check

  __nv_fp8_storage_t buf[16];
  *reinterpret_cast<float4 *>(buf) = *reinterpret_cast<float4 *>(&a[tid]);

  // use __nv_cvt convert to half and get half result from buf
  half sum = __float2half(0.0f);
  for (auto const &i : buf) {
    sum += __nv_cvt_fp8_to_halfraw(i, __NV_E5M2);
  }
  // call warp reduce to get warp result
  sum = warp_reduce_sum_f16_f16(sum);

  // store warp result to smem[32]
  __shared__ half smem[32];      // block level
  int warpId = threadIdx.x / 32; // block level
  int laneId = threadIdx.x % 32;
  if (laneId == 0) {
    smem[warpId] = sum;
  }
  __syncthreads();

  // load smem to warp0
  if (warpId == 0) {
    sum = smem[laneId];
    // call warp reduce to get block result
    sum = warp_reduce_sum_f16_f16(sum);
    // use atomadd to get sum result through blocks
    if (laneId == 0) {
      atomicAdd(y, sum);
    }
  }
}

int main() {
  int N = 1050;

  auto ha = std::vector<__nv_fp8_storage_t>(N);
  for (auto &i : ha)
    i = __nv_cvt_float_to_fp8(2.0f, __NV_SATFINITE, __NV_E5M2);
  float hy = 0.0f;

  auto da = make_cuda_unique<__nv_fp8_storage_t>(N);
  cudaMemcpy(da.get(), ha.data(), ha.size(), cudaMemcpyHostToDevice);
  auto dy = make_cuda_unique<float>(1);
  cudaMemcpy(dy.get(), &hy, 1 * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 64;
  int gridSize = cuda::ceil_div(N, blockSize);

  block_all_reduce_sum_fp8_e5m2x16_pack_f16_kernel<<<gridSize, blockSize>>>(
      da.get(), dy.get(), N);

  auto cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    std::cout << cudaGetErrorString(cudaErr) << std::endl;
    exit(1);
  }

  cudaMemcpy(&hy, dy.get(), 1 * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << hy << std::endl;

  return 0;
}