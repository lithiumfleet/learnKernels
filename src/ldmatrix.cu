#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>

__global__ void LDSM_8x8_x1_b16() {
  int laneId = threadIdx.x;

  __shared__ uint16_t smem[8 * 8];
  if (laneId == 0) {
    for (int i = 0; i < 8 * 8; i++) {
      smem[i] = i;
    }
  }
  __syncthreads();

  // lane0..7s' smemAddrs are used, which point to smem[0], smem[8]...
  uint32_t smemAddr = __cvta_generic_to_shared(smem + laneId * 8);
  uint32_t reg;

  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 "
               "{%0}, [%1];\n"
               : "=r"(reg)
               : "r"(smemAddr));

  uint16_t a[2];
  reinterpret_cast<uint32_t *>(a)[0] = reg;
  printf("lane %d get %u, %u\n", laneId, a[0], a[1]);
}

__global__ void LDSM_8x8_x4_b16() {
  __shared__ half smem[8 * 8 * 4];
  // init smem
  int laneId = threadIdx.x;

  if (laneId == 0) {
    for (int i = 0; i < 4 * 8 * 8; i++) {
      smem[i] = __float2half(float(i));
    }
  }
  __syncthreads();

  // print the smem layout
  if (laneId == 0) {
    for (int r = 0; r < 16; r++) {
      for (int c = 0; c < 16; c++) {
        int i = r * 16 + c;
        if (i == 0 || i == 8 || i == 128 || i == 128 + 8) {
          printf("(%p)", (void *)&smem[i]);
        }
        printf("%.1f ", __half2float(smem[r * 16 + c]));
      }
      printf("\n");
    }
  }

  __syncthreads();
  uint32_t regs[4];
  // also just 0..31 lane's addr will be used.
  uint32_t smemAddr =
      __cvta_generic_to_shared(smem + (laneId % 16 * 16 + laneId / 16 * 8));

  if (laneId < 32) {
    printf("lane %d -> 0x%08x\n", laneId, smemAddr);
  }

  __syncthreads();
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [ %4 ];\n"
      : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
      : "r"(smemAddr));

  __syncthreads();

  half a[8];
  memcpy(a, regs, sizeof(a));

  if (laneId == 0) {
    printf("lane %d get {", laneId);
    for (int i = 0; i < 8; i++)
      printf("%.1f, ", __half2float(a[i]));
    printf("\b\b}\n");
  }
}

int main() {

  std::cout << "== LDSM_8x8_x1_b16 ==" << std::endl;
  LDSM_8x8_x1_b16<<<1, 32>>>();
  cudaDeviceSynchronize();
  cudaDeviceReset();

  std::cout << std::endl << "== LDSM_8x8_x4_b16 ==" << std::endl;
  LDSM_8x8_x4_b16<<<1, 32>>>();
  cudaDeviceSynchronize();
  cudaDeviceReset();

  return 0;
}