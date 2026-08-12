#include <cuda/cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <ostream>
#include <sys/types.h>
#include <vector>

__global__ void elementwise_add_f16x2_kernel(half *a, half *b, half *c, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  tid *= 2;

  if (tid + 2 < N) {
    half2 *ra = reinterpret_cast<half2 *>(&a[tid]);
    half2 *rb = reinterpret_cast<half2 *>(&b[tid]);
    half2 sum;
    sum.x = ra->x + rb->x;
    sum.y = ra->y + rb->y;
    *reinterpret_cast<half2 *>(&c[tid]) = sum;
  } else if (tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}

struct CudaDeleter {
  void operator()(half ptr[]) const {
    if (ptr)
      cudaFree(ptr);
  }
};

std::unique_ptr<half[], CudaDeleter> make_cuda_unique(int N) {
  half *ptr = nullptr;
  cudaMalloc(&ptr, N * sizeof(half));
  return std::unique_ptr<half[], CudaDeleter>(ptr);
}

int main() {

  int N = 1023;
  int blockSize = 256;
  int gridSize = cuda::ceil_div(cuda::ceil_div(N, 2), blockSize);

  auto ha = std::vector<half>(N);
  auto hb = std::vector<half>(N);
  auto hc = std::vector<half>(N);
  for (int i = 0; i < N; i++) {
    ha[i] = __float2half(float(i) / 10);
    hb[i] = __float2half(float(i) / 10);
  }

  auto da = make_cuda_unique(N);
  auto db = make_cuda_unique(N);
  auto dc = make_cuda_unique(N);

  cudaMemcpy(da.get(), ha.data(), N * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(db.get(), hb.data(), N * sizeof(half), cudaMemcpyHostToDevice);

  elementwise_add_f16x2_kernel<<<gridSize, blockSize>>>(da.get(), db.get(),
                                                        dc.get(), N);

  cudaMemcpy(hc.data(), dc.get(), N * sizeof(half), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; i++)
    std::cout << __half2float(ha[i]) << " ";
  std::cout << std::endl;
  for (int i = 0; i < 10; i++)
    std::cout << __half2float(hb[i]) << " ";
  std::cout << std::endl;
  for (int i = 0; i < 10; i++)
    std::cout << __half2float(hc[i]) << " ";
  std::cout << std::endl;

  return 0;
}