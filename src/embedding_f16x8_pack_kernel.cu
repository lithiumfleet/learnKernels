#include <cuda_fp16.h>
#include <iostream>
#include <ostream>
#include <vector>

#include "utils.cuh"

// https://gemini.google.com/app/8f9a4d5109971302
// use cuda check macro to debug
__global__ void embedding_f16x8_pack_kernel(const int *idx, const half *weight,
                                            half *output, int n, int emb_size) {
  int bid = blockIdx.x;
  int tid = threadIdx.x * 8;

  uintptr_t out_ptr = reinterpret_cast<uintptr_t>(&output[bid * emb_size]);
  uintptr_t weight_ptr = reinterpret_cast<uintptr_t>(&weight[idx[bid] * emb_size]);
  bool is_aligned = (out_ptr % 16 == 0) && (weight_ptr % 16 == 0);

  if (is_aligned && tid + 8 <= emb_size) {
    *reinterpret_cast<float4 *>(&output[bid * emb_size + tid]) =
        *reinterpret_cast<const float4 *>(&weight[idx[bid] * emb_size + tid]);
  } else {
    for (int i = tid; i < tid + 8 && i < emb_size; i++) {
      output[bid * emb_size + i] = weight[idx[bid] * emb_size + i];
    }
  }
}
int main() {

  int emb_length = 1024;
  int emb_size = 256;
  int n = 5;
  auto hidx = std::vector<int>{1, 102, 33, 45, 1023};
  auto hweight = std::vector<half>(emb_length * emb_size);
  auto houtput = std::vector<half>(hidx.size() * emb_size);

  for (int i = 0; i < emb_length; i++) {
    for (int j = 0; j < emb_size; j++) {
      hweight[i * emb_size + j] = __float2half(float(i));
    }
  }

  auto didx = make_cuda_unique<int>(hidx.size());
  auto dweight = make_cuda_unique(hweight.size());
  auto doutput = make_cuda_unique(houtput.size());

  cudaMemcpy(didx.get(), hidx.data(), hidx.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dweight.get(), hweight.data(), hweight.size() * sizeof(half),
             cudaMemcpyHostToDevice);

  int blockSize = cuda::ceil_div(emb_size, 8);
  int gridSize = hidx.size();

  embedding_f16x8_pack_kernel<<<gridSize, blockSize>>>(
      didx.get(), dweight.get(), doutput.get(), n, emb_size);

  cudaMemcpy(houtput.data(), doutput.get(), houtput.size() * sizeof(half),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < hidx.size(); i++) {
    for (int j = 0; j < 5; j++) {
      std::cout << __half2float(houtput[i * emb_size + j]) << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}