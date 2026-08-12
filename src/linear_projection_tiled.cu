#include <cuda/cmath>
#include <cuda_tile.h>
#include <iostream>

namespace ct = cuda::tiles;
using namespace ct::literals;

template <auto TM, auto TN, auto TK>
__tile_global__ void linearProject(float *Y, const float *X, const float *W,
                                   const float *B, int M, int N, int K) {
  auto Yview = ct::partition_view{ct::tensor_span{Y, ct::shape{M, N}},
                                  ct::shape{TM, TN}};
  auto Xview = ct::partition_view{ct::tensor_span{X, ct::shape{M, K}},
                                  ct::shape{TM, TK}};
  auto Wview = ct::partition_view{ct::tensor_span{W, ct::shape{K, N}},
                                  ct::shape{TK, TN}};
  auto Bview =
      ct::partition_view{ct::tensor_span{B, ct::shape{N}}, ct::shape{TN}};

  auto acc = ct::zeros<ct::tile<float, ct::shape<TM, TN>>>();
  acc = ct::transpose(acc);

  auto bid = ct::bid();
  for (auto bidk : ct::irange(0, (K + TK - 1) / TK)) {
    auto x = Xview.load_masked(bid.x, bidk);
    auto w = Wview.load_masked(bidk, bid.y);
    acc = ct::mma(x, w, acc);
  }

  auto bias = Bview.load_masked(bid.x);
  acc = acc + bias;
  Yview.store_masked(acc, bid.x, bid.y);
}

int main() {
  int M = 1024, N = 512, K = 512;
  constexpr auto TM = 128_ic;
  constexpr auto TN = 128_ic;
  constexpr auto TK = 32_ic;

  float *Y, *X, *W, *B;
  cudaMallocManaged(&Y, M * N * sizeof(float));
  cudaMallocManaged(&X, M * K * sizeof(float));
  cudaMallocManaged(&W, K * N * sizeof(float));
  cudaMallocManaged(&B, N * sizeof(float));

  for (int i = 0; i < M * N; i++)
    X[i] = 1.0f;
  for (int i = 0; i < M * K; i++)
    W[i] = 2.0f;
  for (int i = 0; i < N; i++)
    B[i] = 0.5f;

  dim3 grid(cuda::ceil_div(M, TM.value), cuda::ceil_div(N, TN.value));
  linearProject<TM, TN, TK><<<grid, 1>>>(Y, X, W, B, M, N, K);

  cudaDeviceSynchronize();

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++)
      std::cout << Y[i * M + j] << " ";
    std::cout << std::endl;
  }

  return 0;
}