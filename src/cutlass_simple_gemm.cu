// https://zhuanlan.zhihu.com/p/667521327
#include "cute/arch/mma_sm80.hpp"
#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <cute/tensor.hpp>

template <typename T, const int TM, int TN, const int TK, typename TiledMMA>
__global__ void simple_gemm(T *C, const T *A, const T *B, int M, int N, int K) {
  using namespace cute;
  // https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html
  // for NT strides: ABC all LayoutLeft
  // for TN strides: AB LayoutRight C LayoutLeft
  // so C always in row-major/N-major, A always uses (M,K) shape, B always uses
  // (N,K) shape and NT for AB: T means use K-major, N means use Non-K-major.

  // A is marked as T so use K-major == col-major == LayoutLeft{}
  auto tA = make_tensor(A, make_layout(make_shape(M, K)));
  auto tB = make_tensor(B, make_layout(make_shape(N, K), LayoutRight{}));
  auto tC = make_tensor(C, make_layout(make_shape(M, N), LayoutRight{}));

  int rid = blockIdx.y, cid = blockIdx.x;

  // tile tensor
  auto gA = local_tile(tA, make_tile(Int<TM>{}, Int<TK>{}), make_coord(rid, _));
  auto gB = local_tile(tB, make_tile(Int<TN>{}, Int<TK>{}), make_coord(cid, _));
  auto gC =
      local_tile(tC, make_tile(Int<TM>{}, Int<TN>{}), make_coord(rid, cid));

  // tmma in SM80_16x8x16_F16F16F16F16_TN & A in (2,2,1) B in (1,2,1) =>
  // 32x16x16
  TiledMMA tmma;
  auto thrMma = tmma.get_slice(threadIdx.x);

  auto tAgA = thrMma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)
  auto tBgB = thrMma.partition_B(gB);
  auto tCgC = thrMma.partition_C(gC);

  auto tArA = thrMma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  auto tBrB = thrMma.partition_fragment_B(gB(_, _, 0));
  auto tCrC = thrMma.partition_fragment_C(gC);

  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    print(tA.layout());
    printf("\n");
    print(gA.layout());
    printf("\n");
    printf("\n");
    print(tAgA.layout());
    printf("\n");
    print(tBgB.layout());
    printf("\n");
    print(tCgC.layout());
    printf("\n");
    printf("\n");
    print(tArA.layout());
    printf("\n");
    print(tBrB.layout());
    printf("\n");
    print(tCrC.layout());
    printf("\n");
  }
  __syncthreads();

  clear(tCrC);

  int tileCnt = size<2>(gA);
  for (int i = 0; i < tileCnt; i++) {
    copy(tAgA(_, _, _, i), tArA);
    copy(tBgB(_, _, _, i), tBrB);

    gemm(tmma, tCrC, tArA, tBrB, tCrC);
  }

  copy(tCrC, tCgC);
}

int main() {
  using namespace cute;

  using T = half_t;
  int M = 8192, N = 4096, K = 2048;

  T *A, *B, *C;
  cudaMallocManaged(&A, M * K * sizeof(T));
  cudaMallocManaged(&B, N * K * sizeof(T));
  cudaMallocManaged(&C, M * N * sizeof(T));

  for (int i = 0; i < M * K; i++) {
    A[i] = __float2half(1.0f);
  }
  for (int i = 0; i < N * K; i++) {
    B[i] = __float2half(2.0f);
  }

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  using MMA =
      decltype(make_tiled_mma(mma_atom{}, make_layout(Shape<_2, _2, _1>{}),
                              make_layout(Shape<_1, _2, _1>{})));
  const int TM = 128, TN = 128, TK = 32;
  dim3 block(size(MMA{}));
  dim3 grid(N / TN, M / TM);
  simple_gemm<T, TM, TN, TK, MMA><<<grid, block>>>(C, A, B, M, N, K);
  cudaDeviceSynchronize();

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << C[i * N + j] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}