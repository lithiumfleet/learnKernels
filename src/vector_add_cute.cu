// https://zhuanlan.zhihu.com/p/663093816

#include "cute/numeric/integral_constant.hpp"
#include "cute/pointer.hpp"
#include "cute/tensor_impl.hpp"
#include <cuda_fp16.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#define ceil_div(a, b) ((a + b - 1) / b)

template <int numElemPreThread = 8>
__global__ void
vector_add_local_tile_multi_elem_per_thread_half(const half *a, const half *b,
                                                 half *c, int N) {
  using namespace cute;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid * numElemPreThread >= N)
    return;

  auto ta = make_tensor(make_gmem_ptr(a), make_layout(N));
  auto tb = make_tensor(make_gmem_ptr(b), make_layout(N));
  auto tc = make_tensor(make_gmem_ptr(c), make_layout(N));

  auto tileA =
      local_tile(ta, make_shape(Int<numElemPreThread>{}), make_coord(tid));
  auto tileB =
      local_tile(tb, make_shape(Int<numElemPreThread>{}), make_coord(tid));
  auto tileC =
      local_tile(tc, make_shape(Int<numElemPreThread>{}), make_coord(tid));

  auto tempA = make_tensor_like(tileA);
  auto tempB = make_tensor_like(tileB);
  auto tempC = make_tensor_like(tileC);

  copy(tileA, tempA);
  copy(tileB, tempB);

  auto regA = recast<half2>(tempA);
  auto regB = recast<half2>(tempB);
  auto regC = recast<half2>(tempC);

  for (int i = 0; i < size(regC); i++) {
    regC(i) = regA(i) + regB(i);
  }

  copy(recast<half>(regC), tileC);
}

int main() {

  using namespace std;

  const int N = 8195;
  half *a;
  half *b;
  half *c;

  cudaMallocManaged(&a, N * sizeof(half));
  cudaMallocManaged(&b, N * sizeof(half));
  cudaMallocManaged(&c, N * sizeof(half));

  for (int i = 0; i < N; ++i) {
    a[i] = __float2half(1.0f);
    b[i] = __float2half(0.5f);
  }

  int blockSize = 1024;
  int gridSize = ceil_div(N / 8, blockSize);

  vector_add_local_tile_multi_elem_per_thread_half<<<gridSize, blockSize>>>(
      a, b, c, N);

  cudaDeviceSynchronize();

  for (int i = 0; i < 5; i++) {
    cout << __half2float(c[i]) << " ";
  }

  return 0;
}