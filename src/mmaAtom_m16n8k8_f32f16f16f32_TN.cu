#include "cute/layout.hpp"
#include "cute/stride.hpp"
#include "cute/tensor_impl.hpp"

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=mma%2520sync%2520aligned%2520m16n8k16#warp-level-matrix-fragment-mma-1688
__global__ void mmaAtom_m16n8k8_f32f16f16f32_TN(const half *a, const half *b,
                                                float *c) {
  using namespace cute;

  // print layout u will c what indices it has...
  // for mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 
  // ta == layoutRight == RowMajor == .row
  // tb == layoutLeft == ColMajor == .col
  // tc == layoutRight == RowMajor (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=mma%2520sync%2520aligned%2520m16n8k16#mma-1688-c-f16-f32)

  auto ta = make_tensor(a, make_layout(make_shape(16, 8), LayoutRight{}));
  auto tb = make_tensor(b, make_layout(make_shape(8, 8)));
  auto tc = make_tensor(c, make_layout(make_shape(16, 8), LayoutRight{}));

  // tx: 0 32 64 96 1 ... -> (4,8):(32,1)
  // vx: 0 16 8 24 -> (2,2):(16,8)
  auto tva = make_layout(make_shape(make_shape(4, 8), make_shape(2, 2)),
                         make_stride(make_stride(32, 1), make_stride(16, 8)));
  // tx: 0 16 32 48 1 ... -> (4,8):(16,1)
  // vx: 0 8 -> 2:8
  auto tvb = make_layout(make_shape(make_shape(4, 8), 2),
                         make_stride(make_stride(16, 1), 8));
  // tx: 0 32 64 128 1 ... -> (4,8):(32,1)
  // vx: 0 16 8 24 -> (2,2):(16,8)
  auto tvc = make_layout(make_shape(make_shape(4, 8), make_shape(2, 2)),
                         make_stride(make_stride(32, 1), make_stride(16, 8)));

  half ra[4];
  half rb[2];
  float rc[4];

  for (int i = 0; i < 4; i++)
    ra[i] = ta(tva(threadIdx.x, i));
  for (int i = 0; i < 2; i++)
    rb[i] = tb(tvb(threadIdx.x, i));
  for (int i = 0; i < 4; i++)
    rc[i] = tc(tvc(threadIdx.x, i));

  auto _ra = reinterpret_cast<uint32_t *>(ra);
  auto _rb = reinterpret_cast<uint32_t *>(rb);
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, "
               "{%4, %5}, "
               "{%6}, "
               "{%0, %1, %2, %3};"
               : "+f"(rc[0]), "+f"(rc[1]), "+f"(rc[2]), "+f"(rc[3])
               : "r"(_ra[0]), "r"(_ra[1]), "r"(_rb[0]));

  for (int i = 0; i < 4; i++)
    tc(tvc(threadIdx.x, i)) = rc[i];
}

int _main() {
  using namespace cutlass;
  int M = 16, N = 8, K = 8;
  half *a = nullptr, *b = nullptr;
  float *c = nullptr, *d = nullptr;

  cudaMallocManaged(&a, M * K * sizeof(half));
  cudaMallocManaged(&b, N * K * sizeof(half));
  cudaMallocManaged(&c, M * N * sizeof(float));
  cudaMallocManaged(&d, M * N * sizeof(float));

  for (int i = 0; i < M * K; i++)
    a[i] = half(i);
  for (int i = 0; i < N * K; i++)
    b[i] = half(i);
  for (int i = 0; i < M * N; i++)
    c[i] = float(i);

  mmaAtom_m16n8k8_f32f16f16f32_TN<<<1, 32>>>(a, b, c);
  cudaDeviceSynchronize();

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << cudaGetErrorString(err) << std::endl;
    return -1;
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << c[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}

void gemm_ijk_tn(const half *A, const half *B, float *C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = C[i * N + j];

      for (int k = 0; k < K; ++k) {
        float a = __half2float(A[i * K + k]);
        float b = __half2float(B[k * N + j]);
        acc += a * b;
      }

      C[i * N + j] = acc;
    }
  }
}


int main() {
  constexpr int M = 16;
  constexpr int N = 8;
  constexpr int K = 8;

  std::vector<half> A(M * K);
  std::vector<half> B(K * N);
  std::vector<float> C(M * N);

  for (int i = 0; i < M * K; ++i)
    A[i] = __float2half(static_cast<float>(i));

  for (int i = 0; i < K * N; ++i)
    B[i] = __float2half(static_cast<float>(i));

  for (int i = 0; i < M * N; ++i)
    C[i] = static_cast<float>(i);

  gemm_ijk_tn(A.data(), B.data(), C.data(), M, N, K);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << C[i * N + j] << ' ';
    }
    std::cout << '\n';
  }
  std::cout << std::endl;
  _main();
}