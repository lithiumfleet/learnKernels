#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/stride.hpp"
#include "cute/tensor_impl.hpp"
#include <cuda_fp16.h>

// https://am17an.bearblog.dev/a-gentle-introduction-to-gemm-using-mma-tensor-cores/
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// C = AB + C
__global__ void mmaAtom_m16n8k16_f32f16f16f32_TN(half *a, half *b, float *c) {
  using namespace cute;

  // https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html
  // "Since CuTe layouts return indices rather than coordinates, we choose a
  // column-major encoding of the (m,n) coordinates ... we again need a similar
  // mapping between (m, k) == m + k * M. ..."

  // "Looking down the M mode"
  // txv0: 0 32 64 96 1 33  ... -> (4,8):(32,1)
  // t0vx: 0 16 8 24 128 144 136 152 -> (2,2,2):(16,8,128)
  using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2, _2>>,
                         Stride<Stride<_32, _1>, Stride<_16, _8, _128>>>;

  // "we go across the N mode"
  // txv0: 0 16 32 48 1 17 ... -> (4,8):(16,1)
  // t0vx: 0 8 64 72 -> (2,2):(8,64)
  using BLayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2>>,
                         Stride<Stride<_16, _1>, Stride<_8, _64>>>;

  // txv0: 0 32 64 128 1 ... -> (4,8):(32,1)
  // t0vx: 0 16 8 24 -> (2,2):(16,8)
  using CLayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2>>,
                         Stride<Stride<_32, _1>, Stride<_16, _8>>>;

  auto ta = make_tensor(a, Shape<_16, _16>{}, LayoutRight{});
  auto tb = make_tensor(b, Shape<_8, _16>{});
  auto tc = make_tensor(c, Shape<_16, _8>{}, LayoutRight{});


  auto aTVLayout = ALayout{};
  auto bTVLayout = BLayout{};
  auto cTVLayout = CLayout{};

  half aRegs[8];
  half bRegs[4];
  float cRegs[4];

  for (int i = 0; i < 8; ++i) {
    aRegs[i] = ta(aTVLayout(threadIdx.x, i));
  }
  for (int i = 0; i < 4; ++i) {
    bRegs[i] = tb(bTVLayout(threadIdx.x, i));
  }
  for (int i = 0; i < 4; ++i) {
    cRegs[i] = tc(cTVLayout(threadIdx.x, i));
  }

  auto _aRegs = reinterpret_cast<uint32_t *>(aRegs);
  auto _bRegs = reinterpret_cast<uint32_t *>(bRegs);

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, "
               "{%4, %5, %6, %7}, "
               "{%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(cRegs[0]), "+f"(cRegs[1]), "+f"(cRegs[2]), "+f"(cRegs[3])
               : "r"(_aRegs[0]), "r"(_aRegs[1]), "r"(_aRegs[2]), "r"(_aRegs[3]),
                 "r"(_bRegs[0]), "r"(_bRegs[1]));

  for (int i = 0; i < 4; ++i) {
    tc(cTVLayout(threadIdx.x, i)) = cRegs[i];
  }
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

int _main() {
  using namespace cutlass;
  int M = 16, N = 8, K = 16;
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

  mmaAtom_m16n8k16_f32f16f16f32_TN<<<1, 32>>>(a, b, c);
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
int main() {
  constexpr int M = 16;
  constexpr int N = 8;
  constexpr int K = 16;

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