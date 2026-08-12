#include <cuda/cmath>
#include <iostream>
#include <vector>

__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c,
                                             int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  tid *= 4;

  if ((tid + 3) < N) {
    float4 *ra = reinterpret_cast<float4 *>(&a[tid]);
    float4 *rb = reinterpret_cast<float4 *>(&b[tid]);
    float4 rc;
    rc.x = ra->x + rb->x;
    rc.y = ra->y + rb->y;
    rc.z = ra->z + rb->z;
    rc.w = ra->w + rb->w;
    *reinterpret_cast<float4 *>(&c[tid]) = rc;
  } else if (tid < N) {
    for (int i = tid; i < N; i++) {
      c[tid] = a[tid] + b[tid];
    }
  }
}

int main() {
  int N = 1023;
  int blockSize = 256;
  int gridSize = cuda::ceil_div(N, blockSize);

  auto ha = std::vector<float>(N);
  auto hb = std::vector<float>(N);
  auto hc = std::vector<float>(N);

  float *da, *db, *dc;
  cudaMalloc(&da, N * sizeof(float));
  cudaMalloc(&db, N * sizeof(float));
  cudaMalloc(&dc, N * sizeof(float));

  for (int i = 0; i < N; i ++) {
    ha[i] = float(i) / 10;
    hb[i] = float(i) / 10;
    hc[i] = 0;
  }

  cudaMemcpy(da, ha.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dc, hc.data(), N * sizeof(float), cudaMemcpyHostToDevice);

  elementwise_add_f32x4_kernel<<<gridSize, blockSize>>>(da, db, dc, N);

  cudaMemcpy(ha.data(), da, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), db, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hc.data(), dc, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);

  for (int i = 0; i < 5; i ++) {
    std::cout << hc[i] << " ";
  }

  return 0;
}