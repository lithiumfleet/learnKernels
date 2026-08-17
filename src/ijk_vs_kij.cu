#include <cassert>
#include <chrono>
#include <ctime>
#include <iostream>
#include <vector>

using namespace std;

using mat = vector<vector<float>>;

mat filledMat(int M, int N, float val) { return mat(M, vector<float>(N, val)); }

void matmulIjk(const mat &a, const mat &b, mat &c) {
  int M = c.size(), N = c[0].size(), K = b.size();

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

void matmulKji(const mat &a, const mat &b, mat &c) {
  int M = c.size(), N = c[0].size(), K = b.size();

  for (int k = 0; k < K; k++) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

struct Timer {
private:
  chrono::time_point<chrono::steady_clock> last;

public:
  void reset() { this->last = std::chrono::steady_clock::now(); }

  void printDiff() {
    auto end = chrono::steady_clock::now();

    auto us =
        chrono::duration_cast<chrono::microseconds>(end - this->last).count();

    cout << "duration: " << us << " us\n";
  }
};

int main() {

  int M = 1024, N = 1024, K = 1024;
  auto a = filledMat(M, K, 1.0f);
  auto b = filledMat(K, N, 2.0f);
  auto c = filledMat(M, N, 0.0f);

  auto timer = Timer{};
  timer.reset();
  matmulIjk(a, b, c);
  timer.printDiff();

  a = filledMat(M, K, 1.0f);
  b = filledMat(K, N, 2.0f);
  c = filledMat(M, N, 0.0f);
  timer.reset();
  matmulKji(a, b, c);
  timer.printDiff();

  return 0;
}