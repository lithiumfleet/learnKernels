#pragma once

#include "cute/container/tuple.hpp"
#include "cute/layout.hpp"

enum InitMethod { Empty = 0, Fill, Increase };
using enum InitMethod;
using namespace cute;

template <typename Scalar> struct InitArgs {
  InitMethod initMethod = Fill;
  Scalar initVal = Scalar{};
  Scalar factor = Scalar{1};
  InitArgs(InitMethod initMethod = Increase, Scalar factor = Scalar{1})
      : initMethod(initMethod), factor(factor) {}
};

template <typename Scalar>
void fillStatic(auto &a, auto &aSize, Scalar initVal) {
  for (int i = 0; i < aSize; i++)
    a[i] = initVal;
}

template <typename Scalar>
void fillIncrease(auto &a, auto &aSize, Scalar factor = Scalar{1}) {
  for (int i = 0; i < aSize; i++)
    a[i] = static_cast<Scalar>(i) * factor;
}

using namespace cute;
template <typename Scalar>
void fill(auto &a, auto &aSize, InitArgs<Scalar> &initArgs) {
  if (initArgs.initMethod == Increase) {
    fillIncrease(a, aSize, initArgs.factor);
  } else {
    fillStatic(a, aSize, initArgs.initVal);
  }
}

template <typename Scalar, typename ALayout>
auto tensorGen(ALayout aLayout, InitArgs<Scalar> initArgs) {

  Scalar *a = nullptr;
  auto aSize = size(aLayout);
  cudaMallocManaged(&a, aSize * sizeof(Scalar));
  fill(a, aSize, initArgs);
  return a;
}

template <typename Scalar>
auto mmaTensorGen(int M, int N, int K, InitArgs<Scalar> initArgs) {
  auto aLayout = make_layout(make_shape(M, K));
  auto bLayout = make_layout(make_shape(N, K));
  auto cLayout = make_layout(make_shape(M, N));

  auto a = tensorGen(aLayout, initArgs);
  auto b = tensorGen(bLayout, initArgs);
  auto c = tensorGen(cLayout, initArgs);

  return tuple(a, b, c);
}