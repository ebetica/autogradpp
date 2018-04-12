#include "test.h"
#include "cuda.h"
#include <thread>

CASE("misc/no_grad/1") {
  no_grad_guard guard;
  auto model = Linear(5, 2).make();
  auto x = Var(at::CPU(at::kFloat).randn({10, 5}), true);
  auto y = model->forward({x})[0];
  Variable s = y.sum();

  backward(s);
  EXPECT(!model->parameters()["weight"].grad().defined());
};

CASE("misc/random/seed_cpu") {
  int size = 100;
  setSeed(7);
  auto x1 = Var(at::CPU(at::kFloat).randn({size}));
  setSeed(7);
  auto x2 = Var(at::CPU(at::kFloat).randn({size}));

  auto l_inf = (x1.data() - x2.data()).abs().max().toCFloat();
  EXPECT(l_inf < 1e-10);
};

CASE("misc/random/seed_cuda") {
  CUDA_GUARD;
  int size = 100;
  setSeed(7);
  auto x1 = Var(at::CUDA(at::kFloat).randn({size}));
  setSeed(7);
  auto x2 = Var(at::CUDA(at::kFloat).randn({size}));

  auto l_inf = (x1.data() - x2.data()).abs().max().toCFloat();
  EXPECT(l_inf < 1e-10);
};

void makeRandomNumber() {
  cudaSetDevice(std::rand() % 2);
  auto x = at::CUDA(at::kFloat).randn({1000});
  std::cout << x.sum() << std::endl;
}

CASE("misc/derp") {
  auto threads = std::vector<std::thread>();
  for (auto i = 0; i < 1000; i++) {
    threads.emplace_back(makeRandomNumber);
  }
  for (auto& t : threads) {
    t.join();
  }
};
