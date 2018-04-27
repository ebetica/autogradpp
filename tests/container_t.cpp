#include "test.h"

AUTOGRAD_CONTAINER_CLASS(TestModel) {
 public:
  void initialize_containers() override {
    add(Linear(10, 3).make(), "l1");
    add(Linear(3, 5).make(), "l2");
    add(Linear(5, 100).make(), "l3");
  }

  Variant forward(Variant const& input) override { return input; };
};

AUTOGRAD_CONTAINER_CLASS(NestedModel) {
 public:
  void initialize_containers() override {
    add(Linear(5, 20).make(), "l1");
    add(TestModel().make(), "test");
  }

  void initialize_parameters() override {
    add(Var(DefaultTensor(at::kFloat).tensor({3, 2, 21}), false), "param");
  }

  Variant forward(Variant const& input) override { return input; };
};

CASE("containers/conv2d/even") {
  auto model = Conv2d(3, 2, 3).stride(2).make();
  auto x = Var(at::CPU(at::kFloat).randn({2, 3, 5, 5}), true);
  auto y = model->forward(x).get();
  Variable s = y.sum();

  backward(s);
  EXPECT(y.ndimension() == 4);
  EXPECT(s.ndimension() == 0);
  for (auto i = 0; i < 4; i++) {
    EXPECT(y.size(i) == 2);
  }

  EXPECT(model->parameters()["weight"].grad().numel() == 3 * 2 * 3 * 3);
};

CASE("containers/conv2d/uneven") {
  auto model = Conv2d(3, 2, IntVec({3, 2})).stride(2).make();
  auto x = Var(at::CPU(at::kFloat).randn({2, 3, 5, 4}), true);
  auto y = model->forward(x).get();
  Variable s = y.sum();

  backward(s);
  EXPECT(y.ndimension() == 4);
  EXPECT(s.ndimension() == 0);
  for (auto i = 0; i < 4; i++) {
    EXPECT(y.size(i) == 2);
  }

  EXPECT(model->parameters()["weight"].grad().numel() == 3 * 2 * 3 * 2);
};

CASE("containers/conv1d/even") {
  auto model = Conv1d(3, 2, 3).stride(2).make();
  auto x = Var(at::CPU(at::kFloat).randn({2, 3, 5}), true);
  auto y = model->forward(x).get();
  Variable s = y.sum();

  backward(s);
  EXPECT(y.ndimension() == 4);
  EXPECT(s.ndimension() == 0);
  for (auto i = 0; i < 3; i++) {
    EXPECT(y.size(i) == 2);
  }

  EXPECT(model->parameters()["weight"].grad().numel() == 3 * 2 * 3);
};

CASE("containers/conv3d/even") {
  auto model = Conv3d(3, 2, 3).stride(2).make();
  auto x = Var(at::CPU(at::kFloat).randn({2, 3, 5, 5, 5}), true);
  auto y = model->forward(x).get();
  Variable s = y.sum();

  backward(s);
  EXPECT(y.ndimension() == 5);
  EXPECT(s.ndimension() == 0);
  for (auto i = 0; i < 5; i++) {
    EXPECT(y.size(i) == 2);
  }

  EXPECT(model->parameters()["weight"].grad().numel() == 3 * 2 * 3 * 3 * 3);
};

CASE("containers/linear/basic1") {
  auto model = Linear(5, 2).make();
  auto x = Var(at::CPU(at::kFloat).randn({10, 5}), true);
  auto y = model->forward(x).get();
  Variable s = y.sum();

  backward(s);
  EXPECT(y.ndimension() == 2);
  EXPECT(s.ndimension() == 0);
  EXPECT(y.size(0) == 10);
  EXPECT(y.size(1) == 2);

  EXPECT(model->parameters()["weight"].grad().numel() == 2 * 5);
};

CASE("containers/linear/sequential") {
  auto model = ContainerList()
    .append(Linear(10, 3).make())
    .append(Linear(3, 5).make())
    .append(Linear(5, 100).make())
    .make();

  auto x = Var(at::CPU(at::kFloat).randn({1000, 10}));
  for (auto layer : *model) {
    x = layer->forward(x).get();
    x = x.clamp_min(0);  // relu
  }

  backward(x);
  EXPECT(x.ndimension() == 2);
  EXPECT(x.size(0) == 1000);
  EXPECT(x.size(1) == 100);
  EXPECT(x.data().min().toCFloat() == 0);
};

CASE("containers/linear/simple") {
  auto model = SimpleContainer().make();
  auto l1 = model->add(Linear(10, 3).make(), "l1");
  auto l2 = model->add(Linear(3, 5).make(), "l2");
  auto l3 = model->add(Linear(5, 100).make(), "l3");

  Variable x = Var(at::CPU(at::kFloat).randn({1000, 10}))
    .m(l1->functor()).m(at::relu)
    .m(l2->functor()).m(at::relu)
    .m(l3->functor()).m(at::relu)
    ;

  backward(x);
  EXPECT(x.ndimension() == 2);
  EXPECT(x.size(0) == 1000);
  EXPECT(x.size(1) == 100);
  EXPECT(x.data().min().toCFloat() == 0);
};

CASE("containers/clone") {
  auto model = TestModel().make();

  auto model2 = model->clone();
  auto m1param = model->parameters();
  auto m2param = model2->parameters();
  for (auto& param : m1param) {
    EXPECT(param.second.allclose(m2param[param.first]));
    param.second.data().mul_(2);
  }
  for (auto& param : m1param) {
    EXPECT(!param.second.allclose(m2param[param.first]));
  }
};

CASE("containers/embedding/basic") {
  int dict_size = 10;
  auto model = Embedding(dict_size, 2).make();
  // Cannot get gradients to change indices (input) - only for embedding params
  auto x = Var(at::CPU(at::kLong).tensor({10}).fill_(dict_size - 1), false);
  auto y = model->forward(x).get();
  Variable s = y.sum();

  backward(s);
  EXPECT(y.ndimension() == 2);
  EXPECT(s.ndimension() == 0);
  EXPECT(y.size(0) == 10);
  EXPECT(y.size(1) == 2);

  EXPECT(model->parameters()["weight"].grad().numel() == 2 * dict_size);
};

CASE("containers/embedding/list") {
  auto model = Embedding(6, 4).make();
  auto x = Var(at::CPU(at::kLong).tensor({2, 3}).fill_(5), false);
  auto y = model->forward(x).get();
  Variable s = y.sum();

  backward(s);
  EXPECT(y.ndimension() == 3);
  EXPECT(y.size(0) == 2);
  EXPECT(y.size(1) == 3);
  EXPECT(y.size(2) == 4);
};

CASE("containers/cuda/1") {
  CUDA_GUARD;
  auto model = Linear(5, 2).make();
  model->cuda();
  auto x = Var(at::CUDA(at::kFloat).randn({10, 5}), true);
  auto y = model->forward(x).get();
  Variable s = y.sum();

  backward(s);
  EXPECT(y.ndimension() == 2);
  EXPECT(s.ndimension() == 0);
  EXPECT(y.size(0) == 10);
  EXPECT(y.size(1) == 2);

  EXPECT(model->parameters()["weight"].grad().numel() == 2 * 5);
};

CASE("containers/cuda/2") {
  CUDA_GUARD;
  auto model = Linear(5, 2).make();
  model->cuda();
  model->cpu();
  auto x = Var(at::CPU(at::kFloat).randn({10, 5}), true);
  auto y = model->forward(x).get();
  Variable s = y.sum();

  backward(s);
  EXPECT(y.ndimension() == 2);
  EXPECT(s.ndimension() == 0);
  EXPECT(y.size(0) == 10);
  EXPECT(y.size(1) == 2);

  EXPECT(model->parameters()["weight"].grad().numel() == 2 * 5);
};

CASE("containers/dropout/1") {
  auto dropout = Dropout(0.5).make();
  Variable x = Var(at::CPU(at::kFloat).ones(100));
  Variable y = dropout->forward(x).get();

  backward(y);
  EXPECT(y.ndimension() == 1);
  EXPECT(y.size(0) == 100);
  EXPECT(y.sum().toCFloat() < 130); // Probably
  EXPECT(y.sum().toCFloat() > 70); // Probably

  dropout->eval();
  y = dropout->forward(x).get();
  EXPECT(y.data().sum().toCFloat() == 100);
};

CASE("containers/param") {
  auto model = NestedModel().make();
  EXPECT(model->param("param").size(0) == 3);
  EXPECT(model->param("param").size(1) == 2);
  EXPECT(model->param("param").size(2) == 21);
  EXPECT(model->param("l1.bias").size(0) == 20);
  EXPECT(model->param("l1.weight").size(0) == 20);
  EXPECT(model->param("l1.weight").size(1) == 5);
  EXPECT(model->param("test.l1.bias").size(0) == 3);
  EXPECT(model->param("test.l1.weight").size(0) == 3);
  EXPECT(model->param("test.l1.weight").size(1) == 10);
  EXPECT(model->param("test.l2.bias").size(0) == 5);
  EXPECT(model->param("test.l2.weight").size(0) == 5);
  EXPECT(model->param("test.l2.weight").size(1) == 3);
  EXPECT(model->param("test.l3.bias").size(0) == 100);
  EXPECT(model->param("test.l3.weight").size(0) == 100);
  EXPECT(model->param("test.l3.weight").size(1) == 5);
}
