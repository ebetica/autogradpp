#include <functional>
#include <map>
#include <regex>
#include <math.h>
#include "cereal/archives/portable_binary.hpp"
#include "autograd.h"
using namespace autograd;

// I'm just testing that the sizes match, and hopefully the pytorch tests
// will handle all the gradient bits...
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define EXPECT(CODE) if (CODE); else { throw std::runtime_error(__FILE__ ":" STR(__LINE__) ": " #CODE); }

#if AT_CUDA_ENABLED()
#define CUDA_GUARD
#else
#define CUDA_GUARD std::cerr << "No cuda, skipping test" << std::endl; return
#endif

class CartPole {
  // Translated from openai/gym's cartpole.py
  public:
    double gravity = 9.8;
    double masscart = 1.0;
    double masspole = 0.1;
    double total_mass = (masspole + masscart);
    double length = 0.5; // actually half the pole's length;
    double polemass_length = (masspole * length);
    double force_mag = 10.0;
    double tau = 0.02;  // seconds between state updates;

    // Angle at which to fail the episode
    double theta_threshold_radians = 12 * 2 * M_PI / 360;
    double x_threshold = 2.4;
    int steps_beyond_done = -1;

    at::Tensor state;
    double reward;
    bool done;
    int step_ = 0;

    at::Tensor getState() {
      return state;
    }

    double getReward() {
      return reward;
    }

    double isDone() {
      return done;
    }

    void reset() {
      state = at::CPU(at::kFloat).tensor({4}).uniform_(-0.05, 0.05);
      steps_beyond_done = -1;
      step_ = 0;
    }

    CartPole() {
      reset();
    }

    void step(int action) {
      auto x = state[0].toCFloat();
      auto x_dot = state[1].toCFloat();
      auto theta = state[2].toCFloat();
      auto theta_dot = state[3].toCFloat();

      auto force = (action == 1) ? force_mag : -force_mag;
      auto costheta = std::cos(theta);
      auto sintheta = std::sin(theta);
      auto temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
      auto thetaacc = (gravity * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass));
      auto xacc  = temp - polemass_length * thetaacc * costheta / total_mass;

      x  = x + tau * x_dot;
      x_dot = x_dot + tau * xacc;
      theta = theta + tau * theta_dot;
      theta_dot = theta_dot + tau * thetaacc;
      state[0] = x;
      state[1] = x_dot;
      state[2] = theta;
      state[3] = theta_dot;
      done =  x < - x_threshold
           || x > x_threshold
           || theta < -theta_threshold_radians
           || theta > theta_threshold_radians
           || step_ > 200;

      if (!done) {
        reward = 1.0;
      } else if (steps_beyond_done == -1) {
        // Pole just fell!
        steps_beyond_done = 0;
        reward = 0;
      } else {
        if (steps_beyond_done == 0) {
          assert(false); // Can't do this
        }
      }
      step_++;

    };
};

std::map<std::string, std::function<void()>> construct_tests() {
 std::map<std::string, std::function<void()>> tests;

 tests["autograd/no_grad/1"] = []() {
   no_grad_guard guard;
   auto model = Linear(5, 2).make();
   auto x = Var(at::CPU(at::kFloat).randn({10, 5}), true);
   auto y = model->forward({x})[0];
   Variable s = y.sum();

   backward(s);
   EXPECT(!model->parameters()["weight"].grad().defined());
 };

 tests["autograd/conv2d/even"] = []() {
    auto model = Conv2d(3, 2, 3).stride(2).make();
    auto x = Var(at::CPU(at::kFloat).randn({2, 3, 5, 5}), true);
    auto y = model->forward({x})[0];
    Variable s = y.sum();

    backward(s);
    EXPECT(y.ndimension() == 4);
    EXPECT(s.ndimension() == 1);
    for (auto i = 0; i < 4; i++) {
      EXPECT(y.size(i) == 2);
    }

    EXPECT(model->parameters()["weight"].grad().numel() == 3 * 2 * 3 * 3);
  };

 tests["autograd/conv2d/uneven"] = []() {
    auto model = Conv2d(3, 2, IntVec({3, 2})).stride(2).make();
    auto x = Var(at::CPU(at::kFloat).randn({2, 3, 5, 4}), true);
    auto y = model->forward({x})[0];
    Variable s = y.sum();

    backward(s);
    EXPECT(y.ndimension() == 4);
    EXPECT(s.ndimension() == 1);
    for (auto i = 0; i < 4; i++) {
      EXPECT(y.size(i) == 2);
    }

    EXPECT(model->parameters()["weight"].grad().numel() == 3 * 2 * 3 * 2);
  };

 tests["autograd/conv1d/even"] = []() {
    auto model = Conv1d(3, 2, 3).stride(2).make();
    auto x = Var(at::CPU(at::kFloat).randn({2, 3, 5}), true);
    auto y = model->forward({x})[0];
    Variable s = y.sum();

    backward(s);
    EXPECT(y.ndimension() == 4);
    EXPECT(s.ndimension() == 1);
    for (auto i = 0; i < 3; i++) {
      EXPECT(y.size(i) == 2);
    }

    EXPECT(model->parameters()["weight"].grad().numel() == 3 * 2 * 3);
  };

 tests["autograd/conv3d/even"] = []() {
    auto model = Conv3d(3, 2, 3).stride(2).make();
    auto x = Var(at::CPU(at::kFloat).randn({2, 3, 5, 5, 5}), true);
    auto y = model->forward({x})[0];
    Variable s = y.sum();

    backward(s);
    EXPECT(y.ndimension() == 5);
    EXPECT(s.ndimension() == 1);
    for (auto i = 0; i < 5; i++) {
      EXPECT(y.size(i) == 2);
    }

    EXPECT(model->parameters()["weight"].grad().numel() == 3 * 2 * 3 * 3 * 3);
  };


 tests["autograd/linear/basic1"] = []() {
   auto model = Linear(5, 2).make();
   auto x = Var(at::CPU(at::kFloat).randn({10, 5}), true);
   auto y = model->forward({x})[0];
   Variable s = y.sum();

   backward(s);
   EXPECT(y.ndimension() == 2);
   EXPECT(s.ndimension() == 1);
   EXPECT(y.size(0) == 10);
   EXPECT(y.size(1) == 2);

   EXPECT(model->parameters()["weight"].grad().numel() == 2 * 5);
 };

 tests["autograd/linear/sequential"] = []() {
   auto model = ContainerList()
     .append(Linear(10, 3).make())
     .append(Linear(3, 5).make())
     .append(Linear(5, 100).make())
     .make();

   auto x = Var(at::CPU(at::kFloat).randn({1000, 10}));
   for (auto layer : *model) {
     x = layer->forward({x})[0];
     x = x.clamp_min(0);  // relu
   }

   backward(x);
   EXPECT(x.ndimension() == 2);
   EXPECT(x.size(0) == 1000);
   EXPECT(x.size(1) == 100);
   EXPECT(x.data().min().toCFloat() == 0);
 };

 tests["autograd/linear/simple"] = []() {
   auto model = SimpleContainer().make();
   auto l1 = model->add(Linear(10, 3).make(), "l1");
   auto l2 = model->add(Linear(3, 5).make(), "l2");
   auto l3 = model->add(Linear(5, 100).make(), "l3");

   auto x = Var(at::CPU(at::kFloat).randn({1000, 10}));
   x = l1->forward({x})[0].clamp_min(0);
   x = l2->forward({x})[0].clamp_min(0);
   x = l3->forward({x})[0].clamp_min(0);

   backward(x);
   EXPECT(x.ndimension() == 2);
   EXPECT(x.size(0) == 1000);
   EXPECT(x.size(1) == 100);
   EXPECT(x.data().min().toCFloat() == 0);
 };

 tests["autograd/embedding/basic"] = []() {
   int dict_size = 10;
   auto model = Embedding(dict_size, 2).make();
   // Cannot get gradients to change indices (input) - only for embedding params
   auto x = Var(at::CPU(at::kLong).tensor({10}).fill_(dict_size - 1), false);
   auto y = model->forward({x})[0];
   Variable s = y.sum();

   backward(s);
   EXPECT(y.ndimension() == 2);
   EXPECT(s.ndimension() == 1);
   EXPECT(y.size(0) == 10);
   EXPECT(y.size(1) == 2);

   EXPECT(model->parameters()["weight"].grad().numel() == 2 * dict_size);
 };

 tests["autograd/embedding/list"] = []() {
   auto model = Embedding(6, 4).make();
   auto x = Var(at::CPU(at::kLong).tensor({2, 3}).fill_(5), false);
   auto y = model->forward({x})[0];
   Variable s = y.sum();

   backward(s);
   EXPECT(y.ndimension() == 3);
   EXPECT(y.size(0) == 2);
   EXPECT(y.size(1) == 3);
   EXPECT(y.size(2) == 4);
 };

 tests["autograd/cuda/1"] = []() {
   CUDA_GUARD;
   auto model = Linear(5, 2).make();
   model->cuda();
   auto x = Var(at::CUDA(at::kFloat).randn({10, 5}), true);
   auto y = model->forward({x})[0];
   Variable s = y.sum();

   backward(s);
   EXPECT(y.ndimension() == 2);
   EXPECT(s.ndimension() == 1);
   EXPECT(y.size(0) == 10);
   EXPECT(y.size(1) == 2);

   EXPECT(model->parameters()["weight"].grad().numel() == 2 * 5);
 };

 tests["autograd/cuda/2"] = []() {
   CUDA_GUARD;
   auto model = Linear(5, 2).make();
   model->cuda();
   model->cpu();
   auto x = Var(at::CPU(at::kFloat).randn({10, 5}), true);
   auto y = model->forward({x})[0];
   Variable s = y.sum();

   backward(s);
   EXPECT(y.ndimension() == 2);
   EXPECT(s.ndimension() == 1);
   EXPECT(y.size(0) == 10);
   EXPECT(y.size(1) == 2);

   EXPECT(model->parameters()["weight"].grad().numel() == 2 * 5);
 };

 tests["autograd/dropout/1"] = []() {
   auto dropout = Dropout(0.5).make();
   Variable x = Var(at::CPU(at::kFloat).ones(100));
   Variable y = dropout->forward({x})[0];

   backward(y);
   EXPECT(y.ndimension() == 1);
   EXPECT(y.size(0) == 100);
   EXPECT(y.sum().toCFloat() < 130); // Probably
   EXPECT(y.sum().toCFloat() > 70); // Probably

   dropout->eval();
   y = dropout->forward({x})[0];
   EXPECT(y.data().sum().toCFloat() == 100);
 };

 tests["autograd/LSTM/1"] = []() {
   auto model = LSTM(128, 64).nlayers(2).dropout(0.2).make();
   Variable x = Var(at::CPU(at::kFloat).randn({10, 16, 128}));
   auto out = model->forward({x})[0];
   auto y = x.mean();

   backward(y);
   EXPECT(out.ndimension() == 3);
   EXPECT(out.size(0) == 10);
   EXPECT(out.size(1) == 16);
   EXPECT(out.size(2) == 64);

   EXPECT(model->hiddens()[0].ndimension() == 2);
   EXPECT(model->hiddens()[0].size(0) == 16);
   EXPECT(model->hiddens()[0].size(1) == 64);
   EXPECT(model->hiddens()[1].ndimension() == 2);
   EXPECT(model->hiddens()[1].size(0) == 16);
   EXPECT(model->hiddens()[1].size(1) == 64);

   // Something is in the hiddens
   EXPECT(model->hiddens()[0].data().norm().toCFloat() > 0);
   EXPECT(model->hiddens()[1].data().norm().toCFloat() > 0);

   Variable saved_hidden = model->hiddens()[0];
   model->forward({x})[0];
   Variable diff = model->hiddens()[0] - saved_hidden;

   // Hiddens changed
   EXPECT(diff.data().abs().sum().toCFloat() > 1e-3)
 };

 tests["autograd/optim/sgd"] = []() {
   // We better be able to learn XOR
   auto model = ContainerList()
     .append(Linear(2, 8).make())
     .append(Linear(8, 1).make())
     .make();

   auto optim = SGD(model, 1e-1).momentum(0.9).nesterov().weight_decay(1e-6).make();

   float running_loss = 1;
   int epoch = 0;
   while (running_loss > 0.1) {
     auto bs = 4U;
     auto inp = at::CPU(at::kFloat).tensor({bs, 2});
     auto lab = at::CPU(at::kFloat).tensor({bs});
     for (auto i = 0U; i < bs; i++) {
       auto a = std::rand() % 2;
       auto b = std::rand() % 2;
       auto c = a ^ b;
       inp[i][0] = a;
       inp[i][1] = b;
       lab[i] = c;
     }
     //
     // forward
     auto x = Var(inp);
     auto y = Var(lab, false);
     for (auto layer : *model) x = layer->forward({x})[0].sigmoid_();
     Variable loss = at::binary_cross_entropy(x, y);

     optim->zero_grad();
     backward(loss);
     optim->step();

     running_loss = running_loss * 0.99 + loss.data().sum().toCFloat() * 0.01;
     EXPECT(epoch < 3000);
     epoch++;
   }
 };

 tests["autograd/serialization/undefined"] = []() {
   auto x = at::Tensor();

   EXPECT(!x.defined());

   auto y = at::CPU(at::kFloat).randn({5});

   std::stringstream ss;
   save(ss, &x);
   load(ss, &y);

   EXPECT(!y.defined());
 };

 tests["autograd/serialization/binary"] = []() {
   auto x = at::CPU(at::kFloat).randn({5, 5});
   auto y = at::Tensor();

   std::stringstream ss;
   {
     cereal::BinaryOutputArchive archive(ss);
     archive(x);
   }
   {
     cereal::BinaryInputArchive archive(ss);
     archive(y);
   }

   EXPECT(y.defined());
   EXPECT(x.sizes().vec() == y.sizes().vec());
   EXPECT(x.eq(y).all());
 };

 tests["autograd/serialization/portable_binary"] = []() {
   auto x = at::CPU(at::kFloat).randn({5, 5});
   auto y = at::Tensor();

   std::stringstream ss;
   {
     cereal::PortableBinaryOutputArchive archive(ss);
     archive(x);
   }
   {
     cereal::PortableBinaryInputArchive archive(ss);
     archive(y);
   }

   EXPECT(y.defined());
   EXPECT(x.sizes().vec() == y.sizes().vec());
   EXPECT(x.eq(y).all());
 };

 tests["autograd/serialization/xor"] = []() {
   // We better be able to save and load a XOR model!
   auto makeModel = []() {
     return ContainerList()
     .append(Linear(2, 8).make())
     .append(Linear(8, 1).make())
     .make();
   };
   auto getLoss = [](std::shared_ptr<ContainerList> model, uint32_t bs) {
     auto inp = at::CPU(at::kFloat).tensor({bs, 2});
     auto lab = at::CPU(at::kFloat).tensor({bs});
     for (auto i = 0U; i < bs; i++) {
       auto a = std::rand() % 2;
       auto b = std::rand() % 2;
       auto c = a ^ b;
       inp[i][0] = a;
       inp[i][1] = b;
       lab[i] = c;
     }

     // forward
     auto x = Var(inp);
     auto y = Var(lab, false);
     for (auto layer : *model) x = layer->forward({x})[0].sigmoid_();
     return at::binary_cross_entropy(x, y);
   };

   auto model = makeModel();
   auto model2 = makeModel();
   auto model3 = makeModel();
   auto optim = SGD(model, 1e-1).momentum(0.9).nesterov().weight_decay(1e-6).make();

   float running_loss = 1;
   int epoch = 0;
   while (running_loss > 0.1) {
     Variable loss = getLoss(model, 4);
     optim->zero_grad();
     backward(loss);
     optim->step();

     running_loss = running_loss * 0.99 + loss.data().sum().toCFloat() * 0.01;
     EXPECT(epoch < 3000);
     epoch++;
   }

   std::stringstream ss;
   save(ss, model);
   load(ss, model2);

   auto loss = getLoss(model2, 100);
   EXPECT(loss.toCFloat() < 0.1);

   CUDA_GUARD;
   model2->cuda();
   ss.clear();
   save(ss, model2);
   load(ss, model3);

   loss = getLoss(model3, 100);
   EXPECT(loss.toCFloat() < 0.1);
 };

 tests["autograd/serialization/optim"] = []() {
   auto model1 = Linear(5, 2).make();
   auto model2 = Linear(5, 2).make();
   auto model3 = Linear(5, 2).make();

   // Models 1, 2, 3 will have the same params
   std::stringstream ss;
   save(ss, model1);
   load(ss, model2);
   ss.seekg(0, std::ios::beg);
   load(ss, model3);

   // Make some optimizers with momentum (and thus state)
   auto optim1 = SGD(model1, 1e-1).momentum(0.9).make();
   auto optim2 = SGD(model2, 1e-1).momentum(0.9).make();
   auto optim2_2 = SGD(model2, 1e-1).momentum(0.9).make();
   auto optim3 = SGD(model3, 1e-1).momentum(0.9).make();
   auto optim3_2 = SGD(model3, 1e-1).momentum(0.9).make();

   auto x = Var(at::CPU(at::kFloat).ones({10, 5}), true);

   auto step = [&](Optimizer optim, Container model) {
     optim->zero_grad();
     auto y = model->forward({x})[0].sum();
     backward(y);
     optim->step();
   };

   // Do 2 steps of model1
   step(optim1, model1);
   step(optim1, model1);

   // Do 2 steps of model 2 without saving the optimizer
   step(optim2, model2);
   step(optim2_2, model2);

   // Do 2 steps of model 3 while saving the optimizer
   step(optim3, model3);
   ss.clear();
   save(ss, optim3);
   load(ss, optim3_2);
   step(optim3_2, model3);

   auto param1 = model1->parameters();
   auto param2 = model2->parameters();
   auto param3 = model3->parameters();
   for (auto& p : param1) {
     auto name = p.first;
     // Model 1 and 3 should be the same
     EXPECT(param1[name].norm().toCFloat() == param3[name].norm().toCFloat());
     EXPECT(param1[name].norm().toCFloat() != param2[name].norm().toCFloat());
   }
 };

 auto test_mnist = [](
     uint32_t batch_size, uint32_t num_epochs, bool useGPU,
     auto& model, auto& forward_op, auto& optim) {
   CUDA_GUARD;
   std::cout << "Training MNIST for " << num_epochs << " epochs, rest your eyes for a bit!\n";
   struct MNIST_Reader
   {
     FILE *fp_;

     MNIST_Reader(const char *path) {
       fp_ = fopen(path, "rb");
       if (!fp_) throw std::runtime_error("failed to open file");
     }

     ~MNIST_Reader() { if (fp_) fclose(fp_); }

     int32_t read_int() {
       uint8_t buf[4];
       if (fread(buf, sizeof(buf), 1, fp_) != 1) throw std::runtime_error("failed to read an integer");
       return int32_t(buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3]);
     }

     uint8_t read_byte() {
       uint8_t i;
       if (fread(&i, sizeof(i), 1, fp_) != 1) throw std::runtime_error("failed to read an byte");
       return i;
     }
   };

   auto readData = [&](std::string fn) {
     MNIST_Reader rd(fn.c_str());

     /* int image_magic = */ rd.read_int();
     int image_count = rd.read_int();
     int image_rows = rd.read_int();
     int image_cols = rd.read_int();

     auto data = at::CPU(at::kFloat).tensor({image_count, 1, image_rows, image_cols});
     auto a_data = data.accessor<float, 4>();

     for (int c = 0; c < image_count; c++) {
       for (int i = 0; i < image_rows; i++) {
         for (int j = 0; j < image_cols; j++) {
           a_data[c][0][i][j] = float(rd.read_byte()) / 255;
         }
       }
     }

     return data.toBackend(useGPU ? at::kCUDA : at::kCPU);
   };

   auto readLabels = [&](std::string fn) {
     MNIST_Reader rd(fn.c_str());
     /* int label_magic = */ rd.read_int();
     int label_count = rd.read_int();

     auto data = at::CPU(at::kLong).tensor({label_count});
     auto a_data = data.accessor<long, 1>();

     for (int i = 0; i < label_count; ++i) {
       a_data[i] = long(rd.read_byte());
     }
     return data.toBackend(useGPU ? at::kCUDA : at::kCPU);
   };

   auto trdata = readData("mnist/train-images-idx3-ubyte");
   auto trlabel = readLabels("mnist/train-labels-idx1-ubyte");
   auto tedata = readData("mnist/t10k-images-idx3-ubyte");
   auto telabel = readLabels("mnist/t10k-labels-idx1-ubyte");

   if (useGPU) {
     model->cuda();
   }

   for (auto epoch = 0U; epoch < num_epochs; epoch++) {
     auto shuffled_inds = std::vector<int>(trdata.size(0));
     for (int i=0; i < trdata.size(0); i++) {
      shuffled_inds[i] = i;
     }
     std::random_shuffle(shuffled_inds.begin(), shuffled_inds.end());

     auto inp = (useGPU ? at::CUDA : at::CPU)(at::kFloat).tensor({batch_size, 1, trdata.size(2), trdata.size(3)});
     auto lab = (useGPU ? at::CUDA : at::CPU)(at::kLong).tensor({batch_size});
     for (auto p = 0U; p < shuffled_inds.size() - batch_size; p++) {
       inp[p % batch_size] = trdata[shuffled_inds[p]];
       lab[p % batch_size] = trlabel[shuffled_inds[p]];

       if (p % batch_size != batch_size - 1) continue;
       Variable x = forward_op(Var(inp));
       Variable y = Var(lab, false);
       Variable loss = at::nll_loss(x, y);

       optim->zero_grad();
       backward(loss);
       optim->step();
     }
   }

   no_grad_guard guard;
   auto result = std::get<1>(forward_op(Var(tedata, false)).max(1));
   Variable correct = (result == Var(telabel)).toType(at::kFloat);
   std::cout << "Num correct: " << correct.data().sum().toCFloat()
     << " out of " << telabel.size(0) << std::endl;
   EXPECT(correct.data().sum().toCFloat() > telabel.size(0) * 0.8);
 };

 tests["autograd/~integration/mnist"] = [test_mnist]() {  // ~ will make it run last :D
   auto model = SimpleContainer().make();
   auto conv1 = model->add(Conv2d(1, 10, 5).make(), "conv1");
   auto conv2 = model->add(Conv2d(10, 20, 5).make(), "conv2");
   auto drop = Dropout(0.3).make();
   auto drop2d = Dropout2d(0.3).make();
   auto linear1 = model->add(Linear(320, 50).make(), "linear1");
   auto linear2 = model->add(Linear(50, 10).make(), "linear2");

   auto forward = [&](Variable x) {
     x = std::get<0>(at::max_pool2d(conv1->forward({x})[0], {2, 2})).clamp_min(0);
     x = conv2->forward({x})[0];
     x = drop2d->forward({x})[0];
     x = std::get<0>(at::max_pool2d(x, {2, 2})).clamp_min(0);

     x = x.view({-1, 320});
     x = linear1->forward({x})[0].clamp_min(0);
     x = drop->forward({x})[0];
     x = linear2->forward({x})[0];
     x = at::log_softmax(x, 1);
     return x;
   };

   auto optim = SGD(model, 1e-2).momentum(0.5).make();

   test_mnist(
       32,  // batch_size
       3,  // num_epochs
       true,  // useGPU
       model, forward, optim);
 };

 tests["autograd/~integration/mnist_batchnorm"] = [test_mnist]() {  // ~ will make it run last :D
   auto model = SimpleContainer().make();
   auto conv1 = model->add(Conv2d(1, 10, 5).make(), "conv1");
   auto batchnorm2d = model->add(
       BatchNorm(10).stateful().make(),
       "batchnorm2d");
   auto conv2 = model->add(Conv2d(10, 20, 5).make(), "conv2");
   auto linear1 = model->add(Linear(320, 50).make(), "linear1");
   auto batchnorm1 = model->add(
       BatchNorm(50).stateful().make(),
       "batchnorm1");
   auto linear2 = model->add(Linear(50, 10).make(), "linear2");

   auto forward = [&](Variable x) {
     x = std::get<0>(at::max_pool2d(conv1->forward({x})[0], {2, 2})).clamp_min(0);
     x = batchnorm2d->forward({x})[0];
     x = conv2->forward({x})[0];
     x = std::get<0>(at::max_pool2d(x, {2, 2})).clamp_min(0);

     x = x.view({-1, 320});
     x = linear1->forward({x})[0].clamp_min(0);
     x = batchnorm1->forward({x})[0];
     x = linear2->forward({x})[0];
     x = at::log_softmax(x, 1);
     return x;
   };

   auto optim = SGD(model, 1e-2).momentum(0.5).make();

   test_mnist(
       32,  // batch_size
       3,  // num_epochs
       true,  // useGPU
       model, forward, optim);
 };

tests["autograd/~integration/cartpole"] = []() {
  std::cout << "Training episodic policy gradient with a critic for up to 3000"
    " episodes, rest your eyes for a bit!\n";
  auto model = SimpleContainer().make();
  auto linear = model->add(Linear(4, 128).make(), "linear");
  auto policyHead = model->add(Linear(128, 2).make(), "policy");
  auto valueHead = model->add(Linear(128, 1).make(), "action");
  auto optim = Adam(model, 1e-3).make();

  std::vector<Variable> saved_log_probs;
  std::vector<Variable> saved_values;
  std::vector<float> rewards;

  auto forward = [&](variable_list inp) {
    auto x = linear->forward(inp)[0].clamp_min(0);
    Variable actions = policyHead->forward({x})[0];
    Variable value = valueHead->forward({x})[0];
    return std::make_tuple(at::softmax(actions, -1), value);
  };

   auto selectAction = [&](at::Tensor state) {
     // Only work on single state right now, change index to gather for batch
     auto out = forward({Var(state, false)});
     auto probs = Variable(std::get<0>(out));
     auto value = Variable(std::get<1>(out));
     auto action = probs.data().multinomial(1)[0].toCInt();
     // Compute the log prob of a multinomial distribution.
     // This should probably be actually implemented in autogradpp...
     auto p = probs / probs.sum(-1, true);
     auto log_prob = p[action].log();
     saved_log_probs.push_back(log_prob);
     saved_values.push_back(value);
     return action;
   };

  auto finishEpisode = [&]() {
    auto R = 0.;
    for (int i = rewards.size() - 1; i >= 0; i--) {
      R = rewards[i] + 0.99 * R;
      rewards[i] = R;
    }
    auto r_t = at::CPU(at::kFloat).tensorFromBlob(rewards.data(), {static_cast<int64_t>(rewards.size())});
    r_t = (r_t - r_t.mean()) / (r_t.std() + 1e-5);

    std::vector<at::Tensor> policy_loss;
    std::vector<at::Tensor> value_loss;
    for (auto i = 0U; i < saved_log_probs.size(); i++) {
      auto r = rewards[i] - saved_values[i].toCFloat();
      policy_loss.push_back(- r * saved_log_probs[i]);
      value_loss.push_back(at::smooth_l1_loss(saved_values[i], Var(at::CPU(at::kFloat).scalarTensor(at::Scalar(rewards[i])), false)));
    }
    auto loss = at::cat(policy_loss).sum() + at::cat(value_loss).sum();

    optim->zero_grad();
    backward(loss);
    optim->step();

    rewards.clear();
    saved_log_probs.clear();
    saved_values.clear();
  };

  auto env = CartPole();
  double running_reward = 10.0;
  for (auto episode = 0; ; episode++) {
    env.reset();
    auto state = env.getState();
    int t = 0;
    for ( ; t < 10000; t++) {
      auto action = selectAction(state);
      env.step(action);
      state = env.getState();
      auto reward = env.getReward();
      auto done = env.isDone();

      rewards.push_back(reward);
      if (done) break;
    }

    running_reward = running_reward * 0.99 + t * 0.01;
    finishEpisode();
    /*
    if (episode % 10 == 0) {
      printf("Episode %i\tLast length: %5d\tAverage length: %.2f\n",
              episode, t, running_reward);
    }
    */
    if (running_reward > 150) break;
    EXPECT(episode < 3000);
  }

};

 return tests;
}

int main(int argc, char** argv) {
  for (auto p : construct_tests()) {
    if (argc == 1) {
      std::cout << "Doing " << p.first << "\n";
      p.second();
    } else {
      auto regex = std::regex(argv[1]);
      if (!std::regex_search(p.first, regex)) continue;
      try {
        std::cout << "Doing " << p.first << "\n";
        p.second();
      } catch(const std::exception & ex) {
        std::cout << "Test failed! " << ex.what() << std::endl;
      }
    }
  }

  std::cout << "Done!\n";
  return 0;
}
