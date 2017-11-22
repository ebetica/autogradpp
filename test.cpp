#include "autograd.h"
using namespace autograd;

// I'm just testing that the sizes match, and hopefully the pytorch tests
// will handle all the gradient bits...
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define EXPECT(CODE) if (CODE); else { throw std::runtime_error(__FILE__ ":" STR(__LINE__) ": " #CODE); }

std::unordered_map<std::string, void (*)()> constuct_tests() {
 std::unordered_map<std::string, void (*)()> tests;

 tests["autograd/conv2d/even"] = []() {
    at::CPU(at::kFloat).randn({2, 3, 5, 5}).size(0);
    Var(at::CPU(at::kFloat).randn({2, 3, 5, 5}), true).size(0);
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

 tests["autograd/cuda/1"] = []() {
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

 tests["autograd/dropout/1"] = []() {
   auto dropout = Dropout(0.5).make();
   Variable x = Var(at::CPU(at::kFloat).ones(100));
   x = dropout->forward({x})[0];

   backward(x);
   EXPECT(x.ndimension() == 1);
   EXPECT(x.size(0) == 100);
   EXPECT(x.data().sum().toCFloat() < 70); // Probably
   EXPECT(x.data().sum().toCFloat() > 30); // Probably
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


 tests["autograd/integration/mnist"] = []() {
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

     int image_magic = rd.read_int();
     int image_count = rd.read_int();
     int image_rows = rd.read_int();
     int image_cols = rd.read_int();

     std::cout << "Images: magic=" << image_magic << ", count=" << image_count << ", rows=" << image_rows << ", cols=" << image_cols << std::endl;

     auto data = at::CPU(at::kFloat).tensor({image_count, 1, image_rows, image_cols});
     auto a_data = data.accessor<float, 4>();

     for (int c = 0; c < image_count; c++)
       for (int i = 0; i < image_rows; i++)
         for (int j = 0; j < image_cols; j++) {
           a_data[c][0][i][j] = float(rd.read_byte()) / 255;
     }     

     return data.toBackend(at::kCPU);
   };

   auto readLabels = [&](std::string fn) {
     MNIST_Reader rd(fn.c_str());
     int label_magic = rd.read_int();
     int label_count = rd.read_int();
     std::cout << "Labels: magic=" << label_magic << ", count=" << label_count << std::endl;

     auto data = at::CPU(at::kLong).tensor({label_count});
     auto a_data = data.accessor<long, 1>();

     for (int i = 0; i < label_count; ++i) {
       a_data[i] = long(rd.read_byte());
     }
     return data.toBackend(at::kCPU);
   };

   auto trdata = readData("mnist/train-images-idx3-ubyte");
   auto trlabel = readLabels("mnist/train-labels-idx1-ubyte");
   auto tedata = readData("mnist/t10k-images-idx3-ubyte");
   auto telabel = readLabels("mnist/t10k-labels-idx1-ubyte");

   auto convolutions = ContainerList()
     .append(Conv2d(1, 32, 3).make())
     .append(Conv2d(32, 32, 3).make())
     .append(Conv2d(32, 32, 3).padding(1).stride(2).make())
     .append(Conv2d(32, 32, 3).make())
     .append(Conv2d(32, 32, 3).make())
     .append(Conv2d(32, 32, 3).stride(2).make())
     .append(Conv2d(32, 32, 3).make())
     .make();
   auto model = SimpleContainer().make();
   model->add(convolutions, "convolutions");
   auto linear = model->add(Linear(32, 10).make(), "linear");
   //model->cuda();
   
   auto optim = SGD(model, 1e-1).momentum(0.9).nesterov().weight_decay(1e-6).make(); 

   auto forward = [&](Variable x) {
     for (auto layer : *convolutions) x = layer->forward({x})[0].clamp_min(0);

     x = x.squeeze();
     x = at::log_softmax(linear->forward({x})[0], 1);
     return x;
   };

   float running_loss = 3;
   int epoch = 0;
   auto bs = 256U;
   for (auto epoch = 0U; epoch < 3; epoch++) {
     auto shuffled_inds = std::vector<int>(trdata.size(0));
     for (int i=0; i < trdata.size(0); i++) {
      shuffled_inds[i] = i;
     }
     std::random_shuffle(shuffled_inds.begin(), shuffled_inds.end());

     auto inp = at::CPU(at::kFloat).tensor({bs, 1, trdata.size(2), trdata.size(3)});
     auto lab = at::CPU(at::kLong).tensor({bs});
     for (auto p = 0U; p < shuffled_inds.size() - bs; p++) {
       inp[p % bs] = trdata[shuffled_inds[p]];
       lab[p % bs] = trlabel[shuffled_inds[p]];

       if (p % bs != bs - 1) continue;
       Variable x = forward(Var(inp));
       Variable y = Var(lab, false);
       Variable loss = at::nll_loss(x, y);

       optim->zero_grad();
       backward(loss);
       optim->step(); 

       auto print_freq = 10;
       if (p % (bs * 10) == bs * 10 - 1) {
         running_loss = running_loss * 0.99 + loss.data().sum().toCFloat() * 0.01;
         std::cout << p << ": " << running_loss << "\n";
       }
     }
   }

   auto result = std::get<1>(forward(Var(tedata, false, true)).max(1));
   auto correct = (result == telabel).sum().toCFloat();
   std::cout << "Num correct: " << correct << std::endl;
   
 };

 return tests;
}

int main(int argc, char** argv) {
  /*
  auto W = Var(at::CPU(at::kFloat).randn({10, 10}), true);
  auto x = Var(at::CPU(at::kFloat).randn({5, 10}), true);
  Variable z = x.mm(W);
  backward(z);
  */
  
  constuct_tests()["autograd/integration/mnist"]();
  /*
  for (auto p : constuct_tests()) {
    if (argc == 1) {
      std::cout << "Doing " << p.first << "\n";
      p.second();
    } else {
      try {
        std::cout << "Doing " << p.first << "\n";
        p.second();
      } catch(const std::exception & ex) {
        std::cout << "Test failed! " << ex.what() << std::endl;
      }
    }
  }
  */

  std::cout << "Done!\n";
  return 0;
}
