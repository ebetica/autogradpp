#pragma once

#include <memory>

#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/variable.h"

#define AUTOGRAD_CONTAINER_CLASS(Type) class Type : public Container_CRTP<Type>
#define AUTOGRAD_OPTIMIZER_CLASS(Type) class Type : public Optimizer_CRTP<Type>
#define AUTOGRAD_KWARG(CLS, TYP, NAME, DEFAULT, OPTION) \
  TYP NAME ## _ = DEFAULT; \
  CLS & NAME(TYP x = OPTION) { NAME ## _ = x; return *this; }


namespace {
namespace tag = torch::autograd;
using IntVec = decltype(std::declval<at::IntList>().vec());
}

namespace autograd {
namespace detail {
extern tag::Engine engine;
}

class ContainerImpl;
using Variable = tag::Variable;
using variable_list = tag::variable_list;
using Tensor = tag::Tensor;
using Container = std::shared_ptr<ContainerImpl>;

void backward(Variable loss, bool keep_graph=false);

inline Variable Var(
    at::Tensor data, bool requires_grad=true, bool is_volatile=false) {
  return tag::make_variable(data, {requires_grad, is_volatile});
}

class ContainerImpl {
 public: 
  virtual void reset_parameters() { };

  virtual variable_list forward(variable_list) = 0;
  virtual void initialize_parameters() { };

  std::unordered_map<std::string, Variable> parameters(); 

  void cuda();
  void train();
  void eval();

  at::Type& DefaultTensor(at::ScalarType s);

  std::unordered_map<std::string, Container> children_;
  std::unordered_map<std::string, Variable> params_;
  bool cuda_ = false;
  bool train_ = true;

 protected:
  Container add(Container, std::string const&);
  // Be careful when registering Tensors that are not variables
  Variable& add(Variable, std::string const&);
};

template <class Derived>
class Container_CRTP : public ContainerImpl {
 public:
  std::shared_ptr<Derived> make() const {
    auto ptr = std::make_shared<Derived>(*static_cast<const Derived*>(this));
    ptr->initialize_parameters();
    ptr->reset_parameters();
    return ptr;
  }
};

AUTOGRAD_CONTAINER_CLASS(ContainerList) {
  // Lets you use a container like a vector without making a new class,
  // just for simple implementations
public:
  virtual variable_list forward(variable_list) override {
    throw std::runtime_error("ContainerList has no forward, maybe you"
        " wanted to subclass and override this function?");
  }

  Container add(Container m) {
    this->children_.push_back(m);
    ContainerImpl::add(this->children_.back(), std::to_string(size() - 1));
    return this->children_.back();
  }

  ContainerList& append(Container m) {
    this->children_.push_back(m);
    ContainerImpl::add(this->children_.back(), std::to_string(size() - 1));
    return *this;
  }

  Container& operator[](int index) {
    return children_[index];
  }

  int size() { return children_.size(); }

  auto begin() { return children_.begin(); }

  auto end() { return children_.end(); }

  std::vector<Container> children_;
};

AUTOGRAD_CONTAINER_CLASS(SimpleContainer) {
  // Lets you use a container without making a new class,
  // for experimental implementations
 public:
  virtual variable_list forward(variable_list) override {
    throw std::runtime_error("SimpleContainer has no forward, maybe you"
        " wanted to subclass and override this function?");
  }
  using ContainerImpl::add;

};

AUTOGRAD_CONTAINER_CLASS(Linear) {
 public:
   Linear(uint32_t nin, uint32_t nout, bool no_bias=true)
     : nin(nin), nout(nout) { }

   variable_list forward(variable_list) override;
   void reset_parameters() override;
   void initialize_parameters() override;
   AUTOGRAD_KWARG(Linear, bool, no_bias, false, true);

  Variable weight, bias;
  uint32_t nin, nout;
};

AUTOGRAD_CONTAINER_CLASS(Conv) {
 private:
  Conv(uint32_t Nd, uint32_t in_chan, uint32_t out_chan) 
    : Nd_(Nd),
      in_channels_(in_chan),
      out_channels_(out_chan),
      stride_(makeTup(1)),
      padding_(makeTup(0)),
      dilation_(makeTup(1)),
      dilated_(false),
      output_padding_(makeTup(0))
      { }

 public:
  Conv(uint32_t Nd, uint32_t in_chan, uint32_t out_chan, int ks) 
    : Conv(Nd, in_chan, out_chan) {
      ks_ = makeTup(ks, 1);
    }

  Conv(uint32_t Nd, uint32_t in_chan, uint32_t out_chan, IntVec ks) 
    : Conv(Nd, in_chan, out_chan) {
      ks_ = makeTup(ks);
    }

  void reset_parameters() override;
  variable_list forward(variable_list) override;
  void initialize_parameters() override;

  template <typename T>
  Conv& stride(T s) { stride_ = makeTup(s, 1); return *this; }
  template <typename T>
  Conv& padding(T s) { padding_ = makeTup(s); return *this; }
  template <typename T>
  Conv& dilation(T s) { dilation_ = makeTup(s, 1); return *this; }
  template <typename T>
  Conv& output_padding(T s) { output_padding_ = makeTup(s); return *this; }

  AUTOGRAD_KWARG(Conv, bool, transposed, false, true)
  AUTOGRAD_KWARG(Conv, bool, no_bias, false, true)
  AUTOGRAD_KWARG(Conv, int, groups, 1, 1)
  
   Variable weight, bias;
   uint32_t Nd_;
   uint32_t in_channels_;
   uint32_t out_channels_;
   IntVec ks_;
   IntVec stride_;
   IntVec padding_;
   IntVec dilation_;
   bool dilated_;
   IntVec output_padding_;
  protected:
   IntVec makeTup(int x, int def=0) {
     IntVec ret;
     if (Nd_ == 1) {
       ret.push_back(x);
       ret.push_back(def);
     } else {
       for (auto i = 0U; i < Nd_; i++) ret.push_back(x);
     }
     return ret;
   }
   IntVec makeTup(IntVec x) {
     return x;
   }
};

class Conv2d : public Conv {
 public:
  Conv2d(uint32_t i, uint32_t o, int ks) : Conv(2, i, o, ks) { } 
  Conv2d(uint32_t i, uint32_t o, IntVec ks) : Conv(2, i, o, ks) { }
};

class Conv1d : public Conv {
 public:
  Conv1d(uint32_t i, uint32_t o, int ks) : Conv(1, i, o, ks) { } 
  Conv1d(uint32_t i, uint32_t o, IntVec ks) : Conv(1, i, o, ks) { }
};

AUTOGRAD_CONTAINER_CLASS(Dropout) {
 public:
  Dropout(double p=0.5) : p_(p) { assert(p < 1 && p >= 0); }
  variable_list forward(variable_list) override;
 protected:
  double p_;
};

AUTOGRAD_CONTAINER_CLASS(Dropout2d) {
 public:
  Dropout2d(double p=0.5) : p_(p) { assert(p < 1 && p >= 0); }
  variable_list forward(variable_list) override;
 protected:
  double p_;
};

AUTOGRAD_CONTAINER_CLASS(LSTM) {
 public:
  LSTM(uint32_t input_size, uint32_t hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size) { }

  AUTOGRAD_KWARG(LSTM, bool, no_bias, false, true)
  AUTOGRAD_KWARG(LSTM, uint32_t, nlayers, false, true)
  AUTOGRAD_KWARG(LSTM, double, dropout, false, true)

  std::vector<Variable>& hiddens() { return hiddens_; }

  void reset_parameters() override;
  variable_list forward(variable_list) override;
  void initialize_parameters() override;

  std::vector<Container> i2h;
  std::vector<Container> h2h;
  std::vector<Variable> hiddens_;
 protected:
  uint32_t input_size_;
  uint32_t hidden_size_;
  Container dropout_module;
};

class OptimizerImpl {
 public:
  OptimizerImpl(Container model) : model_(model) { }
  virtual void init_state() { }
  virtual void step() = 0;
  void zero_grad();

 protected:
  Container model_;
};

template <class Derived>
class Optimizer_CRTP : public OptimizerImpl {
 public:
  Optimizer_CRTP(Container model) : OptimizerImpl(model) { }
  std::shared_ptr<Derived> make() const {
    auto ptr = std::make_shared<Derived>(*static_cast<const Derived*>(this));
    ptr->init_state();
    return ptr;
  }
};

AUTOGRAD_OPTIMIZER_CLASS(SGD) {
 public:
  SGD(Container model, double lr) : Optimizer_CRTP(model), lr_(lr) { }
  AUTOGRAD_KWARG(SGD, double, momentum, 0, 0);
  AUTOGRAD_KWARG(SGD, double, dampening, 0, 0);
  AUTOGRAD_KWARG(SGD, double, weight_decay, 0, 0);
  AUTOGRAD_KWARG(SGD, bool, nesterov, false, true);
  void step() override;
  
  double lr_;
  std::unordered_map<std::string, at::Tensor> momentum_buffers;
};

}  // namespace autograd
