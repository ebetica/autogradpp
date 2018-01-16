#pragma once

#include <map>
#include <memory>
#include <fstream>

// We have to include these to register optimizers
#include "cereal/archives/binary.hpp"
#include "cereal/archives/xml.hpp"
#include "cereal/archives/json.hpp"
#include "cereal/types/polymorphic.hpp"

#include "cereal/types/vector.hpp"
#include "cereal/types/unordered_map.hpp"
#include "cereal/types/string.hpp"

#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/grad_mode.h"

#define AUTOGRAD_CONTAINER_CLASS(Type) class Type : public autograd::Container_CRTP<Type>
#define AUTOGRAD_OPTIMIZER_CLASS(Type) class Type : public autograd::Optimizer_CRTP<Type>
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
class OptimizerImpl;
using Variable = tag::Variable;
using variable_list = tag::variable_list;
using Tensor = tag::Tensor;
using Container = std::shared_ptr<ContainerImpl>;
using Optimizer = std::shared_ptr<OptimizerImpl>;

void backward(Tensor loss, bool keep_graph=false);

template <typename T>
void save(std::string const& fn, T const & obj) {
  std::ofstream os(fn, std::ios::binary);
  save(os, obj);
}
template <typename T>
void load(std::string const& fn, T& obj) {
  std::ifstream is(fn, std::ios::binary);
  load(is, obj);
}
template <typename T>
void save(std::ostream& stream, T const & obj) {
  cereal::BinaryOutputArchive archive(stream);
  archive(*obj);
}
template <typename T>
void load(std::istream& stream, T& obj) {
  cereal::BinaryInputArchive archive(stream);
  archive(*obj);
}

inline Variable Var(at::Tensor data, bool requires_grad=true) {
  return tag::make_variable(data, requires_grad);
}

// This is thread local!!!
inline void set_grad_enabled(bool val=true) {
  tag::GradMode::set_enabled(val);
}

// RAII thread local lock that stops future execution from building gradients
class no_grad_guard {
 public:
  no_grad_guard() {
    tag::GradMode::set_enabled(false);
  }

  ~no_grad_guard() {
    tag::GradMode::set_enabled(true);
  }
};

class ContainerImpl {
 public: 
  virtual void reset_parameters() { };

  virtual variable_list forward(variable_list) = 0;
  virtual void initialize_parameters() { };

  std::map<std::string, Variable> parameters() const; 

  void cuda();
  void cpu();
  void train();
  void eval();

  at::Type& DefaultTensor(at::ScalarType s);

  std::unordered_map<std::string, Container> children_;
  std::unordered_map<std::string, Variable> params_;
  bool cuda_ = false;
  bool train_ = true;

  template<class Archive>
  void save(Archive & ar) const {
    auto params = parameters();
    std::size_t size = params.size();
    ar(size);
    for (auto& p : params) {
      ar(p.first, p.second);
    }
  }

  template<class Archive>
  void load(Archive & ar) {
    auto params = parameters();
    std::size_t size;
    ar(size);
    std::string name;
    for (int i = 0; i < size; i++) {
      ar(name);
      ar(params[name]);
    }
  }

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
   Linear(uint32_t nin, uint32_t nout)
     : nin(nin), nout(nout) { }

   variable_list forward(variable_list) override;
   void reset_parameters() override;
   void initialize_parameters() override;
   AUTOGRAD_KWARG(Linear, bool, no_bias, false, true);

  Variable weight, bias;
  uint32_t nin, nout;
};

AUTOGRAD_CONTAINER_CLASS(Embedding) {
 public:
   Embedding(uint32_t num_embeddings, uint32_t embedding_dim)
     : num_embeddings(num_embeddings), embedding_dim(embedding_dim) { }

   variable_list forward(variable_list) override;
   void reset_parameters() override;
   void initialize_parameters() override;

  Variable weight;
  uint32_t num_embeddings, embedding_dim;
};

AUTOGRAD_CONTAINER_CLASS(Conv) {
 private:
  Conv(uint32_t Nd, uint32_t in_chan, uint32_t out_chan) 
    : Nd_(Nd),
      in_channels_(in_chan),
      out_channels_(out_chan),
      stride_(makeTup(1, 1)),
      padding_(makeTup(0)),
      dilation_(makeTup(1, 1)),
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

class Conv1d : public Conv {
 public:
  Conv1d(uint32_t i, uint32_t o, int ks) : Conv(1, i, o, ks) { } 
  Conv1d(uint32_t i, uint32_t o, IntVec ks) : Conv(1, i, o, ks) { }
};

class Conv2d : public Conv {
 public:
  Conv2d(uint32_t i, uint32_t o, int ks) : Conv(2, i, o, ks) { } 
  Conv2d(uint32_t i, uint32_t o, IntVec ks) : Conv(2, i, o, ks) { }
};

class Conv3d : public Conv {
 public:
  Conv3d(uint32_t i, uint32_t o, int ks) : Conv(3, i, o, ks) { } 
  Conv3d(uint32_t i, uint32_t o, IntVec ks) : Conv(3, i, o, ks) { }
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
  OptimizerImpl() { }
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
 
 protected:
  Optimizer_CRTP() { }
};

AUTOGRAD_OPTIMIZER_CLASS(SGD) {
 public:
  SGD(Container model, double lr) : Optimizer_CRTP(model), lr_(lr) { }
  AUTOGRAD_KWARG(SGD, double, momentum, 0, 0);
  AUTOGRAD_KWARG(SGD, double, dampening, 0, 0);
  AUTOGRAD_KWARG(SGD, double, weight_decay, 0, 0);
  AUTOGRAD_KWARG(SGD, bool, nesterov, false, true);
  void step() override;

  template <class Archive>
  void serialize(Archive & ar) { 
    ar(CEREAL_NVP(momentum_buffers_)); 
  }
  
 private:
  friend class cereal::access;
  SGD() { }
  double lr_;
  std::unordered_map<std::string, at::Tensor> momentum_buffers_;
};

AUTOGRAD_OPTIMIZER_CLASS(Adam) {
 public:
  using OptimizerImpl::OptimizerImpl;
  Adam(Container model, double lr) : Optimizer_CRTP(model), lr_(lr) { }
  AUTOGRAD_KWARG(Adam, double, beta1, 0.9, 0.9);
  AUTOGRAD_KWARG(Adam, double, beta2, 0.999, 0.999);
  AUTOGRAD_KWARG(Adam, double, weight_decay, 0, 0);
  AUTOGRAD_KWARG(Adam, double, eps, 1e-8, 1e-8);
  void step() override;

  template <class Archive>
  void serialize(Archive & ar) { 
    ar(CEREAL_NVP(step_buffer_),
       CEREAL_NVP(exp_avg_buffer_),
       CEREAL_NVP(exp_avg_sq_buffer_)); 
  }
  
 private:
  friend class cereal::access;
  Adam() { }
  double lr_;
  std::unordered_map<std::string, int> step_buffer_;
  std::unordered_map<std::string, at::Tensor> exp_avg_buffer_;
  std::unordered_map<std::string, at::Tensor> exp_avg_sq_buffer_;
};

}  // namespace autograd

// This is super ugly and I don't know how to simplify it
CEREAL_REGISTER_TYPE(autograd::SGD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(autograd::OptimizerImpl, autograd::SGD);
CEREAL_REGISTER_TYPE(autograd::Adam);
CEREAL_REGISTER_POLYMORPHIC_RELATION(autograd::OptimizerImpl, autograd::Adam);

namespace cereal {
template<class Archive>
void save(Archive & archive, at::Tensor const & tensor) { 
  auto sizes = std::vector<int64_t>();
  auto buf = std::vector<uint8_t>();
  for (auto s : tensor.sizes()) {
    sizes.push_back(s);
  }
  auto contig = tensor.toBackend(at::kCPU).contiguous();
  auto size = tensor.storage()->size() * tensor.storage()->elementSize();
  at::Backend backend = tensor.type().backend();
  at::ScalarType type = tensor.type().scalarType();

  buf.resize(size);
  memcpy(buf.data(), contig.storage()->data(), size);

  archive(
      CEREAL_NVP(type),
      CEREAL_NVP(backend), 
      CEREAL_NVP(sizes), 
      CEREAL_NVP(buf)); 
}

/**
 * We follow these rules for loading:
 * 1. If tensor is defined, and the same ScalarType as the saved tensor,
 *    then we simply copy the data into the tensor, with resizing.
 * 2. Otherwise, overwrite the provided tensor with the right type and backend
 **/
template<class Archive>
void load(Archive & archive, at::Tensor & tensor) {
  auto sizes = std::vector<int64_t>();
  auto buf = std::vector<uint8_t>();
  at::Backend backend;
  at::ScalarType type;
  archive(
      CEREAL_NVP(type),
      CEREAL_NVP(backend), 
      CEREAL_NVP(sizes),
      CEREAL_NVP(buf)); 

  if (!tensor.defined() || tensor.type().scalarType() != type) {
    tensor = at::getType(backend, type).tensor();
  }
  if (tensor.type().is_cuda()) {
    // should actually use cudamemcpy probably
    auto cputensor = at::CPU(tensor.type().scalarType()).tensor(sizes);
    tensor.resize_(sizes);
    memcpy(cputensor.storage()->data(), buf.data(), buf.size());
    tensor.copy_(cputensor);
  } else {
    tensor.resize_(sizes);
    memcpy(tensor.storage()->data(), buf.data(), buf.size());
  }
} 

template<class Archive>
void load(Archive & archive, tag::Variable & tensor) {
  load(archive, tensor.data());
} 
}  // namespace cereal
