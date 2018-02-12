#pragma once

#include "detail.h"

#include "torch/csrc/autograd/variable.h"

#define AUTOGRAD_CONTAINER_CLASS(Type) class Type : public autograd::Container_CRTP<Type>

namespace autograd {
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
    for (std::size_t i = 0; i < size; i++) {
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

AUTOGRAD_CONTAINER_CLASS(BatchNorm) {
 public:
  BatchNorm(uint32_t num_features)
    : num_features_(num_features) {}

  AUTOGRAD_KWARG(BatchNorm, double, eps, 1e-5, 1e-5)
  AUTOGRAD_KWARG(BatchNorm, double, momentum, 0.1, 0.1)
  AUTOGRAD_KWARG(BatchNorm, bool, affine, true, true)
  AUTOGRAD_KWARG(BatchNorm, bool, stateful, false, true)

  void reset_parameters() override;
  variable_list forward(variable_list) override;
  void initialize_parameters() override;

  Variable weight;
  Variable bias;
  Variable running_mean;
  Variable running_var;

 protected:
  uint32_t num_features_;
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
} // namespace autograd
