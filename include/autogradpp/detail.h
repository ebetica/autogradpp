#pragma once

#include <map>
#include <memory>

#include <mapbox/variant.hpp>

#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/grad_mode.h"

// for AutoGPU. Usage:
//   AutoGPU gpu_raii(1);
// While this object is in scope, all of your GPU tensors will go to GPU 1
#include "torch/csrc/utils/auto_gpu.h"

#define AUTOGRAD_OPTIMIZER_CLASS(Type) \
  class Type : public autograd::Optimizer_CRTP<Type>
#define AUTOGRAD_KWARG(CLS, TYP, NAME, DEFAULT, OPTION) \
  TYP NAME##_ = DEFAULT;                                \
  CLS& NAME(TYP x = OPTION) {                           \
    NAME##_ = x;                                        \
    return *this;                                       \
  }

namespace {
namespace tag = torch::autograd;
using IntVec = decltype(std::declval<at::IntList>().vec());
} // namespace

namespace autograd {
namespace detail {
extern tag::Engine engine;
}

class ContainerImpl;
class OptimizerImpl;
using Variable = tag::Variable;
using Tensor = at::Tensor;
using Container = std::shared_ptr<ContainerImpl>;
using Optimizer = std::shared_ptr<OptimizerImpl>;

#define GEN_TYPE(TYP, NAME) \
  Variant(TYP);             \
  bool is ## NAME () const; \
  TYP get ## NAME () const;

class Variant {
 public:
  Variant(Tensor);
  Variant(Variable);
  Variant(const std::string&);
  Variant(std::vector<Variant>&);
  Variant(std::vector<Variant>&&);
  Variant(std::initializer_list<Variable>);
  Variant(std::unordered_map<std::string, Variant>&);
  Variant(std::unordered_map<std::string, Variant>&&);

  Variable const&                                       get() const;
  std::string const&                                    getString() const;
  std::vector<Variant> const&                           getList() const;
  std::unordered_map<std::string, Variant> const&       getDict() const;
  bool                                                  isVariable() const;
  bool                                                  isString() const;
  bool                                                  isList() const;
  bool                                                  isDict() const;
  GEN_TYPE(float,      Float);
  GEN_TYPE(double,     Double);
  GEN_TYPE(bool,       Bool);
  GEN_TYPE(int32_t,    Int32);
  GEN_TYPE(int64_t,    Int64);

  /* These functions will automatically assume you did a .get() */
  // The contract for these will be: If you cannot find it in functions.h, you
  // should define it here. If it's not here then it's a bug.
  template <typename F, typename... Args> 
  auto m(F func, Args&&... params) const {
    return func(get(), std::forward<Args>(params)...);
  } 
  template <typename T> Variable operator+(T other) const { return get() + other; }
  template <typename T> Variable operator-(T other) const { return get() * other; }
  template <typename T> Variable operator*(T other) const { return get() - other; }
  template <typename T> Variable operator/(T other) const { return get() / other; }
  template <typename T>
  Variable operator[](T other) const { return get()[other]; }

  Tensor const& data() const;
  bool defined() const;
  Variable detach() const;
  at::Type& type() const;

 private:
  mapbox::util::variant<
    Variable,
    std::string,
    float,
    double, 
    bool, 
    int32_t,
    int64_t,
    mapbox::util::recursive_wrapper<std::vector<Variant>>,
    mapbox::util::recursive_wrapper<std::unordered_map<std::string, Variant>>
      > variant_;
};

template<> inline Variable Variant::operator+(Variant other) const { return get() + other.get(); }
template<> inline Variable Variant::operator-(Variant other) const { return get() - other.get(); }
template<> inline Variable Variant::operator*(Variant other) const { return get() * other.get(); }
template<> inline Variable Variant::operator/(Variant other) const { return get() / other.get(); }

#undef GEN_TYPE

void backward(Tensor loss, bool keep_graph = false);

inline Variable Var(at::Tensor data, bool requires_grad = true) {
  return tag::make_variable(data, requires_grad);
}

// This is thread local!!!
inline void set_grad_enabled(bool val = true) {
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

void setSeed(uint64_t seed);

int getNumGPUs();
bool hasCuda();
bool hasCudnn();

} // namespace autograd
