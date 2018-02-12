#include <ATen/Config.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

#include "detail.h"

namespace autograd {
namespace detail {
tag::Engine engine;
}

void backward(Variable loss, bool keep_graph) {
  tag::function_list funclst;
  tag::variable_list varlst;
  funclst.emplace_back(loss.grad_fn(), loss.output_nr());
  varlst.emplace_back(Var(at::ones_like(loss.data()), false));
  // create_graph should be set to true when we want to support double bwd
  detail::engine.execute(funclst, varlst, keep_graph, false);
}

void backward(Tensor loss, bool keep_graph) {
  Variable tmp(loss);
  backward(tmp, keep_graph);
}

void setSeed(uint64_t seed) {
  for (auto i = 0; i < static_cast<int>(at::Backend::NumOptions); i++) {
    try {
      at::globalContext()
          .defaultGenerator(static_cast<at::Backend>(i))
          .manualSeed(seed);
    } catch (const std::runtime_error &e) {
      // defaultGenerator() will throw a runtime error for backends that are not
      // available (e.g. CUDA on non-GPU machines).
      // We ignore those at the moment.
      continue;
    }
  }
};

bool hasCuda() {
  return AT_CUDA_ENABLED();
}
bool hasCudnn() {
  return AT_CUDNN_ENABLED();
}

} // namespace autograd
