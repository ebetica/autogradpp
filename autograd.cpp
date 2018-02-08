#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

#include "autograd.h"

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

std::map<std::string, Variable> ContainerImpl::parameters() const {
  std::map<std::string, Variable> ret;
  for (auto pair : children_) {
    auto& name = pair.first;
    auto& child = pair.second;
    for (auto& p : child->parameters()) {
      ret[name + "/" + p.first] = p.second;
    }
  }
  for (auto pair : params_) {
    ret[pair.first] = pair.second;
  }
  return ret;
}

void ContainerImpl::cuda() {
  for (auto& pair : children_) {
    pair.second->cuda();
  }
  // Can't do in place operation since .toBackend isn't implemented for variables
  /*
  for (auto& pair : params_) {
    Variable(pair.second.toBackend(at::kCUDA)).detach_();
  }
  */
  cuda_ = true;
  // So we hack it...
  auto copied = params_;
  initialize_parameters();
  for (auto pair : params_) {
    pair.second.data().copy_(copied[pair.first].data());
  }
};

void ContainerImpl::cpu() {
  for (auto& pair : children_) {
    pair.second->cpu();
  }
  cuda_ = false;
  // So we hack it...
  auto copied = params_;
  initialize_parameters();
  for (auto pair : params_) {
    pair.second.data().copy_(copied[pair.first].data());
  }
};

void ContainerImpl::train() {
  for (auto& pair : children_) {
    pair.second->train();
  }
  train_ = true;
}

void ContainerImpl::eval() {
  for (auto& pair : children_) {
    pair.second->eval();
  }
  train_ = false;
}

Container ContainerImpl::add(Container m, std::string const& name) {
  this->children_[name] = std::move(m);
  return this->children_[name];
}

Variable& ContainerImpl::add(Variable v, std::string const& name) {
  this->params_[name] = v;
  return this->params_[name];
}

at::Type& ContainerImpl::DefaultTensor(at::ScalarType s) {
  if (cuda_) return at::CUDA(s);
  else return at::CPU(s);
}

variable_list Linear::forward(variable_list input) {
  auto x = input[0];
  if (x.ndimension() == 2 && !no_bias_) {
    // Fused op is marginally faster
    assert(x.size(1) == weight.size(1));
    return variable_list({at::addmm(bias, x, weight.t())});
  }

  auto output = x.matmul(weight.t());
  if (!no_bias_) {
    output += bias;
  }
  return variable_list({output});
}

void Linear::reset_parameters() {
  auto stdv = 1.0 / std::sqrt(weight.size(1));
  for (auto& p : parameters()) {
    p.second.data().uniform_(-stdv, stdv);
  }
}

void Linear::initialize_parameters() {
  weight = this->add(Var(DefaultTensor(at::kFloat).tensor({nout, nin}), true), "weight");
  if (!no_bias_) {
    bias = this->add(Var(DefaultTensor(at::kFloat).tensor({nout}), true), "bias");
  }
}

variable_list Embedding::forward(variable_list input) {
  auto x = input[0];
  return variable_list({at::embedding(weight, x, -1, false, false)});
}

void Embedding::reset_parameters() {
  for (auto& p : parameters()) {
    p.second.data().normal_(0, 1);
  }
}

void Embedding::initialize_parameters() {
  weight = this->add(Var(DefaultTensor(at::kFloat).tensor({num_embeddings, embedding_dim}), true), "weight");
}

void Conv::initialize_parameters() {
  if (!transposed_) {
    for (auto pad : output_padding_) {
      if (pad != 0) {
        throw std::runtime_error("Only transposed convolutions support output padding!");
      }
    }
  }

  IntVec wsize;
  if (transposed_) {
    wsize.push_back(in_channels_);
    wsize.push_back(out_channels_ / groups_);
  } else {
    wsize.push_back(out_channels_);
    wsize.push_back(in_channels_ / groups_);
  }
  wsize.insert(wsize.end(), ks_.begin(), ks_.end());
  weight = this->add(Var(DefaultTensor(at::kFloat).tensor(wsize), true), "weight");
  if (!no_bias_) {
    bias = this->add(Var(DefaultTensor(at::kFloat).tensor({out_channels_}), true), "bias");
  } else {
    assert(!bias.defined());
  }
}

void Conv::reset_parameters() {
  auto n = in_channels_;
  for (auto k : ks_) n *= k;
  auto stdv = 1.0 / std::sqrt(n);
  for (auto& p : parameters()) {
    p.second.data().uniform_(-stdv, stdv);
  }
}

variable_list Conv::forward(variable_list input) {
  auto x = input[0];
  if (Nd_ == 1) {
    assert(x.ndimension() == 3);
    x = x.unsqueeze(-1);  // TODO: Use conv1d once available
  } else if (Nd_ == 2) {
    assert(x.ndimension() == 4);
  } else if (Nd_ == 3) {
    assert(x.ndimension() == 5);
  } else {
    throw std::runtime_error("Only Conv{1,2,3}d are supported");
  }

  Variable out;
  if (Nd_ == 1 || Nd_ == 2) {
    if (transposed_) {
      out = at::conv_transpose2d(x, weight, bias, stride_, padding_, output_padding_, 1, dilation_);
    } else {
      out = at::conv2d(x, weight, bias, stride_, padding_, dilation_);
    }
  } else if (Nd_ == 3) {
    if (transposed_) {
      out = at::conv_transpose3d(x, weight, bias, stride_, padding_, output_padding_, 1, dilation_);
    } else {
      out = at::conv3d(x, weight, bias, stride_, padding_, dilation_);
    }
  }

  return variable_list({out});
}

void BatchNorm::initialize_parameters() {
  if (affine_) {
    weight = this->add(Var(DefaultTensor(at::kFloat).tensor(num_features_), true), "weight");
    bias = this->add(Var(DefaultTensor(at::kFloat).tensor(num_features_), true), "bias");
  }

  if (stateful_) {
    running_mean = Var(DefaultTensor(at::kFloat).zeros({num_features_}), false);
    running_var = Var(DefaultTensor(at::kFloat).ones({num_features_}), false);
  }
}

void BatchNorm::reset_parameters() {
  if (affine_) {
    weight.data().uniform_();
    bias.data().zero_();
  }

  if (stateful_) {
    running_mean.data().zero_();
    running_var.data().fill_(1);
  }
}

variable_list BatchNorm::forward(variable_list inputs) {
  auto& input = inputs[0];
  auto& running_mean = (stateful_ ? this->running_mean : inputs[1]);
  auto& running_var = (stateful_ ? this->running_var : inputs[2]);

  if (train_) {
    const auto num_channels = input.dim() > 1 ? input.size(1) : 1;
    if (input.numel() / num_channels <= 1) {
      throw std::runtime_error(
          "BatchNorm expected more than 1 value per channel when training!");
    }
  }

  auto output = at::batch_norm(
      input,
      weight, bias,
      running_mean, running_var,
      train_, momentum_, eps_,
      AT_CUDNN_ENABLED());

  return variable_list({output});
}

void LSTM::initialize_parameters() {
  i2h.clear();
  h2h.clear();
  auto gate_size = 4 * hidden_size_;
  for (auto i = 0U; i < nlayers_; i++) {
    auto input_size = (i == 0) ? input_size_ : hidden_size_;
    hiddens_.push_back(tag::Variable());
    i2h.push_back(add(Linear(input_size, gate_size).no_bias(no_bias_).make(), "h2h"));
    h2h.push_back(add(Linear(hidden_size_, gate_size).no_bias(no_bias_).make(), "i2h"));
  }
  if (dropout_ > 0) dropout_module = Dropout(dropout_).make();
}

void LSTM::reset_parameters() {
  auto stdv = 1.0 / std::sqrt(hidden_size_);
  for (auto& p : parameters()) {
    p.second.data().uniform_(-stdv, stdv);
  }
}

variable_list LSTM::forward(variable_list inputs) {
  auto& inp = inputs[0];
  auto bsz = inp.size(1);
  for (auto i = 0U; i < nlayers_; i++) {
    auto reset = false;
    if (!hiddens_[i].defined()) {
      reset = true;
    } else if (bsz != hiddens_[i].size(0)) {
      std::cerr << "Warning! Batch size doesn't match previous, resetting hiddens..." << std::endl;
      reset = true;
    }
    if (reset) {
      hiddens_[i] = Var(DefaultTensor(at::kFloat).zeros({inp.size(1), hidden_size_}));
    }
  }
  auto output = Var(DefaultTensor(at::kFloat).zeros({inp.size(0), inp.size(1), hidden_size_}));
  for (auto t = 0U; t < inp.size(0); t++ ) {
    auto x = inp.select(0, t);
    for (auto i = 0U; i < nlayers_; i++) {
      auto hid = hiddens_[i];
      auto gates = i2h[i]->forward({x})[0] + h2h[i]->forward({hid})[0];
      gates = gates.view({bsz, 4, hidden_size_});
      auto in_gate = gates.select(1, 0).sigmoid();
      auto in_trans = gates.select(1, 1).tanh();
      auto forget_gate = gates.select(1, 2).sigmoid();
      auto out_gate = gates.select(1, 3).sigmoid();
      x = forget_gate * hid + in_gate * in_trans;
      hiddens_[i] = out_gate * x.tanh();
    }
    auto output_slice = output.select(0, t);
    output_slice.copy_(x);
    if (dropout_ > 0 && t != inp.size(0) - 1) {
      output = dropout_module->forward({output})[0];
    }
  }
  return variable_list({output});
}

variable_list Dropout::forward(variable_list inputs) {
  if (p_ == 0 || !this->train_) return inputs;
  variable_list lst;
  for (auto x : inputs) {
    auto noise = x.data().type().tensor(x.sizes());
    noise = (noise.uniform_(0, 1) > p_).toType(x.type().scalarType()).mul_(1. / (1 - p_));
    lst.push_back(x * Var(noise));
  }
  return lst;
}

variable_list Dropout2d::forward(variable_list inputs) {
  if (p_ == 0 || !this->train_) return inputs;
  variable_list lst;
  for (auto x : inputs) {
    auto noise = x.data().type().tensor({x.size(0), x.size(1), 1, 1});
    noise = (noise.uniform_(0, 1) > p_).toType(x.type().scalarType()).mul_(1. / (1 - p_));
    lst.push_back(x * Var(noise));
  }
  return lst;
}

void OptimizerImpl::zero_grad() {
  for (auto p : model_->parameters()) {
    auto& grad = p.second.grad();
    if (grad.defined()) {
      grad = grad.detach();
      grad.data().zero_();
    }
  }
}

void OptimizerImpl::set_model(Container model) {
  model_ = model;
}

void SGD::step() {
  for (auto& pair : model_->parameters()) {
    auto& name = pair.first;
    auto& grad = pair.second.grad();
    auto& p = pair.second.data();
    if (!grad.defined()) continue;

    auto d_p = grad.data();
    if (weight_decay_ > 0) {
      d_p.add_(p, weight_decay_);
    };

    if (momentum_ != 0) {
      at::Tensor buf;
      if (momentum_buffers_.find(name) == momentum_buffers_.end()) {
        buf = momentum_buffers_[name] = at::zeros_like(p);
        buf.mul_(momentum_).add_(d_p);
      } else {
        buf = momentum_buffers_[name];
        buf.mul_(momentum_).add_(d_p, 1 - dampening_);
      }

      if (nesterov_) {
        d_p = d_p.add(buf, momentum_);
      } else {
        d_p = buf;
      }
    }

    p.add_(d_p, - lr_);
  }
}

void SGD::init_state() {
  momentum_buffers_.clear();
}

/// Adapted from
/// https://github.com/pytorch/pytorch/blob/master/torch/optim/adagrad.py
void Adagrad::step() {
  for (auto& pair : model_->parameters()) {
    auto& name = pair.first;
    auto& grad = pair.second.grad();
    auto& p = pair.second.data();
    if (!grad.defined()) continue;

    auto d_p = grad.data();
    if (weight_decay_ > 0) {
      d_p.add_(p, weight_decay_);
    };
    auto &step = step_[name];
    step += 1.0;
    auto clr = lr_ / (1.0 + (step - 1.0) * lr_decay_);
    at::Tensor buf;
    if (sum_.find(name) == sum_.end()) {
      buf = sum_[name] = at::zeros_like(p);
    } else {
      buf = sum_[name];
    }

    buf.addcmul_(d_p, d_p, 1.0);
    at::Tensor std = buf.sqrt().add_(1e-10);
    p.addcdiv_(d_p, std, -clr);
  }
}

void Adagrad::init_state() {
  sum_.clear();
  step_.clear();
}

void Adam::step() {
  for (auto& pair : model_->parameters()) {
    auto& name = pair.first;
    auto& grad = pair.second.grad();
    auto& p = pair.second.data();
    if (!grad.defined()) continue;

    if (step_buffer_.find(name) == step_buffer_.end()) {
      step_buffer_[name] = 0;
      exp_avg_buffer_[name] = at::zeros_like(p);
      exp_avg_sq_buffer_[name] = at::zeros_like(p);
    }

    auto& step = step_buffer_[name];
    auto& exp_avg = exp_avg_buffer_[name];
    auto& exp_avg_sq = exp_avg_sq_buffer_[name];

    step += 1;

    auto d_p = grad.data();
    if (weight_decay_ > 0) {
      d_p.add_(p, weight_decay_);
    }

     exp_avg.mul_(beta1_).add_(d_p, 1 - beta1_);
     exp_avg_sq.mul_(beta2_).addcmul_(d_p, d_p, 1 - beta2_);

     auto denom = exp_avg_sq.sqrt().add_(eps_);
     auto bias_correction1 = 1 - std::pow(beta1_, step);
     auto bias_correction2 = 1 - std::pow(beta2_, step);
     auto step_size = lr_ * std::sqrt(bias_correction2) / bias_correction1;

     p.addcdiv_(exp_avg, denom, -step_size);
  }
}

void Adam::init_state() {
  step_buffer_.clear();
  exp_avg_buffer_.clear();
  exp_avg_sq_buffer_.clear();
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

} // namespace autograd
