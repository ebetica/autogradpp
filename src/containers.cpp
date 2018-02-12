#include "containers.h"


namespace autograd {
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

} // namespace autograd
