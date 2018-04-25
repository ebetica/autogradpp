#include "autogradpp/containers.h"

namespace autograd {
std::function<Variant(Variant const&)> ContainerImpl::functor() {
  return [this](Variant const & x) { return this->forward(x); };
}
Variant ContainerImpl::forward(Variant&& x) {
  return this->forward(x);
}

std::map<std::string, Variable> ContainerImpl::parameters() const {
  std::map<std::string, Variable> ret;
  for (auto pair : children_) {
    auto& name = pair.first;
    auto& child = pair.second;
    for (auto& p : child->parameters()) {
      ret[name + "." + p.first] = p.second;
    }
  }
  for (auto pair : params_) {
    ret[pair.first] = pair.second;
  }
  return ret;
}

Variable& ContainerImpl::param(std::string const& name) {
  ContainerImpl* container = this;
  auto begin = 0;
  while (true) {
    auto dot_pos = name.find('.', begin);
    if (dot_pos == std::string::npos) {
      break;
    }

    auto child_name = name.substr(begin, dot_pos - begin);
    auto it = container->children_.find(child_name);
    if (it == container->children_.end()) {
      throw std::runtime_error("No such child: " + child_name);
    }

    container = it->second.get();
    begin = dot_pos + 1; // Skip the dot
  }

  auto param_name = name.substr(begin);
  auto it = container->params_.find(param_name);
  if (it == params_.end()) {
    throw std::runtime_error("No such param: " + param_name);
  }
  return it->second;
}

void ContainerImpl::cuda() {
  for (auto& pair : children_) {
    pair.second->cuda();
  }
  cuda_ = true;
  auto copied = params_;
  params_.clear();
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
  auto copied = params_;
  params_.clear();
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
  if (this->children_.find(name) != this->children_.end()) {
    throw std::runtime_error("Trying to add container that already exists");
  }
  if (std::find(name.begin(), name.end(), '.') != name.end()) {
    // We can't allow containers with dots in their names, as that would make
    // their parameters not findable with parameters().
    throw std::runtime_error("Trying to add parameter with a '.' in its name");
  }
  this->children_[name] = std::move(m);
  return this->children_[name];
}

Variable& ContainerImpl::add(Variable v, std::string const& name) {
  if (this->params_.find(name) != this->params_.end()) {
    throw std::runtime_error("Trying to add parameter that already exists");
  }
  if (std::find(name.begin(), name.end(), '.') != name.end()) {
    // We can't allow parameters with dots in their names, as that would make
    // them not findable with parameters().
    throw std::runtime_error("Trying to add parameter with a '.' in its name");
  }
  this->params_[name] = v;
  return this->params_[name];
}

at::Type& ContainerImpl::DefaultTensor(at::ScalarType s) {
  if (cuda_)
    return at::CUDA(s);
  else
    return at::CPU(s);
}

Variant Linear::forward(Variant const& input) {
  auto x = input.get();
  if (x.ndimension() == 2 && !no_bias_) {
    // Fused op is marginally faster
    assert(x.size(1) == weight.size(1));
    return at::addmm(bias, x, weight.t());
  }

  auto output = x.matmul(weight.t());
  if (!no_bias_) {
    output += bias;
  }
  return output;
}

void Linear::reset_parameters() {
  auto stdv = 1.0 / std::sqrt(weight.size(1));
  for (auto& p : parameters()) {
    p.second.data().uniform_(-stdv, stdv);
  }
}

void Linear::initialize_parameters() {
  weight = this->add(
      Var(DefaultTensor(at::kFloat).tensor({nout, nin}), true), "weight");
  if (!no_bias_) {
    bias =
        this->add(Var(DefaultTensor(at::kFloat).tensor({nout}), true), "bias");
  }
}

Variant Embedding::forward(Variant const& input) {
  auto x = input.get();
  return at::embedding(weight, x, -1, false, false);
}

void Embedding::reset_parameters() {
  for (auto& p : parameters()) {
    p.second.data().normal_(0, 1);
  }
}

void Embedding::initialize_parameters() {
  weight = this->add(
      Var(DefaultTensor(at::kFloat).tensor({num_embeddings, embedding_dim}),
          true),
      "weight");
}

void Conv::initialize_parameters() {
  if (!transposed_) {
    for (auto pad : output_padding_) {
      if (pad != 0) {
        throw std::runtime_error(
            "Only transposed convolutions support output padding!");
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
  weight =
      this->add(Var(DefaultTensor(at::kFloat).tensor(wsize), true), "weight");
  if (!no_bias_) {
    bias = this->add(
        Var(DefaultTensor(at::kFloat).tensor({out_channels_}), true), "bias");
  } else {
    assert(!bias.defined());
  }
}

void Conv::reset_parameters() {
  auto n = in_channels_;
  for (auto k : ks_)
    n *= k;
  auto stdv = 1.0 / std::sqrt(n);
  for (auto& p : parameters()) {
    p.second.data().uniform_(-stdv, stdv);
  }
}

Variant Conv::forward(Variant const& input) {
  auto x = input.get();
  if (Nd_ == 1) {
    assert(x.ndimension() == 3);
    x = x.unsqueeze(-1); // TODO: Use conv1d once available
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
      out = at::conv_transpose2d(
          x,
          weight,
          bias,
          stride_,
          padding_,
          output_padding_,
          groups_,
          dilation_);
    } else {
      out = at::conv2d(x, weight, bias, stride_, padding_, dilation_, groups_);
    }
  } else if (Nd_ == 3) {
    if (transposed_) {
      out = at::conv_transpose3d(
          x,
          weight,
          bias,
          stride_,
          padding_,
          output_padding_,
          groups_,
          dilation_);
    } else {
      out = at::conv3d(x, weight, bias, stride_, padding_, dilation_, groups_);
    }
  }

  return out;
}

void BatchNorm::initialize_parameters() {
  if (affine_) {
    weight = this->add(
        Var(DefaultTensor(at::kFloat).tensor(num_features_), true), "weight");
    bias = this->add(
        Var(DefaultTensor(at::kFloat).tensor(num_features_), true), "bias");
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

Variant BatchNorm::forward(Variant const& inputs) {
  Variable input, running_mean, running_var;
  if (stateful_) {
    input = inputs.get();
    running_mean = this->running_mean;
    running_var = this->running_var;
  } else {
    auto lst = inputs.getList();
    input = lst[0].get();
    running_mean = lst[1].get();
    running_var = lst[2].get();
  }

  if (train_) {
    const auto num_channels = input.dim() > 1 ? input.size(1) : 1;
    if (input.numel() / num_channels <= 1) {
      throw std::runtime_error(
          "BatchNorm expected more than 1 value per channel when training!");
    }
  }

  auto output = at::batch_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      train_,
      momentum_,
      eps_,
      hasCudnn());

  return output;
}

template <typename Derived>
void RNNBase<Derived>::initialize_containers() {
  auto gate_size = hidden_size_;
  if (mode_ == RNNMode::LSTM) {
    gate_size *= 4;
  } else if (mode_ == RNNMode::GRU) {
    gate_size *= 3;
  }

  for (auto i = 0U; i < nlayers_; i++) {
    auto input_size = (i == 0) ? input_size_ : hidden_size_;
    i2h.push_back(this->add(
        Linear(input_size, gate_size).no_bias(no_bias_).make(),
        "i2h_" + std::to_string(i)));
    h2h.push_back(this->add(
        Linear(hidden_size_, gate_size).no_bias(no_bias_).make(),
        "h2h_" + std::to_string(i)));
  }
  if (dropout_ > 0)
    dropout_module = Dropout(dropout_).make();
  this->flatten_parameters();
}

template <typename Derived>
void RNNBase<Derived>::reset_parameters() {
  auto stdv = 1.0 / std::sqrt(hidden_size_);
  for (auto& p : this->parameters()) {
    p.second.data().uniform_(-stdv, stdv);
  }
}

template <typename Derived>
Variant RNNBase<Derived>::GRU_cell_forward(Variant&& inputs, int i) {
  auto lst = inputs.getList();
  auto x = lst[0].get();
  auto hx = lst[1].get().defined()
      ? lst[1].get()
      : Var(x.data().type().zeros({x.size(0), hidden_size_}));

  auto gi = i2h[i]->forward(x).get();
  auto gh = h2h[i]->forward(hx).get();
  auto gic = gi.chunk(3, 1);
  auto ghc = gh.chunk(3, 1);

  auto reset_gate = (gic[0] + ghc[0]).sigmoid_();
  auto input_gate = (gic[1] + ghc[1]).sigmoid_();
  auto new_gate = (gic[2] + reset_gate * ghc[2]).tanh_();
  auto hy = new_gate + input_gate * (hx - new_gate);

  return hy;
}

template <typename Derived>
Variant RNNBase<Derived>::RNN_TANH_cell_forward(
    Variant&& inputs,
    int i) {
  auto lst = inputs.getList();
  auto x = lst[0].get();
  auto hx = lst[1].get().defined()
      ? lst[1].get()
      : Var(x.data().type().zeros({x.size(0), hidden_size_}));

  auto h = i2h[i]->forward(x) + h2h[i]->forward(hx);
  return h.tanh();
}

template <typename Derived>
Variant RNNBase<Derived>::RNN_RELU_cell_forward(
    Variant&& inputs,
    int i) {
  auto lst = inputs.getList();
  auto x = lst[0].get();
  auto hx = lst[1].get().defined()
      ? lst[1].get()
      : Var(x.data().type().zeros({x.size(0), hidden_size_}));

  auto h = i2h[i]->forward(x) + h2h[i]->forward(hx);
  return at::relu(h);
}

template <typename Derived>
Variant RNNBase<Derived>::LSTM_cell_forward(Variant&& inputs, int i) {
  auto lst = inputs.getList();
  auto x = lst[0].get();
  auto hid = lst[1].get().defined()
      ? lst[1].get()
      : Var(x.data().type().zeros({2, x.size(0), hidden_size_}));
  auto hx = hid[0];
  auto cx = hid[1];

  auto gates = i2h[i]->forward(x) + h2h[i]->forward(hx);

  auto chunked = gates.chunk(4, 1);
  auto in_gate = chunked[0].sigmoid();
  auto forget_gate = chunked[1].sigmoid();
  auto cell_gate = chunked[2].tanh();
  auto out_gate = chunked[3].sigmoid();

  auto cy = (forget_gate * cx) + (in_gate * cell_gate);
  auto hy = out_gate * cy.tanh();

  return at::stack({hy, cy}, 0);
}

template <typename Derived>
Variant RNNBase<Derived>::cell_forward(Variant&& inputs, int i) {
  if (mode_ == RNNMode::LSTM)
    return LSTM_cell_forward(std::move(inputs), i);
  else if (mode_ == RNNMode::GRU)
    return GRU_cell_forward(std::move(inputs), i);
  else if (mode_ == RNNMode::RNN_TANH)
    return RNN_TANH_cell_forward(std::move(inputs), i);
  else if (mode_ == RNNMode::RNN_RELU)
    return RNN_RELU_cell_forward(std::move(inputs), i);
  else
    throw std::runtime_error("No such RNN mode");
}

template <typename Derived>
Variant RNNBase<Derived>::autograd_forward(Variant&& inputs) {
  auto lst = inputs.getList();
  auto inp = lst[0].get();

  std::vector<Tensor> hidden;
  for (size_t i = 0; i < nlayers_; i++) {
    hidden.emplace_back(lst[1].defined() ? lst[1][i] : tag::Variable());
  }

  auto output =
      Var(this->DefaultTensor(at::kFloat)
              .zeros({inp.size(0), inp.size(1), hidden_size_}),
          false);
  for (auto t = 0U; t < inp.size(0); t++) {
    auto x = inp.select(0, t);
    for (size_t i = 0; i < nlayers_; i++) {
      hidden[i] = cell_forward({x, hidden[i]}, i).get();
      if (mode_ == RNNMode::LSTM) {
        x = hidden[i][0];
      } else {
        x = hidden[i];
      }
      auto output_slice = output.select(0, t);
      output_slice.copy_(x);
      if (dropout_ > 0 && i != nlayers_ - 1) {
        x = dropout_module->forward(x).get();
      }
    }
  }

  auto hidout = at::stack(hidden, 0);
  return {output, hidout};
}

template <typename Derived>
bool RNNBase<Derived>::flatten_parameters() {
  data_ptrs_.clear();
  auto anyParam = i2h[0]->params_.begin()->second;
  if (!anyParam.is_cuda() || !at::cudnn_is_acceptable(anyParam)) {
    return false;
  }
  std::unordered_set<void*> unique_data_ptrs;
  auto params = this->parameters();
  for (auto& p : params) {
    unique_data_ptrs.insert(p.second.data().data_ptr());
  }
  // TODO PyTorch says:
  // If any parameters alias, we fall back to the slower, copying code path.
  // This is
  // a sufficient check, because overlapping parameter buffers that don't
  // completely
  // alias would break the assumptions of the uniqueness check in
  // Module.named_parameters().
  // But I'm not sure if this is the case for us
  if (unique_data_ptrs.size() != params.size()) {
    return false;
  }

  std::vector<Tensor> weight_list;
  for (size_t i = 0; i < nlayers_; i++) {
    weight_list.push_back(i2h[i]->param("weight"));
    weight_list.push_back(h2h[i]->param("weight"));
    if (!no_bias_) {
      weight_list.push_back(i2h[i]->param("bias"));
      weight_list.push_back(h2h[i]->param("bias"));
    }
  }
  auto weight_stride0 = no_bias_ ? 2 : 4;

  {
    no_grad_guard guard;
    flat_weight_ = at::_cudnn_rnn_flatten_weight(
        weight_list,
        weight_stride0,
        input_size_,
        mode_,
        hidden_size_,
        nlayers_,
        false,
        false); // batch_first and bidirectional, unsupported
  }
  for (auto& p : params) {
    data_ptrs_.emplace_back(p.second.data().data_ptr());
  }
  return true;
}

template <typename Derived>
Variant RNNBase<Derived>::CUDNN_forward(Variant&& inputs) {
  std::vector<Tensor> weight_list;
  for (size_t i = 0; i < nlayers_; i++) {
    weight_list.push_back(i2h[i]->param("weight"));
    weight_list.push_back(h2h[i]->param("weight"));
    if (!no_bias_) {
      weight_list.push_back(i2h[i]->param("bias"));
      weight_list.push_back(h2h[i]->param("bias"));
    }
  }
  auto weight_stride0 = no_bias_ ? 2 : 4;

  auto lst = inputs.getList();
  auto x = lst[0].get();
  Variable hx, cx;
  if (!lst[1].defined()) {
    hx = x.type().zeros({nlayers_, x.size(1), hidden_size_});
    if (mode_ == RNNMode::LSTM) {
      cx = x.type().zeros({nlayers_, x.size(1), hidden_size_});
    }
  } else {
    hx = mode_ == RNNMode::LSTM ? inputs[1][0] : inputs[1];
    cx = mode_ == RNNMode::LSTM ? inputs[1][1] : Variable();
  }
  auto dropout_state = x.type().tensor();

  std::vector<void*> weight_data_ptrs;
  auto params = this->parameters();
  for (auto& p : params) {
    weight_data_ptrs.emplace_back(p.second.data().data_ptr());
  }
  if (weight_data_ptrs != data_ptrs_) {
    std::cerr << "Parameters are unflattened! Code path might be super slow. "
                 "Please call flatten_parameters() when you muck around with "
                 "storages!"
              << std::endl;
    flat_weight_ = Variable();
  }

  // tup = std::tuple of output, hy, cy, reserve, new_weight_buf
  auto tup = _cudnn_rnn(
      x,
      weight_list,
      weight_stride0,
      flat_weight_,
      hx,
      cx,
      mode_,
      hidden_size_,
      nlayers_,
      false, // batch first
      0, // TODO waiting on dropout state descriptor in C++ pytorch
      this->train_,
      false, // bidirectional
      {}, // packing not supported
      dropout_state // TODO waiting on dropout state descriptor in C++ pytorch
  );

  Variable hidout = mode_ == RNNMode::LSTM
      ? at::stack({std::get<1>(tup), std::get<2>(tup)}, 0)
      : std::get<1>(tup);
  Variable output = std::get<0>(tup);
  return {output, hidout};
}

template <typename Derived>
Variant RNNBase<Derived>::forward(Variant const& input) {
  std::vector<Variant> inp;
  if (input.isList()) {
    auto lst = input.getList();
    if (lst.size() != 2) {
      throw std::runtime_error("RNN takes a list of the input and hidden");
    }
    inp.push_back(lst[0]);
    inp.push_back(lst[1]);
  } else {
    inp.push_back(input.get());
    inp.push_back(Variable());
  }

  // Dropout descriptors aren't in C++ in PyTorch yet...
  auto output = at::cudnn_is_acceptable(inp[0].get()) && dropout_ == 0
      ? CUDNN_forward(std::move(inp))
      : autograd_forward(std::move(inp));

  return output;
}

template <typename Derived>
void RNNBase<Derived>::cuda() {
  Container_CRTP<Derived>::cuda();
  flatten_parameters();
}

template <typename Derived>
void RNNBase<Derived>::cpu() {
  Container_CRTP<Derived>::cpu();
  flatten_parameters();
}

Variant Dropout::forward(Variant const& inputs) {
  if (p_ == 0 || !this->train_)
    return inputs;
  else if (inputs.isVariable()) {
    auto x = inputs.get();
    auto noise = x.data().type().tensor(x.sizes())
      .uniform_(0, 1)
      .m([](const Tensor& self, float other) { return self > other;}, p_)
      .toType(x.type().scalarType())
      .mul_(1. / (1 - p_));
    return x * Var(noise);
  } else if (inputs.isList()) {
    std::vector<Variant> lst;
    for (auto& var : inputs.getList()) {
      lst.emplace_back(this->forward(var));
    }
    return lst;
  } else if (inputs.isDict()) {
    std::unordered_map<std::string, Variant> dict;
    for (auto& var : inputs.getDict()) {
      dict.emplace(var.first, this->forward(var.second));
    }
    return dict;
  } else {
    throw std::runtime_error("Type of input not supported for Dropout");
  }
}

Variant Dropout2d::forward(Variant const& inputs) {
  if (p_ == 0 || !this->train_)
    return inputs;
  else if (inputs.isVariable()) {
    auto x = inputs.get();
    auto noise = x.data().type().tensor({x.size(0), x.size(1), 1, 1})
      .uniform_(0, 1)
      .m([](const Tensor& self, at::Scalar other) { return self > other;}, p_)
      .toType(x.type().scalarType())
      .mul_(1. / (1 - p_));
    return x * Var(noise);
  } else if (inputs.isList()) {
    std::vector<Variant> lst;
    for (auto& var : inputs.getList()) {
      lst.emplace_back(this->forward(var));
    }
    return lst;
  } else if (inputs.isDict()) {
    std::unordered_map<std::string, Variant> dict;
    for (auto& var : inputs.getDict()) {
      dict.emplace(var.first, this->forward(var.second));
    }
    return dict;
  } else {
    throw std::runtime_error("Type of input not supported for Dropout");
  }
}

} // namespace autograd
