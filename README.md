# AUTOGRADPP

This is an experimental C++ frontend to pytorch's C++ backend. Use at your own
risk.

How to build:
```
git submodule update --init --recursive

# On Linux:
cd pytorch; python setup.py build;

# On macOS
cd pytorch; LDSHARED="cc -dynamiclib -undefined dynamic_lookup" python setup.py build;

cd ..; mkdir -p build; cd build
cmake .. -DPYTHON_EXECUTABLE:FILEPATH=$(which python)  # helpful if you use anaconda
make -j
```

# Stuff

- Check out the [MNIST example](https://github.com/ebetica/autogradpp/blob/master/test.cpp#L283), which tries to replicate PyTorch's MNIST model + training loop
- The principled way to write a model is probably something like 
```
AUTOGRAD_CONTAINER_CLASS(MyModel) {
  // This does a 2D convolution, followed by global sum pooling, followed by a linear.
 public:
  void initialize_parameters() override {
    myConv_ = add(Conv2d(1, 50, 3, 3).stride(2).make(), "conv");
    myLinear_ = add(Linear(50, 1).make(), "linear");
  }
  variable_list forward(variable_list x) override {
    auto v = myConv_->forward(x);
    v = v.mean(-1).mean(-1);
    return myLinear_.forward({v});
  }
 private:
  Container myLinear_;
  Container myConv_;
}
```

Some things are not implemented:
- Batchnorm
- Only SGD and Adam are implemented: the rest of the optimizers are just copying Python code from PyTorch over.

Some things to be careful of:
- Variable.detach does not do what you think it does. Better to do Variable(old.data())

Otherwise, everything else works. There may be breaking API changes.
