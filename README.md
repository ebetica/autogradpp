# AUTOGRADPP

This is an experimental C++ frontend to pytorch's C++ backend. Use at your own
risk.

How to build:
```
git submodule update --init --recursive
cd pytorch; python setup.py build; # get a coffee
cd ..; mkdir -p build; cd build
cmake .. -DPYTHON_EXECUTABLE:FILEPATH=$(which python)  # helpful if you use anaconda
make -j
```
