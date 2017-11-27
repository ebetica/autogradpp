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
