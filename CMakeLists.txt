cmake_minimum_required(VERSION 3.15)
project(c_utils)

# Add this line to help find pybind11
set(PYBIND11_PYTHON_VERSION 3)

# Use either of these approaches:
# Option 1: If installed via apt
find_package(pybind11 CONFIG REQUIRED)
# Option 2: If installed via pip
#execute_process(
#    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
#    OUTPUT_VARIABLE pybind11_DIR
#    OUTPUT_STRIP_TRAILING_WHITESPACE
#)
#find_package(pybind11 CONFIG REQUIRED HINTS "${pybind11_DIR}")

pybind11_add_module(c_utils utils_binding.cpp utils.c)
