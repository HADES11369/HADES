cmake_minimum_required(VERSION 3.15.0)
project(EmbCacheLib LANGUAGES CUDA CXX)

# find python
find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_INTERP python)
message(STATUS "Using Python interpreter: ${PYTHON_INTERP}")

# find torch
file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts/find_torch.py FIND_TORCH_PY)
execute_process(
  COMMAND ${PYTHON_INTERP} ${FIND_TORCH_PY}
  OUTPUT_VARIABLE TORCH_PREFIX_VER
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "find_torch.py output: ${TORCH_PREFIX_VER}")
list(GET TORCH_PREFIX_VER 0 TORCH_PREFIX)
list(GET TORCH_PREFIX_VER 1 TORCH_VER)
message(STATUS "Configuring for PyTorch ${TORCH_VER}")
set(Torch_DIR "${TORCH_PREFIX}/Torch")
find_package(Torch REQUIRED)
# link against libtorch_python.so
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
list(APPEND EmbCacheLib_EXTERNAL_LIB ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

# find pybind11
file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts/find_pybind11.py FIND_PYBIND_PY)
execute_process(
  COMMAND ${PYTHON_INTERP} ${FIND_PYBIND_PY}
  OUTPUT_VARIABLE PYBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "find_pybind11.py output: ${PYBIND_CMAKE_DIR}")
set(pybind11_DIR ${PYBIND_CMAKE_DIR})
find_package(pybind11 REQUIRED)

# set flag
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp")

# Set openmp
include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
endif(OPENMP_FOUND)
message(STATUS "Build with OpenMP")

# Define our library target
# include_directories(src)
file(GLOB EmbCacheLib_SRC
  src/cache/*.cc
  src/cache/*.cu
  src/*.cc
  src/cuda/*.cu
)
pybind11_add_module(EmbCacheLib ${EmbCacheLib_SRC})
set_target_properties(EmbCacheLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(EmbCacheLib PROPERTIES CUDA_STANDARD 17)
set_target_properties(EmbCacheLib PROPERTIES POSITION_INDEPENDENT_CODE ON)
TARGET_COMPILE_OPTIONS(
  EmbCacheLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

# Enable C++17
target_compile_features(EmbCacheLib PRIVATE cxx_std_17)

# Link against LibTorch
target_link_libraries(EmbCacheLib PRIVATE ${EmbCacheLib_EXTERNAL_LIB})

# file(GLOB TEST_SRC src/cache/test.cc)
# add_executable(test ${TEST_SRC})
# target_link_libraries(test PRIVATE ${EmbCacheLib_EXTERNAL_LIB})
