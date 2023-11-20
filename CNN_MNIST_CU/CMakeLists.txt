cmake_minimum_required(VERSION 3.10)

project(CNN_MNIST_CUDA LANGUAGES CUDA)

option(DEBUG "Enable debugging" ON)

set(CMAKE_CUDA_ARCHITECTURES 61)  # Adjust based on your GPU's compute capability

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

include_directories("/scr")

set(SOURCES
    scr/cnn_inference.cu
)
# Assuming cnn_inference.cu is in the src directory
add_executable(cnn_mnist_cuda ${SOURCES})

# Specify CUDA-specific flags
set_target_properties(cnn_mnist_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
