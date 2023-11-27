# cmake_minimum_required(VERSION 3.12)

# project(CNN_INFERENCE)

# enable_language(CUDA)

# find_package(CUDA REQUIRED)

# set(SOURCES
#     src/cnn_inference.cpp
#     src/Convolution.cu
#     src/Functions.cpp
#     src/memory_allocation.cu
# )

# # you need to set this according to your nvidia architecture
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=compute_72;-code=sm_72)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

# cuda_add_executable(cnn_inference ${SOURCES})

# target_link_libraries(cnn_inference ${CUDA_LIBRARIES})

cmake_minimum_required(VERSION 3.12)

project(CNN_INFERENCE)

enable_language(CUDA)

find_package(CUDA REQUIRED)

set(SOURCES
    tests/test.cu
    src/Convolution.cu
    src/Functions.cpp
    src/memory_allocation.cu
)

# you need to set this according to your nvidia architecture
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=compute_72;-code=sm_72)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

cuda_add_executable(test ${SOURCES})

target_link_libraries(test ${CUDA_LIBRARIES})
