cmake_minimum_required(VERSION 3.18)

project(RevocableMegakernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(toy_megakernel
    src/main.cu
)

target_include_directories(toy_megakernel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(toy_megakernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
