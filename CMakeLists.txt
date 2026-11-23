cmake_minimum_required(VERSION 3.18)

project(RevocableMegakernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 75 = Turing (original), 80 = A100. Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80")
endif()

# Per-task record logging for the correctness checker (B1). Turn OFF for timing runs.
option(ENABLE_TASK_LOG "Record per-task execution logs" ON)

add_executable(toy_megakernel
    src/main.cu
)

target_include_directories(toy_megakernel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(ENABLE_TASK_LOG)
    target_compile_definitions(toy_megakernel PRIVATE ENABLE_TASK_LOG)
endif()

set_target_properties(toy_megakernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
