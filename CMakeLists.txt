cmake_minimum_required(VERSION 3.25)

project(learn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

add_executable(learn ${CMAKE_SOURCE_DIR}/src/main.cu)

set_target_properties(learn PROPERTIES
    CXX_STANDARD 20
    CUDA_STANDARD 20
)

target_compile_options(${PROJECT_NAME}
    PRIVATE
    -I${PROJECT_SOURCE_DIR}/include
    -I${PROJECT_SOURCE_DIR}/include/cutlass/include
    -I${PROJECT_SOURCE_DIR}/include/cutlass/tools/util/include
    -I/usr/local/cuda/targets/x86_64-linux/include/cccl
    -I/usr/local/cuda/targets/x86_64-linux/include
)

target_compile_options(${PROJECT_NAME}
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -lineinfo
        -enable-tile
    >
)